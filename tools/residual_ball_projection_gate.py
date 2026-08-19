from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from linecaller.dcf.ball_color_residual_isolator import (
    BallColorResidualIsolator,
    LockedResidualColorProfile,
)
from linecaller.dcf.ball_in_mesh_search import BallInMeshSearcher
from linecaller.dcf.models import CellIndex
from linecaller.dcf.residual_ball_projection import ResidualBallProjectionEstimator


RC5_CONFIRMED = (
    (3813, CellIndex(69, 60, 6), 336.1, 251.8, 0.760, "TRANSITION_LOCAL"),
    (3814, CellIndex(70, 65, 9), 340.8, 225.8, 0.758, "TRANSITION_LOCAL"),
    (3815, CellIndex(71, 70, 12), 344.3, 200.5, 0.477, "NEW_PATH_CONFIRMED"),
    (3816, CellIndex(71, 70, 15), 345.3, 181.0, 0.644, "NEW_PATH_CONFIRMED"),
    (3817, CellIndex(71, 65, 18), 347.5, 166.2, 0.750, "NEW_PATH_CONFIRMED"),
    (3818, CellIndex(71, 61, 20), 349.2, 156.5, 0.659, "NEW_PATH_CONFIRMED"),
    (3819, CellIndex(73, 61, 19), 362.6, 163.2, 0.488, "NEW_PATH_CONFIRMED"),
    (3820, CellIndex(72, 66, 22), 352.6, 138.9, 0.612, "NEW_PATH_CONFIRMED"),
    (3821, CellIndex(72, 68, 23), 352.7, 130.7, 0.577, "NEW_PATH_CONFIRMED"),
    (3822, CellIndex(72, 66, 24), 353.6, 125.7, 0.506, "NEW_PATH_CONFIRMED"),
)


def load_points(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(data["image_points"], dtype=np.float64)


def read_frame(cap: cv2.VideoCapture, frame_no: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_no}")
    return frame


def crop_rgb_with_origin(frame_bgr, cx: float, cy: float, half: int):
    h, w = frame_bgr.shape[:2]
    ix, iy = int(round(cx)), int(round(cy))
    x1, y1 = max(0, ix-half), max(0, iy-half)
    x2, y2 = min(w, ix+half+1), min(h, iy+half+1)
    rgb = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    return rgb, (float(x1), float(y1))


def panel_rgb(rgb, size=(240, 240)):
    return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), size, interpolation=cv2.INTER_NEAREST)


def draw_cross(img, pt, radius=7, thickness=2):
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.line(img, (x-radius, y), (x+radius, y), (255,255,255), thickness, cv2.LINE_AA)
    cv2.line(img, (x, y-radius), (x, y+radius), (255,255,255), thickness, cv2.LINE_AA)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--background-frame", type=int, default=3780)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    template_rgb = np.asarray(Image.open(a.template).convert("RGB"), dtype=np.uint8)
    profile = LockedResidualColorProfile.from_template(template_rgb)
    isolator = BallColorResidualIsolator(profile)
    estimator = ResidualBallProjectionEstimator()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise RuntimeError("Invalid FPS")

    background_frame = read_frame(cap, a.background_frame)
    projector = BallInMeshSearcher(load_points(a.calibration)).projector

    records = []
    for frame_no, cell, px, py, identity_score, state in RC5_CONFIRMED:
        frame = read_frame(cap, frame_no)
        expected_d = float(projector.project(cell).expected_diameter_px)
        half = max(10, int(round(expected_d * 2.0)))
        current_rgb, origin = crop_rgb_with_origin(frame, px, py, half)
        background_rgb, bg_origin = crop_rgb_with_origin(background_frame, px, py, half)
        if origin != bg_origin:
            raise RuntimeError("ROI alignment mismatch")
        ev = isolator.isolate(
            current_rgb,
            background_rgb,
            expected_diameter_px=expected_d,
        )
        proj = estimator.measure(ev, roi_origin_xy=origin)
        center_err = None
        if proj.valid and proj.center_xy_global is not None:
            center_err = math.hypot(proj.center_xy_global[0]-px, proj.center_xy_global[1]-py)
        records.append({
            "frame": frame_no,
            "state": state,
            "cell": cell,
            "px": px,
            "py": py,
            "identity_score": identity_score,
            "expected_d": expected_d,
            "current_rgb": current_rgb,
            "origin": origin,
            "evidence": ev,
            "projection": proj,
            "center_err": center_err,
        })
    cap.release()

    print()
    print("============================================================")
    print(" CP-0035.9.4.2 RC7.2 - RESIDUAL BALL PROJECTION")
    print("============================================================")
    print()
    print(f"FPS.............................. {fps:.6f}")
    print(f"BACKGROUND FRAME................. {a.background_frame} (ENGINEERING ONLY)")
    print("BALL IDENTITY..................... WHO / SEPARATE")
    print("RESIDUAL.......................... PRESENCE EVIDENCE")
    print("PROJECTION OUTPUT................. CENTER (u,v) + IMAGE-SCALE EVIDENCE")
    print("CIRCLE REQUIRED................... NO")
    print("CENTER METHOD..................... RESIDUAL-INTENSITY WEIGHTED")
    print("SIZE EVIDENCE..................... AREA + MOMENTS + BBOX")
    print("PROJECTION DEFINES XYZ............ NO")
    print("PROJECTION DEFINES PATH........... NO")
    print()
    print("FRAME | TRACK CENTER    | RESIDUAL CENTER | dAREA | dMIN | dMAJ | BBOX    | ELONG | CENTER DELTA")
    print("-" * 116)
    valid = 0
    deltas = []
    scales = []
    for r in records:
        q = r["projection"]
        if not q.valid:
            print(f"{r['frame']:5d} | ({r['px']:6.1f},{r['py']:6.1f}) | INVALID")
            continue
        valid += 1
        gx, gy = q.center_xy_global
        delta = float(r["center_err"])
        deltas.append(delta)
        if q.apparent_scale_px is not None:
            scales.append(q.apparent_scale_px)
        print(
            f"{r['frame']:5d} | ({r['px']:6.1f},{r['py']:6.1f}) | "
            f"({gx:7.2f},{gy:7.2f}) | "
            f"{q.area_equivalent_diameter_px:5.2f} | "
            f"{q.moment_minor_diameter_px:5.2f} | "
            f"{q.moment_major_diameter_px:5.2f} | "
            f"{q.bbox_width_px:2.0f}x{q.bbox_height_px:<2.0f} | "
            f"{q.elongation:5.2f} | {delta:6.2f}px"
        )

    print()
    print(f"VALID PROJECTIONS................ {valid}/{len(records)}")
    if deltas:
        print(f"MEAN CENTER DELTA................ {np.mean(deltas):.3f} px")
        print(f"MAX CENTER DELTA................. {np.max(deltas):.3f} px")
    if scales:
        print(f"APPARENT SCALE RANGE............. {min(scales):.2f} .. {max(scales):.2f} px")
    print("NOTE.............................. SCALE IS IMAGE EVIDENCE, NOT PHYSICAL DIAMETER")

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    first, last = RC5_CONFIRMED[0][0], RC5_CONFIRMED[-1][0]
    by_frame = {r["frame"]: r for r in records}
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_w = 520
    canvas_w = fw + panel_w
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, fh))
    if not writer.isOpened():
        raise RuntimeError("Could not open output video")

    last_rec = None
    for frame_no in range(first, last+1):
        ok, img = cap.read()
        if not ok:
            break
        rec = by_frame.get(frame_no)
        if rec is not None:
            last_rec = rec
            cv2.circle(img, (round(rec["px"]), round(rec["py"])), 10, (255,255,255), 1, cv2.LINE_AA)
            q = rec["projection"]
            if q.valid and q.center_xy_global is not None:
                draw_cross(img, q.center_xy_global, radius=6, thickness=2)

        canvas = np.zeros((fh, canvas_w, 3), dtype=np.uint8)
        canvas[:, :fw] = img
        panel = canvas[:, fw:]
        cv2.putText(panel, "RC7.2 RESIDUAL BALL PROJECTION", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(panel, "No circle assumption", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1, cv2.LINE_AA)

        if last_rec is not None:
            q = last_rec["projection"]
            e = last_rec["evidence"]
            iso = panel_rgb(e.isolated_rgb, size=(240,240))
            # Draw measured local center onto enlarged isolated panel.
            if q.valid and q.center_xy_local is not None:
                ih, iw = e.isolated_rgb.shape[:2]
                sx = 240.0 / max(1, iw)
                sy = 240.0 / max(1, ih)
                lx, ly = q.center_xy_local
                draw_cross(iso, (lx*sx, ly*sy), radius=10, thickness=2)
                if e.component_bbox is not None:
                    x1,y1,x2,y2 = e.component_bbox
                    cv2.rectangle(iso, (round(x1*sx),round(y1*sy)), (round(x2*sx),round(y2*sy)), (255,255,255), 1)
            panel[70:310, 12:252] = iso
            cv2.putText(panel, "ISOLATED + measured center", (12, 332), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(panel, f"frame {last_rec['frame']}", (280, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(panel, f"ID {last_rec['identity_score']:.3f}", (280, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)
            if q.valid:
                gx,gy = q.center_xy_global
                cv2.putText(panel, f"center=({gx:.2f},{gy:.2f})", (280, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(panel, f"d_area={q.area_equivalent_diameter_px:.2f}px", (280, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(panel, f"d_minor={q.moment_minor_diameter_px:.2f}px", (280, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(panel, f"d_major={q.moment_major_diameter_px:.2f}px", (280, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(panel, f"elong={q.elongation:.2f}", (280, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(panel, f"delta={last_rec['center_err']:.2f}px", (280, 266), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1, cv2.LINE_AA)
            else:
                cv2.putText(panel, "projection INVALID", (280, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        writer.write(canvas)
    writer.release()
    cap.release()

    print()
    print(f"output={out}")
    print("PRODUCTION_OVERLAY=NO")
    print("RC5_CONFIRMED_SEQUENCE_IS_TEST_ONLY=YES")
    print("BACKGROUND_FRAME_IS_TEST_ONLY=YES")
    print("RESIDUAL_BALL_PROJECTION_GATE=COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
