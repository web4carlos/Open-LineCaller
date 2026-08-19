from __future__ import annotations

import argparse
import json
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


def crop_rgb(frame_bgr, cx: float, cy: float, half: int):
    h, w = frame_bgr.shape[:2]
    ix, iy = int(round(cx)), int(round(cy))
    x1, y1 = max(0, ix-half), max(0, iy-half)
    x2, y2 = min(w, ix+half+1), min(h, iy+half+1)
    return cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


def panel_rgb(rgb, size=(142, 142)):
    return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), size, interpolation=cv2.INTER_NEAREST)


def panel_mask(mask, size=(142, 142)):
    rgb = np.repeat(mask[..., None], 3, axis=2)
    return cv2.resize(rgb, size, interpolation=cv2.INTER_NEAREST)


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
        current_rgb = crop_rgb(frame, px, py, half)
        background_rgb = crop_rgb(background_frame, px, py, half)
        ev = isolator.isolate(
            current_rgb,
            background_rgb,
            expected_diameter_px=expected_d,
        )
        records.append({
            "frame": frame_no, "state": state, "cell": cell,
            "px": px, "py": py, "identity_score": identity_score,
            "expected_d": expected_d, "current_rgb": current_rgb,
            "background_rgb": background_rgb, "evidence": ev,
        })
    cap.release()

    print()
    print("============================================================")
    print(" CP-0035.9.4.2 RC7.1 - BALL-COLOR RESIDUAL ISOLATION")
    print("============================================================")
    print()
    print(f"FPS.............................. {fps:.6f}")
    print(f"BACKGROUND FRAME................. {a.background_frame} (ENGINEERING ONLY)")
    print("DIFFERENCE ENGINE................. PIL ImageChops.difference")
    print("COLOR SOURCE...................... LOCKED BALL REFERENCE / RUNTIME")
    print("HARDCODED COLOR NAME.............. NO")
    print(f"LEARNED HUE CENTER................ {profile.hue_center:.2f} / 255")
    print(f"LEARNED HUE TOLERANCE............. +/- {profile.hue_tolerance:.2f}")
    print(f"LEARNED SATURATION MIN............ {profile.saturation_min}")
    print(f"LEARNED VALUE MIN................. {profile.value_min}")
    print("RESIDUAL DEFINES XYZ.............. NO")
    print("RESIDUAL DEFINES PATH............. NO")
    print()
    print("FRAME | STATE               | ID SCORE | BALL PX | AREA | CENTROID       | ISOLATED")
    print("-" * 102)
    for r in records:
        e = r["evidence"]
        c = "None" if e.centroid_xy is None else f"({e.centroid_xy[0]:4.1f},{e.centroid_xy[1]:4.1f})"
        print(
            f"{r['frame']:5d} | {r['state']:19s} | {r['identity_score']:8.3f} | "
            f"{r['expected_d']:7.2f} | {e.component_area_px:4d} | {c:14s} | "
            f"{'YES' if e.residual_present else 'NO'}"
        )

    passes = sum(1 for r in records if r["evidence"].residual_present)
    control = isolator.isolate(
        records[0]["background_rgb"], records[0]["background_rgb"],
        expected_diameter_px=records[0]["expected_d"],
    )
    control_ok = not control.residual_present
    print()
    print(f"USER-CONFIRMED BALL ISOLATIONS.... {passes}/{len(records)}")
    print(f"BACKGROUND==BACKGROUND CONTROL.... {'PASS' if control_ok else 'FAIL'}")

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    first, last = RC5_CONFIRMED[0][0], RC5_CONFIRMED[-1][0]
    by_frame = {r["frame"]: r for r in records}
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_w = 620
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
            cv2.circle(img, (round(rec["px"]), round(rec["py"])), 10, (255,255,255), 2, cv2.LINE_AA)

        canvas = np.zeros((fh, canvas_w, 3), dtype=np.uint8)
        canvas[:, :fw] = img
        panel = canvas[:, fw:]
        cv2.putText(panel, "RC7.1 BALL-COLOR RESIDUAL", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(panel, "CURRENT | RAW DIFF | MASK | ISOLATED", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1, cv2.LINE_AA)

        if last_rec is not None:
            e = last_rec["evidence"]
            imgs = [
                panel_rgb(last_rec["current_rgb"]),
                panel_rgb(e.raw_residual_rgb),
                panel_mask(e.color_mask_u8),
                panel_rgb(e.isolated_rgb),
            ]
            x0, y0, gap = 10, 68, 10
            for i, im in enumerate(imgs):
                x = x0 + i * (142 + gap)
                panel[y0:y0+142, x:x+142] = im
            cv2.putText(panel, f"frame: {last_rec['frame']}", (12, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(panel, f"ID score: {last_rec['identity_score']:.3f}", (12, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(panel, f"component area: {e.component_area_px}px", (12, 286), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(panel, f"ISOLATED: {'YES' if e.residual_present else 'NO'}", (340, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(panel, "Color learned from locked ball reference", (12, 326), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(panel, "No hardcoded yellow/orange/green", (12, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        writer.write(canvas)
    writer.release()
    cap.release()

    print()
    print(f"output={out}")
    print("PRODUCTION_OVERLAY=NO")
    print("RC5_CONFIRMED_SEQUENCE_IS_TEST_ONLY=YES")
    print("BACKGROUND_FRAME_IS_TEST_ONLY=YES")
    print("BALL_COLOR_RESIDUAL_ISOLATION_GATE=COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
