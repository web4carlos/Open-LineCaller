from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from linecaller.dcf.ball_in_mesh_search import BallInMeshSearcher
from linecaller.dcf.cell_residual_ball_verifier import CellResidualBallVerifier
from linecaller.dcf.models import CellIndex


# TEST-ONLY fixture copied from the user-visually-confirmed RC5 real gate.
# It is deliberately NOT runtime tracking logic. RC7 isolates one question:
# does PIL(current aligned ROI - background aligned ROI) expose the ball as a
# compact residual at those already-confirmed same-ball locations?
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
    x1 = max(0, int(round(cx)) - half)
    y1 = max(0, int(round(cy)) - half)
    x2 = min(w, int(round(cx)) + half + 1)
    y2 = min(h, int(round(cy)) + half + 1)
    crop = frame_bgr[y1:y2, x1:x2]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def resize_panel(rgb, size=(170,170)):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, size, interpolation=cv2.INTER_NEAREST)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--background-frame", type=int, default=3780)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    # Identity asset is required so this test remains tied to the SAME BallSession,
    # but RC7 residual math intentionally does not read color/appearance from it.
    ball = cv2.imread(a.template)
    if ball is None:
        raise RuntimeError("Cannot read ball reference")

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise RuntimeError("Invalid FPS")

    background_frame = read_frame(cap, a.background_frame)
    projector = BallInMeshSearcher(load_points(a.calibration)).projector
    verifier = CellResidualBallVerifier()

    records = []
    for frame_no, cell, px, py, identity_score, state in RC5_CONFIRMED:
        frame = read_frame(cap, frame_no)
        projected = projector.project(cell)
        expected_d = float(projected.expected_diameter_px)
        half = max(10, int(round(expected_d * 2.0)))
        current_rgb = crop_rgb(frame, px, py, half)
        background_rgb = crop_rgb(background_frame, px, py, half)
        if current_rgb.shape != background_rgb.shape:
            raise RuntimeError(f"ROI shape mismatch at frame {frame_no}")

        evidence = verifier.verify(
            current_rgb,
            background_rgb,
            expected_diameter_px=expected_d,
        )
        records.append({
            "frame": frame_no,
            "cell": cell,
            "px": px,
            "py": py,
            "identity_score": identity_score,
            "state": state,
            "expected_d": expected_d,
            "current_rgb": current_rgb,
            "background_rgb": background_rgb,
            "evidence": evidence,
        })

    cap.release()

    print()
    print("============================================================")
    print(" CP-0035.9.4.2 RC7 - CELL RESIDUAL BALL VERIFIER")
    print("============================================================")
    print()
    print(f"FPS.............................. {fps:.6f}")
    print(f"BACKGROUND FRAME................. {a.background_frame} (ENGINEERING GATE ONLY)")
    print("INPUT LOCATIONS................... RC5 USER-VISUAL-CONFIRMED SAME BALL")
    print("DIFFERENCE ENGINE................. PIL ImageChops.difference")
    print("BALL IDENTITY..................... WHO / SEPARATE")
    print("RESIDUAL VERIFIER................. FOREGROUND PRESENCE ONLY")
    print("RESIDUAL DEFINES XYZ.............. NO")
    print("RESIDUAL DEFINES PATH............. NO")
    print()
    print("FRAME | STATE               | ID SCORE | CENTER CHANGE | CENTER DIFF | RESIDUAL")
    print("-" * 96)
    for r in records:
        e = r["evidence"]
        print(
            f"{r['frame']:5d} | {r['state']:19s} | "
            f"{r['identity_score']:8.3f} | "
            f"{e.center_changed_fraction:12.3f} | "
            f"{e.center_mean_difference:11.2f} | "
            f"{'YES' if e.residual_present else 'NO'}"
        )

    residual_passes = sum(1 for r in records if r["evidence"].residual_present)
    e0 = verifier.verify(
        records[0]["background_rgb"],
        records[0]["background_rgb"],
        expected_diameter_px=records[0]["expected_d"],
    )
    negative_control_ok = not e0.residual_present and e0.center_mean_difference == 0.0

    print()
    print(f"CONFIRMED-BALL RESIDUAL PASSES.... {residual_passes}/{len(records)}")
    print(f"BACKGROUND==BACKGROUND CONTROL.... {'PASS' if negative_control_ok else 'FAIL'}")
    print("GLOBAL REACQUISITION.............. NOT RUN IN RC7 ISOLATED GATE")

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    first_frame = RC5_CONFIRMED[0][0]
    last_frame = RC5_CONFIRMED[-1][0]
    by_frame = {r["frame"]: r for r in records}
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_w = 540
    canvas_w = fw + panel_w
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, fh)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open output video")

    last_record = None
    for frame_no in range(first_frame, last_frame + 1):
        ok, img = cap.read()
        if not ok:
            break
        rec = by_frame.get(frame_no)
        if rec is not None:
            last_record = rec
            cv2.circle(img, (round(rec["px"]), round(rec["py"])), 10, (255,255,255), 2, cv2.LINE_AA)

        canvas = np.zeros((fh, canvas_w, 3), dtype=np.uint8)
        canvas[:, :fw] = img
        panel = canvas[:, fw:]
        cv2.putText(panel, "CELL RESIDUAL / PIL", (16,28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(panel, f"frame {frame_no}", (16,54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        if last_record is not None:
            e = last_record["evidence"]
            cur = resize_panel(last_record["current_rgb"])
            bg = resize_panel(last_record["background_rgb"])
            res = resize_panel(e.residual_rgb)
            y0 = 72
            panel[y0:y0+170, 10:180] = cur
            panel[y0:y0+170, 185:355] = bg
            panel[y0:y0+170, 360:530] = res
            cv2.putText(panel, "CURRENT", (45,262), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(panel, "BACKGROUND", (215,262), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(panel, "RESIDUAL", (400,262), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(panel, f"ID score: {last_record['identity_score']:.3f}", (16,300), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(panel, f"center change: {e.center_changed_fraction:.3f}", (16,326), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(panel, f"center diff: {e.center_mean_difference:.1f}", (16,350), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(panel, f"RESIDUAL: {'YES' if e.residual_present else 'NO'}", (300,326), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2, cv2.LINE_AA)

        writer.write(canvas)

    writer.release()
    cap.release()

    print()
    print(f"output={out}")
    print("PRODUCTION_OVERLAY=NO")
    print("RC5_CONFIRMED_SEQUENCE_IS_TEST_ONLY=YES")
    print("BACKGROUND_FRAME_IS_TEST_ONLY=YES")
    print("CELL_RESIDUAL_GATE=COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
