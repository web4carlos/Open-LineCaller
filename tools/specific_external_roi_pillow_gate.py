from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from linecaller.dcf.external_grid_pillow_watcher import (
    BallColorProfile,
    ExternalGridCalibration,
    ExternalGridPillowWatcher,
)
from linecaller.dcf.specific_external_roi import (
    calibration_for_specific_roi,
    select_specific_external_roi,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--grid-calibration", required=True)
    ap.add_argument(
        "--zone",
        choices=[
            "FAR_LEFT", "FAR_BASELINE", "FAR_RIGHT",
            "NEAR_LEFT", "NEAR_BASELINE", "NEAR_RIGHT",
        ],
        required=True,
    )
    ap.add_argument("--depth-bu", type=int, default=4)
    ap.add_argument("--start-frame", type=int, required=True)
    ap.add_argument("--end-frame", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--diff-threshold", type=int, default=18)
    ap.add_argument("--min-ball-pixels", type=int, default=2)
    args = ap.parse_args()

    full_cal = ExternalGridCalibration.load(args.grid_calibration)
    roi = select_specific_external_roi(full_cal, args.zone, depth_bu=args.depth_bu)
    active_cal = calibration_for_specific_roi(full_cal, roi)

    profile = BallColorProfile.from_template(args.template)
    watcher = ExternalGridPillowWatcher(
        active_cal,
        profile,
        diff_threshold=args.diff_threshold,
        min_ball_pixels=args.min_ball_pixels,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output: {out_path}")

    print("")
    print("=" * 68)
    print(" CP-0035.9.4.2 RC8.3.2 - SPECIFIC EXTERNAL ROI PILLOW GATE")
    print("=" * 68)
    print(f"SPECIFIC ROI...................... {roi.zone}")
    print(f"ROI DEPTH......................... {roi.depth_bu} BU")
    print(f"FULL EXTERNAL CELLS............... {roi.full_external_cell_count}")
    print(f"ROI CELLS......................... {roi.active_cell_count}")
    print(f"CELLS REMOVED FROM LOOP........... {roi.reduction_ratio*100.0:.2f}%")
    print("GRID GEOMETRY..................... CALIBRATION / UNCHANGED")
    print("RUNTIME LOOP...................... SPECIFIC ROI ONLY")
    print("PILLOW............................ CURRENT vs CALIBRATED REFERENCE")
    print("PREVIOUS FRAME.................... NO")
    print("BALL TRACKER...................... NO")
    print("PATH / XYZ........................ NO")
    print("UP / BINGO........................ NOT RUN YET")
    print("")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    frame_no = args.start_frame
    candidate_frames = []
    strongest = []

    while frame_no <= args.end_frame:
        ok, bgr = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        candidates = watcher.detect(rgb)
        overlay = bgr.copy()

        if candidates:
            candidate_frames.append(frame_no)
            top = candidates[0]
            strongest.append(
                (top.score, frame_no, top.cell_id, top.gx, top.gy, top.ball_pixels)
            )
            for c in candidates[:3]:
                poly = np.round(np.asarray(c.polygon)).astype(np.int32)
                cv2.polylines(overlay, [poly], True, (0, 255, 255), 2, cv2.LINE_AA)
            label = (
                f"ROI={roi.zone} candidate cell={top.cell_id} "
                f"grid=({top.gx},{top.gy}) px={top.ball_pixels}"
            )
        else:
            label = f"ROI={roi.zone} no ball-compatible cell"

        cv2.putText(
            overlay, label, (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "FRAME -> PILLOW -> SPECIFIC EXTERNAL ROI | TRACKER=NO | BOUNCE=NO",
            (12, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        writer.write(overlay)
        frame_no += 1

    cap.release()
    writer.release()

    strongest.sort(reverse=True)
    scanned = max(0, frame_no - args.start_frame)

    print(f"FRAMES SCANNED.................... {scanned}")
    print(f"CANDIDATE FRAMES.................. {len(candidate_frames)}")
    if scanned:
        print(f"CANDIDATE RATE.................... {100.0*len(candidate_frames)/scanned:.2f}%")
    if candidate_frames:
        print(f"FIRST CANDIDATE FRAME............. {candidate_frames[0]}")
        print(f"LAST CANDIDATE FRAME.............. {candidate_frames[-1]}")

    print("")
    print("STRONGEST ROI CANDIDATES:")
    for score, f, cid, gx, gy, px in strongest[:12]:
        print(
            f"  frame={f} cell={cid} grid=({gx},{gy}) "
            f"ball_pixels={px} score={score:.4f}"
        )

    print("")
    print(f"OUTPUT............................ {out_path}")
    print("BINGO / OUT....................... NOT EVALUATED")
    print("RC8_3_2_SPECIFIC_EXTERNAL_ROI_GATE=COMPLETE")


if __name__ == "__main__":
    main()
