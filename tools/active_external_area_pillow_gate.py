from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from linecaller.dcf.active_external_area import (
    calibration_for_active_area,
    select_active_external_area,
)
from linecaller.dcf.external_grid_pillow_watcher import (
    BallColorProfile,
    ExternalGridCalibration,
    ExternalGridPillowWatcher,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--grid-calibration", required=True)
    ap.add_argument("--active-side", choices=["NEAR", "FAR"], required=True)
    ap.add_argument("--start-frame", type=int, required=True)
    ap.add_argument("--end-frame", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--diff-threshold", type=int, default=18)
    ap.add_argument("--min-ball-pixels", type=int, default=2)
    args = ap.parse_args()

    full_cal = ExternalGridCalibration.load(args.grid_calibration)
    area = select_active_external_area(full_cal, args.active_side)
    active_cal = calibration_for_active_area(full_cal, area)

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
    print("=" * 64)
    print(" CP-0035.9.4.2 RC8.3.1 - ACTIVE EXTERNAL AREA PILLOW GATE")
    print("=" * 64)
    print(f"ACTIVE SIDE....................... {area.side}")
    print(f"FULL EXTERNAL CELLS............... {area.full_external_cell_count}")
    print(f"ACTIVE EXTERNAL CELLS............. {area.active_cell_count}")
    print(f"CELLS REMOVED FROM LOOP........... {area.reduction_ratio*100.0:.2f}%")
    print("GRID GEOMETRY..................... CALIBRATION / UNCHANGED")
    print("RUNTIME LOOP...................... ACTIVE AREA ONLY")
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

            # Engineering only: draw candidate polygons, not the full grid.
            for c in candidates[:3]:
                poly = np.round(np.asarray(c.polygon)).astype(np.int32)
                cv2.polylines(overlay, [poly], True, (0, 255, 255), 2, cv2.LINE_AA)

            label = (
                f"ACTIVE={area.side} candidate cell={top.cell_id} "
                f"grid=({top.gx},{top.gy}) px={top.ball_pixels}"
            )
        else:
            label = f"ACTIVE={area.side} no external ball-compatible cell"

        cv2.putText(
            overlay,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "FRAME -> PILLOW -> ACTIVE EXTERNAL AREA | TRACKER=NO | BOUNCE=NO",
            (12, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(overlay)
        frame_no += 1

    cap.release()
    writer.release()

    strongest.sort(reverse=True)
    print(f"FRAMES SCANNED.................... {max(0, frame_no - args.start_frame)}")
    print(f"CANDIDATE FRAMES.................. {len(candidate_frames)}")
    if candidate_frames:
        print(f"FIRST CANDIDATE FRAME............. {candidate_frames[0]}")
        print(f"LAST CANDIDATE FRAME.............. {candidate_frames[-1]}")
    print("")
    print("STRONGEST ACTIVE-AREA CANDIDATES:")
    for score, f, cid, gx, gy, px in strongest[:12]:
        print(
            f"  frame={f} cell={cid} grid=({gx},{gy}) "
            f"ball_pixels={px} score={score:.4f}"
        )
    print("")
    print(f"OUTPUT............................ {out_path}")
    print("BINGO / OUT....................... NOT EVALUATED")
    print("RC8_3_1_ACTIVE_EXTERNAL_AREA_GATE=COMPLETE")


if __name__ == "__main__":
    main()
