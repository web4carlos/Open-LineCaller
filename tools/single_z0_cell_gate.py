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
from linecaller.dcf.single_z0_candidate import SingleZ0CandidateResolver
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
    resolver = SingleZ0CandidateResolver()

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
    print("=" * 70)
    print(" CP-0035.9.4.2 RC8.3.2 - ONE BALL / ONE Z0 CELL GATE")
    print("=" * 70)
    print(f"ROI............................... {roi.zone}")
    print(f"ROI DEPTH......................... {roi.depth_bu} BU")
    print(f"FULL EXTERNAL CELLS............... {roi.full_external_cell_count}")
    print(f"ROI CELLS......................... {roi.active_cell_count}")
    print("RUNTIME........................... FRAME -> PILLOW -> ROI -> ONE CELL")
    print("MAX VISIBLE CANDIDATE CELLS....... 1")
    print("BALL PATH......................... NO")
    print("BALL HISTORY...................... NO")
    print("OTHER BALL POSITIONS.............. NO")
    print("PREVIOUS FRAME REFERENCE.......... NO")
    print("TRACKER / XYZ..................... NO")
    print("Z0 STATUS......................... CANDIDATE ONLY")
    print("UP / BINGO........................ NOT RUN YET")
    print("")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    frame_no = args.start_frame
    candidate_frames = []
    resolved_rows = []

    while frame_no <= args.end_frame:
        ok, bgr = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raw_candidates = watcher.detect(rgb)
        z0 = resolver.resolve(raw_candidates)

        overlay = bgr.copy()

        if z0 is not None:
            candidate_frames.append(frame_no)
            resolved_rows.append(
                (frame_no, z0.cell_id, z0.gx, z0.gy, z0.ball_pixels, z0.score)
            )

            # IMPORTANT: exactly ONE cell is visualized.
            poly = np.round(np.asarray(z0.polygon)).astype(np.int32)
            cv2.polylines(overlay, [poly], True, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(
                overlay,
                f"ONE Z0 CANDIDATE cell={z0.cell_id} grid=({z0.gx},{z0.gy})",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                overlay,
                "NO Z0 CANDIDATE",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            overlay,
            "ONE BALL / ONE Z0 CELL | PATH=NO | HISTORY=NO | BINGO=NOT YET",
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

    scanned = max(0, frame_no - args.start_frame)

    print(f"FRAMES SCANNED.................... {scanned}")
    print(f"FRAMES WITH ONE Z0 CANDIDATE...... {len(candidate_frames)}")
    if scanned:
        print(
            f"CANDIDATE RATE.................... "
            f"{100.0 * len(candidate_frames) / scanned:.2f}%"
        )

    print("")
    print("ONE-CELL RESULTS:")
    for f, cid, gx, gy, px, score in resolved_rows:
        print(
            f"  frame={f} ONE_CELL={cid} grid=({gx},{gy}) "
            f"ball_pixels={px} score={score:.4f}"
        )

    print("")
    print(f"OUTPUT............................ {out_path}")
    print("MAX CELLS DRAWN PER FRAME......... 1")
    print("PATH / HISTORY DRAWN.............. 0")
    print("BINGO / OUT....................... NOT EVALUATED")
    print("RC8_3_2_SINGLE_Z0_CELL_GATE=COMPLETE")


if __name__ == "__main__":
    main()
