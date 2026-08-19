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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--grid-calibration", required=True)
    ap.add_argument("--start-frame", type=int, required=True)
    ap.add_argument("--end-frame", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--diff-threshold", type=int, default=18)
    ap.add_argument("--min-ball-pixels", type=int, default=2)
    args = ap.parse_args()

    cal = ExternalGridCalibration.load(args.grid_calibration)
    profile = BallColorProfile.from_template(args.template)
    watcher = ExternalGridPillowWatcher(
        cal,
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

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    candidate_frames = []
    strongest = []

    print("")
    print("=" * 60)
    print(" CP-0035.9.4.2 RC8.3 - EXTERNAL GRID PILLOW GATE")
    print("=" * 60)
    print("RUNTIME ORDER.................... FRAME -> PILLOW -> EXTERNAL GRID")
    print("PILLOW COMPARES................. CURRENT vs CALIBRATED REFERENCE")
    print("PREVIOUS FRAME.................. NO")
    print("BALL TRACKER.................... NO")
    print("PATH / VELOCITY / XYZ........... NO")
    print("Z0 BOUNCE CLAIM................. NO")
    print("UP AFTER THE FACT............... NOT RUN YET")
    print("")

    frame_no = args.start_frame
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
            strongest.append((top.score, frame_no, top.cell_id, top.gx, top.gy, top.ball_pixels))
            # Engineering gate: draw only the TRUE perspective candidate polygons,
            # never the full production grid.
            for c in candidates[:3]:
                poly = np.round(np.asarray(c.polygon)).astype(np.int32)
                cv2.polylines(overlay, [poly], True, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                overlay,
                f"PILLOW CANDIDATE cell={top.cell_id} grid=({top.gx},{top.gy}) px={top.ball_pixels}",
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
                "PILLOW: no external ball-compatible cell",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            overlay,
            f"frame={frame_no}  REFERENCE=CALIBRATION  TRACKER=NO  BOUNCE=NOT CLAIMED",
            (12, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(overlay)
        frame_no += 1

    cap.release()
    writer.release()

    strongest.sort(reverse=True)
    print(f"FRAMES SCANNED................... {max(0, frame_no - args.start_frame)}")
    print(f"CANDIDATE FRAMES................. {len(candidate_frames)}")
    if candidate_frames:
        print(f"FIRST CANDIDATE FRAME............ {candidate_frames[0]}")
        print(f"LAST CANDIDATE FRAME............. {candidate_frames[-1]}")
    print("")
    print("STRONGEST CANDIDATES:")
    for score, f, cid, gx, gy, px in strongest[:12]:
        print(f"  frame={f} cell={cid} grid=({gx},{gy}) ball_pixels={px} score={score:.4f}")
    print("")
    print(f"OUTPUT........................... {out_path}")
    print("PRODUCTION GRID.................. INVISIBLE")
    print("BOUNCE / BINGO................... NOT EVALUATED IN RC8.3")
    print("RC8_3_EXTERNAL_GRID_PILLOW_GATE=COMPLETE")


if __name__ == "__main__":
    main()
