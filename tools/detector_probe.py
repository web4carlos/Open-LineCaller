from __future__ import annotations

import argparse
import cv2

from linecaller.ball.advanced_motion_detector import AdvancedMotionBallDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--max-frames", type=int, default=1800)
    args = parser.parse_args()

    detector = AdvancedMotionBallDetector()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Unable to open: {args.video}")

    frames = 0
    frames_with_candidates = 0
    total_candidates = 0
    max_candidates = 0

    try:
        while frames < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            candidates = detector.detect(frame)

            if candidates:
                frames_with_candidates += 1
                total_candidates += len(candidates)
                max_candidates = max(max_candidates, len(candidates))

            frames += 1
    finally:
        cap.release()

    t = detector.telemetry

    print(f"frames={frames}")
    print(f"frames_with_candidates={frames_with_candidates}")
    print(f"total_candidates={total_candidates}")
    print(f"max_candidates_per_frame={max_candidates}")
    print(f"telemetry_contours={t.contours}")
    print(f"telemetry_area_rejected={t.area_rejected}")
    print(f"telemetry_shape_rejected={t.shape_rejected}")
    print(f"telemetry_radius_rejected={t.radius_rejected}")
    print(f"telemetry_accepted={t.accepted}")


if __name__ == "__main__":
    main()
