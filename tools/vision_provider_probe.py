from __future__ import annotations

import argparse
import cv2

from linecaller.vision import create_vision_provider


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--provider", default="classical")
    p.add_argument("--tracknet-csv")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--start-frame", type=int, default=0)
    args = p.parse_args()

    kwargs = {}

    if args.provider.lower() in {
        "tracknet",
        "tracknet_csv",
    }:
        if not args.tracknet_csv:
            raise SystemExit(
                "--tracknet-csv is required for TrackNet provider"
            )
        kwargs["csv_path"] = args.tracknet_csv

    provider = create_vision_provider(
        args.provider,
        **kwargs,
    )

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise SystemExit(
            f"Unable to open: {args.video}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    ) or 30.0

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        args.start_frame,
    )

    frames = 0
    detections = 0
    frame_number = int(args.start_frame)

    try:
        while frames < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            result = provider.detect(
                frame=frame,
                frame_number=frame_number,
                timestamp=frame_number / fps,
            )

            if (
                result.x is not None
                and result.y is not None
            ):
                detections += 1

            frames += 1
            frame_number += 1

    finally:
        cap.release()
        provider.shutdown()

    print(f"provider={provider.name}")
    print(f"start_frame={args.start_frame}")
    print(f"frames={frames}")
    print(f"detections={detections}")
    print(
        f"coverage="
        f"{(detections / frames if frames else 0.0):.3%}"
    )


if __name__ == "__main__":
    main()
