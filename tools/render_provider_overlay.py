from __future__ import annotations

import argparse
from collections import deque

import cv2

from linecaller.vision import create_vision_provider


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--provider", required=True)
    p.add_argument("--tracknet-csv")
    p.add_argument("--max-frames", type=int)
    args = p.parse_args()

    kwargs = {}

    if args.provider.lower() in {
        "tracknet",
        "tracknet_csv",
    }:
        if not args.tracknet_csv:
            raise SystemExit(
                "--tracknet-csv is required"
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

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise SystemExit(
            f"Unable to create: {args.output}"
        )

    trail = deque(maxlen=20)
    frame_number = 0
    detections = 0

    try:
        while True:
            if (
                args.max_frames is not None
                and frame_number >= args.max_frames
            ):
                break

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
                x = int(round(result.x))
                y = int(round(result.y))

                trail.append((x, y))
                detections += 1

                pts = list(trail)

                for i in range(1, len(pts)):
                    cv2.line(
                        frame,
                        pts[i - 1],
                        pts[i],
                        (0, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )

                cv2.circle(
                    frame,
                    (x, y),
                    20,
                    (0, 255, 0),
                    4,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    f"{provider.name.upper()} BALL",
                    (max(5, x + 25), max(30, y - 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                trail.clear()

            cv2.putText(
                frame,
                f"Frame {frame_number}",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)
            frame_number += 1

    finally:
        cap.release()
        writer.release()
        provider.shutdown()

    print(f"provider={provider.name}")
    print(f"frames={frame_number}")
    print(f"detections={detections}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
