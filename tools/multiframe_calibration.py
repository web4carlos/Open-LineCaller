from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from linecaller.calibration_center.multiframe import (
    MultiFrameCourtCalibrator,
)


def sampled_frames(
    video,
    start_frame,
    frame_count,
    stride,
):
    cap = cv2.VideoCapture(
        video
    )

    if not cap.isOpened():
        raise SystemExit(
            f"Cannot open video: {video}"
        )

    frames = []

    for i in range(
        frame_count
    ):
        frame_no = (
            start_frame
            + i * stride
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_no,
        )

        ok, frame = cap.read()

        if ok:
            frames.append(
                frame
            )

    cap.release()
    return frames


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--video",
        required=True,
    )
    p.add_argument(
        "--output",
        required=True,
    )
    p.add_argument(
        "--start-frame",
        type=int,
        default=100,
    )
    p.add_argument(
        "--frames",
        type=int,
        default=12,
    )
    p.add_argument(
        "--stride",
        type=int,
        default=10,
    )

    a = p.parse_args()

    frames = sampled_frames(
        a.video,
        a.start_frame,
        a.frames,
        a.stride,
    )

    engine = (
        MultiFrameCourtCalibrator()
    )

    r = engine.calibrate_frames(
        frames
    )

    print(
        f"success={r.success}"
    )
    print(
        f"confidence={r.confidence:.3f}"
    )
    print(
        f"frames_analyzed={r.frames_analyzed}"
    )
    print(
        f"frames_with_candidate={r.frames_with_candidate}"
    )
    print(
        f"reason={r.reason}"
    )

    for line in r.line_confidences:
        print(
            f"{line.name}="
            f"{line.confidence:.3f} "
            f"{line.status}"
        )

    if r.image_points is None:
        raise SystemExit(
            2
        )

    out = Path(
        a.output
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "version": 1,
        "point_order": [
            "near-left",
            "near-right",
            "far-right",
            "far-left",
        ],
        "image_points": [
            [float(x), float(y)]
            for x, y
            in r.image_points
        ],
        "court_width_ft": 20.0,
        "court_length_ft": 44.0,
        "calibration_mode": (
            "AUTO_MULTI_FRAME"
        ),
        "multi_frame": {
            "success": r.success,
            "confidence": r.confidence,
            "frames_analyzed": (
                r.frames_analyzed
            ),
            "frames_with_candidate": (
                r.frames_with_candidate
            ),
            "reason": r.reason,
            "line_confidences": [
                {
                    "name": line.name,
                    "confidence": (
                        line.confidence
                    ),
                    "status": line.status,
                }
                for line
                in r.line_confidences
            ],
        },
    }

    out.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"output={out}"
    )


if __name__ == "__main__":
    main()
