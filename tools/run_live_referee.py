from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

from linecaller.live import LiveRefereePipeline, parse_video_source
from linecaller.tracking import DetectionPoint


def best_detection(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None, None

    best = None

    for i in range(len(result.boxes)):
        conf = float(result.boxes.conf[i].item())
        x1, y1, x2, y2 = map(
            float,
            result.boxes.xyxy[i].cpu().numpy().tolist(),
        )

        candidate = (
            conf,
            x1,
            y1,
            x2,
            y2,
        )

        if best is None or conf > best[0]:
            best = candidate

    if best is None:
        return None, None

    conf, x1, y1, x2, y2 = best
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    detection = DetectionPoint(
        frame=0,
        x=cx,
        y=cy,
        confidence=conf,
    )

    return detection, (x1, y1, x2, y2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--source", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--max-gap", type=int, default=4)
    p.add_argument("--camera-width", type=int, default=1280)
    p.add_argument("--camera-height", type=int, default=720)
    p.add_argument("--record", default="")
    p.add_argument("--max-frames", type=int, default=0)
    a = p.parse_args()

    source = parse_video_source(a.source)

    model = YOLO(a.model)
    pipeline = LiveRefereePipeline(
        max_gap=a.max_gap,
        min_detection_confidence=a.conf,
        bounce_window=3,
    )

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise SystemExit(
            f"Cannot open live source: {source}"
        )

    if isinstance(source, int):
        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            a.camera_width,
        )
        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            a.camera_height,
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ) or a.camera_width

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ) or a.camera_height

    source_fps = (
        cap.get(cv2.CAP_PROP_FPS)
        or 30.0
    )

    writer = None

    if a.record:
        record_path = Path(a.record)
        record_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        writer = cv2.VideoWriter(
            str(record_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (width, height),
        )

    trail = deque(maxlen=60)

    frame_number = 0
    processed = 0
    started = time.perf_counter()
    fps_value = 0.0

    bounce_flash_until = -1
    last_bounce = None

    print("Open-LineCaller LIVE")
    print("Q = quit")
    print("R = reset tracker/bounce history")
    print("")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            result = model.predict(
                frame,
                conf=a.conf,
                imgsz=a.imgsz,
                device="cpu",
                verbose=False,
            )[0]

            detection, bbox = best_detection(result)

            if detection is not None:
                detection = DetectionPoint(
                    frame=frame_number,
                    x=detection.x,
                    y=detection.y,
                    confidence=detection.confidence,
                )

            live = pipeline.update(
                frame_number,
                detection,
            )

            if bbox is not None:
                x1, y1, x2, y2 = bbox

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )

            if (
                live.tracked_x is not None
                and live.tracked_y is not None
            ):
                tx = int(round(live.tracked_x))
                ty = int(round(live.tracked_y))

                trail.append(
                    (tx, ty, live.track_source)
                )

                for i in range(1, len(trail)):
                    p0 = trail[i - 1]
                    p1 = trail[i]

                    cv2.line(
                        frame,
                        (p0[0], p0[1]),
                        (p1[0], p1[1]),
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                predicted = (
                    live.track_source
                    == "KALMAN-PREDICT"
                )

                color = (
                    (0, 165, 255)
                    if predicted
                    else (0, 0, 255)
                )

                label = (
                    "PREDICTED"
                    if predicted
                    else "BALL"
                )

                cv2.circle(
                    frame,
                    (tx, ty),
                    8,
                    color,
                    -1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    (
                        f"{label} "
                        f"x={tx} y={ty} "
                        f"conf={live.confidence:.2f}"
                    ),
                    (
                        max(10, tx - 120),
                        max(35, ty - 24),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            else:
                trail.clear()

            if live.bounce:
                last_bounce = live
                bounce_flash_until = frame_number + 24

                print(
                    f"BOUNCE frame={frame_number} "
                    f"x={live.bounce_x:.1f} "
                    f"y={live.bounce_y:.1f} "
                    f"conf={live.bounce_confidence:.2f}",
                    flush=True,
                )

            if (
                last_bounce is not None
                and frame_number <= bounce_flash_until
            ):
                bx = int(
                    round(last_bounce.bounce_x)
                )
                by = int(
                    round(last_bounce.bounce_y)
                )

                cv2.circle(
                    frame,
                    (bx, by),
                    26,
                    (255, 0, 255),
                    4,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "BOUNCE",
                    (
                        max(10, bx - 55),
                        max(35, by - 40),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

            processed += 1

            elapsed = time.perf_counter() - started

            if elapsed > 0:
                fps_value = processed / elapsed

            cv2.putText(
                frame,
                (
                    f"Open-LineCaller LIVE | "
                    f"frame={frame_number} | "
                    f"pipeline_fps={fps_value:.1f}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "Q quit | R reset",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(frame)

            cv2.imshow(
                "Open-LineCaller LIVE",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                pipeline.reset()
                trail.clear()
                last_bounce = None
                bounce_flash_until = -1

                print(
                    "Tracker/bounce history reset.",
                    flush=True,
                )

            frame_number += 1

            if (
                a.max_frames > 0
                and frame_number >= a.max_frames
            ):
                break

    finally:
        cap.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

    print("")
    print(f"frames={frame_number}")
    print(f"pipeline_fps={fps_value:.3f}")


if __name__ == "__main__":
    main()
