from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

from linecaller.tracking import (
    BallKalmanTracker,
    DetectionPoint,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--max-gap", type=int, default=4)
    return p.parse_args()

def best_detection(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best = None
    for i in range(len(result.boxes)):
        conf = float(result.boxes.conf[i].item())
        x1, y1, x2, y2 = map(
            float,
            result.boxes.xyxy[i].cpu().numpy().tolist(),
        )
        x = (x1 + x2) / 2.0
        y = (y1 + y2) / 2.0

        if best is None or conf > best[0]:
            best = (conf, x, y, x1, y1, x2, y2)

    return best

def main():
    a = parse_args()

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    overlay = out / "ball_tracker_overlay.mp4"
    csv_path = out / "ball_track.csv"
    summary_path = out / "run_summary.txt"

    model = YOLO(a.model)
    tracker = BallKalmanTracker(
        max_gap=a.max_gap,
        min_detection_confidence=a.conf,
    )

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {a.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(overlay),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise SystemExit(f"Cannot create overlay: {overlay}")

    trail = deque(maxlen=60)
    rows = []

    frames = 0
    yolo_frames = 0
    tracked_frames = 0
    predicted_frames = 0
    inference_seconds = 0.0

    try:
        while frames < a.max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.perf_counter()
            result = model.predict(
                frame,
                conf=a.conf,
                imgsz=a.imgsz,
                device="cpu",
                verbose=False,
            )[0]
            inference_seconds += time.perf_counter() - t0

            best = best_detection(result)
            detection = None

            if best is not None:
                conf, x, y, x1, y1, x2, y2 = best
                detection = DetectionPoint(
                    frame=frames,
                    x=x,
                    y=y,
                    confidence=conf,
                )
                yolo_frames += 1

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )

            point = tracker.update(
                frames,
                detection,
            )

            if point.tracked_x is not None:
                tracked_frames += 1

                if point.source == "KALMAN-PREDICT":
                    predicted_frames += 1

                tx = int(round(point.tracked_x))
                ty = int(round(point.tracked_y))
                trail.append((tx, ty, point.source))

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

                if point.source == "KALMAN-PREDICT":
                    marker_color = (0, 165, 255)
                    label = "PREDICTED"
                else:
                    marker_color = (0, 0, 255)
                    label = "TRACKED"

                cv2.circle(
                    frame,
                    (tx, ty),
                    9,
                    marker_color,
                    -1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    (
                        f"{label} "
                        f"x={tx} y={ty} "
                        f"conf={point.confidence:.2f}"
                    ),
                    (max(10, tx - 120), max(35, ty - 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    marker_color,
                    2,
                    cv2.LINE_AA,
                )
            else:
                trail.clear()

            cv2.putText(
                frame,
                (
                    f"FRAME {frames} | "
                    f"YOLO={yolo_frames} | TRACK={tracked_frames}"
                ),
                (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            rows.append({
                "frame": frames,
                "raw_x": "" if point.raw_x is None else round(point.raw_x, 3),
                "raw_y": "" if point.raw_y is None else round(point.raw_y, 3),
                "tracked_x": "" if point.tracked_x is None else round(point.tracked_x, 3),
                "tracked_y": "" if point.tracked_y is None else round(point.tracked_y, 3),
                "confidence": round(point.confidence, 6),
                "source": point.source,
                "missed_frames": point.missed_frames,
            })

            writer.write(frame)

            if frames and frames % 50 == 0:
                print(
                    f"frame={frames} "
                    f"yolo={yolo_frames} "
                    f"tracked={tracked_frames} "
                    f"predicted={predicted_frames}",
                    flush=True,
                )

            frames += 1

    finally:
        cap.release()
        writer.release()

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        csv_writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "raw_x",
                "raw_y",
                "tracked_x",
                "tracked_y",
                "confidence",
                "source",
                "missed_frames",
            ],
        )
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    yolo_coverage = yolo_frames / frames if frames else 0.0
    tracked_coverage = tracked_frames / frames if frames else 0.0
    inference_fps = frames / inference_seconds if inference_seconds else 0.0

    summary = (
        f"frames={frames}\n"
        f"yolo_frames={yolo_frames}\n"
        f"yolo_coverage={yolo_coverage:.3%}\n"
        f"tracked_frames={tracked_frames}\n"
        f"tracked_coverage={tracked_coverage:.3%}\n"
        f"predicted_frames={predicted_frames}\n"
        f"max_gap={a.max_gap}\n"
        f"inference_fps={inference_fps:.3f}\n"
        f"csv={csv_path}\n"
        f"overlay={overlay}\n"
    )

    summary_path.write_text(summary, encoding="utf-8")

    print("")
    print(summary)

if __name__ == "__main__":
    main()
