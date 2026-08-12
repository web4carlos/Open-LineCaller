from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from linecaller.bounce_v2 import BounceEngine, MotionSample


def read_track_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            def opt(name):
                value = row.get(name, "")
                return None if value in ("", None) else float(value)

            yield MotionSample(
                frame=int(row["frame"]),
                raw_x=opt("raw_x"),
                raw_y=opt("raw_y"),
                tracked_x=opt("tracked_x"),
                tracked_y=opt("tracked_y"),
                confidence=float(row.get("confidence", 0) or 0),
                source=row.get("source", ""),
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track-csv", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--min-pre-speed", type=float, default=1.5)
    p.add_argument("--min-post-speed", type=float, default=1.5)
    p.add_argument("--refractory-frames", type=int, default=10)
    p.add_argument("--min-confidence", type=float, default=0.28)
    p.add_argument("--max-horizontal-jump", type=float, default=65.0)
    p.add_argument("--peak-tolerance-px", type=float, default=8.0)
    p.add_argument("--min-raw-ratio", type=float, default=0.35)
    a = p.parse_args()

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    events_path = out / "bounce_events.csv"
    overlay_path = out / "bounce_overlay.mp4"
    summary_path = out / "run_summary.txt"

    engine = BounceEngine(
        window=a.window,
        min_pre_speed=a.min_pre_speed,
        min_post_speed=a.min_post_speed,
        refractory_frames=a.refractory_frames,
        min_confidence=a.min_confidence,
        max_horizontal_jump=a.max_horizontal_jump,
        peak_tolerance_px=a.peak_tolerance_px,
        min_raw_ratio=a.min_raw_ratio,
    )

    samples = list(read_track_csv(a.track_csv))
    events = []

    for sample in samples:
        event = engine.update(sample)
        if event is not None:
            events.append(event)

    with events_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "x",
                "y",
                "confidence",
                "pre_velocity_y",
                "post_velocity_y",
                "source",
            ],
        )
        wr.writeheader()
        for e in events:
            wr.writerow({
                "frame": e.frame,
                "x": round(e.x, 3),
                "y": round(e.y, 3),
                "confidence": round(e.confidence, 6),
                "pre_velocity_y": round(e.pre_velocity_y, 6),
                "post_velocity_y": round(e.post_velocity_y, 6),
                "source": e.source,
            })

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {a.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    event_map = {e.frame: e for e in events}
    active = None
    active_until = -1
    frame_no = 0
    max_overlay_frames = len(samples)

    while frame_no < max_overlay_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_no in event_map:
            active = event_map[frame_no]
            active_until = frame_no + 18

        if active is not None and frame_no <= active_until:
            bx = int(round(active.x))
            by = int(round(active.y))

            cv2.circle(
                frame,
                (bx, by),
                24,
                (0, 0, 255),
                4,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"BOUNCE frame={active.frame}",
                (max(10, bx - 110), max(35, by - 42)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"x={bx} y={by} conf={active.confidence:.2f}",
                (max(10, bx - 110), max(60, by - 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"FRAME {frame_no} | BOUNCES={len(events)}",
            (25, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        if frame_no and frame_no % 50 == 0:
            print(
                f"overlay_frame={frame_no}/{max_overlay_frames}",
                flush=True,
            )

        frame_no += 1

    cap.release()
    writer.release()

    summary = (
        f"track_samples={len(samples)}\n"
        f"bounce_events={len(events)}\n"
        f"window={a.window}\n"
        f"min_pre_speed={a.min_pre_speed}\n"
        f"min_post_speed={a.min_post_speed}\n"
        f"refractory_frames={a.refractory_frames}\n"
        f"min_confidence={a.min_confidence}\n"
        f"max_horizontal_jump={a.max_horizontal_jump}\n"
        f"peak_tolerance_px={a.peak_tolerance_px}\n"
        f"min_raw_ratio={a.min_raw_ratio}\n"
        f"events={events_path}\n"
        f"overlay={overlay_path}\n"
    )

    summary_path.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
