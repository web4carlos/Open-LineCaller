from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from linecaller.court_geometry import CourtCalibration, CourtGeometry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bounce-csv", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--near-line-threshold-in", type=float, default=3.0)
    a = p.parse_args()

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    calibration = CourtCalibration.load(a.calibration)
    geometry = CourtGeometry(
        calibration,
        near_line_threshold_in=a.near_line_threshold_in,
    )

    events = []

    with open(a.bounce_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = geometry.classify(
                float(row["x"]),
                float(row["y"]),
            )

            events.append({
                "frame": int(row["frame"]),
                "image_x": r.image_x,
                "image_y": r.image_y,
                "court_x_ft": r.court_x_ft,
                "court_y_ft": r.court_y_ft,
                "inside_court": r.inside_court,
                "nearest_line": r.nearest_line,
                "signed_distance_ft": r.signed_distance_ft,
                "absolute_distance_ft": r.absolute_distance_ft,
                "geometry_state": r.geometry_state,
                "bounce_confidence": float(row.get("confidence", 0) or 0),
                "bounce_score": float(row.get("bounce_score", 0) or 0),
            })

    csv_path = out / "court_events.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "frame",
            "image_x",
            "image_y",
            "court_x_ft",
            "court_y_ft",
            "inside_court",
            "nearest_line",
            "signed_distance_ft",
            "absolute_distance_ft",
            "geometry_state",
            "bounce_confidence",
            "bounce_score",
        ]

        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()

        for event in events:
            wr.writerow({
                **event,
                "image_x": round(event["image_x"], 3),
                "image_y": round(event["image_y"], 3),
                "court_x_ft": round(event["court_x_ft"], 4),
                "court_y_ft": round(event["court_y_ft"], 4),
                "signed_distance_ft": round(event["signed_distance_ft"], 4),
                "absolute_distance_ft": round(event["absolute_distance_ft"], 4),
            })

    cap = cv2.VideoCapture(a.video)

    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {a.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    overlay_path = out / "court_geometry_overlay.mp4"

    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    event_map = {e["frame"]: e for e in events}
    active = None
    active_until = -1
    frame_no = 0

    # Render only through the last bounce event + short tail.
    last_frame = (
        max(event_map.keys()) + 20
        if event_map
        else 0
    )

    court_pts = [
        (int(x), int(y))
        for x, y in calibration.image_points
    ]

    while frame_no <= last_frame:
        ok, frame = cap.read()
        if not ok:
            break

        for i in range(4):
            cv2.line(
                frame,
                court_pts[i],
                court_pts[(i + 1) % 4],
                (255, 255, 0),
                3,
                cv2.LINE_AA,
            )

        if frame_no in event_map:
            active = event_map[frame_no]
            active_until = frame_no + 18

        if active is not None and frame_no <= active_until:
            bx = int(round(active["image_x"]))
            by = int(round(active["image_y"]))

            cv2.circle(
                frame,
                (bx, by),
                24,
                (255, 0, 255),
                4,
                cv2.LINE_AA,
            )

            inches = active["absolute_distance_ft"] * 12.0

            cv2.putText(
                frame,
                (
                    f"{active['geometry_state']} | "
                    f"{active['nearest_line']} | "
                    f"{inches:.1f} in"
                ),
                (max(10, bx - 180), max(35, by - 45)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                (
                    f"court=({active['court_x_ft']:.2f},"
                    f"{active['court_y_ft']:.2f}) ft"
                ),
                (max(10, bx - 180), max(60, by - 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)
        frame_no += 1

    cap.release()
    writer.release()

    summary = (
        f"events={len(events)}\n"
        f"near_line_threshold_in={a.near_line_threshold_in}\n"
        f"csv={csv_path}\n"
        f"overlay={overlay_path}\n"
    )

    (out / "run_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print(summary)


if __name__ == "__main__":
    main()
