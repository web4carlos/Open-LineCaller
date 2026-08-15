from __future__ import annotations

import argparse
import csv
from pathlib import Path

from linecaller.dcf.real_ball_gate import (
    GateDecision,
    RealBallContinuityGate,
    TrackSample,
)


def opt_float(row, name):
    value = row.get(name, "")
    return None if value in ("", None) else float(value)


def load_track(path):
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            x = opt_float(row, "raw_x")
            y = opt_float(row, "raw_y")

            if x is None or y is None:
                x = opt_float(row, "tracked_x")
                y = opt_float(row, "tracked_y")

            yield TrackSample(
                frame=int(row["frame"]),
                x=x,
                y=y,
                confidence=float(row.get("confidence") or 0.0),
                source=row.get("source", ""),
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--start-frame", type=int)
    p.add_argument("--end-frame", type=int)
    p.add_argument("--base-radius-px", type=float, default=55.0)
    p.add_argument("--confidence-floor", type=float, default=0.10)
    a = p.parse_args()

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gate = RealBallContinuityGate(
        base_radius_px=a.base_radius_px,
        confidence_floor=a.confidence_floor,
    )

    rows = []
    accepted = 0
    rejected = 0
    jumps = 0

    for sample in load_track(a.track_csv):
        if a.start_frame is not None and sample.frame < a.start_frame:
            continue
        if a.end_frame is not None and sample.frame > a.end_frame:
            continue

        d = gate.evaluate(sample)

        if d.accepted:
            accepted += 1
        else:
            rejected += 1
            if d.reason == "IDENTITY_JUMP_REJECTED":
                jumps += 1

        rows.append({
            "frame": d.frame,
            "accepted": d.accepted,
            "reason": d.reason,
            "distance_px": "" if d.distance_px is None else round(d.distance_px, 3),
            "expected_x": "" if d.expected_x is None else round(d.expected_x, 3),
            "expected_y": "" if d.expected_y is None else round(d.expected_y, 3),
            "observed_x": "" if d.observed_x is None else round(d.observed_x, 3),
            "observed_y": "" if d.observed_y is None else round(d.observed_y, 3),
        })

    csv_path = out / "dcf_real_ball_gate.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "frame","accepted","reason","distance_px",
            "expected_x","expected_y","observed_x","observed_y"
        ])
        wr.writeheader()
        wr.writerows(rows)

    summary = (
        f"samples={len(rows)}\n"
        f"accepted={accepted}\n"
        f"rejected={rejected}\n"
        f"identity_jumps_rejected={jumps}\n"
        f"csv={csv_path}\n"
    )

    (out / "run_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    for row in rows:
        if row["reason"] == "IDENTITY_JUMP_REJECTED":
            print(
                f"frame={row['frame']} "
                f"reason={row['reason']} "
                f"distance={row['distance_px']}px "
                f"expected=({row['expected_x']},{row['expected_y']}) "
                f"observed=({row['observed_x']},{row['observed_y']})"
            )


if __name__ == "__main__":
    main()
