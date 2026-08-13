from __future__ import annotations

import argparse
import csv
from pathlib import Path

from linecaller.bounce_v2.contact_validation import (
    PhysicalBounceContactValidator,
    TrackSample,
)


def first_value(row, names, default=None):
    for n in names:
        if n in row and row[n] not in ("", None):
            return row[n]
    return default


def fnum(row, names):
    v = first_value(row, names)
    if v in ("", None):
        return None
    return float(v)


def inum(row, names):
    v = first_value(row, names)
    if v in ("", None):
        return None
    return int(float(v))


def load_track(path):
    samples = []

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame = inum(row, ["frame", "frame_number"])
            if frame is None:
                continue

            raw_x = fnum(row, ["raw_x", "det_x", "yolo_x"])
            raw_y = fnum(row, ["raw_y", "det_y", "yolo_y"])
            tracked_x = fnum(row, ["tracked_x", "x", "track_x"])
            tracked_y = fnum(row, ["tracked_y", "y", "track_y"])
            confidence = fnum(row, ["confidence", "conf"]) or 0.0
            source = first_value(row, ["source", "track_source"], "") or ""

            samples.append(
                TrackSample(
                    frame=frame,
                    raw_x=raw_x,
                    raw_y=raw_y,
                    tracked_x=tracked_x,
                    tracked_y=tracked_y,
                    confidence=confidence,
                    source=str(source),
                )
            )

    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track-csv", required=True)
    p.add_argument("--bounce-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--min-raw-ratio", type=float, default=0.35)
    p.add_argument("--min-reversal-strength", type=float, default=1.5)
    p.add_argument("--min-score", type=float, default=0.45)
    p.add_argument("--max-localization-shift-px", type=float, default=90.0)
    a = p.parse_args()

    samples = load_track(a.track_csv)

    validator = PhysicalBounceContactValidator(
        window=a.window,
        min_raw_ratio=a.min_raw_ratio,
        min_reversal_strength=a.min_reversal_strength,
        min_score=a.min_score,
        max_localization_shift_px=a.max_localization_shift_px,
    )

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    accepted_rows = []

    with open(a.bounce_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame = inum(row, ["frame", "frame_number"])
            if frame is None:
                continue

            event_x = fnum(row, ["x", "bounce_x", "tracked_x"])
            event_y = fnum(row, ["y", "bounce_y", "tracked_y"])

            r = validator.validate(
                event_frame=frame,
                event_x=event_x,
                event_y=event_y,
                samples=samples,
            )

            enriched = {
                **row,
                "contact_valid": r.accepted,
                "contact_frame": r.contact_frame,
                "contact_x": None if r.contact_x is None else round(r.contact_x, 3),
                "contact_y": None if r.contact_y is None else round(r.contact_y, 3),
                "contact_score": round(r.score, 4),
                "contact_raw_ratio": round(r.raw_ratio, 4),
                "contact_reversal_strength": round(r.reversal_strength, 4),
                "contact_localization_shift_px": (
                    None
                    if r.localization_shift_px is None
                    else round(r.localization_shift_px, 3)
                ),
                "contact_reason": r.reason,
            }

            rows.append(enriched)

            if r.accepted:
                accepted = dict(enriched)
                # downstream court geometry should use the validated contact
                accepted["frame"] = r.contact_frame
                accepted["x"] = r.contact_x
                accepted["y"] = r.contact_y
                accepted_rows.append(accepted)

    audit_csv = out / "bounce_contact_audit.csv"
    valid_csv = out / "validated_bounce_events.csv"

    if rows:
        with audit_csv.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)

    if accepted_rows:
        with valid_csv.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(accepted_rows[0].keys()))
            wr.writeheader()
            wr.writerows(accepted_rows)
    else:
        valid_csv.write_text("", encoding="utf-8")

    print(f"candidates={len(rows)}")
    print(f"accepted={len(accepted_rows)}")
    print(f"rejected={len(rows)-len(accepted_rows)}")
    print(f"audit={audit_csv}")
    print(f"validated={valid_csv}")

    for row in rows:
        print(
            f"frame={row.get('frame')} "
            f"valid={row['contact_valid']} "
            f"contact_frame={row['contact_frame']} "
            f"score={row['contact_score']} "
            f"shift={row['contact_localization_shift_px']} "
            f"reason={row['contact_reason']}"
        )


if __name__ == "__main__":
    main()
