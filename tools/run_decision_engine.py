from __future__ import annotations
import argparse
import csv
from pathlib import Path

from linecaller.decision import DecisionEngine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--court-events", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-bounce-confidence", type=float, default=0.35)
    p.add_argument("--min-bounce-score", type=float, default=0.40)
    p.add_argument("--review-band-in", type=float, default=3.0)
    p.add_argument("--ball-contact-radius-in", type=float, default=1.45)
    a = p.parse_args()

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    engine = DecisionEngine(
        min_bounce_confidence=a.min_bounce_confidence,
        min_bounce_score=a.min_bounce_score,
        review_band_in=a.review_band_in,
        ball_contact_radius_in=a.ball_contact_radius_in,
    )

    rows = []
    counts = {"IN": 0, "OUT": 0, "REVIEW": 0}

    with open(a.court_events, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result = engine.decide(
                geometry_state=row["geometry_state"],
                nearest_line=row["nearest_line"],
                signed_distance_ft=float(row["signed_distance_ft"]),
                bounce_confidence=float(row.get("bounce_confidence", 0) or 0),
                bounce_score=float(row.get("bounce_score", 0) or 0),
            )
            counts[result.call] += 1

            rows.append({
                **row,
                "decision": result.call,
                "decision_confidence": round(result.confidence, 6),
                "decision_reason": result.reason,
                "signed_distance_in": round(result.signed_distance_in, 3),
            })

    output_csv = out / "decisions.csv"

    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)

    summary = (
        f"events={len(rows)}\n"
        f"IN={counts['IN']}\n"
        f"OUT={counts['OUT']}\n"
        f"REVIEW={counts['REVIEW']}\n"
        f"decisions={output_csv}\n"
    )

    (out / "decision_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    for row in rows:
        print(
            f"frame={row['frame']} "
            f"decision={row['decision']} "
            f"reason={row['decision_reason']} "
            f"line={row['nearest_line']} "
            f"distance={row['signed_distance_in']}in"
        )


if __name__ == "__main__":
    main()
