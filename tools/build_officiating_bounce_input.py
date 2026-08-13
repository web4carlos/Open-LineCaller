from __future__ import annotations
import argparse, csv
from pathlib import Path

from linecaller.bounce_v2.pipeline_bridge import OfficiatingPipelineBridge


def load_gate(path):
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[int(float(row["frame"]))] = row
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bounce-csv", required=True)
    p.add_argument("--gate-csv", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()

    gate = load_gate(a.gate_csv)
    bridge = OfficiatingPipelineBridge()

    output_rows = []
    dropped_rows = []

    with open(a.bounce_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_fields = reader.fieldnames or []

        for row in reader:
            frame = int(float(row["frame"]))
            g = gate.get(frame)

            if g is None:
                r = bridge.route(
                    frame=frame,
                    gate_decision="REJECTED_BOUNCE",
                    confidence=0.0,
                )
            else:
                r = bridge.route(
                    frame=frame,
                    gate_decision=g.get("decision", ""),
                    confidence=float(g.get("confidence") or 0),
                )

            enriched = dict(row)
            enriched["bounce_gate_decision"] = r.gate_decision
            enriched["bounce_gate_confidence"] = round(r.confidence, 4)
            enriched["pipeline_action"] = r.action.value
            enriched["force_review"] = r.force_review
            enriched["pipeline_reason"] = r.reason

            if r.action.value == "DROP_EVENT":
                dropped_rows.append(enriched)
            else:
                output_rows.append(enriched)

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    officiating = out / "officiating_bounce_events.csv"
    dropped = out / "dropped_bounce_events.csv"

    fields = list(original_fields)
    extras = [
        "bounce_gate_decision",
        "bounce_gate_confidence",
        "pipeline_action",
        "force_review",
        "pipeline_reason",
    ]
    for x in extras:
        if x not in fields:
            fields.append(x)

    def write(path, rows):
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    write(officiating, output_rows)
    write(dropped, dropped_rows)

    auto = sum(r["pipeline_action"] == "AUTO_OFFICIATE" for r in output_rows)
    review = sum(r["pipeline_action"] == "GEOMETRY_REVIEW_ONLY" for r in output_rows)

    print(f"input_events={len(output_rows)+len(dropped_rows)}")
    print(f"auto_officiate={auto}")
    print(f"review_only={review}")
    print(f"dropped={len(dropped_rows)}")
    print(f"officiating={officiating}")
    print(f"dropped_output={dropped}")

    for row in output_rows + dropped_rows:
        print(
            f"frame={row['frame']} "
            f"gate={row['bounce_gate_decision']} "
            f"action={row['pipeline_action']} "
            f"force_review={row['force_review']}"
        )


if __name__ == "__main__":
    main()
