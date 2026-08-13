from __future__ import annotations
import argparse
import csv
from pathlib import Path

from linecaller.decision.officiating_enforcer import ForceReviewEnforcer


def truth(v):
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def load_gate(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(float(row["frame"]))] = row
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decisions-csv", required=True)
    p.add_argument("--gate-csv", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()

    gate = load_gate(a.gate_csv)
    enforcer = ForceReviewEnforcer()
    rows = []

    with open(a.decisions_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])

        for row in reader:
            frame = int(float(row["frame"]))
            g = gate.get(frame, {})

            base = (
                row.get("decision")
                or row.get("final_decision")
                or row.get("call")
                or "REVIEW"
            )

            gd = g.get("decision", "UNKNOWN")
            gc = float(g.get("confidence") or 0)

            # If the decision CSV already carries force_review, honor it too.
            fr = truth(row.get("force_review", False))

            r = enforcer.enforce(
                frame=frame,
                base_decision=base,
                gate_decision=gd,
                gate_confidence=gc,
                force_review=fr,
            )

            row["base_decision"] = r.base_decision
            row["gate_decision"] = r.gate_decision
            row["gate_confidence"] = round(r.confidence, 4)
            row["force_review"] = r.force_review
            row["final_decision"] = r.final_decision
            row["officiating_reason"] = r.reason
            rows.append(row)

    extras = [
        "base_decision",
        "gate_decision",
        "gate_confidence",
        "force_review",
        "final_decision",
        "officiating_reason",
    ]
    for e in extras:
        if e not in fieldnames:
            fieldnames.append(e)

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output = out / "final_officiating_decisions.csv"

    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    counts = {"IN": 0, "OUT": 0, "REVIEW": 0}
    for row in rows:
        d = row["final_decision"]
        counts[d] = counts.get(d, 0) + 1

    print(f"events={len(rows)}")
    print(f"IN={counts.get('IN',0)}")
    print(f"OUT={counts.get('OUT',0)}")
    print(f"REVIEW={counts.get('REVIEW',0)}")
    print(f"output={output}")

    for row in rows:
        print(
            f"frame={row['frame']} "
            f"base={row['base_decision']} "
            f"gate={row['gate_decision']} "
            f"final={row['final_decision']} "
            f"reason={row['officiating_reason']}"
        )


if __name__ == "__main__":
    main()
