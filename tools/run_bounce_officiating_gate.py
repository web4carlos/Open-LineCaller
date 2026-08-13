from __future__ import annotations
import argparse, csv
from pathlib import Path
from linecaller.bounce_v2.officiating_gate import BounceOfficiatingGate


def truth(v):
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def load_by_frame(path):
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[int(float(row["frame"]))] = row
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--physical-audit", required=True)
    p.add_argument("--evidence-audit", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()

    physical = load_by_frame(a.physical_audit)
    evidence = load_by_frame(a.evidence_audit)
    frames = sorted(set(physical) | set(evidence))

    gate = BounceOfficiatingGate()
    rows = []

    for frame in frames:
        pr = physical.get(frame)
        er = evidence.get(frame)

        if pr is None or er is None:
            decision = "REJECTED_BOUNCE"
            confidence = 0.0
            reason = "MISSING_GATE_INPUT"
        else:
            r = gate.decide(
                frame=frame,
                physical_valid=truth(pr.get("contact_valid")),
                physical_score=float(pr.get("contact_score") or 0),
                evidence_accepted=truth(er.get("evidence_accepted")),
                evidence_class=er.get("evidence_class") or "UNVERIFIED_CONTACT",
                evidence_confidence=float(er.get("evidence_confidence") or 0),
            )
            decision = r.decision.value
            confidence = r.confidence
            reason = r.reason

        rows.append({
            "frame": frame,
            "decision": decision,
            "confidence": round(confidence, 4),
            "reason": reason,
        })

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output = out / "bounce_officiating_gate.csv"

    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame","decision","confidence","reason"])
        w.writeheader()
        w.writerows(rows)

    verified = sum(r["decision"] == "VERIFIED_BOUNCE" for r in rows)
    review = sum(r["decision"] == "REVIEW_BOUNCE" for r in rows)
    rejected = sum(r["decision"] == "REJECTED_BOUNCE" for r in rows)

    print(f"events={len(rows)}")
    print(f"verified={verified}")
    print(f"review={review}")
    print(f"rejected={rejected}")
    print(f"output={output}")
    for r in rows:
        print(
            f"frame={r['frame']} decision={r['decision']} "
            f"confidence={r['confidence']} reason={r['reason']}"
        )


if __name__ == "__main__":
    main()
