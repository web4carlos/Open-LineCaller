import json
from pathlib import Path

def export_validation_report(metrics, comparisons, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics.to_dict(),
        "comparisons": [
            {
                "event_id": c.event_id,
                "truth": c.truth,
                "prediction": c.prediction,
                "correct": c.correct,
                "category": c.category,
                "confidence": c.confidence,
                "latency_ms": c.latency_ms,
            }
            for c in comparisons
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
