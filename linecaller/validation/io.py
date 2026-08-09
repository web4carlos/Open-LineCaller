import json
from pathlib import Path
from .models import ValidationTruthEvent, ValidationPrediction

def save_truth_jsonl(events, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event.to_dict()) + "\n")
    return path

def load_truth_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(ValidationTruthEvent(**json.loads(line)))
    return out

def load_predictions_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(ValidationPrediction(**json.loads(line)))
    return out
