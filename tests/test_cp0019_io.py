from linecaller.validation.io import save_truth_jsonl, load_truth_jsonl
from linecaller.validation.models import ValidationTruthEvent

def test_roundtrip(tmp_path):
    path = tmp_path / "truth.jsonl"
    events = [ValidationTruthEvent("x", "x.mp4", 10, "OUT")]
    save_truth_jsonl(events, path)
    assert load_truth_jsonl(path) == events
