import json
from linecaller.validation.models import ValidationTruthEvent, ValidationPrediction
from linecaller.validation.comparison import compare_events
from linecaller.validation.metrics import compute_metrics
from linecaller.validation.report import export_validation_report

def test_report(tmp_path):
    truth = [ValidationTruthEvent("a", "x.mp4", 1, "IN")]
    pred = [ValidationPrediction("a", 1, "IN", .99, 42)]
    c = compare_events(truth, pred)
    m = compute_metrics(c)
    path = export_validation_report(m, c, tmp_path / "r.json")
    data = json.loads(path.read_text())
    assert data["metrics"]["automatic_accuracy"] == 1.0
