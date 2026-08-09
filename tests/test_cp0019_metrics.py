import pytest
from linecaller.validation.comparison import compare_events
from linecaller.validation.metrics import compute_metrics
from linecaller.validation.models import ValidationPrediction, ValidationTruthEvent

def test_metrics_accuracy_and_coverage():
    truth = [
        ValidationTruthEvent("a", "x.mp4", 1, "IN"),
        ValidationTruthEvent("b", "x.mp4", 2, "OUT"),
        ValidationTruthEvent("c", "x.mp4", 3, "OUT"),
        ValidationTruthEvent("d", "x.mp4", 4, "IN"),
    ]
    pred = [
        ValidationPrediction("a", 1, "IN", .99, 40),
        ValidationPrediction("b", 2, "OUT", .98, 50),
        ValidationPrediction("c", 3, "REVIEW", .70, 45),
    ]
    m = compute_metrics(compare_events(truth, pred))
    assert m.automatic_accuracy == pytest.approx(1.0)
    assert m.coverage == pytest.approx(.5)
    assert m.review_rate == pytest.approx(.25)
    assert m.miss_rate == pytest.approx(.25)
