from linecaller.validation.comparison import compare_events
from linecaller.validation.models import ValidationPrediction, ValidationTruthEvent

def test_categories():
    truth = [
        ValidationTruthEvent("a", "x.mp4", 1, "IN"),
        ValidationTruthEvent("b", "x.mp4", 2, "OUT"),
    ]
    pred = [
        ValidationPrediction("a", 1, "OUT"),
        ValidationPrediction("b", 2, "IN"),
    ]
    result = compare_events(truth, pred)
    assert result[0].category == "FALSE_OUT"
    assert result[1].category == "FALSE_IN"

def test_missing():
    truth = [ValidationTruthEvent("a", "x.mp4", 1, "IN")]
    result = compare_events(truth, [])
    assert result[0].category == "MISSING"
