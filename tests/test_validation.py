from linecaller_core.models import Call
from linecaller_core.validation import ValidationStats, score_prediction


def test_validation_stats():
    stats = ValidationStats()
    score_prediction(Call.IN, Call.IN, stats)
    score_prediction(Call.OUT, Call.IN, stats)
    score_prediction(Call.OUT, Call.REVIEW, stats)
    assert stats.total == 3
    assert stats.correct == 1
    assert stats.wrong == 1
    assert stats.review == 1
    assert stats.accuracy == 0.5
