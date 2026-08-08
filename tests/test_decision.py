from linecaller_core.decision import DecisionEngine
from linecaller_core.models import BounceEvent, Call, CourtModel


def test_clear_in():
    engine = DecisionEngine()
    result = engine.evaluate(BounceEvent(1, 3.0, 6.0, 0.99), CourtModel())
    assert result.call == Call.IN


def test_clear_out():
    engine = DecisionEngine(review_margin_m=0.01)
    result = engine.evaluate(BounceEvent(1, -0.05, 6.0, 0.99), CourtModel())
    assert result.call == Call.OUT


def test_close_call_is_review():
    engine = DecisionEngine(review_margin_m=0.02)
    result = engine.evaluate(BounceEvent(1, -0.01, 6.0, 0.99), CourtModel())
    assert result.call == Call.REVIEW


def test_low_bounce_confidence_is_review():
    engine = DecisionEngine(min_bounce_confidence=0.8)
    result = engine.evaluate(BounceEvent(1, 3.0, 6.0, 0.5), CourtModel())
    assert result.call == Call.REVIEW
