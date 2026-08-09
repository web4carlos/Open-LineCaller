from linecaller.product.hud_models import (
    LatencyQuality,
    StatusLevel,
)
from linecaller.product.hud_policy import (
    confidence_metric,
    fps_metric,
    latency_quality,
    last_call_presentation,
    tracking_metric,
)


def test_tracking_locked_is_ok():
    metric = tracking_metric("LOCKED")
    assert metric.level == StatusLevel.OK


def test_tracking_lost_is_error():
    metric = tracking_metric("LOST")
    assert metric.level == StatusLevel.ERROR


def test_latency_quality_bands():
    assert latency_quality(40) == LatencyQuality.EXCELLENT
    assert latency_quality(100) == LatencyQuality.ACCEPTABLE
    assert latency_quality(180) == LatencyQuality.HIGH


def test_confidence_98_is_ok():
    metric = confidence_metric(.98)
    assert metric.level == StatusLevel.OK


def test_low_fps_is_error():
    metric = fps_metric(20)
    assert metric.level == StatusLevel.ERROR


def test_review_call_is_warning():
    p = last_call_presentation("REVIEW")
    assert p.level == StatusLevel.WARN
    assert p.call == "REVIEW"
