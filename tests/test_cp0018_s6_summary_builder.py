import pytest

from linecaller.product.match_summary import MatchSummaryBuilder


def test_summary_builder():
    b = MatchSummaryBuilder()
    b.started_at = 100.0

    b.record_call("IN")
    b.record_call("OUT")
    b.record_call("REVIEW")

    b.record_runtime(
        fps=60,
        latency_ms=40,
        confidence=.98,
        tracking_status="LOCKED",
    )
    b.record_runtime(
        fps=58,
        latency_ms=50,
        confidence=1.0,
        tracking_status="LOST",
    )

    summary = b.build(
        camera_name="Camera 0",
        calibration_mode="AUTO",
        replay_count=1,
        ended_at=110.0,
        match_id="x",
    )

    assert summary.calls_total == 3
    assert summary.calls_in == 1
    assert summary.calls_out == 1
    assert summary.calls_review == 1
    assert summary.average_fps == pytest.approx(59)
    assert summary.average_latency_ms == pytest.approx(45)
    assert summary.average_confidence == pytest.approx(.99)
    assert summary.tracking_losses == 1
    assert summary.duration_seconds == 10
