from linecaller.product.match_summary import MatchSummaryBuilder


def test_unknown_call_not_counted():
    b = MatchSummaryBuilder()
    b.record_call("UNKNOWN")

    summary = b.build(
        camera_name="Camera 0",
        calibration_mode="AUTO",
        replay_count=0,
        ended_at=b.started_at,
    )

    assert summary.calls_total == 0
