from linecaller.product.summary_models import MatchSummary


def test_summary_to_dict():
    summary = MatchSummary(
        match_id="m1",
        started_at=1,
        ended_at=11,
        duration_seconds=10,
        camera_name="Camera 0",
        calibration_mode="AUTO",
        calls_total=3,
        calls_in=1,
        calls_out=1,
        calls_review=1,
        average_fps=60,
        average_latency_ms=40,
        average_confidence=.99,
        tracking_losses=0,
        replay_count=1,
    )

    data = summary.to_dict()

    assert data["calls_total"] == 3
    assert data["camera_name"] == "Camera 0"
