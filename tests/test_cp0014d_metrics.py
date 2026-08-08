from linecaller.dataset.acceleration_metrics import AccelerationMetrics


def test_time_savings():
    m = AccelerationMetrics(
        manual_baseline_seconds_per_frame=3.0,
        assisted_seconds_per_frame=1.0,
        accepted_via_assist=10,
    )

    assert m.estimated_seconds_saved == 20.0
