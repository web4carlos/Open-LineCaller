from linecaller.product.replay_controller import ReplayMetrics


def test_replay_metrics_average():
    m = ReplayMetrics(
        replay_count=2,
        total_frames_shown=20,
    )

    assert m.average_frames_per_replay == 10
