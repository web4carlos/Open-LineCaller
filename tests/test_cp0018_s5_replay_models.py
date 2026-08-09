from linecaller.product.replay_models import (
    ReplayState,
    ReplayViewModel,
)


def test_replay_view_model():
    m = ReplayViewModel(
        state=ReplayState.PLAYING,
        current_index=2,
        total_frames=10,
        speed=.5,
    )

    assert m.active is True
    assert m.speed_label == "0.5x"
