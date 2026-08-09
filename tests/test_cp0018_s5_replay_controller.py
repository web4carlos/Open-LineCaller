import pytest

from linecaller.live.models import LiveFramePacket
from linecaller.product.replay_controller import ReplayPlaybackController
from linecaller.product.replay_models import ReplayState


def packets(n=3):
    return tuple(
        LiveFramePacket(
            frame_number=i,
            captured_at=float(i),
            frame=f"f{i}",
        )
        for i in range(n)
    )


def test_replay_lifecycle():
    c = ReplayPlaybackController(speed=.25)

    assert c.begin(
        packets(),
        decision_frame=1,
    ) is True

    assert c.state == ReplayState.FREEZE

    c.advance()
    assert c.state == ReplayState.PLAYING

    c.advance()
    c.advance()
    c.advance()

    assert c.state == ReplayState.RESUME

    c.finish()
    assert c.state == ReplayState.LIVE


def test_empty_replay_does_not_start():
    c = ReplayPlaybackController()
    assert c.begin(()) is False


def test_invalid_speed_rejected():
    with pytest.raises(ValueError):
        ReplayPlaybackController(speed=.33)
