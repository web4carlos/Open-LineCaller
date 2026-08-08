from linecaller.apps.live_state import LiveUIState


def test_default_state():
    s = LiveUIState()

    assert s.last_call == "-"
    assert s.replay_active is False
    assert s.fps == 0.0
