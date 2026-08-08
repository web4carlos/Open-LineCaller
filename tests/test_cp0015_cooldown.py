from linecaller.live.cooldown import DecisionCooldown


def test_cooldown_suppresses_duplicate():
    c = DecisionCooldown(cooldown_frames=10)

    assert c.should_emit(100) is True
    assert c.should_emit(105) is False
    assert c.should_emit(110) is True
