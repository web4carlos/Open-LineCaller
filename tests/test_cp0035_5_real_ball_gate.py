from linecaller.dcf.real_ball_gate import RealBallContinuityGate, TrackSample


def s(frame, x, y, confidence=.8, source="YOLO+KALMAN"):
    return TrackSample(frame, x, y, confidence, source)


def test_initial_lock():
    g = RealBallContinuityGate()
    d = g.evaluate(s(1, 100, 100))
    assert d.accepted
    assert d.reason == "INITIAL_LOCK"


def test_continuous_motion_is_accepted():
    g = RealBallContinuityGate(base_radius_px=20)
    assert g.evaluate(s(1,100,100)).accepted
    assert g.evaluate(s(2,105,102)).accepted
    assert g.evaluate(s(3,110,104)).accepted


def test_large_identity_jump_is_rejected():
    g = RealBallContinuityGate(base_radius_px=25)
    g.evaluate(s(1,100,100))
    g.evaluate(s(2,105,102))
    d = g.evaluate(s(3,300,300))
    assert not d.accepted
    assert d.reason == "IDENTITY_JUMP_REJECTED"


def test_rejected_jump_does_not_steal_lock():
    g = RealBallContinuityGate(base_radius_px=25)
    g.evaluate(s(1,100,100))
    g.evaluate(s(2,105,100))
    assert not g.evaluate(s(3,300,300)).accepted
    d = g.evaluate(s(4,115,100))
    assert d.accepted


def test_low_confidence_raw_rejected():
    g = RealBallContinuityGate(confidence_floor=.2)
    d = g.evaluate(s(1,100,100,.1,"YOLO"))
    assert not d.accepted
    assert d.reason == "LOW_CONFIDENCE"
