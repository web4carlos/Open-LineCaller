from linecaller.bounce_v2 import BounceEngine, MotionSample


def s(frame, x, y, raw=True, conf=0.9):
    return MotionSample(
        frame=frame,
        raw_x=float(x) if raw else None,
        raw_y=float(y) if raw else None,
        tracked_x=float(x),
        tracked_y=float(y),
        confidence=conf,
        source="YOLO" if raw else "KALMAN-PREDICT",
    )


def test_curvature_bounce_detects_strong_turn():
    e = BounceEngine(
        window=2,
        min_pre_speed=1,
        min_post_speed=1,
        min_curvature=0.10,
        min_bounce_score=0.25,
    )

    seq = [
        s(0, 100, 80),
        s(1, 108, 92),
        s(2, 116, 106),
        s(3, 124, 92),
        s(4, 132, 78),
    ]

    event = None
    for sample in seq:
        event = e.update(sample)

    assert event is not None
    assert event.curvature > 0
    assert event.bounce_score > 0


def test_curvature_rejects_nearly_straight_motion():
    e = BounceEngine(
        window=2,
        min_pre_speed=0.1,
        min_post_speed=0.1,
        min_curvature=0.40,
        min_bounce_score=0.30,
    )

    # Mild turn, intentionally below required curvature.
    seq = [
        s(0, 100, 80),
        s(1, 110, 90),
        s(2, 120, 100),
        s(3, 130, 95),
        s(4, 140, 90),
    ]

    event = None
    for sample in seq:
        maybe = e.update(sample)
        if maybe:
            event = maybe

    assert event is None


def test_event_exposes_curvature_and_score():
    e = BounceEngine(
        window=2,
        min_pre_speed=1,
        min_post_speed=1,
        min_curvature=0.10,
        min_bounce_score=0.20,
    )

    seq = [
        s(0, 100, 80),
        s(1, 105, 91),
        s(2, 110, 104),
        s(3, 115, 90),
        s(4, 120, 76),
    ]

    event = None
    for sample in seq:
        event = e.update(sample)

    assert event is not None
    assert 0 <= event.curvature <= 1
    assert 0 <= event.bounce_score <= 1
