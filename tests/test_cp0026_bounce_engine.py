from linecaller.bounce_v2 import BounceEngine, MotionSample


def s(frame, y, raw=True, conf=0.9):
    return MotionSample(
        frame=frame,
        raw_x=100.0 if raw else None,
        raw_y=float(y) if raw else None,
        tracked_x=100.0,
        tracked_y=float(y),
        confidence=conf,
        source="YOLO" if raw else "KALMAN-PREDICT",
    )


def test_detects_down_to_up_reversal():
    engine = BounceEngine(
        window=2,
        min_pre_speed=1.0,
        min_post_speed=1.0,
    )

    ys = [80, 90, 102, 90, 78]
    event = None

    for i, y in enumerate(ys):
        event = engine.update(s(i, y))

    assert event is not None
    assert event.frame == 2
    assert event.y == 102
    assert event.pre_velocity_y > 0
    assert event.post_velocity_y < 0


def test_no_bounce_on_monotonic_motion():
    engine = BounceEngine(window=2)

    event = None
    for i, y in enumerate([50, 60, 70, 80, 90, 100]):
        maybe = engine.update(s(i, y))
        if maybe:
            event = maybe

    assert event is None


def test_predicted_center_reduces_source_quality():
    engine = BounceEngine(
        window=2,
        min_pre_speed=1.0,
        min_post_speed=1.0,
    )

    samples = [
        s(0, 80),
        s(1, 90),
        s(2, 102, raw=False, conf=0.4),
        s(3, 90),
        s(4, 78),
    ]

    event = None
    for sample in samples:
        event = engine.update(sample)

    assert event is not None
    assert event.source == "TRACKED"
    assert event.confidence < 1.0


def test_refractory_prevents_duplicate_bounce():
    engine = BounceEngine(
        window=1,
        min_pre_speed=1.0,
        min_post_speed=1.0,
        refractory_frames=5,
    )

    events = []

    for i, y in enumerate([80, 100, 80, 100, 80]):
        e = engine.update(s(i, y))
        if e:
            events.append(e)

    assert len(events) == 1
