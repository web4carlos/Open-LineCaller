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


def test_valid_bounce_at_local_y_maximum():
    e = BounceEngine(
        window=2,
        min_pre_speed=1,
        min_post_speed=1,
        peak_tolerance_px=3,
    )

    event = None
    seq = [
        s(0, 100, 80),
        s(1, 105, 90),
        s(2, 110, 103),
        s(3, 115, 91),
        s(4, 120, 79),
    ]

    for sample in seq:
        event = e.update(sample)

    assert event is not None
    assert event.frame == 2
    assert event.y == 103


def test_rejects_vertical_reversal_not_at_peak():
    e = BounceEngine(
        window=2,
        min_pre_speed=1,
        min_post_speed=1,
        peak_tolerance_px=2,
    )

    # center frame reverses, but another nearby sample is much lower
    # on screen (larger y), so center is not the local bounce peak.
    seq = [
        s(0, 100, 80),
        s(1, 105, 90),
        s(2, 110, 95),
        s(3, 115, 110),
        s(4, 120, 80),
    ]

    event = None
    for sample in seq:
        maybe = e.update(sample)
        if maybe:
            event = maybe

    assert event is None


def test_rejects_large_horizontal_discontinuity():
    e = BounceEngine(
        window=2,
        min_pre_speed=1,
        min_post_speed=1,
        max_horizontal_jump=10,
    )

    seq = [
        s(0, 100, 80),
        s(1, 102, 90),
        s(2, 104, 103),
        s(3, 170, 90),
        s(4, 240, 78),
    ]

    event = None
    for sample in seq:
        maybe = e.update(sample)
        if maybe:
            event = maybe

    assert event is None


def test_rejects_low_raw_support():
    e = BounceEngine(
        window=2,
        min_pre_speed=1,
        min_post_speed=1,
        min_raw_ratio=0.8,
    )

    seq = [
        s(0, 100, 80, raw=False, conf=0.4),
        s(1, 105, 90, raw=False, conf=0.4),
        s(2, 110, 103, raw=True),
        s(3, 115, 90, raw=False, conf=0.4),
        s(4, 120, 78, raw=False, conf=0.4),
    ]

    event = None
    for sample in seq:
        maybe = e.update(sample)
        if maybe:
            event = maybe

    assert event is None
