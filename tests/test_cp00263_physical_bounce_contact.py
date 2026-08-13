from linecaller.bounce_v2.contact_validation import (
    PhysicalBounceContactValidator,
    TrackSample,
)


def s(frame, y, raw=True, x=100, conf=.9):
    return TrackSample(
        frame=frame,
        raw_x=x if raw else None,
        raw_y=y if raw else None,
        tracked_x=x,
        tracked_y=y,
        confidence=conf,
        source="YOLO" if raw else "KALMAN-PREDICT",
    )


def test_physical_local_maximum_is_accepted():
    samples = [
        s(8, 80),
        s(9, 90),
        s(10, 100),
        s(11, 90),
        s(12, 80),
    ]

    r = PhysicalBounceContactValidator(
        window=3,
        min_score=.20,
        min_reversal_strength=1.0,
    ).validate(
        event_frame=10,
        event_x=100,
        event_y=100,
        samples=samples,
    )

    assert r.accepted is True
    assert r.contact_frame == 10


def test_monotonic_motion_is_rejected():
    samples = [
        s(8, 70),
        s(9, 80),
        s(10, 90),
        s(11, 100),
        s(12, 110),
    ]

    r = PhysicalBounceContactValidator(
        window=3
    ).validate(
        event_frame=10,
        event_x=100,
        event_y=90,
        samples=samples,
    )

    assert r.accepted is False


def test_large_event_contact_mismatch_is_rejected():
    samples = [
        s(8, 80, x=100),
        s(9, 90, x=100),
        s(10, 100, x=100),
        s(11, 90, x=100),
        s(12, 80, x=100),
    ]

    r = PhysicalBounceContactValidator(
        window=3,
        min_score=.20,
        min_reversal_strength=1.0,
        max_localization_shift_px=20,
    ).validate(
        event_frame=10,
        event_x=300,
        event_y=300,
        samples=samples,
    )

    assert r.accepted is False
    assert r.reason == "EVENT_CONTACT_MISMATCH"


def test_insufficient_raw_support_is_rejected():
    samples = [
        s(8, 80, raw=False),
        s(9, 90, raw=False),
        s(10, 100, raw=True),
        s(11, 90, raw=False),
        s(12, 80, raw=False),
    ]

    r = PhysicalBounceContactValidator(
        window=3,
        min_raw_ratio=.5,
        min_score=.1,
        min_reversal_strength=1.0,
    ).validate(
        event_frame=10,
        event_x=100,
        event_y=100,
        samples=samples,
    )

    assert r.accepted is False
    assert r.reason == "INSUFFICIENT_RAW_DETECTION_SUPPORT"
