from linecaller.tracking import (
    BallKalmanTracker,
    DetectionPoint,
)

def d(frame, x, y, conf=0.9):
    return DetectionPoint(
        frame=frame,
        x=x,
        y=y,
        confidence=conf,
    )

def test_tracker_initializes_on_detection():
    t = BallKalmanTracker(max_gap=3)
    p = t.update(0, d(0, 100, 200))
    assert p.tracked_x == 100
    assert p.tracked_y == 200
    assert p.source == "YOLO"

def test_tracker_predicts_short_gap():
    t = BallKalmanTracker(max_gap=3)
    t.update(0, d(0, 100, 200))
    t.update(1, d(1, 110, 200))
    p = t.update(2, None)
    assert p.tracked_x is not None
    assert p.tracked_y is not None
    assert p.source == "KALMAN-PREDICT"

def test_tracker_drops_after_gap():
    t = BallKalmanTracker(max_gap=2)
    t.update(0, d(0, 100, 200))
    t.update(1, None)
    t.update(2, None)
    p = t.update(3, None)
    assert p.tracked_x is None
    assert p.source == "LOST"

def test_low_confidence_does_not_initialize():
    t = BallKalmanTracker(
        min_detection_confidence=0.2
    )
    p = t.update(
        0,
        d(0, 100, 200, conf=0.1),
    )
    assert p.tracked_x is None
