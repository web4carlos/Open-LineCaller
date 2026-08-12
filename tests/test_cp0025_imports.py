def test_tracking_imports():
    from linecaller.tracking import (
        BallKalmanTracker,
        DetectionPoint,
        TrackPoint,
    )
    assert BallKalmanTracker is not None
    assert DetectionPoint is not None
    assert TrackPoint is not None
