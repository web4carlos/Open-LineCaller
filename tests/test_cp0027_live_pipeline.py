from linecaller.live import LiveRefereePipeline
from linecaller.tracking import DetectionPoint


def d(frame, x, y, conf=0.9):
    return DetectionPoint(
        frame=frame,
        x=x,
        y=y,
        confidence=conf,
    )


def test_live_pipeline_tracks_detection():
    p = LiveRefereePipeline()

    r = p.update(
        0,
        d(0, 100, 200),
    )

    assert r.tracked_x == 100
    assert r.tracked_y == 200
    assert r.track_source == "YOLO"


def test_live_pipeline_predicts_gap():
    p = LiveRefereePipeline(
        max_gap=3
    )

    p.update(
        0,
        d(0, 100, 200),
    )

    p.update(
        1,
        d(1, 110, 200),
    )

    r = p.update(
        2,
        None,
    )

    assert r.tracked_x is not None
    assert r.track_source == "KALMAN-PREDICT"


def test_live_pipeline_reset():
    p = LiveRefereePipeline()

    p.update(
        0,
        d(0, 100, 200),
    )

    p.reset()

    r = p.update(
        1,
        None,
    )

    assert r.tracked_x is None
