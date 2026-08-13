import cv2
import numpy as np

from linecaller.calibration_center.multiframe import (
    MultiFrameCourtCalibrator,
)


def court_frame(dx=0, dy=0):
    f = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8,
    )

    p = np.array(
        [
            [60+dx, 670+dy],
            [1220+dx, 670+dy],
            [840+dx, 240+dy],
            [440+dx, 240+dy],
        ],
        dtype=np.int32,
    )

    cv2.fillConvexPoly(
        f,
        p,
        (180, 80, 30),
    )

    cv2.polylines(
        f,
        [p],
        True,
        (255, 255, 255),
        8,
    )

    return f


def test_multiframe_aggregates_candidates():
    frames = [
        court_frame(0,0),
        court_frame(1,0),
        court_frame(-1,1),
        court_frame(0,-1),
        court_frame(2,0),
    ]

    r = MultiFrameCourtCalibrator(
        min_frame_support=0.2,
        min_line_confidence=0.1,
    ).calibrate_frames(
        frames
    )

    assert r.image_points is not None
    assert r.frames_analyzed == 5
    assert r.frames_with_candidate >= 3


def test_multiframe_reports_line_confidence():
    frames = [
        court_frame(),
        court_frame(1,0),
        court_frame(-1,0),
        court_frame(),
    ]

    r = MultiFrameCourtCalibrator(
        min_frame_support=0.2,
        min_line_confidence=0.1,
    ).calibrate_frames(
        frames
    )

    names = {
        x.name
        for x
        in r.line_confidences
    }

    assert "NEAR_BASELINE" in names
    assert "FAR_BASELINE" in names
    assert "LEFT_SIDELINE" in names
    assert "RIGHT_SIDELINE" in names


def test_no_frames_fails_cleanly():
    r = MultiFrameCourtCalibrator().calibrate_frames(
        []
    )

    assert r.success is False
    assert r.reason == "NO_FRAMES"


def test_blank_frames_fail_cleanly():
    frames = [
        np.zeros(
            (720,1280,3),
            dtype=np.uint8,
        )
        for _ in range(3)
    ]

    r = MultiFrameCourtCalibrator().calibrate_frames(
        frames
    )

    assert r.success is False
