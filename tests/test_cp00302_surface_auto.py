import cv2
import numpy as np

from linecaller.calibration_center import AutoCourtCalibrator


def synthetic_blue_court():
    frame = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8,
    )

    # BGR blue court trapezoid
    pts = np.array(
        [
            [60, 670],
            [1220, 670],
            [840, 240],
            [440, 240],
        ],
        dtype=np.int32,
    )

    cv2.fillConvexPoly(
        frame,
        pts,
        (180, 80, 30),
    )

    cv2.polylines(
        frame,
        [pts],
        True,
        (255, 255, 255),
        8,
    )

    return frame


def test_surface_auto_finds_blue_trapezoid():
    r = AutoCourtCalibrator(
        min_confidence=0.10
    ).detect(
        synthetic_blue_court()
    )

    assert r.calibration is not None

    nl, nr, fr, fl = r.calibration.image_points

    assert nr[0] - nl[0] > 700
    assert fr[0] - fl[0] > 200
    assert nl[1] > fl[1]


def test_surface_auto_blank_fails_safely():
    frame = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8,
    )

    r = AutoCourtCalibrator().detect(frame)

    assert r.success is False
