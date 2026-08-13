import cv2
import numpy as np

from linecaller.calibration_center import AutoCourtCalibrator


def perspective_court():
    frame = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8,
    )

    # Strong perspective trapezoid, similar to end-court camera.
    pts = np.array(
        [
            [40, 665],
            [1240, 665],
            [840, 250],
            [440, 250],
        ],
        dtype=np.int32,
    )

    cv2.polylines(
        frame,
        [pts],
        True,
        (255, 255, 255),
        9,
    )

    # Center line and kitchen-ish horizontal line should not confuse sidelines.
    cv2.line(
        frame,
        (640, 665),
        (640, 250),
        (255, 255, 255),
        6,
    )

    cv2.line(
        frame,
        (260, 500),
        (1020, 500),
        (255, 255, 255),
        6,
    )

    return frame


def test_perspective_auto_finds_wide_court():
    r = AutoCourtCalibrator(
        min_confidence=0.10
    ).detect(
        perspective_court()
    )

    assert r.calibration is not None

    nl, nr, fr, fl = (
        r.calibration.image_points
    )

    assert nr[0] - nl[0] > 500
    assert fr[0] - fl[0] > 150
    assert nl[1] > fl[1]


def test_perspective_result_order():
    r = AutoCourtCalibrator(
        min_confidence=0.10
    ).detect(
        perspective_court()
    )

    assert r.calibration is not None

    nl, nr, fr, fl = (
        r.calibration.image_points
    )

    assert nl[0] < nr[0]
    assert fl[0] < fr[0]
