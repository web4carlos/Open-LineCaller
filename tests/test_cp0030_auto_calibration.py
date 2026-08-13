import cv2
import numpy as np

from linecaller.calibration_center import AutoCourtCalibrator


def synthetic_court():
    frame=np.zeros((720,1280,3),dtype=np.uint8)

    pts=np.array([
        [180,650],
        [1100,650],
        [850,180],
        [430,180],
    ],dtype=np.int32)

    cv2.polylines(
        frame,
        [pts],
        True,
        (255,255,255),
        8,
    )

    return frame


def test_auto_calibrator_returns_result():
    frame=synthetic_court()
    r=AutoCourtCalibrator(min_confidence=0.1).detect(frame)
    assert r.calibration is not None
    assert 0 <= r.confidence <= 1


def test_empty_frame_fails_cleanly():
    r=AutoCourtCalibrator().detect(
        np.zeros((0,0,3),dtype=np.uint8)
    )
    assert r.success is False


def test_blank_frame_has_no_calibration():
    frame=np.zeros((720,1280,3),dtype=np.uint8)
    r=AutoCourtCalibrator().detect(frame)
    assert r.success is False
