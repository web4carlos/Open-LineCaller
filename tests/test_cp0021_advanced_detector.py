import cv2
import numpy as np

from linecaller.ball.advanced_motion_detector import AdvancedMotionBallDetector


def test_detector_returns_list():
    d = AdvancedMotionBallDetector()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    assert isinstance(d.detect(frame), list)


def test_detector_reset_clears_telemetry():
    d = AdvancedMotionBallDetector()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    d.detect(frame)
    assert d.telemetry.frames == 1
    d.reset()
    assert d.telemetry.frames == 0
    assert d.telemetry.accepted == 0


def test_color_affinity_prefers_yellow_green():
    d = AdvancedMotionBallDetector()

    yellow = np.zeros((10, 10, 3), dtype=np.uint8)
    yellow[:] = (0, 255, 255)

    gray = np.full((10, 10, 3), 128, dtype=np.uint8)

    hsv_yellow = cv2.cvtColor(yellow, cv2.COLOR_BGR2HSV)
    hsv_gray = cv2.cvtColor(gray, cv2.COLOR_BGR2HSV)

    assert (
        d._yellow_green_affinity(hsv_yellow)
        > d._yellow_green_affinity(hsv_gray)
    )


def test_temporal_score_prefers_nearby():
    d = AdvancedMotionBallDetector(temporal_radius_px=100.0)
    d._last_best = (50.0, 50.0)

    near = d._temporal_score(60.0, 50.0)
    far = d._temporal_score(200.0, 200.0)

    assert near > far
