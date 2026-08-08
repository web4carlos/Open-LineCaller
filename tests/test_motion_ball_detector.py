import numpy as np
import cv2

from linecaller.ball.motion_detector import MotionBallDetector


def test_motion_detector_finds_small_moving_object():
    detector = MotionBallDetector(
        min_area=5,
        max_area=500,
        min_radius_px=1,
        max_radius_px=20,
        min_circularity=0.1,
    )

    background = np.zeros((120,160,3), dtype=np.uint8)

    # Warm the subtractor.
    for _ in range(5):
        detector.detect(background)

    frame = background.copy()
    cv2.circle(frame, (80,60), 5, (255,255,255), -1)

    candidates = detector.detect(frame)

    assert candidates
    best = candidates[0]
    assert abs(best.x - 80) < 8
    assert abs(best.y - 60) < 8
