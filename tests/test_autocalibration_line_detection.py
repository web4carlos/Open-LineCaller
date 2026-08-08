import cv2
import numpy as np

from linecaller.autocalibration.line_detection import HoughCourtLineDetector


def test_detector_finds_synthetic_court_lines():
    img = np.zeros((600,800,3), dtype=np.uint8)

    cv2.line(img, (100,500), (700,500), (255,255,255), 4)
    cv2.line(img, (200,100), (600,100), (255,255,255), 4)
    cv2.line(img, (100,500), (200,100), (255,255,255), 4)
    cv2.line(img, (700,500), (600,100), (255,255,255), 4)

    detector = HoughCourtLineDetector(
        hough_threshold=40,
        min_line_length=80,
    )
    lines = detector.detect(img)

    assert len(lines) >= 4
