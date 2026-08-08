import numpy as np

from linecaller.autocalibration.line_detection import (
    HoughCourtLineDetector,
)


def test_hough_normalization_accepts_detector_output(monkeypatch):
    import linecaller.autocalibration.line_detection as module

    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Simulate OpenCV variant returning (N,4)
    monkeypatch.setattr(
        module.cv2,
        "HoughLinesP",
        lambda *args, **kwargs: np.array([
            [0, 0, 50, 0],
            [0, 0, 0, 50],
        ], dtype=np.int32),
    )

    detector = HoughCourtLineDetector()
    result = detector.detect(image)

    assert len(result) == 2
    assert result[0].length_px == 50.0
    assert result[1].length_px == 50.0


def test_hough_normalization_accepts_n_1_4(monkeypatch):
    import linecaller.autocalibration.line_detection as module

    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Simulate classic OpenCV return shape (N,1,4)
    monkeypatch.setattr(
        module.cv2,
        "HoughLinesP",
        lambda *args, **kwargs: np.array([
            [[0, 0, 50, 0]],
            [[0, 0, 0, 50]],
        ], dtype=np.int32),
    )

    detector = HoughCourtLineDetector()
    result = detector.detect(image)

    assert len(result) == 2
