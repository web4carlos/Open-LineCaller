import cv2
import numpy as np

from linecaller.calibration.quality import (
    CalibrationQualityGate,
    CalibrationStatus,
)


def _data():
    court = np.asarray([
        (0.0, 0.0), (6.0, 0.0), (6.0, 12.0), (0.0, 12.0),
        (0.0, 4.0), (6.0, 4.0), (0.0, 8.0), (6.0, 8.0),
    ], dtype=np.float32)

    img_corners = np.asarray([
        (100.0, 800.0), (1100.0, 790.0),
        (900.0, 100.0), (300.0, 110.0)
    ], dtype=np.float32)

    h = cv2.getPerspectiveTransform(
        court[:4].astype(np.float32),
        img_corners
    )
    pts = cv2.perspectiveTransform(court.reshape(-1,1,2), h).reshape(-1,2)
    image = [tuple(map(float, p)) for p in pts]
    return image, [tuple(map(float,p)) for p in court]


def test_quality_valid_for_consistent_points():
    image, court = _data()
    h, mask = cv2.findHomography(
        np.asarray(image, dtype=np.float64),
        np.asarray(court, dtype=np.float64),
        cv2.RANSAC,
    )
    q = CalibrationQualityGate().evaluate(
        image, court, h, int(mask.sum())
    )
    assert q.status == CalibrationStatus.VALID
    assert q.metrics.mean_error_px < 0.01


def test_quality_invalid_when_reference_point_is_bad():
    image, court = _data()
    image[-1] = (image[-1][0] + 40.0, image[-1][1] + 40.0)

    h, mask = cv2.findHomography(
        np.asarray(image, dtype=np.float64),
        np.asarray(court, dtype=np.float64),
        cv2.RANSAC,
        3.0,
    )
    q = CalibrationQualityGate().evaluate(
        image, court, h, int(mask.sum())
    )
    assert q.status == CalibrationStatus.INVALID
    assert q.reasons
