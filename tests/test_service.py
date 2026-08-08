import cv2
import numpy as np

from linecaller.calibration.service import CalibrationService
from linecaller.calibration.quality import CalibrationStatus


def test_service_builds_valid_profile():
    court = np.asarray([
        (0.0, 0.0), (6.0, 0.0), (6.0, 12.0), (0.0, 12.0),
        (0.0, 4.0), (6.0, 4.0), (0.0, 8.0), (6.0, 8.0),
    ], dtype=np.float32)

    image_corners = np.asarray([
        (200.0, 800.0), (1000.0, 800.0),
        (850.0, 150.0), (350.0, 150.0),
    ], dtype=np.float32)

    h = cv2.getPerspectiveTransform(court[:4], image_corners)
    image = cv2.perspectiveTransform(court.reshape(-1,1,2), h).reshape(-1,2)

    result = CalibrationService().build(
        name="test",
        image_size=(1280, 960),
        image_points=[tuple(map(float,p)) for p in image],
        court_points=[tuple(map(float,p)) for p in court],
    )

    assert result.profile.status == CalibrationStatus.VALID
    assert result.quality.metrics.point_count == 8
