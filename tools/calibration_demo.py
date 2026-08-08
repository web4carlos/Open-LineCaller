from __future__ import annotations

import numpy as np
import cv2

from linecaller.calibration.court_model import PickleballCourtModel
from linecaller.calibration.service import CalibrationService


def main() -> None:
    court = PickleballCourtModel()
    refs = court.reference_points()

    names = [
        "near_left_corner",
        "near_right_corner",
        "far_right_corner",
        "far_left_corner",
        "near_nvz_left",
        "near_nvz_right",
        "far_nvz_left",
        "far_nvz_right",
    ]

    court_points = [refs[name] for name in names]

    # Synthetic camera projection: court -> image
    court_corners = np.asarray([
        refs["near_left_corner"],
        refs["near_right_corner"],
        refs["far_right_corner"],
        refs["far_left_corner"],
    ], dtype=np.float32)

    image_corners = np.asarray([
        (220.0, 820.0),
        (1060.0, 810.0),
        (880.0, 160.0),
        (390.0, 170.0),
    ], dtype=np.float32)

    court_to_image = cv2.getPerspectiveTransform(court_corners, image_corners)

    cp = np.asarray(court_points, dtype=np.float32).reshape(-1, 1, 2)
    image_points = cv2.perspectiveTransform(cp, court_to_image).reshape(-1, 2)
    image_points = [tuple(map(float, p)) for p in image_points]

    service = CalibrationService()
    result = service.build(
        name="synthetic_demo",
        image_size=(1280, 960),
        image_points=image_points,
        court_points=court_points,
    )

    q = result.quality
    print(f"Calibration status: {q.status.value}")
    print(f"Mean reprojection error: {q.metrics.mean_error_px:.6f} px")
    print(f"Max reprojection error: {q.metrics.max_error_px:.6f} px")
    print(f"RMS reprojection error: {q.metrics.rms_error_px:.6f} px")
    print(f"Inliers: {q.metrics.inlier_count}/{q.metrics.point_count}")


if __name__ == "__main__":
    main()
