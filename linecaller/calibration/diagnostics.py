from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2


@dataclass(frozen=True)
class PointDiagnostic:
    index: int
    error_px: float


def reprojection_diagnostics(
    image_points: list[tuple[float, float]],
    court_points: list[tuple[float, float]],
    image_to_court_h: np.ndarray,
) -> tuple[PointDiagnostic, ...]:
    inverse = np.linalg.inv(np.asarray(image_to_court_h, dtype=np.float64))

    court = np.asarray(court_points, dtype=np.float64).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(court, inverse).reshape(-1, 2)
    image = np.asarray(image_points, dtype=np.float64)

    errors = np.linalg.norm(projected - image, axis=1)

    return tuple(
        PointDiagnostic(index=i, error_px=float(err))
        for i, err in enumerate(errors)
    )
