from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import numpy as np
import cv2


class CalibrationStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"


@dataclass(frozen=True)
class CalibrationMetrics:
    mean_error_px: float
    max_error_px: float
    rms_error_px: float
    point_count: int
    inlier_count: int


@dataclass(frozen=True)
class QualityThresholds:
    mean_error_px: float = 3.0
    max_error_px: float = 6.0
    min_reference_points: int = 6
    min_inlier_ratio: float = 0.75


@dataclass(frozen=True)
class CalibrationQuality:
    status: CalibrationStatus
    metrics: CalibrationMetrics
    reasons: tuple[str, ...]


class CalibrationQualityGate:
    def __init__(self, thresholds: QualityThresholds | None = None):
        self.thresholds = thresholds or QualityThresholds()

    @staticmethod
    def _project_court_to_image(
        court_points: np.ndarray,
        image_to_court_h: np.ndarray,
    ) -> np.ndarray:
        inverse = np.linalg.inv(image_to_court_h)
        pts = court_points.reshape(-1, 1, 2).astype(np.float64)
        projected = cv2.perspectiveTransform(pts, inverse)
        return projected.reshape(-1, 2)

    def evaluate(
        self,
        image_points: list[tuple[float, float]],
        court_points: list[tuple[float, float]],
        image_to_court_h: np.ndarray,
        inlier_count: int,
    ) -> CalibrationQuality:
        image_arr = np.asarray(image_points, dtype=np.float64)
        court_arr = np.asarray(court_points, dtype=np.float64)

        projected = self._project_court_to_image(court_arr, image_to_court_h)
        errors = np.linalg.norm(projected - image_arr, axis=1)

        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))
        rms_error = float(math.sqrt(float(np.mean(errors ** 2))))

        metrics = CalibrationMetrics(
            mean_error_px=mean_error,
            max_error_px=max_error,
            rms_error_px=rms_error,
            point_count=len(image_points),
            inlier_count=inlier_count,
        )

        reasons: list[str] = []
        t = self.thresholds

        if len(image_points) < t.min_reference_points:
            reasons.append(
                f"Only {len(image_points)} reference points; "
                f"minimum is {t.min_reference_points}"
            )

        if mean_error > t.mean_error_px:
            reasons.append(
                f"Mean reprojection error {mean_error:.2f}px "
                f"exceeds {t.mean_error_px:.2f}px"
            )

        if max_error > t.max_error_px:
            reasons.append(
                f"Max reprojection error {max_error:.2f}px "
                f"exceeds {t.max_error_px:.2f}px"
            )

        ratio = inlier_count / max(1, len(image_points))
        if ratio < t.min_inlier_ratio:
            reasons.append(
                f"Inlier ratio {ratio:.2%} below "
                f"{t.min_inlier_ratio:.2%}"
            )

        return CalibrationQuality(
            status=CalibrationStatus.INVALID if reasons else CalibrationStatus.VALID,
            metrics=metrics,
            reasons=tuple(reasons),
        )
