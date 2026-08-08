from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np
import cv2


@dataclass(frozen=True)
class HomographyResult:
    matrix: np.ndarray
    inlier_mask: np.ndarray | None

    @property
    def inlier_count(self) -> int:
        if self.inlier_mask is None:
            return 0
        return int(np.asarray(self.inlier_mask).sum())


class HomographyEstimator:
    @staticmethod
    def estimate(
        image_points: Sequence[tuple[float, float]],
        court_points: Sequence[tuple[float, float]],
        ransac_threshold_px: float = 3.0,
    ) -> HomographyResult:
        if len(image_points) != len(court_points):
            raise ValueError("image_points and court_points must have equal length")
        if len(image_points) < 4:
            raise ValueError("At least four point correspondences are required")

        src = np.asarray(image_points, dtype=np.float64)
        dst = np.asarray(court_points, dtype=np.float64)

        matrix, mask = cv2.findHomography(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold_px,
        )

        if matrix is None:
            raise RuntimeError("Homography estimation failed")

        return HomographyResult(matrix=matrix, inlier_mask=mask)

    @staticmethod
    def transform_points(
        points: Sequence[tuple[float, float]],
        matrix: np.ndarray,
    ) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, matrix)
        return transformed.reshape(-1, 2)
