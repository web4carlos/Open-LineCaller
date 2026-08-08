from __future__ import annotations

import math
import numpy as np
import cv2

from linecaller.bounce.models import BounceEvent
from linecaller.calibration.profile import CalibrationProfile
from linecaller.calibration.quality import CalibrationStatus
from .geometry import CourtBoundaryGeometry
from .models import DecisionContext


class CourtFusion:
    def __init__(
        self,
        geometry: CourtBoundaryGeometry | None = None,
        bounce_localization_error_px: float = 2.0,
    ):
        self.geometry = geometry or CourtBoundaryGeometry()
        self.bounce_localization_error_px = float(bounce_localization_error_px)

    @staticmethod
    def _project_one(x: float, y: float, h: np.ndarray) -> tuple[float, float]:
        pt = np.asarray([[[x, y]]], dtype=np.float64)
        out = cv2.perspectiveTransform(pt, h)[0, 0]
        return float(out[0]), float(out[1])

    @classmethod
    def _local_scale_m_per_px(
        cls,
        x: float,
        y: float,
        h: np.ndarray,
    ) -> float:
        p = np.asarray([x, y], dtype=np.float64)
        p_x = np.asarray([x + 1.0, y], dtype=np.float64)
        p_y = np.asarray([x, y + 1.0], dtype=np.float64)

        q = np.asarray(cls._project_one(*p, h))
        qx = np.asarray(cls._project_one(*p_x, h))
        qy = np.asarray(cls._project_one(*p_y, h))

        sx = float(np.linalg.norm(qx - q))
        sy = float(np.linalg.norm(qy - q))

        scale = max(sx, sy)

        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("Invalid local pixel-to-meter scale")

        return scale

    def build_context(
        self,
        bounce: BounceEvent,
        calibration: CalibrationProfile,
    ) -> DecisionContext:
        calibration_valid = calibration.status == CalibrationStatus.VALID

        if not calibration_valid:
            return DecisionContext(
                bounce_frame=bounce.frame_number,
                image_x=bounce.x,
                image_y=bounce.y,
                court_x_m=None,
                court_y_m=None,
                nearest_line=None,
                signed_distance_m=None,
                local_m_per_px=None,
                calibration_error_px=calibration.mean_error_px,
                bounce_error_px=self.bounce_localization_error_px,
                total_uncertainty_m=None,
                bounce_confidence=bounce.confidence,
                calibration_valid=False,
                reasons=("Calibration is INVALID",),
            )

        try:
            h = np.asarray(calibration.homography, dtype=np.float64)
            if h.shape != (3, 3):
                raise ValueError("Homography must be 3x3")

            court_x, court_y = self._project_one(bounce.x, bounce.y, h)

            if not (math.isfinite(court_x) and math.isfinite(court_y)):
                raise ValueError("Projected court coordinate is non-finite")

            local_scale = self._local_scale_m_per_px(bounce.x, bounce.y, h)

            combined_px = math.sqrt(
                float(calibration.mean_error_px) ** 2
                + self.bounce_localization_error_px ** 2
            )
            uncertainty_m = combined_px * local_scale

            boundary = self.geometry.signed_distance(court_x, court_y)

            return DecisionContext(
                bounce_frame=bounce.frame_number,
                image_x=bounce.x,
                image_y=bounce.y,
                court_x_m=court_x,
                court_y_m=court_y,
                nearest_line=boundary.nearest_line,
                signed_distance_m=boundary.signed_distance_m,
                local_m_per_px=local_scale,
                calibration_error_px=float(calibration.mean_error_px),
                bounce_error_px=self.bounce_localization_error_px,
                total_uncertainty_m=uncertainty_m,
                bounce_confidence=bounce.confidence,
                calibration_valid=True,
                reasons=(),
            )

        except Exception as exc:
            return DecisionContext(
                bounce_frame=bounce.frame_number,
                image_x=bounce.x,
                image_y=bounce.y,
                court_x_m=None,
                court_y_m=None,
                nearest_line=None,
                signed_distance_m=None,
                local_m_per_px=None,
                calibration_error_px=float(calibration.mean_error_px),
                bounce_error_px=self.bounce_localization_error_px,
                total_uncertainty_m=None,
                bounce_confidence=bounce.confidence,
                calibration_valid=True,
                reasons=(f"Geometry fusion failed: {exc}",),
            )
