from __future__ import annotations

from dataclasses import dataclass

from linecaller.calibration.homography import HomographyEstimator
from linecaller.calibration.profile import CalibrationProfile
from linecaller.calibration.quality import (
    CalibrationQuality,
    CalibrationQualityGate,
)


@dataclass(frozen=True)
class CalibrationBuildResult:
    profile: CalibrationProfile
    quality: CalibrationQuality


class CalibrationService:
    def __init__(self, quality_gate: CalibrationQualityGate | None = None):
        self.quality_gate = quality_gate or CalibrationQualityGate()

    def build(
        self,
        *,
        name: str,
        image_size: tuple[int, int],
        image_points: list[tuple[float, float]],
        court_points: list[tuple[float, float]],
    ) -> CalibrationBuildResult:
        result = HomographyEstimator.estimate(image_points, court_points)

        quality = self.quality_gate.evaluate(
            image_points=image_points,
            court_points=court_points,
            image_to_court_h=result.matrix,
            inlier_count=result.inlier_count,
        )

        profile = CalibrationProfile.create(
            name=name,
            image_size=image_size,
            image_points=image_points,
            court_points=court_points,
            homography=result.matrix,
            status=quality.status,
            mean_error_px=quality.metrics.mean_error_px,
            max_error_px=quality.metrics.max_error_px,
            rms_error_px=quality.metrics.rms_error_px,
        )

        return CalibrationBuildResult(profile=profile, quality=quality)
