from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .ball_color_residual_isolator import BallColorResidualEvidence


@dataclass(frozen=True)
class ResidualBallProjection:
    """2-D projection evidence extracted from an irregular ball residual.

    This object intentionally stays in image space.  It does not claim XYZ,
    depth, a flight curve, or Z0.  It measures where the projected ball
    evidence is and how large/elongated that evidence appears in the frame.
    """

    valid: bool
    center_xy_local: tuple[float, float] | None
    center_xy_global: tuple[float, float] | None
    area_equivalent_diameter_px: float | None
    bbox_width_px: float | None
    bbox_height_px: float | None
    moment_major_diameter_px: float | None
    moment_minor_diameter_px: float | None
    elongation: float | None
    support_pixels: int

    @property
    def apparent_scale_px(self) -> float | None:
        """Conservative scalar summary of projected support size.

        It is deliberately an image-space evidence measure, not a physical
        ball diameter.  The residual can be partial/non-circular because of
        blur, compression, holes, lighting, or background subtraction.
        """
        if not self.valid:
            return None
        vals = [
            v for v in (
                self.area_equivalent_diameter_px,
                self.moment_minor_diameter_px,
                self.moment_major_diameter_px,
            )
            if v is not None and math.isfinite(v) and v > 0
        ]
        if not vals:
            return None
        vals.sort()
        return float(vals[len(vals) // 2])


class ResidualBallProjectionEstimator:
    """Measure center + apparent 2-D scale from RC7.1 residual evidence.

    No circle assumption is required.  The center is intensity-weighted by
    the actual PIL residual.  Shape scale is reported several ways:
      * area-equivalent diameter
      * moment-equivalent major/minor diameters
      * bounding-box width/height

    Those are projection measurements only.  A later camera/physics resolver
    may decide how to use them for continuous XYZ.
    """

    @staticmethod
    def _gray_weights(rgb: np.ndarray) -> np.ndarray:
        arr = np.asarray(rgb, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("raw residual must be HxWx3 RGB")
        return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

    def measure(
        self,
        evidence: BallColorResidualEvidence,
        *,
        roi_origin_xy: tuple[float, float] = (0.0, 0.0),
    ) -> ResidualBallProjection:
        mask = np.asarray(evidence.color_mask_u8) > 0
        if mask.ndim != 2:
            raise ValueError("color mask must be 2-D")
        support = int(mask.sum())
        if not evidence.residual_present or support < 1:
            return ResidualBallProjection(
                valid=False,
                center_xy_local=None,
                center_xy_global=None,
                area_equivalent_diameter_px=None,
                bbox_width_px=None,
                bbox_height_px=None,
                moment_major_diameter_px=None,
                moment_minor_diameter_px=None,
                elongation=None,
                support_pixels=support,
            )

        yy, xx = np.nonzero(mask)
        gray = self._gray_weights(evidence.raw_residual_rgb)
        weights = gray[yy, xx].astype(np.float64)
        # If difference energy is extremely weak, do not lose the spatial
        # support already validated by RC7.1; use uniform component weights.
        if not np.isfinite(weights).all() or float(weights.sum()) <= 1e-9:
            weights = np.ones_like(weights, dtype=np.float64)
        else:
            weights = np.maximum(weights, 1.0)

        wsum = float(weights.sum())
        cx = float(np.sum(xx * weights) / wsum)
        cy = float(np.sum(yy * weights) / wsum)

        dx = xx.astype(np.float64) - cx
        dy = yy.astype(np.float64) - cy
        cxx = float(np.sum(weights * dx * dx) / wsum)
        cyy = float(np.sum(weights * dy * dy) / wsum)
        cxy = float(np.sum(weights * dx * dy) / wsum)
        cov = np.array([[cxx, cxy], [cxy, cyy]], dtype=np.float64)
        eig = np.linalg.eigvalsh(cov)
        eig = np.maximum(eig, 0.0)
        minor_var = float(eig[0])
        major_var = float(eig[1])

        # For a uniformly filled circle/ellipse, variance along an axis is
        # radius^2 / 4.  Therefore 4*sqrt(var) is the moment-equivalent
        # diameter.  It remains useful for irregular/partial projections.
        minor_d = 4.0 * math.sqrt(minor_var) if minor_var > 0 else 0.0
        major_d = 4.0 * math.sqrt(major_var) if major_var > 0 else 0.0

        x1 = int(xx.min())
        x2 = int(xx.max()) + 1
        y1 = int(yy.min())
        y2 = int(yy.max()) + 1
        bbox_w = float(x2 - x1)
        bbox_h = float(y2 - y1)
        area_d = 2.0 * math.sqrt(support / math.pi)
        elong = (major_d / minor_d) if minor_d > 1e-9 else math.inf

        ox, oy = map(float, roi_origin_xy)
        return ResidualBallProjection(
            valid=True,
            center_xy_local=(cx, cy),
            center_xy_global=(ox + cx, oy + cy),
            area_equivalent_diameter_px=float(area_d),
            bbox_width_px=bbox_w,
            bbox_height_px=bbox_h,
            moment_major_diameter_px=float(major_d),
            moment_minor_diameter_px=float(minor_d),
            elongation=float(elong),
            support_pixels=support,
        )
