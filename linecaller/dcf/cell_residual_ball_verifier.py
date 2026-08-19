from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from PIL import Image, ImageChops


@dataclass(frozen=True)
class CellResidualConfig:
    """PIL-based foreground evidence for one aligned cell/ROI.

    This verifier answers only: "did a compact foreground change occur where
    the already-identified ball is expected?"  It does NOT identify the ball,
    does NOT define P(x,y,z), and does NOT define a flight curve.
    """

    difference_threshold: int = 20
    center_radius_scale: float = 0.75
    min_center_changed_fraction: float = 0.45
    min_center_mean_difference: float = 28.0

    def __post_init__(self) -> None:
        if not 0 <= self.difference_threshold <= 255:
            raise ValueError("difference_threshold must be in [0,255]")
        if not math.isfinite(self.center_radius_scale) or self.center_radius_scale <= 0:
            raise ValueError("center_radius_scale must be finite and > 0")
        if not 0.0 <= self.min_center_changed_fraction <= 1.0:
            raise ValueError("min_center_changed_fraction must be in [0,1]")
        if not 0.0 <= self.min_center_mean_difference <= 255.0:
            raise ValueError("min_center_mean_difference must be in [0,255]")


@dataclass(frozen=True)
class CellResidualEvidence:
    residual_present: bool
    center_changed_fraction: float
    center_mean_difference: float
    full_changed_fraction: float
    full_mean_difference: float
    changed_bbox: tuple[int, int, int, int] | None
    residual_rgb: np.ndarray


class CellResidualBallVerifier:
    """Compare CURRENT aligned cell image against its BACKGROUND using PIL.

    Contract:
      current aligned crop - background aligned crop -> residual evidence

    Ball Identity remains a separate upstream WHO check.  This class only
    confirms that something changed in the expected ball-centered region.
    """

    def __init__(self, config: CellResidualConfig | None = None) -> None:
        self.config = config or CellResidualConfig()

    @staticmethod
    def _as_rgb_pil(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("image must be RGB-like HxWx3 or a PIL image")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def verify(
        self,
        current_rgb: Any,
        background_rgb: Any,
        *,
        expected_diameter_px: float,
    ) -> CellResidualEvidence:
        if not math.isfinite(expected_diameter_px) or expected_diameter_px <= 0.0:
            raise ValueError("expected_diameter_px must be finite and > 0")

        current = self._as_rgb_pil(current_rgb)
        background = self._as_rgb_pil(background_rgb)
        if current.size != background.size:
            raise ValueError("current and background images must be aligned and same size")

        residual = ImageChops.difference(current, background)
        gray = np.asarray(residual.convert("L"), dtype=np.uint8)
        if gray.size == 0:
            raise ValueError("empty aligned crop")

        changed = gray >= self.config.difference_threshold
        h, w = gray.shape
        yy, xx = np.ogrid[:h, :w]
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        radius = max(1.0, expected_diameter_px * self.config.center_radius_scale)
        center = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        if not bool(center.any()):
            raise ValueError("center support is empty")

        center_changed_fraction = float(changed[center].mean())
        center_mean_difference = float(gray[center].mean())
        full_changed_fraction = float(changed.mean())
        full_mean_difference = float(gray.mean())

        bbox = residual.getbbox()
        residual_present = (
            center_changed_fraction >= self.config.min_center_changed_fraction
            and center_mean_difference >= self.config.min_center_mean_difference
        )

        return CellResidualEvidence(
            residual_present=bool(residual_present),
            center_changed_fraction=center_changed_fraction,
            center_mean_difference=center_mean_difference,
            full_changed_fraction=full_changed_fraction,
            full_mean_difference=full_mean_difference,
            changed_bbox=None if bbox is None else tuple(int(v) for v in bbox),
            residual_rgb=np.asarray(residual, dtype=np.uint8),
        )
