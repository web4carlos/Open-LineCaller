from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from PIL import Image, ImageChops


@dataclass(frozen=True)
class LockedResidualColorProfile:
    """Runtime color evidence learned from the locked ball reference.

    Nothing here defines the flight curve or XYZ.  The profile only answers:
    among pixels that changed relative to background, which pixels are
    compatible with the already locked ball appearance?
    """

    hue_center: float          # PIL HSV hue, circular 0..255
    hue_tolerance: float
    saturation_min: int
    value_min: int

    @staticmethod
    def _circular_hue_distance(values: np.ndarray, center: float) -> np.ndarray:
        d = np.abs(values.astype(np.float32) - float(center))
        return np.minimum(d, 256.0 - d)

    @classmethod
    def from_template(cls, template_rgb: Any) -> "LockedResidualColorProfile":
        image = _as_rgb_pil(template_rgb)
        hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
        h, w = hsv.shape[:2]
        if h < 3 or w < 3:
            raise ValueError("ball template is too small")

        yy, xx = np.ogrid[:h, :w]
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        center_radius = max(1.5, 0.46 * min(w, h))
        center = (xx - cx) ** 2 + (yy - cy) ** 2 <= center_radius ** 2

        sat = hsv[..., 1]
        val = hsv[..., 2]

        # Tight references normally contain dark/neutral border/background.
        # Learn the foreground from the reference itself; never hard-code a
        # color name such as yellow/orange/green.
        center_sat = sat[center]
        center_val = val[center]
        sat_gate = max(20.0, float(np.percentile(center_sat, 45)))
        val_gate = max(55.0, float(np.percentile(center_val, 55)))
        fg = center & (sat >= sat_gate) & (val >= val_gate)

        # Fallback for pale/white balls: use the brightest center support.
        if int(fg.sum()) < 5:
            val_gate = max(40.0, float(np.percentile(center_val, 70)))
            fg = center & (val >= val_gate)
        if int(fg.sum()) < 3:
            fg = center

        hues = hsv[..., 0][fg].astype(np.float32)
        weights = (sat[fg].astype(np.float32) + 24.0) * (val[fg].astype(np.float32) + 24.0)
        angles = hues * (2.0 * math.pi / 256.0)
        s = float(np.sum(np.sin(angles) * weights))
        c = float(np.sum(np.cos(angles) * weights))
        center_angle = math.atan2(s, c)
        if center_angle < 0:
            center_angle += 2.0 * math.pi
        hue_center = center_angle * 256.0 / (2.0 * math.pi)

        hue_spread = cls._circular_hue_distance(hues, hue_center)
        p90 = float(np.percentile(hue_spread, 90)) if hue_spread.size else 0.0

        # Runtime motion blur/compression shifts hue more than the tight
        # reference.  Slack is learned around the reference, not a named color.
        hue_tolerance = min(42.0, max(18.0, p90 * 3.0 + 10.0))
        saturation_min = int(max(25.0, float(np.percentile(sat[fg], 10)) - 75.0))
        value_min = int(max(70.0, float(np.percentile(val[fg], 10)) - 145.0))

        return cls(
            hue_center=float(hue_center),
            hue_tolerance=float(hue_tolerance),
            saturation_min=saturation_min,
            value_min=value_min,
        )


@dataclass(frozen=True)
class BallColorResidualConfig:
    difference_threshold: int = 18
    expected_radius_scale: float = 1.35
    min_component_pixels: int = 3
    min_component_fraction_of_ball: float = 0.035

    def __post_init__(self) -> None:
        if not 0 <= self.difference_threshold <= 255:
            raise ValueError("difference_threshold must be in [0,255]")
        if not math.isfinite(self.expected_radius_scale) or self.expected_radius_scale <= 0:
            raise ValueError("expected_radius_scale must be finite and > 0")
        if self.min_component_pixels < 1:
            raise ValueError("min_component_pixels must be >= 1")
        if not 0.0 <= self.min_component_fraction_of_ball <= 1.0:
            raise ValueError("min_component_fraction_of_ball must be in [0,1]")


@dataclass(frozen=True)
class BallColorResidualEvidence:
    residual_present: bool
    component_area_px: int
    component_bbox: tuple[int, int, int, int] | None
    centroid_xy: tuple[float, float] | None
    changed_fraction: float
    color_candidate_fraction: float
    raw_residual_rgb: np.ndarray
    color_mask_u8: np.ndarray
    isolated_rgb: np.ndarray


def _as_rgb_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("image must be RGB-like HxWx3 or PIL RGB")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _largest_centered_component(mask: np.ndarray, cx: float, cy: float):
    """Return the most plausible 8-connected component near expected center."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    best = None

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            pts = []
            while stack:
                px, py = stack.pop()
                pts.append((px, py))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((nx, ny))

            arr = np.asarray(pts, dtype=np.float32)
            mx = float(arr[:, 0].mean())
            my = float(arr[:, 1].mean())
            area = len(pts)
            dist = math.hypot(mx - cx, my - cy)
            # Area rewards coherent ball signal; distance keeps the component
            # tied to the expected same-ball location.
            score = area / (1.0 + 0.35 * dist)
            if best is None or score > best[0]:
                xs = arr[:, 0]
                ys = arr[:, 1]
                best = (
                    score,
                    pts,
                    area,
                    (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                    (mx, my),
                )
    return best


class BallColorResidualIsolator:
    """PIL residual + locked ball-color isolation.

    Pipeline:
      CURRENT - BACKGROUND (PIL ImageChops)
      INTERSECT changed pixels with locked ball-color pixels
      KEEP compact connected component nearest expected same-ball center

    It confirms PRESENCE/WHO evidence only.  It never defines XYZ, V, path or Z0.
    """

    def __init__(
        self,
        profile: LockedResidualColorProfile,
        config: BallColorResidualConfig | None = None,
    ) -> None:
        self.profile = profile
        self.config = config or BallColorResidualConfig()

    def isolate(
        self,
        current_rgb: Any,
        background_rgb: Any,
        *,
        expected_diameter_px: float,
        expected_center_xy: tuple[float, float] | None = None,
    ) -> BallColorResidualEvidence:
        if not math.isfinite(expected_diameter_px) or expected_diameter_px <= 0:
            raise ValueError("expected_diameter_px must be finite and > 0")

        current = _as_rgb_pil(current_rgb)
        background = _as_rgb_pil(background_rgb)
        if current.size != background.size:
            raise ValueError("current and background must be aligned and same size")

        raw_residual = ImageChops.difference(current, background).convert("RGB")
        raw_rgb = np.asarray(raw_residual, dtype=np.uint8)
        gray = np.asarray(raw_residual.convert("L"), dtype=np.uint8)
        hsv = np.asarray(current.convert("HSV"), dtype=np.uint8)
        h, w = gray.shape

        if expected_center_xy is None:
            cx = (w - 1) / 2.0
            cy = (h - 1) / 2.0
        else:
            cx, cy = map(float, expected_center_xy)

        changed = gray >= self.config.difference_threshold
        hue_distance = LockedResidualColorProfile._circular_hue_distance(
            hsv[..., 0], self.profile.hue_center
        )
        color_ok = (
            (hue_distance <= self.profile.hue_tolerance)
            & (hsv[..., 1] >= self.profile.saturation_min)
            & (hsv[..., 2] >= self.profile.value_min)
        )

        yy, xx = np.ogrid[:h, :w]
        r = max(2.0, expected_diameter_px * self.config.expected_radius_scale)
        local = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        candidates = changed & color_ok & local

        best = _largest_centered_component(candidates, cx, cy)
        final_mask = np.zeros((h, w), dtype=bool)
        bbox = None
        centroid = None
        area = 0
        if best is not None:
            _, pts, area, bbox, centroid = best
            for px, py in pts:
                final_mask[py, px] = True

        expected_area = math.pi * (expected_diameter_px / 2.0) ** 2
        minimum_area = max(
            self.config.min_component_pixels,
            int(math.ceil(expected_area * self.config.min_component_fraction_of_ball)),
        )
        residual_present = area >= minimum_area

        isolated = np.zeros_like(np.asarray(current, dtype=np.uint8))
        isolated[final_mask] = np.asarray(current, dtype=np.uint8)[final_mask]

        return BallColorResidualEvidence(
            residual_present=bool(residual_present),
            component_area_px=int(area),
            component_bbox=bbox,
            centroid_xy=centroid,
            changed_fraction=float(changed.mean()),
            color_candidate_fraction=float(candidates.mean()),
            raw_residual_rgb=raw_rgb,
            color_mask_u8=(final_mask.astype(np.uint8) * 255),
            isolated_rgb=isolated,
        )
