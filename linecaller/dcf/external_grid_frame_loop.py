from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw


class CalibrationCoverage(str, Enum):
    FULL_COURT = "FULL_COURT"
    HALF_COURT = "HALF_COURT"

    @classmethod
    def parse(cls, value: str | "CalibrationCoverage" | None) -> "CalibrationCoverage":
        if isinstance(value, cls):
            return value
        token = str(value or "FULL_COURT").strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "FULL": cls.FULL_COURT,
            "FULLCOURT": cls.FULL_COURT,
            "FULL_COURT": cls.FULL_COURT,
            "WHOLE_COURT": cls.FULL_COURT,
            "HALF": cls.HALF_COURT,
            "HALFCOURT": cls.HALF_COURT,
            "HALF_COURT": cls.HALF_COURT,
            "HALF_NEAR": cls.HALF_COURT,
            "HALF_FAR": cls.HALF_COURT,
        }
        try:
            return aliases[token]
        except KeyError as exc:
            raise ValueError(f"Unsupported calibration coverage: {value!r}") from exc


@dataclass(frozen=True)
class ExternalGridConfig:
    # One logical floor cell = one Ball Unit (one ball diameter).
    court_x_bu: float = 83.0
    full_court_y_bu: float = 182.0
    half_court_y_bu: float = 91.0
    cell_bu: float = 1.0
    margin_bu: float = 10.0
    min_visible_bbox_px: int = 1

    def court_y_bu(self, coverage: CalibrationCoverage) -> float:
        return self.full_court_y_bu if coverage is CalibrationCoverage.FULL_COURT else self.half_court_y_bu


@dataclass(frozen=True)
class ExternalGridCell:
    cell_id: int
    ix: int
    iy: int
    region: str
    top_view_rect_bu: tuple[float, float, float, float]
    polygon_image: tuple[tuple[float, float], ...]
    bbox_image: tuple[int, int, int, int]
    expected_floor_ball_diameter_px: float


@dataclass(frozen=True)
class ExternalGridCalibration:
    version: int
    coverage: str
    image_size: tuple[int, int]
    image_points: tuple[tuple[float, float], ...]
    config: ExternalGridConfig
    background_image: str
    image_up_unit: tuple[float, float]
    cells: tuple[ExternalGridCell, ...]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.version,
            "coverage": self.coverage,
            "image_size": list(self.image_size),
            "image_points": [list(q) for q in self.image_points],
            "config": asdict(self.config),
            "background_image": self.background_image,
            "image_up_unit": list(self.image_up_unit),
            "cells": [
                {
                    "cell_id": c.cell_id,
                    "ix": c.ix,
                    "iy": c.iy,
                    "region": c.region,
                    "top_view_rect_bu": list(c.top_view_rect_bu),
                    "polygon_image": [list(q) for q in c.polygon_image],
                    "bbox_image": list(c.bbox_image),
                    "expected_floor_ball_diameter_px": c.expected_floor_ball_diameter_px,
                }
                for c in self.cells
            ],
        }
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ExternalGridCalibration":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        cfg = ExternalGridConfig(**data["config"])
        cells = tuple(
            ExternalGridCell(
                cell_id=int(c["cell_id"]),
                ix=int(c["ix"]),
                iy=int(c["iy"]),
                region=str(c["region"]),
                top_view_rect_bu=tuple(float(v) for v in c["top_view_rect_bu"]),
                polygon_image=tuple(tuple(float(v) for v in q) for q in c["polygon_image"]),
                bbox_image=tuple(int(v) for v in c["bbox_image"]),
                expected_floor_ball_diameter_px=float(c["expected_floor_ball_diameter_px"]),
            )
            for c in data["cells"]
        )
        # v1 compatibility: old experimental calibration did not persist gravity-up.
        raw_up = data.get("image_up_unit", (0.0, -1.0))
        ux, uy = float(raw_up[0]), float(raw_up[1])
        norm = math.hypot(ux, uy)
        if norm <= 1e-9:
            ux, uy, norm = 0.0, -1.0, 1.0
        return cls(
            version=int(data.get("version", 1)),
            coverage=str(data["coverage"]),
            image_size=tuple(int(v) for v in data["image_size"]),
            image_points=tuple(tuple(float(v) for v in q) for q in data["image_points"]),
            config=cfg,
            background_image=str(data["background_image"]),
            image_up_unit=(ux / norm, uy / norm),
            cells=cells,
        )


class ExternalGridCalibrator:
    """CALIBRATION-TIME ONLY.

    The external grid is built first in 2D top view and then projected through
    the court homography into the camera image. Runtime consumes this immutable
    calibration; it does not construct or move the grid.

    `image_up_unit` is also calibration-owned. It is the image direction that
    represents physical UP for the installed camera. For the current engineering
    gate, an upright camera uses (0,-1).
    """

    def __init__(
        self,
        image_points: Iterable[Iterable[float]],
        image_size: tuple[int, int],
        *,
        coverage: CalibrationCoverage | str = CalibrationCoverage.FULL_COURT,
        config: ExternalGridConfig | None = None,
        image_up_unit: tuple[float, float] = (0.0, -1.0),
    ) -> None:
        self.coverage = CalibrationCoverage.parse(coverage)
        self.config = config or ExternalGridConfig()
        self.image_size = (int(image_size[0]), int(image_size[1]))
        pts = np.asarray(tuple(tuple(float(v) for v in p) for p in image_points), dtype=np.float32)
        if pts.shape != (4, 2):
            raise ValueError("image_points must be 4x2: near-left, near-right, far-right, far-left")
        self.image_points = pts
        ux, uy = float(image_up_unit[0]), float(image_up_unit[1])
        norm = math.hypot(ux, uy)
        if norm <= 1e-9:
            raise ValueError("image_up_unit must be non-zero")
        self.image_up_unit = (ux / norm, uy / norm)
        court_y = self.config.court_y_bu(self.coverage)

        # IMPORTANT: use the SAME court-coordinate convention as LineCaller
        # CourtGeometry.H.  In court space, Y=0 is the FAR baseline and
        # Y=court_y is the NEAR baseline.  The supplied image points are:
        # near-left, near-right, far-right, far-left.
        #
        # We therefore do not invent a camera angle and we do not draw image
        # rectangles.  The grid is a true 2D top-view floor lattice and every
        # one of its four vertices is projected by this homography.
        src = np.asarray(
            (
                (0.0, court_y),
                (self.config.court_x_bu, court_y),
                (self.config.court_x_bu, 0.0),
                (0.0, 0.0),
            ),
            dtype=np.float32,
        )
        self._top_to_image = cv2.getPerspectiveTransform(src, pts)

    def project_top_view(self, x_bu: float, y_bu: float) -> tuple[float, float]:
        p = np.asarray([[[float(x_bu), float(y_bu)]]], dtype=np.float32)
        q = cv2.perspectiveTransform(p, self._top_to_image)[0, 0]
        return float(q[0]), float(q[1])

    @staticmethod
    def _region(cx: float, cy: float, court_x: float, court_y: float) -> str:
        left, right = cx < 0.0, cx > court_x

        # Court-space convention used by this calibrator:
        #   Y=0       -> FAR baseline
        #   Y=court_y -> NEAR baseline
        # Therefore negative Y is OUT_FAR and Y beyond court_y is OUT_NEAR.
        far, near = cy < 0.0, cy > court_y

        if (left or right) and (near or far):
            return "OUT_CORNER"
        if left:
            return "OUT_LEFT"
        if right:
            return "OUT_RIGHT"
        if far:
            return "OUT_FAR"
        if near:
            return "OUT_NEAR"
        raise ValueError("internal cell is not part of the external grid")

    def build(self, *, background_image: str) -> ExternalGridCalibration:
        cfg = self.config
        court_x = cfg.court_x_bu
        court_y = cfg.court_y_bu(self.coverage)
        step = cfg.cell_bu
        if step <= 0 or cfg.margin_bu <= 0:
            raise ValueError("cell_bu and margin_bu must be > 0")

        min_ix = math.floor(-cfg.margin_bu / step)
        max_ix = math.ceil((court_x + cfg.margin_bu) / step) - 1
        min_iy = math.floor(-cfg.margin_bu / step)
        max_iy = math.ceil((court_y + cfg.margin_bu) / step) - 1
        width, height = self.image_size
        cells: list[ExternalGridCell] = []
        cid = 0

        for iy in range(min_iy, max_iy + 1):
            y0, y1 = iy * step, (iy + 1) * step
            for ix in range(min_ix, max_ix + 1):
                x0, x1 = ix * step, (ix + 1) * step
                # Only the EXTERNAL floor is watched by this grid.
                if x0 >= 0.0 and x1 <= court_x and y0 >= 0.0 and y1 <= court_y:
                    continue

                cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
                region = self._region(cx, cy, court_x, court_y)
                poly = tuple(self.project_top_view(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                bx0 = max(0, int(math.floor(min(xs))) - 1)
                by0 = max(0, int(math.floor(min(ys))) - 1)
                bx1 = min(width, int(math.ceil(max(xs))) + 2)
                by1 = min(height, int(math.ceil(max(ys))) + 2)
                if bx1 - bx0 < cfg.min_visible_bbox_px or by1 - by0 < cfg.min_visible_bbox_px:
                    continue

                # One floor cell = one Ball Unit. Its local projected lateral
                # width is calibration evidence for the ball scale at contact.
                top_w = math.dist(poly[0], poly[1]) / step
                bot_w = math.dist(poly[3], poly[2]) / step
                expected_d = max(0.5, 0.5 * (top_w + bot_w))
                cells.append(
                    ExternalGridCell(
                        cell_id=cid,
                        ix=ix,
                        iy=iy,
                        region=region,
                        top_view_rect_bu=(x0, y0, x1, y1),
                        polygon_image=poly,
                        bbox_image=(bx0, by0, bx1, by1),
                        expected_floor_ball_diameter_px=expected_d,
                    )
                )
                cid += 1

        return ExternalGridCalibration(
            version=2,
            coverage=self.coverage.value,
            image_size=self.image_size,
            image_points=tuple(tuple(float(v) for v in p) for p in self.image_points),
            config=cfg,
            background_image=str(background_image),
            image_up_unit=self.image_up_unit,
            cells=tuple(cells),
        )


def rasterized_scale_bounds_px(
    expected_px: float,
    min_ratio: float,
    max_ratio: float,
    *,
    tiny_expected_threshold_px: float = 3.0,
    quantization_tolerance_px: float = 0.5,
) -> tuple[float, float]:
    # CP-0036.2.4.4: only tiny projected balls receive integer-raster allowance.
    expected = max(0.5, float(expected_px))
    lo = float(min_ratio) * expected
    hi = float(max_ratio) * expected
    if expected < float(tiny_expected_threshold_px):
        q = max(0.0, float(quantization_tolerance_px))
        lo = max(0.0, lo - q)
        hi = hi + q
    return float(lo), float(hi)


def rasterized_scale_compatible(
    observed_px: float,
    expected_px: float,
    min_ratio: float,
    max_ratio: float,
) -> bool:
    lo, hi = rasterized_scale_bounds_px(
        expected_px,
        min_ratio,
        max_ratio,
    )
    observed = float(observed_px)
    return lo <= observed <= hi


@dataclass(frozen=True)
class LockedBallColorProfile:
    hue_center: float
    hue_tolerance: float
    saturation_min: int
    value_min: int
    template_fill_ratio: float = 0.65

    @staticmethod
    def _hue_distance(values: np.ndarray, center: float) -> np.ndarray:
        d = np.abs(values.astype(np.float32) - float(center))
        return np.minimum(d, 256.0 - d)

    @classmethod
    def from_template(cls, image: Any) -> "LockedBallColorProfile":
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
        image = image.convert("RGB")
        hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
        h, w = hsv.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        center = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.46 * min(w, h)) ** 2
        sat, val = hsv[..., 1], hsv[..., 2]
        fg = center & (sat >= max(20, np.percentile(sat[center], 45))) & (val >= max(55, np.percentile(val[center], 55)))
        if int(fg.sum()) < 4:
            fg = center & (val >= max(40, np.percentile(val[center], 70)))
        if int(fg.sum()) < 3:
            fg = center
        hues = hsv[..., 0][fg].astype(np.float32)
        weights = (sat[fg].astype(np.float32) + 24.0) * (val[fg].astype(np.float32) + 24.0)
        angles = hues * (2.0 * math.pi / 256.0)
        s = float(np.sum(np.sin(angles) * weights))
        c = float(np.sum(np.cos(angles) * weights))
        angle = math.atan2(s, c)
        if angle < 0:
            angle += 2.0 * math.pi
        hc = angle * 256.0 / (2.0 * math.pi)
        spread = cls._hue_distance(hues, hc)
        p90 = float(np.percentile(spread, 90)) if spread.size else 0.0
        fg_y, fg_x = np.nonzero(fg)
        if fg_x.size:
            tight_fill_area = max(
                1,
                (int(fg_x.max()) - int(fg_x.min()) + 1)
                * (int(fg_y.max()) - int(fg_y.min()) + 1),
            )
            template_fill_ratio = float(fg.sum()) / float(tight_fill_area)
        else:
            template_fill_ratio = 0.65
        return cls(
            hue_center=hc,
            hue_tolerance=min(16.0, max(5.0, p90 * 2.0 + 2.0)),
            saturation_min=int(max(35.0, float(np.percentile(sat[fg], 10)) - 20.0)),
            value_min=int(max(100.0, float(np.percentile(val[fg], 10)) - 55.0)),
            template_fill_ratio=template_fill_ratio,
        )


@dataclass(frozen=True)
class BallComponent:
    bbox: tuple[int, int, int, int]
    area: int
    centroid_xy: tuple[float, float]
    scale_px: float
    # Number of raw same-color fragments merged into this physical footprint.
    # A fast ball can appear as core + halo/streak in one exposure.
    merged_from: int = 1
    # CP-0036.2.4.2: a recovered component temporarily relaxes selected-ball
    # appearance only inside a trajectory-predicted projected-Z0 neighborhood.
    recovered: bool = False
    recovery_age: int = 0


@dataclass(frozen=True)
class Z0Candidate:
    frame_no: int
    cell_id: int
    region: str
    color_pixels: int
    centroid_xy: tuple[float, float]
    observed_scale_px: float
    expected_floor_scale_px: float
    scale_ratio: float
    floor_xy_bu: tuple[float, float] = (0.0, 0.0)
    boundary_clearance_bu: float = 0.0


@dataclass(frozen=True)
class UpConfirmation:
    contact_frame: int
    confirm_frame: int
    cell_id: int
    region: str
    contact_xy: tuple[float, float]
    above_xy: tuple[float, float]
    up_bu: float
    lateral_bu: float
    scale_ratio_after: float


@dataclass(frozen=True)
class ExternalFrameResult:
    frame_no: int
    stage_trace: tuple[str, ...]
    changed_pixels: int
    ball_color_changed_pixels: int
    z0_candidates: tuple[Z0Candidate, ...]
    up_confirmations: tuple[UpConfirmation, ...]
    raw_ball_components: int = 0
    ball_footprints: int = 0
    merged_motion_footprints: int = 0
    boundary_guard_rejections: int = 0
    projected_signature_matches: int = 0
    projected_signature_rejections: int = 0
    approach_rejections: int = 0
    motion_history_observations: int = 0
    predicted_z0_cell: int | None = None
    approach_reason: str | None = None
    approach_total_motion_px: float = 0.0
    approach_min_motion_px: float = 0.0
    approach_prediction_error_px: float | None = None
    approach_allowed_error_px: float | None = None
    candidate_cell: int | None = None
    projected_anchor: str | None = None
    contact_recoveries: int = 0
    contact_recovery_frame: int | None = None
    contact_recovery_cell: int | None = None
    external_cell_component_hits: int = 0
    raw_z0_candidates: int = 0
    scale_low_rejections: int = 0
    scale_high_rejections: int = 0
    scale_quantized_accepts: int = 0
    scale_diag_reason: str | None = None
    scale_diag_cell: int | None = None
    scale_diag_observed_px: float | None = None
    scale_diag_expected_px: float | None = None
    scale_diag_ratio: float | None = None

    @property
    def bingo_cells(self) -> tuple[UpConfirmation, ...]:
        # BINGO exists ONLY after the same ball is observed ABOVE the prior Z0
        # candidate. A Z0 candidate alone is never a BINGO.
        return self.up_confirmations


@dataclass
class _PendingZ0:
    candidate: Z0Candidate


class ExternalGridFrameLoop:
    """GAME RUNTIME.

    Exact runtime order:
        FRAME -> PILLOW -> EXTERNAL CELL LOOP -> Z0 CANDIDATE
              -> UP AFTER THE FACT -> BINGO

    The only post-contact requirement is UP. There is deliberately no DOWN
    prerequisite, no flight path, no velocity model, no XYZ resolver and no
    ball-following tracker.
    """

    def __init__(
        self,
        calibration: ExternalGridCalibration,
        background_rgb: Image.Image,
        ball_profile: LockedBallColorProfile,
        *,
        difference_threshold: int = 18,
        min_color_pixels: int = 2,
        min_floor_scale_ratio: float = 0.45,
        max_floor_scale_ratio: float = 1.80,
        min_up_bu: float = 0.65,
        max_up_bu: float = 2.25,
        max_lateral_bu: float = 1.00,
        min_after_scale_ratio: float = 0.45,
        max_after_scale_ratio: float = 1.80,
        max_after_frames: int = 2,
        bingo_cooldown_frames: int = 6,
        motion_merge_gap_px: float = 8.0,
        min_outside_clearance_bu: float = 0.20,
    ) -> None:
        self.calibration = calibration
        self.background = background_rgb.convert("RGB")
        if self.background.size != tuple(calibration.image_size):
            raise ValueError("background size does not match calibration image_size")
        self.ball_profile = ball_profile
        self.difference_threshold = int(difference_threshold)
        self.min_color_pixels = int(min_color_pixels)
        self.min_floor_scale_ratio = float(min_floor_scale_ratio)
        self.max_floor_scale_ratio = float(max_floor_scale_ratio)
        self.min_up_bu = float(min_up_bu)
        self.max_up_bu = float(max_up_bu)
        self.max_lateral_bu = float(max_lateral_bu)
        self.min_after_scale_ratio = float(min_after_scale_ratio)
        self.max_after_scale_ratio = float(max_after_scale_ratio)
        self.max_after_frames = int(max_after_frames)
        self.bingo_cooldown_frames = int(bingo_cooldown_frames)
        self.motion_merge_gap_px = float(motion_merge_gap_px)
        self.min_outside_clearance_bu = float(min_outside_clearance_bu)
        if self.max_after_frames < 1:
            raise ValueError("max_after_frames must be >= 1")
        if self.motion_merge_gap_px < 0.0:
            raise ValueError("motion_merge_gap_px must be >= 0")
        if self.min_outside_clearance_bu < 0.0:
            raise ValueError("min_outside_clearance_bu must be >= 0")

        coverage = CalibrationCoverage.parse(calibration.coverage)
        court_x = float(calibration.config.court_x_bu)
        court_y = float(calibration.config.court_y_bu(coverage))
        image_quad = np.asarray(calibration.image_points, dtype=np.float32)
        court_quad = np.asarray(
            (
                (0.0, court_y),
                (court_x, court_y),
                (court_x, 0.0),
                (0.0, 0.0),
            ),
            dtype=np.float32,
        )
        self._image_to_top = cv2.getPerspectiveTransform(image_quad, court_quad)
        self._court_x_bu = court_x
        self._court_y_bu = court_y

        self._pending: list[_PendingZ0] = []
        self._last_bingo_frame_by_cell: dict[int, int] = {}
        self._reset_pre_z0_telemetry()

    def _components(self, current: Image.Image) -> tuple[np.ndarray, np.ndarray, list[BallComponent]]:
        # PILLOW happens only after FRAME arrives. The comparison is ALWAYS
        # against the calibration-owned background/reference, never against
        # the previous runtime frame. Each fixed grid cell therefore watches
        # its own calibrated reference region frame after frame.
        diff = ImageChops.difference(current, self.background)
        gray = np.asarray(diff.convert("L"), dtype=np.uint8)
        hsv = np.asarray(current.convert("HSV"), dtype=np.uint8)
        changed = gray >= self.difference_threshold
        hd = LockedBallColorProfile._hue_distance(hsv[..., 0], self.ball_profile.hue_center)
        color_ok = (
            (hd <= self.ball_profile.hue_tolerance)
            & (hsv[..., 1] >= self.ball_profile.saturation_min)
            & (hsv[..., 2] >= self.ball_profile.value_min)
        )
        ball_changed = changed & color_ok

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(ball_changed.astype(np.uint8), 8)
        components: list[BallComponent] = []
        for label in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[label])
            if area < self.min_color_pixels or area > 260:
                continue
            if w < 2 or h < 2:
                continue
            aspect = max(w / h, h / w)
            if aspect > 4.8:
                continue
            cx, cy = (float(v) for v in centroids[label])
            components.append(
                BallComponent(
                    bbox=(x, y, w, h),
                    area=area,
                    centroid_xy=(cx, cy),
                    scale_px=float(min(w, h)),
                )
            )
        return changed, ball_changed, components

    @staticmethod
    def _bbox_gap(a: BallComponent, b: BallComponent) -> float:
        ax, ay, aw, ah = a.bbox
        bx, by, bw, bh = b.bbox
        ax1, ay1 = ax + aw, ay + ah
        bx1, by1 = bx + bw, by + bh
        dx = max(0.0, float(max(ax, bx) - min(ax1, bx1)))
        dy = max(0.0, float(max(ay, by) - min(ay1, by1)))
        return math.hypot(dx, dy)

    def _merge_motion_components(
        self,
        components: list[BallComponent],
    ) -> list[BallComponent]:
        if len(components) < 2:
            return list(components)

        parent = list(range(len(components)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        for i, a in enumerate(components):
            for j in range(i + 1, len(components)):
                b = components[j]
                adaptive_gap = max(
                    self.motion_merge_gap_px,
                    0.65 * max(a.scale_px, b.scale_px),
                )
                if self._bbox_gap(a, b) > adaptive_gap:
                    continue

                ax, ay, aw, ah = a.bbox
                bx, by, bw, bh = b.bbox
                x0, y0 = min(ax, bx), min(ay, by)
                x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
                uw, uh = max(1, x1 - x0), max(1, y1 - y0)
                aspect = max(uw / uh, uh / uw)
                if aspect > 5.5:
                    continue
                if a.area + b.area > 520:
                    continue
                union(i, j)

        groups: dict[int, list[BallComponent]] = {}
        for i, component in enumerate(components):
            groups.setdefault(find(i), []).append(component)

        footprints: list[BallComponent] = []
        for group in groups.values():
            if len(group) == 1:
                footprints.append(group[0])
                continue

            x0 = min(c.bbox[0] for c in group)
            y0 = min(c.bbox[1] for c in group)
            x1 = max(c.bbox[0] + c.bbox[2] for c in group)
            y1 = max(c.bbox[1] + c.bbox[3] for c in group)
            total_area = sum(c.area for c in group)
            cx = sum(c.centroid_xy[0] * c.area for c in group) / total_area
            cy = sum(c.centroid_xy[1] * c.area for c in group) / total_area
            core_scale = max(c.scale_px for c in group)
            footprints.append(
                BallComponent(
                    bbox=(x0, y0, x1 - x0, y1 - y0),
                    area=total_area,
                    centroid_xy=(float(cx), float(cy)),
                    scale_px=float(core_scale),
                    merged_from=len(group),
                )
            )
        return footprints

    def _image_point_to_floor_bu(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        p = np.asarray([[[float(x), float(y)]]], dtype=np.float32)
        q = cv2.perspectiveTransform(p, self._image_to_top)[0, 0]
        return float(q[0]), float(q[1])

    def _outside_clearance_bu(self, x: float, y: float) -> float:
        dx = 0.0
        if x < 0.0:
            dx = -x
        elif x > self._court_x_bu:
            dx = x - self._court_x_bu

        dy = 0.0
        if y < 0.0:
            dy = -y
        elif y > self._court_y_bu:
            dy = y - self._court_y_bu

        return math.hypot(dx, dy)

    def _reset_pre_z0_telemetry(self) -> None:
        self._last_external_cell_component_hits = 0
        self._last_raw_z0_candidates = 0
        self._last_scale_low_rejections = 0
        self._last_scale_high_rejections = 0
        self._last_scale_quantized_accepts = 0
        self._last_scale_diag_reason: str | None = None
        self._last_scale_diag_cell: int | None = None
        self._last_scale_diag_observed_px: float | None = None
        self._last_scale_diag_expected_px: float | None = None
        self._last_scale_diag_ratio: float | None = None

    def _record_scale_diagnostic(
        self,
        *,
        reason: str,
        cell_id: int,
        observed_px: float,
        expected_px: float,
        ratio: float,
    ) -> None:
        priority = {
            "QUANTIZED_ACCEPT": 3,
            "SCALE_HIGH": 2,
            "SCALE_LOW": 1,
        }
        current_priority = priority.get(self._last_scale_diag_reason or "", 0)
        incoming_priority = priority.get(str(reason), 0)
        if incoming_priority < current_priority:
            return
        if (
            incoming_priority == current_priority
            and self._last_scale_diag_ratio is not None
        ):
            current_distance = min(
                abs(
                    self._last_scale_diag_ratio
                    - self.min_floor_scale_ratio
                ),
                abs(
                    self._last_scale_diag_ratio
                    - self.max_floor_scale_ratio
                ),
            )
            incoming_distance = min(
                abs(float(ratio) - self.min_floor_scale_ratio),
                abs(float(ratio) - self.max_floor_scale_ratio),
            )
            if incoming_distance > current_distance:
                return
        self._last_scale_diag_reason = str(reason)
        self._last_scale_diag_cell = int(cell_id)
        self._last_scale_diag_observed_px = float(observed_px)
        self._last_scale_diag_expected_px = float(expected_px)
        self._last_scale_diag_ratio = float(ratio)

    def _find_z0_candidates(
        self,
        frame_no: int,
        components: list[BallComponent],
    ) -> tuple[list[Z0Candidate], int]:
        self._reset_pre_z0_telemetry()
        candidates: list[Z0Candidate] = []
        boundary_guard_rejections = 0
        # Required architecture: fixed EXTERNAL CELL LOOP. Cells watch
        # themselves; no component is followed from one cell to another.
        for cell in self.calibration.cells:
            bx0, by0, bx1, by1 = cell.bbox_image
            poly = np.asarray(cell.polygon_image, dtype=np.float32)
            for comp in components:
                cx, cy = comp.centroid_xy
                if cx < bx0 or cx >= bx1 or cy < by0 or cy >= by1:
                    continue
                if cv2.pointPolygonTest(poly, (cx, cy), False) < 0:
                    continue
                self._last_external_cell_component_hits += 1
                expected = max(
                    0.5,
                    float(cell.expected_floor_ball_diameter_px),
                )
                observed = float(comp.scale_px)
                ratio = observed / expected
                legacy_compatible = (
                    self.min_floor_scale_ratio
                    <= ratio
                    <= self.max_floor_scale_ratio
                )
                min_px, max_px = rasterized_scale_bounds_px(
                    expected,
                    self.min_floor_scale_ratio,
                    self.max_floor_scale_ratio,
                )
                if observed < min_px:
                    self._last_scale_low_rejections += 1
                    self._record_scale_diagnostic(
                        reason="SCALE_LOW",
                        cell_id=cell.cell_id,
                        observed_px=observed,
                        expected_px=expected,
                        ratio=ratio,
                    )
                    continue
                if observed > max_px:
                    self._last_scale_high_rejections += 1
                    self._record_scale_diagnostic(
                        reason="SCALE_HIGH",
                        cell_id=cell.cell_id,
                        observed_px=observed,
                        expected_px=expected,
                        ratio=ratio,
                    )
                    continue
                if not legacy_compatible:
                    self._last_scale_quantized_accepts += 1
                    self._record_scale_diagnostic(
                        reason="QUANTIZED_ACCEPT",
                        cell_id=cell.cell_id,
                        observed_px=observed,
                        expected_px=expected,
                        ratio=ratio,
                    )

                floor_xy = self._image_point_to_floor_bu(cx, cy)
                clearance = self._outside_clearance_bu(*floor_xy)
                if clearance < self.min_outside_clearance_bu:
                    boundary_guard_rejections += 1
                    continue

                candidates.append(
                    Z0Candidate(
                        frame_no=int(frame_no),
                        cell_id=cell.cell_id,
                        region=cell.region,
                        color_pixels=comp.area,
                        centroid_xy=comp.centroid_xy,
                        observed_scale_px=comp.scale_px,
                        expected_floor_scale_px=expected,
                        scale_ratio=ratio,
                        floor_xy_bu=floor_xy,
                        boundary_clearance_bu=float(clearance),
                    )
                )
                break
        candidates.sort(
            key=lambda c: (
                abs(c.scale_ratio - 1.0),
                -c.color_pixels,
                c.cell_id,
            )
        )
        self._last_raw_z0_candidates = len(candidates)
        return candidates, boundary_guard_rejections

    def _confirm_up(self, frame_no: int, components: list[BallComponent]) -> list[UpConfirmation]:
        ux, uy = self.calibration.image_up_unit
        # Image-right vector perpendicular to calibrated image-up.
        rx, ry = -uy, ux
        confirmations: list[UpConfirmation] = []
        survivors: list[_PendingZ0] = []

        for pending in self._pending:
            c = pending.candidate
            age = int(frame_no) - int(c.frame_no)
            if age <= 0:
                survivors.append(pending)
                continue
            if age > self.max_after_frames:
                continue

            last_bingo = self._last_bingo_frame_by_cell.get(c.cell_id, -10**9)
            if frame_no - last_bingo <= self.bingo_cooldown_frames:
                continue

            expected = max(0.5, c.expected_floor_scale_px)
            best: tuple[float, UpConfirmation] | None = None
            for comp in components:
                dx = comp.centroid_xy[0] - c.centroid_xy[0]
                dy = comp.centroid_xy[1] - c.centroid_xy[1]
                up_bu = (dx * ux + dy * uy) / expected
                lateral_bu = abs(dx * rx + dy * ry) / expected
                if not (self.min_up_bu <= up_bu <= self.max_up_bu):
                    continue
                if lateral_bu > self.max_lateral_bu:
                    continue
                scale_after = comp.scale_px / max(0.5, c.observed_scale_px)
                if not (self.min_after_scale_ratio <= scale_after <= self.max_after_scale_ratio):
                    continue
                ev = UpConfirmation(
                    contact_frame=c.frame_no,
                    confirm_frame=int(frame_no),
                    cell_id=c.cell_id,
                    region=c.region,
                    contact_xy=c.centroid_xy,
                    above_xy=comp.centroid_xy,
                    up_bu=float(up_bu),
                    lateral_bu=float(lateral_bu),
                    scale_ratio_after=float(scale_after),
                )
                score = abs(up_bu - 1.0) + 0.35 * lateral_bu + 0.20 * abs(math.log(max(scale_after, 1e-6)))
                if best is None or score < best[0]:
                    best = (score, ev)

            if best is not None:
                confirmations.append(best[1])
                self._last_bingo_frame_by_cell[c.cell_id] = int(frame_no)
            else:
                survivors.append(pending)

        self._pending = survivors
        return confirmations

    def _add_pending(self, candidates: list[Z0Candidate]) -> None:
        # Keep only one recent pending candidate per cell. This is event memory,
        # not a ball tracker and it never predicts where the ball will go.
        by_cell = {p.candidate.cell_id: p for p in self._pending}
        for c in candidates:
            by_cell[c.cell_id] = _PendingZ0(c)
        self._pending = list(by_cell.values())

    def process_frame(self, frame_no: int, current_rgb: Image.Image | np.ndarray) -> ExternalFrameResult:
        trace = ["FRAME"]
        if not isinstance(current_rgb, Image.Image):
            current_rgb = Image.fromarray(np.asarray(current_rgb, dtype=np.uint8), mode="RGB")
        current = current_rgb.convert("RGB")
        if current.size != self.background.size:
            raise ValueError("frame size does not match calibration")

        changed, ball_changed, raw_components = self._components(current)
        trace.append("PILLOW")

        # Preserve the historical public stage contract:
        # FRAME -> PILLOW -> EXTERNAL_CELL_LOOP.
        # CP-0036.2.3 adds motion-footprint and line-touch sub-stages without
        # changing those first three canonical runtime stages.
        trace.append("EXTERNAL_CELL_LOOP")

        components = self._merge_motion_components(raw_components)
        trace.append("BALL_MOTION_FOOTPRINT")

        z0_candidates, boundary_guard_rejections = self._find_z0_candidates(
            frame_no,
            components,
        )
        trace.append("LINE_TOUCH_GUARD")
        trace.append("Z0_CANDIDATE")

        confirmations = self._confirm_up(frame_no, components)
        trace.append("UP_AFTER_FACT")

        # Current-frame candidates become eligible only on later frames, so a
        # single frame can never produce BINGO by itself.
        self._add_pending(z0_candidates)
        trace.append("BINGO")

        return ExternalFrameResult(
            frame_no=int(frame_no),
            stage_trace=tuple(trace),
            changed_pixels=int(changed.sum()),
            ball_color_changed_pixels=int(ball_changed.sum()),
            z0_candidates=tuple(z0_candidates),
            up_confirmations=tuple(confirmations),
            raw_ball_components=len(raw_components),
            ball_footprints=len(components),
            merged_motion_footprints=sum(
                1 for component in components if component.merged_from > 1
            ),
            boundary_guard_rejections=int(boundary_guard_rejections),
            external_cell_component_hits=int(
                self._last_external_cell_component_hits
            ),
            raw_z0_candidates=int(self._last_raw_z0_candidates),
            scale_low_rejections=int(self._last_scale_low_rejections),
            scale_high_rejections=int(self._last_scale_high_rejections),
            scale_quantized_accepts=int(
                self._last_scale_quantized_accepts
            ),
            scale_diag_reason=self._last_scale_diag_reason,
            scale_diag_cell=self._last_scale_diag_cell,
            scale_diag_observed_px=self._last_scale_diag_observed_px,
            scale_diag_expected_px=self._last_scale_diag_expected_px,
            scale_diag_ratio=self._last_scale_diag_ratio,
        )
