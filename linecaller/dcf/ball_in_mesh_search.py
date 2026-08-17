from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import cv2
import numpy as np

from linecaller.dcf.engine import DynamicCourtField
from linecaller.dcf.models import CellIndex, FieldRegion


@dataclass(frozen=True)
class BallInMeshSearchConfig:
    coarse_stride_x: int = 4
    coarse_stride_y: int = 6
    coarse_stride_z: int = 4
    top_color_candidates: int = 96
    refine_radius_x: int = 4
    refine_radius_y: int = 6
    refine_radius_z: int = 4
    top_refined_candidates: int = 600
    component_similarity_threshold: float = 0.70
    min_expected_diameter_px: float = 2.5
    max_expected_diameter_px: float = 40.0
    min_score: float = 0.60
    appearance_distance_scale: float = 42.0


@dataclass(frozen=True)
class ProjectedMeshCell:
    index: CellIndex
    image_x: float
    image_y: float
    expected_diameter_px: float
    region: FieldRegion


@dataclass(frozen=True)
class MeshScannerCell:
    ix: int
    iy: int
    iz: int
    x1: int
    y1: int
    x2: int
    y2: int
    inside: bool
    expected_scale: float
    dcf_index: CellIndex
    projected_x: float
    projected_y: float
    expected_diameter_px: float
    region: FieldRegion

    @property
    def center_x(self) -> float:
        return float(self.projected_x)

    @property
    def center_y(self) -> float:
        return float(self.projected_y)


@dataclass(frozen=True)
class BallInMeshHit:
    found: bool
    score: float
    x: float | None
    y: float | None
    index: CellIndex | None
    region: FieldRegion | None
    template_score: float = 0.0
    color_score: float = 0.0
    center_score: float = 0.0
    offset_px: float | None = None
    ball_generation: int = 0
    coarse_candidates_tested: int = 0
    refined_candidates_tested: int = 0


@dataclass(frozen=True)
class BallPixelProfile:
    hue: float
    saturation: float
    value: float
    hue_tolerance: float
    saturation_tolerance: float
    value_tolerance: float

    @staticmethod
    def _hue_center(hues: np.ndarray) -> float:
        angles = hues.astype(np.float64) * (2.0 * math.pi / 180.0)
        s = float(np.sin(angles).mean())
        c = float(np.cos(angles).mean())
        angle = math.atan2(s, c)
        if angle < 0:
            angle += 2.0 * math.pi
        return angle * 180.0 / (2.0 * math.pi)

    @staticmethod
    def _hue_distance(hues: np.ndarray, center: float) -> np.ndarray:
        diff = np.abs(hues.astype(np.float64) - center)
        return np.minimum(diff, 180.0 - diff)

    @classmethod
    def from_masked_reference(
        cls,
        reference_bgr: np.ndarray,
        mask: np.ndarray,
    ) -> "BallPixelProfile":
        hsv = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2HSV)
        pixels = hsv[mask > 0]
        if len(pixels) < 3:
            pixels = hsv.reshape(-1, 3)

        hues = pixels[:, 0].astype(np.float64)
        sats = pixels[:, 1].astype(np.float64)
        vals = pixels[:, 2].astype(np.float64)

        hue = cls._hue_center(hues)
        saturation = float(np.median(sats))
        value = float(np.median(vals))

        hue_dev = cls._hue_distance(hues, hue)
        sat_dev = np.abs(sats - saturation)
        val_dev = np.abs(vals - value)

        return cls(
            hue=hue,
            saturation=saturation,
            value=value,
            hue_tolerance=max(8.0, float(np.percentile(hue_dev, 90)) + 4.0),
            saturation_tolerance=max(35.0, float(np.percentile(sat_dev, 90)) + 18.0),
            value_tolerance=max(40.0, float(np.percentile(val_dev, 90)) + 20.0),
        )

    def similarity_map(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        hue_diff = np.abs(h - self.hue)
        hue_diff = np.minimum(hue_diff, 180.0 - hue_diff)

        hue_score = np.clip(
            1.0 - hue_diff / max(12.0, self.hue_tolerance * 2.0),
            0.0,
            1.0,
        )
        sat_score = np.clip(
            1.0 - np.abs(s - self.saturation)
            / max(55.0, self.saturation_tolerance * 2.0),
            0.0,
            1.0,
        )
        val_score = np.clip(
            1.0 - np.abs(v - self.value)
            / max(65.0, self.value_tolerance * 2.0),
            0.0,
            1.0,
        )

        score = 0.60 * hue_score + 0.25 * sat_score + 0.15 * val_score

        # A hue match in a dark background is not a ball-color match.
        # Bounds are derived from THIS ball reference, not a named/hardcoded color.
        min_value = max(
            20.0,
            self.value - max(60.0, self.value_tolerance * 1.5),
        )
        gate = v >= min_value

        if self.saturation >= 55.0:
            min_saturation = max(
                10.0,
                self.saturation - max(50.0, self.saturation_tolerance * 1.1),
            )
            gate &= s >= min_saturation

        gate &= hue_diff <= max(12.0, self.hue_tolerance * 2.0)
        return (score * gate.astype(np.float32)).astype(np.float32)


@dataclass(frozen=True)
class LockedBallIdentity:
    reference_bgr: np.ndarray
    mask: np.ndarray
    diameter_px: float
    pixel_profile: BallPixelProfile

    @staticmethod
    def _foreground_mask(reference_bgr: np.ndarray) -> np.ndarray:
        h, w = reference_bgr.shape[:2]
        lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        border = np.concatenate(
            [lab[0, :, :], lab[-1, :, :], lab[:, 0, :], lab[:, -1, :]],
            axis=0,
        )
        background = np.median(border, axis=0)
        distance = np.linalg.norm(lab - background, axis=2)

        if float(distance.max()) <= 1e-6:
            yy, xx = np.ogrid[:h, :w]
            radius = max(1.0, min(h, w) * 0.35)
            return (
                ((xx - (w - 1) / 2.0) ** 2 + (yy - (h - 1) / 2.0) ** 2)
                <= radius * radius
            ).astype(np.uint8) * 255

        normalized = cv2.normalize(
            distance,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)
        _, binary = cv2.threshold(
            normalized,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        count, labels, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
        choices = []
        image_area = float(h * w)
        cx0 = (w - 1) / 2.0
        cy0 = (h - 1) / 2.0

        for i in range(1, count):
            x, y, cw, ch, area = stats[i]
            if area < 3 or area > image_area * 0.72:
                continue
            cx, cy = centers[i]
            center_distance = (
                ((cx - cx0) / max(1.0, w)) ** 2
                + ((cy - cy0) / max(1.0, h)) ** 2
            )
            choices.append((int(area), -float(center_distance), i))

        if not choices:
            return binary

        _, _, winner = max(choices)
        return (labels == winner).astype(np.uint8) * 255

    @classmethod
    def from_reference(cls, reference_bgr: np.ndarray) -> "LockedBallIdentity":
        if reference_bgr is None or reference_bgr.size == 0:
            raise ValueError("Ball reference is empty")

        raw_mask = cls._foreground_mask(reference_bgr)
        ys, xs = np.where(raw_mask > 0)
        if len(xs) < 3:
            raise ValueError("Could not isolate ball foreground from reference")

        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        bw = x2 - x1
        bh = y2 - y1
        diameter = math.sqrt(float(max(1, bw) * max(1, bh)))

        pad = 1
        side = max(5, max(bw, bh) + 2 * pad)
        canonical = np.zeros((side, side, 3), dtype=np.uint8)
        canonical_mask = np.zeros((side, side), dtype=np.uint8)

        object_bgr = reference_bgr[y1:y2, x1:x2]
        object_mask = raw_mask[y1:y2, x1:x2]
        ox = (side - bw) // 2
        oy = (side - bh) // 2

        target = canonical[oy:oy + bh, ox:ox + bw]
        target_mask = canonical_mask[oy:oy + bh, ox:ox + bw]
        keep = object_mask > 0
        target[keep] = object_bgr[keep]
        target_mask[keep] = 255

        profile = BallPixelProfile.from_masked_reference(canonical, canonical_mask)
        return cls(canonical, canonical_mask, float(diameter), profile)


@dataclass(frozen=True)
class _ColorComponent:
    x: float
    y: float
    diameter_px: float
    mean_similarity: float
    area: int


class DCFMeshProjector:
    """
    Runtime DCF projection profile for CP-0035.9.3 RC2.

    The old PerspectiveCourtMesh used camera-specific visual constants such as
    near_height_px and gamma.  This projector derives its scale from the
    CURRENT court quadrilateral and CURRENT DynamicCourtField resolution.

    Z is expressed in DCF cells. One Z cell uses the same projected scale as
    one local lateral DCF cell. The default image vertical is screen-up, but a
    runtime vertical vector can be supplied by calibration later.
    """

    def __init__(
        self,
        image_points,
        *,
        field: DynamicCourtField | None = None,
        vertical_unit: tuple[float, float] | None = None,
    ):
        pts = np.asarray(image_points, dtype=np.float64)
        if pts.shape != (4, 2):
            raise ValueError(
                "image_points must be 4x2 in order: "
                "near-left, near-right, far-right, far-left"
            )

        self.field = field or DynamicCourtField()
        self.mesh = self  # CP-0035.9.3 compatibility: internal projector surface only
        self.nl, self.nr, self.fr, self.fl = pts

        v = np.asarray(vertical_unit or (0.0, -1.0), dtype=np.float64)
        norm = float(np.linalg.norm(v))
        if norm <= 1e-9:
            raise ValueError("vertical_unit cannot be zero")
        self.vertical_unit = v / norm

    def normalized_coordinates(self, index: CellIndex) -> tuple[float, float, float]:
        if not self.field.valid(index):
            raise ValueError(f"Invalid DCF cell: {index}")

        c = self.field.config
        u = (float(index.x - self.field.x0) + 0.5) / float(c.court_x_cells)
        t = (float(index.y - self.field.y0) + 0.5) / float(c.court_y_cells)
        return u, t, float(index.z)

    @staticmethod
    def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        return a * (1.0 - t) + b * t

    def floor_point(self, u: float, t: float) -> np.ndarray:
        left = self._lerp(self.nl, self.fl, t)
        right = self._lerp(self.nr, self.fr, t)
        return self._lerp(left, right, u)

    def expected_diameter_px(self, index: CellIndex) -> float:
        u, t, _ = self.normalized_coordinates(index)
        du = 1.0 / float(self.field.config.court_x_cells)
        p = self.floor_point(u, t)
        px = self.floor_point(u + du, t)
        diameter = float(np.linalg.norm(px - p))
        if not math.isfinite(diameter) or diameter <= 0.0:
            raise ValueError("Invalid runtime DCF scale")
        return diameter

    def image_point(self, index: CellIndex) -> tuple[float, float]:
        u, t, z = self.normalized_coordinates(index)
        floor = self.floor_point(u, t)
        diameter = self.expected_diameter_px(index)
        p = floor + self.vertical_unit * (z * diameter)
        return float(p[0]), float(p[1])

    def project(self, index: CellIndex) -> ProjectedMeshCell:
        x, y = self.image_point(index)
        return ProjectedMeshCell(
            index=index,
            image_x=x,
            image_y=y,
            expected_diameter_px=self.expected_diameter_px(index),
            region=self.field.region_at_xy(index.x, index.y),
        )

    @staticmethod
    def reference_diameter_px(ball_reference: np.ndarray) -> float:
        return LockedBallIdentity.from_reference(ball_reference).diameter_px

    def to_scanner_cell(
        self,
        projected: ProjectedMeshCell,
        ball_reference: np.ndarray,
    ) -> MeshScannerCell:
        reference_diameter = self.reference_diameter_px(ball_reference)
        scale = projected.expected_diameter_px / reference_diameter
        support = max(8.0, projected.expected_diameter_px * 1.6)
        half = support / 2.0
        return MeshScannerCell(
            ix=projected.index.x,
            iy=projected.index.y,
            iz=projected.index.z,
            x1=int(math.floor(projected.image_x - half)),
            y1=int(math.floor(projected.image_y - half)),
            x2=int(math.ceil(projected.image_x + half)),
            y2=int(math.ceil(projected.image_y + half)),
            inside=projected.region in (FieldRegion.INSIDE, FieldRegion.BOUNDARY),
            expected_scale=float(scale),
            dcf_index=projected.index,
            projected_x=float(projected.image_x),
            projected_y=float(projected.image_y),
            expected_diameter_px=float(projected.expected_diameter_px),
            region=projected.region,
        )


class BallInMeshSearcher:
    """
    ONE ACTIVE BALL / ONE DCF / ONE WINNING CELL.

    No YOLO. No court-specific color. No ball-specific color name. No visible
    production mesh.
    """

    def __init__(
        self,
        image_points,
        *,
        field: DynamicCourtField | None = None,
        config: BallInMeshSearchConfig | None = None,
        vertical_unit: tuple[float, float] | None = None,
        min_score: float | None = None,
        min_color_score: float | None = None,  # backward-compatible argument
    ):
        self.field = field or DynamicCourtField()
        cfg = config or BallInMeshSearchConfig()
        if min_score is not None:
            cfg = BallInMeshSearchConfig(**{
                **cfg.__dict__,
                "min_score": float(min_score),
            })
        self.config = cfg
        self.projector = DCFMeshProjector(
            image_points,
            field=self.field,
            vertical_unit=vertical_unit,
        )
        self._identity: LockedBallIdentity | None = None
        self._ball_generation = 0
        self._resized_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    @property
    def ball_locked(self) -> bool:
        return self._identity is not None

    @property
    def ball_generation(self) -> int:
        return self._ball_generation

    @property
    def ball_identity(self) -> LockedBallIdentity | None:
        return self._identity

    def lock_ball(self, ball_reference: np.ndarray) -> LockedBallIdentity:
        self._identity = LockedBallIdentity.from_reference(ball_reference)
        if self._ball_generation == 0:
            self._ball_generation = 1
        self._resized_cache.clear()
        return self._identity

    def replace_ball(self, ball_reference: np.ndarray) -> LockedBallIdentity:
        self._identity = LockedBallIdentity.from_reference(ball_reference)
        self._ball_generation = max(1, self._ball_generation + 1)
        self._resized_cache.clear()
        return self._identity

    def _ensure_locked(self, ball_reference: np.ndarray) -> LockedBallIdentity:
        if self._identity is None:
            return self.lock_ball(ball_reference)
        return self._identity

    def _coarse_indices(self) -> Iterable[CellIndex]:
        c = self.config
        z_values = sorted(set([0] + list(range(0, self.field.total_z, max(1, c.coarse_stride_z)))))
        for z in z_values:
            for y in range(0, self.field.total_y, max(1, c.coarse_stride_y)):
                for x in range(0, self.field.total_x, max(1, c.coarse_stride_x)):
                    yield CellIndex(x, y, z)

    @staticmethod
    def _in_frame(projected: ProjectedMeshCell, shape) -> bool:
        h, w = shape[:2]
        pad = projected.expected_diameter_px
        return (
            -pad <= projected.image_x < w + pad
            and -pad <= projected.image_y < h + pad
        )

    def _valid_projection(self, projected: ProjectedMeshCell, shape) -> bool:
        d = projected.expected_diameter_px
        return (
            self._in_frame(projected, shape)
            and self.config.min_expected_diameter_px <= d <= self.config.max_expected_diameter_px
        )

    def _components(self, similarity: np.ndarray) -> list[_ColorComponent]:
        binary = (similarity >= self.config.component_similarity_threshold).astype(np.uint8) * 255
        count, labels, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
        result: list[_ColorComponent] = []

        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if area < 2 or area > 250 or w > 32 or h > 32:
                continue
            aspect = float(w) / max(1.0, float(h))
            if not 0.30 <= aspect <= 3.30:
                continue
            values = similarity[labels == i]
            result.append(_ColorComponent(
                x=float(centers[i][0]),
                y=float(centers[i][1]),
                diameter_px=math.sqrt(float(max(1, w) * max(1, h))),
                mean_similarity=float(values.mean()) if len(values) else 0.0,
                area=int(area),
            ))
        return result

    @staticmethod
    def _component_score(
        projected: ProjectedMeshCell,
        components: list[_ColorComponent],
        radius: float,
    ) -> float | None:
        best = None
        d = projected.expected_diameter_px
        for component in components:
            distance = math.hypot(
                component.x - projected.image_x,
                component.y - projected.image_y,
            )
            if distance > radius:
                continue
            size_score = math.exp(-abs(math.log(max(1e-6, component.diameter_px / d))))
            distance_score = max(0.0, 1.0 - distance / max(1.0, radius))
            score = (
                0.40 * component.mean_similarity
                + 0.35 * size_score
                + 0.25 * distance_score
            )
            if best is None or score > best:
                best = float(score)
        return best

    def _coarse_seeds(
        self,
        frame_shape,
        components: list[_ColorComponent],
    ) -> tuple[list[tuple[float, CellIndex]], int]:
        ranked = []
        tested = 0
        stride_span = max(self.config.coarse_stride_x, self.config.coarse_stride_z)

        for index in self._coarse_indices():
            projected = self.projector.project(index)
            if not self._valid_projection(projected, frame_shape):
                continue
            tested += 1
            radius = max(4.0, projected.expected_diameter_px * stride_span * 0.80)
            score = self._component_score(projected, components, radius)
            if score is not None:
                ranked.append((score, index))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[:max(1, self.config.top_color_candidates)], tested

    def _refine_indices(self, seeds: Iterable[tuple[float, CellIndex]]) -> set[CellIndex]:
        c = self.config
        result: set[CellIndex] = set()
        for _, seed in seeds:
            for dz in range(-c.refine_radius_z, c.refine_radius_z + 1):
                for dy in range(-c.refine_radius_y, c.refine_radius_y + 1):
                    for dx in range(-c.refine_radius_x, c.refine_radius_x + 1):
                        q = CellIndex(seed.x + dx, seed.y + dy, seed.z + dz)
                        if self.field.valid(q):
                            result.add(q)
        return result

    def _rank_refined(
        self,
        indices: Iterable[CellIndex],
        frame_shape,
        components: list[_ColorComponent],
    ) -> list[tuple[float, ProjectedMeshCell]]:
        ranked = []
        for index in indices:
            projected = self.projector.project(index)
            if not self._valid_projection(projected, frame_shape):
                continue
            radius = max(3.0, projected.expected_diameter_px * 1.15)
            score = self._component_score(projected, components, radius)
            if score is not None:
                ranked.append((score, projected))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[:max(1, self.config.top_refined_candidates)]

    def _resized_identity(self, expected_diameter_px: float) -> tuple[np.ndarray, np.ndarray]:
        assert self._identity is not None
        scale = expected_diameter_px / max(1e-6, self._identity.diameter_px)
        h, w = self._identity.reference_bgr.shape[:2]
        tw = max(3, round(w * scale))
        th = max(3, round(h * scale))
        key = (tw, th)
        cached = self._resized_cache.get(key)
        if cached is not None:
            return cached

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        tpl = cv2.resize(self._identity.reference_bgr, (tw, th), interpolation=interp)
        mask = cv2.resize(self._identity.mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        if cv2.countNonZero(mask) < 2:
            mask[:, :] = 255
        tpl_lab = cv2.cvtColor(tpl, cv2.COLOR_BGR2LAB).astype(np.float32)
        self._resized_cache[key] = (tpl_lab, mask)
        return tpl_lab, mask

    def _detail_score(
        self,
        projected: ProjectedMeshCell,
        frame_lab: np.ndarray,
        similarity: np.ndarray,
    ):
        tpl_lab, mask = self._resized_identity(projected.expected_diameter_px)
        th, tw = mask.shape[:2]
        h, w = similarity.shape[:2]
        tolerance = max(2, min(5, round(min(tw, th) * 0.35)))
        mask_bool = mask > 0
        best = None

        for dy in range(-tolerance, tolerance + 1):
            for dx in range(-tolerance, tolerance + 1):
                cx = projected.image_x + dx
                cy = projected.image_y + dy
                x1 = int(round(cx - tw / 2.0))
                y1 = int(round(cy - th / 2.0))
                x2 = x1 + tw
                y2 = y1 + th
                if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                    continue

                patch_lab = frame_lab[y1:y2, x1:x2]
                distances = np.linalg.norm(
                    patch_lab[mask_bool] - tpl_lab[mask_bool],
                    axis=1,
                )
                if len(distances) == 0:
                    continue

                mean_distance = float(distances.mean())
                appearance = math.exp(
                    -mean_distance / max(1.0, self.config.appearance_distance_scale)
                )

                local_similarity = similarity[y1:y2, x1:x2]
                inside = local_similarity[mask_bool]
                color_mean = float(inside.mean())
                coverage = float((inside >= 0.55).mean())
                color_score = 0.55 * color_mean + 0.45 * coverage

                outside = local_similarity[~mask_bool]
                if len(outside):
                    contrast = float(np.clip(
                        0.5 + color_mean - float(outside.mean()),
                        0.0,
                        1.0,
                    ))
                else:
                    contrast = 0.5

                offset = math.hypot(dx, dy)
                center_score = max(
                    0.0,
                    1.0 - offset / max(1.0, tolerance + 0.75),
                )

                final = (
                    0.45 * appearance
                    + 0.30 * color_score
                    + 0.15 * center_score
                    + 0.10 * contrast
                )
                candidate = (
                    float(final),
                    float(cx),
                    float(cy),
                    float(appearance),
                    float(color_score),
                    float(center_score),
                    float(offset),
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate
        return best

    def search_indices(
        self,
        frame: np.ndarray,
        ball_reference: np.ndarray,
        indices: Iterable[CellIndex],
        *,
        coarse_candidates_tested: int = 0,
        min_score: float | None = None,
    ) -> BallInMeshHit:
        identity = self._ensure_locked(ball_reference)
        similarity = identity.pixel_profile.similarity_map(frame)
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

        best = None
        tested = 0
        for index in indices:
            projected = self.projector.project(index)
            if not self._valid_projection(projected, frame.shape):
                continue
            tested += 1
            detail = self._detail_score(projected, frame_lab, similarity)
            if detail is None:
                continue
            if best is None or detail[0] > best[0][0]:
                best = (detail, projected)

        threshold = self.config.min_score if min_score is None else float(min_score)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("min_score must be in [0,1]")

        if best is None or best[0][0] < threshold:
            return BallInMeshHit(
                found=False,
                score=0.0 if best is None else float(best[0][0]),
                x=None,
                y=None,
                index=None,
                region=None,
                ball_generation=self.ball_generation,
                coarse_candidates_tested=coarse_candidates_tested,
                refined_candidates_tested=tested,
            )

        detail, projected = best
        return BallInMeshHit(
            found=True,
            score=detail[0],
            x=detail[1],
            y=detail[2],
            index=projected.index,
            region=projected.region,
            template_score=detail[3],
            color_score=detail[4],
            center_score=detail[5],
            offset_px=detail[6],
            ball_generation=self.ball_generation,
            coarse_candidates_tested=coarse_candidates_tested,
            refined_candidates_tested=tested,
        )

    def search(self, frame: np.ndarray, ball_reference: np.ndarray) -> BallInMeshHit:
        identity = self._ensure_locked(ball_reference)
        similarity = identity.pixel_profile.similarity_map(frame)
        components = self._components(similarity)
        if not components:
            return BallInMeshHit(
                False, 0.0, None, None, None, None,
                ball_generation=self.ball_generation,
            )

        seeds, coarse_tested = self._coarse_seeds(frame.shape, components)
        if not seeds:
            return BallInMeshHit(
                False, 0.0, None, None, None, None,
                ball_generation=self.ball_generation,
                coarse_candidates_tested=coarse_tested,
            )

        refined = self._refine_indices(seeds)
        ranked = self._rank_refined(refined, frame.shape, components)
        if not ranked:
            return BallInMeshHit(
                False, 0.0, None, None, None, None,
                ball_generation=self.ball_generation,
                coarse_candidates_tested=coarse_tested,
            )

        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        best = None
        tested = 0
        for _, projected in ranked:
            tested += 1
            detail = self._detail_score(projected, frame_lab, similarity)
            if detail is None:
                continue
            if best is None or detail[0] > best[0][0]:
                best = (detail, projected)

        if best is None or best[0][0] < self.config.min_score:
            return BallInMeshHit(
                found=False,
                score=0.0 if best is None else float(best[0][0]),
                x=None,
                y=None,
                index=None,
                region=None,
                ball_generation=self.ball_generation,
                coarse_candidates_tested=coarse_tested,
                refined_candidates_tested=tested,
            )

        detail, projected = best
        return BallInMeshHit(
            found=True,
            score=detail[0],
            x=detail[1],
            y=detail[2],
            index=projected.index,
            region=projected.region,
            template_score=detail[3],
            color_score=detail[4],
            center_score=detail[5],
            offset_px=detail[6],
            ball_generation=self.ball_generation,
            coarse_candidates_tested=coarse_tested,
            refined_candidates_tested=tested,
        )
