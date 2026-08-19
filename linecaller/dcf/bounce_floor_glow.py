from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import cv2
import numpy as np

from linecaller.dcf.ball_in_mesh_search import DCFMeshProjector
from linecaller.dcf.models import CellIndex


@dataclass(frozen=True)
class BounceGlowConfig:
    """Presentation-only timing/tuning for one confirmed DCF floor contact."""

    fade_in_ms: float = 180.0
    hold_ms: float = 1400.0
    fade_out_ms: float = 900.0

    # The DCF evidence cell is ball-sized. The visible mark is deliberately a
    # soft footprint rather than a square cell so production never exposes the
    # technical mesh. Radii remain expressed in DCF-cell (ball-diameter) units.
    core_radius_cells: float = 0.90
    halo_radius_cells: float = 2.80
    core_alpha: float = 0.40
    halo_alpha: float = 0.44
    max_alpha: float = 0.66
    feather_sigma_cells: float = 0.90
    polygon_segments: int = 48

    def __post_init__(self):
        if self.fade_in_ms < 0 or self.hold_ms < 0 or self.fade_out_ms < 0:
            raise ValueError("Glow timing values must be >= 0")
        if self.fade_in_ms + self.hold_ms + self.fade_out_ms <= 0:
            raise ValueError("Glow duration must be > 0")
        if self.core_radius_cells <= 0.0:
            raise ValueError("core_radius_cells must be > 0")
        if self.halo_radius_cells < self.core_radius_cells:
            raise ValueError("halo_radius_cells must be >= core_radius_cells")
        if self.feather_sigma_cells <= 0.0:
            raise ValueError("feather_sigma_cells must be > 0")
        if self.polygon_segments < 16:
            raise ValueError("polygon_segments must be >= 16")
        for name in ("core_alpha", "halo_alpha", "max_alpha"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")

    @property
    def duration_ms(self) -> float:
        return self.fade_in_ms + self.hold_ms + self.fade_out_ms


@dataclass(frozen=True)
class ProjectedContactFloor:
    cell: CellIndex
    center_x: float
    center_y: float
    nominal_diameter_px: float
    core_polygon: np.ndarray
    halo_polygon: np.ndarray


@dataclass(frozen=True)
class BounceGlowEvent:
    cell: CellIndex
    floor: ProjectedContactFloor
    color_bgr: tuple[int, int, int]
    started_at_s: float
    ball_generation: int = 0


class BounceFloorProjector:
    """
    Project one confirmed Z=0 DCF contact into the camera image.

    It never draws the DCF. It projects only a soft, ball-sized floor footprint
    around the contact cell center. X/Y/Z remain the existing DynamicCourtField
    coordinates; no second mesh or alternate coordinate system is introduced.
    """

    def __init__(self, projector: DCFMeshProjector, *, config: BounceGlowConfig | None = None):
        self.projector = projector
        self.field = projector.field
        self.config = config or BounceGlowConfig()

    def _disc_polygon(self, cell: CellIndex, radius_cells: float) -> np.ndarray:
        u, t, _ = self.projector.normalized_coordinates(cell)
        c = self.field.config
        du = float(radius_cells) / float(c.court_x_cells)
        dt = float(radius_cells) / float(c.court_y_cells)

        points = []
        for i in range(self.config.polygon_segments):
            a = (2.0 * math.pi * i) / float(self.config.polygon_segments)
            points.append(
                self.projector.floor_point(
                    u + du * math.cos(a),
                    t + dt * math.sin(a),
                )
            )
        return np.asarray(points, dtype=np.float32)

    def project(self, cell: CellIndex) -> ProjectedContactFloor:
        if cell.z != 0:
            raise ValueError("Bounce floor glow requires the Contact Plane Z=0")
        if not self.field.valid(cell):
            raise ValueError(f"Invalid DCF contact cell: {cell}")

        u, t, _ = self.projector.normalized_coordinates(cell)
        center = self.projector.floor_point(u, t)
        diameter = float(self.projector.expected_diameter_px(cell))

        return ProjectedContactFloor(
            cell=cell,
            center_x=float(center[0]),
            center_y=float(center[1]),
            nominal_diameter_px=diameter,
            core_polygon=self._disc_polygon(cell, self.config.core_radius_cells),
            halo_polygon=self._disc_polygon(cell, self.config.halo_radius_cells),
        )


class BounceFloorGlowRenderer:
    """
    Production renderer: original video + soft contact glow only.

    No mesh lines, cell coordinates, Ball Wave candidates, bounding boxes, or
    decision labels are rendered. Color is a runtime input; IN/OUT/REVIEW color
    policy belongs to the later decision-visual capability.
    """

    def __init__(self, config: BounceGlowConfig | None = None):
        self.config = config or BounceGlowConfig()

    @staticmethod
    def _smoothstep(t: float) -> float:
        t = max(0.0, min(1.0, float(t)))
        return t * t * (3.0 - 2.0 * t)

    def envelope(self, event: BounceGlowEvent, now_s: float) -> float:
        age_ms = (float(now_s) - float(event.started_at_s)) * 1000.0
        if age_ms < 0.0:
            return 0.0

        c = self.config
        if c.fade_in_ms > 0.0 and age_ms < c.fade_in_ms:
            return self._smoothstep(age_ms / c.fade_in_ms)

        after_in = age_ms - c.fade_in_ms
        if after_in < c.hold_ms:
            return 1.0

        after_hold = after_in - c.hold_ms
        if c.fade_out_ms <= 0.0 or after_hold >= c.fade_out_ms:
            return 0.0

        return 1.0 - self._smoothstep(after_hold / c.fade_out_ms)

    def expired(self, event: BounceGlowEvent, now_s: float) -> bool:
        age_ms = (float(now_s) - float(event.started_at_s)) * 1000.0
        return age_ms >= self.config.duration_ms

    def _sigma_px(self, floor: ProjectedContactFloor) -> float:
        return max(1.2, floor.nominal_diameter_px * self.config.feather_sigma_cells)

    @staticmethod
    def _filled_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        q = np.rint(polygon).astype(np.int32)
        cv2.fillConvexPoly(mask, q, 255, lineType=cv2.LINE_AA)
        return mask.astype(np.float32) / 255.0

    def render(self, frame: np.ndarray, event: BounceGlowEvent, now_s: float) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("frame is empty")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image")

        env = self.envelope(event, now_s)
        if env <= 0.0:
            return frame.copy()

        h, w = frame.shape[:2]
        core = self._filled_mask((h, w), event.floor.core_polygon)
        halo_seed = self._filled_mask((h, w), event.floor.halo_polygon)
        sigma = self._sigma_px(event.floor)
        halo = cv2.GaussianBlur(halo_seed, (0, 0), sigmaX=sigma, sigmaY=sigma)

        alpha = self.config.core_alpha * core + self.config.halo_alpha * halo
        alpha = np.clip(alpha, 0.0, self.config.max_alpha) * float(env)

        src = frame.astype(np.float32)
        color = np.asarray(event.color_bgr, dtype=np.float32).reshape(1, 1, 3)
        a3 = alpha[:, :, None]
        out = src * (1.0 - a3) + color * a3
        return np.clip(out, 0, 255).astype(np.uint8)


class BounceFloorGlowController:
    """
    Consume CP-0035.9.4.1's authoritative floor_trigger and visualize it.

    Physics stays in BallWaveTracker. This controller does NOT decide whether a
    bounce happened. It lights only step.predicted_contact_cell when
    step.floor_trigger is True. That supports both:
      * accepted descending Z=0, and
      * Z=1 + DOWN -> predicted Z=0.
    """

    def __init__(
        self,
        projector: DCFMeshProjector,
        *,
        config: BounceGlowConfig | None = None,
    ):
        self.config = config or BounceGlowConfig()
        self.floor_projector = BounceFloorProjector(projector, config=self.config)
        self.renderer = BounceFloorGlowRenderer(self.config)
        self._events: list[BounceGlowEvent] = []
        self._ball_generation: int | None = None

    @property
    def active_events(self) -> tuple[BounceGlowEvent, ...]:
        return tuple(self._events)

    def reset(self, *, ball_generation: int | None = None) -> None:
        self._events.clear()
        self._ball_generation = ball_generation

    @staticmethod
    def _normalize_color(color_bgr: Iterable[int]) -> tuple[int, int, int]:
        values = tuple(int(v) for v in color_bgr)
        if len(values) != 3:
            raise ValueError("color_bgr must contain exactly three values")
        if any(v < 0 or v > 255 for v in values):
            raise ValueError("color_bgr values must be in [0,255]")
        return values

    def _sync_generation(self, generation: int) -> None:
        generation = int(generation)
        if self._ball_generation is None:
            self._ball_generation = generation
        elif generation != self._ball_generation:
            self.reset(ball_generation=generation)

    def trigger_contact(
        self,
        cell: CellIndex,
        *,
        now_s: float,
        color_bgr: Iterable[int],
        ball_generation: int = 0,
    ) -> BounceGlowEvent:
        self._sync_generation(ball_generation)
        if cell.z != 0:
            raise ValueError("Visible contact must be on Z=0")

        event = BounceGlowEvent(
            cell=cell,
            floor=self.floor_projector.project(cell),
            color_bgr=self._normalize_color(color_bgr),
            started_at_s=float(now_s),
            ball_generation=int(ball_generation),
        )
        self._events.append(event)
        return event

    def consume_wave_step(
        self,
        step,
        *,
        now_s: float,
        color_bgr: Iterable[int],
    ) -> BounceGlowEvent | None:
        state = getattr(step, "state", None)
        hit = getattr(step, "hit", None)
        generation = int(
            getattr(state, "ball_generation", 0)
            or getattr(hit, "ball_generation", 0)
            or 0
        )
        self._sync_generation(generation)

        # Important separation of responsibilities: contact_plane_reached on
        # its own is not enough. Only CP-0035.9.4.1's floor_trigger may light.
        if not bool(getattr(step, "floor_trigger", False)):
            return None

        contact = getattr(step, "predicted_contact_cell", None)
        if contact is None:
            raise ValueError("floor_trigger requires predicted_contact_cell")
        if contact.z != 0:
            raise ValueError("floor_trigger contact cell must be Z=0")

        return self.trigger_contact(
            contact,
            now_s=now_s,
            color_bgr=color_bgr,
            ball_generation=generation,
        )

    def render(self, frame: np.ndarray, *, now_s: float) -> np.ndarray:
        live: list[BounceGlowEvent] = []
        out = frame.copy()
        for event in self._events:
            if self.renderer.expired(event, now_s):
                continue
            out = self.renderer.render(out, event, now_s)
            live.append(event)
        self._events = live
        return out
