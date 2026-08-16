from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from linecaller.dcf.ball_in_mesh_search import BallInMeshHit, BallInMeshSearcher
from linecaller.dcf.engine import DynamicCourtField
from linecaller.dcf.models import CellIndex


@dataclass(frozen=True)
class BallWaveConfig:
    """Runtime tuning for the local DCF Ball Wave."""

    forward_bias: int = 2
    velocity_smoothing: float = 0.55
    miss_uncertainty_step: float = 0.22
    hit_uncertainty_recovery: float = 0.35
    max_local_misses: int = 4

    def __post_init__(self):
        if self.forward_bias < 0:
            raise ValueError("forward_bias must be >= 0")
        if not 0.0 <= self.velocity_smoothing <= 1.0:
            raise ValueError("velocity_smoothing must be in [0,1]")
        if self.miss_uncertainty_step <= 0.0:
            raise ValueError("miss_uncertainty_step must be > 0")
        if self.hit_uncertainty_recovery < 0.0:
            raise ValueError("hit_uncertainty_recovery must be >= 0")
        if self.max_local_misses < 1:
            raise ValueError("max_local_misses must be >= 1")


@dataclass(frozen=True)
class BallWaveState:
    active: bool
    ball_generation: int
    last_cell: CellIndex | None
    velocity: tuple[float, float, float]
    uncertainty: float
    missed_frames: int
    accepted_steps: int
    wave_cell_count: int
    requires_reacquisition: bool


@dataclass(frozen=True)
class BallWaveStep:
    hit: BallInMeshHit
    state: BallWaveState
    local_candidates_tested: int
    contact_plane_reached: bool
    floor_illumination: bool
    reason: str


class BallWaveTracker:
    """
    CP-0035.9.4 -- DCF Ball Wave / cell-to-cell tracking.

    The global Ball-in-Mesh search is used for acquisition only. Once a ball
    has a winning DCF cell, every live frame is searched only inside the
    DynamicCourtField prediction wave around that cell.

    The DCF/mesh/wave are internal and must not be rendered in production.
    Z=0 may be reported as a contact-plane candidate, but floor illumination
    is intentionally reserved for CP-0035.9.5.
    """

    def __init__(
        self,
        searcher: BallInMeshSearcher,
        *,
        field: DynamicCourtField | None = None,
        config: BallWaveConfig | None = None,
    ):
        self.searcher = searcher
        self.field = field or searcher.field
        if self.field is not searcher.field:
            raise ValueError("BallWaveTracker and BallInMeshSearcher must share one DCF")
        self.config = config or BallWaveConfig()
        self._state = BallWaveState(
            active=False,
            ball_generation=searcher.ball_generation,
            last_cell=None,
            velocity=(0.0, 0.0, 0.0),
            uncertainty=0.0,
            missed_frames=0,
            accepted_steps=0,
            wave_cell_count=0,
            requires_reacquisition=True,
        )

    @property
    def state(self) -> BallWaveState:
        return self._state

    @staticmethod
    def _empty_hit(generation: int) -> BallInMeshHit:
        return BallInMeshHit(
            found=False,
            score=0.0,
            x=None,
            y=None,
            index=None,
            region=None,
            ball_generation=generation,
        )

    def _fallback_wave(self, p: CellIndex) -> frozenset[CellIndex]:
        """Compatibility fallback for very old DCF engines."""
        c = self.field.config
        u = max(0.0, min(1.0, float(self._state.uncertainty)))
        r = round(c.active_radius_xy + u * (c.max_active_radius - c.active_radius_xy))
        rz = round(c.active_radius_z + u * (c.max_active_radius - c.active_radius_z))
        vx, vy, vz = self._state.velocity
        cx = round(p.x + vx)
        cy = round(p.y + vy)
        cz = round(p.z + vz)
        sx = (vx > 0) - (vx < 0)
        sy = (vy > 0) - (vy < 0)
        sz = (vz > 0) - (vz < 0)
        out: set[CellIndex] = set()

        def add_box(ax: int, ay: int, az: int, rr: int, rrz: int):
            for dx in range(-rr, rr + 1):
                for dy in range(-rr, rr + 1):
                    for dz in range(-rrz, rrz + 1):
                        q = CellIndex(ax + dx, ay + dy, az + dz)
                        if self.field.valid(q):
                            out.add(q)

        add_box(cx, cy, cz, r, rz)
        for step in range(1, max(0, int(self.config.forward_bias)) + 1):
            add_box(
                cx + sx * step,
                cy + sy * step,
                cz + sz * step,
                max(1, r - step),
                max(1, rz - step),
            )
        return frozenset(out)

    def current_wave(self) -> frozenset[CellIndex]:
        if not self._state.active or self._state.last_cell is None:
            return frozenset()

        if hasattr(self.field, "illuminate_prediction"):
            wave = self.field.illuminate_prediction(
                self._state.last_cell,
                self._state.velocity,
                uncertainty=self._state.uncertainty,
                forward_bias=self.config.forward_bias,
            )
        else:
            wave = self._fallback_wave(self._state.last_cell)

        self._state = BallWaveState(
            **{
                **self._state.__dict__,
                "wave_cell_count": len(wave),
            }
        )
        return frozenset(wave)

    def acquire(self, hit: BallInMeshHit) -> BallWaveState:
        if not hit.found or hit.index is None:
            raise ValueError("Ball Wave requires a found Ball-in-Mesh hit")

        generation = hit.ball_generation or self.searcher.ball_generation
        if generation != self.searcher.ball_generation:
            raise ValueError("Acquisition hit belongs to a different ball generation")

        self._state = BallWaveState(
            active=True,
            ball_generation=generation,
            last_cell=hit.index,
            velocity=(0.0, 0.0, 0.0),
            uncertainty=0.0,
            missed_frames=0,
            accepted_steps=1,
            wave_cell_count=0,
            requires_reacquisition=False,
        )
        self.current_wave()
        return self._state

    def _generation_is_current(self) -> bool:
        return self._state.ball_generation == self.searcher.ball_generation

    def _update_velocity(self, new_cell: CellIndex) -> tuple[float, float, float]:
        assert self._state.last_cell is not None
        last = self._state.last_cell
        raw = (
            float(new_cell.x - last.x),
            float(new_cell.y - last.y),
            float(new_cell.z - last.z),
        )
        if self._state.accepted_steps <= 1:
            return raw

        a = self.config.velocity_smoothing
        old = self._state.velocity
        return tuple(a * old[i] + (1.0 - a) * raw[i] for i in range(3))

    def track(self, frame, ball_reference) -> BallWaveStep:
        if not self._state.active or self._state.last_cell is None:
            return BallWaveStep(
                hit=self._empty_hit(self.searcher.ball_generation),
                state=self._state,
                local_candidates_tested=0,
                contact_plane_reached=False,
                floor_illumination=False,
                reason="REACQUISITION_REQUIRED",
            )

        if not self._generation_is_current():
            self._state = BallWaveState(
                active=False,
                ball_generation=self.searcher.ball_generation,
                last_cell=None,
                velocity=(0.0, 0.0, 0.0),
                uncertainty=0.0,
                missed_frames=0,
                accepted_steps=0,
                wave_cell_count=0,
                requires_reacquisition=True,
            )
            return BallWaveStep(
                hit=self._empty_hit(self.searcher.ball_generation),
                state=self._state,
                local_candidates_tested=0,
                contact_plane_reached=False,
                floor_illumination=False,
                reason="BALL_GENERATION_CHANGED",
            )

        wave = self.current_wave()
        hit = self.searcher.search_indices(frame, ball_reference, wave)

        if hit.found and hit.index is not None:
            velocity = self._update_velocity(hit.index)
            uncertainty = max(
                0.0,
                self._state.uncertainty - self.config.hit_uncertainty_recovery,
            )
            self._state = BallWaveState(
                active=True,
                ball_generation=self.searcher.ball_generation,
                last_cell=hit.index,
                velocity=velocity,
                uncertainty=uncertainty,
                missed_frames=0,
                accepted_steps=self._state.accepted_steps + 1,
                wave_cell_count=len(wave),
                requires_reacquisition=False,
            )
            return BallWaveStep(
                hit=hit,
                state=self._state,
                local_candidates_tested=len(wave),
                contact_plane_reached=(hit.index.z == 0),
                floor_illumination=False,
                reason="LOCAL_HIT",
            )

        misses = self._state.missed_frames + 1
        uncertainty = min(
            1.0,
            self._state.uncertainty + self.config.miss_uncertainty_step,
        )
        requires = misses >= self.config.max_local_misses
        self._state = BallWaveState(
            active=not requires,
            ball_generation=self.searcher.ball_generation,
            last_cell=self._state.last_cell,
            velocity=self._state.velocity,
            uncertainty=uncertainty,
            missed_frames=misses,
            accepted_steps=self._state.accepted_steps,
            wave_cell_count=len(wave),
            requires_reacquisition=requires,
        )
        return BallWaveStep(
            hit=hit,
            state=self._state,
            local_candidates_tested=len(wave),
            contact_plane_reached=False,
            floor_illumination=False,
            reason="REACQUISITION_REQUIRED" if requires else "LOCAL_MISS_EXPAND_WAVE",
        )

    def reacquire(self, frame, ball_reference) -> BallWaveStep:
        """
        Explicit global reacquisition of the SAME locked active ball.

        This never calls replace_ball(). A real ball replacement is a separate
        match event and increments the Ball Session generation.
        """
        hit = self.searcher.search(frame, ball_reference)
        if hit.found and hit.index is not None:
            self.acquire(hit)
            return BallWaveStep(
                hit=hit,
                state=self._state,
                local_candidates_tested=0,
                contact_plane_reached=(hit.index.z == 0),
                floor_illumination=False,
                reason="REACQUIRED_SAME_BALL",
            )

        self._state = BallWaveState(
            active=False,
            ball_generation=self.searcher.ball_generation,
            last_cell=None,
            velocity=(0.0, 0.0, 0.0),
            uncertainty=1.0,
            missed_frames=self._state.missed_frames,
            accepted_steps=0,
            wave_cell_count=0,
            requires_reacquisition=True,
        )
        return BallWaveStep(
            hit=hit,
            state=self._state,
            local_candidates_tested=0,
            contact_plane_reached=False,
            floor_illumination=False,
            reason="GLOBAL_REACQUISITION_MISS",
        )
