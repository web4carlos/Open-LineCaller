from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from linecaller.dcf.ball_in_mesh_search import BallInMeshHit, BallInMeshSearcher
from linecaller.dcf.engine import DynamicCourtField
from linecaller.dcf.models import CellIndex


@dataclass(frozen=True)
class BallWaveConfig:
    """CP-0035.9.4.1 tuning for strict DCF next-cell continuity."""

    # Kept for CP-0035.9.4 compatibility.
    forward_bias: int = 2
    velocity_smoothing: float = 0.55
    miss_uncertainty_step: float = 0.22
    hit_uncertainty_recovery: float = 0.35
    max_local_misses: int = 3

    # New continuity rule: remember at most three accepted DCF cells.
    max_history_cells: int = 3

    # First handoff has no measured velocity yet, so it may inspect the
    # immediate 5x5x5 neighborhood. After two cells, prediction reduces this
    # to a 3x3x3 neighborhood around the expected next cell.
    first_radius_xy: int = 2
    first_radius_z: int = 2
    next_radius_xy: int = 1
    next_radius_z: int = 1
    miss_radius_bonus: int = 1

    # The active ball is already identified. Local continuity may therefore
    # accept weaker visual evidence than global acquisition, but only inside
    # the tiny DCF neighborhood above.
    continuity_min_score: float = 0.38

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
        if self.max_history_cells != 3:
            raise ValueError("CP-0035.9.4.1 requires max_history_cells == 3")
        for name in (
            "first_radius_xy",
            "first_radius_z",
            "next_radius_xy",
            "next_radius_z",
            "miss_radius_bonus",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if not 0.0 <= self.continuity_min_score <= 1.0:
            raise ValueError("continuity_min_score must be in [0,1]")


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
    history: tuple[CellIndex, ...] = ()


@dataclass(frozen=True)
class BallWaveStep:
    hit: BallInMeshHit
    state: BallWaveState
    local_candidates_tested: int
    contact_plane_reached: bool
    floor_illumination: bool
    reason: str
    predicted_contact_cell: CellIndex | None = None
    floor_trigger: bool = False


class BallWaveTracker:
    """
    CP-0035.9.4.1 -- Three-Cell DCF Continuity.

    The active pickleball is not rediscovered on every frame. Once one DCF
    cell is known, LineCaller remembers at most three accepted cells and asks
    only which nearby/predicted DCF cell the SAME ball passes through next.

    Production visibility remains unchanged: DCF / mesh / candidate cells are
    internal. ``floor_illumination`` is an event trigger only; CP-0035.9.5 is
    responsible for the visual glow.
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
        self._contact_latched = False
        self._state = self._inactive_state(
            ball_generation=searcher.ball_generation,
            requires_reacquisition=True,
        )

    @property
    def state(self) -> BallWaveState:
        return self._state

    @staticmethod
    def _empty_hit(generation: int, score: float = 0.0) -> BallInMeshHit:
        return BallInMeshHit(
            found=False,
            score=float(score),
            x=None,
            y=None,
            index=None,
            region=None,
            ball_generation=generation,
        )

    @staticmethod
    def _trim_history(history: Iterable[CellIndex]) -> tuple[CellIndex, ...]:
        return tuple(history)[-3:]

    def _inactive_state(
        self,
        *,
        ball_generation: int,
        requires_reacquisition: bool,
        missed_frames: int = 0,
    ) -> BallWaveState:
        return BallWaveState(
            active=False,
            ball_generation=ball_generation,
            last_cell=None,
            velocity=(0.0, 0.0, 0.0),
            uncertainty=1.0 if requires_reacquisition else 0.0,
            missed_frames=missed_frames,
            accepted_steps=0,
            wave_cell_count=0,
            requires_reacquisition=requires_reacquisition,
            history=(),
        )

    @staticmethod
    def _delta(a: CellIndex, b: CellIndex) -> tuple[float, float, float]:
        return (
            float(a.x - b.x),
            float(a.y - b.y),
            float(a.z - b.z),
        )

    @classmethod
    def _velocity_from_history(
        cls,
        history: tuple[CellIndex, ...],
    ) -> tuple[float, float, float]:
        if len(history) < 2:
            return (0.0, 0.0, 0.0)

        latest = cls._delta(history[-1], history[-2])
        if len(history) < 3:
            return latest

        previous = cls._delta(history[-2], history[-3])
        # Latest movement gets more weight while the older cell stabilizes
        # direction. This is trajectory memory, not a long history model.
        return tuple(
            0.35 * previous[i] + 0.65 * latest[i]
            for i in range(3)
        )

    def _prediction_horizon(self) -> int:
        # A miss does not freeze the ball. Keep moving the same short
        # trajectory forward, without widening into a global search.
        return max(1, self._state.missed_frames + 1)

    def _predicted_center(self) -> CellIndex:
        assert self._state.last_cell is not None
        last = self._state.last_cell
        vx, vy, vz = self._state.velocity
        h = self._prediction_horizon()
        return CellIndex(
            round(last.x + vx * h),
            round(last.y + vy * h),
            round(last.z + vz * h),
        )

    def _candidate_radii(self) -> tuple[int, int]:
        established = len(self._state.history) >= 2
        if established:
            rx = self.config.next_radius_xy
            rz = self.config.next_radius_z
        else:
            rx = self.config.first_radius_xy
            rz = self.config.first_radius_z

        if self._state.missed_frames:
            bonus = min(
                self.config.miss_radius_bonus,
                self._state.missed_frames,
            )
            rx += bonus
            rz += bonus
        return rx, rz

    def _candidate_cells(self) -> frozenset[CellIndex]:
        if not self._state.active or self._state.last_cell is None:
            return frozenset()

        center = self._predicted_center()
        rx, rz = self._candidate_radii()
        out: set[CellIndex] = set()

        for dx in range(-rx, rx + 1):
            for dy in range(-rx, rx + 1):
                for dz in range(-rz, rz + 1):
                    q = CellIndex(center.x + dx, center.y + dy, center.z + dz)
                    if self.field.valid(q):
                        out.add(q)

        # Before velocity exists, keep the known cell itself in the local
        # neighborhood. Once velocity exists, the center is already forward.
        if self.field.valid(self._state.last_cell):
            out.add(self._state.last_cell)

        return frozenset(out)

    def current_wave(self) -> frozenset[CellIndex]:
        wave = self._candidate_cells()
        self._state = BallWaveState(
            **{
                **self._state.__dict__,
                "wave_cell_count": len(wave),
            }
        )
        return wave

    def acquire(self, hit: BallInMeshHit) -> BallWaveState:
        if not hit.found or hit.index is None:
            raise ValueError("Ball Wave requires a found Ball-in-Mesh hit")

        generation = hit.ball_generation or self.searcher.ball_generation
        if generation != self.searcher.ball_generation:
            raise ValueError("Acquisition hit belongs to a different ball generation")

        history = (hit.index,)
        self._contact_latched = False
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
            history=history,
        )
        self.current_wave()
        return self._state

    def _generation_is_current(self) -> bool:
        return self._state.ball_generation == self.searcher.ball_generation

    def _local_search(self, frame, ball_reference, wave):
        # CP-0035.9.3 global acquisition keeps its original threshold. Only
        # this tiny continuity neighborhood uses the lower known-ball floor.
        try:
            return self.searcher.search_indices(
                frame,
                ball_reference,
                wave,
                min_score=self.config.continuity_min_score,
            )
        except TypeError as exc:
            # Backward compatibility for test doubles / older searchers.
            if "min_score" not in str(exc):
                raise
            return self.searcher.search_indices(frame, ball_reference, wave)

    def _predict_floor_from_z1(
        self,
        history: tuple[CellIndex, ...],
        velocity: tuple[float, float, float],
    ) -> CellIndex | None:
        if len(history) < 2:
            return None
        current = history[-1]
        if current.z != 1 or velocity[2] >= 0.0:
            return None

        contact = CellIndex(
            round(current.x + velocity[0]),
            round(current.y + velocity[1]),
            0,
        )
        if not self.field.valid(contact):
            return None
        return contact

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
            self._contact_latched = False
            self._state = self._inactive_state(
                ball_generation=self.searcher.ball_generation,
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
        hit = self._local_search(frame, ball_reference, wave)

        if hit.found and hit.index is not None:
            history = self._trim_history((*self._state.history, hit.index))
            velocity = self._velocity_from_history(history)

            # A completed bounce rearms once the same ball has clearly risen
            # back above Z=1.
            if self._contact_latched and hit.index.z >= 2 and velocity[2] > 0.0:
                self._contact_latched = False

            predicted_contact = None
            floor_trigger = False
            if not self._contact_latched:
                if hit.index.z == 0 and velocity[2] < 0.0:
                    # Strongest evidence: the accepted next cell is already
                    # on the Contact Plane.
                    predicted_contact = hit.index
                    self._contact_latched = True
                    floor_trigger = True
                else:
                    predicted_contact = self._predict_floor_from_z1(history, velocity)
                    if predicted_contact is not None:
                        self._contact_latched = True
                        floor_trigger = True

            self._state = BallWaveState(
                active=True,
                ball_generation=self.searcher.ball_generation,
                last_cell=hit.index,
                velocity=velocity,
                uncertainty=0.0,
                missed_frames=0,
                accepted_steps=self._state.accepted_steps + 1,
                wave_cell_count=len(wave),
                requires_reacquisition=False,
                history=history,
            )

            actual_z0 = hit.index.z == 0
            return BallWaveStep(
                hit=hit,
                state=self._state,
                local_candidates_tested=len(wave),
                contact_plane_reached=actual_z0 or floor_trigger,
                floor_illumination=False,
                reason=(
                    "CONTACT_FLOOR_TRIGGER"
                    if floor_trigger
                    else "LOCAL_HIT"
                ),
                predicted_contact_cell=predicted_contact,
                floor_trigger=floor_trigger,
            )

        misses = self._state.missed_frames + 1
        requires = misses >= self.config.max_local_misses
        uncertainty = min(
            1.0,
            self._state.uncertainty + self.config.miss_uncertainty_step,
        )

        if requires:
            # Keep generation semantics, but clear local path. Global
            # reacquisition remains explicit and is never called by track().
            self._state = BallWaveState(
                active=False,
                ball_generation=self.searcher.ball_generation,
                last_cell=self._state.last_cell,
                velocity=self._state.velocity,
                uncertainty=uncertainty,
                missed_frames=misses,
                accepted_steps=self._state.accepted_steps,
                wave_cell_count=len(wave),
                requires_reacquisition=True,
                history=self._state.history,
            )
        else:
            self._state = BallWaveState(
                **{
                    **self._state.__dict__,
                    "uncertainty": uncertainty,
                    "missed_frames": misses,
                    "wave_cell_count": len(wave),
                }
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
        """Explicit global reacquisition of the SAME locked active ball."""
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

        missed = self._state.missed_frames
        self._contact_latched = False
        self._state = self._inactive_state(
            ball_generation=self.searcher.ball_generation,
            requires_reacquisition=True,
            missed_frames=missed,
        )
        return BallWaveStep(
            hit=hit,
            state=self._state,
            local_candidates_tested=0,
            contact_plane_reached=False,
            floor_illumination=False,
            reason="GLOBAL_REACQUISITION_MISS",
        )
