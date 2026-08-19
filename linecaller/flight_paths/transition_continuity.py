from __future__ import annotations

from dataclasses import dataclass
import math

from linecaller.dcf.ball_in_mesh_search import BallInMeshHit, BallInMeshSearcher
from linecaller.dcf.models import CellIndex


@dataclass(frozen=True)
class TransitionContinuityConfig:
    """Local same-ball bridge used immediately after an abrupt path change.

    The current flight vector is intentionally NOT used to aim the search:
    a paddle impact is exactly the event that invalidates that vector.  The
    last trusted P(x,y,z,t) remains valid, however, so the next frame is
    searched only inside a small physically local Ball-Unit neighborhood.

    DCF cells are used only to discretize the local search volume.  They do
    not define the continuous flight path.
    """

    base_radius_bu: float = 6.0
    miss_growth_bu: float = 3.0
    max_radius_bu: float = 12.0
    min_score: float = 0.38
    confirm_hits: int = 3

    def __post_init__(self) -> None:
        if not math.isfinite(self.base_radius_bu) or self.base_radius_bu <= 0.0:
            raise ValueError("base_radius_bu must be finite and > 0")
        if not math.isfinite(self.miss_growth_bu) or self.miss_growth_bu < 0.0:
            raise ValueError("miss_growth_bu must be finite and >= 0")
        if not math.isfinite(self.max_radius_bu) or self.max_radius_bu < self.base_radius_bu:
            raise ValueError("max_radius_bu must be finite and >= base_radius_bu")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be in [0,1]")
        if self.confirm_hits < 2:
            raise ValueError("confirm_hits must be >= 2")


@dataclass(frozen=True)
class TransitionContinuityStep:
    hit: BallInMeshHit
    reason: str
    candidate_count: int
    accepted_hits: int
    missed_frames: int
    new_path_confirmed: bool


class DirectionTransitionContinuity:
    """Preserve SAME BALL continuity while the old path is invalid.

    Contract:
      * Start from the last trusted same-ball hit.
      * Never call global acquisition/reacquisition.
      * Ignore the old direction during the transition bridge.
      * Search a local 3-D Ball-Unit sphere around the last trusted hit.
      * Update the local anchor after every accepted same-ball observation.
      * Require multiple progressive observations before declaring that enough
        data exists to fit the NEW flight path.

    This is a continuity bridge, not a path model and not a ball detector.
    Ball identity remains the responsibility of BallInMeshSearcher.
    """

    def __init__(
        self,
        searcher: BallInMeshSearcher,
        *,
        config: TransitionContinuityConfig | None = None,
    ) -> None:
        self.searcher = searcher
        self.field = searcher.field
        self.config = config or TransitionContinuityConfig()
        self._anchor: CellIndex | None = None
        self._accepted: list[BallInMeshHit] = []
        self._missed_frames = 0

    @property
    def anchor(self) -> CellIndex | None:
        return self._anchor

    @property
    def accepted_hits(self) -> tuple[BallInMeshHit, ...]:
        return tuple(self._accepted)

    @property
    def missed_frames(self) -> int:
        return self._missed_frames

    @property
    def new_path_confirmed(self) -> bool:
        return len(self._accepted) >= self.config.confirm_hits

    def start(self, last_trusted_hit: BallInMeshHit) -> None:
        if not last_trusted_hit.found or last_trusted_hit.index is None:
            raise ValueError("transition bridge requires a trusted same-ball hit")
        generation = last_trusted_hit.ball_generation or self.searcher.ball_generation
        if generation != self.searcher.ball_generation:
            raise ValueError("trusted hit belongs to another ball generation")
        self._anchor = last_trusted_hit.index
        self._accepted = []
        self._missed_frames = 0

    def _radius_bu(self) -> float:
        return min(
            self.config.max_radius_bu,
            self.config.base_radius_bu
            + self.config.miss_growth_bu * self._missed_frames,
        )

    def candidate_indices(self) -> frozenset[CellIndex]:
        if self._anchor is None:
            return frozenset()

        # CP-0035.9.4.1 already locked ONE BALL ~= ONE DCF CELL in XYZ.
        # Therefore one cell step is approximately one Ball Unit for this
        # local search discretization.  The continuous path itself remains
        # independent from cells.
        radius = self._radius_bu()
        extent = int(math.ceil(radius))
        a = self._anchor
        out: set[CellIndex] = set()

        r2 = radius * radius
        for dx in range(-extent, extent + 1):
            for dy in range(-extent, extent + 1):
                for dz in range(-extent, extent + 1):
                    if dx * dx + dy * dy + dz * dz > r2:
                        continue
                    q = CellIndex(a.x + dx, a.y + dy, a.z + dz)
                    if self.field.valid(q):
                        out.add(q)
        return frozenset(out)

    def track(self, frame, ball_reference) -> TransitionContinuityStep:
        if self._anchor is None:
            raise RuntimeError("start() must be called before track()")

        candidates = self.candidate_indices()
        hit = self.searcher.search_indices(
            frame,
            ball_reference,
            candidates,
            min_score=self.config.min_score,
        )

        if hit.found and hit.index is not None:
            self._anchor = hit.index
            self._accepted.append(hit)
            self._missed_frames = 0
            confirmed = self.new_path_confirmed
            return TransitionContinuityStep(
                hit=hit,
                reason=(
                    "NEW_PATH_OBSERVATIONS_READY"
                    if confirmed
                    else "TRANSITION_LOCAL_HIT"
                ),
                candidate_count=len(candidates),
                accepted_hits=len(self._accepted),
                missed_frames=0,
                new_path_confirmed=confirmed,
            )

        self._missed_frames += 1
        return TransitionContinuityStep(
            hit=hit,
            reason="TRANSITION_LOCAL_MISS_EXPAND",
            candidate_count=len(candidates),
            accepted_hits=len(self._accepted),
            missed_frames=self._missed_frames,
            new_path_confirmed=False,
        )
