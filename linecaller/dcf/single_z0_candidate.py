from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from linecaller.dcf.external_grid_pillow_watcher import CellCandidate


@dataclass(frozen=True)
class SingleZ0Candidate:
    """
    Exactly ONE engineering candidate for a possible Z=0 contact cell.

    This is not a bounce decision.
    This is not a path/history object.
    """
    cell_id: int
    gx: int
    gy: int
    ball_pixels: int
    score: float
    polygon: tuple[tuple[float, float], ...]


class SingleZ0CandidateResolver:
    """
    Reduce any detector output to at most ONE Z0 candidate.

    Rules:
    - zero candidates -> None
    - one candidate -> that candidate
    - multiple candidates -> keep only the strongest
    - no history
    - no trajectory
    - no previous ball position
    - no DOWN
    - no UP confirmation yet
    """

    def resolve(self, candidates: Iterable[CellCandidate]) -> Optional[SingleZ0Candidate]:
        items = list(candidates)
        if not items:
            return None

        top = max(items, key=lambda c: (float(c.score), int(c.ball_pixels)))

        return SingleZ0Candidate(
            cell_id=int(top.cell_id),
            gx=int(top.gx),
            gy=int(top.gy),
            ball_pixels=int(top.ball_pixels),
            score=float(top.score),
            polygon=tuple((float(x), float(y)) for x, y in top.polygon),
        )
