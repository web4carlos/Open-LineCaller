from __future__ import annotations

from dataclasses import dataclass

from .models import CourtModel


@dataclass(frozen=True)
class CourtDistances:
    left: float
    right: float
    near_baseline: float
    far_baseline: float

    @property
    def minimum_signed(self) -> float:
        return min(self.left, self.right, self.near_baseline, self.far_baseline)


def signed_distances_to_court(x_m: float, y_m: float, court: CourtModel) -> CourtDistances:
    return CourtDistances(
        left=x_m,
        right=court.width_m - x_m,
        near_baseline=y_m,
        far_baseline=court.length_m - y_m,
    )
