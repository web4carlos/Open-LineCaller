from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from linecaller.calibration.court_model import PickleballCourtModel


@dataclass(frozen=True)
class BoundaryDistance:
    nearest_line: str
    signed_distance_m: float
    inside: bool


class CourtBoundaryGeometry:
    """
    Rally-boundary geometry for the four outer edges of a pickleball court.

    Positive signed distance = inside.
    Negative signed distance = outside.
    """

    def __init__(self, court: PickleballCourtModel | None = None):
        self.court = court or PickleballCourtModel()

    def signed_distance(self, x: float, y: float) -> BoundaryDistance:
        w = self.court.width_m
        l = self.court.length_m

        inside = (0.0 <= x <= w) and (0.0 <= y <= l)

        if inside:
            distances = {
                "left_sideline": x,
                "right_sideline": w - x,
                "near_baseline": y,
                "far_baseline": l - y,
            }
            line = min(distances, key=distances.get)
            return BoundaryDistance(
                nearest_line=line,
                signed_distance_m=float(distances[line]),
                inside=True,
            )

        # Outside: compute Euclidean distance to the legal rectangle.
        clamped_x = min(max(x, 0.0), w)
        clamped_y = min(max(y, 0.0), l)
        dx = x - clamped_x
        dy = y - clamped_y
        outside_distance = math.hypot(dx, dy)

        # Identify the most relevant boundary. At corners choose the axis
        # with the larger violation; this keeps the reason explainable.
        violations = {
            "left_sideline": max(0.0, -x),
            "right_sideline": max(0.0, x - w),
            "near_baseline": max(0.0, -y),
            "far_baseline": max(0.0, y - l),
        }
        line = max(violations, key=violations.get)

        return BoundaryDistance(
            nearest_line=line,
            signed_distance_m=-float(outside_distance),
            inside=False,
        )
