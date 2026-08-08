from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from .geometry import line_intersection, order_quad, polygon_area
from .models import LineSegment


@dataclass(frozen=True)
class CourtHypothesis:
    corners: tuple[tuple[float, float], ...]
    score: float
    area_ratio: float
    line_support: float
    opposite_consistency: float
    separation_score: float


def _length(a, b):
    return math.hypot(b[0]-a[0], b[1]-a[1])


def _opposite_consistency(quad):
    bl, br, tr, tl = quad
    bottom = _length(bl, br)
    top = _length(tl, tr)
    left = _length(bl, tl)
    right = _length(br, tr)

    def ratio(a, b):
        if max(a,b) == 0:
            return 0.0
        return min(a,b) / max(a,b)

    return 0.5 * ratio(bottom, top) + 0.5 * ratio(left, right)


def build_hypothesis(
    a1: LineSegment,
    a2: LineSegment,
    b1: LineSegment,
    b2: LineSegment,
    *,
    image_width: int,
    image_height: int,
    separation_deg: float,
) -> CourtHypothesis | None:

    pts = [
        line_intersection(a1, b1),
        line_intersection(a1, b2),
        line_intersection(a2, b2),
        line_intersection(a2, b1),
    ]

    if any(p is None for p in pts):
        return None

    margin_x = image_width * 0.15
    margin_y = image_height * 0.15

    if not all(
        -margin_x <= p[0] <= image_width + margin_x
        and -margin_y <= p[1] <= image_height + margin_y
        for p in pts
    ):
        return None

    quad = order_quad(pts)
    area = polygon_area(quad)
    image_area = max(1.0, image_width * image_height)
    area_ratio = area / image_area

    if area_ratio <= 0:
        return None

    line_support = (
        a1.length_px + a2.length_px + b1.length_px + b2.length_px
    ) / max(1.0, 2.0 * (image_width + image_height))
    line_support = max(0.0, min(1.0, line_support))

    consistency = _opposite_consistency(quad)
    separation_score = max(0.0, min(1.0, separation_deg / 70.0))
    area_score = max(0.0, min(1.0, area_ratio / 0.45))

    score = (
        0.30 * line_support +
        0.30 * area_score +
        0.25 * consistency +
        0.15 * separation_score
    )

    return CourtHypothesis(
        corners=quad,
        score=float(score),
        area_ratio=float(area_ratio),
        line_support=float(line_support),
        opposite_consistency=float(consistency),
        separation_score=float(separation_score),
    )
