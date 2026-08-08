from __future__ import annotations

import itertools
import math

from .geometry import line_intersection, order_quad, polygon_area
from .models import (
    AutoCalibrationDisposition,
    AutoCalibrationProposal,
    LineSegment,
)


class CourtProposalBuilder:
    def __init__(
        self,
        *,
        min_area_ratio: float = 0.08,
        auto_review_threshold: float = 0.45,
        auto_accept_threshold: float = 0.72,
    ):
        self.min_area_ratio = float(min_area_ratio)
        self.auto_review_threshold = float(auto_review_threshold)
        self.auto_accept_threshold = float(auto_accept_threshold)

    @staticmethod
    def _top_distinct(lines: list[LineSegment], limit=6):
        # Simple first-stage dedupe: keep long lines with different midpoints.
        chosen = []
        for line in sorted(lines, key=lambda l: l.length_px, reverse=True):
            mx = (line.x1 + line.x2) / 2
            my = (line.y1 + line.y2) / 2

            duplicate = False
            for existing in chosen:
                ex = (existing.x1 + existing.x2) / 2
                ey = (existing.y1 + existing.y2) / 2
                if math.hypot(mx-ex, my-ey) < 35.0:
                    duplicate = True
                    break

            if not duplicate:
                chosen.append(line)

            if len(chosen) >= limit:
                break
        return chosen

    def build(
        self,
        *,
        family_a: list[LineSegment],
        family_b: list[LineSegment],
        separation_deg: float,
        image_width: int,
        image_height: int,
        total_line_count: int,
    ) -> AutoCalibrationProposal:

        reasons = []

        if len(family_a) < 2 or len(family_b) < 2:
            return AutoCalibrationProposal(
                corners=(),
                confidence=0.0,
                disposition=AutoCalibrationDisposition.REJECT,
                line_count=total_line_count,
                family_a_count=len(family_a),
                family_b_count=len(family_b),
                orientation_separation_deg=separation_deg,
                quadrilateral_area_ratio=0.0,
                reasons=("Not enough lines in both orientation families",),
            )

        a_lines = self._top_distinct(family_a)
        b_lines = self._top_distinct(family_b)

        best = None

        for a1, a2 in itertools.combinations(a_lines, 2):
            for b1, b2 in itertools.combinations(b_lines, 2):
                pts = [
                    line_intersection(a1, b1),
                    line_intersection(a1, b2),
                    line_intersection(a2, b2),
                    line_intersection(a2, b1),
                ]

                if any(p is None for p in pts):
                    continue

                margin_x = image_width * 0.15
                margin_y = image_height * 0.15

                if not all(
                    -margin_x <= p[0] <= image_width + margin_x
                    and -margin_y <= p[1] <= image_height + margin_y
                    for p in pts
                ):
                    continue

                quad = order_quad(pts)
                area = polygon_area(quad)
                area_ratio = area / max(1.0, image_width * image_height)

                if area_ratio < self.min_area_ratio:
                    continue

                line_support = (
                    a1.length_px + a2.length_px + b1.length_px + b2.length_px
                ) / max(1.0, 2.0 * (image_width + image_height))

                separation_score = min(1.0, separation_deg / 70.0)
                area_score = min(1.0, area_ratio / 0.45)
                support_score = min(1.0, line_support)

                confidence = (
                    0.40 * support_score
                    + 0.35 * area_score
                    + 0.25 * separation_score
                )

                if best is None or confidence > best[0]:
                    best = (confidence, quad, area_ratio)

        if best is None:
            return AutoCalibrationProposal(
                corners=(),
                confidence=0.0,
                disposition=AutoCalibrationDisposition.REJECT,
                line_count=total_line_count,
                family_a_count=len(family_a),
                family_b_count=len(family_b),
                orientation_separation_deg=separation_deg,
                quadrilateral_area_ratio=0.0,
                reasons=("No plausible court quadrilateral found",),
            )

        confidence, quad, area_ratio = best

        if confidence >= self.auto_accept_threshold:
            disposition = AutoCalibrationDisposition.AUTO_ACCEPT
        elif confidence >= self.auto_review_threshold:
            disposition = AutoCalibrationDisposition.AUTO_REVIEW
        else:
            disposition = AutoCalibrationDisposition.REJECT
            reasons.append("Proposal confidence below review threshold")

        return AutoCalibrationProposal(
            corners=quad,
            confidence=float(confidence),
            disposition=disposition,
            line_count=total_line_count,
            family_a_count=len(family_a),
            family_b_count=len(family_b),
            orientation_separation_deg=float(separation_deg),
            quadrilateral_area_ratio=float(area_ratio),
            reasons=tuple(reasons),
        )
