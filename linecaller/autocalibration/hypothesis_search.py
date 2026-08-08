from __future__ import annotations
import itertools
import math

from .hypothesis import CourtHypothesis, build_hypothesis
from .models import LineSegment


class HypothesisSearch:
    def __init__(self, *, max_lines_per_family: int = 8, dedupe_distance_px: float = 35.0):
        self.max_lines_per_family = int(max_lines_per_family)
        self.dedupe_distance_px = float(dedupe_distance_px)

    def _dedupe(self, lines: list[LineSegment]) -> list[LineSegment]:
        chosen = []
        for line in sorted(lines, key=lambda l: l.length_px, reverse=True):
            mx = (line.x1 + line.x2) / 2.0
            my = (line.y1 + line.y2) / 2.0

            duplicate = False
            for existing in chosen:
                ex = (existing.x1 + existing.x2) / 2.0
                ey = (existing.y1 + existing.y2) / 2.0
                if math.hypot(mx-ex, my-ey) < self.dedupe_distance_px:
                    duplicate = True
                    break

            if not duplicate:
                chosen.append(line)

            if len(chosen) >= self.max_lines_per_family:
                break

        return chosen

    def generate(
        self,
        *,
        family_a: list[LineSegment],
        family_b: list[LineSegment],
        image_width: int,
        image_height: int,
        separation_deg: float,
    ) -> list[CourtHypothesis]:

        a_lines = self._dedupe(family_a)
        b_lines = self._dedupe(family_b)

        hypotheses = []

        for a1, a2 in itertools.combinations(a_lines, 2):
            for b1, b2 in itertools.combinations(b_lines, 2):
                h = build_hypothesis(
                    a1, a2, b1, b2,
                    image_width=image_width,
                    image_height=image_height,
                    separation_deg=separation_deg,
                )
                if h is not None:
                    hypotheses.append(h)

        hypotheses.sort(key=lambda h: h.score, reverse=True)
        return hypotheses
