from __future__ import annotations

import math
import numpy as np

from .models import LineSegment


def circular_angle_distance_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


class OrientationFamilies:
    def __init__(
        self,
        *,
        bin_size_deg: float = 5.0,
        family_tolerance_deg: float = 12.0,
        min_family_separation_deg: float = 25.0,
    ):
        self.bin_size_deg = float(bin_size_deg)
        self.family_tolerance_deg = float(family_tolerance_deg)
        self.min_family_separation_deg = float(min_family_separation_deg)

    def split(self, lines: list[LineSegment]):
        if len(lines) < 2:
            return [], [], 0.0

        bins = np.arange(0.0, 180.0 + self.bin_size_deg, self.bin_size_deg)
        angles = np.asarray([l.angle_deg for l in lines], dtype=float)
        weights = np.asarray([l.length_px for l in lines], dtype=float)

        hist, edges = np.histogram(angles, bins=bins, weights=weights)
        order = list(np.argsort(hist)[::-1])

        first_center = None
        second_center = None

        for idx in order:
            if hist[idx] <= 0:
                continue
            center = (edges[idx] + edges[idx + 1]) / 2.0

            if first_center is None:
                first_center = center
                continue

            if circular_angle_distance_deg(center, first_center) >= self.min_family_separation_deg:
                second_center = center
                break

        if first_center is None or second_center is None:
            return [], [], 0.0

        family_a = [
            l for l in lines
            if circular_angle_distance_deg(l.angle_deg, first_center) <= self.family_tolerance_deg
        ]
        family_b = [
            l for l in lines
            if circular_angle_distance_deg(l.angle_deg, second_center) <= self.family_tolerance_deg
        ]

        separation = circular_angle_distance_deg(first_center, second_center)
        return family_a, family_b, separation
