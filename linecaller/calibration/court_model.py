from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class PickleballCourtModel:
    """Official pickleball court geometry in meters."""

    width_m: float = 6.096
    length_m: float = 13.4112
    nvz_depth_m: float = 2.1336

    @property
    def net_y(self) -> float:
        return self.length_m / 2.0

    @property
    def near_nvz_y(self) -> float:
        return self.net_y - self.nvz_depth_m

    @property
    def far_nvz_y(self) -> float:
        return self.net_y + self.nvz_depth_m

    @property
    def center_x(self) -> float:
        return self.width_m / 2.0

    def reference_points(self) -> Dict[str, Point]:
        """
        Reference points useful for homography validation.
        Coordinates are on the court plane.
        """
        w = self.width_m
        l = self.length_m
        c = self.center_x
        n1 = self.near_nvz_y
        n2 = self.far_nvz_y

        return {
            "near_left_corner": (0.0, 0.0),
            "near_right_corner": (w, 0.0),
            "far_right_corner": (w, l),
            "far_left_corner": (0.0, l),
            "near_nvz_left": (0.0, n1),
            "near_nvz_center": (c, n1),
            "near_nvz_right": (w, n1),
            "far_nvz_left": (0.0, n2),
            "far_nvz_center": (c, n2),
            "far_nvz_right": (w, n2),
            "near_baseline_center": (c, 0.0),
            "far_baseline_center": (c, l),
        }

    def line_segments(self) -> Dict[str, tuple[Point, Point]]:
        w = self.width_m
        l = self.length_m
        c = self.center_x
        n1 = self.near_nvz_y
        n2 = self.far_nvz_y

        return {
            "near_baseline": ((0.0, 0.0), (w, 0.0)),
            "far_baseline": ((0.0, l), (w, l)),
            "left_sideline": ((0.0, 0.0), (0.0, l)),
            "right_sideline": ((w, 0.0), (w, l)),
            "near_nvz": ((0.0, n1), (w, n1)),
            "far_nvz": ((0.0, n2), (w, n2)),
            "near_centerline": ((c, 0.0), (c, n1)),
            "far_centerline": ((c, n2), (c, l)),
        }
