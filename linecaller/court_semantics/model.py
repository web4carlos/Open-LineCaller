from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from math import hypot

class CourtZone(str, Enum):
    OUTSIDE = "OUTSIDE"
    FAR_LEFT_SERVICE = "FAR_LEFT_SERVICE"
    FAR_RIGHT_SERVICE = "FAR_RIGHT_SERVICE"
    FAR_KITCHEN = "FAR_KITCHEN"
    NEAR_LEFT_SERVICE = "NEAR_LEFT_SERVICE"
    NEAR_RIGHT_SERVICE = "NEAR_RIGHT_SERVICE"
    NEAR_KITCHEN = "NEAR_KITCHEN"
    NET = "NET"

class CourtLine(str, Enum):
    LEFT_SIDELINE = "LEFT_SIDELINE"
    RIGHT_SIDELINE = "RIGHT_SIDELINE"
    FAR_BASELINE = "FAR_BASELINE"
    NEAR_BASELINE = "NEAR_BASELINE"
    FAR_NVZ_LINE = "FAR_NVZ_LINE"
    NEAR_NVZ_LINE = "NEAR_NVZ_LINE"
    FAR_CENTERLINE = "FAR_CENTERLINE"
    NEAR_CENTERLINE = "NEAR_CENTERLINE"
    NET = "NET"

@dataclass(frozen=True)
class CourtSemanticResult:
    x_ft: float
    y_ft: float
    inside_outer_court: bool
    zone: CourtZone
    nearest_line: CourtLine
    signed_distance_ft: float
    absolute_distance_ft: float
    on_line: bool

class CourtSemanticModel:
    WIDTH_FT = 20.0
    LENGTH_FT = 44.0
    FAR_BASELINE_Y = 0.0
    FAR_NVZ_Y = 15.0
    NET_Y = 22.0
    NEAR_NVZ_Y = 29.0
    NEAR_BASELINE_Y = 44.0
    CENTER_X = 10.0

    def __init__(self, *, line_tolerance_in: float = 2.0, net_band_in: float = 2.0):
        self.line_tolerance_ft = float(line_tolerance_in) / 12.0
        self.net_band_ft = float(net_band_in) / 12.0

    def _inside_outer(self, x, y):
        return 0.0 <= x <= self.WIDTH_FT and 0.0 <= y <= self.LENGTH_FT

    def zone_for(self, x, y):
        x = float(x); y = float(y)
        if not self._inside_outer(x, y):
            return CourtZone.OUTSIDE
        if abs(y - self.NET_Y) <= self.net_band_ft:
            return CourtZone.NET
        if y == self.FAR_NVZ_Y:
            return CourtZone.FAR_KITCHEN
        if y == self.NEAR_NVZ_Y:
            return CourtZone.NEAR_KITCHEN
        if self.FAR_NVZ_Y < y < self.NET_Y:
            return CourtZone.FAR_KITCHEN
        if self.NET_Y < y < self.NEAR_NVZ_Y:
            return CourtZone.NEAR_KITCHEN
        if y < self.FAR_NVZ_Y:
            return CourtZone.FAR_LEFT_SERVICE if x < self.CENTER_X else CourtZone.FAR_RIGHT_SERVICE
        if y > self.NEAR_NVZ_Y:
            return CourtZone.NEAR_LEFT_SERVICE if x < self.CENTER_X else CourtZone.NEAR_RIGHT_SERVICE
        return CourtZone.OUTSIDE

    @staticmethod
    def _segment_distance(px, py, x1, y1, x2, y2):
        vx = x2-x1; vy = y2-y1
        wx = px-x1; wy = py-y1
        denom = vx*vx + vy*vy
        if denom <= 1e-12:
            return hypot(px-x1, py-y1)
        t = max(0.0, min(1.0, (wx*vx + wy*vy)/denom))
        cx = x1 + t*vx; cy = y1 + t*vy
        return hypot(px-cx, py-cy)

    def line_distances(self, x, y):
        x = float(x); y = float(y)
        d = {
            CourtLine.LEFT_SIDELINE: x,
            CourtLine.RIGHT_SIDELINE: self.WIDTH_FT-x,
            CourtLine.FAR_BASELINE: y,
            CourtLine.NEAR_BASELINE: self.LENGTH_FT-y,
            CourtLine.FAR_NVZ_LINE: y-self.FAR_NVZ_Y,
            CourtLine.NEAR_NVZ_LINE: self.NEAR_NVZ_Y-y,
            CourtLine.NET: y-self.NET_Y,
        }
        d[CourtLine.FAR_CENTERLINE] = self._segment_distance(
            x, y, self.CENTER_X, self.FAR_BASELINE_Y, self.CENTER_X, self.FAR_NVZ_Y
        )
        d[CourtLine.NEAR_CENTERLINE] = self._segment_distance(
            x, y, self.CENTER_X, self.NEAR_NVZ_Y, self.CENTER_X, self.NEAR_BASELINE_Y
        )
        return d

    def nearest_line(self, x, y):
        distances = self.line_distances(x, y)
        line = min(distances, key=lambda k: abs(distances[k]))
        return line, float(distances[line])

    def classify(self, x, y):
        x = float(x); y = float(y)
        zone = self.zone_for(x, y)
        line, signed = self.nearest_line(x, y)
        absolute = abs(signed)
        return CourtSemanticResult(
            x_ft=x, y_ft=y, inside_outer_court=self._inside_outer(x,y),
            zone=zone, nearest_line=line, signed_distance_ft=signed,
            absolute_distance_ft=absolute, on_line=absolute <= self.line_tolerance_ft
        )

