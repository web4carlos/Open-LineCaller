from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


# LineCaller's continuous spatial unit is one ball diameter.  The current
# locked X scale is exactly the 20-ft court width divided into 83 ball-sized
# units: 2.891566... inches per unit.  This is a spatial scale only; Ball
# Identity (color/image/appearance) is deliberately separate.
COURT_WIDTH_M = 6.096
COURT_LENGTH_M = 13.4112
COURT_WIDTH_BALL_UNITS = 83.0
BALL_UNIT_M = COURT_WIDTH_M / COURT_WIDTH_BALL_UNITS
COURT_LENGTH_BALL_UNITS = COURT_LENGTH_M / BALL_UNIT_M  # exactly 182.6
STANDARD_GRAVITY_MPS2 = 9.80665
STANDARD_GRAVITY_BU_S2 = STANDARD_GRAVITY_MPS2 / BALL_UNIT_M


class CourtSide(str, Enum):
    NEAR = "NEAR"
    FAR = "FAR"


class LateralSide(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


@dataclass(frozen=True)
class Point3D:
    """One mathematical point P=(x,y,z) in continuous court Ball Units.

    One Ball Unit (BU) is approximately one pickleball diameter.  The tern
    (x,y,z) is a point, not a ball shape, cell, tube, path, or image feature.
    """

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.x, self.y, self.z)):
            raise ValueError("Point3D coordinates must be finite")
        if self.z < 0.0:
            raise ValueError("Point3D.z cannot be below the Contact Plane")


@dataclass(frozen=True)
class TimedPoint3D:
    """One point observed at a real timestamp in seconds."""

    point: Point3D
    t_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.t_s):
            raise ValueError("t_s must be finite")

    @property
    def x(self) -> float:
        return self.point.x

    @property
    def y(self) -> float:
        return self.point.y

    @property
    def z(self) -> float:
        return self.point.z


@dataclass(frozen=True)
class Velocity3D:
    """Velocity V=(vx,vy,vz) in Ball Units per second."""

    vx: float
    vy: float
    vz: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.vx, self.vy, self.vz)):
            raise ValueError("Velocity3D components must be finite")

    @property
    def speed_bu_s(self) -> float:
        return math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)

    @classmethod
    def between(cls, a: TimedPoint3D, b: TimedPoint3D) -> "Velocity3D":
        dt = b.t_s - a.t_s
        if dt <= 0.0:
            raise ValueError("observations must have increasing time")
        return cls(
            (b.x - a.x) / dt,
            (b.y - a.y) / dt,
            (b.z - a.z) / dt,
        )


@dataclass(frozen=True)
class CourtFrame:
    """Continuous pickleball court in Ball Units.

    x=0 left sideline, x increases right.
    y=0 FAR baseline, y increases toward NEAR.
    z=0 Contact Plane, z increases upward.

    The current physical court maps to 83.0 BU wide and 182.6 BU long using
    the same ~2.8916-inch unit on X, Y, and Z.  The existing 83x182 DCF is a
    later discretization/indexing layer; it does not define this curve space.
    """

    width_bu: float = COURT_WIDTH_BALL_UNITS
    length_bu: float = COURT_LENGTH_BALL_UNITS
    ball_unit_m: float = BALL_UNIT_M

    def __post_init__(self) -> None:
        if self.width_bu <= 0.0 or self.length_bu <= 0.0 or self.ball_unit_m <= 0.0:
            raise ValueError("Court frame dimensions and Ball Unit scale must be positive")

    @property
    def mid_x(self) -> float:
        return self.width_bu / 2.0

    @property
    def mid_y(self) -> float:
        return self.length_bu / 2.0

    @property
    def ball_unit_in(self) -> float:
        return self.ball_unit_m / 0.0254

    @property
    def gravity_bu_s2(self) -> float:
        return STANDARD_GRAVITY_MPS2 / self.ball_unit_m

    def side_at(self, point: Point3D) -> CourtSide:
        return CourtSide.NEAR if point.y >= self.mid_y else CourtSide.FAR

    def lateral_at(self, point: Point3D) -> LateralSide:
        return LateralSide.LEFT if point.x < self.mid_x else LateralSide.RIGHT

    @staticmethod
    def opposite(side: CourtSide) -> CourtSide:
        return CourtSide.FAR if side == CourtSide.NEAR else CourtSide.NEAR


@dataclass(frozen=True)
class LandingRegion:
    """Continuous receiver region on Z=0, expressed in Ball Units."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise ValueError("Invalid LandingRegion bounds")

    def contains(self, point: Point3D, *, eps: float = 1e-9) -> bool:
        return (
            abs(point.z) <= eps
            and self.x_min - eps <= point.x <= self.x_max + eps
            and self.y_min - eps <= point.y <= self.y_max + eps
        )
