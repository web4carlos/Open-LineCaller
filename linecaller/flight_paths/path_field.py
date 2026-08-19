from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .models import (
    CourtFrame,
    CourtSide,
    LandingRegion,
    LateralSide,
    Point3D,
    TimedPoint3D,
    Velocity3D,
)


@dataclass(frozen=True)
class FlightPath:
    """One continuous 3D curve P(t) from P0, V0 and a start time.

    Coordinates are continuous Ball Units (BU); time is seconds.  The ball's
    visual identity and the DCF mesh are not inputs to this equation.
    """

    p0: Point3D
    v0: Velocity3D
    t0_s: float = 0.0
    gravity_bu_s2: float = CourtFrame().gravity_bu_s2

    def __post_init__(self) -> None:
        if not math.isfinite(self.t0_s):
            raise ValueError("t0_s must be finite")
        if not math.isfinite(self.gravity_bu_s2) or self.gravity_bu_s2 <= 0.0:
            raise ValueError("gravity_bu_s2 must be finite and positive")

    def point_at_elapsed(self, dt_s: float) -> Point3D:
        if not math.isfinite(dt_s) or dt_s < 0.0:
            raise ValueError("dt_s must be finite and >= 0")
        z = self.p0.z + self.v0.vz * dt_s - 0.5 * self.gravity_bu_s2 * dt_s * dt_s
        if -1e-8 < z < 0.0:
            z = 0.0
        if z < 0.0:
            raise ValueError("Requested time is after this path crossed Z=0")
        return Point3D(
            self.p0.x + self.v0.vx * dt_s,
            self.p0.y + self.v0.vy * dt_s,
            z,
        )

    def point_at_time(self, t_s: float) -> Point3D:
        if not math.isfinite(t_s) or t_s < self.t0_s:
            raise ValueError("t_s must be finite and >= t0_s")
        return self.point_at_elapsed(t_s - self.t0_s)

    def timed_point_at(self, t_s: float) -> TimedPoint3D:
        return TimedPoint3D(self.point_at_time(t_s), t_s)

    def velocity_at_elapsed(self, dt_s: float) -> Velocity3D:
        if not math.isfinite(dt_s) or dt_s < 0.0:
            raise ValueError("dt_s must be finite and >= 0")
        return Velocity3D(
            self.v0.vx,
            self.v0.vy,
            self.v0.vz - self.gravity_bu_s2 * dt_s,
        )

    def velocity_at_time(self, t_s: float) -> Velocity3D:
        if not math.isfinite(t_s) or t_s < self.t0_s:
            raise ValueError("t_s must be finite and >= t0_s")
        return self.velocity_at_elapsed(t_s - self.t0_s)

    @property
    def flight_duration_s(self) -> float:
        # z(dt) = z0 + vz*dt - 1/2*g*dt^2 = 0
        disc = self.v0.vz * self.v0.vz + 2.0 * self.gravity_bu_s2 * self.p0.z
        return (self.v0.vz + math.sqrt(disc)) / self.gravity_bu_s2

    @property
    def contact_time_s(self) -> float:
        return self.t0_s + self.flight_duration_s

    @property
    def landing_point(self) -> Point3D:
        dt = self.flight_duration_s
        return Point3D(
            self.p0.x + self.v0.vx * dt,
            self.p0.y + self.v0.vy * dt,
            0.0,
        )

    @property
    def landing_observation(self) -> TimedPoint3D:
        return TimedPoint3D(self.landing_point, self.contact_time_s)

    def samples(self, count: int) -> tuple[TimedPoint3D, ...]:
        if count < 2:
            raise ValueError("count must be >= 2")
        duration = self.flight_duration_s
        out = []
        for i in range(count):
            if i == count - 1:
                out.append(self.landing_observation)
            else:
                dt = duration * i / (count - 1)
                out.append(TimedPoint3D(self.point_at_elapsed(dt), self.t0_s + dt))
        return tuple(out)

    @classmethod
    def between_points(
        cls,
        p0: Point3D,
        landing: Point3D,
        flight_time_s: float,
        *,
        t0_s: float = 0.0,
        gravity_bu_s2: float | None = None,
    ) -> "FlightPath":
        """Construct one curved path joining one P0 to one chosen Z=0 point.

        One P0 can therefore generate N possible continuous paths to N possible
        Z=0 receiver points.  Runtime P0 + V0 + time selects/updates the active
        member of that family.
        """
        if abs(landing.z) > 1e-9:
            raise ValueError("landing must lie on Z=0")
        if not math.isfinite(flight_time_s) or flight_time_s <= 0.0:
            raise ValueError("flight_time_s must be finite and > 0")
        g = CourtFrame().gravity_bu_s2 if gravity_bu_s2 is None else gravity_bu_s2
        t = float(flight_time_s)
        v0 = Velocity3D(
            (landing.x - p0.x) / t,
            (landing.y - p0.y) / t,
            (landing.z - p0.z + 0.5 * g * t * t) / t,
        )
        return cls(p0=p0, v0=v0, t0_s=t0_s, gravity_bu_s2=g)


@dataclass(frozen=True)
class PathSelection:
    path: FlightPath
    origin_side: CourtSide
    target_side: CourtSide
    target_lateral: LateralSide
    landing: Point3D
    landing_region: LandingRegion
    target_compatible: bool

    @property
    def contact_time_s(self) -> float:
        return self.path.contact_time_s


class CourtPathField:
    """Continuous family of court paths, independent from the mesh.

    The court provides the continuous XYZ frame.  One P0 plus an outgoing V0
    at real time t0 selects a predicted path and a predicted Z=0 landing point.
    """

    def __init__(self, court: CourtFrame | None = None):
        self.court = court or CourtFrame()

    def landing_region(
        self,
        p0: Point3D,
        target_lateral: LateralSide,
        *,
        officiating_margin_bu: float = 0.0,
    ) -> LandingRegion:
        if officiating_margin_bu < 0.0 or not math.isfinite(officiating_margin_bu):
            raise ValueError("officiating_margin_bu must be finite and >= 0")
        target_side = self.court.opposite(self.court.side_at(p0))
        m = float(officiating_margin_bu)

        if target_lateral == LateralSide.LEFT:
            x_min, x_max = -m, self.court.mid_x
        else:
            x_min, x_max = self.court.mid_x, self.court.width_bu + m

        if target_side == CourtSide.FAR:
            y_min, y_max = -m, self.court.mid_y
        else:
            y_min, y_max = self.court.mid_y, self.court.length_bu + m

        return LandingRegion(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    def path_to_landing(
        self,
        p0: Point3D,
        landing: Point3D,
        flight_time_s: float,
        *,
        t0_s: float = 0.0,
    ) -> FlightPath:
        return FlightPath.between_points(
            p0,
            landing,
            flight_time_s,
            t0_s=t0_s,
            gravity_bu_s2=self.court.gravity_bu_s2,
        )

    def possible_paths(
        self,
        p0: Point3D,
        z0_points: Iterable[Point3D],
        flight_time_s: float,
        *,
        t0_s: float = 0.0,
    ) -> tuple[FlightPath, ...]:
        return tuple(
            self.path_to_landing(p0, p, flight_time_s, t0_s=t0_s)
            for p in z0_points
        )

    def select_from_state(
        self,
        p0: Point3D,
        v0: Velocity3D,
        *,
        t0_s: float,
        target_lateral: LateralSide | None = None,
        officiating_margin_bu: float = 0.0,
    ) -> PathSelection:
        path = FlightPath(
            p0=p0,
            v0=v0,
            t0_s=t0_s,
            gravity_bu_s2=self.court.gravity_bu_s2,
        )
        landing = path.landing_point
        origin = self.court.side_at(p0)
        target = self.court.opposite(origin)
        lateral = target_lateral or self.court.lateral_at(landing)
        region = self.landing_region(
            p0,
            lateral,
            officiating_margin_bu=officiating_margin_bu,
        )
        return PathSelection(
            path=path,
            origin_side=origin,
            target_side=target,
            target_lateral=lateral,
            landing=landing,
            landing_region=region,
            target_compatible=(self.court.side_at(landing) == target and region.contains(landing)),
        )
