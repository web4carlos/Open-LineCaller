from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from .models import CourtFrame, Point3D, TimedPoint3D, Velocity3D


@dataclass(frozen=True)
class RollingZ0Prediction:
    """One Z=0 estimate from exactly the latest three trusted observations.

    The predictor does not build or retain a complete FlightPath.  It estimates
    the current local velocity at the newest point and projects only far enough
    to intersect the Contact Plane Z=0.  A later observation replaces the
    oldest point and recomputes the estimate.
    """

    observations: tuple[TimedPoint3D, TimedPoint3D, TimedPoint3D]
    current: TimedPoint3D
    velocity: Velocity3D
    landing: Point3D
    time_to_z0_s: float
    contact_time_s: float


class RollingThreePointZ0Predictor:
    """Estimate the next Z=0 point from a rolling three-point window.

    Spatial coordinates are continuous Ball Units (BU), time is seconds.
    Gravity is used only as local vertical physics.  No mesh/cell index, ball
    appearance, or complete stored path is an input to this calculation.
    """

    def __init__(self, *, gravity_bu_s2: float | None = None) -> None:
        g = CourtFrame().gravity_bu_s2 if gravity_bu_s2 is None else float(gravity_bu_s2)
        if not math.isfinite(g) or g <= 0.0:
            raise ValueError("gravity_bu_s2 must be finite and > 0")
        self.gravity_bu_s2 = g

    @staticmethod
    def _validate(points) -> tuple[TimedPoint3D, TimedPoint3D, TimedPoint3D]:
        pts = tuple(points)
        if len(pts) != 3:
            raise ValueError("rolling Z0 prediction requires exactly 3 observations")
        if not (pts[0].t_s < pts[1].t_s < pts[2].t_s):
            raise ValueError("observation times must increase strictly")
        return pts  # type: ignore[return-value]

    @staticmethod
    def _slope_anchored_at_latest(values, times, latest_value, latest_t) -> float:
        # Least-squares slope through the newest trusted point.  The newest
        # position is therefore preserved exactly while the two older points
        # smooth the local velocity estimate.
        num = 0.0
        den = 0.0
        for value, t in zip(values[:-1], times[:-1]):
            s = t - latest_t  # negative for the two older observations
            num += s * (value - latest_value)
            den += s * s
        if den <= 0.0:
            raise ValueError("observation times are degenerate")
        return num / den

    def predict(self, points) -> RollingZ0Prediction:
        p1, p2, p3 = self._validate(points)
        pts = (p1, p2, p3)
        times = tuple(p.t_s for p in pts)

        vx = self._slope_anchored_at_latest(
            tuple(p.x for p in pts), times, p3.x, p3.t_s
        )
        vy = self._slope_anchored_at_latest(
            tuple(p.y for p in pts), times, p3.y, p3.t_s
        )

        # Local vertical model about the newest point:
        #   z(t3+s) = z3 + vz3*s - 1/2*g*s^2
        # Solve vz3 from both prior observations with least squares.
        num = 0.0
        den = 0.0
        for p in (p1, p2):
            s = p.t_s - p3.t_s
            rhs = p.z - p3.z + 0.5 * self.gravity_bu_s2 * s * s
            num += s * rhs
            den += s * s
        if den <= 0.0:
            raise ValueError("observation times are degenerate")
        vz = num / den

        velocity = Velocity3D(vx, vy, vz)

        if p3.z <= 1e-12:
            dt = 0.0
        else:
            disc = vz * vz + 2.0 * self.gravity_bu_s2 * p3.z
            if disc < 0.0:
                raise ValueError("local vertical state cannot intersect Z=0")
            dt = (vz + math.sqrt(disc)) / self.gravity_bu_s2
            if not math.isfinite(dt) or dt < 0.0:
                raise ValueError("invalid future Z=0 time")

        landing = Point3D(
            p3.x + vx * dt,
            p3.y + vy * dt,
            0.0,
        )
        return RollingZ0Prediction(
            observations=pts,
            current=p3,
            velocity=velocity,
            landing=landing,
            time_to_z0_s=dt,
            contact_time_s=p3.t_s + dt,
        )


class RollingZ0Tracker:
    """Keep MAX 3 points and recompute Z=0 whenever a new point arrives."""

    def __init__(self, *, gravity_bu_s2: float | None = None) -> None:
        self.predictor = RollingThreePointZ0Predictor(gravity_bu_s2=gravity_bu_s2)
        self._points: deque[TimedPoint3D] = deque(maxlen=3)

    @property
    def points(self) -> tuple[TimedPoint3D, ...]:
        return tuple(self._points)

    @property
    def ready(self) -> bool:
        return len(self._points) == 3

    def reset(self) -> None:
        self._points.clear()

    def add(self, observation: TimedPoint3D) -> RollingZ0Prediction | None:
        if self._points and observation.t_s <= self._points[-1].t_s:
            raise ValueError("new observation time must be greater than previous time")
        self._points.append(observation)
        if len(self._points) < 3:
            return None
        return self.predictor.predict(tuple(self._points))
