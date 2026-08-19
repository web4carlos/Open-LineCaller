from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Sequence

from .models import CourtFrame, Point3D, TimedPoint3D, Velocity3D
from .path_field import FlightPath


class TransitionKind(str, Enum):
    NONE = "NONE"
    ABOVE_FLOOR_DIRECTION_CHANGE = "ABOVE_FLOOR_DIRECTION_CHANGE"
    FLOOR_BOUNCE = "FLOOR_BOUNCE"


@dataclass(frozen=True)
class DirectionChangeConfig:
    """Generic flight-discontinuity thresholds in Ball Units and seconds.

    These are not court- or camera-specific constants. They describe how much
    a newly observed motion must disagree with the continuation of the current
    continuous flight path before LineCaller calls it a new-flight candidate.
    """

    window_points: int = 3
    min_angle_deg: float = 24.0
    min_speed_change_ratio: float = 0.18
    min_path_residual_bu: float = 1.25
    floor_threshold_bu: float = 1.25
    max_internal_gap_s: float = 0.075
    max_transition_gap_s: float = 0.250

    def __post_init__(self) -> None:
        if self.window_points < 3:
            raise ValueError("window_points must be >= 3")
        if not 0.0 <= self.min_angle_deg <= 180.0:
            raise ValueError("min_angle_deg must be in [0,180]")
        if self.min_speed_change_ratio < 0.0:
            raise ValueError("min_speed_change_ratio must be >= 0")
        if self.min_path_residual_bu < 0.0:
            raise ValueError("min_path_residual_bu must be >= 0")
        if self.floor_threshold_bu < 0.0:
            raise ValueError("floor_threshold_bu must be >= 0")
        if self.max_internal_gap_s <= 0.0 or self.max_transition_gap_s <= 0.0:
            raise ValueError("time gaps must be positive")


@dataclass(frozen=True)
class BallisticFit:
    point: Point3D
    velocity: Velocity3D
    reference_time_s: float
    rms_residual_bu: float


@dataclass(frozen=True)
class FlightTransition:
    kind: TransitionKind
    before: tuple[TimedPoint3D, ...]
    after: tuple[TimedPoint3D, ...]
    incoming: BallisticFit
    outgoing: BallisticFit
    angle_deg: float
    speed_change_ratio: float
    old_path_residual_bu: float
    transition_gap_s: float
    old_path_crossed_z0_before_after: bool
    score: float

    @property
    def transition_start_s(self) -> float:
        return self.before[-1].t_s

    @property
    def transition_end_s(self) -> float:
        return self.after[0].t_s

    @property
    def transition_height_bu(self) -> float:
        return min(self.before[-1].z, self.after[0].z)

    @property
    def is_new_flight_candidate(self) -> bool:
        return self.kind == TransitionKind.ABOVE_FLOOR_DIRECTION_CHANGE


class TrajectoryDiscontinuityDetector:
    """Detect a new continuous-flight epoch from P(x,y,z,t) observations.

    Ball appearance is intentionally absent.  The detector compares the
    outgoing motion with the gravity-corrected continuation of the incoming
    path.  A floor bounce is classified separately from an above-floor change.
    """

    def __init__(
        self,
        config: DirectionChangeConfig | None = None,
        *,
        gravity_bu_s2: float | None = None,
    ) -> None:
        self.config = config or DirectionChangeConfig()
        self.gravity_bu_s2 = (
            CourtFrame().gravity_bu_s2
            if gravity_bu_s2 is None
            else float(gravity_bu_s2)
        )
        if not math.isfinite(self.gravity_bu_s2) or self.gravity_bu_s2 <= 0.0:
            raise ValueError("gravity_bu_s2 must be finite and > 0")

    @staticmethod
    def _linear_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
        if len(xs) != len(ys) or len(xs) < 2:
            raise ValueError("linear fit requires matching sequences with >=2 points")
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        denom = sum((x - mx) ** 2 for x in xs)
        if denom <= 1e-15:
            raise ValueError("observation times must not be identical")
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
        intercept = my - slope * mx
        return intercept, slope

    def fit_window(
        self,
        observations: Sequence[TimedPoint3D],
        *,
        reference_time_s: float,
    ) -> BallisticFit:
        if len(observations) < 3:
            raise ValueError("ballistic fit requires at least three observations")
        ordered = tuple(observations)
        for a, b in zip(ordered, ordered[1:]):
            if b.t_s <= a.t_s:
                raise ValueError("observations must have strictly increasing time")

        dt = [o.t_s - reference_time_s for o in ordered]
        ix, vx = self._linear_fit(dt, [o.x for o in ordered])
        iy, vy = self._linear_fit(dt, [o.y for o in ordered])

        # z(t)=z_ref+vz_ref*dt-1/2*g*dt^2. Move known gravity term to
        # the observed side so the remaining fit is linear in dt.
        adjusted_z = [
            o.z + 0.5 * self.gravity_bu_s2 * d * d
            for o, d in zip(ordered, dt)
        ]
        iz, vz = self._linear_fit(dt, adjusted_z)

        point = Point3D(ix, iy, max(0.0, iz))
        velocity = Velocity3D(vx, vy, vz)

        squared = []
        for o in ordered:
            d = o.t_s - reference_time_s
            px = point.x + velocity.vx * d
            py = point.y + velocity.vy * d
            pz = point.z + velocity.vz * d - 0.5 * self.gravity_bu_s2 * d * d
            squared.append((px-o.x)**2 + (py-o.y)**2 + (pz-o.z)**2)
        rms = math.sqrt(sum(squared) / len(squared))
        return BallisticFit(point, velocity, reference_time_s, rms)

    @staticmethod
    def _speed(v: Velocity3D) -> float:
        return v.speed_bu_s

    @staticmethod
    def _angle_deg(a: Velocity3D, b: Velocity3D) -> float:
        na = a.speed_bu_s
        nb = b.speed_bu_s
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        dot = a.vx*b.vx + a.vy*b.vy + a.vz*b.vz
        cosine = max(-1.0, min(1.0, dot / (na * nb)))
        return math.degrees(math.acos(cosine))

    def _windows_are_contiguous(self, observations: Sequence[TimedPoint3D]) -> bool:
        return all(
            0.0 < (b.t_s - a.t_s) <= self.config.max_internal_gap_s
            for a, b in zip(observations, observations[1:])
        )

    def evaluate(
        self,
        before: Sequence[TimedPoint3D],
        after: Sequence[TimedPoint3D],
    ) -> FlightTransition | None:
        n = self.config.window_points
        if len(before) < n or len(after) < n:
            return None
        before = tuple(before[-n:])
        after = tuple(after[:n])
        if not self._windows_are_contiguous(before) or not self._windows_are_contiguous(after):
            return None

        gap = after[0].t_s - before[-1].t_s
        if gap <= 0.0 or gap > self.config.max_transition_gap_s:
            return None

        incoming = self.fit_window(before, reference_time_s=before[-1].t_s)
        outgoing = self.fit_window(after, reference_time_s=after[0].t_s)

        dt = gap
        predicted_velocity = Velocity3D(
            incoming.velocity.vx,
            incoming.velocity.vy,
            incoming.velocity.vz - self.gravity_bu_s2 * dt,
        )
        angle = self._angle_deg(predicted_velocity, outgoing.velocity)
        speed_ref = max(predicted_velocity.speed_bu_s, 1e-9)
        speed_change = abs(outgoing.velocity.speed_bu_s - predicted_velocity.speed_bu_s) / speed_ref

        old_path = FlightPath(
            incoming.point,
            incoming.velocity,
            t0_s=incoming.reference_time_s,
            gravity_bu_s2=self.gravity_bu_s2,
        )
        residuals = []
        crossed = False
        landing = old_path.landing_point
        for observed in after:
            if observed.t_s > old_path.contact_time_s + 1e-9:
                crossed = True
                residuals.append(math.sqrt(
                    (landing.x - observed.x) ** 2
                    + (landing.y - observed.y) ** 2
                    + observed.z ** 2
                ))
            else:
                predicted = old_path.point_at_time(observed.t_s)
                residuals.append(math.sqrt(
                    (predicted.x - observed.x) ** 2
                    + (predicted.y - observed.y) ** 2
                    + (predicted.z - observed.z) ** 2
                ))
        # The first outgoing point may be exactly the physical impact point and
        # therefore still lie on the old path. The divergence becomes visible
        # in the next observations, so evaluate the whole outgoing window.
        residual = max(residuals)

        direction_or_speed = (
            angle >= self.config.min_angle_deg
            or speed_change >= self.config.min_speed_change_ratio
        )
        path_disagrees = crossed or residual >= self.config.min_path_residual_bu
        meaningful = direction_or_speed and path_disagrees

        transition_height = min(before[-1].z, after[0].z)
        floor_like = (
            transition_height <= self.config.floor_threshold_bu
            and predicted_velocity.vz < 0.0
            and outgoing.velocity.vz > 0.0
        )

        if meaningful and floor_like:
            kind = TransitionKind.FLOOR_BOUNCE
        elif meaningful:
            kind = TransitionKind.ABOVE_FLOOR_DIRECTION_CHANGE
        else:
            kind = TransitionKind.NONE

        score = (
            angle / 90.0
            + min(2.0, speed_change)
            + min(3.0, residual / max(0.25, self.config.min_path_residual_bu))
            + (0.5 if crossed else 0.0)
        )
        return FlightTransition(
            kind=kind,
            before=before,
            after=after,
            incoming=incoming,
            outgoing=outgoing,
            angle_deg=angle,
            speed_change_ratio=speed_change,
            old_path_residual_bu=residual,
            transition_gap_s=gap,
            old_path_crossed_z0_before_after=crossed,
            score=score,
        )

    def scan(self, observations: Iterable[TimedPoint3D]) -> tuple[FlightTransition, ...]:
        obs = tuple(observations)
        n = self.config.window_points
        out = []
        for split in range(n, len(obs) - n + 1):
            candidate = self.evaluate(obs[split-n:split], obs[split:split+n])
            if candidate is not None and candidate.kind != TransitionKind.NONE:
                out.append(candidate)
        return tuple(sorted(out, key=lambda c: c.score, reverse=True))
