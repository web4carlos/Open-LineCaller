from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable


@dataclass(frozen=True)
class TrackSample:
    frame: int
    x: float | None
    y: float | None
    confidence: float
    source: str


@dataclass(frozen=True)
class GateDecision:
    frame: int
    accepted: bool
    reason: str
    distance_px: float | None
    expected_x: float | None
    expected_y: float | None
    observed_x: float | None
    observed_y: float | None


class RealBallContinuityGate:
    """
    CP-0035.5 diagnostic gate.

    Purpose:
      protect ball identity before Bounce/Geometry by rejecting track points
      that are spatially implausible relative to recent accepted motion.

    This is intentionally image-space first. It is the bridge from the
    existing real ball_track.csv into the future DCF model.

    It does NOT replace YOLO/Kalman yet.
    """

    def __init__(
        self,
        *,
        base_radius_px: float = 55.0,
        velocity_allowance: float = 2.5,
        confidence_floor: float = 0.10,
        max_gap_frames: int = 5,
    ):
        self.base_radius_px = float(base_radius_px)
        self.velocity_allowance = float(velocity_allowance)
        self.confidence_floor = float(confidence_floor)
        self.max_gap_frames = int(max_gap_frames)

        self._accepted: list[TrackSample] = []

    def _predict(self, frame: int) -> tuple[float, float] | None:
        pts = [p for p in self._accepted if p.x is not None and p.y is not None]
        if not pts:
            return None

        last = pts[-1]

        if len(pts) < 2:
            return float(last.x), float(last.y)

        prev = pts[-2]
        dt = max(1, last.frame - prev.frame)

        vx = (float(last.x) - float(prev.x)) / dt
        vy = (float(last.y) - float(prev.y)) / dt

        ahead = max(1, frame - last.frame)

        return (
            float(last.x) + vx * ahead,
            float(last.y) + vy * ahead,
        )

    def evaluate(self, sample: TrackSample) -> GateDecision:
        if sample.x is None or sample.y is None:
            return GateDecision(
                frame=sample.frame,
                accepted=False,
                reason="NO_POSITION",
                distance_px=None,
                expected_x=None,
                expected_y=None,
                observed_x=sample.x,
                observed_y=sample.y,
            )

        if sample.confidence < self.confidence_floor and sample.source != "KALMAN-PREDICT":
            return GateDecision(
                frame=sample.frame,
                accepted=False,
                reason="LOW_CONFIDENCE",
                distance_px=None,
                expected_x=None,
                expected_y=None,
                observed_x=sample.x,
                observed_y=sample.y,
            )

        prediction = self._predict(sample.frame)

        if prediction is None:
            self._accepted.append(sample)
            return GateDecision(
                frame=sample.frame,
                accepted=True,
                reason="INITIAL_LOCK",
                distance_px=0.0,
                expected_x=sample.x,
                expected_y=sample.y,
                observed_x=sample.x,
                observed_y=sample.y,
            )

        ex, ey = prediction
        distance = hypot(float(sample.x) - ex, float(sample.y) - ey)

        gap = 1
        if self._accepted:
            gap = max(1, sample.frame - self._accepted[-1].frame)

        radius = self.base_radius_px * (
            1.0 + self.velocity_allowance * max(0, gap - 1)
        )

        if gap > self.max_gap_frames:
            radius *= 1.8

        accepted = distance <= radius
        reason = "CONTINUITY_OK" if accepted else "IDENTITY_JUMP_REJECTED"

        if accepted:
            self._accepted.append(sample)
            self._accepted = self._accepted[-8:]

        return GateDecision(
            frame=sample.frame,
            accepted=accepted,
            reason=reason,
            distance_px=distance,
            expected_x=ex,
            expected_y=ey,
            observed_x=sample.x,
            observed_y=sample.y,
        )
