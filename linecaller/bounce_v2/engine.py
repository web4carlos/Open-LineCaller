from __future__ import annotations
from collections import deque

from .models import BounceEvent, MotionSample


class BounceEngine:
    """
    Image coordinates:
        y grows downward.

    A typical visible bounce therefore looks like:
        before bounce: vy > 0  (ball moving downward)
        after bounce:  vy < 0  (ball moving upward)

    This engine intentionally uses raw observations whenever available.
    Tracked/predicted positions are allowed for continuity, but predicted-only
    samples reduce confidence.
    """

    def __init__(
        self,
        *,
        window: int = 3,
        min_pre_speed: float = 1.5,
        min_post_speed: float = 1.5,
        refractory_frames: int = 8,
        min_confidence: float = 0.10,
    ):
        self.window = max(1, int(window))
        self.min_pre_speed = float(min_pre_speed)
        self.min_post_speed = float(min_post_speed)
        self.refractory_frames = int(refractory_frames)
        self.min_confidence = float(min_confidence)

        self.samples = deque(maxlen=max(12, self.window * 4 + 4))
        self.last_bounce_frame = -10_000

    @staticmethod
    def _xy(sample: MotionSample):
        if sample.raw_x is not None and sample.raw_y is not None:
            return sample.raw_x, sample.raw_y, True

        if sample.tracked_x is not None and sample.tracked_y is not None:
            return sample.tracked_x, sample.tracked_y, False

        return None, None, False

    def _velocity_y(self, a: MotionSample, b: MotionSample):
        ax, ay, _ = self._xy(a)
        bx, by, _ = self._xy(b)

        if ay is None or by is None:
            return None

        dt = b.frame - a.frame
        if dt <= 0:
            return None

        return (by - ay) / dt

    def _mean_velocity(self, seq):
        values = []
        for a, b in zip(seq, seq[1:]):
            vy = self._velocity_y(a, b)
            if vy is not None:
                values.append(vy)

        if not values:
            return None

        return sum(values) / len(values)

    def update(self, sample: MotionSample):
        self.samples.append(sample)

        needed = self.window * 2 + 1
        if len(self.samples) < needed:
            return None

        current = list(self.samples)[-needed:]
        center_idx = self.window

        before = current[: center_idx + 1]
        after = current[center_idx:]

        pre_vy = self._mean_velocity(before)
        post_vy = self._mean_velocity(after)

        if pre_vy is None or post_vy is None:
            return None

        if pre_vy < self.min_pre_speed:
            return None

        if post_vy > -self.min_post_speed:
            return None

        center = current[center_idx]

        if center.frame - self.last_bounce_frame < self.refractory_frames:
            return None

        x, y, raw = self._xy(center)
        if x is None or y is None:
            return None

        confidences = [
            max(0.0, min(1.0, float(s.confidence)))
            for s in current
        ]

        mean_conf = sum(confidences) / len(confidences)

        raw_ratio = sum(
            1
            for s in current
            if s.raw_x is not None and s.raw_y is not None
        ) / len(current)

        reversal_strength = min(
            1.0,
            (
                min(abs(pre_vy), 25.0)
                + min(abs(post_vy), 25.0)
            ) / 20.0,
        )

        confidence = (
            0.45 * mean_conf
            + 0.35 * raw_ratio
            + 0.20 * reversal_strength
        )

        if confidence < self.min_confidence:
            return None

        self.last_bounce_frame = center.frame

        return BounceEvent(
            frame=center.frame,
            x=float(x),
            y=float(y),
            confidence=float(confidence),
            pre_velocity_y=float(pre_vy),
            post_velocity_y=float(post_vy),
            source="RAW" if raw else "TRACKED",
        )
