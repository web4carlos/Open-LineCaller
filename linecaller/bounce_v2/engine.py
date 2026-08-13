from __future__ import annotations

from collections import deque
from math import acos, hypot, pi
from statistics import mean

from .models import BounceEvent, MotionSample


class BounceEngine:
    """
    CP-0026.2 Curvature Bounce Engine.

    Image coordinates:
      y grows downward.

    A bounce candidate should combine:
      - downward motion before center,
      - upward motion after center,
      - center near local maximum image-y,
      - strong local trajectory curvature,
      - reasonable horizontal continuity,
      - enough RAW/high-confidence support.

    Curvature here is represented as normalized turning angle:
      0.0 = straight trajectory
      1.0 = 180-degree reversal
    """

    def __init__(
        self,
        *,
        window: int = 3,
        min_pre_speed: float = 1.5,
        min_post_speed: float = 1.5,
        refractory_frames: int = 10,
        min_confidence: float = 0.28,
        max_horizontal_jump: float = 65.0,
        peak_tolerance_px: float = 8.0,
        min_raw_ratio: float = 0.35,
        min_curvature: float = 0.16,
        min_bounce_score: float = 0.38,
    ):
        self.window = max(1, int(window))
        self.min_pre_speed = float(min_pre_speed)
        self.min_post_speed = float(min_post_speed)
        self.refractory_frames = int(refractory_frames)
        self.min_confidence = float(min_confidence)
        self.max_horizontal_jump = float(max_horizontal_jump)
        self.peak_tolerance_px = float(peak_tolerance_px)
        self.min_raw_ratio = float(min_raw_ratio)
        self.min_curvature = float(min_curvature)
        self.min_bounce_score = float(min_bounce_score)

        self.samples = deque(
            maxlen=max(18, self.window * 4 + 7)
        )
        self.last_bounce_frame = -10000

    @staticmethod
    def _xy(sample: MotionSample):
        if sample.raw_x is not None and sample.raw_y is not None:
            return float(sample.raw_x), float(sample.raw_y), True

        if sample.tracked_x is not None and sample.tracked_y is not None:
            return float(sample.tracked_x), float(sample.tracked_y), False

        return None, None, False

    def _velocity(self, a: MotionSample, b: MotionSample):
        ax, ay, _ = self._xy(a)
        bx, by, _ = self._xy(b)

        if None in (ax, ay, bx, by):
            return None

        dt = b.frame - a.frame
        if dt <= 0:
            return None

        return (bx - ax) / dt, (by - ay) / dt

    def _mean_velocity(self, seq):
        vx_values = []
        vy_values = []

        for a, b in zip(seq, seq[1:]):
            v = self._velocity(a, b)
            if v is None:
                continue
            vx, vy = v
            vx_values.append(vx)
            vy_values.append(vy)

        if not vy_values:
            return None

        return mean(vx_values), mean(vy_values)

    def _local_curvature(self, prev_s, center_s, next_s):
        px, py, _ = self._xy(prev_s)
        cx, cy, _ = self._xy(center_s)
        nx, ny, _ = self._xy(next_s)

        if None in (px, py, cx, cy, nx, ny):
            return 0.0

        # Incoming vector points toward center.
        v1x = cx - px
        v1y = cy - py

        # Outgoing vector points away from center.
        v2x = nx - cx
        v2y = ny - cy

        mag1 = hypot(v1x, v1y)
        mag2 = hypot(v2x, v2y)

        if mag1 < 1e-6 or mag2 < 1e-6:
            return 0.0

        cos_theta = (
            v1x * v2x + v1y * v2y
        ) / (mag1 * mag2)

        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle = acos(cos_theta)

        # Straight movement => angle~0 => curvature 0.
        # Strong turn => larger angle.
        return max(0.0, min(1.0, angle / pi))

    def _smoothed_curvature(self, seq, center_idx):
        values = []

        lo = max(1, center_idx - 1)
        hi = min(len(seq) - 2, center_idx + 1)

        for i in range(lo, hi + 1):
            values.append(
                self._local_curvature(
                    seq[i - 1],
                    seq[i],
                    seq[i + 1],
                )
            )

        return mean(values) if values else 0.0

    def _is_local_y_peak(self, seq, center_idx):
        _, cy, _ = self._xy(seq[center_idx])

        if cy is None:
            return False, 0.0

        ys = []
        for s in seq:
            _, y, _ = self._xy(s)
            if y is not None:
                ys.append(y)

        if not ys:
            return False, 0.0

        local_max = max(ys)
        delta = local_max - cy

        valid = delta <= self.peak_tolerance_px

        score = max(
            0.0,
            1.0 - delta / max(1.0, self.peak_tolerance_px),
        )

        return valid, score

    def _horizontal_continuity(self, pre_vx, post_vx):
        jump = abs(post_vx - pre_vx)

        if jump > self.max_horizontal_jump:
            return False, 0.0

        score = max(
            0.0,
            1.0 - jump / max(1.0, self.max_horizontal_jump),
        )

        return True, score

    def update(self, sample: MotionSample):
        self.samples.append(sample)

        needed = self.window * 2 + 1

        if len(self.samples) < needed:
            return None

        seq = list(self.samples)[-needed:]
        center_idx = self.window

        before = seq[: center_idx + 1]
        after = seq[center_idx:]

        pre = self._mean_velocity(before)
        post = self._mean_velocity(after)

        if pre is None or post is None:
            return None

        pre_vx, pre_vy = pre
        post_vx, post_vy = post

        # Vertical reversal remains mandatory.
        if pre_vy < self.min_pre_speed:
            return None

        if post_vy > -self.min_post_speed:
            return None

        center = seq[center_idx]

        if (
            center.frame - self.last_bounce_frame
            < self.refractory_frames
        ):
            return None

        peak_ok, peak_score = self._is_local_y_peak(
            seq,
            center_idx,
        )

        if not peak_ok:
            return None

        horizontal_ok, horizontal_score = (
            self._horizontal_continuity(
                pre_vx,
                post_vx,
            )
        )

        if not horizontal_ok:
            return None

        curvature = self._smoothed_curvature(
            seq,
            center_idx,
        )

        if curvature < self.min_curvature:
            return None

        x, y, center_is_raw = self._xy(center)

        if x is None or y is None:
            return None

        confidences = [
            max(0.0, min(1.0, float(s.confidence)))
            for s in seq
        ]
        mean_conf = mean(confidences)

        raw_count = sum(
            1
            for s in seq
            if s.raw_x is not None and s.raw_y is not None
        )
        raw_ratio = raw_count / len(seq)

        if raw_ratio < self.min_raw_ratio:
            return None

        reversal_strength = min(
            1.0,
            (
                min(abs(pre_vy), 30.0)
                + min(abs(post_vy), 30.0)
            ) / 24.0,
        )

        # Main score for CP-0026.2.
        bounce_score = (
            0.30 * curvature
            + 0.25 * reversal_strength
            + 0.15 * peak_score
            + 0.15 * mean_conf
            + 0.10 * raw_ratio
            + 0.05 * horizontal_score
        )

        confidence = (
            0.55 * bounce_score
            + 0.25 * mean_conf
            + 0.20 * raw_ratio
        )

        if confidence < self.min_confidence:
            return None

        if bounce_score < self.min_bounce_score:
            return None

        self.last_bounce_frame = center.frame

        return BounceEvent(
            frame=center.frame,
            x=float(x),
            y=float(y),
            confidence=float(confidence),
            pre_velocity_y=float(pre_vy),
            post_velocity_y=float(post_vy),
            source="RAW" if center_is_raw else "TRACKED",
            curvature=float(curvature),
            bounce_score=float(bounce_score),
        )
