from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math


@dataclass(frozen=True)
class TrackSample:
    frame: int
    raw_x: float | None
    raw_y: float | None
    tracked_x: float | None
    tracked_y: float | None
    confidence: float
    source: str


@dataclass(frozen=True)
class ContactValidationResult:
    accepted: bool
    event_frame: int
    contact_frame: int | None
    contact_x: float | None
    contact_y: float | None
    score: float
    raw_ratio: float
    reversal_strength: float
    localization_shift_px: float | None
    reason: str


class PhysicalBounceContactValidator:
    """
    CP-0026.3

    Validates whether a bounce candidate has a physically plausible contact
    point in the local tracked trajectory.

    Image coordinates:
      y increases downward.

    A ground bounce normally produces a local MAXIMUM in image y around the
    contact frame, with downward motion before contact and upward motion after.

    Validation uses:
      - local y maximum
      - pre/post vertical reversal
      - raw YOLO support near contact
      - tracker confidence
      - localization shift from original event point

    This layer is intentionally conservative.
    """

    def __init__(
        self,
        *,
        window: int = 5,
        min_raw_ratio: float = 0.35,
        min_reversal_strength: float = 1.5,
        min_score: float = 0.45,
        max_localization_shift_px: float = 90.0,
    ):
        self.window = int(window)
        self.min_raw_ratio = float(min_raw_ratio)
        self.min_reversal_strength = float(min_reversal_strength)
        self.min_score = float(min_score)
        self.max_localization_shift_px = float(max_localization_shift_px)

    @staticmethod
    def _xy(sample: TrackSample):
        x = sample.raw_x if sample.raw_x is not None else sample.tracked_x
        y = sample.raw_y if sample.raw_y is not None else sample.tracked_y
        return x, y

    @staticmethod
    def _dist(a, b):
        if a is None or b is None:
            return None
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def validate(
        self,
        *,
        event_frame: int,
        event_x: float | None,
        event_y: float | None,
        samples: Iterable[TrackSample],
    ) -> ContactValidationResult:
        local = [
            s for s in samples
            if abs(int(s.frame) - int(event_frame)) <= self.window
        ]

        usable = []
        for s in local:
            x, y = self._xy(s)
            if x is not None and y is not None:
                usable.append((s, float(x), float(y)))

        if len(usable) < 3:
            return ContactValidationResult(
                False, event_frame, None, None, None,
                0.0, 0.0, 0.0, None,
                "INSUFFICIENT_TRACK_SAMPLES",
            )

        # Contact candidate = deepest image point near event.
        contact_s, cx, cy = max(
            usable,
            key=lambda t: t[2],
        )

        before = [
            (s, x, y)
            for s, x, y in usable
            if s.frame < contact_s.frame
        ]
        after = [
            (s, x, y)
            for s, x, y in usable
            if s.frame > contact_s.frame
        ]

        if not before or not after:
            return ContactValidationResult(
                False, event_frame, contact_s.frame, cx, cy,
                0.0, 0.0, 0.0, None,
                "NO_TRAJECTORY_REVERSAL_WINDOW",
            )

        # Average recent vertical motion into and out of the candidate.
        b = before[-min(3, len(before)):]
        a = after[:min(3, len(after))]

        pre_dy = sum(
            cy - y
            for _, _, y in b
        ) / len(b)

        post_dy = sum(
            cy - y
            for _, _, y in a
        ) / len(a)

        # Both should be positive if contact is a local maximum in y.
        reversal_strength = min(pre_dy, post_dy)

        raw_count = sum(
            1 for s, _, _ in usable
            if s.raw_x is not None and s.raw_y is not None
        )
        raw_ratio = raw_count / len(usable)

        mean_conf = sum(
            max(0.0, min(1.0, float(s.confidence)))
            for s, _, _ in usable
        ) / len(usable)

        reversal_score = max(
            0.0,
            min(1.0, reversal_strength / 8.0),
        )
        raw_score = max(
            0.0,
            min(1.0, raw_ratio),
        )

        score = (
            0.50 * reversal_score
            + 0.30 * raw_score
            + 0.20 * mean_conf
        )

        shift = None
        if event_x is not None and event_y is not None:
            shift = self._dist(
                (float(event_x), float(event_y)),
                (cx, cy),
            )

        if reversal_strength < self.min_reversal_strength:
            reason = "NO_PHYSICAL_Y_REVERSAL"
            accepted = False
        elif raw_ratio < self.min_raw_ratio:
            reason = "INSUFFICIENT_RAW_DETECTION_SUPPORT"
            accepted = False
        elif shift is not None and shift > self.max_localization_shift_px:
            reason = "EVENT_CONTACT_MISMATCH"
            accepted = False
        elif score < self.min_score:
            reason = "LOW_CONTACT_SCORE"
            accepted = False
        else:
            reason = "PHYSICAL_CONTACT_ACCEPTED"
            accepted = True

        return ContactValidationResult(
            accepted=accepted,
            event_frame=int(event_frame),
            contact_frame=int(contact_s.frame),
            contact_x=float(cx),
            contact_y=float(cy),
            score=float(score),
            raw_ratio=float(raw_ratio),
            reversal_strength=float(reversal_strength),
            localization_shift_px=None if shift is None else float(shift),
            reason=reason,
        )
