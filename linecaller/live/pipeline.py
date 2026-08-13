from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from linecaller.bounce_v2 import BounceEngine, MotionSample
from linecaller.tracking import BallKalmanTracker, DetectionPoint


@dataclass(frozen=True)
class LiveFrameResult:
    frame: int
    raw_x: Optional[float]
    raw_y: Optional[float]
    tracked_x: Optional[float]
    tracked_y: Optional[float]
    confidence: float
    track_source: str
    bounce: bool
    bounce_x: Optional[float]
    bounce_y: Optional[float]
    bounce_confidence: float


class LiveRefereePipeline:
    """
    Lightweight live coordinator.

    Input:
        best detector observation for one frame

    Output:
        tracked ball position + optional bounce event
    """

    def __init__(
        self,
        *,
        max_gap: int = 4,
        min_detection_confidence: float = 0.10,
        bounce_window: int = 3,
    ):
        self.tracker = BallKalmanTracker(
            max_gap=max_gap,
            min_detection_confidence=min_detection_confidence,
        )

        self.bounce_engine = BounceEngine(
            window=bounce_window,
            min_pre_speed=1.5,
            min_post_speed=1.5,
            refractory_frames=10,
            min_confidence=0.28,
            max_horizontal_jump=65.0,
            peak_tolerance_px=8.0,
            min_raw_ratio=0.35,
            min_curvature=0.16,
            min_bounce_score=0.38,
        )

    def reset(self):
        self.tracker.reset()

        # Recreate bounce engine so its history/refractory state is clean.
        self.bounce_engine = BounceEngine(
            window=3,
            min_pre_speed=1.5,
            min_post_speed=1.5,
            refractory_frames=10,
            min_confidence=0.28,
            max_horizontal_jump=65.0,
            peak_tolerance_px=8.0,
            min_raw_ratio=0.35,
            min_curvature=0.16,
            min_bounce_score=0.38,
        )

    def update(
        self,
        frame_number: int,
        detection: DetectionPoint | None,
    ) -> LiveFrameResult:
        track = self.tracker.update(
            frame_number,
            detection,
        )

        motion = MotionSample(
            frame=frame_number,
            raw_x=track.raw_x,
            raw_y=track.raw_y,
            tracked_x=track.tracked_x,
            tracked_y=track.tracked_y,
            confidence=track.confidence,
            source=track.source,
        )

        bounce = self.bounce_engine.update(motion)

        return LiveFrameResult(
            frame=frame_number,
            raw_x=track.raw_x,
            raw_y=track.raw_y,
            tracked_x=track.tracked_x,
            tracked_y=track.tracked_y,
            confidence=track.confidence,
            track_source=track.source,
            bounce=bounce is not None,
            bounce_x=None if bounce is None else bounce.x,
            bounce_y=None if bounce is None else bounce.y,
            bounce_confidence=0.0 if bounce is None else bounce.confidence,
        )
