from __future__ import annotations

from typing import Any
import numpy as np

from linecaller.ball.engine import BallEngine
from linecaller.ball.advanced_motion_detector import AdvancedMotionBallDetector
from linecaller.ball.tracker import BallTracker
from linecaller.ball.trajectory import TrajectoryPoint
from linecaller.bounce.engine import BounceEngine
from linecaller.decision.engine import DecisionEngine

from .perception_adapter import LivePerceptionAdapter
from .pipeline_models import LivePipelineResult


class RealArtificialVisionPerceptionAdapter(LivePerceptionAdapter):
    def __init__(
        self,
        *,
        detector=None,
        tracker=None,
        bounce_engine=None,
        decision_engine=None,
        calibration=None,
    ):
        self._detector = detector or AdvancedMotionBallDetector()
        self._tracker = tracker or BallTracker()

        self.ball_engine = BallEngine(self._detector, self._tracker)
        self.bounce_engine = bounce_engine or BounceEngine()
        self.decision_engine = decision_engine or DecisionEngine()
        self.calibration = calibration

    @property
    def detector(self):
        return self._detector

    def set_calibration(self, calibration):
        self.calibration = calibration

    def reset(self):
        reset_detector = getattr(self._detector, "reset", None)
        if callable(reset_detector):
            reset_detector()

        self._tracker = BallTracker()
        self.ball_engine = BallEngine(self._detector, self._tracker)

        reset_bounce = getattr(self.bounce_engine, "reset", None)
        if callable(reset_bounce):
            reset_bounce()

    @staticmethod
    def _status_value(state):
        status = getattr(state, "status", None)
        return str(getattr(status, "value", status or "SEARCHING")).upper()

    def process(
        self,
        *,
        frame_number: int,
        frame: np.ndarray,
        timestamp: float,
    ) -> LivePipelineResult:
        state = self.ball_engine.process(frame_number, frame)

        tracking_status = self._status_value(state)
        x = getattr(state, "x", None)
        y = getattr(state, "y", None)
        confidence = float(
            getattr(state, "tracking_confidence", 0.0) or 0.0
        )

        metadata: dict[str, Any] = {
            "source": "real_artificial_vision_core",
            "detector": type(self._detector).__name__,
            "timestamp": float(timestamp),
        }

        bounce = None
        if x is not None and y is not None:
            point = TrajectoryPoint(
                frame_number=int(frame_number),
                x=float(x),
                y=float(y),
                measured=(tracking_status == "TRACKING"),
                confidence=confidence,
            )
            bounce = self.bounce_engine.process(point)

        if bounce is None:
            return LivePipelineResult(
                frame_number=int(frame_number),
                tracking_status=tracking_status,
                confidence=confidence,
                ball_x=x,
                ball_y=y,
                bounce_detected=False,
                metadata=metadata,
            )

        metadata.update(
            {
                "bounce_frame": int(bounce.frame_number),
                "bounce_x": float(bounce.x),
                "bounce_y": float(bounce.y),
                "bounce_confidence": float(bounce.confidence),
            }
        )

        if self.calibration is None:
            metadata["reason"] = "calibration_required"

            return LivePipelineResult(
                frame_number=int(bounce.frame_number),
                tracking_status=tracking_status,
                confidence=confidence,
                ball_x=float(bounce.x),
                ball_y=float(bounce.y),
                bounce_detected=True,
                decision="REVIEW",
                decision_confidence=float(bounce.confidence),
                metadata=metadata,
            )

        decision = self.decision_engine.decide(
            bounce,
            self.calibration,
        )

        metadata["decision_explanation"] = list(
            getattr(decision, "explanation", ())
        )

        return LivePipelineResult(
            frame_number=int(bounce.frame_number),
            tracking_status=tracking_status,
            confidence=confidence,
            ball_x=float(bounce.x),
            ball_y=float(bounce.y),
            bounce_detected=True,
            decision=decision.decision.value,
            decision_confidence=float(decision.confidence),
            metadata=metadata,
        )
