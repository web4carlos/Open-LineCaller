from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2

from linecaller.ball.engine import BallEngine
from linecaller.ball.motion_detector import MotionBallDetector
from linecaller.ball.tracker import BallTracker
from linecaller.ball.trajectory import TrajectoryPoint
from linecaller.bounce.engine import BounceEngine
from linecaller.calibration.profile import CalibrationProfile
from linecaller.decision.engine import DecisionEngine
from linecaller.pipeline.event_store import DecisionEventStore
from linecaller.pipeline.overlay import OverlayRenderer
from linecaller.pipeline.video_source import VideoSource


@dataclass(frozen=True)
class PipelineSummary:
    frames: int
    tracked: int
    predicted: int
    lost: int
    bounces: int
    decisions_in: int
    decisions_out: int
    decisions_review: int


class IntegratedVideoPipeline:
    def __init__(
        self,
        *,
        detector=None,
        tracker=None,
        bounce_engine=None,
        decision_engine=None,
        overlay=None,
    ):
        self.ball_engine = BallEngine(
            detector or MotionBallDetector(),
            tracker or BallTracker(),
        )
        self.bounce_engine = bounce_engine or BounceEngine()
        self.decision_engine = decision_engine or DecisionEngine()
        self.overlay = overlay or OverlayRenderer()

    def run(
        self,
        *,
        video_path: str | Path,
        calibration_path: str | Path,
        output_video_path: str | Path,
        events_path: str | Path,
    ) -> PipelineSummary:
        source = VideoSource(video_path)
        source.open()

        calibration = CalibrationProfile.load(calibration_path)

        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            source.fps,
            (source.width, source.height),
        )

        if not writer.isOpened():
            source.close()
            raise RuntimeError(f"Cannot open output video: {output_video_path}")

        store = DecisionEventStore(events_path)

        frames = tracked = predicted = lost = bounces = 0
        decisions = {"IN": 0, "OUT": 0, "REVIEW": 0}
        last_bounce = None
        last_decision = None
        decision_hold_frames = 45
        hold_until = -1

        try:
            for frame_number, frame in source.frames():
                frames += 1

                state = self.ball_engine.process(frame_number, frame)

                if state.status.value == "TRACKING":
                    tracked += 1
                elif state.status.value == "PREDICTED":
                    predicted += 1
                else:
                    lost += 1

                if state.x is not None and state.y is not None:
                    point = TrajectoryPoint(
                        frame_number=frame_number,
                        x=state.x,
                        y=state.y,
                        measured=(state.status.value == "TRACKING"),
                        confidence=state.tracking_confidence,
                    )

                    bounce = self.bounce_engine.process(point)

                    if bounce is not None:
                        bounces += 1
                        last_bounce = bounce
                        result = self.decision_engine.decide(
                            bounce,
                            calibration,
                        )
                        last_decision = result
                        decisions[result.decision.value] += 1
                        store.write(result)
                        hold_until = frame_number + decision_hold_frames

                display = frame.copy()
                self.overlay.draw_track(display, state)

                if last_bounce is not None and frame_number <= hold_until:
                    self.overlay.draw_bounce(display, last_bounce)

                if last_decision is not None and frame_number <= hold_until:
                    self.overlay.draw_decision(display, last_decision)

                writer.write(display)

        finally:
            store.close()
            writer.release()
            source.close()

        return PipelineSummary(
            frames=frames,
            tracked=tracked,
            predicted=predicted,
            lost=lost,
            bounces=bounces,
            decisions_in=decisions["IN"],
            decisions_out=decisions["OUT"],
            decisions_review=decisions["REVIEW"],
        )
