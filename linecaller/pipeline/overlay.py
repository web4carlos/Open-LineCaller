from __future__ import annotations

import cv2

from linecaller.ball.models import BallTrackState, TrackStatus
from linecaller.bounce.models import BounceEvent
from linecaller.decision.models import DecisionResult


class OverlayRenderer:
    def draw_track(self, frame, state: BallTrackState):
        if state.x is None or state.y is None:
            return frame

        center = (int(round(state.x)), int(round(state.y)))
        cv2.circle(frame, center, 8, (255, 255, 255), 2)

        label = f"{state.status.value} {state.tracking_confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return frame

    def draw_bounce(self, frame, bounce: BounceEvent):
        center = (int(round(bounce.x)), int(round(bounce.y)))
        cv2.circle(frame, center, 14, (255, 255, 255), 3)
        cv2.putText(
            frame,
            f"BOUNCE {bounce.confidence:.2f}",
            (center[0] + 16, center[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def draw_decision(self, frame, result: DecisionResult):
        text = f"{result.decision.value}  conf={result.confidence:.2f}"
        cv2.putText(
            frame,
            text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        c = result.context
        if c.signed_distance_m is not None:
            cv2.putText(
                frame,
                f"{c.nearest_line}: {c.signed_distance_m*1000.0:.1f} mm",
                (30, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return frame
