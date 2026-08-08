from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .confidence import status_from_confidence
from .models import BallProposal, ProposalSource


@dataclass(frozen=True)
class AcceptedBox:
    frame_number: int
    x: float
    y: float
    width: float
    height: float


class TemporalProposalTracker:
    def __init__(
        self,
        *,
        max_history: int = 4,
        confidence: float = 0.72,
    ):
        self.history = deque(maxlen=max_history)
        self.confidence = float(confidence)

    def reset(self):
        self.history.clear()

    def observe(
        self,
        frame_number: int,
        x: float,
        y: float,
        width: float,
        height: float,
    ):
        self.history.append(
            AcceptedBox(
                frame_number=int(frame_number),
                x=float(x),
                y=float(y),
                width=float(width),
                height=float(height),
            )
        )

    def predict(self, frame_number: int) -> BallProposal | None:
        if not self.history:
            return None

        last = self.history[-1]

        if len(self.history) == 1:
            vx = vy = 0.0
        else:
            prev = self.history[-2]
            dt = max(1, last.frame_number - prev.frame_number)
            vx = (last.x - prev.x) / dt
            vy = (last.y - prev.y) / dt

        dt_future = frame_number - last.frame_number

        if dt_future <= 0:
            return None

        x = last.x + vx * dt_future
        y = last.y + vy * dt_future

        confidence = max(
            0.0,
            min(1.0, self.confidence - 0.05 * max(0, dt_future - 1)),
        )

        return BallProposal(
            frame_number=frame_number,
            x=x,
            y=y,
            width=last.width,
            height=last.height,
            confidence=confidence,
            source=ProposalSource.ENSEMBLE,
            status=status_from_confidence(confidence),
            metadata={
                "temporal_prediction": True,
                "history_size": len(self.history),
            },
        )
