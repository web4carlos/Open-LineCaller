from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from .models import LiveDecision, LiveEvent
from .pipeline_models import LivePipelineResult
from .perception_adapter import LivePerceptionAdapter


@dataclass(frozen=True)
class LiveAdapterOutput:
    result: LivePipelineResult
    processing_ms: float
    event: LiveEvent | None


class RealTimeOfficiatingAdapter:
    def __init__(
        self,
        perception: LivePerceptionAdapter,
        *,
        minimum_call_confidence: float = 0.0,
    ):
        self.perception = perception
        self.minimum_call_confidence = float(minimum_call_confidence)

    def process_frame(
        self,
        *,
        frame_number: int,
        frame: np.ndarray,
        timestamp: float,
    ) -> LiveAdapterOutput:
        started = time.perf_counter()

        result = self.perception.process(
            frame_number=frame_number,
            frame=frame,
            timestamp=timestamp,
        )

        processing_ms = (time.perf_counter() - started) * 1000.0
        event = self._event_from_result(result, timestamp)

        return LiveAdapterOutput(
            result=result,
            processing_ms=processing_ms,
            event=event,
        )

    def _event_from_result(
        self,
        result: LivePipelineResult,
        timestamp: float,
    ) -> LiveEvent | None:
        if result.decision is None:
            return None

        try:
            decision = LiveDecision(str(result.decision).upper())
        except ValueError:
            return None

        confidence = (
            result.decision_confidence
            if result.decision_confidence is not None
            else result.confidence
        )
        confidence = max(0.0, min(1.0, float(confidence)))

        if (
            decision in (LiveDecision.IN, LiveDecision.OUT)
            and confidence < self.minimum_call_confidence
        ):
            return LiveEvent(
                frame_number=result.frame_number,
                decision=LiveDecision.REVIEW,
                confidence=confidence,
                event_timestamp=timestamp,
                metadata={
                    **result.metadata,
                    "reason": "below_minimum_call_confidence",
                    "original_decision": decision.value,
                },
            )

        return LiveEvent(
            frame_number=result.frame_number,
            decision=decision,
            confidence=confidence,
            event_timestamp=timestamp,
            metadata=dict(result.metadata),
        )
