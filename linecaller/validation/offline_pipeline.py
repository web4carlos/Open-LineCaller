from __future__ import annotations

from abc import ABC, abstractmethod

from linecaller.live.pipeline_factory import (
    create_live_pipeline_adapter,
)
from linecaller.validation.models import (
    ValidationPrediction,
)


class OfflinePipelineAdapter(ABC):
    @abstractmethod
    def process(
        self,
        *,
        frame_number: int,
        timestamp: float,
        frame,
    ) -> ValidationPrediction | None:
        raise NotImplementedError


class LivePipelineOfflineAdapter(
    OfflinePipelineAdapter
):
    """
    Reuses the same stable pipeline boundary used by live mode,
    but feeds it frames from a recorded file.
    """

    def __init__(self, pipeline=None):
        self.pipeline = (
            pipeline
            or create_live_pipeline_adapter()
        )

    def process(
        self,
        *,
        frame_number,
        timestamp,
        frame,
    ):
        output = self.pipeline.process_frame(
            frame_number=frame_number,
            frame=frame,
            timestamp=timestamp,
        )

        event = output.event

        if event is None:
            return None

        return ValidationPrediction(
            event_id=f"prediction-{frame_number:08d}",
            frame=frame_number,
            prediction=event.decision.value,
            confidence=event.confidence,
            latency_ms=output.processing_ms,
            metadata=event.metadata,
        )
