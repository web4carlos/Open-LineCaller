from __future__ import annotations

from .perception_adapter import LivePerceptionAdapter
from .pipeline_models import LivePipelineResult


class NullLivePerceptionAdapter(LivePerceptionAdapter):
    """
    Honest fallback when no concrete live pipeline has been attached yet.
    """

    def process(self, *, frame_number, frame, timestamp):
        return LivePipelineResult(
            frame_number=frame_number,
            tracking_status="SEARCHING",
            confidence=0.0,
        )
