import numpy as np

from linecaller.live.models import (
    LiveDecision,
    LiveEvent,
)
from linecaller.live.pipeline_models import (
    LivePipelineResult,
)
from linecaller.live.realtime_adapter import (
    LiveAdapterOutput,
)
from linecaller.validation.offline_pipeline import (
    LivePipelineOfflineAdapter,
)


class FakeLivePipeline:
    def process_frame(
        self,
        *,
        frame_number,
        frame,
        timestamp,
    ):
        return LiveAdapterOutput(
            result=LivePipelineResult(
                frame_number=frame_number,
            ),
            processing_ms=12.0,
            event=LiveEvent(
                frame_number=frame_number,
                decision=LiveDecision.OUT,
                confidence=.99,
                event_timestamp=timestamp,
            ),
        )


def test_offline_adapter_converts_event():
    adapter = LivePipelineOfflineAdapter(
        FakeLivePipeline()
    )

    prediction = adapter.process(
        frame_number=5,
        timestamp=.1,
        frame=np.zeros(
            (10,10,3),
            dtype=np.uint8,
        ),
    )

    assert prediction is not None
    assert prediction.frame == 5
    assert prediction.prediction == "OUT"
    assert prediction.confidence == .99
    assert prediction.latency_ms == 12.0
