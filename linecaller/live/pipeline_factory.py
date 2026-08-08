from __future__ import annotations

from .null_perception import NullLivePerceptionAdapter
from .realtime_adapter import RealTimeOfficiatingAdapter


def create_live_pipeline_adapter():
    """
    Stable factory used by the UI.

    Future capability packs can replace the concrete perception implementation
    here without changing the Live Match window.
    """
    return RealTimeOfficiatingAdapter(
        NullLivePerceptionAdapter(),
        minimum_call_confidence=0.98,
    )
