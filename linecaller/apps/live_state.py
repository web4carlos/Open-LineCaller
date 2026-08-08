from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MatchRunState(str, Enum):
    READY = "READY"
    LIVE = "LIVE"
    STOPPED = "STOPPED"


@dataclass
class LiveUIState:
    run_state: MatchRunState = MatchRunState.READY
    calibration_status: str = "UNKNOWN"
    tracking_status: str = "SEARCHING"
    fps: float = 0.0
    latency_ms: float = 0.0
    confidence: float = 0.0
    last_call: str = "-"
    replay_active: bool = False
