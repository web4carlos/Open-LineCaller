from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class LiveDecision(str, Enum):
    IN = "IN"
    OUT = "OUT"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class LiveFramePacket:
    frame_number: int
    captured_at: float
    frame: Any


@dataclass(frozen=True)
class LiveEvent:
    frame_number: int
    decision: LiveDecision
    confidence: float
    event_timestamp: float
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class LiveEvidence:
    frame_number: int
    decision: LiveDecision
    confidence: float
    replay_requested: bool
    decision_latency_ms: float
    processing_fps: float
    duplicate_suppressed: bool = False
