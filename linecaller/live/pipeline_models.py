from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LivePipelineResult:
    frame_number: int
    tracking_status: str = "SEARCHING"
    confidence: float = 0.0
    ball_x: float | None = None
    ball_y: float | None = None
    bounce_detected: bool = False
    decision: str | None = None
    decision_confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
