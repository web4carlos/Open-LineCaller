from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProposalSource(str, Enum):
    MOCK = "MOCK"
    MOTION = "MOTION"
    YOLO = "YOLO"
    ONNX = "ONNX"
    TENSORRT = "TENSORRT"
    ENSEMBLE = "ENSEMBLE"
    UNKNOWN = "UNKNOWN"


class ProposalStatus(str, Enum):
    AUTO = "AUTO"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


@dataclass(frozen=True)
class BallProposal:
    frame_number: int
    x: float
    y: float
    width: float
    height: float
    confidence: float
    source: ProposalSource = ProposalSource.UNKNOWN
    status: ProposalStatus = ProposalStatus.REVIEW
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def normalized(self) -> "BallProposal":
        confidence = max(0.0, min(1.0, float(self.confidence)))
        return BallProposal(
            frame_number=int(self.frame_number),
            x=float(self.x),
            y=float(self.y),
            width=float(self.width),
            height=float(self.height),
            confidence=confidence,
            source=self.source,
            status=self.status,
            metadata=dict(self.metadata),
        )

    def validate(self) -> tuple[str, ...]:
        errors = []
        if self.frame_number < 0:
            errors.append("frame_number must be >= 0")
        if self.width <= 0:
            errors.append("width must be > 0")
        if self.height <= 0:
            errors.append("height must be > 0")
        if not (0.0 <= self.confidence <= 1.0):
            errors.append("confidence must be within [0,1]")
        return tuple(errors)


@dataclass(frozen=True)
class ProposalResult:
    frame_number: int
    proposals: tuple[BallProposal, ...]
    engine_name: str
    latency_ms: float = 0.0
