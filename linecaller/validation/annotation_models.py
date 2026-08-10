from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any

class AnnotationLabel(str, Enum):
    IN = "IN"
    OUT = "OUT"
    SKIP = "SKIP"

@dataclass(frozen=True)
class AnnotationCandidate:
    frame: int
    confidence: float = 0.0
    x: float | None = None
    y: float | None = None
    suggested_decision: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self):
        return asdict(self)
