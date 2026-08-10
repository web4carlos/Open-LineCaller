from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class VisionDetection:
    frame_number: int
    x: float | None
    y: float | None
    confidence: float
    status: str
    source: str
    metadata: dict[str, Any] | None = None
