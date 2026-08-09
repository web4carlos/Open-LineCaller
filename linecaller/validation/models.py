from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

@dataclass(frozen=True)
class ValidationTruthEvent:
    event_id: str
    video: str
    frame: int
    truth: str
    notes: str = ""

    def __post_init__(self):
        value = self.truth.upper()
        if value not in ("IN", "OUT"):
            raise ValueError("truth must be IN or OUT")
        object.__setattr__(self, "truth", value)

    def to_dict(self):
        return asdict(self)

@dataclass(frozen=True)
class ValidationPrediction:
    event_id: str
    frame: int
    prediction: str
    confidence: float = 0.0
    latency_ms: float = 0.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        value = self.prediction.upper()
        if value not in ("IN", "OUT", "REVIEW", "MISSING"):
            raise ValueError("prediction must be IN, OUT, REVIEW, or MISSING")
        object.__setattr__(self, "prediction", value)

    def to_dict(self):
        return asdict(self)

@dataclass(frozen=True)
class ValidationComparison:
    event_id: str
    truth: str
    prediction: str
    correct: bool
    category: str
    confidence: float
    latency_ms: float

@dataclass(frozen=True)
class ValidationMetrics:
    total_truth_events: int
    automatic_calls: int
    correct_automatic: int
    incorrect_automatic: int
    reviews: int
    misses: int
    false_in: int
    false_out: int
    automatic_accuracy: float
    coverage: float
    review_rate: float
    miss_rate: float
    average_confidence: float
    average_latency_ms: float

    def to_dict(self):
        return asdict(self)
