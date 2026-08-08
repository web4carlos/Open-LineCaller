from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class EventMatch:
    expected_frame: int
    detected_frame: int
    frame_error: int
    expected_decision: str | None
    detected_decision: str | None
    decision_match: bool | None


@dataclass(frozen=True)
class ClipBenchmarkResult:
    clip_id: str
    expected_bounces: int
    detected_bounces: int
    matched_bounces: int
    false_positive_bounces: int
    missed_bounces: int
    precision: float
    recall: float
    median_frame_error: float | None
    decision_agreement: float | None
    calibration_valid: bool
    matches: tuple[EventMatch, ...]


@dataclass(frozen=True)
class BenchmarkReport:
    clips: int
    expected_bounces: int
    detected_bounces: int
    matched_bounces: int
    false_positive_bounces: int
    missed_bounces: int
    precision: float
    recall: float
    median_frame_error: float | None
    decision_agreement: float | None
    calibration_valid_rate: float
    clip_results: tuple[ClipBenchmarkResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
