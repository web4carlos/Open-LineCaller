from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class MatchSummary:
    match_id: str
    started_at: float
    ended_at: float
    duration_seconds: float

    camera_name: str
    calibration_mode: str

    calls_total: int
    calls_in: int
    calls_out: int
    calls_review: int

    average_fps: float
    average_latency_ms: float
    average_confidence: float

    tracking_losses: int
    replay_count: int

    software_version: str = "CP-0018"

    def to_dict(self) -> dict:
        return asdict(self)
