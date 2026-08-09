from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OfflineVideoInfo:
    path: str
    frame_count: int
    source_fps: float
    width: int
    height: int


@dataclass(frozen=True)
class OfflineRunStats:
    frames_processed: int
    wall_seconds: float
    effective_processing_fps: float


@dataclass(frozen=True)
class OfflineValidationResult:
    video_info: OfflineVideoInfo
    run_stats: OfflineRunStats
    predictions: tuple
    comparisons: tuple
    metrics: object
