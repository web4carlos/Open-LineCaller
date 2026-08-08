from __future__ import annotations

from dataclasses import dataclass
from collections import deque


@dataclass
class LiveMetricsSnapshot:
    frames: int
    avg_processing_ms: float
    processing_fps: float
    emitted_calls: int
    replay_requests: int
    duplicate_suppressed: int


class LiveMetrics:
    def __init__(self, *, window_size: int = 120):
        self.frames = 0
        self.emitted_calls = 0
        self.replay_requests = 0
        self.duplicate_suppressed = 0
        self._durations_ms = deque(maxlen=int(window_size))

    def record_frame(self, processing_ms: float):
        self.frames += 1
        self._durations_ms.append(max(0.0, float(processing_ms)))

    def record_call(self, *, replay: bool = False):
        self.emitted_calls += 1
        if replay:
            self.replay_requests += 1

    def record_duplicate(self):
        self.duplicate_suppressed += 1

    @property
    def avg_processing_ms(self) -> float:
        if not self._durations_ms:
            return 0.0
        return sum(self._durations_ms) / len(self._durations_ms)

    @property
    def processing_fps(self) -> float:
        avg = self.avg_processing_ms
        return 1000.0 / avg if avg > 0 else 0.0

    def snapshot(self) -> LiveMetricsSnapshot:
        return LiveMetricsSnapshot(
            frames=self.frames,
            avg_processing_ms=self.avg_processing_ms,
            processing_fps=self.processing_fps,
            emitted_calls=self.emitted_calls,
            replay_requests=self.replay_requests,
            duplicate_suppressed=self.duplicate_suppressed,
        )
