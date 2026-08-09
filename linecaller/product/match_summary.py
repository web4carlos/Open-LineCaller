from __future__ import annotations

import time
import uuid

from .summary_models import MatchSummary


class MatchSummaryBuilder:
    def __init__(self):
        self.started_at = time.time()
        self.calls_in = 0
        self.calls_out = 0
        self.calls_review = 0
        self.tracking_losses = 0

        self._fps_samples = []
        self._latency_samples = []
        self._confidence_samples = []

    def record_call(self, decision: str):
        decision = str(decision).upper()

        if decision == "IN":
            self.calls_in += 1
        elif decision == "OUT":
            self.calls_out += 1
        elif decision == "REVIEW":
            self.calls_review += 1

    def record_runtime(
        self,
        *,
        fps: float | None = None,
        latency_ms: float | None = None,
        confidence: float | None = None,
        tracking_status: str | None = None,
    ):
        if fps is not None and fps >= 0:
            self._fps_samples.append(float(fps))

        if latency_ms is not None and latency_ms >= 0:
            self._latency_samples.append(float(latency_ms))

        if confidence is not None and 0.0 <= confidence <= 1.0:
            self._confidence_samples.append(float(confidence))

        if str(tracking_status or "").upper() == "LOST":
            self.tracking_losses += 1

    @staticmethod
    def _avg(values):
        return sum(values) / len(values) if values else 0.0

    def build(
        self,
        *,
        camera_name: str,
        calibration_mode: str,
        replay_count: int,
        ended_at: float | None = None,
        match_id: str | None = None,
    ) -> MatchSummary:
        ended_at = time.time() if ended_at is None else float(ended_at)

        total = (
            self.calls_in
            + self.calls_out
            + self.calls_review
        )

        return MatchSummary(
            match_id=match_id or str(uuid.uuid4()),
            started_at=float(self.started_at),
            ended_at=ended_at,
            duration_seconds=max(0.0, ended_at - self.started_at),
            camera_name=str(camera_name),
            calibration_mode=str(calibration_mode),
            calls_total=total,
            calls_in=self.calls_in,
            calls_out=self.calls_out,
            calls_review=self.calls_review,
            average_fps=self._avg(self._fps_samples),
            average_latency_ms=self._avg(self._latency_samples),
            average_confidence=self._avg(self._confidence_samples),
            tracking_losses=self.tracking_losses,
            replay_count=int(replay_count),
        )
