from __future__ import annotations

import time

from .cooldown import DecisionCooldown
from .metrics import LiveMetrics
from .models import (
    LiveDecision,
    LiveEvent,
    LiveEvidence,
    LiveFramePacket,
)
from .replay import ReplayBuffer


class LiveOfficiatingEngine:
    def __init__(
        self,
        *,
        replay_buffer=None,
        cooldown=None,
        metrics=None,
        replay_pre_frames: int = 60,
        replay_post_frames: int = 15,
    ):
        self.replay_buffer = replay_buffer or ReplayBuffer()
        self.cooldown = cooldown or DecisionCooldown()
        self.metrics = metrics or LiveMetrics()
        self.replay_pre_frames = int(replay_pre_frames)
        self.replay_post_frames = int(replay_post_frames)
        self.last_replay = ()

    def ingest_frame(
        self,
        packet: LiveFramePacket,
        *,
        processing_ms: float = 0.0,
    ) -> None:
        self.replay_buffer.append(packet)
        self.metrics.record_frame(processing_ms)

    def handle_event(
        self,
        event: LiveEvent,
        *,
        now: float | None = None,
    ) -> LiveEvidence:
        now = time.perf_counter() if now is None else float(now)

        latency_ms = max(
            0.0,
            (now - float(event.event_timestamp)) * 1000.0,
        )

        if not self.cooldown.should_emit(event.frame_number):
            self.metrics.record_duplicate()

            return LiveEvidence(
                frame_number=event.frame_number,
                decision=event.decision,
                confidence=event.confidence,
                replay_requested=False,
                decision_latency_ms=latency_ms,
                processing_fps=self.metrics.processing_fps,
                duplicate_suppressed=True,
            )

        replay = event.decision == LiveDecision.REVIEW

        if replay:
            self.last_replay = self.replay_buffer.snapshot(
                event_frame=event.frame_number,
                pre_frames=self.replay_pre_frames,
                post_frames=self.replay_post_frames,
            )
        else:
            self.last_replay = ()

        self.metrics.record_call(replay=replay)

        return LiveEvidence(
            frame_number=event.frame_number,
            decision=event.decision,
            confidence=max(0.0, min(1.0, float(event.confidence))),
            replay_requested=replay,
            decision_latency_ms=latency_ms,
            processing_fps=self.metrics.processing_fps,
            duplicate_suppressed=False,
        )
