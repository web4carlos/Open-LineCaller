from __future__ import annotations

from linecaller.live.models import LiveDecision, LiveEvidence
from .live_state import LiveUIState, MatchRunState


class LiveMatchController:
    def __init__(self):
        self.state = LiveUIState()

    def set_ready(self, *, calibration_valid: bool):
        self.state.run_state = MatchRunState.READY
        self.state.calibration_status = "VALID" if calibration_valid else "REQUIRED"

    def start(self):
        self.state.run_state = MatchRunState.LIVE
        self.state.replay_active = False

    def stop(self):
        self.state.run_state = MatchRunState.STOPPED
        self.state.tracking_status = "SEARCHING"

    def update_runtime(
        self,
        *,
        fps: float | None = None,
        latency_ms: float | None = None,
        tracking_status: str | None = None,
        confidence: float | None = None,
    ):
        if fps is not None:
            self.state.fps = float(fps)
        if latency_ms is not None:
            self.state.latency_ms = float(latency_ms)
        if tracking_status is not None:
            self.state.tracking_status = str(tracking_status)
        if confidence is not None:
            self.state.confidence = max(0.0, min(1.0, float(confidence)))

    def handle_evidence(self, evidence: LiveEvidence):
        if evidence.duplicate_suppressed:
            return self.state

        self.state.latency_ms = evidence.decision_latency_ms
        self.state.confidence = evidence.confidence

        if evidence.decision == LiveDecision.REVIEW:
            self.state.last_call = "REVIEW"
            self.state.replay_active = True
        else:
            self.state.last_call = evidence.decision.value
            self.state.replay_active = False

        return self.state
