from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AccelerationMetrics:
    manual_baseline_seconds_per_frame: float = 3.0
    assisted_seconds_per_frame: float = 0.8

    proposals_generated: int = 0
    confirmed_frames_skipped: int = 0
    accepted_via_assist: int = 0
    rejected_via_assist: int = 0

    @property
    def estimated_seconds_saved(self) -> float:
        saved_per_accept = max(
            0.0,
            self.manual_baseline_seconds_per_frame
            - self.assisted_seconds_per_frame,
        )
        return self.accepted_via_assist * saved_per_accept
