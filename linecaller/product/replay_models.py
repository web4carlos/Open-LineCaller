from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ReplayState(str, Enum):
    LIVE = "LIVE"
    FREEZE = "FREEZE"
    PLAYING = "PLAYING"
    RESUME = "RESUME"


@dataclass(frozen=True)
class ReplayViewModel:
    state: ReplayState
    current_index: int
    total_frames: int
    speed: float
    decision_frame: int | None = None
    zoom_x: float | None = None
    zoom_y: float | None = None

    @property
    def active(self) -> bool:
        return self.state != ReplayState.LIVE

    @property
    def speed_label(self) -> str:
        return f"{self.speed:g}x"
