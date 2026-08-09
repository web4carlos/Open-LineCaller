from __future__ import annotations

from dataclasses import dataclass

from .replay_models import ReplayState, ReplayViewModel


@dataclass
class ReplayMetrics:
    replay_count: int = 0
    total_frames_shown: int = 0

    @property
    def average_frames_per_replay(self) -> float:
        if self.replay_count == 0:
            return 0.0
        return self.total_frames_shown / self.replay_count


class ReplayPlaybackController:
    def __init__(self, *, speed: float = 0.25):
        self._frames = ()
        self._index = 0
        self._decision_frame = None
        self._zoom_x = None
        self._zoom_y = None
        self._state = ReplayState.LIVE
        self._speed = 0.25
        self.metrics = ReplayMetrics()
        self.set_speed(speed)

    @property
    def state(self) -> ReplayState:
        return self._state

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def frames(self):
        return self._frames

    def set_speed(self, speed: float):
        speed = float(speed)
        if speed not in (1.0, 0.5, 0.25):
            raise ValueError("Replay speed must be 1.0, 0.5, or 0.25")
        self._speed = speed

    def begin(
        self,
        frames,
        *,
        decision_frame: int | None = None,
        zoom_x: float | None = None,
        zoom_y: float | None = None,
    ) -> bool:
        frames = tuple(frames)

        if not frames:
            return False

        self._frames = frames
        self._index = 0
        self._decision_frame = decision_frame
        self._zoom_x = zoom_x
        self._zoom_y = zoom_y
        self._state = ReplayState.FREEZE
        self.metrics.replay_count += 1
        return True

    def start_playback(self):
        if self._frames:
            self._state = ReplayState.PLAYING

    def current_packet(self):
        if not self._frames:
            return None
        return self._frames[min(self._index, len(self._frames) - 1)]

    def advance(self):
        if self._state == ReplayState.FREEZE:
            self._state = ReplayState.PLAYING
            return self.current_packet()

        if self._state != ReplayState.PLAYING:
            return None

        packet = self.current_packet()
        if packet is None:
            self._state = ReplayState.RESUME
            return None

        self.metrics.total_frames_shown += 1
        self._index += 1

        if self._index >= len(self._frames):
            self._state = ReplayState.RESUME

        return packet

    def finish(self):
        self._frames = ()
        self._index = 0
        self._decision_frame = None
        self._zoom_x = None
        self._zoom_y = None
        self._state = ReplayState.LIVE

    def view_model(self) -> ReplayViewModel:
        return ReplayViewModel(
            state=self._state,
            current_index=self._index,
            total_frames=len(self._frames),
            speed=self._speed,
            decision_frame=self._decision_frame,
            zoom_x=self._zoom_x,
            zoom_y=self._zoom_y,
        )
