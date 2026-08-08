from __future__ import annotations


class DecisionCooldown:
    def __init__(self, *, cooldown_frames: int = 12):
        self.cooldown_frames = int(cooldown_frames)
        self._last_call_frame: int | None = None

    def reset(self):
        self._last_call_frame = None

    def should_emit(self, frame_number: int) -> bool:
        if self._last_call_frame is None:
            self._last_call_frame = int(frame_number)
            return True

        if int(frame_number) - self._last_call_frame >= self.cooldown_frames:
            self._last_call_frame = int(frame_number)
            return True

        return False
