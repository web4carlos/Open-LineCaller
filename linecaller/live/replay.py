from __future__ import annotations

from collections import deque

from .models import LiveFramePacket


class ReplayBuffer:
    def __init__(self, *, max_frames: int = 240):
        if max_frames <= 0:
            raise ValueError("max_frames must be > 0")
        self._frames = deque(maxlen=int(max_frames))

    def append(self, packet: LiveFramePacket) -> None:
        self._frames.append(packet)

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    def snapshot(
        self,
        *,
        event_frame: int,
        pre_frames: int = 60,
        post_frames: int = 15,
    ) -> tuple[LiveFramePacket, ...]:
        low = event_frame - int(pre_frames)
        high = event_frame + int(post_frames)

        return tuple(
            packet
            for packet in self._frames
            if low <= packet.frame_number <= high
        )
