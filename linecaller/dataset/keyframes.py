from __future__ import annotations


class AnnotationKeyframeIndex:
    def __init__(self):
        self._frames: set[int] = set()

    def mark(self, frame_number: int):
        self._frames.add(int(frame_number))

    def unmark(self, frame_number: int):
        self._frames.discard(int(frame_number))

    def toggle(self, frame_number: int) -> bool:
        f = int(frame_number)
        if f in self._frames:
            self._frames.remove(f)
            return False
        self._frames.add(f)
        return True

    def is_keyframe(self, frame_number: int) -> bool:
        return int(frame_number) in self._frames

    def all(self) -> tuple[int, ...]:
        return tuple(sorted(self._frames))

    def previous(self, frame_number: int):
        values = [f for f in self._frames if f < frame_number]
        return max(values) if values else None

    def next(self, frame_number: int):
        values = [f for f in self._frames if f > frame_number]
        return min(values) if values else None
