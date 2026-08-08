from collections import deque
from dataclasses import dataclass

@dataclass(frozen=True)
class TrajectoryPoint:
    frame_number: int
    x: float
    y: float
    measured: bool
    confidence: float

class TrajectoryHistory:
    def __init__(self, max_points=120):
        self._points = deque(maxlen=max_points)

    def add(self, point):
        self._points.append(point)

    def points(self):
        return tuple(self._points)

    def clear(self):
        self._points.clear()

    def __len__(self):
        return len(self._points)
