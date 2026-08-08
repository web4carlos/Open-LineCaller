from .detector import BounceDetector

class BounceEngine:
    def __init__(self, detector=None, cooldown_frames=8, max_history=60):
        self.detector = detector or BounceDetector()
        self.cooldown_frames = int(cooldown_frames)
        self.max_history = int(max_history)
        self._points = []
        self._last_event_frame = None

    def reset(self):
        self._points.clear()
        self._last_event_frame = None

    def process(self, point):
        if self._points and point.frame_number <= self._points[-1].frame_number:
            raise ValueError("Trajectory points must arrive in increasing frame order")
        self._points.append(point)
        if len(self._points) > self.max_history:
            self._points.pop(0)

        hw = self.detector.analyzer.half_window
        if len(self._points) < 2 * hw + 1:
            return None

        center_index = len(self._points) - hw - 1
        event = self.detector.detect_at(self._points, center_index)
        if event is None:
            return None

        if self._last_event_frame is not None:
            if event.frame_number - self._last_event_frame < self.cooldown_frames:
                return None

        self._last_event_frame = event.frame_number
        return event
