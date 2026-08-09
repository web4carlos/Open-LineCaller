from pathlib import Path
from .models import ValidationTruthEvent

class ValidationSession:
    def __init__(self):
        self.video_path = None
        self.truth_events = []

    def set_video(self, path):
        self.video_path = Path(path)

    def add_truth(self, *, frame, truth, notes=""):
        if self.video_path is None:
            raise RuntimeError("Load a video first")

        event_id = f"{self.video_path.stem}-{int(frame):08d}"
        event = ValidationTruthEvent(
            event_id=event_id,
            video=self.video_path.name,
            frame=int(frame),
            truth=truth,
            notes=notes,
        )

        self.truth_events = [e for e in self.truth_events if e.event_id != event_id]
        self.truth_events.append(event)
        self.truth_events.sort(key=lambda e: e.frame)
        return event
