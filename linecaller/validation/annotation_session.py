from pathlib import Path
from .annotation_models import AnnotationLabel
from .models import ValidationTruthEvent

class AnnotationSession:
    def __init__(self):
        self.video_path = None
        self.candidates = []
        self.current_index = 0
        self.labels = {}

    def set_video(self, path):
        self.video_path = Path(path)
        self.candidates = []
        self.current_index = 0
        self.labels = {}

    def set_candidates(self, candidates):
        self.candidates = sorted(list(candidates), key=lambda c: c.frame)
        self.current_index = 0

    @property
    def current(self):
        return None if not self.candidates else self.candidates[self.current_index]

    def next(self):
        if self.candidates:
            self.current_index = min(len(self.candidates)-1, self.current_index+1)
        return self.current

    def previous(self):
        if self.candidates:
            self.current_index = max(0, self.current_index-1)
        return self.current

    def label_current(self, label):
        if self.current is None:
            return None
        label = AnnotationLabel(label)
        self.labels[int(self.current.frame)] = label
        return label

    def label_for_frame(self, frame):
        return self.labels.get(int(frame))

    def truth_events(self):
        if self.video_path is None:
            return []
        out=[]
        for c in self.candidates:
            label=self.labels.get(int(c.frame))
            if label not in (AnnotationLabel.IN, AnnotationLabel.OUT):
                continue
            out.append(ValidationTruthEvent(
                event_id=f"{self.video_path.stem}-{int(c.frame):08d}",
                video=self.video_path.name,
                frame=int(c.frame),
                truth=label.value,
                notes="CP-0019.2 Smart Annotation Studio",
            ))
        return out
