from dataclasses import dataclass

@dataclass
class MatchSessionStats:
    frames: int = 0
    events: int = 0
    calls_in: int = 0
    calls_out: int = 0
    calls_review: int = 0
    last_processing_ms: float = 0.0

    def record_event(self, decision):
        if not decision:
            return
        self.events += 1
        value = str(decision).upper()
        if value == "IN":
            self.calls_in += 1
        elif value == "OUT":
            self.calls_out += 1
        else:
            self.calls_review += 1

    def reset(self):
        self.frames = self.events = self.calls_in = self.calls_out = self.calls_review = 0
        self.last_processing_ms = 0.0
