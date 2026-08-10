from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path

@dataclass(frozen=True)
class DetectorProfile:
    min_area: float = 4.0
    max_area: float = 1400.0
    min_radius_px: float = 1.0
    max_radius_px: float = 35.0
    min_circularity: float = 0.08
    history: int = 300
    var_threshold: float = 16.0
    max_candidates: int = 24
    temporal_radius_px: float = 180.0
    min_confidence: float = 0.0

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
