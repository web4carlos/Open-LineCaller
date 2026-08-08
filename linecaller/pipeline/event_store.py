from __future__ import annotations

import json
from pathlib import Path

from linecaller.decision.models import DecisionResult


class DecisionEventStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")

    def write(self, result: DecisionResult) -> None:
        c = result.context
        payload = {
            "decision": result.decision.value,
            "confidence": result.confidence,
            "explanation": list(result.explanation),
            "bounce_frame": c.bounce_frame,
            "image": {"x": c.image_x, "y": c.image_y},
            "court": {
                "x_m": c.court_x_m,
                "y_m": c.court_y_m,
            },
            "nearest_line": c.nearest_line,
            "signed_distance_m": c.signed_distance_m,
            "total_uncertainty_m": c.total_uncertainty_m,
            "bounce_confidence": c.bounce_confidence,
            "calibration_valid": c.calibration_valid,
        }
        self._file.write(json.dumps(payload) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
