from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict


class LiveEvidenceStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, evidence):
        payload = asdict(evidence)
        payload["decision"] = evidence.decision.value

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
