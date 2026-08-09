from __future__ import annotations

import json
from pathlib import Path

from .summary_models import MatchSummary


class MatchHistoryStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, summary: MatchSummary):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(summary.to_dict(), ensure_ascii=False)
                + "\n"
            )

    def load_all(self) -> list[dict]:
        if not self.path.exists():
            return []

        rows = []

        for line in self.path.read_text(
            encoding="utf-8"
        ).splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))

        return rows
