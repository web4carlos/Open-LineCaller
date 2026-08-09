from __future__ import annotations

import json
from pathlib import Path

from .summary_models import MatchSummary


class MatchReportExporter:
    def export_json(
        self,
        summary: MatchSummary,
        path,
    ) -> Path:
        path = Path(path)

        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")

        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(
                summary.to_dict(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return path
