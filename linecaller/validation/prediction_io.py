from __future__ import annotations

import json
from pathlib import Path


def save_predictions_jsonl(
    predictions,
    path,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for prediction in predictions:
            handle.write(
                json.dumps(
                    prediction.to_dict(),
                    ensure_ascii=False,
                )
                + "\n"
            )

    return path
