from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkClip:
    clip_id: str
    video: Path
    calibration: Path
    expected: Path


def load_manifest(root: str | Path) -> list[BenchmarkClip]:
    root = Path(root)
    path = root / "manifest.json"

    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("clips", data)

    clips = []
    for item in entries:
        clips.append(
            BenchmarkClip(
                clip_id=str(item["id"]),
                video=root / item["video"],
                calibration=root / item["calibration"],
                expected=root / item["expected"],
            )
        )
    return clips
