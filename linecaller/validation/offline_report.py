from __future__ import annotations

import json
from pathlib import Path


def export_offline_validation_report(
    result,
    path,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "mode": "OFFLINE",
        "video": {
            "path": result.video_info.path,
            "frame_count": (
                result.video_info.frame_count
            ),
            "source_fps": (
                result.video_info.source_fps
            ),
            "width": result.video_info.width,
            "height": result.video_info.height,
        },
        "processing": {
            "frames_processed": (
                result.run_stats.frames_processed
            ),
            "wall_seconds": (
                result.run_stats.wall_seconds
            ),
            "effective_processing_fps": (
                result.run_stats.effective_processing_fps
            ),
        },
        "metrics": result.metrics.to_dict(),
        "comparisons": [
            {
                "event_id": c.event_id,
                "truth": c.truth,
                "prediction": c.prediction,
                "correct": c.correct,
                "category": c.category,
                "confidence": c.confidence,
                "latency_ms": c.latency_ms,
            }
            for c in result.comparisons
        ],
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path
