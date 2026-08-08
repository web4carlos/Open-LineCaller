from __future__ import annotations

from collections import Counter
from pathlib import Path

from .validation import validate_clip


def dataset_quality_report(project):
    manifest = project.load_manifest()

    total_frames = 0
    ball_frames = 0
    total_bounces = 0
    issues = []

    for clip_id in manifest.clips:
        annotation = project.load_clip_annotation(clip_id)
        total_frames += annotation.frame_count
        ball_frames += sum(1 for f in annotation.frames if f.ball is not None)
        total_bounces += len(annotation.bounces)
        issues.extend(validate_clip(annotation))

    severity = Counter(issue.severity for issue in issues)

    return {
        "clips": len(manifest.clips),
        "total_frames": total_frames,
        "annotated_ball_frames": ball_frames,
        "bounce_annotations": total_bounces,
        "errors": severity.get("ERROR", 0),
        "warnings": severity.get("WARNING", 0),
        "issues": [
            {
                "clip_id": i.clip_id,
                "severity": i.severity,
                "message": i.message,
            }
            for i in issues
        ],
    }
