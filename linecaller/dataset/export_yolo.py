from __future__ import annotations

from pathlib import Path

from .models import ClipAnnotation


def _normalize_box(ball, width, height):
    cx = (ball.x + ball.width / 2.0) / width
    cy = (ball.y + ball.height / 2.0) / height
    w = ball.width / width
    h = ball.height / height
    return cx, cy, w, h


def export_clip_yolo(
    annotation: ClipAnnotation,
    output_dir: str | Path,
) -> int:
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for frame in annotation.frames:
        if frame.ball is None:
            continue

        cx, cy, w, h = _normalize_box(
            frame.ball,
            annotation.width,
            annotation.height,
        )

        path = labels_dir / f"{annotation.clip_id}_{frame.frame_number:06d}.txt"
        path.write_text(
            f"0 {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}\n",
            encoding="utf-8",
        )
        count += 1

    return count
