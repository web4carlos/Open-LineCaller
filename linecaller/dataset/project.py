from __future__ import annotations

import json
from pathlib import Path

from .models import (
    BallBox,
    BounceAnnotation,
    ClipAnnotation,
    DatasetManifest,
    FrameAnnotation,
)


class DatasetProject:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.annotations_dir = self.root / "annotations"
        self.clips_dir = self.root / "clips"
        self.splits_dir = self.root / "splits"
        self.exports_dir = self.root / "exports"
        self.reports_dir = self.root / "reports"

    def initialize(self, *, name: str | None = None, version: str = "1.0") -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for path in (
            self.annotations_dir,
            self.clips_dir,
            self.splits_dir,
            self.exports_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        if not self.manifest_path.exists():
            manifest = DatasetManifest(
                name=name or self.root.name,
                version=version,
            )
            self.save_manifest(manifest)

    def load_manifest(self) -> DatasetManifest:
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return DatasetManifest(
            name=data["name"],
            version=data["version"],
            clips=list(data.get("clips", [])),
            split_seed=int(data.get("split_seed", 42)),
        )

    def save_manifest(self, manifest: DatasetManifest) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2),
            encoding="utf-8",
        )

    def annotation_path(self, clip_id: str) -> Path:
        return self.annotations_dir / f"{clip_id}.json"

    def save_clip_annotation(self, annotation: ClipAnnotation) -> None:
        self.annotation_path(annotation.clip_id).write_text(
            json.dumps(annotation.to_dict(), indent=2),
            encoding="utf-8",
        )

    def load_clip_annotation(self, clip_id: str) -> ClipAnnotation:
        data = json.loads(
            self.annotation_path(clip_id).read_text(encoding="utf-8")
        )

        frames = []
        for item in data.get("frames", []):
            ball_data = item.get("ball")
            ball = BallBox(**ball_data) if ball_data else None
            frames.append(
                FrameAnnotation(
                    frame_number=int(item["frame_number"]),
                    ball=ball,
                    visible=bool(item.get("visible", True)),
                    occluded=bool(item.get("occluded", False)),
                    quality=str(item.get("quality", "OK")),
                )
            )

        bounces = [
            BounceAnnotation(**item)
            for item in data.get("bounces", [])
        ]

        return ClipAnnotation(
            clip_id=data["clip_id"],
            source_video=data["source_video"],
            width=int(data["width"]),
            height=int(data["height"]),
            fps=float(data["fps"]),
            frame_count=int(data["frame_count"]),
            frames=frames,
            bounces=bounces,
        )
