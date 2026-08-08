from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass(frozen=True)
class BallBox:
    x: float
    y: float
    width: float
    height: float

    def validate(self) -> list[str]:
        errors = []
        if self.width <= 0:
            errors.append("BallBox width must be > 0")
        if self.height <= 0:
            errors.append("BallBox height must be > 0")
        return errors


@dataclass(frozen=True)
class FrameAnnotation:
    frame_number: int
    ball: BallBox | None = None
    visible: bool = True
    occluded: bool = False
    quality: str = "OK"


@dataclass(frozen=True)
class BounceAnnotation:
    frame_number: int
    x: float
    y: float
    decision: str | None = None
    court_x_m: float | None = None
    court_y_m: float | None = None


@dataclass
class ClipAnnotation:
    clip_id: str
    source_video: str
    width: int
    height: int
    fps: float
    frame_count: int
    frames: list[FrameAnnotation] = field(default_factory=list)
    bounces: list[BounceAnnotation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetManifest:
    name: str
    version: str
    clips: list[str] = field(default_factory=list)
    split_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
