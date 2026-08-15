from __future__ import annotations

from dataclasses import dataclass
import mimetypes
from pathlib import Path
from typing import Optional

import cv2


@dataclass(frozen=True, eq=False)
class VideoSource:
    path: Path
    fps: float
    width: int
    height: int

    def __eq__(self, other):
        if isinstance(other, VideoSource):
            return (
                self.path == other.path
                and self.fps == other.fps
                and self.width == other.width
                and self.height == other.height
            )

        if isinstance(other, (str, Path)):
            return self.path == Path(other).expanduser().resolve()

        return NotImplemented

    def describe(self, court_id: str) -> dict:
        return {
            "court_id": court_id,
            "available": self.path.is_file(),
            "filename": self.path.name,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
        }


class VideoSourceRegistry:
    def __init__(self):
        self._sources: dict[str, VideoSource] = {}

    def bind(self, court_id: str, path: str | Path) -> VideoSource:
        p = Path(path).expanduser().resolve()

        if not p.is_file():
            raise FileNotFoundError(p)

        # Metadata probing is best-effort.
        # The HTTP bridge may still bind/serve an existing file even when
        # OpenCV cannot decode it (important for browser-supported media).
        fps = 60.0
        width = 0
        height = 0

        cap = cv2.VideoCapture(str(p))

        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        cap.release()

        source = VideoSource(
            path=p,
            fps=fps,
            width=width,
            height=height,
        )

        self._sources[str(court_id)] = source
        return source

    def get(self, court_id: str) -> Optional[VideoSource]:
        return self._sources.get(str(court_id))

    def describe(self, court_id: str) -> dict:
        source = self.get(court_id)

        if source is None:
            return {
                "court_id": court_id,
                "available": False,
                "filename": "",
                "fps": None,
                "width": None,
                "height": None,
            }

        return source.describe(court_id)


def content_type_for(path: str | Path) -> str:
    p = Path(path)

    return (
        mimetypes.guess_type(p.name)[0]
        or "application/octet-stream"
    )


def parse_range_header(
    value: str | None,
    size: int,
) -> tuple[int, int] | None:

    if not value or not value.startswith("bytes="):
        return None

    spec = value[6:].split(",", 1)[0].strip()

    if "-" not in spec:
        return None

    left, right = spec.split("-", 1)

    if left == "":
        length = int(right)

        if length <= 0:
            return None

        return max(0, size - length), size - 1

    start = int(left)
    end = int(right) if right else size - 1

    if start < 0 or start >= size or end < start:
        return None

    return start, min(end, size - 1)
