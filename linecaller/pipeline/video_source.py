from __future__ import annotations

from pathlib import Path
import cv2


class VideoSource:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            self.capture = None
            raise RuntimeError(f"Cannot open video: {self.path}")

    @property
    def fps(self) -> float:
        if self.capture is None:
            return 0.0
        value = float(self.capture.get(cv2.CAP_PROP_FPS))
        return value if value > 0 else 30.0

    @property
    def width(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.capture else 0

    @property
    def height(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.capture else 0

    @property
    def frame_count(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) if self.capture else 0

    def frames(self):
        if self.capture is None:
            raise RuntimeError("VideoSource is not open")

        frame_number = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            yield frame_number, frame
            frame_number += 1

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
