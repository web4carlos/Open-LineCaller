from __future__ import annotations

from dataclasses import dataclass
import cv2


@dataclass(frozen=True)
class OfflineFrame:
    frame_number: int
    timestamp: float
    frame: object


class OfflineFrameIterator:
    def __init__(self, video_path):
        self.video_path = str(video_path)

    def metadata(self):
        capture = cv2.VideoCapture(self.video_path)

        if not capture.isOpened():
            raise RuntimeError(
                f"Unable to open video: {self.video_path}"
            )

        try:
            fps = float(
                capture.get(cv2.CAP_PROP_FPS)
            )
            frame_count = int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )
            width = int(
                capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            )
            height = int(
                capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            return {
                "frame_count": frame_count,
                "source_fps": fps,
                "width": width,
                "height": height,
            }
        finally:
            capture.release()

    def __iter__(self):
        capture = cv2.VideoCapture(self.video_path)

        if not capture.isOpened():
            raise RuntimeError(
                f"Unable to open video: {self.video_path}"
            )

        fps = float(
            capture.get(cv2.CAP_PROP_FPS)
        )

        if fps <= 0:
            fps = 30.0

        frame_number = 0

        try:
            while True:
                ok, frame = capture.read()

                if not ok:
                    break

                yield OfflineFrame(
                    frame_number=frame_number,
                    timestamp=frame_number / fps,
                    frame=frame,
                )

                frame_number += 1
        finally:
            capture.release()
