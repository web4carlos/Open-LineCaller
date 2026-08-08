from __future__ import annotations

import time
import cv2

from PySide6.QtCore import QThread, Signal


class LiveCameraWorker(QThread):
    frame_ready = Signal(object, float, int)
    camera_error = Signal(str)

    def __init__(self, camera_index: int = 0):
        super().__init__()
        self.camera_index = int(camera_index)
        self._running = False

    def stop(self):
        self._running = False

    def run(self):
        capture = cv2.VideoCapture(self.camera_index)

        if not capture.isOpened():
            self.camera_error.emit(
                f"Unable to open camera {self.camera_index}"
            )
            return

        self._running = True
        frame_number = 0
        started = time.perf_counter()

        try:
            while self._running:
                ok, frame = capture.read()

                if not ok:
                    self.camera_error.emit("Camera frame read failed")
                    break

                now = time.perf_counter()
                elapsed = max(1e-9, now - started)
                fps = (frame_number + 1) / elapsed

                self.frame_ready.emit(
                    frame.copy(),
                    float(fps),
                    int(frame_number),
                )

                frame_number += 1
        finally:
            capture.release()
