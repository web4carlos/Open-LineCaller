from __future__ import annotations

import cv2

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel

from linecaller.calibration.studio_mapping import DisplayMapping


class CalibrationVideoWidget(QLabel):
    frame_clicked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(900, 560)
        self.setStyleSheet("background:#111; color:#ddd;")
        self.setText("Open a pickleball video")
        self._frame = None
        self._points = []
        self._segments = []

    def set_scene(self, frame, points, segments):
        self._frame = frame.copy() if frame is not None else None
        self._points = list(points)
        self._segments = list(segments)
        self._render()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    def mousePressEvent(self, event):
        if self._frame is None:
            return

        h, w = self._frame.shape[:2]
        mapping = DisplayMapping(
            frame_width=w,
            frame_height=h,
            widget_width=max(1, self.width()),
            widget_height=max(1, self.height()),
        )

        point = mapping.widget_to_frame(
            event.position().x(),
            event.position().y(),
        )

        if point is not None:
            self.frame_clicked.emit(point[0], point[1])

    def _render(self):
        if self._frame is None:
            return

        display = self._frame.copy()

        for segment in self._segments:
            p1 = (int(round(segment.p1[0])), int(round(segment.p1[1])))
            p2 = (int(round(segment.p2[0])), int(round(segment.p2[1])))
            cv2.line(display, p1, p2, (0, 255, 255), 2)

        for index, point in enumerate(self._points, start=1):
            p = (int(round(point[0])), int(round(point[1])))
            cv2.circle(display, p, 7, (0, 255, 0), -1)
            cv2.putText(
                display,
                str(index),
                (p[0] + 8, p[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        image = QImage(
            rgb.data,
            w,
            h,
            ch * w,
            QImage.Format.Format_RGB888,
        ).copy()

        pixmap = QPixmap.fromImage(image).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)
