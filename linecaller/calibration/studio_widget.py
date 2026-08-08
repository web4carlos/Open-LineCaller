from __future__ import annotations

import cv2
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor, QImage, QPixmap
from PySide6.QtWidgets import QLabel

from linecaller.calibration.studio_mapping import DisplayMapping


class CalibrationVideoWidget(QLabel):
    frame_clicked = Signal(float, float)
    frame_dragged = Signal(int, float, float)
    cursor_frame_position = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(900, 560)
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.setStyleSheet("background:#111; color:#ddd;")
        self.setText("Open a pickleball video")

        self._frame = None
        self._points = []
        self._segments = []
        self._errors = {}
        self._drag_index = None

    def set_scene(self, frame, points, segments, diagnostics=()):
        self._frame = frame.copy() if frame is not None else None
        self._points = list(points)
        self._segments = list(segments)
        self._errors = {d.index: d.error_px for d in diagnostics}
        self._render()

    def _mapping(self):
        if self._frame is None:
            return None
        h, w = self._frame.shape[:2]
        return DisplayMapping(
            frame_width=w,
            frame_height=h,
            widget_width=max(1, self.width()),
            widget_height=max(1, self.height()),
        )

    def _widget_to_frame(self, event):
        mapping = self._mapping()
        if mapping is None:
            return None
        return mapping.widget_to_frame(event.position().x(), event.position().y())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    def mouseMoveEvent(self, event):
        point = self._widget_to_frame(event)
        if point is None:
            return

        x, y = point
        self.cursor_frame_position.emit(x, y)

        if self._drag_index is not None:
            self.frame_dragged.emit(self._drag_index, x, y)

    def mousePressEvent(self, event):
        point = self._widget_to_frame(event)
        if point is None:
            return

        x, y = point

        # Select an existing point for drag if close enough.
        if self._points:
            pts = np.asarray(self._points, dtype=float)
            d = np.linalg.norm(pts - np.asarray([x, y]), axis=1)
            idx = int(np.argmin(d))
            if float(d[idx]) <= 18.0:
                self._drag_index = idx
                return

        self.frame_clicked.emit(x, y)

    def mouseReleaseEvent(self, event):
        self._drag_index = None

    @staticmethod
    def _quality_color(error_px):
        if error_px is None:
            return (0, 255, 0)
        if error_px <= 2.0:
            return (0, 220, 0)
        if error_px <= 4.0:
            return (0, 220, 255)
        return (0, 0, 255)

    def _render(self):
        if self._frame is None:
            return

        display = self._frame.copy()

        for seg in self._segments:
            p1 = (int(round(seg.p1[0])), int(round(seg.p1[1])))
            p2 = (int(round(seg.p2[0])), int(round(seg.p2[1])))
            cv2.line(display, p1, p2, (0, 255, 255), 2)

        for i, point in enumerate(self._points):
            p = (int(round(point[0])), int(round(point[1])))
            err = self._errors.get(i)
            color = self._quality_color(err)
            cv2.circle(display, p, 8, color, -1)
            label = f"{i+1}" if err is None else f"{i+1} ({err:.1f}px)"
            cv2.putText(
                display, label, (p[0]+10, p[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
            )

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)
