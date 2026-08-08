from __future__ import annotations

import cv2

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel

from linecaller.calibration.studio_mapping import DisplayMapping


class AnnotationVideoWidget(QLabel):
    box_changed = Signal(float, float, float, float)
    cursor_frame_position = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(900, 560)
        self.setMouseTracking(True)
        self.setStyleSheet("background:#111; color:#ddd;")
        self.setText("Open a video clip")

        self._frame = None
        self._box = None
        self._bounce = None
        self._proposal = None
        self._start = None
        self._current = None

    def set_scene(self, frame, box=None, bounce=None, proposal=None):
        self._frame = frame.copy() if frame is not None else None
        self._box = box
        self._bounce = bounce
        self._proposal = proposal
        self._render()

    def _point(self, event):
        if self._frame is None:
            return None

        h, w = self._frame.shape[:2]
        mapping = DisplayMapping(
            frame_width=w,
            frame_height=h,
            widget_width=max(1, self.width()),
            widget_height=max(1, self.height()),
        )

        return mapping.widget_to_frame(
            event.position().x(),
            event.position().y(),
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            point = self._point(event)
            if point is not None:
                self._start = point
                self._current = point

    def mouseMoveEvent(self, event):
        point = self._point(event)
        if point is None:
            return

        self.cursor_frame_position.emit(*point)

        if self._start is not None:
            self._current = point
            self._render()

    def mouseReleaseEvent(self, event):
        if self._start is None:
            return

        point = self._point(event)
        start = self._start

        self._start = None
        self._current = None

        if point is None:
            return

        x = min(start[0], point[0])
        y = min(start[1], point[1])
        w = abs(point[0] - start[0])
        h = abs(point[1] - start[1])

        if w >= 2 and h >= 2:
            self.box_changed.emit(x, y, w, h)

        self._render()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    @staticmethod
    def _proposal_color(status):
        if status is None:
            return (255, 255, 0)

        value = getattr(status, "value", str(status))

        if value == "AUTO":
            return (0, 255, 255)
        if value == "REVIEW":
            return (0, 200, 255)
        return (0, 0, 255)

    def _render(self):
        if self._frame is None:
            return

        display = self._frame.copy()

        if self._proposal is not None and self._box is None:
            p = self._proposal
            color = self._proposal_color(p.status)

            cv2.rectangle(
                display,
                (int(p.x), int(p.y)),
                (int(p.x + p.width), int(p.y + p.height)),
                color,
                2,
            )

            cv2.putText(
                display,
                f"PROPOSAL {p.confidence:.2f}",
                (int(p.x), max(20, int(p.y) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        if self._box is not None:
            cv2.rectangle(
                display,
                (int(self._box.x), int(self._box.y)),
                (
                    int(self._box.x + self._box.width),
                    int(self._box.y + self._box.height),
                ),
                (0, 255, 0),
                2,
            )

        if self._bounce is not None:
            point = (int(self._bounce.x), int(self._bounce.y))
            cv2.circle(display, point, 12, (0, 255, 255), 2)

        if self._start is not None and self._current is not None:
            x1, y1 = self._start
            x2, y2 = self._current
            cv2.rectangle(
                display,
                (int(min(x1, x2)), int(min(y1, y2))),
                (int(max(x1, x2)), int(max(y1, y2))),
                (255, 255, 255),
                1,
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

        self.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
