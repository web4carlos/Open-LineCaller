from __future__ import annotations

import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel


class MagnifierWidget(QLabel):
    def __init__(self, zoom=5, crop_radius=24):
        super().__init__()
        self.zoom = int(zoom)
        self.crop_radius = int(crop_radius)
        self.setFixedSize(260, 260)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#000; border:1px solid #666;")

    def update_from_frame(self, frame, x, y):
        if frame is None:
            return

        h, w = frame.shape[:2]
        r = self.crop_radius
        cx, cy = int(round(x)), int(round(y))

        x1, x2 = max(0, cx-r), min(w, cx+r+1)
        y1, y2 = max(0, cy-r), min(h, cy+r+1)

        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return

        crop = cv2.resize(
            crop,
            None,
            fx=self.zoom,
            fy=self.zoom,
            interpolation=cv2.INTER_NEAREST,
        )

        hh, ww = crop.shape[:2]
        cv2.line(crop, (ww//2, 0), (ww//2, hh), (0,255,0), 1)
        cv2.line(crop, (0, hh//2), (ww, hh//2), (0,255,0), 1)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        h2, w2, ch = rgb.shape
        img = QImage(rgb.data, w2, h2, ch*w2, QImage.Format.Format_RGB888).copy()
        self.setPixmap(
            QPixmap.fromImage(img).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )
