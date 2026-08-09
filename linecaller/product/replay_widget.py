from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QWidget,
)


class ReplayOverlay(QWidget):
    def __init__(self):
        super().__init__()

        layout = QHBoxLayout(self)

        self.label = QLabel("REPLAY")
        self.label.setStyleSheet(
            "font-size:18px; font-weight:900;"
        )

        self.speed = QLabel("0.25x")
        self.timeline = QProgressBar()
        self.timeline.setRange(0, 100)
        self.timeline.setTextVisible(False)

        layout.addWidget(self.label)
        layout.addWidget(self.speed)
        layout.addWidget(self.timeline, 1)

        self.setVisible(False)

    def update_from_model(self, model):
        self.setVisible(model.active)
        self.speed.setText(model.speed_label)

        if model.total_frames <= 0:
            self.timeline.setValue(0)
            return

        progress = int(
            round(
                min(
                    1.0,
                    model.current_index / model.total_frames,
                )
                * 100
            )
        )

        self.timeline.setValue(progress)
