from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)

from .hud_models import StatusLevel


def level_style(level: StatusLevel) -> str:
    mapping = {
        StatusLevel.OK: "color:#d9f7df;",
        StatusLevel.WARN: "color:#ffe8a3;",
        StatusLevel.ERROR: "color:#ffb4b4;",
        StatusLevel.UNKNOWN: "color:#c7cbd3;",
    }
    return mapping[level]


class MetricCard(QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)

        layout = QVBoxLayout(self)

        self.value = QLabel("-")
        self.value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value.setStyleSheet("font-size:22px; font-weight:800;")

        self.detail = QLabel("")
        self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail.setStyleSheet("font-size:12px; color:#aeb4bf;")

        layout.addWidget(self.value)
        layout.addWidget(self.detail)

    def set_metric(self, metric):
        self.value.setText(metric.value)
        self.value.setStyleSheet(
            f"font-size:22px; font-weight:800; {level_style(metric.level)}"
        )
        self.detail.setText(metric.detail)


class ConfidenceCard(MetricCard):
    def __init__(self):
        super().__init__("Confidence")

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)

        self.layout().addWidget(self.bar)

    def set_metric(self, metric):
        super().set_metric(metric)

        text = metric.value.replace("%", "")
        try:
            value = int(round(float(text)))
        except Exception:
            value = 0

        self.bar.setValue(value)
