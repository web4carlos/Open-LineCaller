from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .navigation import ProductRoute


class HomeScreen(QWidget):
    navigate_requested = Signal(object)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        brand = QLabel("Open-LineCaller")
        brand.setObjectName("brandTitle")

        subtitle = QLabel("Live pickleball officiating")
        subtitle.setStyleSheet("font-size:18px; color:#b8bcc5;")

        start = QPushButton("START MATCH")
        start.setObjectName("primaryButton")
        start.clicked.connect(
            lambda: self.navigate_requested.emit(
                ProductRoute.START_MATCH
            )
        )

        history = QPushButton("History")
        history.clicked.connect(
            lambda: self.navigate_requested.emit(
                ProductRoute.HISTORY
            )
        )

        settings = QPushButton("Settings")
        settings.clicked.connect(
            lambda: self.navigate_requested.emit(
                ProductRoute.SETTINGS
            )
        )

        diagnostics = QPushButton("Diagnostics")
        diagnostics.clicked.connect(
            lambda: self.navigate_requested.emit(
                ProductRoute.DIAGNOSTICS
            )
        )

        layout.addStretch(1)
        layout.addWidget(brand)
        layout.addWidget(subtitle)
        layout.addSpacing(24)
        layout.addWidget(start)
        layout.addWidget(history)
        layout.addWidget(settings)
        layout.addWidget(diagnostics)
        layout.addStretch(2)


class PlaceholderScreen(QWidget):
    navigate_requested = Signal(object)

    def __init__(self, title: str, message: str):
        super().__init__()

        layout = QVBoxLayout(self)

        page = QLabel(title)
        page.setObjectName("pageTitle")

        text = QLabel(message)
        text.setWordWrap(True)
        text.setStyleSheet("color:#b8bcc5;")

        home = QPushButton("Back to Home")
        home.clicked.connect(
            lambda: self.navigate_requested.emit(
                ProductRoute.HOME
            )
        )

        layout.addStretch(1)
        layout.addWidget(page)
        layout.addWidget(text)
        layout.addSpacing(20)
        layout.addWidget(home)
        layout.addStretch(2)
