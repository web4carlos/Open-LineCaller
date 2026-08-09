from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .match_history import MatchHistoryStore
from .report_exporter import MatchReportExporter


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class MatchSummaryScreen(QWidget):
    home_requested = Signal()

    def __init__(
        self,
        *,
        history_path="data/match_history.jsonl",
    ):
        super().__init__()

        self.summary = None
        self.history = MatchHistoryStore(history_path)
        self.exporter = MatchReportExporter()

        layout = QVBoxLayout(self)

        title = QLabel("MATCH SUMMARY")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        self.grid = QGridLayout()
        layout.addLayout(self.grid)

        self.values = {}

        fields = [
            ("Duration", "duration"),
            ("Calls", "calls"),
            ("IN", "in"),
            ("OUT", "out"),
            ("REVIEW", "review"),
            ("Average FPS", "fps"),
            ("Average Latency", "latency"),
            ("Average Confidence", "confidence"),
            ("Tracking Losses", "tracking_losses"),
            ("Replay Count", "replays"),
            ("Camera", "camera"),
            ("Calibration", "calibration"),
        ]

        for row, (label, key) in enumerate(fields):
            self.grid.addWidget(QLabel(label), row, 0)
            value = QLabel("-")
            value.setStyleSheet(
                "font-size:18px; font-weight:700;"
            )
            self.grid.addWidget(value, row, 1)
            self.values[key] = value

        self.save_btn = QPushButton("Save Match")
        self.export_btn = QPushButton("Export JSON")
        self.home_btn = QPushButton("Back to Home")

        self.save_btn.clicked.connect(self.save_match)
        self.export_btn.clicked.connect(self.export_json)
        self.home_btn.clicked.connect(self.home_requested.emit)

        layout.addWidget(self.save_btn)
        layout.addWidget(self.export_btn)
        layout.addWidget(self.home_btn)
        layout.addStretch(1)

    def set_summary(self, summary):
        self.summary = summary

        self.values["duration"].setText(
            format_duration(summary.duration_seconds)
        )
        self.values["calls"].setText(str(summary.calls_total))
        self.values["in"].setText(str(summary.calls_in))
        self.values["out"].setText(str(summary.calls_out))
        self.values["review"].setText(str(summary.calls_review))
        self.values["fps"].setText(
            f"{summary.average_fps:.1f}"
        )
        self.values["latency"].setText(
            f"{summary.average_latency_ms:.1f} ms"
        )
        self.values["confidence"].setText(
            f"{summary.average_confidence*100:.1f}%"
        )
        self.values["tracking_losses"].setText(
            str(summary.tracking_losses)
        )
        self.values["replays"].setText(
            str(summary.replay_count)
        )
        self.values["camera"].setText(summary.camera_name)
        self.values["calibration"].setText(
            summary.calibration_mode
        )

    def save_match(self):
        if self.summary is None:
            return

        self.history.append(self.summary)

        QMessageBox.information(
            self,
            "Match",
            "Match saved to local history.",
        )

    def export_json(self):
        if self.summary is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Match Report",
            f"{self.summary.match_id}.json",
            "JSON (*.json)",
        )

        if not filename:
            return

        path = self.exporter.export_json(
            self.summary,
            filename,
        )

        QMessageBox.information(
            self,
            "Export",
            f"Report exported:\n{path}",
        )
