import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSlider, QVBoxLayout, QWidget
)
from .session import ValidationSession
from .io import save_truth_jsonl, load_truth_jsonl, load_predictions_jsonl
from .comparison import compare_events
from .metrics import compute_metrics
from .report import export_validation_report

class ValidationLabWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open-LineCaller Validation Lab")
        self.resize(1300, 820)

        self.session = ValidationSession()
        self.capture = None
        self.frame_count = 0
        self.current_frame_number = 0
        self.predictions = []
        self.comparisons = ()
        self.metrics = None

        root = QWidget()
        layout = QVBoxLayout(root)

        buttons = QHBoxLayout()
        for text, handler in [
            ("Open Video", self.open_video),
            ("Load Ground Truth", self.load_truth),
            ("Save Ground Truth", self.save_truth),
            ("Load Predictions", self.load_predictions),
            ("Compare", self.compare),
            ("Export Report", self.export_report),
        ]:
            b = QPushButton(text)
            b.clicked.connect(handler)
            buttons.addWidget(b)
        layout.addLayout(buttons)

        self.preview = QLabel("Open a real pickleball video")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(520)
        self.preview.setStyleSheet("background:#111;color:#ddd;")
        layout.addWidget(self.preview, 1)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.goto_frame)
        layout.addWidget(self.slider)

        controls = QHBoxLayout()
        for text, handler in [
            ("◀ Frame", lambda: self.goto_frame(self.current_frame_number - 1)),
            ("Frame ▶", lambda: self.goto_frame(self.current_frame_number + 1)),
            ("GROUND TRUTH: IN", lambda: self.mark_truth("IN")),
            ("GROUND TRUTH: OUT", lambda: self.mark_truth("OUT")),
        ]:
            b = QPushButton(text)
            b.clicked.connect(handler)
            controls.addWidget(b)
        layout.addLayout(controls)

        self.status = QLabel("No video loaded")
        self.metrics_label = QLabel("")
        self.metrics_label.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addWidget(self.metrics_label)

        self.setCentralWidget(root)

    def open_video(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open validation video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if not filename:
            return

        if self.capture:
            self.capture.release()

        self.capture = cv2.VideoCapture(filename)
        if not self.capture.isOpened():
            QMessageBox.critical(self, "Validation Lab", "Unable to open video.")
            return

        self.session.set_video(filename)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setEnabled(True)
        self.slider.setRange(0, max(0, self.frame_count - 1))
        self.goto_frame(0)

    def goto_frame(self, frame_number):
        if self.capture is None:
            return

        frame_number = max(0, min(int(frame_number), max(0, self.frame_count - 1)))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = self.capture.read()
        if not ok:
            return

        self.current_frame_number = frame_number
        self.slider.blockSignals(True)
        self.slider.setValue(frame_number)
        self.slider.blockSignals(False)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888).copy()
        self.preview.setPixmap(
            QPixmap.fromImage(img).scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        self.status.setText(
            f"{self.session.video_path.name} | Frame {frame_number}/{self.frame_count - 1} | "
            f"Ground truth events: {len(self.session.truth_events)}"
        )

    def mark_truth(self, truth):
        if self.session.video_path is None:
            return
        event = self.session.add_truth(frame=self.current_frame_number, truth=truth)
        self.status.setText(f"Marked {event.event_id}: {truth}")

    def save_truth(self):
        if not self.session.truth_events:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Ground Truth", "ground_truth.jsonl", "JSON Lines (*.jsonl)"
        )
        if filename:
            save_truth_jsonl(self.session.truth_events, filename)

    def load_truth(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Ground Truth", "", "JSON Lines (*.jsonl)"
        )
        if filename:
            self.session.truth_events = load_truth_jsonl(filename)

    def load_predictions(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Predictions", "", "JSON Lines (*.jsonl)"
        )
        if filename:
            self.predictions = load_predictions_jsonl(filename)

    def compare(self):
        self.comparisons = compare_events(self.session.truth_events, self.predictions)
        self.metrics = compute_metrics(self.comparisons)
        m = self.metrics
        self.metrics_label.setText(
            f"Accuracy: {m.automatic_accuracy:.2%} | Coverage: {m.coverage:.2%} | "
            f"Review: {m.review_rate:.2%} | Miss: {m.miss_rate:.2%} | "
            f"False IN: {m.false_in} | False OUT: {m.false_out} | "
            f"Avg confidence: {m.average_confidence:.2%} | "
            f"Avg latency: {m.average_latency_ms:.1f} ms"
        )

    def export_report(self):
        if self.metrics is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Validation Report", "validation_report.json", "JSON (*.json)"
        )
        if filename:
            export_validation_report(self.metrics, self.comparisons, filename)

    def closeEvent(self, event):
        if self.capture:
            self.capture.release()
        event.accept()
