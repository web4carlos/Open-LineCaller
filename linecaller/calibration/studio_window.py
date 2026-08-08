from __future__ import annotations

from pathlib import Path
import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from linecaller.calibration.quality import CalibrationStatus
from linecaller.calibration.session import CalibrationSession
from linecaller.calibration.studio_widget import CalibrationVideoWidget


class CalibrationStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Open-LineCaller — Calibration Studio")
        self.resize(1400, 900)

        self.capture = None
        self.video_path: Path | None = None
        self.frame_count = 0
        self.current_frame_number = 0
        self.current_frame = None

        self.session = CalibrationSession()

        self._build_menu()
        self._build_ui()
        self._update_instruction()

    def _build_menu(self):
        menu = self.menuBar().addMenu("&File")

        open_action = QAction("Open Video...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_video)
        menu.addAction(open_action)

        save_action = QAction("Save Calibration...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_calibration)
        menu.addAction(save_action)

        menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        menu.addAction(exit_action)

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)

        self.instruction = QLabel()
        self.instruction.setStyleSheet(
            "font-size:16px; font-weight:600; padding:8px;"
        )
        layout.addWidget(self.instruction)

        self.video = CalibrationVideoWidget()
        self.video.frame_clicked.connect(self._add_point)
        layout.addWidget(self.video, stretch=1)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.goto_frame)
        layout.addWidget(self.slider)

        row = QHBoxLayout()

        prev_button = QPushButton("◀ Previous Frame")
        next_button = QPushButton("Next Frame ▶")
        undo_button = QPushButton("Undo Point")
        reset_button = QPushButton("Reset Points")
        compute_button = QPushButton("Compute Calibration")
        save_button = QPushButton("Save Calibration...")

        prev_button.clicked.connect(lambda: self.goto_frame(self.current_frame_number - 1))
        next_button.clicked.connect(lambda: self.goto_frame(self.current_frame_number + 1))
        undo_button.clicked.connect(self.undo_point)
        reset_button.clicked.connect(self.reset_points)
        compute_button.clicked.connect(self.compute_calibration)
        save_button.clicked.connect(self.save_calibration)

        for button in (
            prev_button,
            next_button,
            undo_button,
            reset_button,
            compute_button,
            save_button,
        ):
            row.addWidget(button)

        layout.addLayout(row)

        self.status = QLabel("No video loaded")
        self.status.setStyleSheet("padding:8px;")
        layout.addWidget(self.status)

        self.setCentralWidget(root)

    def open_video(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Pickleball Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
        )

        if not filename:
            return

        if self.capture is not None:
            self.capture.release()

        capture = cv2.VideoCapture(filename)

        if not capture.isOpened():
            QMessageBox.critical(self, "Calibration Studio", "Unable to open video.")
            return

        self.capture = capture
        self.video_path = Path(filename)
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_number = 0
        self.session.reset()

        self.slider.blockSignals(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, self.frame_count - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.slider.setEnabled(True)

        self.goto_frame(0)

    def goto_frame(self, frame_number: int):
        if self.capture is None:
            return

        frame_number = max(0, min(frame_number, max(0, self.frame_count - 1)))

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = self.capture.read()

        if not ok:
            return

        self.current_frame_number = frame_number
        self.current_frame = frame

        self.slider.blockSignals(True)
        self.slider.setValue(frame_number)
        self.slider.blockSignals(False)

        self._refresh_scene()
        self._update_instruction()

    def _add_point(self, x: float, y: float):
        if self.current_frame is None:
            return

        try:
            self.session.add_point(x, y)
        except ValueError:
            return

        self._refresh_scene()
        self._update_instruction()

    def undo_point(self):
        self.session.undo()
        self._refresh_scene()
        self._update_instruction()

    def reset_points(self):
        self.session.reset()
        self._refresh_scene()
        self._update_instruction()

    def compute_calibration(self):
        if self.current_frame is None:
            return

        if not self.session.complete:
            QMessageBox.warning(
                self,
                "Calibration Studio",
                "Select all requested court points first.",
            )
            return

        h, w = self.current_frame.shape[:2]

        try:
            result = self.session.build(
                name=self.video_path.stem if self.video_path else "court",
                image_size=(w, h),
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Calibration Studio",
                f"Calibration failed:\n{exc}",
            )
            return

        q = result.quality

        self.status.setText(
            f"Calibration: {q.status.value}   |   "
            f"Mean: {q.metrics.mean_error_px:.2f} px   |   "
            f"Max: {q.metrics.max_error_px:.2f} px   |   "
            f"RMS: {q.metrics.rms_error_px:.2f} px   |   "
            f"Inliers: {q.metrics.inlier_count}/{q.metrics.point_count}"
        )

        self._refresh_scene()

        if q.status == CalibrationStatus.INVALID:
            reasons = "\n".join(q.reasons) or "Quality gate failed."
            QMessageBox.warning(
                self,
                "Calibration INVALID",
                reasons,
            )

    def save_calibration(self):
        if self.session.result is None:
            QMessageBox.warning(
                self,
                "Calibration Studio",
                "Compute the calibration before saving.",
            )
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Calibration",
            "court.json",
            "JSON (*.json)",
        )

        if not filename:
            return

        self.session.result.profile.save(filename)

        QMessageBox.information(
            self,
            "Calibration Studio",
            f"Saved:\n{filename}",
        )

    def _refresh_scene(self):
        segments = self.session.overlay_segments()
        self.video.set_scene(
            self.current_frame,
            self.session.image_points,
            segments,
        )

    def _update_instruction(self):
        if self.current_frame is None:
            self.instruction.setText("Open a video to begin calibration.")
            return

        next_name = self.session.next_reference_name

        if next_name is None:
            self.instruction.setText(
                f"Frame {self.current_frame_number} — all 8 points selected. "
                "Press Compute Calibration."
            )
        else:
            index = len(self.session.image_points) + 1
            self.instruction.setText(
                f"Frame {self.current_frame_number} — "
                f"Point {index}/{len(self.session.reference_names)}: "
                f"click {next_name}"
            )

    def closeEvent(self, event):
        if self.capture is not None:
            self.capture.release()
        event.accept()
