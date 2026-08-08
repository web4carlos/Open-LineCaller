from __future__ import annotations

from pathlib import Path
import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QListWidget, QMainWindow,
    QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget
)

from linecaller.calibration.labels import human_label
from linecaller.calibration.quality import CalibrationStatus
from linecaller.calibration.session import CalibrationSession
from linecaller.calibration.studio_widget import CalibrationVideoWidget
from linecaller.calibration.magnifier_widget import MagnifierWidget


class CalibrationStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open-LineCaller — Precision Calibration Studio")
        self.resize(1500, 920)

        self.capture = None
        self.video_path = None
        self.frame_count = 0
        self.current_frame_number = 0
        self.current_frame = None
        self.session = CalibrationSession()

        self._build_menu()
        self._build_ui()
        self._update_instruction()
        self._update_point_list()

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

    def _build_ui(self):
        root = QWidget()
        outer = QHBoxLayout(root)

        left = QVBoxLayout()
        right = QVBoxLayout()

        self.instruction = QLabel()
        self.instruction.setStyleSheet("font-size:16px; font-weight:600; padding:8px;")
        left.addWidget(self.instruction)

        self.video = CalibrationVideoWidget()
        self.video.frame_clicked.connect(self._add_point)
        self.video.frame_dragged.connect(self._move_point)
        self.video.cursor_frame_position.connect(self._cursor_moved)
        left.addWidget(self.video, stretch=1)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.goto_frame)
        left.addWidget(self.slider)

        buttons = QHBoxLayout()
        specs = [
            ("◀ Previous Frame", lambda: self.goto_frame(self.current_frame_number - 1)),
            ("Next Frame ▶", lambda: self.goto_frame(self.current_frame_number + 1)),
            ("Undo Point", self.undo_point),
            ("Reset Points", self.reset_points),
            ("Compute Calibration", self.compute_calibration),
            ("Save Calibration...", self.save_calibration),
        ]
        for label, callback in specs:
            b = QPushButton(label)
            b.clicked.connect(callback)
            buttons.addWidget(b)
        left.addLayout(buttons)

        self.status = QLabel("No video loaded")
        self.status.setStyleSheet("padding:8px;")
        left.addWidget(self.status)

        right.addWidget(QLabel("Precision Magnifier"))
        self.magnifier = MagnifierWidget()
        right.addWidget(self.magnifier)

        right.addWidget(QLabel("Calibration Points"))
        self.point_list = QListWidget()
        right.addWidget(self.point_list, stretch=1)

        outer.addLayout(left, stretch=1)
        outer.addLayout(right)

        self.setCentralWidget(root)

    def open_video(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Pickleball Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if not filename:
            return

        if self.capture is not None:
            self.capture.release()

        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            QMessageBox.critical(self, "Calibration Studio", "Unable to open video.")
            return

        self.capture = cap
        self.video_path = Path(filename)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.session.reset()

        self.slider.setEnabled(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, self.frame_count - 1))

        self.goto_frame(0)

    def goto_frame(self, frame_number):
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

        self._refresh()
        self._update_instruction()

    def _add_point(self, x, y):
        try:
            self.session.add_point(x, y)
        except ValueError:
            return
        self._refresh()
        self._update_instruction()
        self._update_point_list()

    def _move_point(self, index, x, y):
        self.session.move_point(index, x, y)
        self._refresh()
        self._update_point_list()

    def _cursor_moved(self, x, y):
        self.magnifier.update_from_frame(self.current_frame, x, y)

    def undo_point(self):
        self.session.undo()
        self._refresh()
        self._update_instruction()
        self._update_point_list()

    def reset_points(self):
        self.session.reset()
        self._refresh()
        self._update_instruction()
        self._update_point_list()

    def compute_calibration(self):
        if self.current_frame is None:
            return

        if not self.session.complete:
            QMessageBox.warning(self, "Calibration Studio", "Select all 8 points first.")
            return

        h, w = self.current_frame.shape[:2]

        try:
            result = self.session.build(
                name=self.video_path.stem if self.video_path else "court",
                image_size=(w, h),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Calibration Studio", f"Calibration failed:\n{exc}")
            return

        q = result.quality
        self.status.setText(
            f"Calibration: {q.status.value} | "
            f"Mean {q.metrics.mean_error_px:.2f}px | "
            f"Max {q.metrics.max_error_px:.2f}px | "
            f"RMS {q.metrics.rms_error_px:.2f}px | "
            f"Inliers {q.metrics.inlier_count}/{q.metrics.point_count}"
        )

        self._refresh()
        self._update_point_list()

        if q.status == CalibrationStatus.INVALID:
            worst = sorted(
                self.session.point_diagnostics(),
                key=lambda d: d.error_px,
                reverse=True,
            )
            text = "\n".join(
                f"Point {d.index+1}: {d.error_px:.2f}px"
                for d in worst[:4]
            )
            QMessageBox.warning(
                self,
                "Calibration INVALID",
                "Highest point errors:\n" + text,
            )

    def save_calibration(self):
        if self.session.result is None:
            QMessageBox.warning(self, "Calibration Studio", "Compute calibration first.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Calibration", "court.json", "JSON (*.json)"
        )
        if not filename:
            return

        self.session.result.profile.save(filename)
        QMessageBox.information(self, "Calibration Studio", f"Saved:\n{filename}")

    def _refresh(self):
        self.video.set_scene(
            self.current_frame,
            self.session.image_points,
            self.session.overlay_segments(),
            self.session.point_diagnostics(),
        )

    def _update_instruction(self):
        if self.current_frame is None:
            self.instruction.setText("Open a video to begin calibration.")
            return

        name = self.session.next_reference_name
        if name is None:
            self.instruction.setText(
                f"Frame {self.current_frame_number} — all points selected. "
                "Drag any point to refine, then Compute Calibration."
            )
        else:
            idx = len(self.session.image_points) + 1
            self.instruction.setText(
                f"Frame {self.current_frame_number} — Point {idx}/8: "
                f"{human_label(name)}"
            )

    def _update_point_list(self):
        self.point_list.clear()
        diagnostics = {d.index: d.error_px for d in self.session.point_diagnostics()}

        for i, name in enumerate(self.session.reference_names):
            if i < len(self.session.image_points):
                x, y = self.session.image_points[i]
                err = diagnostics.get(i)
                suffix = "" if err is None else f"  |  error {err:.2f}px"
                self.point_list.addItem(
                    f"{i+1}. {human_label(name)}  ({x:.1f}, {y:.1f}){suffix}"
                )
            else:
                self.point_list.addItem(
                    f"{i+1}. {human_label(name)}  — pending"
                )

    def closeEvent(self, event):
        if self.capture is not None:
            self.capture.release()
        event.accept()
