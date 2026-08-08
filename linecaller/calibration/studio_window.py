from __future__ import annotations

from pathlib import Path
import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QListWidget, QMainWindow,
    QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget
)

from linecaller.autocalibration.engine_v2 import MultiHypothesisAutoCalibrationEngine
from linecaller.calibration.labels import human_label
from linecaller.calibration.quality import CalibrationStatus
from linecaller.calibration.session import CalibrationSession
from linecaller.calibration.smart_adapter import SmartCalibrationAdapter
from linecaller.calibration.studio_widget import CalibrationVideoWidget
from linecaller.calibration.magnifier_widget import MagnifierWidget


class CalibrationStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open-LineCaller — Smart Calibration Studio")
        self.resize(1500, 920)

        self.capture = None
        self.video_path = None
        self.frame_count = 0
        self.current_frame_number = 0
        self.current_frame = None

        self.session = CalibrationSession()
        self.auto_engine = MultiHypothesisAutoCalibrationEngine()
        self.smart_adapter = SmartCalibrationAdapter()

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

        actions = [
            ("◀ Previous Frame", lambda: self.goto_frame(self.current_frame_number - 1)),
            ("Next Frame ▶", lambda: self.goto_frame(self.current_frame_number + 1)),
            ("AUTO CALIBRATE", self.auto_calibrate),
            ("Undo Point", self.undo_point),
            ("Reset Points", self.reset_points),
            ("Compute Calibration", self.compute_calibration),
            ("Save Calibration...", self.save_calibration),
        ]

        for label, callback in actions:
            button = QPushButton(label)
            button.clicked.connect(callback)
            if label == "AUTO CALIBRATE":
                button.setStyleSheet("font-weight:700; padding:8px;")
            buttons.addWidget(button)

        left.addLayout(buttons)

        self.status = QLabel("No video loaded")
        self.status.setStyleSheet("padding:8px;")
        left.addWidget(self.status)

        self.auto_status = QLabel("Auto-calibration: not run")
        self.auto_status.setStyleSheet("padding:8px; border:1px solid #555;")
        left.addWidget(self.auto_status)

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

        self.auto_status.setText("Auto-calibration: not run")
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

    def auto_calibrate(self):
        if self.current_frame is None:
            QMessageBox.warning(
                self,
                "Auto Calibration",
                "Open a video and select a frame first.",
            )
            return

        try:
            result = self.auto_engine.propose(self.current_frame)
            ranked = result["ranked"]
            prefill = self.smart_adapter.from_ranked(ranked)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Auto Calibration",
                f"Auto-calibration failed:\n{exc}",
            )
            return

        best_score = ranked.best.score if ranked.best else 0.0

        self.auto_status.setText(
            f"Auto-calibration: {ranked.disposition.value} | "
            f"Best score {best_score:.3f} | "
            f"Margin {ranked.confidence_margin:.3f} | "
            f"Hypotheses {result['hypothesis_count']} | "
            f"Lines {result['line_count']}"
        )

        if prefill.outer_corners:
            self.session.prefill_outer_corners(prefill.outer_corners)

        self._refresh()
        self._update_instruction()
        self._update_point_list()

        if ranked.disposition.value == "REJECT":
            reasons = "\n".join(ranked.reasons) or prefill.reason
            QMessageBox.warning(
                self,
                "Auto Calibration — REJECT",
                "The automatic proposal is not reliable enough.\n\n"
                f"{reasons}\n\n"
                "You may choose another frame or calibrate manually.",
            )

        elif ranked.disposition.value == "AUTO_REVIEW":
            QMessageBox.information(
                self,
                "Auto Calibration — REVIEW",
                "Outer corners were proposed.\n\n"
                "Inspect and drag them carefully, then add the four Kitchen points.",
            )

        else:
            QMessageBox.information(
                self,
                "Auto Calibration — ACCEPT",
                "Strong outer-court proposal loaded.\n\n"
                "Inspect it, then add the four Kitchen points before computing calibration.",
            )

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
        self.auto_status.setText("Auto-calibration: reset")

    def compute_calibration(self):
        if self.current_frame is None:
            return

        if not self.session.complete:
            QMessageBox.warning(
                self,
                "Calibration Studio",
                f"Need all 8 points. Current: {len(self.session.image_points)}/8",
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
            QMessageBox.warning(
                self,
                "Calibration Studio",
                "Compute calibration first.",
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
                f"Frame {self.current_frame_number} — all 8 points selected. "
                "Refine with drag, then Compute Calibration."
            )
        else:
            idx = len(self.session.image_points) + 1

            if idx <= 4:
                self.instruction.setText(
                    f"Frame {self.current_frame_number} — Point {idx}/8: "
                    f"{human_label(name)} "
                    "(or use AUTO CALIBRATE to prefill outer corners)"
                )
            else:
                self.instruction.setText(
                    f"Frame {self.current_frame_number} — Point {idx}/8: "
                    f"{human_label(name)}"
                )

    def _update_point_list(self):
        self.point_list.clear()

        diagnostics = {
            d.index: d.error_px
            for d in self.session.point_diagnostics()
        }

        for i, name in enumerate(self.session.reference_names):
            if i < len(self.session.image_points):
                x, y = self.session.image_points[i]
                err = diagnostics.get(i)
                suffix = "" if err is None else f" | error {err:.2f}px"

                self.point_list.addItem(
                    f"{i+1}. {human_label(name)} "
                    f"({x:.1f}, {y:.1f}){suffix}"
                )
            else:
                self.point_list.addItem(
                    f"{i+1}. {human_label(name)} — pending"
                )

    def closeEvent(self, event):
        if self.capture is not None:
            self.capture.release()
        event.accept()
