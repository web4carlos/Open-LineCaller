from __future__ import annotations

import time
import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from linecaller.apps.live_audio import LiveAudioNotifier
from linecaller.apps.live_camera_worker import LiveCameraWorker
from linecaller.apps.live_controller import LiveMatchController
from linecaller.live.engine import LiveOfficiatingEngine
from linecaller.live.models import LiveDecision, LiveFramePacket
from linecaller.live.pipeline_factory import create_live_pipeline_adapter


class LiveMatchWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Open-LineCaller LIVE")
        self.resize(1400, 900)

        self.controller = LiveMatchController()
        self.controller.set_ready(calibration_valid=False)

        self.live_engine = LiveOfficiatingEngine()
        self.pipeline_adapter = create_live_pipeline_adapter()
        self.audio = LiveAudioNotifier()

        self.worker = None
        self.current_frame = None
        self.current_frame_number = 0
        self.last_ball = None

        self._build_ui()
        self._refresh_status()

    def _build_ui(self):
        root = QWidget()
        outer = QVBoxLayout(root)

        top = QHBoxLayout()

        self.camera_combo = QComboBox()
        self.camera_combo.addItems(
            ["Camera 0", "Camera 1", "Camera 2", "Camera 3"]
        )

        self.start_btn = QPushButton("START MATCH")
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setEnabled(False)

        self.calibration_btn = QPushButton("Calibration")
        self.replay_btn = QPushButton("Replay")
        self.replay_btn.setEnabled(False)
        self.dev_cb = QCheckBox("Developer Diagnostics")

        for widget in (
            QLabel("Camera:"),
            self.camera_combo,
            self.start_btn,
            self.stop_btn,
            self.calibration_btn,
            self.replay_btn,
        ):
            top.addWidget(widget)

        top.addStretch(1)
        top.addWidget(self.dev_cb)
        outer.addLayout(top)

        self.preview = QLabel("Camera preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(560)
        self.preview.setStyleSheet("background:#111; color:#ddd;")
        outer.addWidget(self.preview, 1)

        grid = QGridLayout()

        self.tracking_value = QLabel("SEARCHING")
        self.calibration_value = QLabel("REQUIRED")
        self.fps_value = QLabel("0.0")
        self.latency_value = QLabel("0.0 ms")
        self.conf_value = QLabel("0.0%")

        for col, (name, widget) in enumerate([
            ("Tracking", self.tracking_value),
            ("Calibration", self.calibration_value),
            ("FPS", self.fps_value),
            ("Latency", self.latency_value),
            ("Confidence", self.conf_value),
        ]):
            box = QGroupBox(name)
            layout = QVBoxLayout(box)
            widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            widget.setStyleSheet("font-size:22px; font-weight:700;")
            layout.addWidget(widget)
            grid.addWidget(box, 0, col)

        outer.addLayout(grid)

        call_box = QGroupBox("LAST CALL")
        call_layout = QVBoxLayout(call_box)

        self.call_label = QLabel("-")
        self.call_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.call_label.setStyleSheet("font-size:72px; font-weight:900;")

        self.call_detail = QLabel("Waiting for match")
        self.call_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)

        call_layout.addWidget(self.call_label)
        call_layout.addWidget(self.call_detail)
        outer.addWidget(call_box)

        self.dev_label = QLabel("")
        self.dev_label.setVisible(False)
        self.dev_label.setStyleSheet(
            "font-family:Consolas; background:#222; color:#ddd; padding:8px;"
        )
        outer.addWidget(self.dev_label)

        self.setCentralWidget(root)

        self.start_btn.clicked.connect(self.start_match)
        self.stop_btn.clicked.connect(self.stop_match)
        self.calibration_btn.clicked.connect(self.mark_calibration_valid)
        self.replay_btn.clicked.connect(self.show_replay)
        self.dev_cb.toggled.connect(self.dev_label.setVisible)

    def mark_calibration_valid(self):
        # CP-0018 will wire the real Auto -> Assisted -> Manual flow.
        self.controller.set_ready(calibration_valid=True)
        self._refresh_status()

    def start_match(self):
        if self.controller.state.calibration_status != "VALID":
            QMessageBox.warning(
                self,
                "Calibration required",
                "Run Auto / Assisted / Manual calibration before starting.",
            )
            return

        self.controller.start()
        camera_index = self.camera_combo.currentIndex()

        self.worker = LiveCameraWorker(camera_index)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.camera_error.connect(self._on_camera_error)
        self.worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.call_detail.setText("LIVE")

    def stop_match(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None

        self.controller.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._refresh_status()

    def _on_frame(self, frame, capture_fps, frame_number):
        self.current_frame = frame
        self.current_frame_number = frame_number

        now = time.perf_counter()

        output = self.pipeline_adapter.process_frame(
            frame_number=frame_number,
            frame=frame,
            timestamp=now,
        )

        result = output.result

        self.live_engine.ingest_frame(
            LiveFramePacket(
                frame_number=frame_number,
                captured_at=now,
                frame=frame.copy(),
            ),
            processing_ms=output.processing_ms,
        )

        self.controller.update_runtime(
            fps=capture_fps,
            latency_ms=output.processing_ms,
            tracking_status=result.tracking_status,
            confidence=result.confidence,
        )

        if result.ball_x is not None and result.ball_y is not None:
            self.last_ball = (result.ball_x, result.ball_y)
        else:
            self.last_ball = None

        if output.event is not None:
            evidence = self.live_engine.handle_event(output.event)

            if not evidence.duplicate_suppressed:
                self.controller.handle_evidence(evidence)

                if evidence.decision in (
                    LiveDecision.IN,
                    LiveDecision.OUT,
                ):
                    self.audio.announce(evidence.decision.value)

                self.replay_btn.setEnabled(
                    evidence.replay_requested
                )

        self._render_frame(frame)
        self._refresh_status()

    def _render_frame(self, frame):
        display = frame.copy()

        if self.last_ball is not None:
            x, y = self.last_ball
            cv2.circle(
                display,
                (int(x), int(y)),
                8,
                (0,255,0),
                2,
            )

        if self.dev_cb.isChecked():
            cv2.putText(
                display,
                f"Frame {self.current_frame_number}",
                (20,35),
                cv2.FONT_HERSHEY_SIMPLEX,
                .7,
                (255,255,255),
                2,
            )

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        image = QImage(
            rgb.data,
            w,
            h,
            ch*w,
            QImage.Format.Format_RGB888,
        ).copy()

        self.preview.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _on_camera_error(self, message):
        QMessageBox.critical(self, "Camera", message)
        self.stop_match()

    def show_replay(self):
        if not self.live_engine.last_replay:
            QMessageBox.information(
                self,
                "Replay",
                "No replay is currently available.",
            )
            return

        QMessageBox.information(
            self,
            "Replay",
            f"Replay ready: {len(self.live_engine.last_replay)} buffered frames.",
        )

    def _refresh_status(self):
        s = self.controller.state

        self.tracking_value.setText(s.tracking_status)
        self.calibration_value.setText(s.calibration_status)
        self.fps_value.setText(f"{s.fps:.1f}")
        self.latency_value.setText(f"{s.latency_ms:.1f} ms")
        self.conf_value.setText(f"{s.confidence*100:.1f}%")
        self.call_label.setText(s.last_call)

        if s.replay_active:
            self.call_detail.setText("INSTANT REPLAY")
        elif s.run_state.value == "LIVE":
            self.call_detail.setText("LIVE")

        self.dev_label.setText(
            f"run_state={s.run_state.value} | "
            f"frame={self.current_frame_number} | "
            f"replay={s.replay_active} | "
            f"buffer={len(self.live_engine.replay_buffer)} | "
            f"adapter={type(self.pipeline_adapter.perception).__name__}"
        )

    def closeEvent(self, event):
        self.stop_match()
        event.accept()
