from __future__ import annotations

import time
import cv2

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QVBoxLayout,
    QWidget,
)

from linecaller.apps.live_audio import LiveAudioNotifier
from linecaller.apps.live_camera_worker import LiveCameraWorker
from linecaller.apps.live_controller import LiveMatchController
from linecaller.live.engine import LiveOfficiatingEngine
from linecaller.live.models import LiveDecision, LiveFramePacket
from linecaller.live.pipeline_factory import create_live_pipeline_adapter
from linecaller.product.hud_models import HUDMetric, StatusLevel
from linecaller.product.hud_policy import (
    confidence_metric,
    fps_metric,
    last_call_presentation,
    latency_metric,
    replay_metric,
    tracking_metric,
)
from linecaller.product.hud_widgets import ConfidenceCard, MetricCard, level_style
from linecaller.product.match_summary import MatchSummaryBuilder
from linecaller.product.replay_controller import ReplayPlaybackController
from linecaller.product.replay_models import ReplayState
from linecaller.product.replay_render import replay_zoom
from linecaller.product.replay_widget import ReplayOverlay


class LiveMatchWindow(QMainWindow):
    match_finished = Signal(object)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Open-LineCaller LIVE")
        self.resize(1450, 960)

        self.controller = LiveMatchController()
        self.controller.set_ready(calibration_valid=False)

        self.live_engine = LiveOfficiatingEngine()
        self.pipeline_adapter = create_live_pipeline_adapter()
        self.audio = LiveAudioNotifier()

        self.worker = None
        self.current_frame = None
        self.current_frame_number = 0
        self.last_ball = None

        self.summary_builder = MatchSummaryBuilder()

        self.replay_controller = ReplayPlaybackController(
            speed=0.25
        )

        self.replay_timer = QTimer(self)
        self.replay_timer.timeout.connect(self._replay_tick)

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

        self.end_btn = QPushButton("END MATCH")
        self.end_btn.setEnabled(False)

        self.calibration_btn = QPushButton("Calibration")
        self.replay_btn = QPushButton("Replay")
        self.replay_btn.setEnabled(False)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x"])
        self.speed_combo.currentTextChanged.connect(
            self._replay_speed_changed
        )

        self.dev_cb = QCheckBox("Developer Diagnostics")

        for widget in (
            QLabel("Camera:"),
            self.camera_combo,
            self.start_btn,
            self.stop_btn,
            self.end_btn,
            self.calibration_btn,
            self.replay_btn,
            QLabel("Replay Speed:"),
            self.speed_combo,
        ):
            top.addWidget(widget)

        top.addStretch(1)
        top.addWidget(self.dev_cb)
        outer.addLayout(top)

        self.preview = QLabel("Camera preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(540)
        self.preview.setStyleSheet("background:#111; color:#ddd;")
        outer.addWidget(self.preview, 1)

        self.replay_overlay = ReplayOverlay()
        outer.addWidget(self.replay_overlay)

        grid = QGridLayout()

        self.tracking_card = MetricCard("Tracking")
        self.fps_card = MetricCard("FPS")
        self.latency_card = MetricCard("Latency")
        self.confidence_card = ConfidenceCard()
        self.replay_card = MetricCard("Replay")
        self.calibration_card = MetricCard("Calibration")

        for col, card in enumerate([
            self.tracking_card,
            self.fps_card,
            self.latency_card,
            self.confidence_card,
            self.calibration_card,
            self.replay_card,
        ]):
            grid.addWidget(card, 0, col)

        outer.addLayout(grid)

        call_box = QGroupBox("LAST CALL")
        call_layout = QVBoxLayout(call_box)

        self.call_label = QLabel("-")
        self.call_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.call_label.setStyleSheet(
            "font-size:72px; font-weight:900;"
        )

        self.call_detail = QLabel("LIVE")
        self.call_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.call_detail.setStyleSheet("font-size:18px;")

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
        self.end_btn.clicked.connect(self.end_match)
        self.calibration_btn.clicked.connect(
            self.mark_calibration_valid
        )
        self.replay_btn.clicked.connect(self.show_replay)
        self.dev_cb.toggled.connect(
            self.dev_label.setVisible
        )

    def mark_calibration_valid(self):
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

        self.summary_builder = MatchSummaryBuilder()
        self.controller.start()

        camera_index = self.camera_combo.currentIndex()

        self.worker = LiveCameraWorker(camera_index)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.camera_error.connect(
            self._on_camera_error
        )
        self.worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.end_btn.setEnabled(True)
        self.call_detail.setText("LIVE")

    def stop_match(self):
        self.replay_timer.stop()
        self.replay_controller.finish()

        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None

        self.controller.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.end_btn.setEnabled(False)
        self._refresh_status()

    def end_match(self):
        camera_name = self.camera_combo.currentText()

        summary = self.summary_builder.build(
            camera_name=camera_name,
            calibration_mode="AUTO",
            replay_count=self.replay_controller.metrics.replay_count,
        )

        self.stop_match()
        self.match_finished.emit(summary)
        self.hide()

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

        self.summary_builder.record_runtime(
            fps=capture_fps,
            latency_ms=output.processing_ms,
            confidence=result.confidence,
            tracking_status=result.tracking_status,
        )

        if result.ball_x is not None and result.ball_y is not None:
            self.last_ball = (
                result.ball_x,
                result.ball_y,
            )
        else:
            self.last_ball = None

        if output.event is not None:
            evidence = self.live_engine.handle_event(
                output.event
            )

            if not evidence.duplicate_suppressed:
                self.controller.handle_evidence(evidence)
                self.summary_builder.record_call(
                    evidence.decision.value
                )

                if evidence.decision in (
                    LiveDecision.IN,
                    LiveDecision.OUT,
                ):
                    self.audio.announce(
                        evidence.decision.value
                    )

                if evidence.replay_requested:
                    self._begin_replay(
                        decision_frame=evidence.frame_number,
                        zoom_x=(
                            result.ball_x
                            if result.ball_x is not None
                            else None
                        ),
                        zoom_y=(
                            result.ball_y
                            if result.ball_y is not None
                            else None
                        ),
                    )

        if self.replay_controller.state == ReplayState.LIVE:
            self._render_frame(frame)

        self._refresh_status()

    def _begin_replay(
        self,
        *,
        decision_frame,
        zoom_x=None,
        zoom_y=None,
    ):
        started = self.replay_controller.begin(
            self.live_engine.last_replay,
            decision_frame=decision_frame,
            zoom_x=zoom_x,
            zoom_y=zoom_y,
        )

        if not started:
            return

        self.replay_btn.setEnabled(True)
        self.replay_overlay.update_from_model(
            self.replay_controller.view_model()
        )

        self.replay_timer.start(
            self._replay_interval_ms()
        )

    def _replay_interval_ms(self):
        base_fps = max(
            1.0,
            self.controller.state.fps or 30.0,
        )
        speed = self.replay_controller.speed

        return max(
            1,
            int(round(1000.0 / (base_fps * speed))),
        )

    def _replay_speed_changed(self, text):
        mapping = {
            "0.25x": 0.25,
            "0.5x": 0.5,
            "1x": 1.0,
        }

        self.replay_controller.set_speed(
            mapping[text]
        )

        if self.replay_timer.isActive():
            self.replay_timer.start(
                self._replay_interval_ms()
            )

    def _replay_tick(self):
        packet = self.replay_controller.advance()
        model = self.replay_controller.view_model()

        self.replay_overlay.update_from_model(
            model
        )

        if packet is not None:
            frame = replay_zoom(
                packet.frame.copy(),
                x=model.zoom_x,
                y=model.zoom_y,
                scale=2.0,
            )

            cv2.putText(
                frame,
                "REPLAY",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255,255,255),
                2,
                cv2.LINE_AA,
            )

            self._render_frame(
                frame,
                overlay_ball=False,
            )

        if self.replay_controller.state == ReplayState.RESUME:
            self.replay_timer.stop()
            self.replay_controller.finish()

            self.replay_overlay.update_from_model(
                self.replay_controller.view_model()
            )

            self.controller.state.replay_active = False
            self.call_detail.setText("LIVE")

            if self.current_frame is not None:
                self._render_frame(
                    self.current_frame
                )

    def _render_frame(
        self,
        frame,
        *,
        overlay_ball=True,
    ):
        display = frame.copy()

        if overlay_ball and self.last_ball is not None:
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
                (20,70),
                cv2.FONT_HERSHEY_SIMPLEX,
                .65,
                (255,255,255),
                2,
            )

        rgb = cv2.cvtColor(
            display,
            cv2.COLOR_BGR2RGB,
        )

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
        QMessageBox.critical(
            self,
            "Camera",
            message,
        )
        self.stop_match()

    def show_replay(self):
        if not self.live_engine.last_replay:
            QMessageBox.information(
                self,
                "Replay",
                "No replay is currently available.",
            )
            return

        self._begin_replay(
            decision_frame=self.current_frame_number,
            zoom_x=(
                self.last_ball[0]
                if self.last_ball is not None
                else None
            ),
            zoom_y=(
                self.last_ball[1]
                if self.last_ball is not None
                else None
            ),
        )

    def _refresh_status(self):
        s = self.controller.state

        self.tracking_card.set_metric(
            tracking_metric(
                s.tracking_status
            )
        )
        self.fps_card.set_metric(
            fps_metric(s.fps)
        )
        self.latency_card.set_metric(
            latency_metric(
                s.latency_ms
            )
        )
        self.confidence_card.set_metric(
            confidence_metric(
                s.confidence
            )
        )
        self.replay_card.set_metric(
            replay_metric(
                bool(
                    self.live_engine.last_replay
                )
            )
        )

        calibration_ok = (
            s.calibration_status == "VALID"
        )

        self.calibration_card.set_metric(
            HUDMetric(
                "Calibration",
                s.calibration_status,
                StatusLevel.OK
                if calibration_ok
                else StatusLevel.ERROR,
                "Ready"
                if calibration_ok
                else "Calibration required",
            )
        )

        presentation = last_call_presentation(
            s.last_call
        )

        self.call_label.setText(
            presentation.call
        )
        self.call_label.setStyleSheet(
            f"font-size:72px; font-weight:900; "
            f"{level_style(presentation.level)}"
        )

        if (
            self.replay_controller.state
            != ReplayState.LIVE
        ):
            self.call_detail.setText(
                "INSTANT REPLAY"
            )
        else:
            self.call_detail.setText(
                presentation.detail
            )

        self.dev_label.setText(
            f"run_state={s.run_state.value} | "
            f"frame={self.current_frame_number} | "
            f"tracking={s.tracking_status} | "
            f"replay_state={self.replay_controller.state.value} | "
            f"buffer={len(self.live_engine.replay_buffer)} | "
            f"replays={self.replay_controller.metrics.replay_count}"
        )

    def closeEvent(self, event):
        self.stop_match()
        event.accept()
