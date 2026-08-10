from __future__ import annotations

import time
from pathlib import Path
import cv2

from PySide6.QtCore import Qt,QTimer,Signal
from PySide6.QtWidgets import (
    QFileDialog,QFrame,QHBoxLayout,QLabel,QMessageBox,
    QPushButton,QSlider,QVBoxLayout,QWidget
)

from linecaller.live.pipeline_factory import create_live_pipeline_adapter
from .session import MatchSessionStats
from .video_widget import VideoSurface


class MatchWorkspace(QWidget):
    back_requested=Signal()

    def __init__(self,*,mode="video",camera_index=0):
        super().__init__()
        self.mode=mode
        self.camera_index=int(camera_index)
        self.capture=None
        self.adapter=None
        self.source_path=None
        self.frame_number=0
        self.frame_count=0
        self.fps=30.0
        self.running=False
        self.stats=MatchSessionStats()

        self.timer=QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._build_ui()

    def _card(self):
        f=QFrame(); f.setObjectName("PanelCard"); return f

    def _build_ui(self):
        outer=QVBoxLayout(self); outer.setContentsMargins(20,18,20,18); outer.setSpacing(14)

        top=QHBoxLayout()
        back=QPushButton("← Home"); back.clicked.connect(self.back_requested.emit); top.addWidget(back)
        title=QLabel("Analyze Match" if self.mode=="video" else "Live Match")
        title.setObjectName("SectionTitle"); top.addWidget(title); top.addStretch(1)
        self.source_label=QLabel("No video selected" if self.mode=="video" else f"Camera {self.camera_index}")
        self.source_label.setObjectName("Muted"); top.addWidget(self.source_label)
        outer.addLayout(top)

        body=QHBoxLayout(); body.setSpacing(14)

        left_card=self._card()
        left=QVBoxLayout(left_card); left.setContentsMargins(14,14,14,14)
        self.video=VideoSurface("Choose a video to analyze" if self.mode=="video" else "Press Start Live")
        left.addWidget(self.video,1)

        self.seek=QSlider(Qt.Orientation.Horizontal)
        self.seek.setEnabled(False)
        self.seek.sliderMoved.connect(self._seek_to)
        if self.mode=="video":
            left.addWidget(self.seek)

        controls=QHBoxLayout()
        if self.mode=="video":
            open_btn=QPushButton("Open Video"); open_btn.setObjectName("PrimaryButton")
            open_btn.clicked.connect(self.open_video); controls.addWidget(open_btn)

        self.start_btn=QPushButton("Start" if self.mode=="video" else "Start Live")
        self.start_btn.setObjectName("PrimaryButton"); self.start_btn.clicked.connect(self.start)
        controls.addWidget(self.start_btn)

        pause_btn=QPushButton("Pause"); pause_btn.clicked.connect(self.pause); controls.addWidget(pause_btn)
        stop_btn=QPushButton("Stop"); stop_btn.setObjectName("DangerButton"); stop_btn.clicked.connect(self.stop)
        controls.addWidget(stop_btn); controls.addStretch(1)
        left.addLayout(controls)
        body.addWidget(left_card,4)

        right=self._card()
        rl=QVBoxLayout(right); rl.setContentsMargins(18,18,18,18); rl.setSpacing(12)
        kicker=QLabel("OFFICIATING"); kicker.setObjectName("Muted"); rl.addWidget(kicker)
        self.decision=QLabel("—"); self.decision.setObjectName("Decision"); rl.addWidget(self.decision)
        self.confidence=QLabel("Confidence  —"); self.confidence.setObjectName("Muted"); rl.addWidget(self.confidence)
        rl.addSpacing(12)

        self.tracking_label=QLabel()
        self.fps_label=QLabel()
        self.latency_label=QLabel()
        self.events_label=QLabel()
        self.summary_label=QLabel(); self.summary_label.setWordWrap(True)

        for widget in [self.tracking_label,self.fps_label,self.latency_label,self.events_label,self.summary_label]:
            rl.addWidget(widget)

        rl.addStretch(1)
        reset=QPushButton("Reset Session"); reset.clicked.connect(self.reset_session); rl.addWidget(reset)
        body.addWidget(right,1)

        outer.addLayout(body,1)
        self._refresh_metrics(tracking="SEARCHING",processing_ms=0.0)

    def _new_adapter(self):
        self.adapter=create_live_pipeline_adapter(minimum_call_confidence=0.98)

    def open_video(self):
        path,_=QFileDialog.getOpenFileName(
            self,"Open match video","","Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if not path:
            return

        self.stop()
        cap=cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self,"Open-LineCaller","The selected video could not be opened.")
            return

        self.capture=cap
        self.source_path=Path(path)
        self.source_label.setText(self.source_path.name)
        self.frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.seek.setEnabled(True)
        self.seek.setRange(0,max(0,self.frame_count-1))
        self.frame_number=0
        self.reset_session()

        visible_frame=None
        visible_number=0
        for n in range(min(121,max(1,self.frame_count))):
            cap.set(cv2.CAP_PROP_POS_FRAMES,n)
            ok,frame=cap.read()
            if not ok:
                break
            if float(frame.mean())>2.0:
                visible_frame=frame
                visible_number=n
                break

        if visible_frame is not None:
            self.frame_number=visible_number
            self.video.show_bgr(visible_frame)
            self.seek.setValue(visible_number)

        cap.set(cv2.CAP_PROP_POS_FRAMES,self.frame_number)

    def start(self):
        if self.mode=="live":
            if self.capture is None:
                self.capture=cv2.VideoCapture(self.camera_index)
                if not self.capture.isOpened():
                    self.capture.release()
                    self.capture=None
                    QMessageBox.critical(self,"Open-LineCaller",f"Camera {self.camera_index} could not be opened.")
                    return
                self.fps=float(self.capture.get(cv2.CAP_PROP_FPS)) or 30.0
                self.source_label.setText(f"Camera {self.camera_index} • LIVE")
        elif self.capture is None:
            QMessageBox.information(self,"Open-LineCaller","Open a video first.")
            return

        if self.adapter is None:
            self._new_adapter()

        self.running=True
        interval=1 if self.mode=="live" else max(1,int(round(1000.0/max(1.0,self.fps))))
        self.timer.start(interval)

    def pause(self):
        self.running=False
        self.timer.stop()

    def stop(self):
        self.running=False
        self.timer.stop()
        if self.mode=="live" and self.capture is not None:
            self.capture.release()
            self.capture=None
            self.source_label.setText(f"Camera {self.camera_index}")

    def reset_session(self):
        self.stats.reset()
        self.adapter=None
        self.decision.setText("—")
        self.confidence.setText("Confidence  —")
        self._refresh_metrics(tracking="SEARCHING",processing_ms=0.0)

    def _seek_to(self,frame_number):
        if self.capture is None or self.mode!="video":
            return
        self.pause()
        self.frame_number=int(frame_number)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES,self.frame_number)
        ok,frame=self.capture.read()
        if ok:
            self.video.show_bgr(frame)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES,self.frame_number)
        self.adapter=None

    def _tick(self):
        if not self.running or self.capture is None:
            return

        ok,frame=self.capture.read()
        if not ok:
            self.pause()
            return

        if self.adapter is None:
            self._new_adapter()

        timestamp=time.perf_counter() if self.mode=="live" else self.frame_number/max(1.0,self.fps)

        try:
            output=self.adapter.process_frame(
                frame_number=self.frame_number,
                frame=frame,
                timestamp=timestamp,
            )
        except Exception as exc:
            self.pause()
            QMessageBox.critical(self,"Artificial Vision",f"Processing stopped:\n{exc}")
            return

        self.stats.frames += 1
        self.stats.last_processing_ms=float(output.processing_ms)
        result=output.result

        if result.ball_x is not None and result.ball_y is not None:
            cv2.circle(frame,(int(result.ball_x),int(result.ball_y)),10,(0,255,0),2)

        if output.event is not None:
            decision=output.event.decision.value
            self.stats.record_event(decision)
            self.decision.setText(decision)
            self.confidence.setText(f"Confidence  {output.event.confidence*100:.1f}%")

        self._refresh_metrics(tracking=result.tracking_status,processing_ms=output.processing_ms)
        self.video.show_bgr(frame)

        if self.mode=="video":
            self.seek.blockSignals(True)
            self.seek.setValue(min(self.frame_number,self.seek.maximum()))
            self.seek.blockSignals(False)

        self.frame_number += 1

    def _refresh_metrics(self,*,tracking,processing_ms):
        effective_fps=1000.0/processing_ms if processing_ms and processing_ms>0 else 0.0
        self.tracking_label.setText(f"Tracking\n{tracking}")
        self.fps_label.setText(f"Vision FPS\n{effective_fps:.1f}")
        self.latency_label.setText(f"Processing\n{processing_ms:.1f} ms")
        self.events_label.setText(f"Events\n{self.stats.events}")
        self.summary_label.setText(
            f"Calls\nIN {self.stats.calls_in}   OUT {self.stats.calls_out}   REVIEW {self.stats.calls_review}"
        )

    def close_source(self):
        self.pause()
        if self.capture is not None:
            self.capture.release()
            self.capture=None
