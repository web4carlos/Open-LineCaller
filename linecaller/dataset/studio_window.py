from pathlib import Path
import json
import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSlider,
    QVBoxLayout, QWidget
)

from linecaller.dataset.assisted_controller import AssistedAnnotationController
from linecaller.dataset.models import BallBox
from linecaller.dataset.studio_session import DatasetStudioSession
from linecaller.dataset.studio_widget import AnnotationVideoWidget
from linecaller.dataset.validation import validate_clip
from linecaller.dataset.proposal_engine_factory import create_dataset_studio_proposal_engine


class DatasetStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open-LineCaller — AI-Assisted Dataset Studio")
        self.resize(1500, 920)

        self.capture = None
        self.video_path = None
        self.current_frame = None
        self.current_frame_number = 0
        self.frame_count = 0
        self.fps = 0.0
        self.session = None
        self.controller = None
        self.annotation_path = None
        self.cx = self.cy = 0.0

        self._menu()
        self._ui()
        self._shortcuts()
        self._status()

    def _menu(self):
        menu = self.menuBar().addMenu("&File")

        open_action = QAction("Open Video...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_video)
        menu.addAction(open_action)

        save_action = QAction("Save Annotation...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_annotation)
        menu.addAction(save_action)

    def _ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)

        self.video = AnnotationVideoWidget()
        self.video.box_changed.connect(self._set_box)
        self.video.cursor_frame_position.connect(self._cursor)
        layout.addWidget(self.video, 1)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.goto_frame)
        layout.addWidget(self.slider)

        row = QHBoxLayout()

        actions = [
            ("◀ Previous", lambda: self.goto_frame(self.current_frame_number - 1)),
            ("Next ▶", lambda: self.goto_frame(self.current_frame_number + 1)),
            ("Propose", self.request_proposal),
            ("Accept", self.accept_proposal),
            ("Reject", self.reject_proposal),
            ("Delete Box", self.delete_box),
            ("Mark Bounce", self.toggle_bounce),
        ]

        for text, callback in actions:
            button = QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)

        self.auto_advance = QCheckBox("Auto Advance")
        self.auto_advance.setChecked(True)

        self.visible = QCheckBox("Visible")
        self.visible.setChecked(True)

        self.occluded = QCheckBox("Occluded")

        self.decision = QComboBox()
        self.decision.addItems(["", "IN", "OUT", "REVIEW"])

        row.addWidget(self.auto_advance)
        row.addWidget(self.visible)
        row.addWidget(self.occluded)
        row.addWidget(QLabel("Decision:"))
        row.addWidget(self.decision)

        layout.addLayout(row)

        self.status = QLabel()
        self.metrics_label = QLabel()

        layout.addWidget(self.status)
        layout.addWidget(self.metrics_label)

        self.setCentralWidget(root)

    def _shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self.goto_frame(self.current_frame_number - 1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self.goto_frame(self.current_frame_number + 1))
        QShortcut(QKeySequence("P"), self, activated=self.request_proposal)
        QShortcut(QKeySequence("A"), self, activated=self.accept_proposal)
        QShortcut(QKeySequence("R"), self, activated=self.reject_proposal)
        QShortcut(QKeySequence("Space"), self, activated=self.accept_proposal)

    def open_video(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Pickleball Clip",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
        )

        if not filename:
            return

        if self.capture:
            self.capture.release()

        cap = cv2.VideoCapture(filename)

        if not cap.isOpened():
            QMessageBox.critical(self, "Dataset Studio", "Unable to open video.")
            return

        self.capture = cap
        self.video_path = Path(filename)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.session = DatasetStudioSession(
            clip_id=self.video_path.stem,
            source_video=str(self.video_path),
            width=width,
            height=height,
            fps=self.fps,
            frame_count=self.frame_count,
        )

        self.controller = AssistedAnnotationController(
            self.session,
            engine=create_dataset_studio_proposal_engine(),
        )

        self.slider.setEnabled(True)
        self.slider.setRange(0, max(0, self.frame_count - 1))

        self.goto_frame(0)

    def goto_frame(self, frame_number):
        if not self.capture:
            return

        frame_number = max(
            0,
            min(frame_number, max(0, self.frame_count - 1)),
        )

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = self.capture.read()

        if not ok:
            return

        self.current_frame_number = frame_number
        self.current_frame = frame

        self.slider.blockSignals(True)
        self.slider.setValue(frame_number)
        self.slider.blockSignals(False)

        if self.controller:
            self.controller.clear_proposal()

        self._refresh()

    def request_proposal(self):
        if not self.controller or self.current_frame is None:
            return

        self.controller.request_proposal(
            self.current_frame_number,
            self.current_frame,
        )
        self._refresh()

    def accept_proposal(self):
        if not self.controller:
            return

        box = self.controller.accept_current(
            visible=self.visible.isChecked(),
            occluded=self.occluded.isChecked(),
        )

        if box is not None and self.auto_advance.isChecked():
            self.controller.metrics.auto_advanced += 1
            self.goto_frame(self.current_frame_number + 1)
            self.request_proposal()
        else:
            self._refresh()

    def reject_proposal(self):
        if not self.controller:
            return

        rejected = self.controller.reject_current()

        if rejected and self.auto_advance.isChecked():
            self.controller.metrics.auto_advanced += 1
            self.goto_frame(self.current_frame_number + 1)
            self.request_proposal()
        else:
            self._refresh()

    def _set_box(self, x, y, w, h):
        if not self.session:
            return

        box = BallBox(x, y, w, h)

        if self.controller and self.controller.current_proposal is not None:
            self.controller.adjust_current(
                self.current_frame_number,
                box,
                visible=self.visible.isChecked(),
                occluded=self.occluded.isChecked(),
            )
        else:
            self.session.set_ball_box(
                self.current_frame_number,
                box,
                visible=self.visible.isChecked(),
                occluded=self.occluded.isChecked(),
            )

        self._refresh()

    def delete_box(self):
        if self.session:
            self.session.remove_ball_box(self.current_frame_number)
            self._refresh()

    def toggle_bounce(self):
        if not self.session:
            return

        frame = self.current_frame_number
        existing = self.session.bounce_annotation(frame)

        if existing:
            self.session.toggle_bounce(
                frame,
                x=existing.x,
                y=existing.y,
                decision=existing.decision,
            )
        else:
            ann = self.session.frame_annotation(frame)

            if ann and ann.ball:
                x = ann.ball.x + ann.ball.width / 2.0
                y = ann.ball.y + ann.ball.height
            else:
                x, y = self.cx, self.cy

            self.session.toggle_bounce(
                frame,
                x=x,
                y=y,
                decision=self.decision.currentText() or None,
            )

        self._refresh()

    def _cursor(self, x, y):
        self.cx = x
        self.cy = y

    def _refresh(self):
        if not self.session:
            return

        ann = self.session.frame_annotation(self.current_frame_number)
        bounce = self.session.bounce_annotation(self.current_frame_number)
        proposal = self.controller.current_proposal if self.controller else None

        self.video.set_scene(
            self.current_frame,
            box=ann.ball if ann else None,
            bounce=bounce,
            proposal=proposal,
        )

        self._status()

    def _status(self):
        if not self.session:
            self.status.setText("No clip loaded")
            self.metrics_label.setText("")
            return

        annotation = self.session.to_annotation()
        boxes = sum(1 for f in annotation.frames if f.ball)

        self.status.setText(
            f"Clip: {annotation.clip_id} | "
            f"Frame {self.current_frame_number + 1}/{annotation.frame_count} | "
            f"Boxes: {boxes} | "
            f"Bounces: {len(annotation.bounces)}"
        )

        if self.controller:
            m = self.controller.metrics
            self.metrics_label.setText(
                f"Proposals requested: {m.proposals_requested} | "
                f"Shown: {m.proposals_shown} | "
                f"Accepted: {m.accepted} | "
                f"Adjusted: {m.adjusted} | "
                f"Rejected: {m.rejected} | "
                f"Acceptance rate: {m.acceptance_rate:.1%}"
            )

    def save_annotation(self):
        if not self.session:
            return

        annotation = self.session.to_annotation()
        issues = validate_clip(annotation)
        errors = [i for i in issues if i.severity == "ERROR"]

        if errors:
            QMessageBox.critical(
                self,
                "Validation failed",
                "\n".join(i.message for i in errors[:10]),
            )
            return

        if not self.annotation_path:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Annotation",
                f"{annotation.clip_id}.json",
                "JSON (*.json)",
            )

            if not filename:
                return

            self.annotation_path = Path(filename)

        self.annotation_path.write_text(
            json.dumps(annotation.to_dict(), indent=2),
            encoding="utf-8",
        )

        QMessageBox.information(
            self,
            "Dataset Studio",
            f"Saved:\n{self.annotation_path}",
        )

    def closeEvent(self, event):
        if self.capture:
            self.capture.release()
        event.accept()
