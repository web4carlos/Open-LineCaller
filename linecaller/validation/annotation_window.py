import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,QHBoxLayout,QLabel,QMainWindow,QMessageBox,
    QProgressBar,QPushButton,QVBoxLayout,QWidget
)
from .annotation_models import AnnotationLabel
from .annotation_render import zoom_around_candidate
from .annotation_session import AnnotationSession
from .candidate_scanner import CandidateScanner
from .io import save_truth_jsonl

class SmartAnnotationStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LineCaller Artificial Vision — Smart Event Annotation Studio")
        self.resize(1400,900)
        self.session=AnnotationSession()
        self.capture=None
        self._build_ui()
        self._install_shortcuts()

    def _build_ui(self):
        root=QWidget(); outer=QVBoxLayout(root)
        top=QHBoxLayout()
        for text,handler in [
            ("Open Video",self.open_video),
            ("Scan Candidates",self.scan_candidates),
            ("Save Ground Truth",self.save_ground_truth),
        ]:
            b=QPushButton(text); b.clicked.connect(handler); top.addWidget(b)
        top.addStretch(1); outer.addLayout(top)

        self.preview=QLabel("Open a pickleball match video")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(560)
        self.preview.setStyleSheet("background:#111;color:#ddd;")
        outer.addWidget(self.preview,1)

        self.event_label=QLabel("No candidates loaded")
        self.event_label.setStyleSheet("font-size:18px;font-weight:700;")
        self.meta_label=QLabel(""); self.meta_label.setWordWrap(True)
        outer.addWidget(self.event_label); outer.addWidget(self.meta_label)

        row=QHBoxLayout()
        for text,handler in [
            ("← Previous",self.previous_event),
            ("IN [I]",lambda:self.label_current(AnnotationLabel.IN)),
            ("OUT [O]",lambda:self.label_current(AnnotationLabel.OUT)),
            ("SKIP [S]",lambda:self.label_current(AnnotationLabel.SKIP)),
            ("Next →",self.next_event),
        ]:
            b=QPushButton(text); b.clicked.connect(handler); row.addWidget(b)
        outer.addLayout(row)

        self.progress=QProgressBar(); self.progress.setRange(0,100)
        self.status=QLabel("Ready")
        outer.addWidget(self.progress); outer.addWidget(self.status)
        self.setCentralWidget(root)

    def _install_shortcuts(self):
        for key,handler in [
            ("I",lambda:self.label_current(AnnotationLabel.IN)),
            ("O",lambda:self.label_current(AnnotationLabel.OUT)),
            ("S",lambda:self.label_current(AnnotationLabel.SKIP)),
            ("Left",self.previous_event),
            ("Right",self.next_event),
        ]:
            a=QAction(self); a.setShortcut(QKeySequence(key)); a.triggered.connect(handler); self.addAction(a)

    def open_video(self):
        filename,_=QFileDialog.getOpenFileName(
            self,"Open pickleball video","",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if not filename: return
        if self.capture: self.capture.release()
        self.capture=cv2.VideoCapture(filename)
        if not self.capture.isOpened():
            QMessageBox.critical(self,"Smart Annotation Studio","Unable to open video.")
            self.capture=None; return
        self.session.set_video(filename)
        first_visible=0
        for n in range(121):
            self.capture.set(cv2.CAP_PROP_POS_FRAMES,n)
            ok,frame=self.capture.read()
            if not ok: break
            if float(frame.mean()) > 2.0:
                first_visible=n; break
        self._show_frame(first_visible,None)
        self.status.setText(f"Loaded: {filename}")

    def scan_candidates(self):
        if self.session.video_path is None:
            QMessageBox.information(self,"Smart Annotation Studio","Open a video first."); return
        scanner=CandidateScanner()
        self.status.setText("Scanning candidate events...")
        def prog(n,total):
            if total>0: self.progress.setValue(int(min(100,round(n/total*100))))
        candidates=scanner.scan(self.session.video_path,progress_callback=prog)
        self.session.set_candidates(candidates)
        self.progress.setValue(100)
        if not candidates:
            self.event_label.setText("No candidates found")
            self.status.setText("No candidate events found."); return
        self.status.setText(f"{len(candidates)} candidate events found.")
        self._refresh_current()

    def _refresh_current(self):
        c=self.session.current
        if c is None: return
        self._show_frame(c.frame,c)
        label=self.session.label_for_frame(c.frame)
        label_text=label.value if label else "UNLABELED"
        self.event_label.setText(
            f"Event {self.session.current_index+1} of {len(self.session.candidates)} | "
            f"Frame {c.frame} | Label: {label_text}"
        )
        self.meta_label.setText(
            f"Candidate confidence: {c.confidence:.2%} | "
            f"Suggested decision: {c.suggested_decision or '-'} | Ball: ({c.x}, {c.y})"
        )

    def _show_frame(self, frame_number, candidate):
        if self.capture is None: return
        self.capture.set(cv2.CAP_PROP_POS_FRAMES,int(frame_number))
        ok,frame=self.capture.read()
        if not ok:
            self.status.setText(f"Unable to read frame {frame_number}"); return
        if candidate is not None:
            frame=zoom_around_candidate(frame,x=candidate.x,y=candidate.y,scale=2.5)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w=rgb.shape[:2]
        image=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format.Format_RGB888).copy()
        pixmap=QPixmap.fromImage(image)
        self.preview.setPixmap(pixmap.scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def label_current(self,label):
        if self.session.current is None: return
        self.session.label_current(label)
        self._refresh_current()
        if self.session.current_index < len(self.session.candidates)-1:
            self.next_event()

    def next_event(self):
        self.session.next(); self._refresh_current()

    def previous_event(self):
        self.session.previous(); self._refresh_current()

    def save_ground_truth(self):
        events=self.session.truth_events()
        if not events:
            QMessageBox.information(self,"Smart Annotation Studio","No IN/OUT events labeled yet."); return
        filename,_=QFileDialog.getSaveFileName(
            self,"Save Ground Truth","ground_truth.jsonl","JSON Lines (*.jsonl)"
        )
        if filename:
            save_truth_jsonl(events,filename)
            QMessageBox.information(self,"Smart Annotation Studio",f"Saved {len(events)} events.")

    def closeEvent(self,event):
        if self.capture: self.capture.release()
        event.accept()
