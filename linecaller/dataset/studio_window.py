from pathlib import Path
import json, cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction,QKeySequence,QShortcut
from PySide6.QtWidgets import QCheckBox,QComboBox,QFileDialog,QHBoxLayout,QLabel,QMainWindow,QMessageBox,QPushButton,QSlider,QVBoxLayout,QWidget
from linecaller.dataset.models import BallBox
from linecaller.dataset.studio_session import DatasetStudioSession
from linecaller.dataset.studio_widget import AnnotationVideoWidget
from linecaller.dataset.validation import validate_clip

class DatasetStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Open-LineCaller — Dataset Studio"); self.resize(1450,900)
        self.capture=None; self.video_path=None; self.current_frame=None; self.current_frame_number=0
        self.frame_count=0; self.fps=0.; self.session=None; self.annotation_path=None; self.cx=self.cy=0.
        self._menu(); self._ui(); self._shortcuts(); self._status()

    def _menu(self):
        m=self.menuBar().addMenu("&File")
        for label,shortcut,cb in [("Open Video...","Ctrl+O",self.open_video),("Save Annotation...","Ctrl+S",self.save_annotation)]:
            a=QAction(label,self); a.setShortcut(shortcut); a.triggered.connect(cb); m.addAction(a)

    def _ui(self):
        r=QWidget(); l=QVBoxLayout(r); self.video=AnnotationVideoWidget()
        self.video.box_changed.connect(self._set_box); self.video.cursor_frame_position.connect(lambda x,y:setattr(self,"cx",x) or setattr(self,"cy",y))
        l.addWidget(self.video,1); self.slider=QSlider(Qt.Orientation.Horizontal); self.slider.setEnabled(False); self.slider.valueChanged.connect(self.goto_frame); l.addWidget(self.slider)
        row=QHBoxLayout(); self.vis=QCheckBox("Visible"); self.vis.setChecked(True); self.occ=QCheckBox("Occluded"); self.dec=QComboBox(); self.dec.addItems(["","IN","OUT","REVIEW"])
        buttons=[("◀ Previous",lambda:self.goto_frame(self.current_frame_number-1)),("Next ▶",lambda:self.goto_frame(self.current_frame_number+1)),("Delete Ball Box",self.delete_box),("Mark Bounce",self.toggle_bounce)]
        for t,cb in buttons:
            b=QPushButton(t); b.clicked.connect(cb); row.addWidget(b)
        row.addWidget(self.vis); row.addWidget(self.occ); row.addWidget(QLabel("Decision:")); row.addWidget(self.dec); l.addLayout(row)
        self.vis.toggled.connect(self._flags); self.occ.toggled.connect(self._flags); self.dec.currentTextChanged.connect(self._decision)
        self.status=QLabel(); l.addWidget(self.status); self.setCentralWidget(r)

    def _shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Left),self,activated=lambda:self.goto_frame(self.current_frame_number-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right),self,activated=lambda:self.goto_frame(self.current_frame_number+1))
        QShortcut(QKeySequence("B"),self,activated=self.toggle_bounce)
        QShortcut(QKeySequence("Delete"),self,activated=self.delete_box)
        for k,d in [("1","IN"),("2","OUT"),("3","REVIEW")]: QShortcut(QKeySequence(k),self,activated=lambda d=d:self.dec.setCurrentText(d))

    def open_video(self):
        f,_=QFileDialog.getOpenFileName(self,"Open Pickleball Clip","","Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)")
        if not f:return
        if self.capture:self.capture.release()
        c=cv2.VideoCapture(f)
        if not c.isOpened(): QMessageBox.critical(self,"Dataset Studio","Unable to open video."); return
        self.capture=c; self.video_path=Path(f); w=int(c.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps=float(c.get(cv2.CAP_PROP_FPS)) or 30.; self.frame_count=int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        self.session=DatasetStudioSession(clip_id=self.video_path.stem,source_video=str(self.video_path),width=w,height=h,fps=self.fps,frame_count=self.frame_count)
        self.slider.setEnabled(True); self.slider.setRange(0,max(0,self.frame_count-1)); self.goto_frame(0)

    def goto_frame(self,n):
        if not self.capture:return
        n=max(0,min(n,max(0,self.frame_count-1))); self.capture.set(cv2.CAP_PROP_POS_FRAMES,n); ok,fr=self.capture.read()
        if not ok:return
        self.current_frame_number=n; self.current_frame=fr; self.slider.blockSignals(True); self.slider.setValue(n); self.slider.blockSignals(False)
        a=self.session.frame_annotation(n) if self.session else None; b=self.session.bounce_annotation(n) if self.session else None
        self.vis.blockSignals(True); self.occ.blockSignals(True); self.dec.blockSignals(True)
        self.vis.setChecked(a.visible if a else True); self.occ.setChecked(a.occluded if a else False); self.dec.setCurrentText(b.decision if b and b.decision else "")
        self.vis.blockSignals(False); self.occ.blockSignals(False); self.dec.blockSignals(False); self._refresh()

    def _refresh(self):
        if self.session:
            a=self.session.frame_annotation(self.current_frame_number); b=self.session.bounce_annotation(self.current_frame_number)
            self.video.set_scene(self.current_frame,a.ball if a else None,b)
        else:self.video.set_scene(self.current_frame)
        self._status()

    def _set_box(self,x,y,w,h):
        self.session.set_ball_box(self.current_frame_number,BallBox(x,y,w,h),visible=self.vis.isChecked(),occluded=self.occ.isChecked()); self._refresh()

    def delete_box(self):
        if self.session:self.session.remove_ball_box(self.current_frame_number); self._refresh()

    def _flags(self):
        if self.session:self.session.set_flags(self.current_frame_number,visible=self.vis.isChecked(),occluded=self.occ.isChecked()); self._refresh()

    def toggle_bounce(self):
        if not self.session:return
        f=self.current_frame_number; b=self.session.bounce_annotation(f)
        if b:self.session.toggle_bounce(f,x=b.x,y=b.y,decision=b.decision)
        else:
            a=self.session.frame_annotation(f)
            x=(a.ball.x+a.ball.width/2) if a and a.ball else self.cx
            y=(a.ball.y+a.ball.height) if a and a.ball else self.cy
            self.session.toggle_bounce(f,x=x,y=y,decision=self.dec.currentText() or None)
        self._refresh()

    def _decision(self,t):
        if self.session and self.session.bounce_annotation(self.current_frame_number):
            self.session.set_bounce_decision(self.current_frame_number,t or None); self._refresh()

    def save_annotation(self):
        if not self.session:return
        a=self.session.to_annotation(); issues=validate_clip(a); errs=[i for i in issues if i.severity=="ERROR"]
        if errs: QMessageBox.critical(self,"Validation failed","\n".join(i.message for i in errs[:10])); return
        if not self.annotation_path:
            f,_=QFileDialog.getSaveFileName(self,"Save Annotation",f"{a.clip_id}.json","JSON (*.json)")
            if not f:return
            self.annotation_path=Path(f)
        self.annotation_path.write_text(json.dumps(a.to_dict(),indent=2),encoding="utf-8")
        QMessageBox.information(self,"Dataset Studio",f"Saved:\n{self.annotation_path}")

    def _status(self):
        if not self.session:self.status.setText("No clip loaded"); return
        a=self.session.to_annotation(); boxes=sum(1 for f in a.frames if f.ball)
        self.status.setText(f"Clip: {a.clip_id} | Frame {self.current_frame_number+1}/{a.frame_count} | {a.width}x{a.height} @ {a.fps:.2f} fps | Ball boxes: {boxes} | Bounces: {len(a.bounces)}")

    def closeEvent(self,e):
        if self.capture:self.capture.release()
        e.accept()
