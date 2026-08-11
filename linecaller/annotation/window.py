from pathlib import Path
import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction,QKeySequence,QShortcut
from PySide6.QtWidgets import QFileDialog,QFrame,QHBoxLayout,QLabel,QMainWindow,QMessageBox,QPushButton,QSpinBox,QVBoxLayout,QWidget
from .image_view import AnnotationImageView
from .models import BallAnnotation
from .session import AnnotationSession
from .video_source import AnnotationVideoSource
from .yolo import YoloLabelWriter
from .smart_bbox import SmartBBoxDetector
from .zoom_widget import AnnotationZoomWidget

APP_QSS="""
QMainWindow,QWidget{background:#0b1020;color:#e9eefb;font-family:"Segoe UI";font-size:10.5pt;}
QFrame#Card{background:#111a2d;border:1px solid #263758;border-radius:16px;}
QLabel#Title{font-size:20pt;font-weight:800;}
QLabel#Muted{color:#93a4bf;}
QLabel#Metric{font-size:16pt;font-weight:700;}
QPushButton{background:#182542;border:1px solid #2c4068;border-radius:10px;padding:10px 15px;font-weight:650;}
QPushButton:hover{background:#213259;}
QPushButton#Primary{background:#2469ee;border:1px solid #3c7cf3;font-size:11pt;}
QPushButton#Negative{background:#50301e;border:1px solid #775038;}
QPushButton#Danger{background:#4a202b;border:1px solid #733649;}
QSpinBox{background:#0e1627;border:1px solid #2a3d61;border-radius:8px;padding:7px;}
"""

class AnnotationStudioWindow(QMainWindow):
    DEFAULT_DATASET_ROOT=r"C:\validation\pickleball-dataset\raw"
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open-LineCaller — Pickleball Annotation Studio")
        self.resize(1500,920);self.setMinimumSize(1150,740);self.setStyleSheet(APP_QSS)
        self.source=None;self.session=None;self.current_frame=None
        self.smart_bbox = SmartBBoxDetector()
        self._build_ui();self._build_menu();self._build_shortcuts()
        self.statusBar().showMessage("Ready — Open a video")
    def _card(self):
        f=QFrame();f.setObjectName("Card");return f
    def _build_ui(self):
        root=QWidget();self.setCentralWidget(root)
        outer=QVBoxLayout(root);outer.setContentsMargins(18,16,18,16);outer.setSpacing(12)
        top=QHBoxLayout()
        t=QLabel("Pickleball Annotation Studio");t.setObjectName("Title");top.addWidget(t);top.addStretch(1)
        self.video_label=QLabel("No video");self.video_label.setObjectName("Muted");top.addWidget(self.video_label)
        outer.addLayout(top)
        body=QHBoxLayout();body.setSpacing(14)

        left_card=self._card();left=QVBoxLayout(left_card);left.setContentsMargins(12,12,12,12)
        self.image_view=AnnotationImageView();self.image_view.image_clicked.connect(self._on_image_clicked);left.addWidget(self.image_view,1)
        nav=QHBoxLayout()
        b=QPushButton("◀ PREV");b.clicked.connect(lambda:self.move_frame(-1));nav.addWidget(b)
        self.frame_label=QLabel("Frame —");self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter);nav.addWidget(self.frame_label,1)
        b=QPushButton("NEXT ▶");b.clicked.connect(lambda:self.move_frame(1));nav.addWidget(b)
        left.addLayout(nav)

        actions=QHBoxLayout()
        self.open_btn=QPushButton("OPEN VIDEO");self.open_btn.setObjectName("Primary");self.open_btn.clicked.connect(self.open_video);actions.addWidget(self.open_btn)
        b=QPushButton("SAVE + NEXT   [SPACE]");b.setObjectName("Primary");b.clicked.connect(self.save_and_next);actions.addWidget(b)
        b=QPushButton("NO BALL   [N]");b.setObjectName("Negative");b.clicked.connect(self.mark_negative);actions.addWidget(b)
        b=QPushButton("DELETE LABEL");b.setObjectName("Danger");b.clicked.connect(self.delete_label);actions.addWidget(b)
        left.addLayout(actions);body.addWidget(left_card,5)

        side_card=self._card();side=QVBoxLayout(side_card);side.setContentsMargins(18,18,18,18);side.setSpacing(12)
        h=QLabel("ZERO-TYPING WORKFLOW");h.setObjectName("Muted");side.addWidget(h)
        ins=QLabel("1. Click the ball\n2. Press SPACE\n\nNo ball visible?\nPress N");ins.setWordWrap(True);ins.setObjectName("Metric");side.addWidget(ins)
        row=QHBoxLayout();row.addWidget(QLabel("Box size"))
        self.box_size=QSpinBox();self.box_size.setRange(8,100);self.box_size.setSingleStep(2);self.box_size.setValue(24);self.box_size.valueChanged.connect(self._box_size_changed);row.addWidget(self.box_size);side.addLayout(row)
        self.zoom_widget=AnnotationZoomWidget();side.addWidget(self.zoom_widget)
        self.candidate_label=QLabel('Candidate\n—');self.candidate_label.setObjectName('Metric');side.addWidget(self.candidate_label)
        self.positive_label=QLabel("Positive\n0");self.positive_label.setObjectName("Metric");side.addWidget(self.positive_label)
        self.negative_label=QLabel("Negative\n0");self.negative_label.setObjectName("Metric");side.addWidget(self.negative_label)
        self.labeled_label=QLabel("Labeled\n0");self.labeled_label.setObjectName("Metric");side.addWidget(self.labeled_label)
        self.remaining_label=QLabel("Remaining\n—");self.remaining_label.setObjectName("Metric");side.addWidget(self.remaining_label)
        side.addStretch(1)
        d=QLabel("Dataset output:\nC:\\validation\\pickleball-dataset\\raw");d.setObjectName("Muted");d.setWordWrap(True);side.addWidget(d)
        body.addWidget(side_card,1);outer.addLayout(body,1)

    def _build_menu(self):
        m=self.menuBar().addMenu("File")
        a=QAction("Open Video",self);a.triggered.connect(self.open_video);m.addAction(a)
        a=QAction("Exit",self);a.triggered.connect(self.close);m.addAction(a)
    def _build_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Space),self,activated=self.save_and_next)
        QShortcut(QKeySequence(Qt.Key.Key_N),self,activated=self.mark_negative)
        QShortcut(QKeySequence(Qt.Key.Key_Left),self,activated=lambda:self.move_frame(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right),self,activated=lambda:self.move_frame(1))
        QShortcut(QKeySequence("Shift+Left"),self,activated=lambda:self.move_frame(-10))
        QShortcut(QKeySequence("Shift+Right"),self,activated=lambda:self.move_frame(10))
        QShortcut(QKeySequence("Ctrl+Left"),self,activated=lambda:self.move_frame(-100))
        QShortcut(QKeySequence("Ctrl+Right"),self,activated=lambda:self.move_frame(100))
        QShortcut(QKeySequence(Qt.Key.Key_Delete),self,activated=self.delete_label)

    def open_video(self):
        path,_=QFileDialog.getOpenFileName(self,"Open pickleball video",r"C:\validation","Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)")
        if not path:return
        try: source=AnnotationVideoSource(path)
        except Exception as exc:
            QMessageBox.critical(self,"Annotation Studio",str(exc));return
        if self.source is not None:self.source.close()
        self.source=source
        self.session=AnnotationSession.load_or_create(video_path=path,dataset_root=self.DEFAULT_DATASET_ROOT)
        self.box_size.setValue(self.session.box_size_px)
        self.video_label.setText(Path(path).name)
        self.load_frame(self.session.current_frame)
        self.statusBar().showMessage("Click the pickleball, then SPACE")

    def _box_size_changed(self,value):
        if self.session is None:return
        self.session.box_size_px=int(value)
        data=self.session.annotation_for(self.session.current_frame)
        if data and data.get("type")=="positive":data["box_size_px"]=int(value)
        self.session.save();self.image_view.set_annotation(data)

    def _on_image_clicked(self,x,y):
        if self.session is None or self.current_frame is None:return
        c=self.smart_bbox.detect(self.current_frame,x,y)
        self.session.mark_positive(
            self.session.current_frame,c.center_x,c.center_y,
            box_width_px=c.width,box_height_px=c.height,
            score=c.score,method=c.method
        )
        self.session.save()
        data=self.session.annotation_for(self.session.current_frame)
        self.image_view.set_annotation(data)
        self.zoom_widget.show_candidate(self.current_frame,c.center_x,c.center_y,c.width,c.height)
        self.candidate_label.setText(f'Candidate\n{c.score*100:.0f}%' if c.score>0 else 'Candidate\nMANUAL')
        self._refresh_stats()
        self.statusBar().showMessage('Smart box ready — SPACE accepts; click again to correct')

    def load_frame(self,frame_number):
        if self.source is None or self.session is None:return
        frame_number=max(0,min(int(frame_number),max(0,self.source.frame_count-1)))
        frame=self.source.read(frame_number)
        if frame is None:return
        self.session.current_frame=frame_number;self.session.save();self.current_frame=frame
        self.image_view.set_frame(frame);self.image_view.set_annotation(self.session.annotation_for(frame_number))
        self.frame_label.setText(f"Frame {frame_number:,} / {max(0,self.source.frame_count-1):,}")
        self._refresh_stats()

    def move_frame(self,delta):
        if self.session is not None:self.load_frame(self.session.current_frame+int(delta))

    def _export_current(self):
        if self.source is None or self.session is None or self.current_frame is None:return False
        data=self.session.annotation_for(self.session.current_frame)
        if data is None:
            QMessageBox.information(self,"Annotation Studio","Click the ball first, or press NO BALL.");return False
        image_dir=Path(self.DEFAULT_DATASET_ROOT)/"images";label_dir=Path(self.DEFAULT_DATASET_ROOT)/"labels"
        image_dir.mkdir(parents=True,exist_ok=True);label_dir.mkdir(parents=True,exist_ok=True)
        stem=f"{Path(self.session.video_path).stem}_frame_{self.session.current_frame:06d}"
        image_path=image_dir/f"{stem}.jpg";label_path=label_dir/f"{stem}.txt"
        cv2.imwrite(str(image_path),self.current_frame)
        if data.get("type")=="negative":
            YoloLabelWriter.write_negative(label_path)
        else:
            old_size=int(data.get('box_size_px',self.session.box_size_px))
            ann=BallAnnotation(
                self.session.current_frame,
                float(data['x']),float(data['y']),
                int(data.get('box_width_px',old_size)),
                int(data.get('box_height_px',old_size))
            )
            YoloLabelWriter.write_positive(label_path,ann,self.source.width,self.source.height)
        return True

    def save_and_next(self):
        if self._export_current():self.move_frame(1)
    def mark_negative(self):
        if self.session is None:return
        self.session.mark_negative(self.session.current_frame);self.session.save()
        self.image_view.set_annotation(self.session.annotation_for(self.session.current_frame));self._refresh_stats()
        if self._export_current():self.move_frame(1)
    def delete_label(self):
        if self.session is None:return
        self.session.remove(self.session.current_frame);self.session.save();self.image_view.set_annotation(None);self._refresh_stats()
    def _refresh_stats(self):
        if self.session is None:return
        s=self.session.stats()
        self.positive_label.setText(f"Positive\n{s.positives}");self.negative_label.setText(f"Negative\n{s.negatives}");self.labeled_label.setText(f"Labeled\n{s.labeled}")
        if self.source is not None:self.remaining_label.setText(f"Remaining\n{max(0,self.source.frame_count-s.labeled):,}")
    def closeEvent(self,event):
        if self.session is not None:self.session.save()
        if self.source is not None:self.source.close()
        event.accept()
