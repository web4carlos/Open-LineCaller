import sys
import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage,QPixmap
from PySide6.QtWidgets import (
    QApplication,QFileDialog,QFormLayout,QHBoxLayout,QLabel,QMainWindow,
    QPushButton,QSlider,QSpinBox,QDoubleSpinBox,QVBoxLayout,QWidget
)

from linecaller.ball.advanced_motion_detector import AdvancedMotionBallDetector
from linecaller.ball.detector_profile import DetectorProfile

class DetectorTuningLabWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LineCaller Artificial Vision — Detector Tuning Lab")
        self.resize(1500,900)
        self.capture=None
        self.frame_count=0
        self.frame_number=0
        self.detector=AdvancedMotionBallDetector()
        self._build()

    def _build(self):
        root=QWidget()
        outer=QVBoxLayout(root)

        top=QHBoxLayout()
        for text,handler in [
            ("Open Video",self.open_video),
            ("Reset Detector",self.reset_detector),
            ("Load Profile",self.load_profile),
            ("Save Profile",self.save_profile),
        ]:
            b=QPushButton(text); b.clicked.connect(handler); top.addWidget(b)
        outer.addLayout(top)

        body=QHBoxLayout()

        left=QVBoxLayout()
        self.preview=QLabel("Open video")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(800,500)
        self.preview.setStyleSheet("background:#111;color:#ddd;")
        left.addWidget(self.preview,1)

        self.mask=QLabel("Foreground mask")
        self.mask.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask.setMinimumHeight(220)
        self.mask.setStyleSheet("background:#111;color:#ddd;")
        left.addWidget(self.mask)

        self.slider=QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.show_frame)
        left.addWidget(self.slider)

        nav=QHBoxLayout()
        prev=QPushButton("◀ Frame")
        nxt=QPushButton("Frame ▶")
        prev.clicked.connect(lambda:self.show_frame(self.frame_number-1))
        nxt.clicked.connect(lambda:self.show_frame(self.frame_number+1))
        nav.addWidget(prev); nav.addWidget(nxt)
        left.addLayout(nav)

        self.status=QLabel("Ready")
        left.addWidget(self.status)
        body.addLayout(left,3)

        form=QFormLayout()
        self.controls={}

        def dbl(name,value,lo,hi,step):
            c=QDoubleSpinBox()
            c.setRange(lo,hi); c.setDecimals(3); c.setSingleStep(step); c.setValue(value)
            c.valueChanged.connect(self.parameters_changed)
            self.controls[name]=c; form.addRow(name,c)

        def integer(name,value,lo,hi):
            c=QSpinBox()
            c.setRange(lo,hi); c.setValue(value)
            c.valueChanged.connect(self.parameters_changed)
            self.controls[name]=c; form.addRow(name,c)

        dbl("min_area",4.0,0.0,5000.0,1.0)
        dbl("max_area",1400.0,1.0,20000.0,10.0)
        dbl("min_radius_px",1.0,0.1,50.0,0.5)
        dbl("max_radius_px",35.0,1.0,200.0,1.0)
        dbl("min_circularity",0.08,0.0,1.0,0.01)
        dbl("min_confidence",0.0,0.0,1.0,0.01)
        integer("max_candidates",24,1,100)
        dbl("temporal_radius_px",180.0,1.0,1000.0,10.0)

        right=QVBoxLayout()
        right.addLayout(form)
        self.telemetry=QLabel("")
        self.telemetry.setWordWrap(True)
        right.addWidget(self.telemetry)
        right.addStretch(1)
        body.addLayout(right,1)

        outer.addLayout(body)
        self.setCentralWidget(root)

    def profile(self):
        return DetectorProfile(
            min_area=self.controls["min_area"].value(),
            max_area=self.controls["max_area"].value(),
            min_radius_px=self.controls["min_radius_px"].value(),
            max_radius_px=self.controls["max_radius_px"].value(),
            min_circularity=self.controls["min_circularity"].value(),
            max_candidates=self.controls["max_candidates"].value(),
            temporal_radius_px=self.controls["temporal_radius_px"].value(),
            min_confidence=self.controls["min_confidence"].value(),
            history=self.detector.history,
            var_threshold=self.detector.var_threshold,
        )

    def parameters_changed(self,*_):
        self.detector.apply_profile(self.profile(),reset=True)
        if self.capture is not None:
            self.show_frame(self.frame_number)

    def reset_detector(self):
        self.detector.reset()
        if self.capture is not None:
            self.show_frame(self.frame_number)

    def open_video(self):
        path,_=QFileDialog.getOpenFileName(
            self,"Open Video","","Video (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:return
        if self.capture:self.capture.release()
        self.capture=cv2.VideoCapture(path)
        self.frame_count=int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setEnabled(True)
        self.slider.setRange(0,max(0,self.frame_count-1))
        self.detector.reset()
        self.show_frame(20 if self.frame_count>20 else 0)

    def _qimage(self,frame):
        if len(frame.shape)==2:
            rgb=cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        else:
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w=rgb.shape[:2]
        return QImage(rgb.data,w,h,rgb.strides[0],QImage.Format.Format_RGB888).copy()

    def show_frame(self,n):
        if self.capture is None:return
        n=max(0,min(int(n),max(0,self.frame_count-1)))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES,n)
        ok,frame=self.capture.read()
        if not ok:return

        self.frame_number=n
        self.slider.blockSignals(True); self.slider.setValue(n); self.slider.blockSignals(False)

        analysis=self.detector.analyze(frame)
        overlay=frame.copy()

        for c in analysis.candidates:
            cv2.circle(overlay,(int(c.x),int(c.y)),max(3,int(c.radius_px)),(0,255,0),2)
            cv2.putText(
                overlay,f"{c.confidence:.2f}",
                (int(c.x)+5,int(c.y)-5),
                cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0),1,cv2.LINE_AA
            )

        p=QPixmap.fromImage(self._qimage(overlay))
        self.preview.setPixmap(p.scaled(
            self.preview.size(),Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

        m=QPixmap.fromImage(self._qimage(analysis.mask))
        self.mask.setPixmap(m.scaled(
            self.mask.size(),Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        ))

        t=self.detector.telemetry
        self.status.setText(
            f"Frame {n}/{max(0,self.frame_count-1)} | accepted this frame: {len(analysis.candidates)}"
        )
        self.telemetry.setText(
            f"Frames analyzed: {t.frames}\nContours: {t.contours}\n"
            f"Area rejected: {t.area_rejected}\nShape rejected: {t.shape_rejected}\n"
            f"Radius rejected: {t.radius_rejected}\nConfidence rejected: {t.confidence_rejected}\n"
            f"Accepted: {t.accepted}"
        )

    def save_profile(self):
        path,_=QFileDialog.getSaveFileName(
            self,"Save Detector Profile","detector_profile.json","JSON (*.json)"
        )
        if path:self.profile().save(path)

    def load_profile(self):
        path,_=QFileDialog.getOpenFileName(
            self,"Load Detector Profile","","JSON (*.json)"
        )
        if not path:return
        p=DetectorProfile.load(path)
        for name,value in p.to_dict().items():
            if name in self.controls:
                self.controls[name].blockSignals(True)
                self.controls[name].setValue(value)
                self.controls[name].blockSignals(False)
        self.detector.apply_profile(p,reset=True)
        if self.capture is not None:self.show_frame(self.frame_number)

    def closeEvent(self,event):
        if self.capture:self.capture.release()
        event.accept()

def main():
    app=QApplication(sys.argv)
    w=DetectorTuningLabWindow()
    w.show()
    return app.exec()

if __name__=="__main__":
    raise SystemExit(main())
