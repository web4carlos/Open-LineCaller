import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage,QPixmap
from PySide6.QtWidgets import QLabel

class AnnotationZoomWidget(QLabel):
    def __init__(self):
        super().__init__("Click the ball\nfor zoom preview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(260,260)
        self.setStyleSheet("background:#050914;border:1px solid #263453;border-radius:12px;color:#8190aa;")
    def show_candidate(self,frame,x,y,width,height,crop_size=120):
        if frame is None:return
        fh,fw=frame.shape[:2];half=crop_size//2;cx=int(round(x));cy=int(round(y))
        x1=max(0,cx-half);y1=max(0,cy-half);x2=min(fw,cx+half);y2=min(fh,cy+half)
        crop=frame[y1:y2,x1:x2].copy()
        if crop.size==0:return
        lx=cx-x1;ly=cy-y1
        cv2.rectangle(crop,(int(lx-width/2),int(ly-height/2)),(int(lx+width/2),int(ly+height/2)),(0,255,0),2)
        rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
        img=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(img).scaled(self.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
