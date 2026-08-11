import cv2
from PySide6.QtCore import Qt,Signal
from PySide6.QtGui import QImage,QPainter,QPen,QPixmap
from PySide6.QtWidgets import QLabel
class AnnotationImageView(QLabel):
    image_clicked=Signal(float,float)
    def __init__(self):
        super().__init__("Open a video to begin");self.setAlignment(Qt.AlignmentFlag.AlignCenter);self.setMinimumSize(900,560)
        self.setStyleSheet("background:#050914;border:1px solid #263453;border-radius:14px;color:#8190aa;")
        self._pixmap_original=None;self._image_width=0;self._image_height=0;self._annotation=None
    def set_frame(self,frame):
        if frame is None:return
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
        image=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format.Format_RGB888).copy()
        self._pixmap_original=QPixmap.fromImage(image);self._image_width=w;self._image_height=h;self._render()
    def set_annotation(self,annotation):self._annotation=annotation;self._render()
    def _scaled_pixmap(self):
        if self._pixmap_original is None:return None
        return self._pixmap_original.scaled(self.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)
    def _render(self):
        pix=self._scaled_pixmap()
        if pix is None:return
        rendered=QPixmap(pix);p=QPainter(rendered)
        if self._annotation and self._annotation.get("type")=="positive":
            x=float(self._annotation["x"]);y=float(self._annotation["y"]);box=int(self._annotation.get("box_size_px",24))
            sx=pix.width()/max(1,self._image_width);sy=pix.height()/max(1,self._image_height)
            cx=x*sx;cy=y*sy;bw=max(8.0,box*sx);bh=max(8.0,box*sy)
            pen=QPen(Qt.GlobalColor.green);pen.setWidth(3);p.setPen(pen)
            p.drawRect(int(cx-bw/2),int(cy-bh/2),int(bw),int(bh))
            p.drawLine(int(cx-14),int(cy),int(cx+14),int(cy));p.drawLine(int(cx),int(cy-14),int(cx),int(cy+14))
        p.end();self.setPixmap(rendered)
    def resizeEvent(self,event):super().resizeEvent(event);self._render()
    def mousePressEvent(self,event):
        if self._pixmap_original is None:return
        pix=self._scaled_pixmap()
        if pix is None:return
        left=(self.width()-pix.width())/2.0;top=(self.height()-pix.height())/2.0
        px=event.position().x()-left;py=event.position().y()-top
        if px<0 or py<0 or px>=pix.width() or py>=pix.height():return
        self.image_clicked.emit(float(px/pix.width()*self._image_width),float(py/pix.height()*self._image_height))
