import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel
from linecaller.calibration.studio_mapping import DisplayMapping

class AnnotationVideoWidget(QLabel):
    box_changed=Signal(float,float,float,float)
    cursor_frame_position=Signal(float,float)

    def __init__(self):
        super().__init__(); self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(900,560); self.setMouseTracking(True)
        self.setStyleSheet("background:#111; color:#ddd;"); self.setText("Open a video clip")
        self._frame=None; self._box=None; self._bounce=None; self._start=None; self._current=None

    def set_scene(self,frame,box=None,bounce=None):
        self._frame=frame.copy() if frame is not None else None
        self._box=box; self._bounce=bounce; self._render()

    def _point(self,e):
        if self._frame is None: return None
        h,w=self._frame.shape[:2]
        m=DisplayMapping(frame_width=w,frame_height=h,widget_width=max(1,self.width()),widget_height=max(1,self.height()))
        return m.widget_to_frame(e.position().x(),e.position().y())

    def mousePressEvent(self,e):
        if e.button()==Qt.MouseButton.LeftButton:
            p=self._point(e)
            if p is not None: self._start=p; self._current=p

    def mouseMoveEvent(self,e):
        p=self._point(e)
        if p is None:return
        self.cursor_frame_position.emit(*p)
        if self._start is not None: self._current=p; self._render()

    def mouseReleaseEvent(self,e):
        if self._start is None:return
        p=self._point(e); s=self._start
        self._start=self._current=None
        if p is None:return
        x=min(s[0],p[0]); y=min(s[1],p[1]); w=abs(p[0]-s[0]); h=abs(p[1]-s[1])
        if w>=2 and h>=2:self.box_changed.emit(x,y,w,h)
        self._render()

    def resizeEvent(self,e): super().resizeEvent(e); self._render()

    def _render(self):
        if self._frame is None:return
        d=self._frame.copy()
        if self._box:
            cv2.rectangle(d,(int(self._box.x),int(self._box.y)),(int(self._box.x+self._box.width),int(self._box.y+self._box.height)),(0,255,0),2)
        if self._bounce:
            p=(int(self._bounce.x),int(self._bounce.y)); cv2.circle(d,p,12,(0,255,255),2)
            cv2.putText(d,"BOUNCE "+(self._bounce.decision or ""), (p[0]+14,p[1]-10), cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2,cv2.LINE_AA)
        if self._start and self._current:
            x1,y1=self._start; x2,y2=self._current
            cv2.rectangle(d,(int(min(x1,x2)),int(min(y1,y2))),(int(max(x1,x2)),int(max(y1,y2))),(255,255,255),1)
        rgb=cv2.cvtColor(d,cv2.COLOR_BGR2RGB); h,w,ch=rgb.shape
        q=QImage(rgb.data,w,h,ch*w,QImage.Format.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(q).scaled(self.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
