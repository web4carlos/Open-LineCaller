import cv2
class AnnotationVideoSource:
    def __init__(self,path):
        self.path=str(path);self.cap=cv2.VideoCapture(self.path)
        if not self.cap.isOpened(): raise RuntimeError(f"Could not open video: {self.path}")
        self.frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps=float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
    def read(self,frame_number):
        frame_number=max(0,min(int(frame_number),max(0,self.frame_count-1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,frame_number);ok,frame=self.cap.read()
        return frame if ok else None
    def close(self):
        if self.cap is not None:self.cap.release();self.cap=None
