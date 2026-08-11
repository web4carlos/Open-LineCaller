from dataclasses import dataclass
import cv2, numpy as np

@dataclass(frozen=True)
class SmartBBoxCandidate:
    center_x: float
    center_y: float
    width: int
    height: int
    score: float
    method: str = "local-color-contrast"

class SmartBBoxDetector:
    def __init__(self, roi_size=96, min_box=10, max_box=52, fallback_box=24):
        self.roi_size=int(roi_size); self.min_box=int(min_box); self.max_box=int(max_box); self.fallback_box=int(fallback_box)

    def _fallback(self,x,y):
        return SmartBBoxCandidate(float(x),float(y),self.fallback_box,self.fallback_box,0.0,"fallback-click")

    def detect(self, frame, click_x, click_y):
        if frame is None or frame.size==0: return self._fallback(click_x,click_y)
        h,w=frame.shape[:2]; cx=int(round(click_x)); cy=int(round(click_y)); half=max(16,self.roi_size//2)
        x1=max(0,cx-half); y1=max(0,cy-half); x2=min(w,cx+half); y2=min(h,cy+half)
        roi=frame[y1:y2,x1:x2]
        if roi.size==0: return self._fallback(click_x,click_y)

        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        color=cv2.inRange(hsv,np.array([20,55,70],np.uint8),np.array([95,255,255],np.uint8))
        gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        _,bright=cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
        mask=cv2.bitwise_and(color,bright)
        k=np.ones((3,3),np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        click_local=np.array([click_x-x1,click_y-y1],np.float32)
        diag=max(1.0,float(np.hypot(roi.shape[1],roi.shape[0])))
        best=None; best_score=-1.0

        for c in contours:
            area=float(cv2.contourArea(c))
            if area<3: continue
            rx,ry,rw,rh=cv2.boundingRect(c)
            if rw<=0 or rh<=0 or rw>self.max_box*2 or rh>self.max_box*2: continue
            cc=np.array([rx+rw/2,ry+rh/2],np.float32)
            distance=float(np.linalg.norm(cc-click_local))
            proximity=max(0.0,1.0-distance/diag)
            per=float(cv2.arcLength(c,True))
            circularity=min(1.0,4*np.pi*area/(per*per)) if per>0 else 0.0
            fill=min(1.0,area/max(1.0,float(rw*rh)))
            score=.58*proximity+.22*circularity+.20*fill
            if score>best_score:
                best_score=score; best=(cc[0],cc[1],rw,rh)

        if best is None: return self._fallback(click_x,click_y)
        lx,ly,rw,rh=best
        bw=int(np.clip(round(rw*1.45),self.min_box,self.max_box))
        bh=int(np.clip(round(rh*1.45),self.min_box,self.max_box))
        return SmartBBoxCandidate(float(x1+lx),float(y1+ly),bw,bh,float(np.clip(best_score,0,1)))
