from dataclasses import dataclass
import cv2
@dataclass(frozen=True)
class DCFCell:
    ix:int; iy:int; x1:int; y1:int; x2:int; y2:int; inside:bool; expected_scale:float
@dataclass(frozen=True)
class DCFHit:
    found:bool; score:float; x:float|None; y:float|None; cell:DCFCell|None
class OneBallOneDCF:
    def __init__(self,min_score=.55): self.min_score=float(min_score)
    def search(self,frame,ball,cells):
        fg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); bg=cv2.cvtColor(ball,cv2.COLOR_BGR2GRAY); best=None
        for c in cells:
            crop=fg[c.y1:c.y2,c.x1:c.x2]
            tw=max(3,round(bg.shape[1]*c.expected_scale)); th=max(3,round(bg.shape[0]*c.expected_scale))
            if crop.size==0 or tw>crop.shape[1] or th>crop.shape[0]: continue
            tpl=cv2.resize(bg,(tw,th))
            _,score,_,loc=cv2.minMaxLoc(cv2.matchTemplate(crop,tpl,cv2.TM_CCOEFF_NORMED))
            hit=(float(score),c.x1+loc[0]+tw/2,c.y1+loc[1]+th/2,c)
            if best is None or hit[0]>best[0]: best=hit
        if best is None or best[0]<self.min_score:return DCFHit(False,0 if best is None else best[0],None,None,None)
        return DCFHit(True,*best)
