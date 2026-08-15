from dataclasses import dataclass
import cv2
@dataclass(frozen=True)
class MatchResult:
    found: bool; x: float|None; y: float|None; score: float; scale: float|None; roi: tuple
class DCFBallTemplateMatcher:
    def __init__(self,min_score=.42,scales=(.65,.8,.95,1.,1.15,1.3,1.5)):
        self.min_score=min_score; self.scales=scales
    def match(self,frame,template,roi):
        x1,y1,x2,y2=map(int,roi); h,w=frame.shape[:2]
        x1=max(0,x1);y1=max(0,y1);x2=min(w,x2);y2=min(h,y2)
        crop=cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)
        base=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        best=None
        for s in self.scales:
            tw=max(3,round(base.shape[1]*s));th=max(3,round(base.shape[0]*s))
            if tw>crop.shape[1] or th>crop.shape[0]: continue
            t=cv2.resize(base,(tw,th))
            _,score,_,loc=cv2.minMaxLoc(cv2.matchTemplate(crop,t,cv2.TM_CCOEFF_NORMED))
            if best is None or score>best[0]: best=(score,x1+loc[0]+tw/2,y1+loc[1]+th/2,s)
        if best is None:return MatchResult(False,None,None,0,None,(x1,y1,x2,y2))
        score,x,y,s=best; found=score>=self.min_score
        return MatchResult(found,x if found else None,y if found else None,float(score),s,(x1,y1,x2,y2))
