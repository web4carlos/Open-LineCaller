from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
from .smart_bbox import SmartBBoxDetector

@dataclass(frozen=True)
class AssistSuggestion:
    x: float
    y: float
    width: int
    height: int
    confidence: float
    source: str

class HSVAutoAssist:
    def suggest(self, frame, previous_xy: Optional[tuple[float,float]]=None):
        if frame is None or frame.size == 0:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([20,45,75], dtype=np.uint8),
            np.array([95,255,255], dtype=np.uint8),
        )
        k = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h,w = frame.shape[:2]
        diag = max(1.0, float(np.hypot(w,h)))

        best = None
        best_score = -1.0

        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 3 or area > 1800:
                continue

            (cx,cy), radius = cv2.minEnclosingCircle(c)
            if radius < 2 or radius > 30:
                continue

            per = float(cv2.arcLength(c, True))
            circ = min(1.0, 4*np.pi*area/(per*per)) if per > 0 else 0.0

            x,y,bw,bh = cv2.boundingRect(c)
            fill = min(1.0, area/max(1.0, float(bw*bh)))

            temporal = 0.5
            if previous_xy is not None:
                d = float(np.hypot(cx-previous_xy[0], cy-previous_xy[1]))
                temporal = max(0.0, 1.0-d/diag)

            score = 0.40*temporal + 0.30*circ + 0.30*fill

            if score > best_score:
                best_score = score
                best = (cx,cy,bw,bh,score)

        if best is None:
            return None

        cx,cy,bw,bh,score = best
        bw = int(np.clip(round(bw*1.5), 10, 56))
        bh = int(np.clip(round(bh*1.5), 10, 56))

        return AssistSuggestion(
            float(cx), float(cy), bw, bh,
            float(np.clip(score,0,1)),
            "HSV-AUTO"
        )

class CombinedAutoAssist:
    def __init__(self):
        self.hsv = HSVAutoAssist()
        self.local = SmartBBoxDetector()

    def suggest(self, frame, previous_xy=None):
        return self.hsv.suggest(frame, previous_xy=previous_xy)

    def refine_click(self, frame, x, y):
        c = self.local.detect(frame, x, y)
        return AssistSuggestion(
            c.center_x, c.center_y, c.width, c.height,
            c.score, "CLICK-REFINE"
        )
