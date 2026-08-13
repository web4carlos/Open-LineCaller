from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class EditableCourt:
    # order: near-left, near-right, far-right, far-left
    points: list[list[float]]
    original_points: list[list[float]]
    source: str = "AUTO"

    @classmethod
    def create(cls, points, source="AUTO"):
        p=[[float(x),float(y)] for x,y in points]
        return cls([x[:] for x in p],[x[:] for x in p],source)

    def reset(self):
        self.points=[x[:] for x in self.original_points]

    def move_corner(self,index,x,y):
        self.points[index]=[float(x),float(y)]

    def move_boundary(self,boundary,dx,dy):
        ids={
            "NEAR_BASELINE":(0,1),
            "RIGHT_SIDELINE":(1,2),
            "FAR_BASELINE":(2,3),
            "LEFT_SIDELINE":(3,0),
        }[boundary]
        for i in ids:
            self.points[i][0]+=float(dx)
            self.points[i][1]+=float(dy)

    def valid(self):
        q=np.asarray(self.points,np.float32)
        if q.shape != (4,2) or not np.isfinite(q).all():
            return False
        # Convex and meaningful area.
        return bool(cv2.isContourConvex(q.astype(np.int32)) and abs(cv2.contourArea(q)) > 1000)
