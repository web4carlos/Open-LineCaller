from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class RefinementResult:
    accepted: bool
    image_points: tuple[tuple[float,float], ...]
    original_score: float
    refined_score: float
    confidence_gain: float
    reason: str

class CourtGeometryRefiner:
    """
    CP-0030.4 Color-Agnostic Court Geometry Refinement.

    Geometry first:
    - starts from an existing 4-corner court estimate
    - searches locally for strong image edges, independent of paint color
    - fits four boundary lines
    - intersects them to produce a refined quadrilateral
    - accepts only if measurable edge support improves
    """

    def __init__(self, search_radius_px: int = 24, min_gain: float = 0.025):
        self.search_radius_px = int(search_radius_px)
        self.min_gain = float(min_gain)

    @staticmethod
    def _edge_map(frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        channels = cv2.split(lab)
        edges = []
        for ch in channels:
            ch = cv2.GaussianBlur(ch, (5,5), 0)
            edges.append(cv2.Canny(ch, 35, 120))
        return np.maximum.reduce(edges)

    @staticmethod
    def _sample_segment(p1, p2, n=180):
        x=np.linspace(p1[0],p2[0],n)
        y=np.linspace(p1[1],p2[1],n)
        return np.stack([x,y],axis=1)

    def _support(self, edge, p1, p2):
        h,w=edge.shape[:2]
        pts=self._sample_segment(p1,p2)
        vals=[]
        for x,y in pts:
            xi=int(round(x)); yi=int(round(y))
            x0=max(0,xi-2); x1=min(w,xi+3)
            y0=max(0,yi-2); y1=min(h,yi+3)
            vals.append(1.0 if edge[y0:y1,x0:x1].max(initial=0)>0 else 0.0)
        return float(np.mean(vals))

    def _fit_near_segment(self, edge, p1, p2):
        h,w=edge.shape[:2]
        mask=np.zeros_like(edge)
        a=(int(round(p1[0])),int(round(p1[1])))
        b=(int(round(p2[0])),int(round(p2[1])))
        cv2.line(mask,a,b,255,max(3,self.search_radius_px*2))
        ys,xs=np.nonzero((edge>0)&(mask>0))
        if len(xs)<30:
            return None
        pts=np.column_stack([xs,ys]).astype(np.float32)
        vx,vy,x0,y0=cv2.fitLine(pts,cv2.DIST_L2,0,0.01,0.01).reshape(-1)
        return np.array([float(vx),float(vy),float(x0),float(y0)],dtype=float)

    @staticmethod
    def _intersection(l1,l2):
        vx1,vy1,x1,y1=l1; vx2,vy2,x2,y2=l2
        A=np.array([[vx1,-vx2],[vy1,-vy2]],dtype=float)
        b=np.array([x2-x1,y2-y1],dtype=float)
        det=np.linalg.det(A)
        if abs(det)<1e-6:
            return None
        t=np.linalg.solve(A,b)[0]
        return (float(x1+t*vx1),float(y1+t*vy1))

    @staticmethod
    def _valid_quad(q,w,h):
        if q is None or len(q)!=4: return False
        pts=np.asarray(q,np.float32)
        if not np.isfinite(pts).all(): return False
        if (pts[:,0] < -0.1*w).any() or (pts[:,0] > 1.1*w).any(): return False
        if (pts[:,1] < -0.1*h).any() or (pts[:,1] > 1.1*h).any(): return False
        area=abs(cv2.contourArea(pts))
        return area > 0.04*w*h

    def refine(self, frame, image_points):
        q=tuple((float(x),float(y)) for x,y in image_points)
        if len(q)!=4:
            return RefinementResult(False,q,0,0,0,"EXPECTED_4_POINTS")

        edge=self._edge_map(frame)
        # CP-0030 order: near-left, near-right, far-right, far-left
        nl,nr,fr,fl=q
        original=np.mean([
            self._support(edge,nl,nr),
            self._support(edge,nr,fr),
            self._support(edge,fr,fl),
            self._support(edge,fl,nl),
        ])

        near=self._fit_near_segment(edge,nl,nr)
        right=self._fit_near_segment(edge,nr,fr)
        far=self._fit_near_segment(edge,fr,fl)
        left=self._fit_near_segment(edge,fl,nl)

        if any(v is None for v in (near,right,far,left)):
            return RefinementResult(False,q,original,original,0,"INSUFFICIENT_EDGE_SUPPORT")

        refined=(
            self._intersection(left,near),
            self._intersection(near,right),
            self._intersection(right,far),
            self._intersection(far,left),
        )

        h,w=edge.shape[:2]
        if not self._valid_quad(refined,w,h) or any(p is None for p in refined):
            return RefinementResult(False,q,original,original,0,"INVALID_REFINED_GEOMETRY")

        rnl,rnr,rfr,rfl=refined
        score=np.mean([
            self._support(edge,rnl,rnr),
            self._support(edge,rnr,rfr),
            self._support(edge,rfr,rfl),
            self._support(edge,rfl,rnl),
        ])
        gain=float(score-original)

        if gain < self.min_gain:
            return RefinementResult(False,q,float(original),float(score),gain,"NO_MEASURABLE_IMPROVEMENT")

        return RefinementResult(
            True, tuple(refined), float(original), float(score), gain,
            "GEOMETRY_REFINEMENT_ACCEPTED"
        )
