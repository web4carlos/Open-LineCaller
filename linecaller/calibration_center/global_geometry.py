from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class GlobalGeometryResult:
    accepted: bool
    image_points: tuple[tuple[float, float], ...]
    original_score: float
    best_score: float
    gain: float
    candidate_count: int
    reason: str


class GlobalPickleballGeometryFitter:
    """
    CP-0030.5 — Global Pickleball Geometry Fitting

    Unlike local edge snapping, this stage scores the complete court hypothesis.

    The court model supplies expected parallel court lines at:
      y = 0, 15, 22, 29, 44 ft
      x = 0, 10, 20 ft (centerline only in service areas)

    Candidate outer quadrilaterals are generated near the current calibration.
    Each candidate homography projects ALL expected pickleball lines into the
    image. The complete hypothesis is then scored against color-agnostic edge
    evidence.

    A candidate is accepted only when the global score improves enough.
    """

    COURT_W = 20.0
    COURT_L = 44.0

    def __init__(
        self,
        search_px: int = 18,
        step_px: int = 6,
        min_gain: float = 0.025,
        boundary_weight: float = 1.0,
        interior_weight: float = 1.35,
    ):
        self.search_px = int(search_px)
        self.step_px = max(1, int(step_px))
        self.min_gain = float(min_gain)
        self.boundary_weight = float(boundary_weight)
        self.interior_weight = float(interior_weight)

    @staticmethod
    def _edges(frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        chans = list(cv2.split(lab)) + [cv2.split(hsv)[1]]
        maps = []
        for ch in chans:
            ch = cv2.GaussianBlur(ch, (5, 5), 0)
            maps.append(cv2.Canny(ch, 35, 120))
        return np.maximum.reduce(maps)

    @staticmethod
    def _H(q):
        # q order: near-left, near-right, far-right, far-left
        src = np.array(
            [[0.,44.], [20.,44.], [20.,0.], [0.,0.]],
            dtype=np.float32,
        )
        dst = np.asarray(q, dtype=np.float32)
        return cv2.getPerspectiveTransform(src, dst)

    @staticmethod
    def _project(H, pts):
        a = np.asarray([[[float(x), float(y)]] for x, y in pts], np.float32)
        z = cv2.perspectiveTransform(a, H)
        return [(float(p[0][0]), float(p[0][1])) for p in z]

    @staticmethod
    def _segment_support(edge, p1, p2, band=2, n=160):
        h, w = edge.shape[:2]
        xs = np.linspace(p1[0], p2[0], n)
        ys = np.linspace(p1[1], p2[1], n)
        hit = 0
        valid = 0
        for x, y in zip(xs, ys):
            xi = int(round(x)); yi = int(round(y))
            if xi < 0 or yi < 0 or xi >= w or yi >= h:
                continue
            valid += 1
            x0=max(0,xi-band); x1=min(w,xi+band+1)
            y0=max(0,yi-band); y1=min(h,yi+band+1)
            if edge[y0:y1,x0:x1].max(initial=0) > 0:
                hit += 1
        return float(hit / valid) if valid else 0.0

    def _score(self, edge, q):
        H = self._H(q)

        outer = [
            ((0,0),(20,0)),
            ((0,44),(20,44)),
            ((0,0),(0,44)),
            ((20,0),(20,44)),
        ]
        interior = [
            ((0,15),(20,15)),   # far NVZ
            ((0,22),(20,22)),   # net axis / geometry constraint
            ((0,29),(20,29)),   # near NVZ
            ((10,0),(10,15)),   # far center
            ((10,29),(10,44)),  # near center
        ]

        total = 0.0
        weights = 0.0

        for a,b in outer:
            p1,p2 = self._project(H,[a,b])
            total += self.boundary_weight*self._segment_support(edge,p1,p2)
            weights += self.boundary_weight

        # Interior geometry is deliberately weighted more strongly because it
        # helps reject nearby tennis/multisport rectangles.
        for a,b in interior:
            p1,p2 = self._project(H,[a,b])
            total += self.interior_weight*self._segment_support(edge,p1,p2)
            weights += self.interior_weight

        return total/weights if weights else 0.0

    @staticmethod
    def _valid(q, w, h):
        a=np.asarray(q,np.float32)
        if a.shape != (4,2) or not np.isfinite(a).all():
            return False
        if abs(cv2.contourArea(a)) < 0.04*w*h:
            return False

        nl,nr,fr,fl=a
        # Basic perspective ordering.
        if nl[0] >= nr[0] or fl[0] >= fr[0]:
            return False
        if (nl[1]+nr[1])/2 <= (fl[1]+fr[1])/2:
            return False
        return True

    def _offsets(self):
        vals = list(range(-self.search_px, self.search_px+1, self.step_px))
        if 0 not in vals:
            vals.append(0)
        return sorted(set(vals))

    def fit(self, frame, image_points):
        q0=tuple((float(x),float(y)) for x,y in image_points)
        if len(q0) != 4:
            return GlobalGeometryResult(False,q0,0,0,0,0,"EXPECTED_4_POINTS")

        edge=self._edges(frame)
        h,w=edge.shape[:2]
        original=self._score(edge,q0)

        # Search vertical offsets for near/far baselines and horizontal offsets
        # for left/right sidelines. This preserves a stable low-dimensional
        # search while allowing the four physical boundaries to move.
        offsets=self._offsets()
        best_q=q0
        best=original
        count=0

        nl,nr,fr,fl=q0
        for near_dy, far_dy, left_dx, right_dx in product(offsets, repeat=4):
            count += 1
            q=(
                (nl[0]+left_dx,  nl[1]+near_dy),
                (nr[0]+right_dx, nr[1]+near_dy),
                (fr[0]+right_dx, fr[1]+far_dy),
                (fl[0]+left_dx,  fl[1]+far_dy),
            )
            if not self._valid(q,w,h):
                continue
            s=self._score(edge,q)
            if s > best:
                best=s
                best_q=q

        gain=float(best-original)
        if gain < self.min_gain:
            return GlobalGeometryResult(
                False,q0,float(original),float(best),gain,count,
                "NO_GLOBAL_GEOMETRY_IMPROVEMENT"
            )

        return GlobalGeometryResult(
            True,tuple(best_q),float(original),float(best),gain,count,
            "GLOBAL_PICKLEBALL_GEOMETRY_ACCEPTED"
        )
