from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from linecaller.court_geometry import CourtCalibration


@dataclass(frozen=True)
class AutoCalibrationResult:
    success: bool
    confidence: float
    calibration: CourtCalibration | None
    reason: str


class AutoCourtCalibrator:
    """
    CP-0030.3 Hybrid Auto Calibration

    Detection order:
      1. blue/cyan playing-surface detector
      2. perspective white-line detector
      3. caller/UI falls back to manual calibration

    Surface detection is preferred for real footage.
    Line detection preserves CP-0030/0030.1 compatibility and works when
    the court surface cannot be segmented reliably.
    """

    def __init__(self, *, min_confidence: float = 0.55):
        self.min_confidence = float(min_confidence)

    @staticmethod
    def _length(x1, y1, x2, y2):
        return float(np.hypot(x2 - x1, y2 - y1))

    @staticmethod
    def _angle(x1, y1, x2, y2):
        a = abs(float(np.degrees(np.arctan2(y2-y1, x2-x1))))
        return 180.0-a if a > 90.0 else a

    @staticmethod
    def _x_at_y(line, target_y):
        x1, y1, x2, y2, _ = line
        dy = y2-y1
        if abs(dy) < 1e-9:
            return None
        return float(x1 + ((target_y-y1)/dy)*(x2-x1))

    @staticmethod
    def _order_quad(pts):
        pts = np.asarray(pts, dtype=np.float32)
        idx = np.argsort(pts[:, 1])
        far = pts[idx[:2]]
        near = pts[idx[-2:]]
        far = far[np.argsort(far[:, 0])]
        near = near[np.argsort(near[:, 0])]
        fl, fr = far
        nl, nr = near
        return (
            (float(nl[0]), float(nl[1])),
            (float(nr[0]), float(nr[1])),
            (float(fr[0]), float(fr[1])),
            (float(fl[0]), float(fl[1])),
        )

    def _surface_candidate(self, frame):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            np.array([82, 28, 25], dtype=np.uint8),
            np.array([140, 255, 255], dtype=np.uint8),
        )

        k = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours[:10]:
            area = float(cv2.contourArea(c))
            if area < 0.06*w*h:
                continue

            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)

            for eps in (0.015, 0.02, 0.03, 0.04, 0.055, 0.075):
                approx = cv2.approxPolyDP(hull, eps*peri, True)
                if len(approx) != 4:
                    continue

                quad = self._order_quad(approx.reshape(4, 2))
                ok, score, reason = self._validate_quad(quad, w, h)
                if ok:
                    area_score = min(1.0, area/(0.38*w*h))
                    return quad, max(score, 0.55*area_score), "SURFACE"

        return None, 0.0, "NO_SURFACE_CANDIDATE"

    def _validate_quad(self, quad, w, h):
        nl, nr, fr, fl = quad
        near_w = nr[0]-nl[0]
        far_w = fr[0]-fl[0]
        near_y = (nl[1]+nr[1])/2.0
        far_y = (fl[1]+fr[1])/2.0
        depth = near_y-far_y

        if near_w <= 0 or far_w <= 0:
            return False, 0.0, "INVALID_ORDER"
        if near_w < 0.25*w or far_w < 0.08*w:
            return False, 0.0, "TOO_NARROW"
        if depth < 0.10*h:
            return False, 0.0, "TOO_SHALLOW"
        if far_w > 1.15*near_w:
            return False, 0.0, "BAD_PERSPECTIVE"

        width_score = min(1.0, near_w/(0.70*w))
        depth_score = min(1.0, depth/(0.35*h))
        taper = far_w/max(near_w, 1.0)
        taper_score = max(0.0, 1.0-abs(taper-0.50)/0.65)
        score = 0.40*width_score + 0.35*depth_score + 0.25*taper_score
        return True, float(np.clip(score, 0, 1)), "OK"

    def _line_candidate(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 50, 155)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            threshold=55,
            minLineLength=max(55, int(w*0.06)),
            maxLineGap=55,
        )
        if lines is None:
            return None, 0.0, "NO_LINES"

        horizontal, left, right = [], [], []

        for item in lines:
            x1, y1, x2, y2 = map(int, item.reshape(-1))
            ln = self._length(x1,y1,x2,y2)
            angle = self._angle(x1,y1,x2,y2)
            row = (x1,y1,x2,y2,ln)

            if angle <= 20:
                horizontal.append(row)
            elif 20 < angle <= 78:
                if y1 >= y2:
                    bx, tx = x1, x2
                else:
                    bx, tx = x2, x1
                if tx > bx:
                    left.append(row)
                elif tx < bx:
                    right.append(row)

        if len(horizontal) < 2 or not left or not right:
            return None, 0.0, "INSUFFICIENT_LINE_SUPPORT"

        horizontal = sorted(horizontal, key=lambda r:r[4], reverse=True)[:35]
        left = sorted(left, key=lambda r:r[4], reverse=True)[:30]
        right = sorted(right, key=lambda r:r[4], reverse=True)[:30]

        ys = [
            (r[1]+r[3])/2.0 for r in horizontal
            if (r[1]+r[3])/2.0 > 0.15*h
        ]
        if len(ys) < 2:
            return None, 0.0, "INSUFFICIENT_BASELINES"

        # Try multiple baseline percentile pairs instead of one fragile choice.
        pairs = ((95,20),(95,30),(90,20),(90,35),(85,20),(85,35))
        best = None

        for near_p, far_p in pairs:
            near_y = float(np.percentile(ys, near_p))
            far_y = float(np.percentile(ys, far_p))
            if near_y-far_y < 0.10*h:
                continue

            def projections(group):
                near, far = [], []
                for r in group:
                    xn = self._x_at_y(r, near_y)
                    xf = self._x_at_y(r, far_y)
                    if xn is not None and np.isfinite(xn): near.append(xn)
                    if xf is not None and np.isfinite(xf): far.append(xf)
                return near, far

            ln, lf = projections(left)
            rn, rf = projections(right)
            if not (ln and lf and rn and rf):
                continue

            # Robust central estimates; Hough often yields both edges of a stripe.
            nl = float(np.median(ln))
            fl = float(np.median(lf))
            nr = float(np.median(rn))
            fr = float(np.median(rf))

            quad = ((nl,near_y),(nr,near_y),(fr,far_y),(fl,far_y))
            ok, geometry_score, _ = self._validate_quad(quad, w, h)
            if not ok:
                continue

            support = min(
                1.0,
                (min(len(horizontal),12)+min(len(left),8)+min(len(right),8))/24.0
            )
            score = 0.72*geometry_score + 0.28*support
            if best is None or score > best[1]:
                best = (quad, score)

        if best is None:
            return None, 0.0, "NO_VALID_LINE_QUAD"

        return best[0], float(np.clip(best[1],0,1)), "LINES"

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return AutoCalibrationResult(False, 0.0, None, "EMPTY_FRAME")

        # 1) Preferred real-footage cue.
        quad, score, method = self._surface_candidate(frame)

        # 2) Compatibility + fallback cue.
        if quad is None:
            quad, score, method = self._line_candidate(frame)

        if quad is None:
            return AutoCalibrationResult(
                False, float(score), None,
                f"AUTO_FAILED:{method}"
            )

        calibration = CourtCalibration(image_points=quad)
        success = score >= self.min_confidence

        return AutoCalibrationResult(
            success=success,
            confidence=float(np.clip(score,0,1)),
            calibration=calibration,
            reason=(
                f"OK:{method}"
                if success
                else f"LOW_CONFIDENCE:{method}"
            ),
        )
