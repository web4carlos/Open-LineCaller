from __future__ import annotations

import math
import numpy as np

from .models import LineSegment


def line_coefficients(seg: LineSegment) -> tuple[float, float, float]:
    # ax + by + c = 0
    a = seg.y1 - seg.y2
    b = seg.x2 - seg.x1
    c = seg.x1 * seg.y2 - seg.x2 * seg.y1
    norm = math.hypot(a, b)
    if norm == 0:
        raise ValueError("Degenerate line")
    return a / norm, b / norm, c / norm


def line_intersection(a: LineSegment, b: LineSegment):
    a1, b1, c1 = line_coefficients(a)
    a2, b2, c2 = line_coefficients(b)

    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:
        return None

    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return float(x), float(y)


def polygon_area(points):
    pts = np.asarray(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(
        float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    )


def order_quad(points):
    pts = np.asarray(points, dtype=float)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
    ordered = pts[np.argsort(angles)]

    # rotate so bottom-left (largest y, then smallest x) starts first
    start = max(range(4), key=lambda i: (ordered[i,1], -ordered[i,0]))
    ordered = np.roll(ordered, -start, axis=0)

    # ensure sequence: bottom-left, bottom-right, top-right, top-left
    if ordered[1,0] < ordered[-1,0]:
        ordered = np.array([ordered[0], ordered[-1], ordered[-2], ordered[-3]])

    return tuple((float(x), float(y)) for x, y in ordered)
