from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Mapping, Sequence

import cv2
import numpy as np


Point = tuple[float, float]


@dataclass(frozen=True)
class BoundaryLine:
    name: str
    p0: Point
    p1: Point

    @property
    def length_px(self) -> float:
        return math.dist(self.p0, self.p1)


@dataclass(frozen=True)
class BoundaryInference:
    image_points: tuple[Point, Point, Point, Point]
    offscreen_corners: int
    lines: tuple[BoundaryLine, BoundaryLine, BoundaryLine, BoundaryLine]


_BOUNDARY_ORDER = ("NEAR", "RIGHT", "FAR", "LEFT")


def _point(value: Sequence[float], label: str, image_size: tuple[int, int]) -> Point:
    if len(value) != 2:
        raise ValueError(f"{label} must contain x,y")
    x, y = float(value[0]), float(value[1])
    width, height = image_size
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"{label} contains a non-finite coordinate")
    if not (0.0 <= x < width and 0.0 <= y < height):
        raise ValueError(
            f"{label} must be clicked on a visible part of the boundary line"
        )
    return (x, y)


def parse_boundary_lines_json(
    raw: str,
    image_size: tuple[int, int],
) -> tuple[BoundaryLine, BoundaryLine, BoundaryLine, BoundaryLine]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("boundary_lines must be valid JSON") from exc

    if not isinstance(value, Mapping):
        raise ValueError(
            "boundary_lines must contain NEAR, RIGHT, FAR and LEFT"
        )

    width, height = image_size
    min_line = max(8.0, 0.0125 * math.hypot(width, height))
    out: list[BoundaryLine] = []

    for name in _BOUNDARY_ORDER:
        pair = value.get(name)
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(
                f"{name} boundary needs exactly two visible clicks"
            )
        p0 = _point(pair[0], f"{name}[0]", image_size)
        p1 = _point(pair[1], f"{name}[1]", image_size)
        line = BoundaryLine(name, p0, p1)
        if line.length_px < min_line:
            raise ValueError(
                f"{name} boundary clicks are too close together"
            )
        out.append(line)

    return tuple(out)  # type: ignore[return-value]


def _homogeneous_line(line: BoundaryLine) -> np.ndarray:
    a = np.asarray((line.p0[0], line.p0[1], 1.0), dtype=np.float64)
    b = np.asarray((line.p1[0], line.p1[1], 1.0), dtype=np.float64)
    result = np.cross(a, b)
    norm = math.hypot(float(result[0]), float(result[1]))
    if norm <= 1e-12:
        raise ValueError(f"{line.name} boundary is degenerate")
    return result / norm


def _intersection(a: BoundaryLine, b: BoundaryLine) -> Point:
    la = _homogeneous_line(a)
    lb = _homogeneous_line(b)
    q = np.cross(la, lb)
    w = float(q[2])
    if abs(w) <= 1e-6:
        raise ValueError(
            f"{a.name} and {b.name} boundaries are too parallel to infer a corner"
        )
    x, y = float(q[0] / w), float(q[1] / w)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(
            f"{a.name}/{b.name} produced an invalid corner"
        )
    return (x, y)


def _distance_outside_frame(point: Point, image_size: tuple[int, int]) -> float:
    x, y = point
    width, height = image_size
    dx = 0.0 if 0.0 <= x <= width else (-x if x < 0.0 else x - width)
    dy = 0.0 if 0.0 <= y <= height else (-y if y < 0.0 else y - height)
    return math.hypot(dx, dy)


def _is_offscreen(point: Point, image_size: tuple[int, int]) -> bool:
    x, y = point
    width, height = image_size
    return not (0.0 <= x < width and 0.0 <= y < height)


def infer_court_corners_from_boundary_lines(
    lines: Sequence[BoundaryLine],
    image_size: tuple[int, int],
) -> BoundaryInference:
    if len(lines) != 4:
        raise ValueError("Exactly four named boundary lines are required")

    by_name = {line.name: line for line in lines}
    if set(by_name) != set(_BOUNDARY_ORDER):
        raise ValueError("Boundary lines must be NEAR, RIGHT, FAR and LEFT")

    near = by_name["NEAR"]
    right = by_name["RIGHT"]
    far = by_name["FAR"]
    left = by_name["LEFT"]

    corners = (
        _intersection(near, left),
        _intersection(near, right),
        _intersection(far, right),
        _intersection(far, left),
    )

    width, height = image_size
    diagonal = math.hypot(width, height)
    max_offscreen_distance = 1.5 * diagonal
    for i, corner in enumerate(corners):
        if _distance_outside_frame(corner, image_size) > max_offscreen_distance:
            raise ValueError(
                f"Inferred corner {i + 1} is implausibly far outside the frame; "
                "reselect the boundary-line points"
            )

    polygon = np.asarray(corners, dtype=np.float32)
    area = abs(float(cv2.contourArea(polygon)))
    if area < max(250.0, 0.002 * width * height):
        raise ValueError("Inferred court quadrilateral is too small")
    if not cv2.isContourConvex(np.rint(polygon).astype(np.int32)):
        raise ValueError(
            "Inferred court corners cross; verify NEAR/RIGHT/FAR/LEFT line labels"
        )

    src = np.asarray(
        ((0.0, 182.0), (83.0, 182.0), (83.0, 0.0), (0.0, 0.0)),
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, polygon)
    if not np.isfinite(H).all() or abs(float(np.linalg.det(H))) <= 1e-12:
        raise ValueError("Inferred court homography is degenerate")

    return BoundaryInference(
        image_points=corners,
        offscreen_corners=sum(
            1 for corner in corners if _is_offscreen(corner, image_size)
        ),
        lines=(near, right, far, left),
    )