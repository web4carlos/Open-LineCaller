from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2

from linecaller.calibration.court_model import PickleballCourtModel
from linecaller.calibration.service import CalibrationBuildResult, CalibrationService
from linecaller.calibration.diagnostics import reprojection_diagnostics

DEFAULT_REFERENCE_NAMES = (
    "near_left_corner",
    "near_right_corner",
    "far_right_corner",
    "far_left_corner",
    "near_nvz_left",
    "near_nvz_right",
    "far_nvz_left",
    "far_nvz_right",
)

@dataclass(frozen=True)
class OverlaySegment:
    name: str
    p1: tuple[float, float]
    p2: tuple[float, float]

class CalibrationSession:
    def __init__(self, court=None, service=None, reference_names=DEFAULT_REFERENCE_NAMES):
        self.court = court or PickleballCourtModel()
        self.service = service or CalibrationService()
        self.reference_names = tuple(reference_names)
        self.image_points: list[tuple[float, float]] = []
        self.result: CalibrationBuildResult | None = None

    @property
    def complete(self):
        return len(self.image_points) == len(self.reference_names)

    @property
    def next_reference_name(self):
        return None if self.complete else self.reference_names[len(self.image_points)]

    def add_point(self, x, y):
        if self.complete:
            raise ValueError("All calibration points are already selected")
        self.image_points.append((float(x), float(y)))
        self.result = None

    def move_point(self, index: int, x: float, y: float):
        if not (0 <= index < len(self.image_points)):
            raise IndexError(index)
        self.image_points[index] = (float(x), float(y))
        self.result = None

    def nearest_point_index(self, x: float, y: float, max_distance_px: float = 18.0):
        if not self.image_points:
            return None
        target = np.asarray([x, y], dtype=float)
        pts = np.asarray(self.image_points, dtype=float)
        d = np.linalg.norm(pts - target, axis=1)
        idx = int(np.argmin(d))
        return idx if float(d[idx]) <= max_distance_px else None

    def undo(self):
        if self.image_points:
            self.image_points.pop()
        self.result = None

    def reset(self):
        self.image_points.clear()
        self.result = None

    def court_points(self):
        refs = self.court.reference_points()
        return [refs[name] for name in self.reference_names]

    def build(self, *, name, image_size):
        if not self.complete:
            raise ValueError(f"Need {len(self.reference_names)} points; have {len(self.image_points)}")
        self.result = self.service.build(
            name=name,
            image_size=image_size,
            image_points=list(self.image_points),
            court_points=self.court_points(),
        )
        return self.result

    def point_diagnostics(self):
        if self.result is None:
            return ()
        return reprojection_diagnostics(
            self.image_points,
            self.court_points(),
            self.result.profile.homography,
        )

    def overlay_segments(self):
        if self.result is None:
            return ()

        h = np.asarray(self.result.profile.homography, dtype=np.float64)
        inverse = np.linalg.inv(h)
        segments = []

        for name, (c1, c2) in self.court.line_segments().items():
            pts = np.asarray([[c1, c2]], dtype=np.float64)
            image_pts = cv2.perspectiveTransform(pts, inverse)[0]
            segments.append(
                OverlaySegment(
                    name=name,
                    p1=(float(image_pts[0][0]), float(image_pts[0][1])),
                    p2=(float(image_pts[1][0]), float(image_pts[1][1])),
                )
            )
        return tuple(segments)
