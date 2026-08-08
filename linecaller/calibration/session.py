from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2

from linecaller.calibration.court_model import PickleballCourtModel
from linecaller.calibration.service import CalibrationBuildResult, CalibrationService


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
    def __init__(
        self,
        *,
        court: PickleballCourtModel | None = None,
        service: CalibrationService | None = None,
        reference_names: tuple[str, ...] = DEFAULT_REFERENCE_NAMES,
    ):
        self.court = court or PickleballCourtModel()
        self.service = service or CalibrationService()
        self.reference_names = reference_names
        self.image_points: list[tuple[float, float]] = []
        self.result: CalibrationBuildResult | None = None

    @property
    def complete(self) -> bool:
        return len(self.image_points) == len(self.reference_names)

    @property
    def next_reference_name(self) -> str | None:
        if self.complete:
            return None
        return self.reference_names[len(self.image_points)]

    def add_point(self, x: float, y: float) -> None:
        if self.complete:
            raise ValueError("All calibration reference points are already selected")
        self.image_points.append((float(x), float(y)))
        self.result = None

    def undo(self) -> None:
        if self.image_points:
            self.image_points.pop()
        self.result = None

    def reset(self) -> None:
        self.image_points.clear()
        self.result = None

    def court_points(self) -> list[tuple[float, float]]:
        refs = self.court.reference_points()
        return [refs[name] for name in self.reference_names]

    def build(
        self,
        *,
        name: str,
        image_size: tuple[int, int],
    ) -> CalibrationBuildResult:
        if not self.complete:
            raise ValueError(
                f"Need {len(self.reference_names)} points; "
                f"have {len(self.image_points)}"
            )

        self.result = self.service.build(
            name=name,
            image_size=image_size,
            image_points=list(self.image_points),
            court_points=self.court_points(),
        )
        return self.result

    def overlay_segments(self) -> tuple[OverlaySegment, ...]:
        if self.result is None:
            return ()

        h = np.asarray(self.result.profile.homography, dtype=np.float64)
        inverse = np.linalg.inv(h)

        segments: list[OverlaySegment] = []

        for name, (c1, c2) in self.court.line_segments().items():
            court_pts = np.asarray([[c1, c2]], dtype=np.float64)
            image_pts = cv2.perspectiveTransform(court_pts, inverse)[0]

            p1 = (float(image_pts[0][0]), float(image_pts[0][1]))
            p2 = (float(image_pts[1][0]), float(image_pts[1][1]))

            segments.append(OverlaySegment(name=name, p1=p1, p2=p2))

        return tuple(segments)
