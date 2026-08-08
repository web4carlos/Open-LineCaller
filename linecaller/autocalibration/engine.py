from __future__ import annotations

import numpy as np

from .line_detection import HoughCourtLineDetector
from .orientation import OrientationFamilies
from .proposal import CourtProposalBuilder


class AutoCalibrationEngine:
    def __init__(
        self,
        *,
        line_detector=None,
        orientation_families=None,
        proposal_builder=None,
    ):
        self.line_detector = line_detector or HoughCourtLineDetector()
        self.orientation_families = orientation_families or OrientationFamilies()
        self.proposal_builder = proposal_builder or CourtProposalBuilder()

    def propose(self, frame: np.ndarray):
        lines = self.line_detector.detect(frame)
        family_a, family_b, separation = self.orientation_families.split(lines)

        h, w = frame.shape[:2]

        return self.proposal_builder.build(
            family_a=family_a,
            family_b=family_b,
            separation_deg=separation,
            image_width=w,
            image_height=h,
            total_line_count=len(lines),
        )
