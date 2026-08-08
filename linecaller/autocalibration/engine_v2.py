from __future__ import annotations
import numpy as np

from .line_detection import HoughCourtLineDetector
from .orientation import OrientationFamilies
from .hypothesis_search import HypothesisSearch
from .ranking import HypothesisRanker


class MultiHypothesisAutoCalibrationEngine:
    def __init__(
        self,
        *,
        line_detector=None,
        orientation_families=None,
        hypothesis_search=None,
        ranker=None,
    ):
        self.line_detector = line_detector or HoughCourtLineDetector()
        self.orientation_families = orientation_families or OrientationFamilies()
        self.hypothesis_search = hypothesis_search or HypothesisSearch()
        self.ranker = ranker or HypothesisRanker()

    def propose(self, frame: np.ndarray):
        lines = self.line_detector.detect(frame)
        family_a, family_b, separation = self.orientation_families.split(lines)

        h, w = frame.shape[:2]

        hypotheses = self.hypothesis_search.generate(
            family_a=family_a,
            family_b=family_b,
            image_width=w,
            image_height=h,
            separation_deg=separation,
        )

        ranked = self.ranker.rank(hypotheses)

        return {
            "line_count": len(lines),
            "family_a_count": len(family_a),
            "family_b_count": len(family_b),
            "orientation_separation_deg": separation,
            "hypothesis_count": len(hypotheses),
            "ranked": ranked,
            "hypotheses": tuple(hypotheses[:10]),
        }
