import math

from linecaller.autocalibration.models import LineSegment, AutoCalibrationDisposition
from linecaller.autocalibration.line_detection import normalized_angle_deg
from linecaller.autocalibration.proposal import CourtProposalBuilder


def seg(x1,y1,x2,y2):
    return LineSegment(
        x1,y1,x2,y2,
        math.hypot(x2-x1,y2-y1),
        normalized_angle_deg(x1,y1,x2,y2),
    )


def test_rectangle_proposal_is_plausible():
    # Synthetic perspective-like quadrilateral boundaries.
    family_a = [
        seg(100,800,900,800),
        seg(250,150,750,150),
    ]
    family_b = [
        seg(100,800,250,150),
        seg(900,800,750,150),
    ]

    p = CourtProposalBuilder(
        min_area_ratio=0.05,
        auto_review_threshold=0.1,
        auto_accept_threshold=0.2,
    ).build(
        family_a=family_a,
        family_b=family_b,
        separation_deg=75,
        image_width=1000,
        image_height=900,
        total_line_count=4,
    )

    assert len(p.corners) == 4
    assert p.disposition in (
        AutoCalibrationDisposition.AUTO_ACCEPT,
        AutoCalibrationDisposition.AUTO_REVIEW,
    )


def test_rejects_insufficient_lines():
    p = CourtProposalBuilder().build(
        family_a=[seg(0,0,100,0)],
        family_b=[],
        separation_deg=0,
        image_width=1000,
        image_height=900,
        total_line_count=1,
    )
    assert p.disposition == AutoCalibrationDisposition.REJECT
