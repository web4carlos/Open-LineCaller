import math

from linecaller.autocalibration.hypothesis_search import HypothesisSearch
from linecaller.autocalibration.line_detection import normalized_angle_deg
from linecaller.autocalibration.models import LineSegment


def seg(x1,y1,x2,y2):
    return LineSegment(
        x1,y1,x2,y2,
        math.hypot(x2-x1,y2-y1),
        normalized_angle_deg(x1,y1,x2,y2),
    )


def test_search_generates_hypothesis():
    family_a=[
        seg(100,800,900,800),
        seg(250,150,750,150),
    ]
    family_b=[
        seg(100,800,250,150),
        seg(900,800,750,150),
    ]

    hs=HypothesisSearch().generate(
        family_a=family_a,
        family_b=family_b,
        image_width=1000,
        image_height=900,
        separation_deg=75,
    )

    assert hs
    assert hs[0].score > 0
