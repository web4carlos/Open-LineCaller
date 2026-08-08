import math

from linecaller.autocalibration.hypothesis import build_hypothesis
from linecaller.autocalibration.line_detection import normalized_angle_deg
from linecaller.autocalibration.models import LineSegment


def seg(x1,y1,x2,y2):
    return LineSegment(
        x1,y1,x2,y2,
        math.hypot(x2-x1,y2-y1),
        normalized_angle_deg(x1,y1,x2,y2),
    )


def test_build_hypothesis():
    a1=seg(100,800,900,800)
    a2=seg(250,150,750,150)
    b1=seg(100,800,250,150)
    b2=seg(900,800,750,150)

    h=build_hypothesis(
        a1,a2,b1,b2,
        image_width=1000,
        image_height=900,
        separation_deg=75,
    )

    assert h is not None
    assert len(h.corners)==4
    assert h.score > 0
