import pytest

from linecaller.autocalibration.geometry import (
    line_intersection,
    polygon_area,
    order_quad,
)
from linecaller.autocalibration.models import LineSegment


def seg(x1,y1,x2,y2):
    import math
    from linecaller.autocalibration.line_detection import normalized_angle_deg
    return LineSegment(
        x1,y1,x2,y2,
        math.hypot(x2-x1,y2-y1),
        normalized_angle_deg(x1,y1,x2,y2),
    )


def test_line_intersection():
    a = seg(0,0,100,0)
    b = seg(50,-50,50,50)
    p = line_intersection(a,b)
    assert p[0] == pytest.approx(50)
    assert p[1] == pytest.approx(0)


def test_polygon_area():
    assert polygon_area([(0,0),(10,0),(10,10),(0,10)]) == pytest.approx(100)


def test_order_quad_has_four_points():
    q = order_quad([(10,10),(0,0),(10,0),(0,10)])
    assert len(q) == 4
