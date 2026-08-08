import pytest

from linecaller.autocalibration.models import LineSegment
from linecaller.autocalibration.orientation import (
    OrientationFamilies,
    circular_angle_distance_deg,
)


def line(angle, length=200):
    import math
    r = math.radians(angle)
    return LineSegment(
        0, 0,
        math.cos(r)*length,
        math.sin(r)*length,
        length,
        angle % 180,
    )


def test_circular_angle_distance():
    assert circular_angle_distance_deg(2, 178) == pytest.approx(4)


def test_two_orientation_families():
    lines = [
        line(10), line(11), line(12),
        line(100), line(101), line(99),
    ]

    a, b, sep = OrientationFamilies().split(lines)

    assert len(a) >= 2
    assert len(b) >= 2
    assert sep > 60
