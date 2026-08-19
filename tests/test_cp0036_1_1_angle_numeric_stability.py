from __future__ import annotations

from linecaller.flight_paths.discontinuity import TrajectoryDiscontinuityDetector
from linecaller.flight_paths.models import Velocity3D


def test_identical_velocity_angle_is_exactly_zero():
    v = Velocity3D(10.0, -80.0, 44.123456789)
    angle = TrajectoryDiscontinuityDetector._angle_deg(v, v)
    assert angle == 0.0


def test_parallel_scaled_velocity_angle_is_effectively_zero():
    a = Velocity3D(10.0, -80.0, 44.123456789)
    b = Velocity3D(20.0, -160.0, 88.246913578)
    angle = TrajectoryDiscontinuityDetector._angle_deg(a, b)
    assert angle < 1e-12