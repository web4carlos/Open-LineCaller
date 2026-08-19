import inspect
import math

from linecaller.flight_paths.models import CourtFrame, Point3D, TimedPoint3D
from linecaller.flight_paths.rolling_z0 import (
    RollingThreePointZ0Predictor,
    RollingZ0Tracker,
)
import linecaller.flight_paths.rolling_z0 as rolling_module


def ballistic_point(p0, v0, t, g):
    return Point3D(
        p0.x + v0[0] * t,
        p0.y + v0[1] * t,
        p0.z + v0[2] * t - 0.5 * g * t * t,
    )


def test_core_uses_three_timed_points_and_no_complete_path_object():
    source = inspect.getsource(rolling_module)
    assert "from .path_field import FlightPath" not in source
    assert "CellIndex" not in source
    assert "ball_reference" not in source
    assert "cv2" not in source


def test_exact_ballistic_three_points_recover_z0_from_latest_state():
    court = CourtFrame()
    g = court.gravity_bu_s2
    p0 = Point3D(20.0, 150.0, 12.0)
    v0 = (35.0, -120.0, 42.0)
    times = (0.00, 0.04, 0.08)
    obs = tuple(TimedPoint3D(ballistic_point(p0, v0, t, g), 10.0 + t) for t in times)

    pred = RollingThreePointZ0Predictor(gravity_bu_s2=g).predict(obs)

    # Compare with the exact future landing from the latest state.
    latest = obs[-1]
    exact_vz = v0[2] - g * times[-1]
    dt = (exact_vz + math.sqrt(exact_vz**2 + 2*g*latest.z)) / g
    assert math.isclose(pred.time_to_z0_s, dt, abs_tol=1e-8)
    assert math.isclose(pred.landing.x, latest.x + v0[0]*dt, abs_tol=1e-8)
    assert math.isclose(pred.landing.y, latest.y + v0[1]*dt, abs_tol=1e-8)
    assert pred.landing.z == 0.0


def test_tracker_keeps_only_latest_three_points():
    tracker = RollingZ0Tracker()
    pts = [
        TimedPoint3D(Point3D(10+i, 100-i, 8+i), 1.0 + i*0.03)
        for i in range(5)
    ]
    assert tracker.add(pts[0]) is None
    assert tracker.add(pts[1]) is None
    assert tracker.add(pts[2]) is not None
    tracker.add(pts[3])
    tracker.add(pts[4])
    assert tracker.points == tuple(pts[-3:])


def test_new_flight_reset_discards_old_three_point_window():
    tracker = RollingZ0Tracker()
    for i in range(3):
        tracker.add(TimedPoint3D(Point3D(10+i, 100-i, 8+i), 1+i*0.03))
    assert tracker.ready
    tracker.reset()
    assert tracker.points == ()
    assert not tracker.ready


def test_each_new_observation_recomputes_from_shifted_window():
    tracker = RollingZ0Tracker()
    a = TimedPoint3D(Point3D(20, 100, 12), 1.00)
    b = TimedPoint3D(Point3D(21, 96, 11), 1.04)
    c = TimedPoint3D(Point3D(22, 92, 9), 1.08)
    d = TimedPoint3D(Point3D(23, 88, 6), 1.12)
    tracker.add(a); tracker.add(b)
    first = tracker.add(c)
    second = tracker.add(d)
    assert first is not None and second is not None
    assert first.observations == (a,b,c)
    assert second.observations == (b,c,d)
    assert first.current == c
    assert second.current == d


def test_z0_now_means_zero_time_to_contact():
    pred = RollingThreePointZ0Predictor().predict((
        TimedPoint3D(Point3D(10,10,2), 1.00),
        TimedPoint3D(Point3D(11,11,1), 1.03),
        TimedPoint3D(Point3D(12,12,0), 1.06),
    ))
    assert pred.time_to_z0_s == 0.0
    assert pred.landing == Point3D(12,12,0)
