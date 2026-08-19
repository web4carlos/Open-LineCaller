from __future__ import annotations

import math
import pytest

from linecaller.flight_paths.models import CourtFrame, Point3D, TimedPoint3D
from linecaller.flight_paths.hit_anchored_z0 import HitAnchoredRollingZ0


def tp(x, y, z, t):
    return TimedPoint3D(Point3D(float(x), float(y), float(z)), float(t))


def ballistic(p0, vx, vy, vz, t, g):
    return tp(
        p0.x + vx*t,
        p0.y + vy*t,
        p0.z + vz*t - 0.5*g*t*t,
        t,
    )


def true_contact(p0, vx, vy, vz, g):
    dt = (vz + math.sqrt(vz*vz + 2*g*p0.z)) / g
    return p0.x + vx*dt, p0.y + vy*dt, dt


def test_hit_is_first_point_of_first_window():
    e = HitAnchoredRollingZ0()
    h = tp(10, 100, 8, 1.0)
    e.start_epoch(h)
    assert e.state.points == (h,)
    assert e.add(tp(11, 95, 9, 1.03)) is None
    pred = e.add(tp(12, 90, 9, 1.06))
    assert pred is not None
    assert pred.observations[0] == h


def test_window_moves_one_point_at_a_time_and_never_exceeds_three():
    e = HitAnchoredRollingZ0()
    p0 = tp(10, 100, 8, 1.00)
    p1 = tp(11, 95, 9, 1.03)
    p2 = tp(12, 90, 9, 1.06)
    p3 = tp(13, 85, 8, 1.09)
    e.start_epoch(p0)
    e.add(p1)
    first = e.add(p2)
    second = e.add(p3)
    assert first is not None and second is not None
    assert first.observations == (p0, p1, p2)
    assert second.observations == (p1, p2, p3)
    assert len(e.state.points) == 3


def test_new_hit_resets_old_window_but_starts_next_epoch():
    e = HitAnchoredRollingZ0()
    e.start_epoch(tp(10, 100, 8, 1.00))
    e.add(tp(11, 95, 9, 1.03))
    e.add(tp(12, 90, 9, 1.06))
    old_epoch = e.state.epoch
    new_hit = tp(50, 20, 5, 2.00)
    e.start_epoch(new_hit)
    assert e.state.epoch == old_epoch + 1
    assert e.state.points == (new_hit,)
    assert e.state.prediction is None


def test_exact_ballistic_three_points_predict_true_z0():
    court = CourtFrame()
    g = court.gravity_bu_s2
    p0 = Point3D(20.0, 150.0, 12.0)
    vx, vy, vz = 40.0, -120.0, 28.0
    xz, yz, dtz = true_contact(p0, vx, vy, vz, g)

    e = HitAnchoredRollingZ0(gravity_bu_s2=g)
    e.start_epoch(ballistic(p0, vx, vy, vz, 0.0, g))
    e.add(ballistic(p0, vx, vy, vz, 0.02, g))
    pred = e.add(ballistic(p0, vx, vy, vz, 0.04, g))
    assert pred is not None
    assert pred.landing.x == pytest.approx(xz, abs=1e-6)
    assert pred.landing.y == pytest.approx(yz, abs=1e-6)
    assert pred.contact_time_s == pytest.approx(dtz, abs=1e-6)


def test_later_exact_ballistic_window_keeps_same_z0():
    court = CourtFrame()
    g = court.gravity_bu_s2
    p0 = Point3D(20.0, 150.0, 12.0)
    vx, vy, vz = 40.0, -120.0, 28.0
    xz, yz, _ = true_contact(p0, vx, vy, vz, g)

    e = HitAnchoredRollingZ0(gravity_bu_s2=g)
    e.start_epoch(ballistic(p0, vx, vy, vz, 0.0, g))
    predictions = []
    for t in (0.02, 0.04, 0.06, 0.08, 0.10):
        p = e.add(ballistic(p0, vx, vy, vz, t, g))
        if p is not None:
            predictions.append(p)
    assert len(predictions) == 4
    for pred in predictions:
        assert pred.landing.x == pytest.approx(xz, abs=1e-5)
        assert pred.landing.y == pytest.approx(yz, abs=1e-5)


def test_time_must_increase():
    e = HitAnchoredRollingZ0()
    e.start_epoch(tp(10, 100, 8, 1.0))
    with pytest.raises(ValueError):
        e.add(tp(11, 95, 9, 1.0))


def test_no_complete_path_object_is_stored():
    e = HitAnchoredRollingZ0()
    e.start_epoch(tp(10, 100, 8, 1.00))
    e.add(tp(11, 95, 9, 1.03))
    e.add(tp(12, 90, 9, 1.06))
    state = e.state
    assert len(state.points) == 3
    assert not hasattr(state, "path")
