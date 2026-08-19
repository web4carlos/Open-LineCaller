import inspect
import math

from linecaller.flight_paths import CourtFrame, FlightPath, Point3D, TimedPoint3D, Velocity3D
from linecaller.flight_paths.discontinuity import (
    DirectionChangeConfig,
    TrajectoryDiscontinuityDetector,
    TransitionKind,
)
import linecaller.flight_paths.discontinuity as discontinuity_module


def samples(path, times):
    return [TimedPoint3D(path.point_at_time(t), t) for t in times]


def test_smooth_ballistic_path_does_not_start_new_epoch():
    path = FlightPath(Point3D(20, 150, 15), Velocity3D(25, -100, 35), t0_s=5.0)
    obs = samples(path, [5.00, 5.03, 5.06, 5.09, 5.12, 5.15, 5.18])
    detector = TrajectoryDiscontinuityDetector()
    assert detector.scan(obs) == ()


def test_above_floor_redirect_is_new_flight_candidate():
    court = CourtFrame()
    incoming = FlightPath(Point3D(20, 150, 18), Velocity3D(20, -90, 5), t0_s=1.0)
    hit_t = 1.09
    hit = incoming.point_at_time(hit_t)
    outgoing = FlightPath(hit, Velocity3D(-35, 125, 42), t0_s=hit_t)

    before = samples(incoming, [1.00, 1.03, 1.06])
    after = samples(outgoing, [1.09, 1.12, 1.15])
    detector = TrajectoryDiscontinuityDetector(gravity_bu_s2=court.gravity_bu_s2)
    candidate = detector.evaluate(before, after)
    assert candidate is not None
    assert candidate.kind == TransitionKind.ABOVE_FLOOR_DIRECTION_CHANGE
    assert candidate.is_new_flight_candidate
    assert candidate.angle_deg > 24.0


def test_floor_bounce_is_not_classified_as_paddle_direction_change():
    court = CourtFrame()
    incoming = FlightPath(Point3D(35, 100, 3), Velocity3D(4, -20, -18), t0_s=2.0)
    tc = incoming.contact_time_s
    landing = incoming.landing_point
    outgoing = FlightPath(landing, Velocity3D(3, -17, 42), t0_s=tc)

    before_times = [tc - 0.06, tc - 0.03, tc]
    # FlightPath before at exact contact is valid.
    before = samples(incoming, before_times)
    after = samples(outgoing, [tc + 0.001, tc + 0.031, tc + 0.061])
    detector = TrajectoryDiscontinuityDetector(
        DirectionChangeConfig(floor_threshold_bu=1.25),
        gravity_bu_s2=court.gravity_bu_s2,
    )
    candidate = detector.evaluate(before, after)
    assert candidate is not None
    assert candidate.kind == TransitionKind.FLOOR_BOUNCE
    assert not candidate.is_new_flight_candidate


def test_internal_gap_prevents_windows_from_mixing_two_segments():
    detector = TrajectoryDiscontinuityDetector()
    obs = [
        TimedPoint3D(Point3D(10, 100, 8), 0.00),
        TimedPoint3D(Point3D(11, 98, 7), 0.03),
        TimedPoint3D(Point3D(12, 96, 6), 0.06),
        TimedPoint3D(Point3D(20, 70, 15), 0.20),
        TimedPoint3D(Point3D(21, 72, 16), 0.23),
        TimedPoint3D(Point3D(22, 75, 17), 0.26),
    ]
    # The gap between the two three-point windows is allowed as the transition,
    # but no individual fit window may itself bridge that gap.
    candidates = detector.scan(obs)
    assert all(c.transition_gap_s <= detector.config.max_transition_gap_s for c in candidates)


def test_observation_time_is_required_and_ball_identity_is_not_input():
    sig = inspect.signature(TrajectoryDiscontinuityDetector.evaluate)
    assert list(sig.parameters) == ["self", "before", "after"]
    source = inspect.getsource(discontinuity_module)
    for token in ("BallPixelProfile", "ball_reference", "cv2", "YOLO", "CellIndex"):
        assert token not in source


def test_gravity_is_compensated_before_comparing_velocity():
    court = CourtFrame()
    path = FlightPath(Point3D(15, 140, 20), Velocity3D(10, -80, 55), t0_s=3.0)
    before = samples(path, [3.00, 3.03, 3.06])
    after = samples(path, [3.09, 3.12, 3.15])
    detector = TrajectoryDiscontinuityDetector(gravity_bu_s2=court.gravity_bu_s2)
    candidate = detector.evaluate(before, after)
    assert candidate is not None
    assert candidate.kind == TransitionKind.NONE
    assert candidate.angle_deg < 1e-6
    assert candidate.old_path_residual_bu < 1e-6
