import inspect
import math

from linecaller.flight_paths import (
    BALL_UNIT_M,
    COURT_LENGTH_BALL_UNITS,
    CourtFrame,
    CourtPathField,
    CourtSide,
    FlightEpochManager,
    FlightPath,
    LateralSide,
    Point3D,
    TimedPoint3D,
    Velocity3D,
)
import linecaller.flight_paths.path_field as path_field_module


def assert_point_close(a, b, tol=1e-7):
    assert math.isclose(a.x, b.x, abs_tol=tol)
    assert math.isclose(a.y, b.y, abs_tol=tol)
    assert math.isclose(a.z, b.z, abs_tol=tol)


def test_xyz_tern_is_one_point_not_a_figure():
    p = Point3D(12.5, 77.0, 8.0)
    assert (p.x, p.y, p.z) == (12.5, 77.0, 8.0)


def test_one_ball_unit_matches_locked_x_scale():
    court = CourtFrame()
    assert math.isclose(court.width_bu, 83.0, abs_tol=1e-12)
    assert math.isclose(court.ball_unit_in, 2.8915662650602414, abs_tol=1e-9)
    assert math.isclose(BALL_UNIT_M, 6.096 / 83.0, abs_tol=1e-12)


def test_same_ball_unit_is_used_for_continuous_y_and_z():
    court = CourtFrame()
    assert math.isclose(court.length_bu, COURT_LENGTH_BALL_UNITS, abs_tol=1e-12)
    assert math.isclose(court.length_bu, 182.6, abs_tol=1e-9)
    assert court.gravity_bu_s2 > 100.0


def test_path_core_has_no_mesh_cell_or_ball_appearance_dependency():
    source = inspect.getsource(path_field_module)
    for token in ("CellIndex", "ball_reference", "BallPixelProfile", "cv2", "YOLO"):
        assert token not in source


def test_time_is_part_of_the_path_contract():
    sig = inspect.signature(CourtPathField.select_from_state)
    assert "t0_s" in sig.parameters
    assert sig.parameters["t0_s"].default is inspect.Parameter.empty


def test_velocity_uses_position_change_over_real_time():
    a = TimedPoint3D(Point3D(10.0, 100.0, 12.0), 4.000)
    b = TimedPoint3D(Point3D(12.0, 96.0, 13.0), 4.040)
    v = Velocity3D.between(a, b)
    assert math.isclose(v.vx, 50.0)
    assert math.isclose(v.vy, -100.0)
    assert math.isclose(v.vz, 25.0)


def test_same_positions_with_different_dt_produce_different_velocity():
    p0 = Point3D(10.0, 100.0, 12.0)
    p1 = Point3D(12.0, 96.0, 13.0)
    fast = Velocity3D.between(TimedPoint3D(p0, 1.0), TimedPoint3D(p1, 1.02))
    slow = Velocity3D.between(TimedPoint3D(p0, 1.0), TimedPoint3D(p1, 1.10))
    assert fast.speed_bu_s > slow.speed_bu_s


def test_one_p0_can_join_multiple_possible_z0_points():
    field = CourtPathField()
    p0 = Point3D(25.0, 145.0, 14.0)
    z0_points = (
        Point3D(48.0, 30.0, 0.0),
        Point3D(58.0, 35.0, 0.0),
        Point3D(70.0, 40.0, 0.0),
    )
    paths = field.possible_paths(p0, z0_points, 0.75, t0_s=8.0)
    assert len(paths) == 3
    for path, expected in zip(paths, z0_points):
        assert_point_close(path.landing_point, expected)
        assert math.isclose(path.contact_time_s, 8.75)


def test_p0_v0_t0_select_one_active_path_and_one_z0():
    field = CourtPathField()
    selection = field.select_from_state(
        Point3D(25.0, 145.0, 14.0),
        Velocity3D(40.0, -140.0, 35.0),
        t0_s=12.5,
    )
    assert selection.path.t0_s == 12.5
    assert selection.landing.z == 0.0
    assert selection.contact_time_s > 12.5


def test_absolute_time_and_elapsed_time_are_consistent():
    path = FlightPath(
        Point3D(25.0, 145.0, 14.0),
        Velocity3D(40.0, -140.0, 35.0),
        t0_s=3.25,
    )
    a = path.point_at_elapsed(0.04)
    b = path.point_at_time(3.29)
    assert_point_close(a, b)


def test_path_is_curved_in_z_but_continuous_in_xyz():
    path = FlightPath(
        Point3D(25.0, 145.0, 14.0),
        Velocity3D(40.0, -140.0, 35.0),
    )
    t = path.flight_duration_s
    a = path.point_at_elapsed(t * 0.25)
    b = path.point_at_elapsed(t * 0.50)
    c = path.point_at_elapsed(t * 0.75)
    assert abs(a.z - 2.0 * b.z + c.z) > 1e-6
    assert a.x != path.p0.x and a.y != path.p0.y


def test_near_to_far_and_right_target_are_derived_from_path():
    field = CourtPathField()
    selection = field.select_from_state(
        Point3D(20.0, 150.0, 12.0),
        Velocity3D(55.0, -160.0, 25.0),
        t0_s=1.0,
    )
    assert selection.origin_side == CourtSide.NEAR
    assert selection.target_side == CourtSide.FAR
    assert selection.target_lateral == LateralSide.RIGHT
    assert selection.target_compatible


def test_far_return_reverses_target_half():
    field = CourtPathField()
    selection = field.select_from_state(
        Point3D(62.0, 30.0, 13.0),
        Velocity3D(-45.0, 170.0, 30.0),
        t0_s=2.0,
    )
    assert selection.origin_side == CourtSide.FAR
    assert selection.target_side == CourtSide.NEAR


def test_direction_change_starts_new_epoch_with_new_p0_v0_t0():
    manager = FlightEpochManager()
    first = manager.on_direction_change(
        Point3D(20.0, 150.0, 12.0),
        Velocity3D(55.0, -160.0, 25.0),
        t0_s=5.0,
    )
    second = manager.on_direction_change(
        Point3D(62.0, 30.0, 13.0),
        Velocity3D(-45.0, 170.0, 30.0),
        t0_s=6.2,
    )
    assert second.epoch_id == first.epoch_id + 1
    assert second.p0 != first.p0
    assert second.v0 != first.v0
    assert second.t0_s == 6.2


def test_pre_serve_floor_contact_does_not_create_epoch():
    manager = FlightEpochManager()
    assert manager.on_floor_contact(Point3D(30.0, 150.0, 0.0)) is None
    assert manager.active is None


def test_floor_contact_closes_active_epoch():
    manager = FlightEpochManager()
    epoch = manager.on_direction_change(
        Point3D(20.0, 150.0, 12.0),
        Velocity3D(55.0, -160.0, 25.0),
        t0_s=5.0,
    )
    assert manager.on_floor_contact(epoch.predicted_z0) == epoch
    assert manager.active is None


def test_ball_identity_is_not_an_input_to_flight_equations():
    sig = inspect.signature(CourtPathField.select_from_state)
    assert list(sig.parameters) == [
        "self", "p0", "v0", "t0_s", "target_lateral", "officiating_margin_bu"
    ]
