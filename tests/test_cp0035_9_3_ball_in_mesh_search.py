import cv2
import numpy as np
import pytest

from linecaller.dcf.ball_in_mesh_search import (
    BallInMeshSearchConfig,
    BallInMeshSearcher,
    DCFMeshProjector,
)
from linecaller.dcf.engine import DynamicCourtField
from linecaller.dcf.models import CellIndex


IMAGE_POINTS = (
    (120.0, 420.0),   # near-left
    (760.0, 420.0),   # near-right
    (620.0, 80.0),    # far-right
    (260.0, 80.0),    # far-left
)


def make_ball(size=21, color=(0, 220, 255)):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), max(3, size // 2 - 2), color, -1, cv2.LINE_AA)
    cv2.circle(img, (c - 3, c - 3), 2, (255, 255, 255), -1, cv2.LINE_AA)
    return img


def paste_center(frame, patch, x, y):
    h, w = patch.shape[:2]
    x1 = int(round(x - w / 2))
    y1 = int(round(y - h / 2))
    frame[y1:y1+h, x1:x1+w] = patch


def test_contact_plane_z0_maps_to_floor_point():
    field = DynamicCourtField()
    projector = DCFMeshProjector(IMAGE_POINTS, field=field)

    index = CellIndex(field.x0 + 40, field.y0 + 80, 0)
    u, t, h = projector.normalized_coordinates(index)

    assert h == 0.0

    expected = projector.mesh.floor_point(u, t)
    actual = projector.image_point(index)

    assert actual[0] == pytest.approx(expected[0])
    assert actual[1] == pytest.approx(expected[1])


def test_positive_z_projects_above_floor():
    field = DynamicCourtField()
    projector = DCFMeshProjector(IMAGE_POINTS, field=field)

    floor = CellIndex(field.x0 + 40, field.y0 + 80, 0)
    air = CellIndex(field.x0 + 40, field.y0 + 80, 20)

    fx, fy = projector.image_point(floor)
    ax, ay = projector.image_point(air)

    assert ax == pytest.approx(fx)
    assert ay < fy


def test_external_margin_is_part_of_same_internal_dcf():
    field = DynamicCourtField()
    projector = DCFMeshProjector(IMAGE_POINTS, field=field)

    outside = CellIndex(field.x0 - 2, field.y0 + 50, 0)
    u, _, _ = projector.normalized_coordinates(outside)

    assert u < 0.0
    assert projector.project(outside).region.value == "OUT_LEFT"


def test_mesh_scanner_cell_carries_xyz_identity():
    field = DynamicCourtField()
    projector = DCFMeshProjector(IMAGE_POINTS, field=field)
    ball = make_ball()

    index = CellIndex(field.x0 + 20, field.y0 + 30, 7)
    projected = projector.project(index)
    cell = projector.to_scanner_cell(projected, ball)

    assert cell.dcf_index == index
    assert cell.ix == index.x
    assert cell.iy == index.y
    assert cell.iz == index.z
    assert cell.center_x == pytest.approx(projected.image_x)
    assert cell.center_y == pytest.approx(projected.image_y)


def test_search_indices_finds_one_known_ball_in_one_dcf_cell():
    field = DynamicCourtField()
    config = BallInMeshSearchConfig(
        min_expected_diameter_px=1.0,
        max_expected_diameter_px=200.0,
    )
    searcher = BallInMeshSearcher(
        IMAGE_POINTS,
        field=field,
        config=config,
        min_score=0.45,
        min_color_score=0.45,
    )

    target = CellIndex(field.x0 + 41, field.y0 + 91, 0)
    projected = searcher.projector.project(target)

    ball = make_ball()
    scale = projected.expected_diameter_px / min(ball.shape[:2])
    resized = cv2.resize(
        ball,
        (
            max(3, round(ball.shape[1] * scale)),
            max(3, round(ball.shape[0] * scale)),
        ),
    )

    frame = np.zeros((500, 900, 3), dtype=np.uint8)
    paste_center(
        frame,
        resized,
        projected.image_x,
        projected.image_y,
    )

    neighbour = CellIndex(target.x + 7, target.y, target.z)

    hit = searcher.search_indices(
        frame,
        ball,
        [neighbour, target],
    )

    assert hit.found
    assert hit.index == target
    assert hit.x == pytest.approx(projected.image_x, abs=2.0)
    assert hit.y == pytest.approx(projected.image_y, abs=2.0)


def test_first_search_locks_one_active_ball():
    searcher = BallInMeshSearcher(
        IMAGE_POINTS,
        config=BallInMeshSearchConfig(
            min_expected_diameter_px=1.0,
            max_expected_diameter_px=200.0,
        ),
    )
    ball = make_ball()

    assert not searcher.ball_locked
    searcher.lock_ball(ball)
    assert searcher.ball_locked
    assert searcher.ball_generation == 1


def test_replace_ball_is_explicit_generation_change():
    searcher = BallInMeshSearcher(IMAGE_POINTS)
    ball_a = make_ball(color=(0, 220, 255))
    ball_b = make_ball(color=(255, 80, 0))

    searcher.lock_ball(ball_a)
    assert searcher.ball_generation == 1

    searcher.replace_ball(ball_b)
    assert searcher.ball_generation == 2


def test_production_searcher_has_no_mesh_drawing_api():
    # Mesh drawing remains a debug concern in PerspectiveCourtMesh. The
    # production searcher itself exposes search, not a draw-mesh operation.
    searcher = BallInMeshSearcher(IMAGE_POINTS)
    assert not hasattr(searcher, "draw_mesh")


def test_runtime_projector_has_no_fixed_near_height_or_gamma():
    projector = DCFMeshProjector(IMAGE_POINTS)
    assert not hasattr(projector, "near_height_px")
    assert not hasattr(projector, "gamma")


def test_off_center_reference_isolated_from_background():
    from linecaller.dcf.ball_in_mesh_search import LockedBallIdentity

    ref = np.full((24, 24, 3), (30, 35, 35), dtype=np.uint8)
    cv2.circle(ref, (13, 17), 4, (0, 220, 255), -1, cv2.LINE_AA)

    identity = LockedBallIdentity.from_reference(ref)

    assert 5.0 <= identity.diameter_px <= 10.0
    assert cv2.countNonZero(identity.mask) > 10
    # The locked profile must describe the ball, not the dark court/background.
    assert identity.pixel_profile.value > 100


def test_true_cell_evaluation_ignores_reference_background_change():
    field = DynamicCourtField()
    searcher = BallInMeshSearcher(
        IMAGE_POINTS,
        field=field,
        config=BallInMeshSearchConfig(
            min_expected_diameter_px=1.0,
            max_expected_diameter_px=200.0,
            min_score=0.40,
        ),
    )

    # Reference ball sits on a dark background and is deliberately off-centre.
    ref = np.full((24, 24, 3), (28, 32, 32), dtype=np.uint8)
    cv2.circle(ref, (13, 17), 4, (0, 220, 255), -1, cv2.LINE_AA)

    identity = searcher.lock_ball(ref)
    target = CellIndex(field.x0 + 41, field.y0 + 91, 0)
    projected = searcher.projector.project(target)

    # Target frame has a completely different court/background color.
    frame = np.full((500, 900, 3), (150, 80, 30), dtype=np.uint8)
    radius = max(2, round(projected.expected_diameter_px / 2.0))
    cv2.circle(
        frame,
        (round(projected.image_x), round(projected.image_y)),
        radius,
        (0, 220, 255),
        -1,
        cv2.LINE_AA,
    )

    hit = searcher.search_indices(frame, ref, [target])
    assert hit.found
    assert hit.index == target


def test_locked_ball_identity_cannot_change_via_search_reference():
    field = DynamicCourtField()
    searcher = BallInMeshSearcher(
        IMAGE_POINTS,
        field=field,
        config=BallInMeshSearchConfig(
            min_expected_diameter_px=1.0,
            max_expected_diameter_px=200.0,
            min_score=0.40,
        ),
    )

    yellow = np.zeros((21, 21, 3), dtype=np.uint8)
    blue = np.zeros((21, 21, 3), dtype=np.uint8)
    cv2.circle(yellow, (10, 10), 7, (0, 220, 255), -1)
    cv2.circle(blue, (10, 10), 7, (255, 80, 0), -1)

    searcher.lock_ball(yellow)
    locked_hue = searcher.ball_identity.pixel_profile.hue

    # Passing blue to the search call must NOT replace the locked identity.
    frame = np.zeros((500, 900, 3), dtype=np.uint8)
    target = CellIndex(field.x0 + 30, field.y0 + 50, 0)
    searcher.search_indices(frame, blue, [target])

    assert searcher.ball_generation == 1
    assert searcher.ball_identity.pixel_profile.hue == pytest.approx(locked_hue)
