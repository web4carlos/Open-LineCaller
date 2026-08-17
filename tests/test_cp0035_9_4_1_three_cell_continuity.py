import math

from linecaller.dcf.ball_in_mesh_search import BallInMeshHit, DCFMeshProjector
from linecaller.dcf.ball_wave import BallWaveConfig, BallWaveTracker
from linecaller.dcf.engine import DynamicCourtField
from linecaller.dcf.models import CellIndex, FieldRegion


def hit(cell, generation=1, score=0.55):
    return BallInMeshHit(
        found=True,
        score=score,
        x=float(cell.x),
        y=float(cell.y),
        index=cell,
        region=FieldRegion.INSIDE,
        ball_generation=generation,
    )


def miss(generation=1):
    return BallInMeshHit(
        False, 0.0, None, None, None, None,
        ball_generation=generation,
    )


class FakeSearcher:
    def __init__(self):
        self.field = DynamicCourtField()
        self.ball_generation = 1
        self.local_results = []
        self.local_calls = []
        self.local_thresholds = []
        self.global_results = []
        self.global_calls = 0

    def search_indices(self, frame, ball_reference, indices, **kwargs):
        self.local_calls.append(frozenset(indices))
        self.local_thresholds.append(kwargs.get("min_score"))
        if self.local_results:
            return self.local_results.pop(0)
        return miss(self.ball_generation)

    def search(self, frame, ball_reference):
        self.global_calls += 1
        if self.global_results:
            return self.global_results.pop(0)
        return miss(self.ball_generation)


def test_history_is_hard_capped_at_three_cells():
    s = FakeSearcher()
    t = BallWaveTracker(s)
    t.acquire(hit(CellIndex(60, 80, 8)))

    for cell in (
        CellIndex(61, 81, 7),
        CellIndex(62, 82, 6),
        CellIndex(63, 83, 5),
        CellIndex(64, 84, 4),
    ):
        s.local_results.append(hit(cell))
        step = t.track(None, None)
        assert step.hit.found
        assert len(step.state.history) <= 3

    assert t.state.history == (
        CellIndex(62, 82, 6),
        CellIndex(63, 83, 5),
        CellIndex(64, 84, 4),
    )


def test_first_handoff_searches_only_immediate_dcf_neighborhood():
    s = FakeSearcher()
    t = BallWaveTracker(s)
    start = CellIndex(60, 80, 10)
    t.acquire(hit(start))

    wave = t.current_wave()

    assert start in wave
    assert len(wave) <= 125
    assert CellIndex(63, 80, 10) not in wave
    assert CellIndex(60, 83, 10) not in wave


def test_after_two_cells_next_search_is_small_predicted_neighborhood():
    s = FakeSearcher()
    t = BallWaveTracker(s)
    t.acquire(hit(CellIndex(60, 80, 10)))
    s.local_results.append(hit(CellIndex(61, 81, 9)))
    t.track(None, None)

    wave = t.current_wave()
    expected = CellIndex(62, 82, 8)

    assert expected in wave
    assert len(wave) <= 28
    assert CellIndex(70, 90, 8) not in wave


def test_local_continuity_uses_lower_known_ball_threshold_only_locally():
    s = FakeSearcher()
    cfg = BallWaveConfig(continuity_min_score=0.38)
    t = BallWaveTracker(s, config=cfg)
    t.acquire(hit(CellIndex(60, 80, 10)))
    s.local_results.append(hit(CellIndex(61, 80, 9), score=0.40))

    step = t.track(None, None)

    assert step.hit.found
    assert s.local_thresholds == [0.38]
    assert s.global_calls == 0


def test_miss_expands_only_one_local_ring_not_global_field():
    s = FakeSearcher()
    t = BallWaveTracker(s)
    t.acquire(hit(CellIndex(60, 80, 10)))
    base = len(t.current_wave())

    s.local_results.append(miss())
    step = t.track(None, None)
    expanded = len(t.current_wave())

    assert step.reason == "LOCAL_MISS_EXPAND_WAVE"
    assert expanded > base
    assert expanded <= 343
    assert s.global_calls == 0


def test_z1_while_descending_predicts_z0_and_triggers_floor():
    s = FakeSearcher()
    t = BallWaveTracker(s)
    t.acquire(hit(CellIndex(50, 70, 3)))

    s.local_results.append(hit(CellIndex(51, 72, 2)))
    first = t.track(None, None)
    assert not first.floor_illumination

    s.local_results.append(hit(CellIndex(52, 74, 1)))
    contact = t.track(None, None)

    assert contact.hit.found
    assert contact.hit.index == CellIndex(52, 74, 1)
    assert contact.reason == "CONTACT_FLOOR_TRIGGER"
    assert contact.contact_plane_reached is True
    assert contact.floor_illumination is False
    assert contact.floor_trigger is True
    assert contact.predicted_contact_cell == CellIndex(53, 76, 0)


def test_z1_while_rising_does_not_trigger_floor():
    s = FakeSearcher()
    t = BallWaveTracker(s)
    t.acquire(hit(CellIndex(50, 70, 0)))
    s.local_results.append(hit(CellIndex(51, 72, 1)))

    step = t.track(None, None)

    assert step.hit.found
    assert step.state.velocity[2] > 0
    assert step.floor_illumination is False
    assert step.floor_trigger is False
    assert step.predicted_contact_cell is None


def test_z1_floor_trigger_is_latched_not_repeated():
    s = FakeSearcher()
    t = BallWaveTracker(s)
    t.acquire(hit(CellIndex(50, 70, 3)))
    s.local_results.extend([
        hit(CellIndex(51, 71, 2)),
        hit(CellIndex(52, 72, 1)),
        hit(CellIndex(52, 72, 1)),
    ])

    t.track(None, None)
    first = t.track(None, None)
    duplicate = t.track(None, None)

    assert first.floor_trigger is True
    assert first.floor_illumination is False
    assert duplicate.floor_trigger is False


def test_track_never_performs_global_scan():
    s = FakeSearcher()
    t = BallWaveTracker(s)
    t.acquire(hit(CellIndex(60, 80, 10)))
    s.local_results.extend([miss(), miss()])

    t.track(None, None)
    t.track(None, None)

    assert s.global_calls == 0
    assert len(s.local_calls) == 2


def test_explicit_reacquisition_resets_three_cell_history():
    s = FakeSearcher()
    t = BallWaveTracker(s, config=BallWaveConfig(max_local_misses=1))
    t.acquire(hit(CellIndex(60, 80, 10)))
    s.local_results.append(miss())
    t.track(None, None)

    new_cell = CellIndex(70, 90, 8)
    s.global_results.append(hit(new_cell))
    step = t.reacquire(None, None)

    assert step.reason == "REACQUIRED_SAME_BALL"
    assert step.state.history == (new_cell,)
    assert step.state.velocity == (0.0, 0.0, 0.0)


def test_direct_descending_z0_hit_triggers_contact_floor_event():
    s = FakeSearcher()
    t = BallWaveTracker(s)
    t.acquire(hit(CellIndex(60, 80, 3)))
    s.local_results.append(hit(CellIndex(60, 81, 2)))
    t.track(None, None)
    s.local_results.append(hit(CellIndex(60, 82, 0)))

    step = t.track(None, None)

    assert step.hit.found
    assert step.hit.index == CellIndex(60, 82, 0)
    assert step.contact_plane_reached is True
    assert step.floor_trigger is True
    assert step.floor_illumination is False
    assert step.predicted_contact_cell == CellIndex(60, 82, 0)


def test_dcf_xy_cells_are_ball_sized_physical_units():
    """Court XY resolution is intentionally about one pickleball per cell."""
    width_cell_in = 20.0 * 12.0 / 83.0
    depth_cell_in = 44.0 * 12.0 / 182.0

    # USA Pickleball equipment diameter range: 2.874..2.972 inches.
    assert 2.874 <= width_cell_in <= 2.972
    assert 2.874 <= depth_cell_in <= 2.972
    assert math.isclose(width_cell_in, depth_cell_in, abs_tol=0.02)


def test_one_z_step_is_one_local_lateral_dcf_cell_in_projection():
    """Z uses the same local physical cell scale as X, not an arbitrary height."""
    field = DynamicCourtField()
    projector = DCFMeshProjector(
        [(20.0, 340.0), (620.0, 340.0), (390.0, 90.0), (150.0, 90.0)],
        field=field,
    )
    x = field.x0 + field.config.court_x_cells // 2
    y = field.y0 + field.config.court_y_cells // 2
    z0 = CellIndex(x, y, 0)
    z1 = CellIndex(x, y, 1)

    p0 = projector.image_point(z0)
    p1 = projector.image_point(z1)
    step_px = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    local_cell_px = projector.expected_diameter_px(z0)

    assert math.isclose(step_px, local_cell_px, rel_tol=1e-9, abs_tol=1e-9)


def test_z_zero_is_floor_and_z_increases_upward():
    field = DynamicCourtField()
    projector = DCFMeshProjector(
        [(20.0, 340.0), (620.0, 340.0), (390.0, 90.0), (150.0, 90.0)],
        field=field,
    )
    x = field.x0 + field.config.court_x_cells // 2
    y = field.y0 + field.config.court_y_cells // 2
    z0 = CellIndex(x, y, 0)
    z1 = CellIndex(x, y, 1)

    u, t, _ = projector.normalized_coordinates(z0)
    floor = projector.floor_point(u, t)
    p0 = projector.image_point(z0)
    p1 = projector.image_point(z1)

    assert math.isclose(p0[0], float(floor[0]), abs_tol=1e-9)
    assert math.isclose(p0[1], float(floor[1]), abs_tol=1e-9)
    assert p1[1] < p0[1]  # default runtime vertical vector is screen-up
