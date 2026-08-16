from linecaller.dcf.ball_in_mesh_search import BallInMeshHit
from linecaller.dcf.ball_wave import BallWaveConfig, BallWaveTracker
from linecaller.dcf.engine import DynamicCourtField
from linecaller.dcf.models import CellIndex, FieldRegion


def hit(cell, generation=1, score=0.9):
    return BallInMeshHit(
        found=True,
        score=score,
        x=100.0,
        y=100.0,
        index=cell,
        region=FieldRegion.INSIDE,
        ball_generation=generation,
    )


def miss(generation=1):
    return BallInMeshHit(False, 0.0, None, None, None, None, ball_generation=generation)


class FakeSearcher:
    def __init__(self):
        self.field = DynamicCourtField()
        self.ball_generation = 1
        self.local_results = []
        self.global_results = []
        self.local_calls = []
        self.global_calls = 0

    def search_indices(self, frame, ball_reference, indices, **kwargs):
        indices = frozenset(indices)
        self.local_calls.append(indices)
        return self.local_results.pop(0) if self.local_results else miss(self.ball_generation)

    def search(self, frame, ball_reference):
        self.global_calls += 1
        return self.global_results.pop(0) if self.global_results else miss(self.ball_generation)


def test_acquisition_creates_small_internal_wave():
    s = FakeSearcher()
    t = BallWaveTracker(s)
    start = CellIndex(60, 80, 10)
    state = t.acquire(hit(start))
    wave = t.current_wave()

    assert state.active
    assert start in wave
    assert 1 < len(wave) < 1000
    assert state.requires_reacquisition is False


def test_local_tracking_never_calls_global_search():
    s = FakeSearcher()
    t = BallWaveTracker(s)
    t.acquire(hit(CellIndex(60, 80, 10)))
    s.local_results.append(hit(CellIndex(61, 80, 9)))

    step = t.track(object(), object())

    assert step.hit.found
    assert step.reason == "LOCAL_HIT"
    assert s.global_calls == 0
    assert len(s.local_calls) == 1
    assert step.local_candidates_tested < 1000


def test_cell_to_cell_motion_updates_velocity_and_forward_wave():
    s = FakeSearcher()
    t = BallWaveTracker(s)
    t.acquire(hit(CellIndex(60, 80, 10)))

    s.local_results.append(hit(CellIndex(62, 81, 9)))
    first = t.track(None, None)
    assert first.state.velocity == (2.0, 1.0, -1.0)

    wave = t.current_wave()
    predicted = CellIndex(64, 82, 8)
    assert predicted in wave


def test_misses_expand_wave_before_reacquisition():
    s = FakeSearcher()
    t = BallWaveTracker(s, config=BallWaveConfig(max_local_misses=4))
    t.acquire(hit(CellIndex(60, 80, 10)))
    base = len(t.current_wave())

    s.local_results.append(miss())
    step1 = t.track(None, None)
    expanded = len(t.current_wave())

    assert step1.reason == "LOCAL_MISS_EXPAND_WAVE"
    assert step1.state.active
    assert expanded > base


def test_repeated_local_misses_request_explicit_reacquisition():
    s = FakeSearcher()
    t = BallWaveTracker(s, config=BallWaveConfig(max_local_misses=3))
    t.acquire(hit(CellIndex(60, 80, 10)))
    s.local_results.extend([miss(), miss(), miss()])

    t.track(None, None)
    t.track(None, None)
    final = t.track(None, None)

    assert final.reason == "REACQUISITION_REQUIRED"
    assert final.state.requires_reacquisition
    assert not final.state.active
    assert s.global_calls == 0


def test_explicit_reacquisition_uses_same_ball_session():
    s = FakeSearcher()
    t = BallWaveTracker(s, config=BallWaveConfig(max_local_misses=1))
    t.acquire(hit(CellIndex(60, 80, 10)))
    s.local_results.append(miss())
    t.track(None, None)

    s.global_results.append(hit(CellIndex(72, 91, 7), generation=1))
    step = t.reacquire(None, None)

    assert step.reason == "REACQUIRED_SAME_BALL"
    assert step.state.active
    assert step.state.ball_generation == 1
    assert s.global_calls == 1


def test_ball_replacement_generation_invalidates_old_wave():
    s = FakeSearcher()
    t = BallWaveTracker(s)
    t.acquire(hit(CellIndex(60, 80, 10), generation=1))

    s.ball_generation = 2
    step = t.track(None, None)

    assert step.reason == "BALL_GENERATION_CHANGED"
    assert step.state.requires_reacquisition
    assert not step.state.active
    assert len(s.local_calls) == 0


def test_z0_is_contact_candidate_but_floor_not_illuminated_yet():
    s = FakeSearcher()
    t = BallWaveTracker(s)
    t.acquire(hit(CellIndex(60, 80, 1)))
    s.local_results.append(hit(CellIndex(60, 81, 0)))

    step = t.track(None, None)

    assert step.contact_plane_reached is True
    assert step.floor_illumination is False
    assert step.hit.index.z == 0


def test_wave_can_include_external_officiating_field_cells():
    s = FakeSearcher()
    t = BallWaveTracker(s)
    start = CellIndex(s.field.x0, s.field.y0 + 30, 5)
    t.acquire(hit(start))
    wave = t.current_wave()

    assert any(c.x < s.field.x0 for c in wave)


def test_no_mesh_rendering_api_on_ball_wave_tracker():
    s = FakeSearcher()
    t = BallWaveTracker(s)
    assert not hasattr(t, "draw_mesh")
