from __future__ import annotations

from dataclasses import dataclass

from linecaller.dcf.ball_in_mesh_search import BallInMeshHit
from linecaller.dcf.models import CellIndex, FieldRegion
from linecaller.flight_paths.transition_continuity import (
    DirectionTransitionContinuity,
    TransitionContinuityConfig,
)


class FakeField:
    def valid(self, p):
        return 0 <= p.x < 200 and 0 <= p.y < 300 and 0 <= p.z < 100


class FakeSearcher:
    def __init__(self, hits):
        self.field = FakeField()
        self.ball_generation = 7
        self._hits = list(hits)
        self.local_calls = 0
        self.global_calls = 0
        self.last_indices = None

    def search(self, *_args, **_kwargs):
        self.global_calls += 1
        raise AssertionError("RC5 transition continuity must never global-search")

    def search_indices(self, _frame, _reference, indices, *, min_score=None):
        self.local_calls += 1
        self.last_indices = frozenset(indices)
        if self._hits:
            return self._hits.pop(0)
        return miss()


def found(x, y, z, score=0.8):
    return BallInMeshHit(
        True,
        score,
        100.0,
        100.0,
        CellIndex(x, y, z),
        FieldRegion.INSIDE,
        ball_generation=7,
    )


def miss(score=0.2):
    return BallInMeshHit(
        False,
        score,
        None,
        None,
        None,
        None,
        ball_generation=7,
    )


def test_transition_sphere_is_not_aimed_by_old_velocity():
    searcher = FakeSearcher([])
    bridge = DirectionTransitionContinuity(
        searcher,
        config=TransitionContinuityConfig(base_radius_bu=3.0, max_radius_bu=6.0),
    )
    bridge.start(found(50, 60, 10))
    cells = bridge.candidate_indices()
    # All directions around the last trusted P are available.  No old vector
    # selects one forward cone.
    assert CellIndex(53, 60, 10) in cells
    assert CellIndex(47, 60, 10) in cells
    assert CellIndex(50, 63, 10) in cells
    assert CellIndex(50, 57, 10) in cells
    assert CellIndex(50, 60, 13) in cells
    assert CellIndex(50, 60, 7) in cells


def test_three_local_hits_confirm_new_path_observation_window_without_global_search():
    searcher = FakeSearcher([
        found(51, 61, 12),
        found(52, 63, 14),
        found(53, 66, 16),
    ])
    bridge = DirectionTransitionContinuity(searcher)
    bridge.start(found(50, 60, 10))

    a = bridge.track(None, None)
    b = bridge.track(None, None)
    c = bridge.track(None, None)

    assert not a.new_path_confirmed
    assert not b.new_path_confirmed
    assert c.new_path_confirmed
    assert c.reason == "NEW_PATH_OBSERVATIONS_READY"
    assert bridge.anchor == CellIndex(53, 66, 16)
    assert searcher.local_calls == 3
    assert searcher.global_calls == 0


def test_miss_expands_local_volume_but_keeps_last_trusted_anchor():
    searcher = FakeSearcher([miss(), found(54, 60, 10)])
    bridge = DirectionTransitionContinuity(
        searcher,
        config=TransitionContinuityConfig(
            base_radius_bu=3.0,
            miss_growth_bu=2.0,
            max_radius_bu=7.0,
        ),
    )
    initial = found(50, 60, 10)
    bridge.start(initial)

    n0 = len(bridge.candidate_indices())
    first = bridge.track(None, None)
    n1 = len(bridge.candidate_indices())
    second = bridge.track(None, None)

    assert first.reason == "TRANSITION_LOCAL_MISS_EXPAND"
    assert bridge.anchor == second.hit.index
    assert n1 > n0
    assert second.hit.index == CellIndex(54, 60, 10)
    assert searcher.global_calls == 0


def test_wrong_ball_generation_is_rejected_at_transition_start():
    searcher = FakeSearcher([])
    bridge = DirectionTransitionContinuity(searcher)
    bad = BallInMeshHit(
        True, 0.9, 1.0, 1.0, CellIndex(10, 10, 10), FieldRegion.INSIDE,
        ball_generation=99,
    )
    try:
        bridge.start(bad)
    except ValueError as exc:
        assert "generation" in str(exc)
    else:
        raise AssertionError("expected generation mismatch to be rejected")
