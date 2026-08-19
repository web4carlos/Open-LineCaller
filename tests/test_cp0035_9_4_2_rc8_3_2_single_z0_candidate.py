from __future__ import annotations

from dataclasses import dataclass

from linecaller.dcf.single_z0_candidate import SingleZ0CandidateResolver


@dataclass(frozen=True)
class FakeCandidate:
    cell_id: int
    gx: int
    gy: int
    ball_pixels: int
    changed_pixels: int
    visible_area_px: int
    score: float
    polygon: tuple[tuple[float, float], ...]


def _c(cell_id, score, pixels=3):
    return FakeCandidate(
        cell_id=cell_id,
        gx=cell_id,
        gy=10,
        ball_pixels=pixels,
        changed_pixels=pixels,
        visible_area_px=20,
        score=score,
        polygon=((0,0),(1,0),(1,1),(0,1)),
    )


def test_zero_candidates_returns_none():
    r = SingleZ0CandidateResolver()
    assert r.resolve([]) is None


def test_one_candidate_returns_same_cell():
    r = SingleZ0CandidateResolver()
    z0 = r.resolve([_c(7, 1.0)])
    assert z0 is not None
    assert z0.cell_id == 7


def test_multiple_candidates_reduce_to_exactly_one():
    r = SingleZ0CandidateResolver()
    z0 = r.resolve([_c(1, 0.4), _c(2, 1.2), _c(3, 0.9)])
    assert z0 is not None
    assert z0.cell_id == 2


def test_score_wins_before_pixel_count():
    r = SingleZ0CandidateResolver()
    z0 = r.resolve([_c(1, 1.0, 100), _c(2, 1.1, 2)])
    assert z0 is not None
    assert z0.cell_id == 2


def test_pixel_count_breaks_equal_score_tie():
    r = SingleZ0CandidateResolver()
    z0 = r.resolve([_c(1, 1.0, 3), _c(2, 1.0, 7)])
    assert z0 is not None
    assert z0.cell_id == 2


def test_result_has_no_history_or_path_fields():
    r = SingleZ0CandidateResolver()
    z0 = r.resolve([_c(5, 1.0)])
    assert z0 is not None
    assert not hasattr(z0, "history")
    assert not hasattr(z0, "path")
    assert not hasattr(z0, "trajectory")
