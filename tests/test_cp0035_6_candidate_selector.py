from linecaller.dcf.candidate_selector import (
    BallCandidateObservation,
    DCFBallCandidateSelector,
)

def c(x,y,conf,w=10,h=10):
    return BallCandidateObservation(1,x,y,conf,w,h)

def test_high_confidence_far_candidate_loses():
    s=DCFBallCandidateSelector(max_distance_px=80)
    r=s.select(
        [c(500,500,.95),c(110,100,.40)],
        expected_x=100,
        expected_y=100,
        expected_size=10,
    )
    assert r.selected.x == 110

def test_implausible_size_rejected():
    s=DCFBallCandidateSelector(max_distance_px=80,max_size_ratio=2.0)
    r=s.select(
        [c(102,101,.9,50,50),c(105,100,.4,10,10)],
        expected_x=100,
        expected_y=100,
        expected_size=10,
    )
    assert r.selected.width == 10

def test_no_plausible_candidate():
    s=DCFBallCandidateSelector(max_distance_px=20)
    r=s.select(
        [c(500,500,.9)],
        expected_x=100,
        expected_y=100,
        expected_size=10,
    )
    assert r.selected is None
    assert r.reason == "NO_PLAUSIBLE_CANDIDATE"

def test_initial_selection_can_use_confidence():
    s=DCFBallCandidateSelector()
    r=s.select(
        [c(100,100,.2),c(200,200,.8)],
        expected_x=None,
        expected_y=None,
        expected_size=None,
    )
    assert r.selected.confidence == .8

def test_size_continuity_affects_score():
    s=DCFBallCandidateSelector(max_distance_px=80)
    r=s.select(
        [c(105,100,.6,10,10),c(105,100,.6,25,25)],
        expected_x=100,
        expected_y=100,
        expected_size=10,
    )
    assert r.selected.width == 10
