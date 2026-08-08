from linecaller.autocalibration.hypothesis import CourtHypothesis
from linecaller.autocalibration.models import AutoCalibrationDisposition
from linecaller.autocalibration.ranking import HypothesisRanker


def h(score):
    return CourtHypothesis(
        corners=((0,0),(1,0),(1,1),(0,1)),
        score=score,
        area_ratio=.3,
        line_support=.8,
        opposite_consistency=.9,
        separation_score=.9,
    )


def test_accept_clear_winner():
    r=HypothesisRanker(
        accept_score=.7,
        review_score=.4,
        min_accept_margin=.08,
    ).rank([h(.85),h(.60)])

    assert r.disposition == AutoCalibrationDisposition.AUTO_ACCEPT


def test_review_when_good_but_close():
    r=HypothesisRanker(
        accept_score=.7,
        review_score=.4,
        min_accept_margin=.08,
        min_review_margin=.02,
    ).rank([h(.80),h(.75)])

    assert r.disposition == AutoCalibrationDisposition.AUTO_REVIEW


def test_reject_ambiguous():
    r=HypothesisRanker(
        accept_score=.7,
        review_score=.4,
        min_accept_margin=.08,
        min_review_margin=.03,
    ).rank([h(.80),h(.79)])

    assert r.disposition == AutoCalibrationDisposition.REJECT
