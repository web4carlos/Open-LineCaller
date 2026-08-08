from linecaller.autocalibration.hypothesis import CourtHypothesis
from linecaller.autocalibration.models import AutoCalibrationDisposition
from linecaller.autocalibration.ranking import RankedAutoCalibration
from linecaller.calibration.smart_adapter import SmartCalibrationAdapter


def hyp(score=.8):
    return CourtHypothesis(
        corners=((0,10),(10,10),(8,0),(2,0)),
        score=score,
        area_ratio=.4,
        line_support=.8,
        opposite_consistency=.9,
        separation_score=.9,
    )


def test_adapter_accepts_ranked_proposal():
    ranked = RankedAutoCalibration(
        best=hyp(),
        runner_up=hyp(.5),
        confidence_margin=.3,
        disposition=AutoCalibrationDisposition.AUTO_ACCEPT,
        reasons=(),
    )

    p = SmartCalibrationAdapter().from_ranked(ranked)

    assert p.accepted is True
    assert len(p.outer_corners) == 4


def test_adapter_reject_without_best():
    ranked = RankedAutoCalibration(
        best=None,
        runner_up=None,
        confidence_margin=0.0,
        disposition=AutoCalibrationDisposition.REJECT,
        reasons=("none",),
    )

    p = SmartCalibrationAdapter().from_ranked(ranked)

    assert p.accepted is False
    assert p.outer_corners == ()
