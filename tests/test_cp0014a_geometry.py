import pytest

from linecaller.proposals.geometry import center_distance, iou
from linecaller.proposals.models import BallProposal


def p(x,y,w=10,h=10):
    return BallProposal(0,x,y,w,h,.9)


def test_iou_same_box():
    assert iou(p(0,0), p(0,0)) == pytest.approx(1.0)


def test_center_distance():
    assert center_distance(p(0,0), p(3,4)) == pytest.approx(5.0)
