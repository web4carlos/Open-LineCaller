import pytest
from linecaller.ball.trajectory import TrajectoryPoint
from linecaller.bounce.analyzer import TrajectoryAnalyzer

def p(f,y,m=True,c=.9): return TrajectoryPoint(f,100.0,float(y),m,c)

def test_velocity_reversal():
    pts=[p(0,10),p(1,15),p(2,20),p(3,15),p(4,10)]
    a=TrajectoryAnalyzer(2).analyze(pts,2)
    assert a.pre_vertical_velocity == pytest.approx(5)
    assert a.post_vertical_velocity == pytest.approx(-5)
