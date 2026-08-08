from linecaller.ball.trajectory import TrajectoryPoint
from linecaller.bounce.detector import BounceDetector

def p(f,y,m=True,c=.9): return TrajectoryPoint(f,50.0,float(y),m,c)

def test_clear_bounce():
    assert BounceDetector().detect_at([p(0,10),p(1,15),p(2,21),p(3,15),p(4,9)],2) is not None

def test_monotonic_fall():
    assert BounceDetector().detect_at([p(0,10),p(1,15),p(2,20),p(3,25),p(4,30)],2) is None

def test_slow_reversal_rejected():
    assert BounceDetector().detect_at([p(0,10),p(1,10.5),p(2,11),p(3,10.5),p(4,10)],2) is None

def test_predicted_dominated_rejected():
    pts=[p(0,10,False),p(1,15,False),p(2,21,True),p(3,15,False),p(4,9,False)]
    assert BounceDetector().detect_at(pts,2) is None

def test_low_confidence_rejected():
    pts=[p(0,10,True,.2),p(1,15,True,.2),p(2,21,True,.2),p(3,15,True,.2),p(4,9,True,.2)]
    assert BounceDetector().detect_at(pts,2) is None
