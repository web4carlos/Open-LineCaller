import pytest
from linecaller.ball.trajectory import TrajectoryPoint
from linecaller.bounce.engine import BounceEngine

def p(f,y): return TrajectoryPoint(f,100.0,float(y),True,.95)

def test_streaming_emits_one_bounce():
    engine=BounceEngine(cooldown_frames=8)
    seq=[p(0,5),p(1,10),p(2,16),p(3,23),p(4,16),p(5,10),p(6,6)]
    events=[e for point in seq if (e:=engine.process(point))]
    assert len(events)==1
    assert events[0].frame_number==3

def test_non_increasing_frames_rejected():
    engine=BounceEngine()
    engine.process(p(1,10))
    with pytest.raises(ValueError):
        engine.process(p(1,12))
