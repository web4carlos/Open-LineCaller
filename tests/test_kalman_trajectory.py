from linecaller.ball.kalman import Kalman2D
from linecaller.ball.trajectory import TrajectoryHistory, TrajectoryPoint

def test_kalman():
    k = Kalman2D()
    assert k.update(10,20) == (10.0,20.0)
    x,y = k.predict()
    assert isinstance(x,float) and isinstance(y,float)

def test_trajectory_bounded():
    h = TrajectoryHistory(max_points=3)
    for i in range(5):
        h.add(TrajectoryPoint(i,float(i),0.0,True,1.0))
    assert len(h) == 3
    assert h.points()[0].frame_number == 2
