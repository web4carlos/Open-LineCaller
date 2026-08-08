from linecaller.ball.models import BallCandidate, TrackStatus
from linecaller.ball.tracker import BallTracker

def c(x,y,conf=0.9):
    return BallCandidate(x,y,5.0,conf,"test")

def test_tracking_measurement():
    t = BallTracker()
    s = t.update(1,[c(100,100)])
    assert s.status == TrackStatus.TRACKING

def test_prediction_on_brief_miss():
    t = BallTracker(max_lost_frames=3)
    t.update(1,[c(100,100)])
    t.update(2,[c(105,103)])
    s = t.update(3,[])
    assert s.status == TrackStatus.PREDICTED

def test_lost_after_limit():
    t = BallTracker(max_lost_frames=2)
    t.update(1,[c(100,100)])
    t.update(2,[])
    t.update(3,[])
    s = t.update(4,[])
    assert s.status == TrackStatus.LOST

def test_prefers_near_candidate():
    t = BallTracker(max_match_distance_px=80)
    t.update(1,[c(100,100)])
    t.update(2,[c(108,102)])
    s = t.update(3,[c(115,105,0.8), c(500,500,0.99)])
    assert s.status == TrackStatus.TRACKING
    assert s.x < 200
