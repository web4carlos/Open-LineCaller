import pytest
from linecaller.dataset.models import BallBox
from linecaller.dataset.studio_session import DatasetStudioSession
def s(): return DatasetStudioSession(clip_id="c",source_video="c.mp4",width=640,height=480,fps=60,frame_count=100)
def test_set_ball_box():
    x=s(); x.set_ball_box(5,BallBox(10,20,8,8)); assert x.to_annotation().frames[0].ball.width==8
def test_remove_ball_box():
    x=s(); x.set_ball_box(5,BallBox(10,20,8,8)); x.remove_ball_box(5); assert x.frame_annotation(5).ball is None
def test_toggle_bounce():
    x=s(); assert x.toggle_bounce(10,x=20,y=30,decision="OUT") is True; assert x.bounce_annotation(10).decision=="OUT"; assert x.toggle_bounce(10,x=20,y=30) is False
def test_invalid_frame():
    with pytest.raises(ValueError): s().set_ball_box(1000,BallBox(0,0,10,10))
