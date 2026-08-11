from linecaller.annotation.models import BallAnnotation
from linecaller.annotation.yolo import YoloLabelWriter
def test_rectangular():
    a=BallAnnotation(0,960,540,30,18);x,y,w,h=YoloLabelWriter.normalize(a,1920,1080)
    assert x==0.5 and y==0.5 and w==30/1920 and h==18/1080
