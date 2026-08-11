from linecaller.annotation.models import BallAnnotation
from linecaller.annotation.yolo import YoloLabelWriter
def test_yolo_normalization():
    a=BallAnnotation(0,960,540,24);x,y,w,h=YoloLabelWriter.normalize(a,1920,1080)
    assert x==0.5 and y==0.5 and 0<w<1 and 0<h<1
def test_negative_file(tmp_path):
    p=tmp_path/'negative.txt';YoloLabelWriter.write_negative(p);assert p.read_text()==''
