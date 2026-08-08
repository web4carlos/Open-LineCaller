from linecaller.dataset.models import BallBox,BounceAnnotation,ClipAnnotation,FrameAnnotation
from linecaller.dataset.studio_session import DatasetStudioSession
def test_roundtrip():
    a=ClipAnnotation("x","x.mp4",100,100,30,20,[FrameAnnotation(3,BallBox(10,10,5,5),True,True,"OK")],[BounceAnnotation(4,20,30,"IN")])
    out=DatasetStudioSession.from_annotation(a).to_annotation()
    assert out.frames[0].occluded is True and out.bounces[0].decision=="IN"
