from linecaller.vision.models import VisionDetection

def test_model():
    d = VisionDetection(10, 12.0, 30.0, .8, "TRACKING", "classical")
    assert d.frame_number == 10
    assert d.source == "classical"
