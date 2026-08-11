from linecaller.annotation.models import AnnotationStats,BallAnnotation
def test_stats_labeled():
    s=AnnotationStats(positives=7,negatives=3);assert s.labeled==10
def test_ball_annotation():
    a=BallAnnotation(12,100,200,24);assert a.frame_number==12
