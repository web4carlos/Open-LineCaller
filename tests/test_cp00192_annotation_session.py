from linecaller.validation.annotation_models import AnnotationCandidate, AnnotationLabel
from linecaller.validation.annotation_session import AnnotationSession

def test_navigation(tmp_path):
    video=tmp_path/"match.mp4"; video.write_bytes(b"")
    s=AnnotationSession(); s.set_video(video)
    s.set_candidates([AnnotationCandidate(frame=10),AnnotationCandidate(frame=20)])
    assert s.current.frame==10
    s.next(); assert s.current.frame==20
    s.previous(); assert s.current.frame==10

def test_truth_export(tmp_path):
    video=tmp_path/"match.mp4"; video.write_bytes(b"")
    s=AnnotationSession(); s.set_video(video)
    s.set_candidates([AnnotationCandidate(frame=10),AnnotationCandidate(frame=20),AnnotationCandidate(frame=30)])
    s.label_current(AnnotationLabel.IN); s.next()
    s.label_current(AnnotationLabel.SKIP); s.next()
    s.label_current(AnnotationLabel.OUT)
    truth=s.truth_events()
    assert [e.truth for e in truth]==["IN","OUT"]
