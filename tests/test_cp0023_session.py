from linecaller.annotation.session import AnnotationSession
def test_session_roundtrip(tmp_path):
    video=tmp_path/'match.mp4';dataset=tmp_path/'dataset'
    s=AnnotationSession(str(video),str(dataset));s.current_frame=42;s.mark_positive(42,100,200);s.mark_negative(43);s.save()
    loaded=AnnotationSession.load_or_create(str(video),str(dataset))
    assert loaded.current_frame==42
    assert loaded.annotation_for(42)['type']=='positive'
    assert loaded.annotation_for(43)['type']=='negative'
    assert loaded.stats().labeled==2
