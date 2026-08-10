from linecaller.validation.annotation_models import AnnotationCandidate, AnnotationLabel

def test_candidate():
    c=AnnotationCandidate(frame=100,confidence=.9,suggested_decision="OUT")
    assert c.frame==100
    assert c.suggested_decision=="OUT"

def test_labels():
    assert AnnotationLabel.IN.value=="IN"
    assert AnnotationLabel.OUT.value=="OUT"
    assert AnnotationLabel.SKIP.value=="SKIP"
