from linecaller.dataset.keyframes import AnnotationKeyframeIndex


def test_keyframes():
    k = AnnotationKeyframeIndex()
    k.mark(10)
    k.mark(20)

    assert k.is_keyframe(10)
    assert k.previous(15) == 10
    assert k.next(15) == 20

    assert k.toggle(10) is False
    assert not k.is_keyframe(10)
