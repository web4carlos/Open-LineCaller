from linecaller.annotation.session import AnnotationSession
def test_migration(tmp_path):
    v=tmp_path/"m.mp4";d=tmp_path/"d"
    s=AnnotationSession(str(v),str(d));s.labels={"10":{"type":"positive","x":100,"y":200,"box_size_px":24}};s.save()
    item=AnnotationSession.load_or_create(str(v),str(d)).annotation_for(10)
    assert item["box_width_px"]==24 and item["box_height_px"]==24
