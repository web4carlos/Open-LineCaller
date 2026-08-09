from linecaller.validation.session import ValidationSession

def test_session(tmp_path):
    video = tmp_path / "match.mp4"
    video.write_bytes(b"")
    s = ValidationSession()
    s.set_video(video)
    e = s.add_truth(frame=123, truth="OUT")
    assert e.event_id == "match-00000123"

def test_replace_same_frame(tmp_path):
    video = tmp_path / "match.mp4"
    video.write_bytes(b"")
    s = ValidationSession()
    s.set_video(video)
    s.add_truth(frame=10, truth="IN")
    s.add_truth(frame=10, truth="OUT")
    assert len(s.truth_events) == 1
    assert s.truth_events[0].truth == "OUT"
