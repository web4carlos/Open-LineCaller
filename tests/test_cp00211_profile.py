from linecaller.ball.detector_profile import DetectorProfile

def test_profile_round_trip(tmp_path):
    p=DetectorProfile(min_circularity=.3,min_confidence=.5,max_candidates=5)
    path=tmp_path/"profile.json"
    p.save(path)
    loaded=DetectorProfile.load(path)
    assert loaded == p
