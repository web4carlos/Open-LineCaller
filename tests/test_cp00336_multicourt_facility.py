import pytest
from linecaller.facility import FacilityManager,CourtMode,CourtStatus
def test_empty(): assert FacilityManager().court_count()==0
def test_multiple():
    f=FacilityManager(); f.add_court("c1","One"); f.add_court("c2","Two",mode=CourtMode.OPEN_PLAY); assert f.court_count()==2
def test_duplicate():
    f=FacilityManager(); f.add_court("c1","One")
    with pytest.raises(ValueError): f.add_court("c1","Again")
def test_camera_independent():
    f=FacilityManager(); f.add_court("c1","One"); f.add_court("c2","Two"); f.bind_camera("c1","CAM1"); assert f.get_court("c2").camera_id is None
def test_calibration():
    f=FacilityManager(); f.add_court("c1","One"); assert f.bind_calibration("c1","CAL1").calibration_id=="CAL1"
def test_one_live():
    f=FacilityManager(); f.add_court("c1","One"); f.add_court("c2","Two"); f.start_session("c1","M1"); assert len(f.live_courts())==1
def test_two_live():
    f=FacilityManager(); f.add_court("c1","One"); f.add_court("c2","Two"); f.start_session("c1","M1"); f.start_session("c2","M2"); assert len(f.live_courts())==2
def test_review_is_operational():
    f=FacilityManager(); f.add_court("c1","One"); f.start_session("c1","M1"); f.set_status("c1",CourtStatus.REVIEW); assert len(f.live_courts())==1
def test_offline_rejects_session():
    f=FacilityManager(); f.add_court("c1","One"); f.set_status("c1",CourtStatus.OFFLINE)
    with pytest.raises(RuntimeError): f.start_session("c1","M1")
def test_end_independent():
    f=FacilityManager(); f.add_court("c1","One"); f.add_court("c2","Two"); f.start_session("c1","M1"); f.start_session("c2","M2"); f.end_session("c1"); assert f.get_court("c2").status==CourtStatus.LIVE
