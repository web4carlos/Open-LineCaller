from linecaller.live_bridge import LiveCourtState,LiveStateStore

def court(cid="c1"): return LiveCourtState(cid,"Court","LIVE","MATCH","0-0-2","IN",.91)

def test_empty_snapshot():
    s=LiveStateStore("Club"); x=s.snapshot(); assert x.sequence==0 and x.courts==()

def test_upsert_adds_court():
    s=LiveStateStore(); s.upsert(court()); assert len(s.snapshot().courts)==1

def test_sequence_advances():
    s=LiveStateStore(); assert s.upsert(court())==1; assert s.upsert(court())==2

def test_upsert_replaces_same_court():
    s=LiveStateStore(); s.upsert(court()); s.upsert(LiveCourtState("c1","Court","REVIEW","MATCH","1-0-2","REVIEW")); assert len(s.snapshot().courts)==1

def test_lookup():
    s=LiveStateStore(); s.upsert(court()); assert s.court("c1").last_call=="IN"

def test_missing_lookup():
    assert LiveStateStore().court("missing") is None

def test_snapshot_serializable_shape():
    s=LiveStateStore("Club"); s.upsert(court()); d=s.snapshot().to_dict(); assert d["facility_name"]=="Club" and d["courts"][0]["court_id"]=="c1"

def test_confidence_preserved():
    s=LiveStateStore(); s.upsert(court()); assert s.court("c1").confidence==.91

def test_multiple_courts():
    s=LiveStateStore(); s.upsert(court("c1")); s.upsert(court("c2")); assert len(s.snapshot().courts)==2

def test_facility_name_preserved():
    assert LiveStateStore("My Club").snapshot().facility_name=="My Club"
