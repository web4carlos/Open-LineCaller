from linecaller.facility import FacilityManager,CourtMode,CourtStatus
f=FacilityManager("Open LineCaller Demo Club")
f.add_court("court-1","Center Court"); f.add_court("court-2","Court 2",mode=CourtMode.OPEN_PLAY)
f.add_court("court-3","Court 3"); f.add_court("court-4","Court 4",mode=CourtMode.OPEN_PLAY)
f.bind_camera("court-1","CAM-01"); f.bind_camera("court-2","CAM-02"); f.bind_camera("court-3","CAM-03")
f.start_session("court-1","MATCH-001",score_display="7-5-1")
f.start_session("court-2","OPEN-PLAY-021",score_display="4-3-2"); f.set_status("court-2",CourtStatus.REVIEW)
f.set_status("court-3",CourtStatus.READY); f.set_status("court-4",CourtStatus.OFFLINE)
print(f"facility={f.state.name} courts={f.court_count()} live={len(f.live_courts())}")
for c in f.state.courts:
 print(f"{c.court_id}: name={c.name} mode={c.mode.value} status={c.status.value} camera={c.camera_id or '-'} session={c.active_session_id or '-'} score={c.score_display or '-'}")
