from linecaller.product_shell.session import MatchSessionStats

def test_session_counts_calls():
    s=MatchSessionStats()
    s.record_event("IN"); s.record_event("OUT"); s.record_event("REVIEW"); s.record_event(None)
    assert s.events==3
    assert s.calls_in==1
    assert s.calls_out==1
    assert s.calls_review==1

def test_session_reset():
    s=MatchSessionStats(frames=100,events=2,calls_in=1,calls_out=1)
    s.reset()
    assert s.frames==0
    assert s.events==0
    assert s.calls_in==0
    assert s.calls_out==0
