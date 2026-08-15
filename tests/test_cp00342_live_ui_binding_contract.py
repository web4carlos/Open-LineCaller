from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def app():
    return (ROOT/"ui"/"src"/"App.tsx").read_text(encoding="utf-8")

def test_app_uses_live_facility_hook():
    assert "useLiveFacility" in app()

def test_static_mock_courts_removed():
    assert "const courts:Court[]=" not in app()

def test_ui_reads_python_court_id():
    assert "court_id" in app()

def test_ui_reads_python_last_call():
    assert "last_call" in app()

def test_ui_reads_python_score():
    assert "court.score" in app()

def test_ui_reads_python_confidence():
    assert "court.confidence" in app()

def test_ui_reads_python_officiating_geometry():
    s=app(); assert "nearest_line" in s and "distance_in" in s

def test_ui_reads_python_service_state():
    s=app(); assert "serving_team" in s and "server_number" in s and "service_court" in s

def test_ui_has_bridge_connection_state():
    s=app(); assert "PYTHON LIVE" in s and "BRIDGE OFFLINE" in s

def test_ui_preserves_facility_and_live_views():
    s=app(); assert "COURT GRID" in s and "LIVE REFEREE / PYTHON STATE" in s
