from __future__ import annotations

from fastapi.testclient import TestClient

from linecaller.api.live_app import app


def test_root_is_video_first_but_keeps_live_camera_fallback():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    text = response.text

    assert "Video-First Calibration Wizard" in text
    assert "Upload Video" in text
    assert "Live Camera" in text
    assert "getUserMedia" in text


def test_video_transport_supports_scrub_play_and_frame_steps():
    client = TestClient(app)
    text = client.get("/").text

    assert 'id="scrubber"' in text
    assert 'id="playPause"' in text
    assert "Step Frame" in text
    assert 'id="analysisFps"' in text


def test_video_analysis_is_deterministic_frame_by_frame():
    client = TestClient(app)
    text = client.get("/").text

    assert "frameStepSeconds" in text
    assert "analyzeLoop" in text
    assert "video.currentTime + frameStepSeconds()" in text
    assert "/api/live/frame" in text


def test_existing_guided_calibration_and_ball_replacement_remain_present():
    client = TestClient(app)
    text = client.get("/").text

    assert "Capture Clean Court" in text
    assert "Build Outside Grid" in text
    assert "Select Ball From Current Frame" in text
    assert "Change Ball" in text


def test_engineering_console_is_still_available():
    client = TestClient(app)
    response = client.get("/engineering")
    assert response.status_code == 200
    assert "Configure live court" in response.text


def test_health_reports_cp0036_2():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["version"] == "CP-0036.2"