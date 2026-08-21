from __future__ import annotations

from fastapi.testclient import TestClient

from linecaller.api.live_app import app


def test_cp0036_2_4_1_exact_frame_controls_are_exposed():
    root = TestClient(app).get("/")
    assert root.status_code == 200
    html = root.text
    for token in (
        'id="gotoFrame"',
        'id="gotoFrameBtn"',
        'id="rangeFrom"',
        'id="rangeTo"',
        'id="analyzeRange"',
        'Exact Frame Navigation',
        'Analyze Exact Range',
    ):
        assert token in html


def test_cp0036_2_4_1_range_is_inclusive_and_auto_stops():
    html = TestClient(app).get("/").text
    assert "analysisStopFrame" in html
    assert "processedVideoFrame >= analysisStopFrame" in html
    assert "Exact analysis range complete at video frame" in html
    assert "await seekToFrame(fromFrame)" in html
    assert "await startLineCaller(true)" in html


def test_cp0036_2_4_1_keyboard_navigation_is_present():
    html = TestClient(app).get("/").text
    assert "ev.key === 'ArrowLeft'" in html
    assert "ev.key === 'ArrowRight'" in html
    assert "ev.shiftKey ? 10 : 1" in html
    assert "ev.key === 'g' || ev.key === 'G'" in html
    assert "ev.code === 'Space'" in html


def test_cp0036_2_4_1_frame_math_uses_analysis_fps():
    html = TestClient(app).get("/").text
    assert "Math.round((video.currentTime || 0) * fps())" in html
    assert "await seekTo(targetFrame / fps())" in html
    assert "Math.ceil(duration * fps()) - 1" in html
