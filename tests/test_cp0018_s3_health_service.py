from linecaller.product.health_models import HealthStatus
from linecaller.product.health_service import HealthCheckService


def test_good_runtime_is_ready(monkeypatch):
    service = HealthCheckService(
        min_fps=30,
        max_latency_ms=100,
        min_free_gb=0.0,
    )

    report = service.evaluate(
        camera_ok=True,
        calibration_ok=True,
        replay_ok=True,
        audio_ok=True,
        tracking_status="LOCKED",
        fps=60,
        latency_ms=40,
        storage_path=".",
    )

    assert report.ready is True


def test_low_fps_blocks_ready():
    service = HealthCheckService(
        min_fps=30,
        min_free_gb=0.0,
    )

    report = service.evaluate(
        camera_ok=True,
        calibration_ok=True,
        replay_ok=True,
        audio_ok=True,
        tracking_status="LOCKED",
        fps=20,
        latency_ms=40,
        storage_path=".",
    )

    assert report.by_name("FPS").status == HealthStatus.FAIL
    assert report.ready is False


def test_high_latency_blocks_ready():
    service = HealthCheckService(
        max_latency_ms=100,
        min_free_gb=0.0,
    )

    report = service.evaluate(
        camera_ok=True,
        calibration_ok=True,
        replay_ok=True,
        audio_ok=True,
        tracking_status="LOCKED",
        fps=60,
        latency_ms=150,
        storage_path=".",
    )

    assert report.by_name("Latency").status == HealthStatus.FAIL
    assert report.ready is False


def test_audio_warning_is_nonblocking():
    service = HealthCheckService(
        min_free_gb=0.0,
    )

    report = service.evaluate(
        camera_ok=True,
        calibration_ok=True,
        replay_ok=True,
        audio_ok=False,
        tracking_status="LOCKED",
        fps=60,
        latency_ms=40,
        storage_path=".",
    )

    assert report.by_name("Audio").status == HealthStatus.WARN
    assert report.ready is True
