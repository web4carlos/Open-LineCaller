from linecaller.product.health_models import HealthStatus
from linecaller.product.health_service import HealthCheckService


def test_searching_tracking_is_warning_not_failure():
    service = HealthCheckService(
        min_free_gb=0.0,
    )

    report = service.evaluate(
        camera_ok=True,
        calibration_ok=True,
        replay_ok=True,
        audio_ok=True,
        tracking_status="SEARCHING",
        fps=60,
        latency_ms=40,
        storage_path=".",
    )

    item = report.by_name("Tracking")

    assert item.status == HealthStatus.WARN
    assert item.required is False
    assert report.ready is True
