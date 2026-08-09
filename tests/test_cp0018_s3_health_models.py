from linecaller.product.health_models import (
    HealthCheckItem,
    HealthReport,
    HealthStatus,
)


def test_health_report_ready_when_required_pass():
    report = HealthReport((
        HealthCheckItem("A", HealthStatus.PASS, "ok"),
        HealthCheckItem(
            "Optional",
            HealthStatus.WARN,
            "warn",
            required=False,
        ),
    ))

    assert report.ready is True


def test_health_report_blocks_required_failure():
    report = HealthReport((
        HealthCheckItem("A", HealthStatus.FAIL, "bad"),
    ))

    assert report.ready is False
