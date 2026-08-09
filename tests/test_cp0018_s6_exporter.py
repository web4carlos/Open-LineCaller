import json

from linecaller.product.report_exporter import MatchReportExporter
from linecaller.product.summary_models import MatchSummary


def test_export_json(tmp_path):
    summary = MatchSummary(
        match_id="m1",
        started_at=1,
        ended_at=2,
        duration_seconds=1,
        camera_name="Camera 0",
        calibration_mode="AUTO",
        calls_total=1,
        calls_in=0,
        calls_out=1,
        calls_review=0,
        average_fps=60,
        average_latency_ms=40,
        average_confidence=.99,
        tracking_losses=0,
        replay_count=0,
    )

    path = MatchReportExporter().export_json(
        summary,
        tmp_path / "report",
    )

    assert path.suffix == ".json"

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert data["calls_out"] == 1
