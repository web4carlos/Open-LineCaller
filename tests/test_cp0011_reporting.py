from linecaller.benchmark.models import BenchmarkReport
from linecaller.benchmark.reporting import write_report


def test_write_report(tmp_path):
    report = BenchmarkReport(
        clips=0,
        expected_bounces=0,
        detected_bounces=0,
        matched_bounces=0,
        false_positive_bounces=0,
        missed_bounces=0,
        precision=0.0,
        recall=0.0,
        median_frame_error=None,
        decision_agreement=None,
        calibration_valid_rate=0.0,
        clip_results=(),
    )

    json_path, txt_path = write_report(report, tmp_path)

    assert json_path.exists()
    assert txt_path.exists()
    assert "Benchmark Report" in txt_path.read_text(encoding="utf-8")
