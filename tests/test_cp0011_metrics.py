from linecaller.benchmark.metrics import aggregate
from linecaller.benchmark.models import ClipBenchmarkResult, EventMatch


def result():
    return ClipBenchmarkResult(
        clip_id="x",
        expected_bounces=2,
        detected_bounces=2,
        matched_bounces=1,
        false_positive_bounces=1,
        missed_bounces=1,
        precision=.5,
        recall=.5,
        median_frame_error=1.0,
        decision_agreement=1.0,
        calibration_valid=True,
        matches=(
            EventMatch(
                expected_frame=10,
                detected_frame=11,
                frame_error=1,
                expected_decision="OUT",
                detected_decision="OUT",
                decision_match=True,
            ),
        ),
    )


def test_aggregate():
    r = aggregate([result(), result()])

    assert r.clips == 2
    assert r.expected_bounces == 4
    assert r.matched_bounces == 2
    assert r.precision == .5
    assert r.recall == .5
    assert r.calibration_valid_rate == 1.0
