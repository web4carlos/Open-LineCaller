from __future__ import annotations

from statistics import median

from .models import BenchmarkReport, ClipBenchmarkResult


def safe_ratio(a: int, b: int) -> float:
    return a / b if b else 0.0


def aggregate(results: list[ClipBenchmarkResult]) -> BenchmarkReport:
    expected = sum(r.expected_bounces for r in results)
    detected = sum(r.detected_bounces for r in results)
    matched = sum(r.matched_bounces for r in results)
    fp = sum(r.false_positive_bounces for r in results)
    fn = sum(r.missed_bounces for r in results)

    frame_errors = [
        match.frame_error
        for result in results
        for match in result.matches
    ]

    decision_flags = [
        match.decision_match
        for result in results
        for match in result.matches
        if match.decision_match is not None
    ]

    calibration_valid = sum(1 for r in results if r.calibration_valid)

    return BenchmarkReport(
        clips=len(results),
        expected_bounces=expected,
        detected_bounces=detected,
        matched_bounces=matched,
        false_positive_bounces=fp,
        missed_bounces=fn,
        precision=safe_ratio(matched, matched + fp),
        recall=safe_ratio(matched, matched + fn),
        median_frame_error=median(frame_errors) if frame_errors else None,
        decision_agreement=(
            sum(1 for flag in decision_flags if flag) / len(decision_flags)
            if decision_flags else None
        ),
        calibration_valid_rate=safe_ratio(calibration_valid, len(results)),
        clip_results=tuple(results),
    )
