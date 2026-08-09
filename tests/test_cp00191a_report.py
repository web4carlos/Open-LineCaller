import json

from linecaller.validation.models import (
    ValidationMetrics,
)
from linecaller.validation.offline_models import (
    OfflineRunStats,
    OfflineValidationResult,
    OfflineVideoInfo,
)
from linecaller.validation.offline_report import (
    export_offline_validation_report,
)


def test_offline_report(tmp_path):
    result = OfflineValidationResult(
        video_info=OfflineVideoInfo(
            path="x.mp4",
            frame_count=100,
            source_fps=60,
            width=1920,
            height=1080,
        ),
        run_stats=OfflineRunStats(
            frames_processed=100,
            wall_seconds=2,
            effective_processing_fps=50,
        ),
        predictions=(),
        comparisons=(),
        metrics=ValidationMetrics(
            total_truth_events=0,
            automatic_calls=0,
            correct_automatic=0,
            incorrect_automatic=0,
            reviews=0,
            misses=0,
            false_in=0,
            false_out=0,
            automatic_accuracy=0,
            coverage=0,
            review_rate=0,
            miss_rate=0,
            average_confidence=0,
            average_latency_ms=0,
        ),
    )

    path = export_offline_validation_report(
        result,
        tmp_path / "report.json",
    )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert data["mode"] == "OFFLINE"
    assert data["video"]["source_fps"] == 60
    assert (
        data["processing"][
            "effective_processing_fps"
        ]
        == 50
    )
