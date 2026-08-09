from linecaller.validation.offline_models import (
    OfflineRunStats,
)


def test_offline_stats():
    stats = OfflineRunStats(
        frames_processed=120,
        wall_seconds=2,
        effective_processing_fps=60,
    )

    assert stats.frames_processed == 120
    assert stats.effective_processing_fps == 60
