from linecaller.validation.models import (
    ValidationPrediction,
    ValidationTruthEvent,
)
from linecaller.validation.offline_runner import (
    OfflineValidationRunner,
)


class FakePipeline:
    def process(
        self,
        *,
        frame_number,
        timestamp,
        frame,
    ):
        if frame_number == 2:
            return ValidationPrediction(
                event_id="p2",
                frame=2,
                prediction="IN",
                confidence=.99,
                latency_ms=5,
            )

        if frame_number == 4:
            return ValidationPrediction(
                event_id="p4",
                frame=4,
                prediction="OUT",
                confidence=.98,
                latency_ms=6,
            )

        return None


class FakeIterator:
    def metadata(self):
        return {
            "frame_count": 5,
            "source_fps": 30.0,
            "width": 100,
            "height": 100,
        }

    def __iter__(self):
        from linecaller.validation.video_iterator import OfflineFrame

        for i in range(5):
            yield OfflineFrame(
                frame_number=i,
                timestamp=i / 30.0,
                frame=f"f{i}",
            )


def test_runner_metrics(monkeypatch):
    monkeypatch.setattr(
        "linecaller.validation.offline_runner.OfflineFrameIterator",
        lambda path: FakeIterator(),
    )

    truth = [
        ValidationTruthEvent(
            "t2",
            "x.mp4",
            2,
            "IN",
        ),
        ValidationTruthEvent(
            "t4",
            "x.mp4",
            4,
            "OUT",
        ),
    ]

    runner = OfflineValidationRunner(
        pipeline=FakePipeline(),
        frame_tolerance=0,
    )

    result = runner.run(
        video_path="x.mp4",
        truth_events=truth,
    )

    assert result.run_stats.frames_processed == 5
    assert result.metrics.automatic_accuracy == 1.0
    assert result.metrics.coverage == 1.0
