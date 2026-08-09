from __future__ import annotations

import time

from .comparison import compare_events
from .event_matcher import (
    match_predictions_to_truth,
)
from .metrics import compute_metrics
from .offline_models import (
    OfflineRunStats,
    OfflineValidationResult,
    OfflineVideoInfo,
)
from .offline_pipeline import (
    LivePipelineOfflineAdapter,
)
from .video_iterator import (
    OfflineFrameIterator,
)


class OfflineValidationRunner:
    def __init__(
        self,
        *,
        pipeline=None,
        frame_tolerance: int = 3,
    ):
        self.pipeline = (
            pipeline
            or LivePipelineOfflineAdapter()
        )
        self.frame_tolerance = int(
            frame_tolerance
        )

    def run(
        self,
        *,
        video_path,
        truth_events,
    ):
        iterator = OfflineFrameIterator(
            video_path
        )

        meta = iterator.metadata()

        predictions = []
        frames_processed = 0

        started = time.perf_counter()

        for item in iterator:
            prediction = self.pipeline.process(
                frame_number=item.frame_number,
                timestamp=item.timestamp,
                frame=item.frame,
            )

            if prediction is not None:
                predictions.append(
                    prediction
                )

            frames_processed += 1

        wall_seconds = max(
            0.0,
            time.perf_counter() - started,
        )

        effective_fps = (
            frames_processed / wall_seconds
            if wall_seconds > 0
            else 0.0
        )

        matched_predictions = (
            match_predictions_to_truth(
                truth_events,
                predictions,
                frame_tolerance=self.frame_tolerance,
            )
        )

        comparisons = compare_events(
            truth_events,
            matched_predictions,
        )

        metrics = compute_metrics(
            comparisons
        )

        return OfflineValidationResult(
            video_info=OfflineVideoInfo(
                path=str(video_path),
                frame_count=meta[
                    "frame_count"
                ],
                source_fps=meta[
                    "source_fps"
                ],
                width=meta["width"],
                height=meta["height"],
            ),
            run_stats=OfflineRunStats(
                frames_processed=frames_processed,
                wall_seconds=wall_seconds,
                effective_processing_fps=effective_fps,
            ),
            predictions=tuple(
                matched_predictions
            ),
            comparisons=tuple(
                comparisons
            ),
            metrics=metrics,
        )
