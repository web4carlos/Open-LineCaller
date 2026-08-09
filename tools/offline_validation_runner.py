from __future__ import annotations

import argparse

from linecaller.validation.io import (
    load_truth_jsonl,
)
from linecaller.validation.offline_report import (
    export_offline_validation_report,
)
from linecaller.validation.offline_runner import (
    OfflineValidationRunner,
)
from linecaller.validation.prediction_io import (
    save_predictions_jsonl,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
    )
    parser.add_argument(
        "--predictions",
        required=True,
    )
    parser.add_argument(
        "--report",
        required=True,
    )
    parser.add_argument(
        "--frame-tolerance",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    truth = load_truth_jsonl(
        args.ground_truth
    )

    runner = OfflineValidationRunner(
        frame_tolerance=args.frame_tolerance
    )

    result = runner.run(
        video_path=args.video,
        truth_events=truth,
    )

    save_predictions_jsonl(
        result.predictions,
        args.predictions,
    )

    export_offline_validation_report(
        result,
        args.report,
    )

    m = result.metrics
    s = result.run_stats
    v = result.video_info

    print("")
    print(
        "Open-LineCaller OFFLINE Validation"
    )
    print(
        f"video={v.path}"
    )
    print(
        f"frames_processed={s.frames_processed}"
    )
    print(
        f"source_fps={v.source_fps:.2f}"
    )
    print(
        f"effective_processing_fps="
        f"{s.effective_processing_fps:.2f}"
    )
    print(
        f"automatic_accuracy="
        f"{m.automatic_accuracy:.4f}"
    )
    print(
        f"coverage={m.coverage:.4f}"
    )
    print(
        f"review_rate={m.review_rate:.4f}"
    )
    print(
        f"miss_rate={m.miss_rate:.4f}"
    )
    print(
        f"false_in={m.false_in}"
    )
    print(
        f"false_out={m.false_out}"
    )


if __name__ == "__main__":
    main()
