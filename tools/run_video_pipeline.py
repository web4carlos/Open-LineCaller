from __future__ import annotations

import argparse

from linecaller.pipeline.integrated import IntegratedVideoPipeline


def parser():
    p = argparse.ArgumentParser(
        description="Open-LineCaller integrated video decision pipeline"
    )
    p.add_argument("--video", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--events", required=True)
    return p


def main():
    args = parser().parse_args()

    pipeline = IntegratedVideoPipeline()

    summary = pipeline.run(
        video_path=args.video,
        calibration_path=args.calibration,
        output_video_path=args.output,
        events_path=args.events,
    )

    print("")
    print("Open-LineCaller CP-0006 summary")
    print("--------------------------------")
    print(f"Frames:       {summary.frames}")
    print(f"Tracked:      {summary.tracked}")
    print(f"Predicted:    {summary.predicted}")
    print(f"Lost:         {summary.lost}")
    print(f"Bounces:      {summary.bounces}")
    print(f"IN:           {summary.decisions_in}")
    print(f"OUT:          {summary.decisions_out}")
    print(f"REVIEW:       {summary.decisions_review}")


if __name__ == "__main__":
    main()
