from __future__ import annotations
import argparse
import cv2

from linecaller.autocalibration.engine_v2 import (
    MultiHypothesisAutoCalibrationEngine,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Unable to read image: {args.image}")

    result = MultiHypothesisAutoCalibrationEngine().propose(frame)
    ranked = result["ranked"]

    print("")
    print("Open-LineCaller CP-0009 Multi-Hypothesis Auto-Calibration")
    print("----------------------------------------------------------")
    print(f"Lines:              {result['line_count']}")
    print(f"Family A:           {result['family_a_count']}")
    print(f"Family B:           {result['family_b_count']}")
    print(f"Hypotheses:         {result['hypothesis_count']}")
    print(f"Disposition:        {ranked.disposition.value}")
    print(f"Score margin:       {ranked.confidence_margin:.3f}")

    if ranked.best:
        print(f"Best score:         {ranked.best.score:.3f}")
        print("Best corners:")
        for i, (x, y) in enumerate(ranked.best.corners, start=1):
            print(f"  {i}: ({x:.1f}, {y:.1f})")

    if ranked.runner_up:
        print(f"Runner-up score:    {ranked.runner_up.score:.3f}")

    for reason in ranked.reasons:
        print(f"Reason: {reason}")


if __name__ == "__main__":
    main()
