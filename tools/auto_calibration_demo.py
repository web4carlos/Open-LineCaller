from __future__ import annotations

import argparse
import cv2

from linecaller.autocalibration.engine import AutoCalibrationEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Unable to read image: {args.image}")

    proposal = AutoCalibrationEngine().propose(frame)

    print("")
    print("Open-LineCaller CP-0008 Assisted Auto-Calibration")
    print("--------------------------------------------------")
    print(f"Disposition:        {proposal.disposition.value}")
    print(f"Confidence:         {proposal.confidence:.3f}")
    print(f"Total lines:        {proposal.line_count}")
    print(f"Family A lines:     {proposal.family_a_count}")
    print(f"Family B lines:     {proposal.family_b_count}")
    print(f"Orientation sep:    {proposal.orientation_separation_deg:.1f} deg")
    print(f"Quad area ratio:    {proposal.quadrilateral_area_ratio:.3f}")

    if proposal.corners:
        print("Corners:")
        for i, (x, y) in enumerate(proposal.corners, start=1):
            print(f"  {i}: ({x:.1f}, {y:.1f})")

    if proposal.reasons:
        print("Reasons:")
        for reason in proposal.reasons:
            print(f"  - {reason}")


if __name__ == "__main__":
    main()
