from __future__ import annotations

import argparse
import cv2

from linecaller.proposals.yolo_engine import YOLOProposalEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--class-name", default="sports ball")
    parser.add_argument("--confidence", type=float, default=0.15)
    args = parser.parse_args()

    frame = cv2.imread(args.image)

    if frame is None:
        raise SystemExit(f"Unable to read image: {args.image}")

    engine = YOLOProposalEngine(
        weights=args.weights,
        class_name=args.class_name or None,
        min_confidence=args.confidence,
        fail_open=False,
    )

    result = engine.propose(0, frame)

    print("")
    print("Open-LineCaller CP-0014C YOLO Proposal Demo")
    print("--------------------------------------------")
    print(f"Engine:       {result.engine_name}")
    print(f"Latency:      {result.latency_ms:.1f} ms")
    print(f"Proposals:    {len(result.proposals)}")

    for index, proposal in enumerate(result.proposals, 1):
        print(
            f"{index}: "
            f"x={proposal.x:.1f} y={proposal.y:.1f} "
            f"w={proposal.width:.1f} h={proposal.height:.1f} "
            f"conf={proposal.confidence:.3f} "
            f"class={proposal.metadata.get('class_name')}"
        )


if __name__ == "__main__":
    main()
