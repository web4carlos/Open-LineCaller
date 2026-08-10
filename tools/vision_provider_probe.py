import argparse
import cv2
from linecaller.vision import create_vision_provider

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--provider", default="classical")
    p.add_argument("--max-frames", type=int, default=300)
    args = p.parse_args()

    provider = create_vision_provider(args.provider)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Unable to open: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frames = detections = 0
    try:
        while frames < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            result = provider.detect(
                frame=frame,
                frame_number=frames,
                timestamp=frames / fps,
            )
            if result.x is not None and result.y is not None:
                detections += 1
            frames += 1
    finally:
        cap.release()
        provider.shutdown()

    print(f"provider={provider.name}")
    print(f"frames={frames}")
    print(f"detections={detections}")

if __name__ == "__main__":
    main()
