from __future__ import annotations

import argparse
import time
import cv2

from linecaller.live.models import LiveFramePacket
from linecaller.live.replay import ReplayBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--buffer-frames", type=int, default=240)
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.camera)

    if not capture.isOpened():
        raise SystemExit(f"Unable to open camera {args.camera}")

    replay = ReplayBuffer(max_frames=args.buffer_frames)

    frame_number = 0
    started = time.perf_counter()
    last_report = started

    print("Open-LineCaller CP-0015 Live Camera Test")
    print("Press Q in the video window to quit.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            now = time.perf_counter()

            replay.append(
                LiveFramePacket(
                    frame_number=frame_number,
                    captured_at=now,
                    frame=frame.copy(),
                )
            )

            elapsed = max(1e-9, now - started)
            fps = (frame_number + 1) / elapsed

            cv2.putText(
                frame,
                f"LIVE  FPS {fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Replay buffer: {len(replay)} frames",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Open-LineCaller Live Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

            frame_number += 1

            if now - last_report >= 2.0:
                print(
                    f"frames={frame_number} "
                    f"fps={fps:.1f} "
                    f"replay_buffer={len(replay)}"
                )
                last_report = now

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
