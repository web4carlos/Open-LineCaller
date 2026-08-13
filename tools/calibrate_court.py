from __future__ import annotations

import argparse
from pathlib import Path
import cv2

from linecaller.court_geometry import CourtCalibration


POINT_NAMES = [
    "NEAR-LEFT",
    "NEAR-RIGHT",
    "FAR-RIGHT",
    "FAR-LEFT",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--frame", type=int, default=0)
    a = p.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {a.video}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, base = cap.read()
    cap.release()

    if not ok:
        raise SystemExit(f"Cannot read frame {a.frame}")

    points = []
    display = base.copy()

    window = "Open-LineCaller Court Calibration"

    def redraw():
        nonlocal display
        display = base.copy()

        for i, (x, y) in enumerate(points):
            cv2.circle(display, (int(x), int(y)), 8, (0, 255, 255), -1)
            cv2.putText(
                display,
                str(i + 1),
                (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        next_name = (
            POINT_NAMES[len(points)]
            if len(points) < 4
            else "DONE - press S"
        )

        cv2.rectangle(
            display,
            (10, 10),
            (800, 90),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            display,
            f"Click {len(points)+1}/4: {next_name}" if len(points)<4 else next_name,
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            "U=undo  R=reset  S=save  Q=quit",
            (25, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    print("Click corners in this exact order:")
    print("1. NEAR-LEFT")
    print("2. NEAR-RIGHT")
    print("3. FAR-RIGHT")
    print("4. FAR-LEFT")
    print("Then press S to save.")

    while True:
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break

        if key == ord("u") and points:
            points.pop()
            redraw()

        if key == ord("r"):
            points.clear()
            redraw()

        if key == ord("s"):
            if len(points) != 4:
                print("Need exactly 4 points before saving.")
                continue

            calibration = CourtCalibration(
                image_points=tuple(points)
            )
            calibration.save(a.output)
            print(f"saved={Path(a.output)}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
