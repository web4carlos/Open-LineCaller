from collections import deque
import cv2

class VisionOverlay:
    def __init__(self, trail_length=24):
        self.trail = deque(maxlen=int(trail_length))

    def reset(self):
        self.trail.clear()

    def draw(self, frame, result):
        x = getattr(result, "ball_x", None)
        y = getattr(result, "ball_y", None)
        status = str(getattr(result, "tracking_status", "SEARCHING")).upper()

        if x is None or y is None:
            return frame

        x = int(round(float(x)))
        y = int(round(float(y)))
        self.trail.append((x, y))

        pts = list(self.trail)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], (0,255,255), 3, cv2.LINE_AA)

        cv2.circle(frame, (x,y), 24, (0,255,0), 4, cv2.LINE_AA)
        cv2.circle(frame, (x,y), 5, (0,255,0), -1, cv2.LINE_AA)
        cv2.line(frame, (x-34,y), (x+34,y), (0,255,0), 2, cv2.LINE_AA)
        cv2.line(frame, (x,y-34), (x,y+34), (0,255,0), 2, cv2.LINE_AA)

        cv2.putText(
            frame,
            f"BALL {status} ({x},{y})",
            (max(5,x+30), max(30,y-30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2,
            cv2.LINE_AA,
        )
        return frame
