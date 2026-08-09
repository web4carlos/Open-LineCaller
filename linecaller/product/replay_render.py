from __future__ import annotations

import cv2


def replay_zoom(frame, *, x=None, y=None, scale: float = 2.0):
    if frame is None or x is None or y is None:
        return frame

    h, w = frame.shape[:2]

    crop_w = max(40, int(w / scale))
    crop_h = max(40, int(h / scale))

    cx = int(x)
    cy = int(y)

    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    x1 = max(0, x2 - crop_w)
    y1 = max(0, y2 - crop_h)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return frame

    return cv2.resize(
        crop,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )
