from __future__ import annotations
import cv2
import numpy as np

def render_soft_floor_glow(frame_bgr, polygon, *, strength=0.68):
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.round(np.asarray(polygon, dtype=np.float32)).astype(np.int32)
    cv2.fillConvexPoly(mask, poly, 255)
    k = max(5, int(round(max(h, w) * 0.015)) | 1)
    glow = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0
    out = frame_bgr.astype(np.float32).copy()
    out += 105.0 * float(strength) * glow[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)
