from __future__ import annotations

import cv2

from linecaller.apps.live_audio import LiveAudioNotifier


def camera_reachable(camera_index: int) -> bool:
    capture = cv2.VideoCapture(int(camera_index))
    try:
        return bool(capture.isOpened())
    finally:
        capture.release()


def audio_hook_available() -> bool:
    try:
        notifier = LiveAudioNotifier()
        return notifier is not None
    except Exception:
        return False
