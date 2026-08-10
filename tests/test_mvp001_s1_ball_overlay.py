import numpy as np
from linecaller.product_shell.vision_overlay import VisionOverlay

class Result:
    ball_x = 100.0
    ball_y = 80.0
    tracking_status = "TRACKING"

def test_overlay_draws_pixels():
    frame = np.zeros((200,300,3), dtype=np.uint8)
    overlay = VisionOverlay()
    out = overlay.draw(frame, Result())
    assert out.sum() > 0
    assert len(overlay.trail) == 1

def test_overlay_reset():
    frame = np.zeros((200,300,3), dtype=np.uint8)
    overlay = VisionOverlay()
    overlay.draw(frame, Result())
    overlay.reset()
    assert len(overlay.trail) == 0
