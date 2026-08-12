import cv2
import numpy as np
from linecaller.annotation.auto_assist import HSVAutoAssist, CombinedAutoAssist

def test_hsv_detects_neon_ball():
    frame=np.zeros((300,500,3),dtype=np.uint8)
    cv2.circle(frame,(250,140),8,(0,255,255),-1)
    s=HSVAutoAssist().suggest(frame)
    assert s is not None
    assert abs(s.x-250)<5
    assert abs(s.y-140)<5

def test_click_refine():
    frame=np.zeros((200,300,3),dtype=np.uint8)
    cv2.ellipse(frame,(140,90),(8,5),0,0,360,(0,255,255),-1)
    s=CombinedAutoAssist().refine_click(frame,138,92)
    assert abs(s.x-140)<5
    assert abs(s.y-90)<5
