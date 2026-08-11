import cv2,numpy as np
from linecaller.annotation.smart_bbox import SmartBBoxDetector
def test_fallback():
    frame=np.zeros((200,300,3),dtype=np.uint8);c=SmartBBoxDetector().detect(frame,100,80)
    assert c.center_x==100 and c.center_y==80 and c.method=="fallback-click"
def test_yellow_blob():
    frame=np.zeros((240,320,3),dtype=np.uint8)
    cv2.ellipse(frame,(160,120),(8,5),0,0,360,(0,255,255),-1)
    c=SmartBBoxDetector().detect(frame,158,121)
    assert abs(c.center_x-160)<5 and abs(c.center_y-120)<5
    assert c.width>=10 and c.height>=10 and c.score>0
