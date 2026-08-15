import cv2,numpy as np
from linecaller.dcf.template_matcher import DCFBallTemplateMatcher
def test_local_match():
 f=np.zeros((120,180,3),np.uint8);cv2.circle(f,(100,60),7,(220,220,220),-1);t=f[51:69,91:109].copy();r=DCFBallTemplateMatcher(.5,(1.,)).match(f,t,(70,30,130,90));assert r.found and abs(r.x-100)<2
def test_roi_excludes_ball():
 f=np.zeros((120,180,3),np.uint8);cv2.circle(f,(100,60),7,(220,220,220),-1);t=f[51:69,91:109].copy();r=DCFBallTemplateMatcher(.5,(1.,)).match(f,t,(0,0,50,50));assert not r.found
