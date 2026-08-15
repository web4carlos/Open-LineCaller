import cv2,numpy as np
from linecaller.dcf.one_ball_field import OneBallOneDCF,DCFCell
def test_one_ball_one_field():
 f=np.zeros((100,160,3),np.uint8);cv2.circle(f,(120,60),6,(230,230,230),-1);t=f[53:67,113:127].copy();cells=[DCFCell(0,0,0,20,70,90,True,1),DCFCell(1,0,80,20,150,90,False,1)];r=OneBallOneDCF(.5).search(f,t,cells);assert r.found and not r.cell.inside
