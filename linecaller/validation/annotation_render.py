import cv2

def zoom_around_candidate(frame, *, x=None, y=None, scale=2.5):
    if frame is None or x is None or y is None:
        return frame
    h,w=frame.shape[:2]
    cw=max(80,int(w/float(scale))); ch=max(80,int(h/float(scale)))
    cx=int(x); cy=int(y)
    x1=max(0,min(w-cw,cx-cw//2)); y1=max(0,min(h-ch,cy-ch//2))
    crop=frame[y1:y1+ch, x1:x1+cw]
    if crop.size == 0:
        return frame
    return cv2.resize(crop,(w,h),interpolation=cv2.INTER_LINEAR)
