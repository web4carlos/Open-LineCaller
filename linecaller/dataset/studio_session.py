from dataclasses import replace
from linecaller.dataset.models import BallBox, BounceAnnotation, ClipAnnotation, FrameAnnotation

class DatasetStudioSession:
    def __init__(self, *, clip_id, source_video, width, height, fps, frame_count):
        self.clip_id=clip_id; self.source_video=source_video
        self.width=int(width); self.height=int(height)
        self.fps=float(fps); self.frame_count=int(frame_count)
        self._frames={}; self._bounces={}

    @classmethod
    def from_annotation(cls, a):
        s=cls(clip_id=a.clip_id, source_video=a.source_video, width=a.width, height=a.height, fps=a.fps, frame_count=a.frame_count)
        s._frames={f.frame_number:f for f in a.frames}
        s._bounces={b.frame_number:b for b in a.bounces}
        return s

    def _check(self,f):
        if not 0 <= f < self.frame_count: raise ValueError(f"Frame out of range: {f}")

    def frame_annotation(self,f): return self._frames.get(f)
    def bounce_annotation(self,f): return self._bounces.get(f)

    def set_ball_box(self,f,box,visible=True,occluded=False,quality="OK"):
        self._check(f)
        errs=box.validate()
        if errs: raise ValueError("; ".join(errs))
        self._frames[f]=FrameAnnotation(f,box,bool(visible),bool(occluded),str(quality))

    def remove_ball_box(self,f):
        cur=self._frames.get(f)
        if cur: self._frames[f]=FrameAnnotation(f,None,cur.visible,cur.occluded,cur.quality)

    def set_flags(self,f,*,visible,occluded):
        self._check(f)
        cur=self._frames.get(f,FrameAnnotation(frame_number=f))
        self._frames[f]=replace(cur,visible=bool(visible),occluded=bool(occluded))

    def toggle_bounce(self,f,*,x,y,decision=None):
        self._check(f)
        if f in self._bounces:
            del self._bounces[f]; return False
        if decision not in (None,"IN","OUT","REVIEW"): raise ValueError("Invalid decision")
        self._bounces[f]=BounceAnnotation(f,float(x),float(y),decision)
        return True

    def set_bounce_decision(self,f,decision):
        if decision not in (None,"IN","OUT","REVIEW"): raise ValueError("Invalid decision")
        cur=self._bounces.get(f)
        if cur is None: raise ValueError("No bounce on current frame")
        self._bounces[f]=replace(cur,decision=decision)

    def to_annotation(self):
        return ClipAnnotation(
            clip_id=self.clip_id, source_video=self.source_video,
            width=self.width, height=self.height, fps=self.fps, frame_count=self.frame_count,
            frames=sorted(self._frames.values(),key=lambda x:x.frame_number),
            bounces=sorted(self._bounces.values(),key=lambda x:x.frame_number)
        )
