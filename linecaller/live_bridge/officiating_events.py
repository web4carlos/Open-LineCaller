from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path

@dataclass(frozen=True)
class OfficiatingEvent:
    frame:int; decision:str; confidence:float
    image_x:float|None=None; image_y:float|None=None
    nearest_line:str|None=None; signed_distance_m:float|None=None
    total_uncertainty_m:float|None=None; bounce_confidence:float|None=None
    calibration_valid:bool|None=None; explanation:tuple[str,...]=()
    def to_dict(self):
        d=asdict(self); d['explanation']=list(self.explanation); return d

class OfficiatingTimeline:
    def __init__(self,events=()): self.events=tuple(sorted(events,key=lambda e:e.frame))
    @classmethod
    def load(cls,path):
        p=Path(path)
        if not p.is_file(): raise FileNotFoundError(p)
        out=[]
        for i,raw in enumerate(p.read_text(encoding='utf-8-sig').splitlines(),1):
            if not raw.strip(): continue
            x=json.loads(raw); image=x.get('image') or {}; frame=x.get('bounce_frame',x.get('frame'))
            if frame is None: raise ValueError(f'missing frame at line {i}')
            decision=str(x.get('decision','')).upper()
            if decision not in {'IN','OUT','REVIEW'}: raise ValueError(f'invalid decision {decision!r} at line {i}')
            out.append(OfficiatingEvent(int(frame),decision,float(x.get('confidence',0.0)),_f(image.get('x',x.get('image_x'))),_f(image.get('y',x.get('image_y'))),_s(x.get('nearest_line')),_f(x.get('signed_distance_m')),_f(x.get('total_uncertainty_m')),_f(x.get('bounce_confidence')),_b(x.get('calibration_valid')),tuple(str(v) for v in x.get('explanation',()))))
        return cls(tuple(out))
    def event_for_frame(self,frame,hold_frames=45):
        hit=None
        for e in self.events:
            if e.frame>frame: break
            hit=e
        return hit if hit and frame<=hit.frame+max(0,int(hold_frames)) else None
    def to_dict(self): return {'count':len(self.events),'events':[e.to_dict() for e in self.events]}
def _f(v): return None if v in (None,'') else float(v)
def _s(v): return None if v in (None,'') else str(v)
def _b(v):
    if v in (None,''): return None
    if isinstance(v,bool): return v
    return str(v).strip().lower() in {'1','true','yes'}
