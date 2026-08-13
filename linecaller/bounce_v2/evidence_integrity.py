from dataclasses import dataclass
from enum import Enum

class ContactEvidence(str,Enum):
    OBSERVED_CONTACT="OBSERVED_CONTACT"
    INFERRED_CONTACT="INFERRED_CONTACT"
    UNVERIFIED_CONTACT="UNVERIFIED_CONTACT"

@dataclass(frozen=True)
class EvidenceSample:
    frame:int
    raw_x:float|None
    raw_y:float|None
    tracked_x:float|None
    tracked_y:float|None
    confidence:float
    source:str
    missed_frames:int=0
    @property
    def observed(self): return self.raw_x is not None and self.raw_y is not None

@dataclass(frozen=True)
class EvidenceIntegrityResult:
    accepted:bool
    evidence:ContactEvidence
    event_frame:int
    contact_frame:int|None
    contact_x:float|None
    contact_y:float|None
    confidence:float
    before_observed_frame:int|None
    after_observed_frame:int|None
    reason:str

class BounceEvidenceIntegrity:
    """Kalman may guide search, but cannot alone certify ball-floor contact."""
    def __init__(self,window=5,max_inference_gap=6,min_observed_confidence=.10,min_inferred_confidence=.10):
        self.window=int(window); self.max_inference_gap=int(max_inference_gap)
        self.min_observed_confidence=float(min_observed_confidence)
        self.min_inferred_confidence=float(min_inferred_confidence)

    def evaluate(self,event_frame,samples):
        local=sorted([s for s in samples if abs(s.frame-event_frame)<=self.window],key=lambda s:s.frame)
        observed=[s for s in local if s.observed]
        exact=[s for s in observed if s.frame==event_frame]
        if exact:
            s=exact[0]; c=max(0.,min(1.,float(s.confidence)))
            if c < self.min_observed_confidence:
                return EvidenceIntegrityResult(False,ContactEvidence.UNVERIFIED_CONTACT,event_frame,s.frame,s.raw_x,s.raw_y,c,None,None,"OBSERVATION_CONFIDENCE_TOO_LOW")
            return EvidenceIntegrityResult(True,ContactEvidence.OBSERVED_CONTACT,event_frame,s.frame,float(s.raw_x),float(s.raw_y),c,None,None,"RAW_CONTACT_OBSERVED")
        before=[s for s in observed if s.frame<event_frame]
        after=[s for s in observed if s.frame>event_frame]
        if not before or not after:
            return EvidenceIntegrityResult(False,ContactEvidence.UNVERIFIED_CONTACT,event_frame,None,None,None,0.,before[-1].frame if before else None,after[0].frame if after else None,"CONTACT_NOT_BRACKETED_BY_OBSERVATIONS")
        b,a=before[-1],after[0]; gap=a.frame-b.frame
        if gap>self.max_inference_gap:
            return EvidenceIntegrityResult(False,ContactEvidence.UNVERIFIED_CONTACT,event_frame,None,None,None,0.,b.frame,a.frame,"OBSERVATION_GAP_TOO_LARGE")
        t=(event_frame-b.frame)/gap
        x=float(b.raw_x)+t*(float(a.raw_x)-float(b.raw_x))
        y=float(b.raw_y)+t*(float(a.raw_y)-float(b.raw_y))
        endpoint=min(float(b.confidence),float(a.confidence))
        gap_factor=max(0.,1.-(gap-2)/max(1.,self.max_inference_gap))
        c=max(0.,min(1.,endpoint*(.65+.35*gap_factor)))
        if c<self.min_inferred_confidence:
            return EvidenceIntegrityResult(False,ContactEvidence.UNVERIFIED_CONTACT,event_frame,event_frame,x,y,c,b.frame,a.frame,"INFERRED_CONFIDENCE_TOO_LOW")
        return EvidenceIntegrityResult(True,ContactEvidence.INFERRED_CONTACT,event_frame,event_frame,x,y,c,b.frame,a.frame,"CONTACT_BRACKETED_BY_RAW_OBSERVATIONS")
