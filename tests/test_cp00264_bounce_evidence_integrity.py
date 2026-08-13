from linecaller.bounce_v2.evidence_integrity import *
def s(f,raw=True,x=100,y=100,c=.8):
    return EvidenceSample(f,x if raw else None,y if raw else None,x,y,c,"YOLO+KALMAN" if raw else "KALMAN-PREDICT")
def test_observed():
    r=BounceEvidenceIntegrity().evaluate(10,[s(9),s(10),s(11)]); assert r.accepted and r.evidence==ContactEvidence.OBSERVED_CONTACT
def test_kalman_not_observed():
    r=BounceEvidenceIntegrity().evaluate(10,[s(9),s(10,False),s(11)]); assert r.evidence!=ContactEvidence.OBSERVED_CONTACT
def test_bracketed_inferred():
    r=BounceEvidenceIntegrity().evaluate(10,[s(9,x=90),s(10,False),s(11,x=110)]); assert r.accepted and r.evidence==ContactEvidence.INFERRED_CONTACT
def test_unverified():
    r=BounceEvidenceIntegrity().evaluate(10,[s(9,False),s(10,False),s(11,False)]); assert not r.accepted
def test_gap():
    r=BounceEvidenceIntegrity(window=10,max_inference_gap=4).evaluate(10,[s(7),s(13)]); assert not r.accepted and r.reason=="OBSERVATION_GAP_TOO_LARGE"
