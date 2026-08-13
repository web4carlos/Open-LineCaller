from linecaller.bounce_v2.officiating_gate import (
    BounceOfficiatingGate,
    BounceGateDecision,
)

def gate():
    return BounceOfficiatingGate()

def test_observed_good_is_verified():
    r=gate().decide(frame=29,physical_valid=True,physical_score=.78,
        evidence_accepted=True,evidence_class="OBSERVED_CONTACT",evidence_confidence=.24)
    assert r.decision==BounceGateDecision.VERIFIED_BOUNCE

def test_inferred_low_is_review():
    r=gate().decide(frame=94,physical_valid=True,physical_score=.79,
        evidence_accepted=True,evidence_class="INFERRED_CONTACT",evidence_confidence=.1024)
    assert r.decision==BounceGateDecision.REVIEW_BOUNCE

def test_inferred_017_is_review():
    r=gate().decide(frame=157,physical_valid=True,physical_score=.7465,
        evidence_accepted=True,evidence_class="INFERRED_CONTACT",evidence_confidence=.1701)
    assert r.decision==BounceGateDecision.REVIEW_BOUNCE

def test_physical_failure_is_rejected():
    r=gate().decide(frame=260,physical_valid=False,physical_score=.442,
        evidence_accepted=True,evidence_class="OBSERVED_CONTACT",evidence_confidence=.2126)
    assert r.decision==BounceGateDecision.REJECTED_BOUNCE

def test_unverified_is_rejected():
    r=gate().decide(frame=1,physical_valid=True,physical_score=.8,
        evidence_accepted=False,evidence_class="UNVERIFIED_CONTACT",evidence_confidence=0)
    assert r.decision==BounceGateDecision.REJECTED_BOUNCE
