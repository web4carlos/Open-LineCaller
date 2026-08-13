from linecaller.decision.officiating_enforcer import ForceReviewEnforcer

def test_verified_preserves_out():
    r=ForceReviewEnforcer().enforce(
        frame=29,base_decision="OUT",gate_decision="VERIFIED_BOUNCE",
        gate_confidence=.56,force_review=False)
    assert r.final_decision=="OUT"

def test_verified_preserves_in():
    r=ForceReviewEnforcer().enforce(
        frame=1,base_decision="IN",gate_decision="VERIFIED_BOUNCE",
        gate_confidence=.9,force_review=False)
    assert r.final_decision=="IN"

def test_review_gate_forces_review():
    r=ForceReviewEnforcer().enforce(
        frame=94,base_decision="IN",gate_decision="REVIEW_BOUNCE",
        gate_confidence=.51,force_review=False)
    assert r.final_decision=="REVIEW"

def test_force_review_flag_wins():
    r=ForceReviewEnforcer().enforce(
        frame=157,base_decision="OUT",gate_decision="VERIFIED_BOUNCE",
        gate_confidence=.8,force_review=True)
    assert r.final_decision=="REVIEW"

def test_unknown_gate_fails_safe():
    r=ForceReviewEnforcer().enforce(
        frame=5,base_decision="OUT",gate_decision="UNKNOWN",
        gate_confidence=.9)
    assert r.final_decision=="REVIEW"
