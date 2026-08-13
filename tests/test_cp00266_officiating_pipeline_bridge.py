from linecaller.bounce_v2.pipeline_bridge import (
    OfficiatingPipelineBridge,
    BridgeAction,
)

def test_verified_routes_to_auto_officiate():
    r=OfficiatingPipelineBridge().route(
        frame=29,gate_decision="VERIFIED_BOUNCE",confidence=.56)
    assert r.action==BridgeAction.AUTO_OFFICIATE
    assert r.force_review is False

def test_review_routes_to_geometry_but_forces_review():
    r=OfficiatingPipelineBridge().route(
        frame=94,gate_decision="REVIEW_BOUNCE",confidence=.51)
    assert r.action==BridgeAction.GEOMETRY_REVIEW_ONLY
    assert r.force_review is True

def test_rejected_is_dropped():
    r=OfficiatingPipelineBridge().route(
        frame=260,gate_decision="REJECTED_BOUNCE",confidence=0)
    assert r.action==BridgeAction.DROP_EVENT

def test_unknown_gate_decision_fails_closed():
    r=OfficiatingPipelineBridge().route(
        frame=1,gate_decision="SOMETHING_NEW",confidence=.9)
    assert r.action==BridgeAction.DROP_EVENT
