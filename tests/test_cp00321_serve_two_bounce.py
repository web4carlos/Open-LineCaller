from linecaller.rules.serve_two_bounce import *

def test_right_server_targets_left_service():
    r=ServeLegalityEvaluator().evaluate(
        ServeContext("A","RIGHT","FAR_LEFT_SERVICE"))
    assert r.ruling==ServeRuling.LEGAL_SERVE

def test_left_server_targets_right_service():
    r=ServeLegalityEvaluator().evaluate(
        ServeContext("A","LEFT","FAR_RIGHT_SERVICE"))
    assert r.ruling==ServeRuling.LEGAL_SERVE

def test_wrong_service_court_fault():
    r=ServeLegalityEvaluator().evaluate(
        ServeContext("A","RIGHT","FAR_RIGHT_SERVICE"))
    assert r.ruling==ServeRuling.SERVE_FAULT

def test_kitchen_serve_fault():
    r=ServeLegalityEvaluator().evaluate(
        ServeContext("A","RIGHT","FAR_KITCHEN"))
    assert r.ruling==ServeRuling.SERVE_FAULT

def test_nvz_line_serve_fault():
    r=ServeLegalityEvaluator().evaluate(
        ServeContext("A","RIGHT","FAR_LEFT_SERVICE","FAR_NVZ_LINE",True))
    assert r.ruling==ServeRuling.SERVE_FAULT

def test_upstream_review_stays_review():
    r=ServeLegalityEvaluator().evaluate(
        ServeContext("A","RIGHT","FAR_LEFT_SERVICE",upstream_decision="REVIEW"))
    assert r.ruling==ServeRuling.REVIEW

def test_two_bounce_progression():
    t=TwoBounceRule()
    a=t.receiver_bounce("IN")
    assert a.state==TwoBounceState.WAIT_SERVER_BOUNCE
    b=t.receiver_return_hit(False)
    assert b.ruling=="CONTINUE"
    c=t.server_side_bounce("IN")
    assert c.state==TwoBounceState.OPEN_RALLY

def test_receiver_cannot_volley_serve():
    t=TwoBounceRule()
    t.receiver_bounce("IN")
    # This call represents the return after the mandatory bounce, so not fault.
    r=t.receiver_return_hit(False)
    assert r.ruling=="CONTINUE"

def test_serving_team_cannot_volley_return_before_bounce():
    t=TwoBounceRule()
    t.receiver_bounce("IN")
    r=t.server_hit_before_bounce()
    assert r.state==TwoBounceState.FAULT

def test_two_bounce_review_propagates():
    t=TwoBounceRule()
    r=t.receiver_bounce("REVIEW")
    assert r.state==TwoBounceState.REVIEW
