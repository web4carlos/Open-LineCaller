from linecaller.rules.integrated_pipeline import (
    IntegratedPickleballRulesPipeline,
    IntegratedRulesStatus,
)
from linecaller.rules.models import TeamSide


def test_legal_serve_continues():
    p=IntegratedPickleballRulesPipeline()
    r=p.evaluate_serve(
        serving_team=TeamSide.A,
        receiving_team=TeamSide.B,
        server_side="RIGHT",
        receiving_zone="FAR_LEFT_SERVICE",
        decision="IN",
    )
    assert r.status==IntegratedRulesStatus.CONTINUE


def test_wrong_service_court_awards_rally_to_receiver():
    p=IntegratedPickleballRulesPipeline()
    r=p.evaluate_serve(
        serving_team=TeamSide.A,
        receiving_team=TeamSide.B,
        server_side="RIGHT",
        receiving_zone="FAR_RIGHT_SERVICE",
        decision="IN",
    )
    assert r.status==IntegratedRulesStatus.RALLY_WON
    assert r.winner==TeamSide.B


def test_serve_review_has_no_winner():
    p=IntegratedPickleballRulesPipeline()
    r=p.evaluate_serve(
        serving_team=TeamSide.A,
        receiving_team=TeamSide.B,
        server_side="RIGHT",
        receiving_zone="FAR_LEFT_SERVICE",
        decision="REVIEW",
    )
    assert r.status==IntegratedRulesStatus.REVIEW
    assert r.winner==TeamSide.UNKNOWN


def test_valid_two_bounce_sequence_reaches_continue():
    p=IntegratedPickleballRulesPipeline()
    p.evaluate_serve(
        serving_team=TeamSide.A,
        receiving_team=TeamSide.B,
        server_side="RIGHT",
        receiving_zone="FAR_LEFT_SERVICE",
        decision="IN",
    )
    a=p.evaluate_return(
        receiving_team=TeamSide.B,
        serving_team=TeamSide.A,
        return_volley=False,
    )
    b=p.evaluate_server_side_bounce(
        serving_team=TeamSide.A,
        receiving_team=TeamSide.B,
        decision="IN",
    )
    assert a.status==IntegratedRulesStatus.CONTINUE
    assert b.status==IntegratedRulesStatus.CONTINUE


def test_server_volley_before_required_bounce_loses_rally():
    p=IntegratedPickleballRulesPipeline()
    p.evaluate_serve(
        serving_team=TeamSide.A,
        receiving_team=TeamSide.B,
        server_side="RIGHT",
        receiving_zone="FAR_LEFT_SERVICE",
        decision="IN",
    )
    r=p.evaluate_server_volley_before_required_bounce(
        serving_team=TeamSide.A,
        receiving_team=TeamSide.B,
    )
    assert r.status==IntegratedRulesStatus.RALLY_WON
    assert r.winner==TeamSide.B


def test_nvz_volley_fault_awards_rally_to_opponent():
    p=IntegratedPickleballRulesPipeline()
    r=p.evaluate_nvz_contact(
        acting_team=TeamSide.A,
        opponent_team=TeamSide.B,
        volley=True,
        player_in_nvz=True,
    )
    assert r.status==IntegratedRulesStatus.RALLY_WON
    assert r.winner==TeamSide.B


def test_nvz_groundstroke_is_legal_continue():
    p=IntegratedPickleballRulesPipeline()
    r=p.evaluate_nvz_contact(
        acting_team=TeamSide.A,
        opponent_team=TeamSide.B,
        volley=False,
        player_in_nvz=True,
    )
    assert r.status==IntegratedRulesStatus.CONTINUE


def test_uncertain_momentum_propagates_review():
    p=IntegratedPickleballRulesPipeline()
    r=p.evaluate_nvz_contact(
        acting_team=TeamSide.A,
        opponent_team=TeamSide.B,
        volley=True,
        momentum_complete=False,
    )
    assert r.status==IntegratedRulesStatus.REVIEW
    assert r.winner==TeamSide.UNKNOWN


def test_out_awards_rally_to_opponent():
    p=IntegratedPickleballRulesPipeline()
    r=p.evaluate_ball_decision(
        acting_team=TeamSide.A,
        opponent_team=TeamSide.B,
        decision="OUT",
    )
    assert r.status==IntegratedRulesStatus.RALLY_WON
    assert r.winner==TeamSide.B


def test_review_ball_decision_never_awards_rally():
    p=IntegratedPickleballRulesPipeline()
    r=p.evaluate_ball_decision(
        acting_team=TeamSide.A,
        opponent_team=TeamSide.B,
        decision="REVIEW",
    )
    assert r.status==IntegratedRulesStatus.REVIEW
    assert r.winner==TeamSide.UNKNOWN
