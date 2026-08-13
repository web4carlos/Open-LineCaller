from linecaller.rules.models import TeamSide
from linecaller.rules.rally_outcome import (
    RallyOutcome,
    RallyOutcomeResult,
)
from linecaller.score import SideOutScoreEngine
from linecaller.score.rally_bridge import RallyOutcomeScoreBridge


def result(outcome, winner=TeamSide.UNKNOWN, loser=TeamSide.UNKNOWN, reason="TEST"):
    return RallyOutcomeResult(
        outcome=outcome,
        winner=winner,
        loser=loser,
        fault_team=loser,
        reason=reason,
    )


def test_serving_team_rally_win_scores_point():
    e=SideOutScoreEngine()
    b=RallyOutcomeScoreBridge(e)
    r=b.apply(result(RallyOutcome.RALLY_WON, TeamSide.A, TeamSide.B))
    assert r.action=="POINT"
    assert r.state.score_a==1
    assert r.score_changed is True


def test_receiver_rally_win_causes_opening_sideout_not_point():
    e=SideOutScoreEngine()
    b=RallyOutcomeScoreBridge(e)
    r=b.apply(result(RallyOutcome.RALLY_WON, TeamSide.B, TeamSide.A))
    assert r.action=="SIDE_OUT"
    assert r.state.score_b==0
    assert r.state.serving_team==TeamSide.B


def test_review_never_changes_score():
    e=SideOutScoreEngine()
    b=RallyOutcomeScoreBridge(e)
    before=e.state
    r=b.apply(result(RallyOutcome.REVIEW))
    assert r.action=="NO_CHANGE"
    assert r.state==before
    assert e.state==before
    assert r.score_changed is False


def test_continue_never_changes_score():
    e=SideOutScoreEngine()
    b=RallyOutcomeScoreBridge(e)
    before=e.state
    r=b.apply(result(RallyOutcome.RALLY_CONTINUES))
    assert r.state==before
    assert r.score_changed is False


def test_unknown_winner_fails_safe():
    e=SideOutScoreEngine()
    b=RallyOutcomeScoreBridge(e)
    before=e.state
    r=b.apply(result(RallyOutcome.RALLY_WON))
    assert r.reason=="RALLY_WINNER_UNKNOWN"
    assert e.state==before


def test_second_server_transition_through_bridge():
    e=SideOutScoreEngine()
    b=RallyOutcomeScoreBridge(e)
    b.apply(result(RallyOutcome.RALLY_WON, TeamSide.B, TeamSide.A))
    r=b.apply(result(RallyOutcome.RALLY_WON, TeamSide.A, TeamSide.B))
    assert r.action=="SECOND_SERVER"
    assert r.state.serving_team==TeamSide.B
    assert r.state.server_number==2


def test_second_server_loss_sideout_through_bridge():
    e=SideOutScoreEngine()
    b=RallyOutcomeScoreBridge(e)
    b.apply(result(RallyOutcome.RALLY_WON, TeamSide.B, TeamSide.A))
    b.apply(result(RallyOutcome.RALLY_WON, TeamSide.A, TeamSide.B))
    r=b.apply(result(RallyOutcome.RALLY_WON, TeamSide.A, TeamSide.B))
    assert r.action=="SIDE_OUT"
    assert r.state.serving_team==TeamSide.A


def test_score_call_updates_after_trusted_rally():
    e=SideOutScoreEngine()
    b=RallyOutcomeScoreBridge(e)
    b.apply(result(RallyOutcome.RALLY_WON, TeamSide.A, TeamSide.B))
    assert e.score_call()=="1-0-2"


def test_review_after_point_does_not_rollback_or_advance():
    e=SideOutScoreEngine()
    b=RallyOutcomeScoreBridge(e)
    b.apply(result(RallyOutcome.RALLY_WON, TeamSide.A, TeamSide.B))
    before=e.state
    r=b.apply(result(RallyOutcome.REVIEW))
    assert r.state==before
    assert e.score_call()=="1-0-2"


def test_bridge_can_reach_game_winner():
    e=SideOutScoreEngine(target_score=3, win_by=2)
    b=RallyOutcomeScoreBridge(e)
    for _ in range(3):
        b.apply(result(RallyOutcome.RALLY_WON, TeamSide.A, TeamSide.B))
    assert e.state.game_over is True
    assert e.state.winner==TeamSide.A
