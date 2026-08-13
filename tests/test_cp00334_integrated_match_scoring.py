from linecaller.rules.models import TeamSide
from linecaller.rules.rally_outcome import (
    RallyOutcome,
    RallyOutcomeResult,
)
from linecaller.score.integrated_match import IntegratedMatchScoringPipeline


def outcome(kind, winner=TeamSide.UNKNOWN, loser=TeamSide.UNKNOWN):
    return RallyOutcomeResult(
        outcome=kind,
        winner=winner,
        loser=loser,
        fault_team=loser,
        reason="TEST",
    )


def win(team):
    other=TeamSide.B if team==TeamSide.A else TeamSide.A
    return outcome(RallyOutcome.RALLY_WON,team,other)


def test_trusted_server_win_updates_score():
    p=IntegratedMatchScoringPipeline(target_score=3)
    r=p.apply_rally_outcome(win(TeamSide.A))
    assert r.score_changed is True
    assert r.announcer.score_a==1
    assert r.action=="POINT"


def test_review_is_recorded_but_does_not_change_score():
    p=IntegratedMatchScoringPipeline(target_score=3)
    before=p.match.score_engine.state
    r=p.apply_rally_outcome(outcome(RallyOutcome.REVIEW))
    assert r.score_changed is False
    assert p.match.score_engine.state==before
    assert r.action=="NO_CHANGE"
    assert len(p.history.entries)==1


def test_continue_is_no_change():
    p=IntegratedMatchScoringPipeline(target_score=3)
    r=p.apply_rally_outcome(outcome(RallyOutcome.RALLY_CONTINUES))
    assert r.action=="NO_CHANGE"
    assert r.score_changed is False


def test_sideout_flows_through_integrated_pipeline():
    p=IntegratedMatchScoringPipeline(target_score=3)
    r=p.apply_rally_outcome(win(TeamSide.B))
    assert r.action=="SIDE_OUT"
    assert r.announcer.serving_team==TeamSide.B


def test_game_win_auto_commits_new_game():
    p=IntegratedMatchScoringPipeline(best_of=3,target_score=3,win_by=2)
    for _ in range(2):
        p.apply_rally_outcome(win(TeamSide.A))
    r=p.apply_rally_outcome(win(TeamSide.A))
    assert r.action=="NEW_GAME"
    assert r.game_committed is True
    assert p.match.state.games_a==1
    assert p.match.state.current_game==2


def test_new_game_announcer_is_reset():
    p=IntegratedMatchScoringPipeline(best_of=3,target_score=3,win_by=2)
    for _ in range(3):
        r=p.apply_rally_outcome(win(TeamSide.A))
    assert r.announcer.score_a==0
    assert r.announcer.score_b==0
    assert r.announcer.game_number==2


def test_history_records_every_input_event():
    p=IntegratedMatchScoringPipeline(target_score=5)
    p.apply_rally_outcome(win(TeamSide.A))
    p.apply_rally_outcome(outcome(RallyOutcome.REVIEW))
    p.apply_rally_outcome(outcome(RallyOutcome.RALLY_CONTINUES))
    assert len(p.history.entries)==3
    assert [e.sequence for e in p.history.entries]==[1,2,3]


def test_best_of_one_can_finish_match():
    p=IntegratedMatchScoringPipeline(best_of=1,target_score=3,win_by=2)
    for _ in range(2):
        p.apply_rally_outcome(win(TeamSide.A))
    r=p.apply_rally_outcome(win(TeamSide.A))
    assert r.action=="MATCH_WON"
    assert r.match_over is True
    assert r.announcer.match_winner==TeamSide.A


def test_match_complete_announcer_text():
    p=IntegratedMatchScoringPipeline(best_of=1,target_score=3,win_by=2)
    for _ in range(3):
        r=p.apply_rally_outcome(win(TeamSide.A))
    assert "Match complete" in r.announcer.spoken_score()
    assert "Team A wins" in r.announcer.spoken_score()


def test_post_match_rally_cannot_modify_score():
    p=IntegratedMatchScoringPipeline(best_of=1,target_score=3,win_by=2)
    for _ in range(3):
        p.apply_rally_outcome(win(TeamSide.A))
    before=p.match.score_engine.state
    r=p.apply_rally_outcome(win(TeamSide.B))
    assert r.action=="NO_CHANGE"
    assert r.reason=="MATCH_ALREADY_OVER"
    assert p.match.score_engine.state==before
