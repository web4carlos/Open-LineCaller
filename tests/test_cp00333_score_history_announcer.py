from linecaller.rules.models import TeamSide
from linecaller.score import MatchFormat, SideOutScoreEngine
from linecaller.score.announcer import build_announcer_state
from linecaller.score.history import ScoreHistory


def test_history_starts_empty():
    h=ScoreHistory()
    assert h.entries==()
    assert h.last is None


def test_history_records_snapshot():
    e=SideOutScoreEngine()
    h=ScoreHistory()
    x=h.record(
        game_number=1,event="RALLY",action="START",reason="TEST",
        score_state=e.state,games_a=0,games_b=0,
    )
    assert x.sequence==1
    assert x.score_a==0
    assert x.serving_team==TeamSide.A


def test_history_is_append_only_sequence():
    e=SideOutScoreEngine()
    h=ScoreHistory()
    h.record(game_number=1,event="START",action="START",reason="",
             score_state=e.state,games_a=0,games_b=0)
    e.apply_rally_winner(TeamSide.A)
    h.record(game_number=1,event="RALLY",action="POINT",reason="",
             score_state=e.state,games_a=0,games_b=0)
    assert [x.sequence for x in h.entries]==[1,2]
    assert h.entries[0].score_a==0
    assert h.entries[1].score_a==1


def test_announcer_opening_call_is_zero_zero_two():
    e=SideOutScoreEngine()
    a=build_announcer_state(
        score_state=e.state,game_number=1,games_a=0,games_b=0
    )
    assert a.score_call=="0-0-2"


def test_announcer_uses_serving_team_perspective():
    e=SideOutScoreEngine()
    e.apply_rally_winner(TeamSide.A)
    e.apply_rally_winner(TeamSide.B)
    a=build_announcer_state(
        score_state=e.state,game_number=1,games_a=0,games_b=0
    )
    assert a.serving_team==TeamSide.B
    assert a.score_call=="0-1-1"


def test_announcer_tracks_server_court():
    e=SideOutScoreEngine()
    e.apply_rally_winner(TeamSide.A)
    a=build_announcer_state(
        score_state=e.state,game_number=1,games_a=0,games_b=0
    )
    assert a.server_court=="LEFT"


def test_spoken_score_contains_serving_team():
    e=SideOutScoreEngine()
    a=build_announcer_state(
        score_state=e.state,game_number=1,games_a=0,games_b=0
    )
    assert "Team A serving" in a.spoken_score()


def test_singles_announcer_omits_server_number():
    e=SideOutScoreEngine(match_format=MatchFormat.SINGLES)
    a=build_announcer_state(
        score_state=e.state,game_number=1,games_a=0,games_b=0
    )
    assert a.score_call=="0-0"


def test_match_complete_spoken_state():
    e=SideOutScoreEngine()
    a=build_announcer_state(
        score_state=e.state,game_number=3,games_a=2,games_b=1,
        match_over=True,match_winner=TeamSide.A,
    )
    assert a.spoken_score()=="Match complete. Team A wins. Games 2 to 1."


def test_history_can_capture_review_without_score_change():
    e=SideOutScoreEngine()
    h=ScoreHistory()
    before=e.state
    x=h.record(
        game_number=1,event="REVIEW",action="NO_CHANGE",
        reason="REVIEW_CANNOT_CHANGE_SCORE",
        score_state=e.state,games_a=0,games_b=0,
    )
    assert x.score_a==before.score_a
    assert x.score_b==before.score_b
    assert x.action=="NO_CHANGE"
