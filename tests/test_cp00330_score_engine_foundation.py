from linecaller.rules.models import TeamSide
from linecaller.score import MatchFormat, SideOutScoreEngine


def test_doubles_starts_zero_zero_two():
    e=SideOutScoreEngine(match_format=MatchFormat.DOUBLES)
    assert e.score_call()=="0-0-2"


def test_server_wins_point():
    e=SideOutScoreEngine()
    r=e.apply_rally_winner(TeamSide.A)
    assert r.action=="POINT"
    assert r.state.score_a==1
    assert r.state.serving_team==TeamSide.A


def test_opening_server_loss_is_sideout():
    e=SideOutScoreEngine()
    r=e.apply_rally_winner(TeamSide.B)
    assert r.action=="SIDE_OUT"
    assert r.state.serving_team==TeamSide.B
    assert r.state.server_number==1


def test_after_sideout_first_server_loss_goes_second_server():
    e=SideOutScoreEngine()
    e.apply_rally_winner(TeamSide.B)  # opening side-out to B
    r=e.apply_rally_winner(TeamSide.A)
    assert r.action=="SECOND_SERVER"
    assert r.state.serving_team==TeamSide.B
    assert r.state.server_number==2


def test_second_server_loss_causes_sideout():
    e=SideOutScoreEngine()
    e.apply_rally_winner(TeamSide.B)
    e.apply_rally_winner(TeamSide.A)
    r=e.apply_rally_winner(TeamSide.A)
    assert r.action=="SIDE_OUT"
    assert r.state.serving_team==TeamSide.A
    assert r.state.server_number==1


def test_receiver_does_not_score_in_sideout():
    e=SideOutScoreEngine()
    e.apply_rally_winner(TeamSide.B)
    assert e.state.score_b==0


def test_server_court_even_right_odd_left():
    e=SideOutScoreEngine()
    assert e.state.server_court=="RIGHT"
    e.apply_rally_winner(TeamSide.A)
    assert e.state.server_court=="LEFT"


def test_singles_receiver_win_immediate_sideout():
    e=SideOutScoreEngine(match_format=MatchFormat.SINGLES)
    r=e.apply_rally_winner(TeamSide.B)
    assert r.action=="SIDE_OUT"
    assert r.state.serving_team==TeamSide.B
    assert e.score_call()=="0-0"


def test_game_to_11_win_by_two():
    e=SideOutScoreEngine(target_score=11,win_by=2)
    for _ in range(11):
        e.apply_rally_winner(TeamSide.A)
    assert e.state.game_over is True
    assert e.state.winner==TeamSide.A


def test_game_not_over_at_11_10():
    e=SideOutScoreEngine(target_score=11,win_by=2)
    # Build A 10-0 while serving.
    for _ in range(10):
        e.apply_rally_winner(TeamSide.A)

    # Force a side-out cycle so B can score 10.
    e.apply_rally_winner(TeamSide.B)  # opening side-out to B
    for _ in range(10):
        e.apply_rally_winner(TeamSide.B)

    # Get serve back to A through B server 1/2.
    e.apply_rally_winner(TeamSide.A)  # B server1 -> server2
    e.apply_rally_winner(TeamSide.A)  # side-out to A

    e.apply_rally_winner(TeamSide.A)  # A reaches 11-10

    assert e.state.score_a==11
    assert e.state.score_b==10
    assert e.state.game_over is False
