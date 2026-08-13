from linecaller.rules.models import TeamSide
from linecaller.score import MatchFormat
from linecaller.score.match import MatchController


def finish_game(controller, winner):
    e=controller.score_engine
    # Directly exercise trusted score engine input until target is reached.
    while not e.state.game_over:
        if e.state.serving_team != winner:
            # receiver wins enough service rallies to obtain serve
            e.apply_rally_winner(winner)
            continue
        e.apply_rally_winner(winner)


def test_best_of_three_requires_two_games():
    m=MatchController(best_of=3,target_score=3,win_by=2)
    assert m.games_to_win==2


def test_cannot_commit_unfinished_game():
    m=MatchController()
    r=m.commit_finished_game()
    assert r.action=="IGNORED"
    assert r.reason=="CURRENT_GAME_NOT_FINISHED"


def test_finished_game_updates_match_games():
    m=MatchController(best_of=3,target_score=3,win_by=2)
    finish_game(m,TeamSide.A)
    r=m.commit_finished_game()
    assert r.action=="NEW_GAME"
    assert r.state.games_a==1
    assert r.state.games_b==0
    assert r.state.current_game==2


def test_new_game_resets_point_score():
    m=MatchController(best_of=3,target_score=3,win_by=2)
    finish_game(m,TeamSide.A)
    m.commit_finished_game()
    assert m.score_engine.state.score_a==0
    assert m.score_engine.state.score_b==0


def test_game_start_alternates_by_default():
    m=MatchController(
        best_of=3,
        target_score=3,
        win_by=2,
        starting_team=TeamSide.A,
    )
    finish_game(m,TeamSide.A)
    m.commit_finished_game()
    assert m.score_engine.state.serving_team==TeamSide.B


def test_second_game_win_can_finish_match():
    m=MatchController(best_of=3,target_score=3,win_by=2)
    finish_game(m,TeamSide.A)
    m.commit_finished_game()
    finish_game(m,TeamSide.A)
    r=m.commit_finished_game()
    assert r.action=="MATCH_WON"
    assert r.state.match_over is True
    assert r.state.match_winner==TeamSide.A
    assert m.match_score()=="2-0"


def test_split_games_requires_decider():
    m=MatchController(best_of=3,target_score=3,win_by=2)
    finish_game(m,TeamSide.A)
    m.commit_finished_game()
    finish_game(m,TeamSide.B)
    r=m.commit_finished_game()
    assert r.action=="NEW_GAME"
    assert r.state.current_game==3
    assert r.state.match_over is False
    assert m.match_score()=="1-1"


def test_deciding_game_can_finish_match():
    m=MatchController(best_of=3,target_score=3,win_by=2)
    finish_game(m,TeamSide.A); m.commit_finished_game()
    finish_game(m,TeamSide.B); m.commit_finished_game()
    finish_game(m,TeamSide.B)
    r=m.commit_finished_game()
    assert r.action=="MATCH_WON"
    assert r.state.match_winner==TeamSide.B
    assert m.match_score()=="1-2"


def test_match_over_rejects_extra_commit():
    m=MatchController(best_of=1,target_score=3,win_by=2)
    finish_game(m,TeamSide.A)
    m.commit_finished_game()
    r=m.commit_finished_game()
    assert r.reason=="MATCH_ALREADY_OVER"


def test_singles_match_uses_singles_score_engine():
    m=MatchController(
        best_of=3,
        match_format=MatchFormat.SINGLES,
        target_score=3,
    )
    assert m.score_engine.state.match_format==MatchFormat.SINGLES
    assert m.score_engine.score_call()=="0-0"
