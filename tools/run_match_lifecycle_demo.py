from linecaller.rules.models import TeamSide
from linecaller.score.match import MatchController


def win_game(m, team):
    while not m.score_engine.state.game_over:
        m.score_engine.apply_rally_winner(team)


def show(label, m, update=None):
    s=m.state
    action=update.action if update else "START"
    print(
        f"{label}: games={m.match_score()} "
        f"game={s.current_game} "
        f"point_score={m.score_engine.score_call()} "
        f"start_team={s.starting_team.value} "
        f"action={action} "
        f"match_over={s.match_over} "
        f"winner={s.match_winner.value}"
    )


def main():
    m=MatchController(best_of=3,target_score=3,win_by=2)

    show("match_start",m)

    win_game(m,TeamSide.A)
    show("game1_finished",m)
    show("game1_committed",m,m.commit_finished_game())

    win_game(m,TeamSide.B)
    show("game2_finished",m)
    show("game2_committed",m,m.commit_finished_game())

    win_game(m,TeamSide.A)
    show("game3_finished",m)
    show("match_finished",m,m.commit_finished_game())


if __name__=="__main__":
    main()
