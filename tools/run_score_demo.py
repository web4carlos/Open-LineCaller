from linecaller.rules.models import TeamSide
from linecaller.score import MatchFormat, SideOutScoreEngine


def show(label, engine, update=None):
    s=engine.state
    action=update.action if update else "START"
    print(
        f"{label}: score={engine.score_call()} "
        f"A={s.score_a} B={s.score_b} "
        f"serve={s.serving_team.value} "
        f"server={s.server_number} "
        f"court={s.server_court} "
        f"action={action} "
        f"game_over={s.game_over}"
    )


def main():
    e=SideOutScoreEngine(
        match_format=MatchFormat.DOUBLES,
        starting_team=TeamSide.A,
    )

    show("start",e)

    show("A_wins_rally",e,e.apply_rally_winner(TeamSide.A))
    show("B_wins_rally",e,e.apply_rally_winner(TeamSide.B))
    show("A_wins_vs_B_server1",e,e.apply_rally_winner(TeamSide.A))
    show("A_wins_vs_B_server2",e,e.apply_rally_winner(TeamSide.A))
    show("A_scores_again",e,e.apply_rally_winner(TeamSide.A))


if __name__=="__main__":
    main()
