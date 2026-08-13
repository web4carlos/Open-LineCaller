from linecaller.rules.models import TeamSide
from linecaller.score import SideOutScoreEngine
from linecaller.score.announcer import build_announcer_state
from linecaller.score.history import ScoreHistory


def snapshot(label,e,h,action,reason):
    entry=h.record(
        game_number=1,
        event=label,
        action=action,
        reason=reason,
        score_state=e.state,
        games_a=0,
        games_b=0,
    )
    a=build_announcer_state(
        score_state=e.state,
        game_number=1,
        games_a=0,
        games_b=0,
        last_action=action,
        last_reason=reason,
    )
    print(
        f"{label}: seq={entry.sequence} score={a.score_call} "
        f"serve={a.serving_team.value} server={a.server_number} "
        f"court={a.server_court} action={a.last_action}"
    )
    print(f"  spoken={a.spoken_score()}")


def main():
    e=SideOutScoreEngine()
    h=ScoreHistory()

    snapshot("start",e,h,"START","MATCH_STARTED")

    u=e.apply_rally_winner(TeamSide.A)
    snapshot("A_point",e,h,u.action,u.reason)

    snapshot(
        "review",
        e,h,
        "NO_CHANGE",
        "REVIEW_CANNOT_CHANGE_SCORE",
    )

    u=e.apply_rally_winner(TeamSide.B)
    snapshot("side_out_to_B",e,h,u.action,u.reason)

    print(f"history_entries={len(h.entries)}")


if __name__=="__main__":
    main()
