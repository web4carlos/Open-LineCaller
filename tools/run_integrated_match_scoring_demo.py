from linecaller.rules.models import TeamSide
from linecaller.rules.rally_outcome import RallyOutcome, RallyOutcomeResult
from linecaller.score.integrated_match import IntegratedMatchScoringPipeline


def event(kind,winner=TeamSide.UNKNOWN,loser=TeamSide.UNKNOWN):
    return RallyOutcomeResult(
        outcome=kind,
        winner=winner,
        loser=loser,
        fault_team=loser,
        reason="DEMO",
    )


def win(team):
    other=TeamSide.B if team==TeamSide.A else TeamSide.A
    return event(RallyOutcome.RALLY_WON,team,other)


def show(label,p,r):
    a=r.announcer
    print(
        f"{label}: action={r.action} changed={r.score_changed} "
        f"games={a.games_a}-{a.games_b} game={a.game_number} "
        f"score={a.score_call} serve={a.serving_team.value} "
        f"history={r.history_entry.sequence} match_over={r.match_over}"
    )


def main():
    # Short best-of-1 game for a compact end-to-end demonstration.
    p=IntegratedMatchScoringPipeline(
        best_of=1,
        target_score=3,
        win_by=2,
    )

    show("A_point_1",p,p.apply_rally_outcome(win(TeamSide.A)))
    show("review",p,p.apply_rally_outcome(event(RallyOutcome.REVIEW)))
    show("A_point_2",p,p.apply_rally_outcome(win(TeamSide.A)))
    show("A_game_match_point",p,p.apply_rally_outcome(win(TeamSide.A)))
    show("post_match_event",p,p.apply_rally_outcome(win(TeamSide.B)))

    print(f"history_entries={len(p.history.entries)}")
    print(f"spoken={p._announcer('STATE','').spoken_score()}")


if __name__=="__main__":
    main()
