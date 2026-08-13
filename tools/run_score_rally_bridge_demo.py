from linecaller.rules.models import TeamSide
from linecaller.rules.rally_outcome import (
    RallyOutcome,
    RallyOutcomeResult,
)
from linecaller.score import SideOutScoreEngine
from linecaller.score.rally_bridge import RallyOutcomeScoreBridge


def outcome(kind, winner=TeamSide.UNKNOWN, loser=TeamSide.UNKNOWN):
    return RallyOutcomeResult(
        outcome=kind,
        winner=winner,
        loser=loser,
        fault_team=loser,
        reason="DEMO",
    )


def show(label, engine, bridge_result):
    s=engine.state
    print(
        f"{label}: action={bridge_result.action} "
        f"changed={bridge_result.score_changed} "
        f"score={engine.score_call()} "
        f"A={s.score_a} B={s.score_b} "
        f"serve={s.serving_team.value} "
        f"server={s.server_number} "
        f"reason={bridge_result.reason}"
    )


def main():
    e=SideOutScoreEngine()
    b=RallyOutcomeScoreBridge(e)

    show(
        "A_trusted_win",
        e,
        b.apply(outcome(RallyOutcome.RALLY_WON, TeamSide.A, TeamSide.B)),
    )

    show(
        "review_event",
        e,
        b.apply(outcome(RallyOutcome.REVIEW)),
    )

    show(
        "B_receiver_win",
        e,
        b.apply(outcome(RallyOutcome.RALLY_WON, TeamSide.B, TeamSide.A)),
    )

    show(
        "A_beats_B_server1",
        e,
        b.apply(outcome(RallyOutcome.RALLY_WON, TeamSide.A, TeamSide.B)),
    )

    show(
        "A_beats_B_server2",
        e,
        b.apply(outcome(RallyOutcome.RALLY_WON, TeamSide.A, TeamSide.B)),
    )


if __name__=="__main__":
    main()
