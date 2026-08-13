from linecaller.rules.models import TeamSide
from linecaller.rules.rally_outcome import (
    FaultType,
    RallyOutcomeContext,
    RallyOutcomeEngine,
)


def show(name, **kwargs):
    r = RallyOutcomeEngine().evaluate(RallyOutcomeContext(**kwargs))
    print(
        f"{name}: outcome={r.outcome.value} "
        f"winner={r.winner.value} loser={r.loser.value} "
        f"fault_team={r.fault_team.value} reason={r.reason}"
    )


def main():
    show(
        "ball_out_by_A",
        acting_team=TeamSide.A,
        opponent_team=TeamSide.B,
        decision="OUT",
    )
    show(
        "nvz_fault_by_B",
        acting_team=TeamSide.A,
        opponent_team=TeamSide.B,
        fault_type=FaultType.NVZ_FAULT,
        fault_team=TeamSide.B,
    )
    show(
        "legal_in_ball",
        acting_team=TeamSide.A,
        opponent_team=TeamSide.B,
        decision="IN",
    )
    show(
        "review_event",
        acting_team=TeamSide.A,
        opponent_team=TeamSide.B,
        decision="REVIEW",
    )


if __name__ == "__main__":
    main()
