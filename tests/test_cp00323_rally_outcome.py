from linecaller.rules.models import TeamSide
from linecaller.rules.rally_outcome import (
    FaultType,
    RallyOutcome,
    RallyOutcomeContext,
    RallyOutcomeEngine,
)


def run(**kwargs):
    return RallyOutcomeEngine().evaluate(RallyOutcomeContext(**kwargs))


def test_out_by_a_awards_rally_to_b():
    r=run(acting_team=TeamSide.A,opponent_team=TeamSide.B,decision="OUT")
    assert r.outcome==RallyOutcome.RALLY_WON
    assert r.winner==TeamSide.B
    assert r.loser==TeamSide.A


def test_out_by_b_awards_rally_to_a():
    r=run(acting_team=TeamSide.B,opponent_team=TeamSide.A,decision="OUT")
    assert r.winner==TeamSide.A


def test_in_continues_rally():
    r=run(acting_team=TeamSide.A,opponent_team=TeamSide.B,decision="IN")
    assert r.outcome==RallyOutcome.RALLY_CONTINUES


def test_serve_fault_owner_loses_rally():
    r=run(
        acting_team=TeamSide.A,
        opponent_team=TeamSide.B,
        fault_type=FaultType.SERVE_FAULT,
        fault_team=TeamSide.A,
    )
    assert r.winner==TeamSide.B
    assert r.fault_team==TeamSide.A


def test_nvz_fault_owner_loses_rally():
    r=run(
        acting_team=TeamSide.A,
        opponent_team=TeamSide.B,
        fault_type=FaultType.NVZ_FAULT,
        fault_team=TeamSide.B,
    )
    assert r.winner==TeamSide.A


def test_two_bounce_fault_owner_loses_rally():
    r=run(
        acting_team=TeamSide.A,
        opponent_team=TeamSide.B,
        fault_type=FaultType.TWO_BOUNCE_FAULT,
        fault_team=TeamSide.A,
    )
    assert r.winner==TeamSide.B


def test_unknown_fault_team_requires_review():
    r=run(
        acting_team=TeamSide.A,
        opponent_team=TeamSide.B,
        fault_type=FaultType.OTHER_FAULT,
    )
    assert r.outcome==RallyOutcome.REVIEW


def test_review_decision_never_produces_winner():
    r=run(
        acting_team=TeamSide.A,
        opponent_team=TeamSide.B,
        decision="REVIEW",
    )
    assert r.outcome==RallyOutcome.REVIEW
    assert r.winner==TeamSide.UNKNOWN


def test_upstream_review_overrides_apparent_fault():
    r=run(
        acting_team=TeamSide.A,
        opponent_team=TeamSide.B,
        fault_type=FaultType.NVZ_FAULT,
        fault_team=TeamSide.A,
        upstream_review=True,
    )
    assert r.outcome==RallyOutcome.REVIEW


def test_unknown_decision_fails_safe():
    r=run(
        acting_team=TeamSide.A,
        opponent_team=TeamSide.B,
        decision="MAYBE",
    )
    assert r.outcome==RallyOutcome.REVIEW
