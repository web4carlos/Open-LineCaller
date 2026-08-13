from linecaller.rules.nvz import (
    NVZContext,
    NVZRuleEvaluator,
    NVZRuling,
)


def ev(**kwargs):
    return NVZRuleEvaluator().evaluate(NVZContext(**kwargs))


def test_groundstroke_inside_nvz_is_legal():
    r=ev(volley=False,player_in_nvz=True)
    assert r.ruling==NVZRuling.LEGAL


def test_groundstroke_on_nvz_boundary_is_not_volley_fault():
    r=ev(volley=False,player_in_nvz=True,entered_nvz_after_volley=False)
    assert r.reason=="BOUNCED_BALL_MAY_BE_PLAYED_FROM_NVZ"


def test_volley_while_in_nvz_is_fault():
    r=ev(volley=True,player_in_nvz=True)
    assert r.ruling==NVZRuling.NVZ_FAULT


def test_legal_volley_outside_nvz():
    r=ev(volley=True,player_in_nvz=False,momentum_complete=True)
    assert r.ruling==NVZRuling.LEGAL


def test_momentum_into_nvz_is_fault():
    r=ev(
        volley=True,
        player_in_nvz=False,
        entered_nvz_after_volley=True,
        momentum_complete=False,
    )
    assert r.ruling==NVZRuling.NVZ_FAULT


def test_momentum_fault_even_if_ball_later_dead():
    r=ev(
        volley=True,
        player_in_nvz=False,
        entered_nvz_after_volley=True,
        momentum_complete=True,
    )
    assert r.reason=="VOLLEY_MOMENTUM_ENTERED_NVZ"


def test_incomplete_momentum_evidence_requires_review():
    r=ev(
        volley=True,
        player_in_nvz=False,
        entered_nvz_after_volley=False,
        momentum_complete=False,
    )
    assert r.ruling==NVZRuling.REVIEW


def test_upstream_review_propagates():
    r=ev(
        volley=False,
        player_in_nvz=False,
        upstream_decision="REVIEW",
    )
    assert r.ruling==NVZRuling.REVIEW


def test_upstream_review_wins_over_apparent_fault():
    r=ev(
        volley=True,
        player_in_nvz=True,
        upstream_decision="REVIEW",
    )
    assert r.ruling==NVZRuling.REVIEW


def test_player_can_enter_nvz_without_volley():
    r=ev(
        volley=False,
        player_in_nvz=True,
        entered_nvz_after_volley=True,
    )
    assert r.ruling==NVZRuling.LEGAL
