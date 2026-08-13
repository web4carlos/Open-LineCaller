from linecaller.decision import DecisionEngine


def engine():
    return DecisionEngine(
        min_bounce_confidence=0.35,
        min_bounce_score=0.40,
        review_band_in=3.0,
        ball_contact_radius_in=1.45,
    )


def decide(distance_in, conf=0.8, score=0.8, state="INSIDE"):
    return engine().decide(
        geometry_state=state,
        nearest_line="LEFT_SIDELINE",
        signed_distance_ft=distance_in / 12.0,
        bounce_confidence=conf,
        bounce_score=score,
    )


def test_clear_inside_is_in():
    assert decide(10).call == "IN"


def test_clear_outside_is_out():
    assert decide(-5, state="OUTSIDE").call == "OUT"


def test_ball_touching_line_is_in():
    r = decide(-1.0, state="NEAR_LINE")
    assert r.call == "IN"
    assert r.reason == "BALL_CONTACTS_LINE"


def test_ambiguous_outside_is_review():
    r = decide(-2.0, state="NEAR_LINE")
    assert r.call == "REVIEW"


def test_low_bounce_confidence_is_review():
    r = decide(10, conf=0.20)
    assert r.call == "REVIEW"


def test_low_bounce_score_is_review():
    r = decide(10, score=0.20)
    assert r.call == "REVIEW"
