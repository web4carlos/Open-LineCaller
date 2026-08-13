from linecaller.rules import (
    PickleballRulesEngine,
    RallyEvent,
    RallyEventType,
    RallyPhase,
    TeamSide,
)


def start(engine, frame=1):
    return engine.apply(
        RallyEvent(
            RallyEventType.START_RALLY,
            frame,
            team=TeamSide.A,
        )
    )


def test_start_rally_sets_serve_phase():
    e=PickleballRulesEngine()
    r=start(e)
    assert r.state.phase==RallyPhase.SERVE_FLIGHT
    assert r.state.rally_active is True
    assert r.state.serving_team==TeamSide.A
    assert r.state.receiving_team==TeamSide.B


def test_first_bounce_advances_sequence():
    e=PickleballRulesEngine()
    start(e)
    r=e.apply(RallyEvent(RallyEventType.BOUNCE,10,decision="IN"))
    assert r.state.phase==RallyPhase.RETURN_FLIGHT
    assert r.state.bounce_count==1


def test_second_bounce_advances_sequence():
    e=PickleballRulesEngine()
    start(e)
    e.apply(RallyEvent(RallyEventType.BOUNCE,10,decision="IN"))
    r=e.apply(RallyEvent(RallyEventType.BOUNCE,20,decision="IN"))
    assert r.state.phase==RallyPhase.THIRD_SHOT_FLIGHT
    assert r.state.bounce_count==2


def test_review_is_never_upgraded():
    e=PickleballRulesEngine()
    start(e)
    r=e.apply(RallyEvent(RallyEventType.BOUNCE,10,decision="REVIEW"))
    assert r.ruling=="REVIEW"
    assert r.state.review_required is True
    assert r.state.phase==RallyPhase.REVIEW


def test_fault_ends_rally():
    e=PickleballRulesEngine()
    start(e)
    r=e.apply(RallyEvent(RallyEventType.FAULT,20,reason="TEST_FAULT"))
    assert r.ruling=="FAULT"
    assert r.state.phase==RallyPhase.ENDED
    assert r.state.rally_active is False


def test_event_before_current_frame_is_rejected():
    e=PickleballRulesEngine()
    start(e,10)
    r=e.apply(RallyEvent(RallyEventType.HIT,9))
    assert r.accepted is False
    assert r.reason=="EVENT_FRAME_PRECEDES_CURRENT_RALLY_STATE"


def test_event_without_active_rally_is_ignored():
    e=PickleballRulesEngine()
    r=e.apply(RallyEvent(RallyEventType.BOUNCE,10,decision="IN"))
    assert r.accepted is False
    assert r.reason=="NO_ACTIVE_RALLY"


def test_hit_counter():
    e=PickleballRulesEngine()
    start(e)
    r=e.apply(RallyEvent(RallyEventType.HIT,5,team=TeamSide.A))
    assert r.state.hit_count==1
