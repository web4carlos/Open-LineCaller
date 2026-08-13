from linecaller.rules import (
    PickleballRulesEngine,
    RallyEvent,
    RallyEventType,
    TeamSide,
)


def main():
    engine = PickleballRulesEngine()

    events = [
        RallyEvent(RallyEventType.START_RALLY, 0, team=TeamSide.A),
        RallyEvent(RallyEventType.BOUNCE, 29, decision="OUT", zone="OUTSIDE"),
    ]

    for event in events:
        result = engine.apply(event)
        print(
            f"frame={event.frame} "
            f"event={event.event_type.value} "
            f"phase={result.state.phase.value} "
            f"ruling={result.ruling} "
            f"active={result.state.rally_active} "
            f"review={result.state.review_required} "
            f"reason={result.reason}"
        )


if __name__ == "__main__":
    main()
