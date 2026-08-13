from linecaller.rules.serve_two_bounce import (
    ServeContext,
    ServeLegalityEvaluator,
    TwoBounceRule,
)

def main():
    serve = ServeLegalityEvaluator().evaluate(
        ServeContext(
            serving_team="A",
            server_side="RIGHT",
            receiving_zone="FAR_LEFT_SERVICE",
            nearest_line="FAR_BASELINE",
            on_line=False,
            upstream_decision="IN",
        )
    )
    print(f"serve={serve.ruling.value} reason={serve.reason}")

    t = TwoBounceRule()
    r1=t.receiver_bounce("IN")
    print(f"step=receiver_bounce state={r1.state.value} ruling={r1.ruling} reason={r1.reason}")
    r2=t.receiver_return_hit(False)
    print(f"step=receiver_return state={r2.state.value} ruling={r2.ruling} reason={r2.reason}")
    r3=t.server_side_bounce("IN")
    print(f"step=server_bounce state={r3.state.value} ruling={r3.ruling} reason={r3.reason}")

if __name__=="__main__":
    main()
