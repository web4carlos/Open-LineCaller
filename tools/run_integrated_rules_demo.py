from linecaller.rules.integrated_pipeline import IntegratedPickleballRulesPipeline
from linecaller.rules.models import TeamSide


def show(name, result):
    print(
        f"{name}: status={result.status.value} "
        f"winner={result.winner.value} "
        f"loser={result.loser.value} "
        f"reason={result.reason}"
    )


def main():
    p = IntegratedPickleballRulesPipeline()

    show("legal_serve", p.evaluate_serve(
        serving_team=TeamSide.A,
        receiving_team=TeamSide.B,
        server_side="RIGHT",
        receiving_zone="FAR_LEFT_SERVICE",
        decision="IN",
    ))

    show("receiver_return", p.evaluate_return(
        receiving_team=TeamSide.B,
        serving_team=TeamSide.A,
        return_volley=False,
    ))

    show("second_required_bounce", p.evaluate_server_side_bounce(
        serving_team=TeamSide.A,
        receiving_team=TeamSide.B,
        decision="IN",
    ))

    show("open_rally_ball_out", p.evaluate_ball_decision(
        acting_team=TeamSide.A,
        opponent_team=TeamSide.B,
        decision="OUT",
    ))

    p.reset_rally()

    show("nvz_fault", p.evaluate_nvz_contact(
        acting_team=TeamSide.B,
        opponent_team=TeamSide.A,
        volley=True,
        player_in_nvz=True,
    ))

    show("uncertain_event", p.evaluate_ball_decision(
        acting_team=TeamSide.A,
        opponent_team=TeamSide.B,
        decision="REVIEW",
    ))


if __name__ == "__main__":
    main()
