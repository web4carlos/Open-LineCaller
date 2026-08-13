from linecaller.rules.nvz import (
    NVZContext,
    NVZRuleEvaluator,
)


def show(name, **kwargs):
    r = NVZRuleEvaluator().evaluate(NVZContext(**kwargs))
    print(f"{name}: ruling={r.ruling.value} reason={r.reason}")


def main():
    show(
        "groundstroke_from_kitchen",
        volley=False,
        player_in_nvz=True,
    )
    show(
        "volley_from_kitchen",
        volley=True,
        player_in_nvz=True,
    )
    show(
        "volley_momentum_into_kitchen",
        volley=True,
        player_in_nvz=False,
        entered_nvz_after_volley=True,
    )
    show(
        "clean_volley",
        volley=True,
        player_in_nvz=False,
        entered_nvz_after_volley=False,
        momentum_complete=True,
    )
    show(
        "momentum_uncertain",
        volley=True,
        player_in_nvz=False,
        entered_nvz_after_volley=False,
        momentum_complete=False,
    )


if __name__ == "__main__":
    main()
