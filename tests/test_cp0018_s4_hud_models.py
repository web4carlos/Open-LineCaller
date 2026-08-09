from linecaller.product.hud_models import (
    HUDMetric,
    StatusLevel,
)


def test_hud_metric():
    metric = HUDMetric(
        label="FPS",
        value="60.0",
        level=StatusLevel.OK,
        detail="Excellent",
    )

    assert metric.label == "FPS"
    assert metric.level == StatusLevel.OK
