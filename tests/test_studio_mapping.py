import pytest

from linecaller.calibration.studio_mapping import DisplayMapping


def test_mapping_centered_letterbox():
    mapping = DisplayMapping(
        frame_width=1920,
        frame_height=1080,
        widget_width=1000,
        widget_height=800,
    )

    # Image is 1000x562.5, centered vertically.
    point = mapping.widget_to_frame(500, 400)

    assert point is not None
    assert point[0] == pytest.approx(960, abs=1)
    assert point[1] == pytest.approx(540, abs=1)


def test_mapping_rejects_letterbox_area():
    mapping = DisplayMapping(
        frame_width=1920,
        frame_height=1080,
        widget_width=1000,
        widget_height=800,
    )

    assert mapping.widget_to_frame(500, 20) is None
