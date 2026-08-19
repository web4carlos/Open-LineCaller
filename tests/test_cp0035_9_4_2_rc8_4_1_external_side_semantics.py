from linecaller.dcf.external_grid_frame_loop import ExternalGridCalibrator


def test_y_zero_external_edge_is_far():
    assert ExternalGridCalibrator._region(
        40.0, -0.5, 83.0, 182.0
    ) == "OUT_FAR"


def test_y_max_external_edge_is_near():
    assert ExternalGridCalibrator._region(
        40.0, 182.5, 83.0, 182.0
    ) == "OUT_NEAR"


def test_left_and_right_semantics_are_unchanged():
    assert ExternalGridCalibrator._region(
        -0.5, 80.0, 83.0, 182.0
    ) == "OUT_LEFT"
    assert ExternalGridCalibrator._region(
        83.5, 80.0, 83.0, 182.0
    ) == "OUT_RIGHT"


def test_external_corners_remain_corner():
    assert ExternalGridCalibrator._region(
        -0.5, -0.5, 83.0, 182.0
    ) == "OUT_CORNER"
    assert ExternalGridCalibrator._region(
        83.5, 182.5, 83.0, 182.0
    ) == "OUT_CORNER"