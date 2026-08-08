from linecaller.product.session import ProductSession


def test_product_session_defaults():
    s = ProductSession()

    assert s.selected_camera == 0
    assert s.has_calibration is False
    assert s.active_match_id is None


def test_product_session_calibration():
    s = ProductSession(
        calibration_profile="court_1.json",
    )

    assert s.has_calibration is True
