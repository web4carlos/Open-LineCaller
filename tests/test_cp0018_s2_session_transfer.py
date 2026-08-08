from linecaller.product.session import ProductSession


def test_session_can_store_wizard_camera():
    s = ProductSession()
    s.selected_camera = 3

    assert s.selected_camera == 3
