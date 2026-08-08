from linecaller.product.navigation import ProductRoute


def test_expected_product_routes():
    assert {
        route.value
        for route in ProductRoute
    } == {
        "HOME",
        "START_MATCH",
        "HISTORY",
        "SETTINGS",
        "DIAGNOSTICS",
    }
