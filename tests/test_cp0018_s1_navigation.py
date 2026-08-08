from linecaller.product.navigation import (
    NavigationController,
    ProductRoute,
)


def test_navigation_starts_home():
    nav = NavigationController()
    assert nav.current == ProductRoute.HOME


def test_navigation_and_back():
    nav = NavigationController()

    nav.navigate(ProductRoute.SETTINGS)
    nav.navigate(ProductRoute.DIAGNOSTICS)

    assert nav.current == ProductRoute.DIAGNOSTICS
    assert nav.back() == ProductRoute.SETTINGS
    assert nav.back() == ProductRoute.HOME
