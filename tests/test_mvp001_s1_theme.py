from linecaller.product_shell.theme import APP_QSS

def test_theme_contains_product_shell_styles():
    assert "PrimaryButton" in APP_QSS
    assert "HeroCard" in APP_QSS
    assert "Decision" in APP_QSS
