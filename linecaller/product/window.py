from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QStackedWidget

from .navigation import NavigationController, ProductRoute
from .screens import HomeScreen, PlaceholderScreen
from .session import ProductSession
from .theme import APP_STYLESHEET


class ProductAppWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Open-LineCaller")
        self.resize(1100, 760)

        self.navigation = NavigationController()
        self.session = ProductSession()

        self.setStyleSheet(APP_STYLESHEET)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.screens = {
            ProductRoute.HOME: HomeScreen(),
            ProductRoute.START_MATCH: PlaceholderScreen(
                "Start Match",
                "Match Wizard arrives in CP-0018 Sprint 2.",
            ),
            ProductRoute.HISTORY: PlaceholderScreen(
                "History",
                "Match history arrives in a later sprint.",
            ),
            ProductRoute.SETTINGS: PlaceholderScreen(
                "Settings",
                "Product settings will be connected incrementally.",
            ),
            ProductRoute.DIAGNOSTICS: PlaceholderScreen(
                "Diagnostics",
                "Developer diagnostics will be exposed here.",
            ),
        }

        for route in ProductRoute:
            screen = self.screens[route]
            self.stack.addWidget(screen)
            screen.navigate_requested.connect(self.navigate)

        self.navigate(ProductRoute.HOME)

    def navigate(self, route):
        route = self.navigation.navigate(route)
        self.stack.setCurrentWidget(self.screens[route])
