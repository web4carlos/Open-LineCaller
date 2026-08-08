from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QStackedWidget

from linecaller.apps.live_window import LiveMatchWindow

from .match_wizard_screen import MatchWizardScreen
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
        self.live_window = None

        self.setStyleSheet(APP_STYLESHEET)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.match_wizard = MatchWizardScreen()
        self.match_wizard.live_requested.connect(
            self._open_live_match
        )

        self.screens = {
            ProductRoute.HOME: HomeScreen(),
            ProductRoute.START_MATCH: self.match_wizard,
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

    def _open_live_match(self):
        self.session.selected_camera = (
            self.match_wizard.controller.state.selected_camera
        )

        self.session.calibration_profile = (
            "wizard-calibration"
            if self.match_wizard.controller.state.calibration_valid
            else None
        )

        self.live_window = LiveMatchWindow()

        # carry selected camera into live shell
        self.live_window.camera_combo.setCurrentIndex(
            self.session.selected_camera
        )

        if self.session.has_calibration:
            self.live_window.controller.set_ready(
                calibration_valid=True
            )
            self.live_window._refresh_status()

        self.live_window.show()
