from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow,QMessageBox,QStackedWidget

from .developer_page import DeveloperToolsPage
from .home import HomePage
from .match_workspace import MatchWorkspace
from .settings_page import SettingsPage
from .theme import APP_QSS

class OpenLineCallerMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open-LineCaller — Artificial Vision")
        self.resize(1440,900)
        self.setMinimumSize(1100,720)
        self.setStyleSheet(APP_QSS)

        self.camera_index=0
        self.stack=QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home=HomePage()
        self.settings=SettingsPage(self.camera_index)
        self.developer=DeveloperToolsPage()

        self.stack.addWidget(self.home)
        self.stack.addWidget(self.settings)
        self.stack.addWidget(self.developer)

        self.video_workspace=None
        self.live_workspace=None

        self._wire()
        self._build_menu()
        self.statusBar().showMessage("Ready • Artificial Vision Core available")
        self.show_home()

    def _wire(self):
        self.home.analyze_requested.connect(self.show_analyze)
        self.home.live_requested.connect(self.show_live)
        self.home.settings_requested.connect(self.show_settings)
        self.home.developer_requested.connect(self.show_developer)
        self.settings.back_requested.connect(self.show_home)
        self.settings.camera_changed.connect(self._set_camera)
        self.developer.back_requested.connect(self.show_home)

    def _build_menu(self):
        file_menu=self.menuBar().addMenu("File")
        analyze=QAction("Analyze Match",self); analyze.triggered.connect(self.show_analyze); file_menu.addAction(analyze)
        live=QAction("Live Match",self); live.triggered.connect(self.show_live); file_menu.addAction(live)
        file_menu.addSeparator()
        exit_action=QAction("Exit",self); exit_action.triggered.connect(self.close); file_menu.addAction(exit_action)

        tools_menu=self.menuBar().addMenu("Tools")
        developer=QAction("Developer Tools",self); developer.triggered.connect(self.show_developer); tools_menu.addAction(developer)
        settings_action=QAction("Settings",self); settings_action.triggered.connect(self.show_settings); tools_menu.addAction(settings_action)

        help_menu=self.menuBar().addMenu("Help")
        about=QAction("About Open-LineCaller",self); about.triggered.connect(self._about); help_menu.addAction(about)

    def _set_camera(self,value):
        self.camera_index=int(value)
        if self.live_workspace is not None:
            self.live_workspace.camera_index=self.camera_index

    def _stop_workspaces(self):
        if self.video_workspace is not None:
            self.video_workspace.pause()
        if self.live_workspace is not None:
            self.live_workspace.stop()

    def show_home(self):
        self._stop_workspaces()
        self.stack.setCurrentWidget(self.home)
        self.statusBar().showMessage("Ready")

    def show_settings(self):
        self._stop_workspaces()
        self.stack.setCurrentWidget(self.settings)
        self.statusBar().showMessage("Settings")

    def show_developer(self):
        self._stop_workspaces()
        self.stack.setCurrentWidget(self.developer)
        self.statusBar().showMessage("Developer Tools")

    def show_analyze(self):
        self._stop_workspaces()
        if self.video_workspace is None:
            self.video_workspace=MatchWorkspace(mode="video",camera_index=self.camera_index)
            self.video_workspace.back_requested.connect(self.show_home)
            self.stack.addWidget(self.video_workspace)
        self.stack.setCurrentWidget(self.video_workspace)
        self.statusBar().showMessage("Analyze Match")

    def show_live(self):
        self._stop_workspaces()
        if self.live_workspace is None:
            self.live_workspace=MatchWorkspace(mode="live",camera_index=self.camera_index)
            self.live_workspace.back_requested.connect(self.show_home)
            self.stack.addWidget(self.live_workspace)
        self.stack.setCurrentWidget(self.live_workspace)
        self.statusBar().showMessage("Live Match")

    def _about(self):
        QMessageBox.information(
            self,
            "Open-LineCaller",
            "Open-LineCaller v0.1 Basic\nArtificial Vision for Pickleball\n\nSprint 1 — Product Shell",
        )

    def closeEvent(self,event):
        if self.video_workspace is not None:
            self.video_workspace.close_source()
        if self.live_workspace is not None:
            self.live_workspace.close_source()
        event.accept()
