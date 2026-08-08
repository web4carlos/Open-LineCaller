from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .navigation import ProductRoute
from .wizard_controller import MatchWizardController
from .wizard_models import CalibrationMode, WizardStep


class MatchWizardScreen(QWidget):
    navigate_requested = Signal(object)
    live_requested = Signal()

    def __init__(self):
        super().__init__()

        self.controller = MatchWizardController()

        outer = QVBoxLayout(self)

        title = QLabel("Start Match")
        title.setObjectName("pageTitle")
        outer.addWidget(title)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)

        self.camera_page = self._build_camera_page()
        self.calibration_page = self._build_calibration_page()
        self.health_page = self._build_health_page()
        self.ready_page = self._build_ready_page()

        for page in (
            self.camera_page,
            self.calibration_page,
            self.health_page,
            self.ready_page,
        ):
            self.stack.addWidget(page)

        self._sync()

    def _build_camera_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(QLabel("Select Camera"))

        self.camera_combo = QComboBox()
        self.camera_combo.addItems(
            ["Camera 0", "Camera 1", "Camera 2", "Camera 3"]
        )

        continue_btn = QPushButton("Continue")
        continue_btn.setObjectName("primaryButton")
        continue_btn.clicked.connect(self._continue_camera)

        back_btn = QPushButton("Back")
        back_btn.clicked.connect(
            lambda: self.navigate_requested.emit(ProductRoute.HOME)
        )

        layout.addWidget(self.camera_combo)
        layout.addWidget(continue_btn)
        layout.addWidget(back_btn)
        layout.addStretch(1)

        return page

    def _build_calibration_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        self.calibration_status = QLabel(
            "Auto Calibration is recommended."
        )

        auto_btn = QPushButton("Run Auto Calibration")
        assisted_btn = QPushButton("Assisted Calibration")
        manual_btn = QPushButton("Manual Calibration")

        auto_btn.setObjectName("primaryButton")

        auto_btn.clicked.connect(
            lambda: self._calibration_attempt(
                CalibrationMode.AUTO
            )
        )

        assisted_btn.clicked.connect(
            lambda: self._calibration_attempt(
                CalibrationMode.ASSISTED
            )
        )

        manual_btn.clicked.connect(
            lambda: self._calibration_attempt(
                CalibrationMode.MANUAL
            )
        )

        layout.addWidget(QLabel("Calibration"))
        layout.addWidget(self.calibration_status)
        layout.addWidget(auto_btn)
        layout.addWidget(assisted_btn)
        layout.addWidget(manual_btn)
        layout.addStretch(1)

        return page

    def _build_health_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        self.health_status = QLabel(
            "System check not yet run."
        )

        run_btn = QPushButton("Run Health Check")
        run_btn.setObjectName("primaryButton")
        run_btn.clicked.connect(self._run_health)

        layout.addWidget(QLabel("System Check"))
        layout.addWidget(self.health_status)
        layout.addWidget(run_btn)
        layout.addStretch(1)

        return page

    def _build_ready_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        ready = QLabel("READY")
        ready.setObjectName("brandTitle")

        message = QLabel(
            "Camera, calibration and system checks are ready."
        )

        start = QPushButton("START MATCH")
        start.setObjectName("primaryButton")
        start.clicked.connect(self._start_live)

        restart = QPushButton("Restart Wizard")
        restart.clicked.connect(self._restart)

        layout.addStretch(1)
        layout.addWidget(ready)
        layout.addWidget(message)
        layout.addWidget(start)
        layout.addWidget(restart)
        layout.addStretch(1)

        return page

    def _continue_camera(self):
        self.controller.select_camera(
            self.camera_combo.currentIndex()
        )
        self.controller.continue_from_camera()
        self._sync()

    def _calibration_attempt(self, mode):
        self.controller.set_calibration_mode(mode)

        # Sprint 2 controlled integration point.
        # Real auto/assisted/manual execution arrives incrementally.
        if mode == CalibrationMode.AUTO:
            self.calibration_status.setText(
                "Auto Calibration completed successfully."
            )
        elif mode == CalibrationMode.ASSISTED:
            self.calibration_status.setText(
                "Assisted Calibration completed successfully."
            )
        else:
            self.calibration_status.setText(
                "Manual Calibration completed successfully."
            )

        self.controller.calibration_success()
        self._sync()

    def _run_health(self):
        # Sprint 3 replaces this controlled PASS
        # with subsystem checks.
        self.controller.health_result(True)

        self.health_status.setText(
            "Camera ✓  Calibration ✓  Replay ✓  Audio ✓  System ✓"
        )

        self._sync()

    def _start_live(self):
        if self.controller.can_start_live:
            self.live_requested.emit()

    def _restart(self):
        self.controller.reset()
        self._sync()

    def _sync(self):
        step = self.controller.state.step

        mapping = {
            WizardStep.CAMERA: self.camera_page,
            WizardStep.CALIBRATION: self.calibration_page,
            WizardStep.HEALTH: self.health_page,
            WizardStep.READY: self.ready_page,
        }

        self.stack.setCurrentWidget(mapping[step])
