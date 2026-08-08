import sys

from PySide6.QtWidgets import QApplication

from linecaller.calibration.studio_window import CalibrationStudioWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Open-LineCaller Calibration Studio")

    window = CalibrationStudioWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
