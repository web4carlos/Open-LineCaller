import sys

from PySide6.QtWidgets import QApplication

from linecaller.apps.live_window import LiveMatchWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Open-LineCaller LIVE")

    window = LiveMatchWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
