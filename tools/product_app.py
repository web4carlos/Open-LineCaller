import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from linecaller.product.splash import ProductSplash
from linecaller.product.window import ProductAppWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Open-LineCaller")

    splash = ProductSplash()
    splash.show()

    window = ProductAppWindow()

    QTimer.singleShot(
        700,
        lambda: (
            splash.finish(window),
            window.show(),
        ),
    )

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
