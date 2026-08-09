import sys
from PySide6.QtWidgets import QApplication
from linecaller.validation.window import ValidationLabWindow

def main():
    app = QApplication(sys.argv)
    window = ValidationLabWindow()
    window.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
