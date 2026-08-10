import sys
from PySide6.QtWidgets import QApplication
from linecaller.product_shell import OpenLineCallerMainWindow

def main():
    app=QApplication(sys.argv)
    app.setApplicationName("Open-LineCaller")
    app.setOrganizationName("Open-LineCaller")
    window=OpenLineCallerMainWindow()
    window.show()
    return app.exec()

if __name__=="__main__":
    raise SystemExit(main())
