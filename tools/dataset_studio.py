import sys
from PySide6.QtWidgets import QApplication
from linecaller.dataset.studio_window import DatasetStudioWindow
def main():
    app=QApplication(sys.argv); w=DatasetStudioWindow(); w.show(); return app.exec()
if __name__=="__main__": raise SystemExit(main())
