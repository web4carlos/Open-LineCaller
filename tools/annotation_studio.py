import sys
from PySide6.QtWidgets import QApplication
from linecaller.annotation.window import AnnotationStudioWindow
def main():
    app=QApplication(sys.argv);app.setApplicationName("Open-LineCaller Annotation Studio")
    window=AnnotationStudioWindow();window.show()
    return app.exec()
if __name__=="__main__": raise SystemExit(main())
