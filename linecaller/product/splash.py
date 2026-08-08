from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSplashScreen
from PySide6.QtGui import QPixmap


class ProductSplash(QSplashScreen):
    def __init__(self):
        pixmap = QPixmap(640, 360)
        pixmap.fill(Qt.GlobalColor.black)

        super().__init__(pixmap)

        self.showMessage(
            "Open-LineCaller\nLive Pickleball Officiating",
            Qt.AlignmentFlag.AlignCenter,
            Qt.GlobalColor.white,
        )
