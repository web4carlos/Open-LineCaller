from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame,QHBoxLayout,QLabel,QPushButton,QVBoxLayout,QWidget

class HomePage(QWidget):
    analyze_requested=Signal()
    live_requested=Signal()
    settings_requested=Signal()
    developer_requested=Signal()

    def __init__(self):
        super().__init__(); self._build()

    def _card(self,title,subtitle,button_text,handler):
        card=QFrame(); card.setObjectName("PanelCard")
        layout=QVBoxLayout(card); layout.setContentsMargins(22,22,22,22)
        h=QLabel(title); h.setObjectName("SectionTitle"); layout.addWidget(h)
        s=QLabel(subtitle); s.setObjectName("Muted"); s.setWordWrap(True); layout.addWidget(s)
        layout.addStretch(1)
        b=QPushButton(button_text); b.setObjectName("PrimaryButton"); b.clicked.connect(handler); layout.addWidget(b)
        return card

    def _build(self):
        outer=QVBoxLayout(self); outer.setContentsMargins(34,30,34,30); outer.setSpacing(18)
        hero=QFrame(); hero.setObjectName("HeroCard")
        hl=QVBoxLayout(hero); hl.setContentsMargins(30,28,30,28)
        brand=QLabel("OPEN-LINECALLER"); brand.setObjectName("Brand"); hl.addWidget(brand)
        title=QLabel("Artificial Vision for Pickleball"); title.setObjectName("HeroTitle"); hl.addWidget(title)
        sub=QLabel("Analyze recorded matches or officiate live using the same Artificial Vision Core.")
        sub.setObjectName("HeroSubtitle"); sub.setWordWrap(True); hl.addWidget(sub)
        outer.addWidget(hero)

        row=QHBoxLayout(); row.setSpacing(16)
        row.addWidget(self._card("Analyze Match","Open a recorded match and run Artificial Vision.","Open Video",self.analyze_requested.emit))
        row.addWidget(self._card("Live Match","Connect a camera and run the same core in real time.","Start Live",self.live_requested.emit))
        outer.addLayout(row,1)

        footer=QHBoxLayout()
        settings=QPushButton("Settings"); settings.clicked.connect(self.settings_requested.emit); footer.addWidget(settings)
        dev=QPushButton("Developer Tools"); dev.clicked.connect(self.developer_requested.emit); footer.addWidget(dev)
        footer.addStretch(1)
        version=QLabel("v0.1 Basic • Sprint 1"); version.setObjectName("Muted"); footer.addWidget(version)
        outer.addLayout(footer)
