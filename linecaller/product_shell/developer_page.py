import subprocess,sys
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame,QGridLayout,QHBoxLayout,QLabel,QMessageBox,QPushButton,QVBoxLayout,QWidget

class DeveloperToolsPage(QWidget):
    back_requested=Signal()

    def __init__(self):
        super().__init__(); self._build()

    def _launch(self,module):
        try:
            subprocess.Popen([sys.executable,"-m",module])
        except Exception as exc:
            QMessageBox.critical(self,"Developer Tools",str(exc))

    def _tool(self,title,subtitle,module):
        card=QFrame(); card.setObjectName("PanelCard")
        l=QVBoxLayout(card); l.setContentsMargins(18,18,18,18)
        h=QLabel(title); h.setObjectName("SectionTitle"); l.addWidget(h)
        s=QLabel(subtitle); s.setObjectName("Muted"); s.setWordWrap(True); l.addWidget(s)
        l.addStretch(1)
        b=QPushButton("Open"); b.clicked.connect(lambda:self._launch(module)); l.addWidget(b)
        return card

    def _build(self):
        outer=QVBoxLayout(self); outer.setContentsMargins(28,24,28,24)
        top=QHBoxLayout()
        back=QPushButton("← Home"); back.clicked.connect(self.back_requested.emit); top.addWidget(back)
        title=QLabel("Developer Tools"); title.setObjectName("SectionTitle"); top.addWidget(title); top.addStretch(1)
        outer.addLayout(top)

        grid=QGridLayout(); grid.setSpacing(14)
        tools=[
            ("Validation Lab","Ground-truth and validation workflow.","tools.validation_lab"),
            ("Smart Annotation Studio","Review candidate events and label IN / OUT / SKIP.","tools.smart_annotation_studio"),
            ("Detector Tuning Lab","Inspect masks and detector candidates.","tools.detector_tuning_lab"),
        ]
        for i,item in enumerate(tools):
            grid.addWidget(self._tool(*item),i//2,i%2)
        outer.addLayout(grid,1)
        note=QLabel("Engineering tools are separated from the normal match workflow."); note.setObjectName("Muted"); outer.addWidget(note)
