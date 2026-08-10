from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout,QFrame,QHBoxLayout,QLabel,QPushButton,QSpinBox,QVBoxLayout,QWidget

class SettingsPage(QWidget):
    back_requested=Signal()
    camera_changed=Signal(int)

    def __init__(self,camera_index=0):
        super().__init__(); self._build(camera_index)

    def _build(self,camera_index):
        outer=QVBoxLayout(self); outer.setContentsMargins(28,24,28,24)
        top=QHBoxLayout()
        back=QPushButton("← Home"); back.clicked.connect(self.back_requested.emit); top.addWidget(back)
        title=QLabel("Settings"); title.setObjectName("SectionTitle"); top.addWidget(title); top.addStretch(1)
        outer.addLayout(top)

        card=QFrame(); card.setObjectName("PanelCard")
        form=QFormLayout(card); form.setContentsMargins(24,24,24,24)
        self.camera=QSpinBox(); self.camera.setRange(0,16); self.camera.setValue(camera_index)
        self.camera.valueChanged.connect(self.camera_changed.emit)
        form.addRow("Default camera",self.camera)
        note=QLabel("Sprint 1 keeps settings intentionally small. Audio, model selection and performance profiles are deferred until the Basic flow is validated.")
        note.setObjectName("Muted"); note.setWordWrap(True); form.addRow(note)
        outer.addWidget(card); outer.addStretch(1)
