from __future__ import annotations

from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout


class ProjectModeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open project")
        self.selected_mode = None

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Choose how to open the project:"))

        btns = QHBoxLayout()

        self.btn_edit = QPushButton("Enter in edit mode")
        self.btn_visu = QPushButton("Enter in visualization mode")

        self.btn_edit.clicked.connect(self._choose_edit)
        self.btn_visu.clicked.connect(self._choose_visu)

        btns.addWidget(self.btn_edit)
        btns.addWidget(self.btn_visu)
        layout.addLayout(btns)

    def _choose_edit(self):
        self.selected_mode = "edit"
        self.accept()

    def _choose_visu(self):
        self.selected_mode = "visualization"
        self.accept()
