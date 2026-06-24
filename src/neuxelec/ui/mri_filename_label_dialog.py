from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


class NeuXelecDialogHeader(QFrame):
    """
    Minimal frameless-window header with a custom close button.
    Same visual behavior as the other NeuXelec dialogs.
    """

    def __init__(self, dialog: QDialog, parent=None):
        super().__init__(parent)

        self.dialog = dialog
        self._drag_offset: QPoint | None = None

        self.setObjectName("customDialogHeader")
        self.setFixedHeight(34)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addStretch(1)

        self.btn_close_window = QPushButton("✕")
        self.btn_close_window.setObjectName("closeWindowButton")
        self.btn_close_window.setCursor(Qt.PointingHandCursor)
        self.btn_close_window.setFixedSize(30, 30)
        self.btn_close_window.clicked.connect(self.dialog.reject)

        layout.addWidget(
            self.btn_close_window,
            0,
            Qt.AlignRight | Qt.AlignTop,
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_offset = (
                event.globalPosition().toPoint() - self.dialog.frameGeometry().topLeft()
            )
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and bool(event.buttons() & Qt.LeftButton):
            self.dialog.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_offset = None
            event.accept()
            return

        super().mouseReleaseEvent(event)


class MRIFilenameLabelDialog(QDialog):
    """
    NeuXelec-styled dialog used to define how MRI 1 / MRI 2 should be named
    in exported filenames.
    """

    def __init__(self, items: Iterable[dict], parent=None):
        super().__init__(parent)

        self.items = list(items or [])
        self._line_edits: dict[str, QLineEdit] = {}
        self._positioned_once = False

        self.setWindowTitle("MRI filename labels")
        self.setModal(True)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setSizeGripEnabled(False)

        self.setFixedSize(560, 300 if len(self.items) > 1 else 245)

        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.dialog_shell = QFrame()
        self.dialog_shell.setObjectName("dialogShell")
        outer.addWidget(self.dialog_shell)

        root = QVBoxLayout(self.dialog_shell)
        root.setContentsMargins(14, 8, 14, 14)
        root.setSpacing(8)

        self.custom_header = NeuXelecDialogHeader(self)
        root.addWidget(self.custom_header)

        content = QVBoxLayout()
        content.setContentsMargins(20, 0, 20, 6)
        content.setSpacing(12)

        self.lbl_title = QLabel("MRI FILENAME LABELS")
        self.lbl_title.setObjectName("dialogTitle")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        content.addWidget(self.lbl_title)

        self.lbl_subtitle = QLabel("Choose the label used in saved filenames for each loaded MRI.")
        self.lbl_subtitle.setObjectName("dialogSubtitle")
        self.lbl_subtitle.setAlignment(Qt.AlignCenter)
        self.lbl_subtitle.setWordWrap(True)
        content.addWidget(self.lbl_subtitle)

        for item in self.items:
            role = str(item.get("role", ""))
            display_name = str(item.get("display_name", role))
            default = str(item.get("default", ""))
            source_name = str(item.get("source_name", ""))

            row_card = QFrame()
            row_card.setObjectName("inputCard")

            row_layout = QVBoxLayout(row_card)
            row_layout.setContentsMargins(14, 10, 14, 12)
            row_layout.setSpacing(7)

            lbl = QLabel(display_name)
            lbl.setObjectName("fieldTitle")
            row_layout.addWidget(lbl)

            if source_name:
                source_lbl = QLabel(source_name)
                source_lbl.setObjectName("sourceLabel")
                source_lbl.setToolTip(source_name)
                source_lbl.setWordWrap(False)
                row_layout.addWidget(source_lbl)

            edit = QLineEdit()
            edit.setObjectName("labelInput")
            edit.setText(default)
            edit.setPlaceholderText(default or display_name.replace(" ", ""))
            edit.setMinimumHeight(38)
            row_layout.addWidget(edit)

            self._line_edits[role] = edit
            content.addWidget(row_card)

        root.addLayout(content)
        root.addStretch(1)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(20, 0, 20, 4)
        buttons.setSpacing(10)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("secondaryButton")
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setMinimumHeight(42)
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_ok = QPushButton("Save")
        self.btn_ok.setObjectName("primaryButton")
        self.btn_ok.setCursor(Qt.PointingHandCursor)
        self.btn_ok.setMinimumHeight(42)
        self.btn_ok.setMinimumWidth(118)
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self.accept)

        buttons.addStretch(1)
        buttons.addWidget(self.btn_cancel)
        buttons.addWidget(self.btn_ok)
        root.addLayout(buttons)

        if self._line_edits:
            first_edit = next(iter(self._line_edits.values()))
            first_edit.selectAll()
            first_edit.setFocus()

    def showEvent(self, event) -> None:
        super().showEvent(event)

        if self._positioned_once:
            return

        self._positioned_once = True

        try:
            parent = self.parentWidget()
            screen = parent.screen() if parent is not None else self.screen()

            if screen is None:
                screen = QGuiApplication.primaryScreen()

            geometry = self.frameGeometry()

            if parent is not None:
                geometry.moveCenter(parent.frameGeometry().center())
            elif screen is not None:
                geometry.moveCenter(screen.availableGeometry().center())

            self.move(geometry.topLeft())
        except Exception:
            pass

    def values(self) -> dict[str, str]:
        return {role: edit.text().strip() for role, edit in self._line_edits.items()}

    @staticmethod
    def get_labels(items: Iterable[dict], parent=None) -> dict[str, str] | None:
        dlg = MRIFilenameLabelDialog(items=items, parent=parent)
        result = dlg.exec()

        if result == QDialog.Accepted:
            return dlg.values()

        return None

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QDialog {
                background: transparent;
            }

            QFrame#dialogShell {
                background-color: #06070D;
                border: 1.5px solid #FF487D;
                border-radius: 16px;
            }

            QFrame#customDialogHeader {
                background-color: transparent;
                border: none;
            }

            QPushButton#closeWindowButton {
                color: #D8DAE4;
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 700;
                padding: 0px;
                min-width: 30px;
                max-width: 30px;
                min-height: 30px;
                max-height: 30px;
            }

            QPushButton#closeWindowButton:hover {
                color: white;
                border: 1px solid transparent;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
            }

            QPushButton#closeWindowButton:pressed {
                color: white;
                border: 1px solid transparent;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #E56F00,
                    stop:1 #E0008C
                );
            }

            QLabel {
                background-color: transparent;
                border: none;
            }

            QLabel#dialogTitle {
                color: #F4D9D0;
                font-size: 18px;
                font-weight: 500;
                letter-spacing: 1px;
            }

            QLabel#dialogSubtitle {
                color: #8E8E98;
                font-size: 12px;
                font-weight: 400;
                padding-bottom: 3px;
            }

            QFrame#inputCard {
                background-color: #10121A;
                border: 1px solid #2B2D38;
                border-radius: 12px;
            }

            QLabel#fieldTitle {
                color: #CFCFD6;
                font-size: 13px;
                font-weight: 600;
            }

            QLabel#sourceLabel {
                color: #8E8E98;
                font-size: 11px;
                font-weight: 500;
            }

            QLineEdit#labelInput {
                min-height: 38px;
                background-color: #171922;
                color: #F2F2F5;
                border: 1px solid #2B2D38;
                border-radius: 8px;
                padding-left: 12px;
                padding-right: 12px;
                selection-background-color: #FF008F;
                selection-color: white;
                font-size: 12px;
                font-weight: 500;
            }

            QLineEdit#labelInput:hover {
                border: 1px solid #3A3D4A;
            }

            QLineEdit#labelInput:focus {
                border: 1px solid #FF487D;
            }

            QPushButton {
                min-height: 42px;
                border-radius: 10px;
                font-size: 13px;
                font-weight: 600;
                padding-left: 18px;
                padding-right: 18px;
            }

            QPushButton#secondaryButton {
                color: white;
                background-color: #17181F;
                border: 1px solid #2B2D38;
            }

            QPushButton#secondaryButton:hover {
                background-color: #20222B;
                border: 1px solid #FF487D;
            }

            QPushButton#secondaryButton:pressed {
                background-color: #14151B;
                border: 1px solid #FF487D;
            }

            QPushButton#primaryButton {
                color: white;
                border: none;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
            }

            QPushButton#primaryButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF922B,
                    stop:1 #FF33B8
                );
            }

            QPushButton#primaryButton:pressed {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #E56F00,
                    stop:1 #E0008C
                );
            }
            """)
