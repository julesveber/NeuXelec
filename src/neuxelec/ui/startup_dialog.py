from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from neuxelec.utils.resources import resource_path

from .open_project_dialog import OpenProjectDialog

FRAMELESS_DIALOG_BASE_STYLE = """
    QDialog {
        background: transparent;
    }

    QFrame#dialogShell {
        background-color: #06070D;
        border: 1px solid #FF487D;
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
"""


class NeuXelecDialogHeader(QFrame):
    """
    Minimal custom title bar used by frameless NeuXelec dialogs.
    The empty area can be dragged to move the window.
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


class CreateProjectDialog(QDialog):
    """
    Compact styled dialog used to enter the patient ID for a new project.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Create new project")

        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setSizeGripEnabled(False)

        # Height includes the custom header.
        self.setFixedSize(470, 268)

        self._build_ui()
        self._apply_style()

        self.edit_patient_id.setFocus()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.dialog_shell = QFrame()
        self.dialog_shell.setObjectName("dialogShell")

        shell_layout = QVBoxLayout(self.dialog_shell)
        shell_layout.setContentsMargins(14, 8, 14, 14)
        shell_layout.setSpacing(8)

        outer_layout.addWidget(self.dialog_shell)

        self.custom_header = NeuXelecDialogHeader(self)
        shell_layout.addWidget(self.custom_header)

        self.content_widget = QWidget()
        self.content_widget.setObjectName("createProjectContent")

        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(12, 4, 12, 8)
        content_layout.setSpacing(12)

        self.lbl_patient_id = QLabel("Enter patient ID")
        self.lbl_patient_id.setObjectName("patientIdLabel")
        self.lbl_patient_id.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.lbl_patient_id)

        self.edit_patient_id = QLineEdit()
        self.edit_patient_id.setObjectName("patientIdInput")
        self.edit_patient_id.setPlaceholderText("")
        self.edit_patient_id.setMinimumHeight(44)
        self.edit_patient_id.returnPressed.connect(self._validate_and_accept)
        content_layout.addWidget(self.edit_patient_id)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(8)
        buttons_layout.addStretch(1)

        self.btn_continue = QPushButton("Continue")
        self.btn_continue.setObjectName("continueButton")
        self.btn_continue.setCursor(Qt.PointingHandCursor)
        self.btn_continue.setFixedSize(126, 44)
        self.btn_continue.clicked.connect(self._validate_and_accept)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("cancelButton")
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setFixedSize(108, 44)
        self.btn_cancel.clicked.connect(self.reject)

        buttons_layout.addWidget(self.btn_continue)
        buttons_layout.addWidget(self.btn_cancel)
        buttons_layout.addStretch(1)

        content_layout.addLayout(buttons_layout)

        shell_layout.addWidget(self.content_widget, 1)

    def _apply_style(self) -> None:
        self.setStyleSheet(FRAMELESS_DIALOG_BASE_STYLE + """
            QWidget#createProjectContent {
                background-color: transparent;
                border: none;
            }

            QLabel#patientIdLabel {
                color: #F2F2F5;
                font-size: 14px;
                font-weight: 500;
                background-color: transparent;
                border: none;
            }

            QLineEdit#patientIdInput {
                min-height: 44px;
                background-color: #171922;
                color: white;
                border: 1px solid #2B2D38;
                border-radius: 10px;
                padding-left: 12px;
                padding-right: 12px;
                selection-background-color: #FF008F;
                font-size: 13px;
            }

            QLineEdit#patientIdInput:hover {
                border: 1px solid #3A3D4A;
            }

            QLineEdit#patientIdInput:focus {
                border: 1px solid #FF487D;
            }

            QPushButton#continueButton,
            QPushButton#cancelButton {
                min-height: 44px;
                border-radius: 10px;
                font-size: 13px;
                font-weight: 600;
                padding-left: 18px;
                padding-right: 18px;
            }

            QPushButton#continueButton {
                color: white;
                border: none;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
            }

            QPushButton#continueButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF922B,
                    stop:1 #FF33B8
                );
            }

            QPushButton#continueButton:pressed {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #E56F00,
                    stop:1 #E0008C
                );
            }

            QPushButton#cancelButton {
                color: white;
                background-color: #17181F;
                border: 1px solid #2B2D38;
            }

            QPushButton#cancelButton:hover {
                background-color: #20222B;
                border: 1px solid #FF487D;
            }

            QPushButton#cancelButton:pressed {
                background-color: #14151B;
                border: 1px solid #FF487D;
            }
            """)

    def patient_id(self) -> str:
        return self.edit_patient_id.text().strip()

    def _validate_and_accept(self) -> None:
        if not self.patient_id():
            QMessageBox.warning(
                self,
                "Missing patient ID",
                "Please enter a patient ID.",
            )
            self.edit_patient_id.setFocus()
            return

        self.accept()


class StartupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("NeuXelec")
        self.result_data = None

        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setSizeGripEnabled(False)

        # Welcome window dimensions.
        # Modify these values if you want to fine-tune its proportions.
        self.setFixedSize(500, 466)

        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.dialog_shell = QFrame()
        self.dialog_shell.setObjectName("dialogShell")

        shell_layout = QVBoxLayout(self.dialog_shell)
        shell_layout.setContentsMargins(14, 8, 14, 14)
        shell_layout.setSpacing(8)

        outer_layout.addWidget(self.dialog_shell)

        self.custom_header = NeuXelecDialogHeader(self)
        shell_layout.addWidget(self.custom_header)

        self.content_widget = QWidget()
        self.content_widget.setObjectName("startupContent")

        layout = QVBoxLayout(self.content_widget)
        layout.setContentsMargins(28, 0, 28, 22)
        layout.setSpacing(12)

        layout.addItem(QSpacerItem(20, 4, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # Welcome text
        self.lbl_welcome = QLabel("WELCOME TO")
        self.lbl_welcome.setObjectName("welcomeLabel")
        self.lbl_welcome.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_welcome)

        # Logo PNG
        logo_path = resource_path("resources/images/neuxelec_logo.png")

        self.logo_widget = QLabel()
        self.logo_widget.setObjectName("logoWidget")
        self.logo_widget.setAlignment(Qt.AlignCenter)
        self.logo_widget.setFixedHeight(122)

        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))

            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    360,
                    122,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.logo_widget.setPixmap(scaled_pixmap)
            else:
                self.logo_widget.setText("Logo could not be loaded")
                self.logo_widget.setObjectName("logoErrorLabel")
        else:
            self.logo_widget.setText("Logo not found")
            self.logo_widget.setObjectName("logoErrorLabel")

        layout.addWidget(self.logo_widget, alignment=Qt.AlignCenter)

        self.lbl_subtitle = QLabel("Multimodal Electrode Hub")
        self.lbl_subtitle.setObjectName("subtitleLabel")
        self.lbl_subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_subtitle)

        layout.addItem(QSpacerItem(20, 14, QSizePolicy.Minimum, QSizePolicy.Fixed))

        self.btn_open = QPushButton("Open project")
        self.btn_create = QPushButton("Create new project")

        self.btn_open.setObjectName("btnOpenProject")
        self.btn_create.setObjectName("btnCreateProject")

        self.btn_open.setCursor(Qt.PointingHandCursor)
        self.btn_create.setCursor(Qt.PointingHandCursor)

        self.btn_open.clicked.connect(self._open_project)
        self.btn_create.clicked.connect(self._create_project)

        layout.addWidget(self.btn_open)
        layout.addWidget(self.btn_create)

        layout.addStretch(1)

        shell_layout.addWidget(self.content_widget, 1)

    def _apply_style(self) -> None:
        self.setStyleSheet(FRAMELESS_DIALOG_BASE_STYLE + """
            QWidget#startupContent {
                background-color: transparent;
                border: none;
            }

            QLabel#welcomeLabel {
                color: #F4D9D0;
                font-size: 18px;
                font-weight: 500;
                letter-spacing: 1px;
                background-color: transparent;
                border: none;
            }

            QLabel#logoWidget {
                background-color: transparent;
                border: none;
            }

            QLabel#logoErrorLabel {
                color: #FF00A0;
                font-size: 14px;
                font-weight: 500;
                background-color: transparent;
                border: none;
            }

            QLabel#subtitleLabel {
                color: #8E8E98;
                font-size: 12px;
                font-weight: 400;
                background-color: transparent;
                border: none;
            }

            QPushButton#btnOpenProject,
            QPushButton#btnCreateProject {
                min-height: 48px;
                border-radius: 14px;
                font-size: 14px;
                font-weight: 600;
                padding-left: 18px;
                padding-right: 18px;
            }

            QPushButton#btnOpenProject {
                background-color: #17181F;
                color: white;
                border: 1px solid #2B2D38;
            }

            QPushButton#btnOpenProject:hover {
                background-color: #20222B;
                border: 1px solid #FF487D;
            }

            QPushButton#btnOpenProject:pressed {
                background-color: #14151B;
                border: 1px solid #FF487D;
            }

            QPushButton#btnCreateProject {
                color: white;
                border: none;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
            }

            QPushButton#btnCreateProject:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF922B,
                    stop:1 #FF33B8
                );
            }

            QPushButton#btnCreateProject:pressed {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #E56F00,
                    stop:1 #E0008C
                );
            }
            """)

    def _center_child_dialog(self, dialog: QDialog) -> None:
        """
        Center a secondary dialog relative to the startup welcome window.
        """
        try:
            parent_geometry = self.frameGeometry()
            dialog_geometry = dialog.frameGeometry()

            dialog_geometry.moveCenter(parent_geometry.center())
            dialog.move(dialog_geometry.topLeft())

        except Exception:
            pass

    def _open_project(self):
        """
        Open an existing NeuXelec project through the recent-projects browser.

        The dialog combines:
            - recent JSON project selection;
            - manual file browsing;
            - Edit mode / Visualization only selection.
        """
        dlg = OpenProjectDialog(self)

        # Center the Open project window relative to the welcome window
        # once Qt has fully created and sized the dialog.
        QTimer.singleShot(0, lambda: self._center_child_dialog(dlg))

        if dlg.exec() != QDialog.Accepted:
            return

        if not dlg.selected_project_path or not dlg.selected_mode:
            return

        self.result_data = {
            "action": "open",
            "project_path": str(dlg.selected_project_path),
            "mode": str(dlg.selected_mode),
            "patient_id": None,
        }

        self.accept()

    def _create_project(self):
        """
        Create a new NeuXelec project after retrieving a patient ID from the
        custom styled creation dialog.
        """
        dlg = CreateProjectDialog(self)

        # Center the Create new project window relative to the welcome window
        # once Qt has fully created and sized the dialog.
        QTimer.singleShot(0, lambda: self._center_child_dialog(dlg))

        if dlg.exec() != QDialog.Accepted:
            return

        patient_id = dlg.patient_id()

        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose project folder",
        )

        if not folder:
            return

        project_path = str(Path(folder) / f"{patient_id}.json")

        # Register the newly created project in recent-project history.
        # It will appear in Open project after the JSON has been written.
        OpenProjectDialog.register_recent_project(project_path)

        self.result_data = {
            "action": "create",
            "project_path": project_path,
            "mode": "edit",
            "patient_id": patient_id,
        }

        self.accept()
