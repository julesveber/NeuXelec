from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class NeuXelecDialogHeader(QFrame):
    """
    Minimal frameless-window header with a custom close button.
    The empty header area can be dragged to move the dialog.
    """

    def __init__(self, dialog: QDialog, parent=None):
        super().__init__(parent)

        self.dialog = dialog
        self._drag_offset = None

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
        self.btn_close_window.clicked.connect(self.dialog.close)

        layout.addWidget(self.btn_close_window, 0, Qt.AlignRight | Qt.AlignTop)

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


class EditContactDialog(QDialog):
    """
    Styled dialog used to manually edit a contact position in LPS coordinates.

    Public attributes preserved for compatibility with the current application:
        - editX, editY, editZ
        - btnPick
        - btnOk
        - set_lps()
        - get_lps()
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._positioned_once = False

        self.setWindowTitle("Edit Contact Coordinates")

        # Frameless dialog with transparent background so rounded corners
        # are actually visible.
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # Disable the native white size grip in the corner.
        self.setSizeGripEnabled(False)

        # The dialog can be reduced, but remains large enough to keep
        # coordinate fields readable. A scrollbar appears if needed.
        self.setMinimumSize(390, 300)
        self._set_adapted_initial_size()

        self._build_ui()
        self._apply_style()

    def _set_adapted_initial_size(self) -> None:
        """
        Open the dialog at a comfortable size on a regular screen while
        automatically fitting it inside smaller displays.
        """
        preferred_width = 510
        preferred_height = 510

        try:
            screen = None

            parent = self.parentWidget()
            if parent is not None:
                screen = parent.screen()

            if screen is None:
                screen = QGuiApplication.primaryScreen()

            if screen is None:
                self.resize(preferred_width, preferred_height)
                return

            available = screen.availableGeometry()

            # Keep visible margins around the dialog and leave space for
            # the operating-system title bar and task bar.
            max_width = max(390, int(available.width()) - 80)
            max_height = max(300, int(available.height()) - 100)

            initial_width = min(preferred_width, max_width)
            initial_height = min(preferred_height, max_height)

            self.resize(initial_width, initial_height)

        except Exception:
            self.resize(preferred_width, preferred_height)

    def showEvent(self, event) -> None:
        super().showEvent(event)

        if self._positioned_once:
            return

        self._positioned_once = True

        try:
            parent = self.parentWidget()

            if parent is None:
                return

            geometry = self.frameGeometry()
            geometry.moveCenter(parent.frameGeometry().center())
            self.move(geometry.topLeft())

        except Exception:
            pass

    # ============================================================
    # UI
    # ============================================================

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Outer rounded shell
        self.dialog_shell = QFrame()
        self.dialog_shell.setObjectName("dialogShell")

        shell_layout = QVBoxLayout(self.dialog_shell)
        shell_layout.setContentsMargins(14, 8, 14, 14)
        shell_layout.setSpacing(8)

        root.addWidget(self.dialog_shell)

        # ---------------------------------------------------------
        # Custom frameless header
        # ---------------------------------------------------------
        self.custom_header = NeuXelecDialogHeader(self)
        shell_layout.addWidget(self.custom_header)

        # ---------------------------------------------------------
        # Scrollable content
        # ---------------------------------------------------------
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("editCoordinatesScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("editCoordinatesScrollContent")

        # Preserve sufficient space for the title, the three complete
        # coordinate fields and the crosshair button.
        self.scroll_content.setMinimumHeight(355)

        content = QVBoxLayout(self.scroll_content)
        content.setContentsMargins(12, 8, 12, 10)
        content.setSpacing(14)

        # ---------------------------------------------------------
        # Header
        # ---------------------------------------------------------
        self.lbl_title = QLabel("EDIT COORDINATES")
        self.lbl_title.setObjectName("dialogTitle")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        content.addWidget(self.lbl_title)

        self.lbl_subtitle = QLabel("Modify the contact location in anatomical space")
        self.lbl_subtitle.setObjectName("dialogSubtitle")
        self.lbl_subtitle.setAlignment(Qt.AlignCenter)
        content.addWidget(self.lbl_subtitle)

        # ---------------------------------------------------------
        # Coordinate fields
        # ---------------------------------------------------------
        self.coordinates_group = QFrame()
        self.coordinates_group.setObjectName("coordinatesCard")
        self.coordinates_group.setMinimumHeight(188)

        coordinates_layout = QVBoxLayout(self.coordinates_group)
        coordinates_layout.setContentsMargins(16, 12, 16, 14)
        coordinates_layout.setSpacing(10)

        self.lbl_coordinate_system = QLabel("LPS coordinates  ·  millimeters")
        self.lbl_coordinate_system.setObjectName("sectionLabel")
        coordinates_layout.addWidget(self.lbl_coordinate_system)

        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 0)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(10)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.editX = self._make_coordinate_field("Left / Right coordinate")
        self.editY = self._make_coordinate_field("Posterior / Anterior coordinate")
        self.editZ = self._make_coordinate_field("Superior / Inferior coordinate")

        self.lbl_x = QLabel("X")
        self.lbl_y = QLabel("Y")
        self.lbl_z = QLabel("Z")

        for label in (self.lbl_x, self.lbl_y, self.lbl_z):
            label.setObjectName("axisLabel")
            label.setFixedWidth(24)

        form.addRow(self.lbl_x, self._coordinate_row(self.editX, "mm"))
        form.addRow(self.lbl_y, self._coordinate_row(self.editY, "mm"))
        form.addRow(self.lbl_z, self._coordinate_row(self.editZ, "mm"))

        coordinates_layout.addLayout(form)
        content.addWidget(self.coordinates_group)

        # ---------------------------------------------------------
        # Crosshair picking action
        # ---------------------------------------------------------
        self.btnPick = QPushButton("Pick with crosshair")
        self.btnPick.setObjectName("secondaryButton")
        self.btnPick.setCursor(Qt.PointingHandCursor)
        self.btnPick.setMinimumHeight(42)
        content.addWidget(self.btnPick)

        content.addStretch(1)

        self.scroll_area.setWidget(self.scroll_content)
        shell_layout.addWidget(self.scroll_area, 1)

        # ---------------------------------------------------------
        # Fixed bottom validation action
        # ---------------------------------------------------------
        bottom = QHBoxLayout()
        bottom.setContentsMargins(10, 0, 10, 2)
        bottom.setSpacing(10)

        bottom.addStretch(1)

        self.btnOk = QPushButton("Validate")
        self.btnOk.setObjectName("primaryButton")
        self.btnOk.setCursor(Qt.PointingHandCursor)
        self.btnOk.setMinimumHeight(42)
        self.btnOk.setMinimumWidth(124)

        bottom.addWidget(self.btnOk)

        shell_layout.addLayout(bottom)

    def _make_coordinate_field(self, tooltip: str) -> QLineEdit:
        field = QLineEdit()
        field.setObjectName("coordinateField")
        field.setFixedHeight(42)
        field.setMinimumWidth(250)
        field.setToolTip(tooltip)
        field.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        field.setPlaceholderText("0.00")
        return field

    def _coordinate_row(self, field: QLineEdit, unit: str) -> QWidget:
        widget = QWidget()
        widget.setObjectName("coordinateRow")

        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(9)

        unit_label = QLabel(unit)
        unit_label.setObjectName("unitLabel")
        unit_label.setFixedWidth(26)

        layout.addWidget(field, 1)
        layout.addWidget(unit_label)

        return widget

    # ============================================================
    # Existing public API
    # ============================================================

    def set_lps(self, xyz) -> None:
        self.editX.setText(f"{float(xyz[0]):.2f}")
        self.editY.setText(f"{float(xyz[1]):.2f}")
        self.editZ.setText(f"{float(xyz[2]):.2f}")

    def get_lps(self):
        return (
            float(self.editX.text().strip().replace(",", ".")),
            float(self.editY.text().strip().replace(",", ".")),
            float(self.editZ.text().strip().replace(",", ".")),
        )

    # ============================================================
    # Styling
    # ============================================================

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
                text-align: center;
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

            QWidget#editCoordinatesScrollContent {
                background-color: transparent;
            }

            QScrollArea#editCoordinatesScrollArea {
                background-color: transparent;
                border: none;
            }

            QScrollArea#editCoordinatesScrollArea > QWidget > QWidget {
                background-color: transparent;
            }

            QLabel {
                color: #F2F2F5;
                font-size: 12px;
                border: none;
                background: transparent;
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
                padding-bottom: 2px;
            }

            QFrame#coordinatesCard {
                background-color: #10121A;
                border: 1px solid #2B2D38;
                border-radius: 12px;
            }

            QLabel#sectionLabel {
                color: #F2F2F5;
                font-size: 12px;
                font-weight: 600;
                padding-bottom: 3px;
            }

            QLabel#axisLabel {
                color: #FF487D;
                font-size: 13px;
                font-weight: 700;
            }

            QLabel#unitLabel {
                color: #A6A8B2;
                font-size: 12px;
                font-weight: 500;
            }

            QLineEdit#coordinateField {
                min-height: 42px;
                background-color: #171922;
                color: #F2F2F5;
                border: 1px solid #2B2D38;
                border-radius: 9px;
                padding-left: 12px;
                padding-right: 12px;
                selection-background-color: #FF008F;
                font-size: 13px;
                font-weight: 500;
            }

            QLineEdit#coordinateField:hover {
                border: 1px solid #3A3D4A;
            }

            QLineEdit#coordinateField:focus {
                border: 1px solid #FF487D;
            }

            QPushButton {
                min-height: 42px;
                border-radius: 10px;
                font-size: 13px;
                font-weight: 600;
                padding-left: 16px;
                padding-right: 16px;
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

            QScrollArea#editCoordinatesScrollArea QScrollBar:vertical {
                background-color: #111218;
                width: 13px;
                margin: 5px 2px 5px 2px;
                border: none;
                border-radius: 6px;
            }

            QScrollArea#editCoordinatesScrollArea QScrollBar::handle:vertical {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
                border-radius: 6px;
                min-height: 22px;
            }

            QScrollArea#editCoordinatesScrollArea QScrollBar::handle:vertical:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF922B,
                    stop:1 #FF33B8
                );
            }

            QScrollArea#editCoordinatesScrollArea QScrollBar::handle:vertical:pressed {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #E56F00,
                    stop:1 #E0008C
                );
            }

            QScrollArea#editCoordinatesScrollArea QScrollBar::add-line:vertical,
            QScrollArea#editCoordinatesScrollArea QScrollBar::sub-line:vertical {
                height: 0px;
                background: transparent;
                border: none;
            }

            QScrollArea#editCoordinatesScrollArea QScrollBar::add-page:vertical,
            QScrollArea#editCoordinatesScrollArea QScrollBar::sub-page:vertical {
                background: transparent;
            }
            """)
