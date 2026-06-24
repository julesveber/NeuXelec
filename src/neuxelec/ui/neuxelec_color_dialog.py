from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QVBoxLayout,
    QWidget,
)


class NeuXelecColorDialogHeader(QFrame):
    """
    Frameless NeuXelec header for the reusable color picker.
    Drag the empty header area to move the dialog.
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


class NeuXelecColorDialog(QDialog):
    """
    Reusable NeuXelec-styled color picker.

    Usage:
        color_hex = NeuXelecColorDialog.get_color(
            initial_color="#FF3B30",
            parent=self,
            title="Choose marker color",
        )

        if color_hex is not None:
            ...
    """

    def __init__(
        self,
        initial_color: str | QColor = "#FF3B30",
        parent=None,
        title: str = "Choose color",
    ):
        super().__init__(parent)

        self._positioned_once = False

        if isinstance(initial_color, QColor):
            initial = initial_color
        else:
            initial = QColor(str(initial_color))

        if not initial.isValid():
            initial = QColor("#FF3B30")

        self.setWindowTitle(title)
        self.setModal(True)

        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setSizeGripEnabled(False)

        self._set_adapted_initial_size()

        self._build_ui(initial)
        self._apply_style()

    def _set_adapted_initial_size(self) -> None:
        """
        Open large enough to avoid cropping, while fitting on smaller screens.
        The content is scrollable if the user makes the dialog smaller.
        """
        preferred_width = 690
        preferred_height = 535

        min_width = 520
        min_height = 420

        try:
            parent = self.parentWidget()
            screen = parent.screen() if parent is not None else None

            if screen is None:
                screen = QGuiApplication.primaryScreen()

            if screen is None:
                self.setMinimumSize(min_width, min_height)
                self.resize(preferred_width, preferred_height)
                return

            available = screen.availableGeometry()
            margin = 50

            max_width = max(min_width, available.width() - margin * 2)
            max_height = max(min_height, available.height() - margin * 2)

            self.setMinimumSize(min_width, min_height)
            self.resize(
                min(preferred_width, max_width),
                min(preferred_height, max_height),
            )

        except Exception:
            self.setMinimumSize(min_width, min_height)
            self.resize(preferred_width, preferred_height)

    def _build_ui(self, initial: QColor) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.dialog_shell = QFrame()
        self.dialog_shell.setObjectName("dialogShell")

        shell_layout = QVBoxLayout(self.dialog_shell)
        shell_layout.setContentsMargins(12, 6, 12, 12)
        shell_layout.setSpacing(8)

        outer_layout.addWidget(self.dialog_shell)

        self.custom_header = NeuXelecColorDialogHeader(self)
        shell_layout.addWidget(self.custom_header)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("colorDialogScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("colorDialogScrollContent")

        content_layout = QVBoxLayout(self.scroll_content)
        content_layout.setContentsMargins(8, 4, 8, 8)
        content_layout.setSpacing(0)
        content_layout.setAlignment(Qt.AlignCenter)

        self.color_dialog = QColorDialog(initial, self.scroll_content)
        self.color_dialog.setObjectName("embeddedColorDialog")
        self.color_dialog.setOption(QColorDialog.DontUseNativeDialog, True)
        self.color_dialog.setOption(QColorDialog.ShowAlphaChannel, False)

        # IMPORTANT:
        # We hide the native QColorDialog OK/Cancel buttons.
        # The NeuXelec parent dialog below is the only one that accepts/rejects.
        self.color_dialog.setOption(QColorDialog.NoButtons, True)

        self.color_dialog.setWindowFlags(Qt.Widget)

        # The native Qt color widget has a natural minimum size. Keeping this
        # avoids clipped controls at normal size. The surrounding scroll area
        # preserves access if the user resizes the popup smaller.
        self.color_dialog.setMinimumSize(600, 410)
        self.color_dialog.setMaximumSize(620, 430)

        content_layout.addWidget(self.color_dialog)

        self.scroll_area.setWidget(self.scroll_content)
        shell_layout.addWidget(self.scroll_area, 1)

        # ---------------------------------------------------------
        # NeuXelec bottom buttons
        # ---------------------------------------------------------
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(12, 0, 12, 4)
        buttons_layout.setSpacing(10)
        buttons_layout.addStretch(1)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("secondaryButton")
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setMinimumHeight(40)
        self.btn_cancel.setMinimumWidth(104)
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_ok = QPushButton("OK")
        self.btn_ok.setObjectName("primaryButton")
        self.btn_ok.setCursor(Qt.PointingHandCursor)
        self.btn_ok.setMinimumHeight(40)
        self.btn_ok.setMinimumWidth(104)
        self.btn_ok.clicked.connect(self.accept)

        buttons_layout.addWidget(self.btn_cancel)
        buttons_layout.addWidget(self.btn_ok)

        shell_layout.addLayout(buttons_layout)

        self.resize_grip = QSizeGrip(self.dialog_shell)
        self.resize_grip.setObjectName("dialogResizeGrip")
        self.resize_grip.setFixedSize(16, 16)
        self.resize_grip.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

        try:
            margin = 4
            self.resize_grip.move(
                self.dialog_shell.width() - self.resize_grip.width() - margin,
                self.dialog_shell.height() - self.resize_grip.height() - margin,
            )
            self.resize_grip.raise_()
        except Exception:
            pass

    def showEvent(self, event) -> None:
        super().showEvent(event)

        if self._positioned_once:
            return

        self._positioned_once = True

        try:
            parent = self.parentWidget()

            if parent is not None:
                geometry = self.frameGeometry()
                geometry.moveCenter(parent.frameGeometry().center())
                self.move(geometry.topLeft())
                return

            screen = QGuiApplication.primaryScreen()

            if screen is not None:
                geometry = self.frameGeometry()
                geometry.moveCenter(screen.availableGeometry().center())
                self.move(geometry.topLeft())

        except Exception:
            pass

    def selected_color(self) -> QColor | None:
        color = self.color_dialog.currentColor()

        if not color.isValid():
            return None

        return color

    def selected_hex(self) -> str | None:
        color = self.selected_color()

        if color is None:
            return None

        return color.name().upper()

    @staticmethod
    def get_color(
        initial_color: str | QColor = "#FF3B30",
        parent=None,
        title: str = "Choose color",
    ) -> str | None:
        dialog = NeuXelecColorDialog(
            initial_color=initial_color,
            parent=parent,
            title=title,
        )
        dialog.setWindowTitle(title)

        if dialog.exec() != QDialog.Accepted:
            return None

        return dialog.selected_hex()

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

            QSizeGrip#dialogResizeGrip {
                background-color: transparent;
                border: none;
                image: none;
            }

            QScrollArea#colorDialogScrollArea,
            QWidget#colorDialogScrollContent {
                background-color: transparent;
                border: none;
            }

            QScrollArea#colorDialogScrollArea > QWidget > QWidget {
                background-color: transparent;
            }

            QColorDialog#embeddedColorDialog,
            QColorDialog#embeddedColorDialog QWidget {
                background-color: #06070D;
                color: #F2F2F5;
            }

            QColorDialog#embeddedColorDialog QLabel {
                color: #F2F2F5;
                background-color: transparent;
                border: none;
                font-size: 12px;
            }

            QColorDialog#embeddedColorDialog QLineEdit,
            QColorDialog#embeddedColorDialog QSpinBox {
                background-color: #171922;
                color: #F2F2F5;
                border: 1px solid #2B2D38;
                border-radius: 6px;
                padding: 4px 24px 4px 8px;
                selection-background-color: #FF008F;
            }

            QColorDialog#embeddedColorDialog QSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 18px;
                background-color: #171922;
                border-left: 1px solid #2B2D38;
                border-top-right-radius: 6px;
            }

            QColorDialog#embeddedColorDialog QSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 18px;
                background-color: #171922;
                border-left: 1px solid #2B2D38;
                border-bottom-right-radius: 6px;
            }

            QColorDialog#embeddedColorDialog QSpinBox::up-button:hover,
            QColorDialog#embeddedColorDialog QSpinBox::down-button:hover {
                background-color: #20222B;
                border-left: 1px solid #FF487D;
            }

            QColorDialog#embeddedColorDialog QSpinBox::up-arrow {
                image: url(resources/images/spin_up.svg);
                width: 8px;
                height: 8px;
            }

            QColorDialog#embeddedColorDialog QSpinBox::down-arrow {
                image: url(resources/images/spin_down.svg);
                width: 8px;
                height: 8px;
            }

            QColorDialog#embeddedColorDialog QLineEdit:hover,
            QColorDialog#embeddedColorDialog QSpinBox:hover {
                border: 1px solid #3A3D4A;
            }

            QColorDialog#embeddedColorDialog QLineEdit:focus,
            QColorDialog#embeddedColorDialog QSpinBox:focus {
                border: 1px solid #FF487D;
            }

            QColorDialog#embeddedColorDialog QPushButton {
                color: white;
                background-color: #17181F;
                border: 1px solid #2B2D38;
                border-radius: 8px;
                min-height: 30px;
                padding-left: 12px;
                padding-right: 12px;
                font-size: 12px;
                font-weight: 600;
            }

            QColorDialog#embeddedColorDialog QPushButton:hover {
                background-color: #20222B;
                border: 1px solid #FF487D;
            }

            QColorDialog#embeddedColorDialog QPushButton:pressed {
                background-color: #14151B;
                border: 1px solid #FF487D;
            }

            QColorDialog#embeddedColorDialog QDialogButtonBox QPushButton {
                min-width: 82px;
                min-height: 34px;
            }

            QPushButton#secondaryButton,
            QPushButton#primaryButton {
                min-height: 40px;
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

            QScrollBar:vertical {
                background-color: transparent;
                border: none;
                width: 10px;
                margin: 5px 2px 5px 2px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical {
                min-height: 28px;
                background-color: #3B3E48;
                border: 1px solid #FF487D;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #4A4D58;
                border: 1px solid #FF6B98;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
                width: 0px;
                border: none;
                background: transparent;
            }

            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
                border: none;
            }

            QScrollBar:horizontal {
                background-color: transparent;
                border: none;
                height: 10px;
                margin: 2px 5px 2px 5px;
                border-radius: 5px;
            }

            QScrollBar::handle:horizontal {
                min-width: 28px;
                background-color: #3B3E48;
                border: 1px solid #FF487D;
                border-radius: 5px;
            }

            QScrollBar::handle:horizontal:hover {
                background-color: #4A4D58;
                border: 1px solid #FF6B98;
            }

            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                height: 0px;
                width: 0px;
                border: none;
                background: transparent;
            }

            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: transparent;
                border: none;
            }
            """)
