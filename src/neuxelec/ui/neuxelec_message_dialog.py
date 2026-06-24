from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from neuxelec.utils.resources import resource_path


class NeuXelecMessageHeader(QFrame):
    """
    Frameless NeuXelec header shared by information, warning, error
    and confirmation dialogs.
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


class NeuXelecMessageDialog(QDialog):
    """
    Generic NeuXelec-styled dialog replacing QMessageBox throughout the app.

    Supported uses:
        - information(...)
        - warning(...)
        - critical(...)
        - question(...)
        - choice(...)
    """

    def __init__(
        self,
        title: str,
        message: str,
        parent=None,
        kind: str = "information",
    ):
        super().__init__(parent)

        self._title = str(title)
        self._message = str(message)
        self._kind = str(kind)
        self._selected_value: str | None = None
        self._positioned_once = False
        self._resize_margin = 8

        self.setWindowTitle(self._title)
        self.setModal(True)

        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setSizeGripEnabled(False)

        self.setMinimumSize(430, 245)
        self._set_adapted_initial_size()

        self._build_ui()
        self._apply_style()

    # ============================================================
    # Public helper constructors
    # ============================================================

    @classmethod
    def information(
        cls,
        parent,
        title: str,
        message: str,
        button_text: str = "OK",
    ) -> None:
        dlg = cls(title, message, parent=parent, kind="information")
        dlg._add_button(button_text, primary=True, accepted=True)
        dlg.exec()

    @classmethod
    def warning(
        cls,
        parent,
        title: str,
        message: str,
        button_text: str = "OK",
    ) -> None:
        dlg = cls(title, message, parent=parent, kind="warning")
        dlg._add_button(button_text, primary=True, accepted=True)
        dlg.exec()

    @classmethod
    def critical(
        cls,
        parent,
        title: str,
        message: str,
        button_text: str = "Close",
    ) -> None:
        dlg = cls(title, message, parent=parent, kind="critical")
        dlg._add_button(button_text, primary=True, accepted=True)
        dlg.exec()

    @classmethod
    def question(
        cls,
        parent,
        title: str,
        message: str,
        accept_text: str = "Confirm",
        reject_text: str = "Cancel",
    ) -> bool:
        dlg = cls(title, message, parent=parent, kind="question")
        dlg._add_button(reject_text, primary=False, accepted=False)
        dlg._add_button(accept_text, primary=True, accepted=True)
        return dlg.exec() == QDialog.Accepted

    @classmethod
    def choice(
        cls,
        parent,
        title: str,
        message: str,
        choices: Sequence[tuple[str, str, bool]],
        cancel_text: str = "Cancel",
    ) -> str | None:
        """
        choices:
            [
                ("returned_value", "Button text", is_primary),
                ...
            ]
        """
        dlg = cls(title, message, parent=parent, kind="question")

        dlg._add_button(cancel_text, primary=False, accepted=False)

        for value, label, primary in choices:
            dlg._add_choice_button(
                text=label,
                value=value,
                primary=bool(primary),
            )

        dlg.exec()
        return dlg._selected_value

    # ============================================================
    # Geometry and resize
    # ============================================================

    def _set_adapted_initial_size(self) -> None:
        preferred_width = 570

        message_lines = self._message.count("\n") + 1
        preferred_height = min(
            460,
            max(285, 250 + (message_lines * 18)),
        )

        try:
            parent = self.parentWidget()
            screen = parent.screen() if parent is not None else None

            if screen is None:
                screen = QGuiApplication.primaryScreen()

            if screen is None:
                self.resize(preferred_width, preferred_height)
                return

            available = screen.availableGeometry()
            margin = 45

            max_width = max(
                self.minimumWidth(),
                available.width() - margin * 2,
            )
            max_height = max(
                self.minimumHeight(),
                available.height() - margin * 2,
            )

            self.resize(
                min(preferred_width, max_width),
                min(preferred_height, max_height),
            )

        except Exception:
            self.resize(preferred_width, preferred_height)

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

            if screen is not None:
                available = screen.availableGeometry()

                x = max(
                    available.left(),
                    min(
                        geometry.left(),
                        available.right() - geometry.width() + 1,
                    ),
                )
                y = max(
                    available.top(),
                    min(
                        geometry.top(),
                        available.bottom() - geometry.height() + 1,
                    ),
                )

                self.move(x, y)
            else:
                self.move(geometry.topLeft())

        except Exception:
            pass

    def _resize_edges_at_position(self, pos: QPoint):
        if not hasattr(self, "dialog_shell"):
            return Qt.Edge(0)

        rect = self.dialog_shell.rect()
        margin = int(self._resize_margin)

        on_left = pos.x() <= margin
        on_right = pos.x() >= rect.width() - margin
        on_top = pos.y() <= margin
        on_bottom = pos.y() >= rect.height() - margin

        edges = Qt.Edge(0)

        if on_left:
            edges |= Qt.Edge.LeftEdge
        if on_right:
            edges |= Qt.Edge.RightEdge
        if on_top:
            edges |= Qt.Edge.TopEdge
        if on_bottom:
            edges |= Qt.Edge.BottomEdge

        return edges

    def _update_resize_cursor(self, edges) -> None:
        if edges == (Qt.Edge.LeftEdge | Qt.Edge.TopEdge) or edges == (
            Qt.Edge.RightEdge | Qt.Edge.BottomEdge
        ):
            self.dialog_shell.setCursor(Qt.CursorShape.SizeFDiagCursor)

        elif edges == (Qt.Edge.RightEdge | Qt.Edge.TopEdge) or edges == (
            Qt.Edge.LeftEdge | Qt.Edge.BottomEdge
        ):
            self.dialog_shell.setCursor(Qt.CursorShape.SizeBDiagCursor)

        elif edges & (Qt.Edge.LeftEdge | Qt.Edge.RightEdge):
            self.dialog_shell.setCursor(Qt.CursorShape.SizeHorCursor)

        elif edges & (Qt.Edge.TopEdge | Qt.Edge.BottomEdge):
            self.dialog_shell.setCursor(Qt.CursorShape.SizeVerCursor)

        else:
            self.dialog_shell.unsetCursor()

    def _start_native_resize(self, edges) -> bool:
        if not edges:
            return False

        try:
            handle = self.windowHandle()

            if handle is None:
                return False

            return bool(handle.startSystemResize(edges))

        except Exception:
            return False

    def eventFilter(self, obj, event):
        try:
            if obj is self.dialog_shell:
                if event.type() == QEvent.MouseMove:
                    edges = self._resize_edges_at_position(event.position().toPoint())
                    self._update_resize_cursor(edges)

                elif event.type() == QEvent.MouseButtonPress:
                    if event.button() == Qt.LeftButton:
                        edges = self._resize_edges_at_position(event.position().toPoint())

                        if self._start_native_resize(edges):
                            event.accept()
                            return True

                elif event.type() == QEvent.Leave:
                    self.dialog_shell.unsetCursor()

        except Exception:
            pass

        return super().eventFilter(obj, event)

    # ============================================================
    # UI
    # ============================================================

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.dialog_shell = QFrame()
        self.dialog_shell.setObjectName("dialogShell")
        self.dialog_shell.setProperty("kind", self._kind)
        self.dialog_shell.setMouseTracking(True)
        self.dialog_shell.installEventFilter(self)

        root = QVBoxLayout(self.dialog_shell)
        root.setContentsMargins(14, 8, 14, 14)
        root.setSpacing(10)

        outer_layout.addWidget(self.dialog_shell)

        self.custom_header = NeuXelecMessageHeader(self)
        root.addWidget(self.custom_header)

        self.lbl_title = QLabel(self._title.upper())
        self.lbl_title.setObjectName("dialogTitle")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        root.addWidget(self.lbl_title)

        self.message_frame = QFrame()
        self.message_frame.setObjectName("messageCard")

        message_layout = QVBoxLayout(self.message_frame)
        message_layout.setContentsMargins(14, 12, 14, 12)
        message_layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("messageScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        message_content = QWidget()
        message_content.setObjectName("messageScrollContent")

        message_content_layout = QVBoxLayout(message_content)
        message_content_layout.setContentsMargins(0, 0, 0, 0)
        message_content_layout.setSpacing(0)

        self.lbl_message = QLabel(self._message)
        self.lbl_message.setObjectName("messageLabel")
        self.lbl_message.setWordWrap(True)
        self.lbl_message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_message.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )

        message_content_layout.addWidget(self.lbl_message)
        message_content_layout.addStretch(1)

        self.scroll_area.setWidget(message_content)
        message_layout.addWidget(self.scroll_area)

        root.addWidget(self.message_frame, 1)

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setContentsMargins(12, 0, 12, 4)
        self.buttons_layout.setSpacing(10)
        self.buttons_layout.addStretch(1)

        root.addLayout(self.buttons_layout)

    def _add_button(
        self,
        text: str,
        primary: bool,
        accepted: bool,
    ) -> None:
        button = QPushButton(text)
        button.setObjectName("primaryButton" if primary else "secondaryButton")
        button.setCursor(Qt.PointingHandCursor)
        button.setMinimumHeight(42)
        button.setMinimumWidth(108)

        if accepted:
            button.clicked.connect(self.accept)
        else:
            button.clicked.connect(self.reject)

        self.buttons_layout.addWidget(button)

    def _add_choice_button(
        self,
        text: str,
        value: str,
        primary: bool,
    ) -> None:
        button = QPushButton(text)
        button.setObjectName("primaryButton" if primary else "secondaryButton")
        button.setCursor(Qt.PointingHandCursor)
        button.setMinimumHeight(42)

        def _choose() -> None:
            self._selected_value = str(value)
            self.accept()

        button.clicked.connect(_choose)
        self.buttons_layout.addWidget(button)

    # ============================================================
    # Style
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
            }

            QPushButton#closeWindowButton:hover {
                color: white;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
            }

            QPushButton#closeWindowButton:pressed {
                color: white;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #E56F00,
                    stop:1 #E0008C
                );
            }

            QLabel {
                color: #F2F2F5;
                background-color: transparent;
                border: none;
            }

            QLabel#dialogTitle {
                color: #F4D9D0;
                font-size: 18px;
                font-weight: 500;
                letter-spacing: 1px;
                padding-bottom: 4px;
            }

            QFrame#messageCard {
                background-color: #10121A;
                border: 1px solid #2B2D38;
                border-radius: 10px;
            }

            QScrollArea#messageScrollArea,
            QWidget#messageScrollContent {
                background-color: transparent;
                border: none;
            }

            QLabel#messageLabel {
                color: #CFCFD6;
                font-size: 12px;
                font-weight: 500;
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

            QScrollArea#messageScrollArea QScrollBar:vertical {
                background-color: #111218;
                width: 11px;
                margin: 4px 2px 4px 2px;
                border: none;
                border-radius: 5px;
            }

            QScrollArea#messageScrollArea QScrollBar::handle:vertical {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
                border-radius: 5px;
                min-height: 22px;
            }

            QScrollArea#messageScrollArea QScrollBar::add-line:vertical,
            QScrollArea#messageScrollArea QScrollBar::sub-line:vertical {
                height: 0px;
                background: transparent;
                border: none;
            }

            QScrollArea#messageScrollArea QScrollBar::add-page:vertical,
            QScrollArea#messageScrollArea QScrollBar::sub-page:vertical {
                background: transparent;
            }
            """)


class NeuXelecSelectionDialog(NeuXelecMessageDialog):
    """
    NeuXelec-styled dialog used to select one item from a list.
    Replaces native QInputDialog.getItem() popups.
    """

    def __init__(
        self,
        title: str,
        message: str,
        options: list[str],
        current_index: int = 0,
        parent=None,
        accept_text: str = "Apply",
        reject_text: str = "Cancel",
    ):
        self._options = [str(option) for option in options]

        super().__init__(
            title=title,
            message=message,
            parent=parent,
            kind="question",
        )

        self.setMinimumSize(470, 285)

        self.combo_selection = QComboBox()
        self.combo_selection.setObjectName("selectionCombo")
        self.combo_selection.setMinimumHeight(40)
        self.combo_selection.addItems(self._options)

        if self._options:
            safe_index = max(
                0,
                min(int(current_index), len(self._options) - 1),
            )
            self.combo_selection.setCurrentIndex(safe_index)

        try:
            message_layout = self.message_frame.layout()
            message_layout.addSpacing(10)
            message_layout.addWidget(self.combo_selection)
        except Exception:
            pass

        self._add_button(
            reject_text,
            primary=False,
            accepted=False,
        )
        self._add_button(
            accept_text,
            primary=True,
            accepted=True,
        )

        icon_dir = resource_path("resources/images")

        spin_down_path = (icon_dir / "spin_down.svg").as_posix()

        selection_style = """
            QComboBox#selectionCombo {
                color: #F2F2F5;
                background-color: #151720;
                border: 1px solid #2B2D38;
                border-radius: 9px;
                padding: 8px 38px 8px 12px;
                font-size: 12px;
                font-weight: 500;
            }

            QComboBox#selectionCombo:hover {
                border: 1px solid #3B3D48;
            }

            QComboBox#selectionCombo:focus,
            QComboBox#selectionCombo:on {
                border: 1px solid #FF487D;
                background-color: #181A24;
            }

            QComboBox#selectionCombo::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 34px;
                background-color: #1D1F29;
                border: none;
                border-left: 1px solid #2B2D38;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
            }

            QComboBox#selectionCombo::drop-down:hover {
                background-color: #282A34;
            }

            QComboBox#selectionCombo::drop-down:pressed {
                background-color: #343641;
            }

            QComboBox#selectionCombo::down-arrow {
                image: url(__SPIN_DOWN__);
                width: 12px;
                height: 8px;
            }

            QComboBox#selectionCombo QAbstractItemView {
                color: #F2F2F5;
                background-color: #10121A;
                border: 1px solid #FF487D;
                selection-background-color: #3A2132;
                selection-color: white;
                outline: none;
                padding: 4px;
            }
        """

        selection_style = selection_style.replace(
            "__SPIN_DOWN__",
            spin_down_path,
        )

        self.setStyleSheet(self.styleSheet() + selection_style)

    @classmethod
    def select_item(
        cls,
        parent,
        title: str,
        message: str,
        options: list[str],
        current_index: int = 0,
        accept_text: str = "Apply",
        reject_text: str = "Cancel",
    ) -> str | None:
        dlg = cls(
            title=title,
            message=message,
            options=options,
            current_index=current_index,
            parent=parent,
            accept_text=accept_text,
            reject_text=reject_text,
        )

        if dlg.exec() != QDialog.Accepted:
            return None

        return str(dlg.combo_selection.currentText())


class NeuXelecTextInputDialog(NeuXelecMessageDialog):
    """
    NeuXelec-styled text input dialog.
    Replaces native QInputDialog.getText() popups.
    """

    def __init__(
        self,
        title: str,
        message: str,
        initial_text: str = "",
        parent=None,
        accept_text: str = "Apply",
        reject_text: str = "Cancel",
    ):
        super().__init__(
            title=title,
            message=message,
            parent=parent,
            kind="question",
        )

        self.setMinimumSize(470, 285)

        self.text_input = QLineEdit()
        self.text_input.setObjectName("textInput")
        self.text_input.setMinimumHeight(40)
        self.text_input.setText(str(initial_text))
        self.text_input.selectAll()

        try:
            message_layout = self.message_frame.layout()
            message_layout.addSpacing(10)
            message_layout.addWidget(self.text_input)
        except Exception:
            pass

        self._add_button(
            reject_text,
            primary=False,
            accepted=False,
        )
        self._add_button(
            accept_text,
            primary=True,
            accepted=True,
        )

        self.text_input.returnPressed.connect(self.accept)

        self.setStyleSheet(self.styleSheet() + """
            QLineEdit#textInput {
                color: #F2F2F5;
                background-color: #151720;
                border: 1px solid #2B2D38;
                border-radius: 9px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: 500;
            }

            QLineEdit#textInput:hover {
                border: 1px solid #FF487D;
            }

            QLineEdit#textInput:focus {
                border: 1px solid #FF487D;
                background-color: #181A24;
            }
            """)

    @classmethod
    def get_text(
        cls,
        parent,
        title: str,
        message: str,
        initial_text: str = "",
        accept_text: str = "Apply",
        reject_text: str = "Cancel",
    ) -> str | None:
        dlg = cls(
            title=title,
            message=message,
            initial_text=initial_text,
            parent=parent,
            accept_text=accept_text,
            reject_text=reject_text,
        )

        if dlg.exec() != QDialog.Accepted:
            return None

        return str(dlg.text_input.text())


class NeuXelecColorDialog(QDialog):
    """
    NeuXelec-styled color selection dialog.
    Replaces native QColorDialog popups while keeping Qt's color picker.
    """

    def __init__(
        self,
        title: str,
        initial_color: QColor,
        parent=None,
    ):
        super().__init__(parent)

        self._title = str(title)
        self._initial_color = QColor(initial_color)
        self._positioned_once = False

        self.setWindowTitle(self._title)
        self.setModal(True)

        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setSizeGripEnabled(False)

        self.setMinimumSize(660, 560)
        self.resize(720, 610)

        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.dialog_shell = QFrame()
        self.dialog_shell.setObjectName("colorDialogShell")

        root = QVBoxLayout(self.dialog_shell)
        root.setContentsMargins(14, 8, 14, 16)
        root.setSpacing(10)

        outer_layout.addWidget(self.dialog_shell)

        self.custom_header = NeuXelecMessageHeader(self)
        root.addWidget(self.custom_header)

        self.lbl_title = QLabel(self._title.upper())
        self.lbl_title.setObjectName("colorDialogTitle")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        root.addWidget(self.lbl_title)

        self.picker_frame = QFrame()
        self.picker_frame.setObjectName("colorPickerCard")

        picker_layout = QVBoxLayout(self.picker_frame)
        picker_layout.setContentsMargins(10, 10, 10, 10)
        picker_layout.setSpacing(0)

        self.color_picker = QColorDialog(self._initial_color, self)
        self.color_picker.setObjectName("embeddedColorPicker")
        self.color_picker.setOption(QColorDialog.DontUseNativeDialog, True)
        self.color_picker.setOption(QColorDialog.NoButtons, True)

        picker_layout.addWidget(self.color_picker)
        root.addWidget(self.picker_frame, 1)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(12, 0, 12, 0)
        buttons.setSpacing(10)
        buttons.addStretch(1)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("secondaryButton")
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setMinimumSize(108, 42)
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setObjectName("primaryButton")
        self.btn_apply.setCursor(Qt.PointingHandCursor)
        self.btn_apply.setMinimumSize(108, 42)
        self.btn_apply.clicked.connect(self.accept)

        buttons.addWidget(self.btn_cancel)
        buttons.addWidget(self.btn_apply)

        root.addLayout(buttons)

    def selected_color(self) -> QColor:
        return QColor(self.color_picker.currentColor())

    @classmethod
    def get_color(
        cls,
        parent,
        title: str,
        initial_color: QColor,
    ) -> QColor | None:
        dlg = cls(
            title=title,
            initial_color=initial_color,
            parent=parent,
        )

        if dlg.exec() != QDialog.Accepted:
            return None

        color = dlg.selected_color()

        if not color.isValid():
            return None

        return color

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

        except Exception:
            pass

    def _apply_style(self) -> None:
        icon_dir = resource_path("resources/images")

        spin_up_path = (icon_dir / "spin_up.svg").as_posix()
        spin_down_path = (icon_dir / "spin_down.svg").as_posix()

        style = """
            QDialog {
                background: transparent;
            }

            QFrame#colorDialogShell {
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
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
            }

            QPushButton#closeWindowButton:pressed {
                color: white;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #E56F00,
                    stop:1 #E0008C
                );
            }

            QLabel#colorDialogTitle {
                color: #F4D9D0;
                background-color: transparent;
                border: none;
                font-size: 18px;
                font-weight: 500;
                letter-spacing: 1px;
                padding-bottom: 4px;
            }

            QFrame#colorPickerCard {
                background-color: #10121A;
                border: 1px solid #2B2D38;
                border-radius: 10px;
            }

            QColorDialog#embeddedColorPicker {
                color: #F2F2F5;
                background-color: #10121A;
            }

            QColorDialog#embeddedColorPicker QLabel {
                color: #D8DAE4;
                background-color: transparent;
                border: none;
            }

            /* Native color dialog action buttons */
            QColorDialog#embeddedColorPicker QPushButton {
                color: #F2F2F5;
                background-color: #17181F;
                border: 1px solid #2B2D38;
                border-radius: 6px;
                padding: 4px 10px;
                min-height: 22px;
            }

            QColorDialog#embeddedColorPicker QPushButton:hover {
                border: 1px solid #FF487D;
                background-color: #20222B;
            }

            /* HTML hexadecimal field */
            QColorDialog#embeddedColorPicker QLineEdit {
                color: #F2F2F5;
                background-color: #151720;
                border: 1px solid #2B2D38;
                border-radius: 7px;
                padding: 6px 10px;
                selection-background-color: #FF487D;
                selection-color: white;
            }

            QColorDialog#embeddedColorPicker QLineEdit:hover {
                border: 1px solid #3B3D48;
            }

            QColorDialog#embeddedColorPicker QLineEdit:focus {
                border: 1px solid #FF487D;
                background-color: #181A24;
            }

            /* Hue / Sat / Val / Red / Green / Blue numeric fields */
            QColorDialog#embeddedColorPicker QSpinBox {
                color: #F2F2F5;
                background-color: #151720;
                border: 1px solid #2B2D38;
                border-radius: 7px;
                padding: 5px 28px 5px 10px;
                min-height: 24px;
                min-width: 74px;
                selection-background-color: #FF487D;
                selection-color: white;
            }

            QColorDialog#embeddedColorPicker QSpinBox:hover {
                border: 1px solid #3B3D48;
            }

            QColorDialog#embeddedColorPicker QSpinBox:focus {
                border: 1px solid #FF487D;
                background-color: #181A24;
            }

            /*
            The editor contained inside each QSpinBox must not draw its own
            outline, otherwise the rose contour appears fragmented.
            */
            QColorDialog#embeddedColorPicker QSpinBox QLineEdit {
                color: #F2F2F5;
                background-color: transparent;
                border: none;
                border-radius: 0px;
                padding: 0px;
                selection-background-color: #FF487D;
                selection-color: white;
            }

            QColorDialog#embeddedColorPicker QSpinBox QLineEdit:hover,
            QColorDialog#embeddedColorPicker QSpinBox QLineEdit:focus {
                background-color: transparent;
                border: none;
            }

            QColorDialog#embeddedColorPicker QSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 23px;
                background-color: #1D1F29;
                border: none;
                border-left: 1px solid #2B2D38;
                border-bottom: 1px solid #2B2D38;
                border-top-right-radius: 6px;
                margin: 1px 1px 0px 0px;
            }

            QColorDialog#embeddedColorPicker QSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 23px;
                background-color: #1D1F29;
                border: none;
                border-left: 1px solid #2B2D38;
                border-bottom-right-radius: 6px;
                margin: 0px 1px 1px 0px;
            }

            QColorDialog#embeddedColorPicker QSpinBox::up-button:hover,
            QColorDialog#embeddedColorPicker QSpinBox::down-button:hover {
                background-color: #282A34;
            }

            QColorDialog#embeddedColorPicker QSpinBox::up-button:pressed,
            QColorDialog#embeddedColorPicker QSpinBox::down-button:pressed {
                background-color: #343641;
            }

            QColorDialog#embeddedColorPicker QSpinBox::up-arrow {
                image: url(__SPIN_UP__);
                width: 12px;
                height: 8px;
            }

            QColorDialog#embeddedColorPicker QSpinBox::down-arrow {
                image: url(__SPIN_DOWN__);
                width: 12px;
                height: 8px;
            }

            QPushButton#secondaryButton,
            QPushButton#primaryButton {
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
        """

        style = style.replace("__SPIN_UP__", spin_up_path)
        style = style.replace("__SPIN_DOWN__", spin_down_path)

        self.setStyleSheet(style)
