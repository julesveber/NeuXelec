from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QVBoxLayout,
    QWidget,
)


class NeuXelecFileAssignmentHeader(QFrame):
    """Frameless NeuXelec header for the file assignment dialog."""

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


class FileAssignmentDialog(QDialog):
    """
    NeuXelec-styled dialog used after bulk loading files.

    Returns a list of dicts:
        [{"path": "...", "role": "T1"}, ...]
    """

    def __init__(
        self,
        files: Sequence[str],
        roles: Sequence[tuple[str, str]],
        suggestions: dict[str, str] | None = None,
        parent=None,
        title: str = "Assign files",
        subtitle: str = "Choose the modality corresponding to each selected file.",
    ):
        super().__init__(parent)
        self._files = [str(f) for f in files]
        self._roles = [(str(value), str(label)) for value, label in roles]
        self._suggestions = suggestions or {}
        self._combos: list[QComboBox] = []
        self._positioned_once = False

        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # Keep the NeuXelec frameless look, but allow manual resizing.
        # The custom QSizeGrip in the bottom-right corner handles resizing.
        self.setSizeGripEnabled(False)
        self.setMinimumSize(540, 300)
        self.setMaximumSize(16777215, 16777215)

        self._build_ui(title, subtitle)
        self._apply_style()

        # Compute the initial size after the UI is built so it can adapt
        # to the number of selected files and to the current screen.
        self._set_adapted_initial_size()

    def _set_adapted_initial_size(self) -> None:
        """
        Open at a comfortable size depending on the number of selected files,
        without exceeding the available screen.

        Small selections open compactly.
        Large selections open taller and use the scroll area.
        The user can still resize the dialog with the bottom-right grip.
        """
        n_files = max(1, len(self._files))

        try:
            parent = self.parentWidget()
            screen = parent.screen() if parent is not None else None
            if screen is None:
                screen = QGuiApplication.primaryScreen()

            if screen is not None:
                available = screen.availableGeometry()
                screen_w = int(available.width())
                screen_h = int(available.height())
            else:
                screen_w = 1400
                screen_h = 900

            margin = 70
            max_width = max(540, screen_w - 2 * margin)
            max_height = max(300, screen_h - 2 * margin)

            # Width: compact by default, slightly wider for long file names,
            # capped by screen size.
            longest_name = 0
            try:
                longest_name = max(len(Path(f).name) for f in self._files)
            except Exception:
                longest_name = 24

            preferred_width = 680 + min(160, max(0, longest_name - 28) * 5)
            preferred_width = min(preferred_width, 820)

            # Height: title/header/buttons take roughly 205 px.
            # Each file row uses around 44 px.
            preferred_height = 205 + 44 * n_files

            # Keep 1-2 files compact, but never too small.
            preferred_height = max(330, preferred_height)

            # Large selections should not make a huge window:
            # the scroll area will take over.
            preferred_height = min(preferred_height, int(max_height * 0.82))

            final_width = min(max(self.minimumWidth(), preferred_width), max_width)
            final_height = min(max(self.minimumHeight(), preferred_height), max_height)

            self.resize(int(final_width), int(final_height))

        except Exception:
            self.resize(700, max(330, min(620, 205 + 44 * n_files)))

    def _build_ui(self, title: str, subtitle: str) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.dialog_shell = QFrame()
        self.dialog_shell.setObjectName("dialogShell")
        outer.addWidget(self.dialog_shell)

        root = QVBoxLayout(self.dialog_shell)
        root.setContentsMargins(14, 8, 14, 14)
        root.setSpacing(10)

        self.custom_header = NeuXelecFileAssignmentHeader(self)
        root.addWidget(self.custom_header)

        self.lbl_title = QLabel(str(title).upper())
        self.lbl_title.setObjectName("dialogTitle")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        root.addWidget(self.lbl_title)

        self.lbl_subtitle = QLabel(str(subtitle))
        self.lbl_subtitle.setObjectName("dialogSubtitle")
        self.lbl_subtitle.setAlignment(Qt.AlignCenter)
        self.lbl_subtitle.setWordWrap(True)
        root.addWidget(self.lbl_subtitle)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("assignmentScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("assignmentScrollContent")
        grid = QGridLayout(self.scroll_content)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)

        header_file = QLabel("FILE")
        header_file.setObjectName("columnHeader")
        header_role = QLabel("ASSIGN TO")
        header_role.setObjectName("columnHeader")
        grid.addWidget(header_file, 0, 0)
        grid.addWidget(header_role, 0, 1)

        role_values = [value for value, _label in self._roles]
        role_labels = [label for _value, label in self._roles]

        for row, path in enumerate(self._files, start=1):
            file_label = QLabel(Path(path).name)
            file_label.setObjectName("fileNameLabel")
            file_label.setToolTip(path)
            file_label.setMinimumHeight(34)

            combo = QComboBox()
            combo.setObjectName("assignmentCombo")
            combo.setMinimumHeight(36)
            combo.addItems(role_labels)

            suggested = self._suggestions.get(path, "IGNORE")
            try:
                idx = role_values.index(suggested)
            except ValueError:
                idx = role_values.index("IGNORE") if "IGNORE" in role_values else 0
            combo.setCurrentIndex(idx)

            self._combos.append(combo)
            grid.addWidget(file_label, row, 0)
            grid.addWidget(combo, row, 1)

        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)

        self.scroll_area.setWidget(self.scroll_content)
        root.addWidget(self.scroll_area, 1)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(12, 0, 12, 4)
        buttons.setSpacing(10)
        buttons.addStretch(1)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("secondaryButton")
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setMinimumHeight(42)
        self.btn_cancel.setMinimumWidth(112)
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_import = QPushButton("Import")
        self.btn_import.setObjectName("primaryButton")
        self.btn_import.setCursor(Qt.PointingHandCursor)
        self.btn_import.setMinimumHeight(42)
        self.btn_import.setMinimumWidth(112)
        self.btn_import.clicked.connect(self.accept)

        buttons.addWidget(self.btn_cancel)
        buttons.addWidget(self.btn_import)
        root.addLayout(buttons)

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

    def assignments(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        values = [value for value, _label in self._roles]
        for path, combo in zip(self._files, self._combos):
            idx = int(combo.currentIndex())
            role = values[idx] if 0 <= idx < len(values) else "IGNORE"
            if role != "IGNORE":
                out.append({"path": str(path), "role": str(role)})
        return out

    @classmethod
    def get_assignments(
        cls,
        files: Sequence[str],
        roles: Sequence[tuple[str, str]],
        suggestions: dict[str, str] | None = None,
        parent=None,
        title: str = "Assign files",
        subtitle: str = "Choose the modality corresponding to each selected file.",
    ) -> list[dict[str, str]] | None:
        dlg = cls(
            files, roles, suggestions=suggestions, parent=parent, title=title, subtitle=subtitle
        )
        if dlg.exec() != QDialog.Accepted:
            return None
        return dlg.assignments()

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QDialog { background: transparent; }
            QFrame#dialogShell {
                background-color: #06070D;
                border: 1.5px solid #FF487D;
                border-radius: 16px;
            }
            QFrame#customDialogHeader { background: transparent; border: none; }
            QPushButton#closeWindowButton {
                color: #D8DAE4;
                background: transparent;
                border: 1px solid transparent;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 700;
                padding: 0px;
            }
            QPushButton#closeWindowButton:hover {
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FF8000, stop:1 #FF00A0);
            }
            QLabel { color: #F2F2F5; background: transparent; border: none; }
            QLabel#dialogTitle {
                color: #F4D9D0;
                font-size: 18px;
                font-weight: 600;
                letter-spacing: 1px;
            }
            QLabel#dialogSubtitle { color: #9398A8; font-size: 12px; }
            QLabel#columnHeader {
                color: #717786;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QLabel#fileNameLabel {
                color: #D5D7E1;
                background-color: #10121A;
                border: 1px solid #242734;
                border-radius: 8px;
                padding-left: 10px;
                padding-right: 10px;
            }
            QScrollArea#assignmentScrollArea,
            QWidget#assignmentScrollContent {
                background: transparent;
                border: none;
            }
            QComboBox#assignmentCombo {
                color: #F2F2F5;
                background-color: #171922;
                border: 1px solid #2B2D38;
                border-radius: 8px;
                padding: 6px 28px 6px 10px;
                font-size: 12px;
                font-weight: 600;
            }
            QComboBox#assignmentCombo:hover { border: 1px solid #FF487D; }
            QComboBox#assignmentCombo::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 26px;
                border-left: 1px solid #2B2D38;
                background-color: #171922;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QComboBox#assignmentCombo::down-arrow {
                image: url(resources/images/spin_down.svg);
                width: 9px;
                height: 9px;
            }
            QComboBox#assignmentCombo QAbstractItemView {
                color: #F2F2F5;
                background-color: #10121A;
                border: 1px solid #FF487D;
                selection-background-color: #FF487D;
                selection-color: white;
                outline: none;
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
            QPushButton#primaryButton {
                color: white;
                border: none;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FF8000, stop:1 #FF00A0);
            }
            QPushButton#primaryButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FF922B, stop:1 #FF33B8);
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
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical { height: 0px; border: none; background: transparent; }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical { background: transparent; border: none; }
            QSizeGrip#dialogResizeGrip { background: transparent; border: none; image: none; }
            """)
