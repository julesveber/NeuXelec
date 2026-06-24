from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QSettings, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
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


class OpenProjectDialog(QDialog):
    """
    Dialog used to open an existing NeuXelec JSON project.

    Features:
        - displays recently opened or created JSON projects;
        - allows browsing for another JSON project;
        - enables Edit mode / Visualization only only after selection;
        - stores recent project paths locally with QSettings.

    The project JSON itself is never modified by this dialog.
    """

    SETTINGS_ORGANIZATION = "NeuXelec"
    SETTINGS_APPLICATION = "NeuXelec"
    RECENT_PROJECTS_KEY = "recent_projects/json_paths"
    MAX_RECENT_PROJECTS = 30

    def __init__(self, parent=None):
        super().__init__(parent)

        self.selected_project_path: str | None = None
        self.selected_mode: str | None = None
        self._positioned_once = False

        # Invisible interactive border used to resize the frameless window.
        self._resize_margin = 8

        self.setWindowTitle("Open project")

        # Frameless rounded NeuXelec dialog.
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # Use an internal transparent grip instead of the native light grip.
        self.setSizeGripEnabled(False)

        # Default opening size remains unchanged, but the user can reduce
        # the window further if needed on a smaller screen.
        self.setMinimumSize(500, 400)
        self._set_adapted_initial_size()

        self._build_ui()
        self._apply_style()
        self._load_recent_projects()
        self._update_mode_buttons_enabled()

    # ============================================================
    # UI
    # ============================================================
    def _set_adapted_initial_size(self) -> None:
        """
        Keep the current default opening size when possible, while ensuring
        that the dialog remains accessible on smaller screens.
        """
        # Keep these values: this is your current preferred default size.
        preferred_width = 670
        preferred_height = 640

        try:
            parent = self.parentWidget()
            screen = parent.screen() if parent is not None else None

            if screen is None:
                screen = QGuiApplication.primaryScreen()

            if screen is None:
                self.resize(preferred_width, preferred_height)
                return

            available = screen.availableGeometry()
            screen_margin = 45

            max_width = max(
                self.minimumWidth(),
                available.width() - (screen_margin * 2),
            )
            max_height = max(
                self.minimumHeight(),
                available.height() - (screen_margin * 2),
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
        """
        Detect whether the mouse is over one of the resize borders or corners
        of the frameless rounded window.
        """
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
        """
        Show the appropriate resize cursor above each border or corner.
        """
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
        """
        Start the native Qt/Windows resize action without restoring the
        system title bar.
        """
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
            if hasattr(self, "dialog_shell") and obj is self.dialog_shell:
                if event.type() == QEvent.MouseMove:
                    edges = self._resize_edges_at_position(event.position().toPoint())
                    self._update_resize_cursor(edges)

                elif event.type() == QEvent.MouseButtonPress:
                    if event.button() == Qt.MouseButton.LeftButton:
                        edges = self._resize_edges_at_position(event.position().toPoint())

                        if self._start_native_resize(edges):
                            event.accept()
                            return True

                elif event.type() == QEvent.Leave:
                    self.dialog_shell.unsetCursor()

        except Exception:
            pass

        return super().eventFilter(obj, event)

    def _build_ui(self) -> None:
        # ============================================================
        # Transparent dialog and rounded NeuXelec shell
        # ============================================================
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.dialog_shell = QFrame()
        self.dialog_shell.setObjectName("dialogShell")

        # Manual resizing from every border and corner.
        self.dialog_shell.setMouseTracking(True)
        self.dialog_shell.installEventFilter(self)

        shell_layout = QVBoxLayout(self.dialog_shell)
        shell_layout.setContentsMargins(14, 8, 14, 14)
        shell_layout.setSpacing(8)

        outer_layout.addWidget(self.dialog_shell)

        # ---------------------------------------------------------
        # Custom frameless header
        # ---------------------------------------------------------
        self.custom_header = NeuXelecDialogHeader(self)
        shell_layout.addWidget(self.custom_header)

        # ============================================================
        # Scrollable central content
        # ============================================================
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("openProjectScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("openProjectScrollContent")
        self.scroll_content.setMinimumHeight(462)

        layout = QVBoxLayout(self.scroll_content)
        layout.setContentsMargins(20, 4, 20, 10)
        layout.setSpacing(12)

        self.lbl_title = QLabel("OPEN PROJECT")
        self.lbl_title.setObjectName("dialogTitle")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_title)

        self.lbl_subtitle = QLabel("Select a recent project or browse for another JSON file")
        self.lbl_subtitle.setObjectName("dialogSubtitle")
        self.lbl_subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_subtitle)

        layout.addSpacing(8)

        self.lbl_recent = QLabel("Recent projects")
        self.lbl_recent.setObjectName("sectionLabel")
        self.lbl_recent.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_recent)

        # ---------------------------------------------------------
        # Recent project table
        # ---------------------------------------------------------
        self.table_recent = QTableWidget(0, 2)
        self.table_recent.setObjectName("recentProjectsTable")
        self.table_recent.setHorizontalHeaderLabels(["Project name", "Last modified"])

        name_header = self.table_recent.horizontalHeaderItem(0)
        date_header = self.table_recent.horizontalHeaderItem(1)

        if name_header is not None:
            name_header.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        if date_header is not None:
            date_header.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.table_recent.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_recent.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_recent.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_recent.setShowGrid(False)
        self.table_recent.setAlternatingRowColors(False)

        self.table_recent.setVerticalScrollMode(QAbstractItemView.ScrollPerItem)

        self.table_recent.verticalHeader().setVisible(False)
        self.table_recent.verticalHeader().setDefaultSectionSize(38)
        self.table_recent.verticalHeader().setMinimumSectionSize(38)

        # Four full project rows remain visible by default.
        self.table_recent.setFixedHeight(212)

        header = self.table_recent.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)

        self.table_recent.itemSelectionChanged.connect(self._on_recent_project_selected)

        layout.addWidget(self.table_recent)

        self.lbl_empty_history = QLabel(
            "No recent project found. Browse for an existing JSON project file."
        )
        self.lbl_empty_history.setObjectName("emptyHistoryLabel")
        self.lbl_empty_history.setAlignment(Qt.AlignCenter)
        self.lbl_empty_history.hide()
        layout.addWidget(self.lbl_empty_history)

        layout.addSpacing(6)

        self.btn_browse = QPushButton("Browse another project...")
        self.btn_browse.setObjectName("secondaryButton")
        self.btn_browse.setCursor(Qt.PointingHandCursor)
        self.btn_browse.clicked.connect(self._browse_project)
        layout.addWidget(self.btn_browse)

        layout.addSpacing(6)

        # ---------------------------------------------------------
        # Selected project card
        # ---------------------------------------------------------
        self.selected_frame = QFrame()
        self.selected_frame.setObjectName("selectedProjectFrame")
        self.selected_frame.setProperty("projectSelected", False)

        selected_layout = QVBoxLayout(self.selected_frame)
        selected_layout.setContentsMargins(14, 10, 14, 10)
        selected_layout.setSpacing(4)

        self.lbl_selected_title = QLabel("Selected project")
        self.lbl_selected_title.setObjectName("selectedTitle")

        self.lbl_selected_project = QLabel("No project selected")
        self.lbl_selected_project.setObjectName("selectedProjectLabel")
        self.lbl_selected_project.setTextInteractionFlags(Qt.TextSelectableByMouse)

        selected_layout.addWidget(self.lbl_selected_title)
        selected_layout.addWidget(self.lbl_selected_project)

        layout.addWidget(self.selected_frame)
        layout.addStretch(1)

        self.scroll_area.setWidget(self.scroll_content)
        shell_layout.addWidget(self.scroll_area, 1)

        # ============================================================
        # Fixed bottom mode buttons
        # ============================================================
        mode_layout = QHBoxLayout()
        mode_layout.setContentsMargins(20, 0, 20, 4)
        mode_layout.setSpacing(10)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("secondaryButton")
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setMinimumHeight(42)
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_visualization = QPushButton("Visualization only")
        self.btn_visualization.setObjectName("secondaryButton")
        self.btn_visualization.setCursor(Qt.PointingHandCursor)
        self.btn_visualization.setMinimumHeight(42)
        self.btn_visualization.clicked.connect(lambda: self._accept_with_mode("visualization"))

        self.btn_edit = QPushButton("Edit mode")
        self.btn_edit.setObjectName("primaryButton")
        self.btn_edit.setCursor(Qt.PointingHandCursor)
        self.btn_edit.setMinimumHeight(42)
        self.btn_edit.setMinimumWidth(108)
        self.btn_edit.clicked.connect(lambda: self._accept_with_mode("edit"))

        mode_layout.addWidget(self.btn_cancel)
        mode_layout.addStretch(1)
        mode_layout.addWidget(self.btn_visualization)
        mode_layout.addWidget(self.btn_edit)

        shell_layout.addLayout(mode_layout)

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

            QWidget#openProjectScrollContent {
                background-color: transparent;
            }

            QScrollArea#openProjectScrollArea {
                background-color: transparent;
                border: none;
            }

            QScrollArea#openProjectScrollArea > QWidget > QWidget {
                background-color: transparent;
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
            }

            QLabel#sectionLabel {
                color: #F2F2F5;
                font-size: 13px;
                font-weight: 600;
                padding-bottom: 2px;
            }

            QLabel#emptyHistoryLabel {
                color: #747782;
                font-size: 12px;
                padding: 4px;
            }

            QTableWidget#recentProjectsTable {
                background-color: #10121A;
                alternate-background-color: #10121A;
                color: #F2F2F5;
                border: 1px solid #2B2D38;
                border-radius: 12px;
                outline: none;
                padding: 4px;
                padding-bottom: 8px;
                selection-background-color: #35202F;
                selection-color: white;
            }

            QTableWidget#recentProjectsTable::item {
                border: none;
                padding: 8px;
            }

            QTableWidget#recentProjectsTable::item:selected {
                background-color: #35202F;
                color: white;
                border-bottom: 1px solid #FF487D;
            }

            QHeaderView::section {
                background-color: #171922;
                color: #A6A8B2;
                border: none;
                border-bottom: 1px solid #2B2D38;
                padding: 9px;
                font-size: 12px;
                font-weight: 600;
            }

            QFrame#selectedProjectFrame {
                background-color: #10121A;
                border: 1px solid #2B2D38;
                border-radius: 10px;
            }

            QFrame#selectedProjectFrame[projectSelected="true"] {
                background-color: #10121A;
                border: 1px solid #FF487D;
                border-radius: 10px;
            }

            QLabel#selectedTitle {
                color: #8E8E98;
                font-size: 11px;
                font-weight: 500;
            }

            QLabel#selectedProjectLabel {
                color: #F2F2F5;
                font-size: 13px;
                font-weight: 600;
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
                background-color: #17181F;
                color: white;
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

            QPushButton#secondaryButton:disabled {
                background-color: #121319;
                color: #62646E;
                border: 1px solid #20222A;
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

            QPushButton#primaryButton:disabled {
                color: #85858C;
                background: #24252C;
                border: 1px solid #292B34;
            }

            QScrollBar:vertical {
                background-color: #111218;
                width: 14px;
                margin: 5px 3px 8px 2px;
                border: none;
                border-radius: 6px;
            }

            QScrollBar::handle:vertical {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
                border-radius: 6px;
                min-height: 18px;
            }

            QScrollBar::handle:vertical:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF922B,
                    stop:1 #FF33B8
                );
                border-radius: 6px;
            }

            QScrollBar::handle:vertical:pressed {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #E56F00,
                    stop:1 #E0008C
                );
                border-radius: 6px;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
                background: transparent;
                border: none;
            }

            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
            }
            """)

    # ============================================================
    # Recent project storage
    # ============================================================

    @classmethod
    def _settings(cls) -> QSettings:
        return QSettings(
            cls.SETTINGS_ORGANIZATION,
            cls.SETTINGS_APPLICATION,
        )

    @classmethod
    def _read_recent_paths(cls) -> list[str]:
        settings = cls._settings()
        raw_value = settings.value(cls.RECENT_PROJECTS_KEY, "[]")

        try:
            if isinstance(raw_value, str):
                paths = json.loads(raw_value)
            elif isinstance(raw_value, list):
                paths = raw_value
            else:
                paths = []
        except Exception:
            paths = []

        cleaned_paths = []
        seen = set()

        for raw_path in paths:
            try:
                path = str(Path(str(raw_path)).resolve())
            except Exception:
                continue

            if path in seen:
                continue

            seen.add(path)
            cleaned_paths.append(path)

        return cleaned_paths

    @classmethod
    def _write_recent_paths(cls, paths: list[str]) -> None:
        settings = cls._settings()
        settings.setValue(
            cls.RECENT_PROJECTS_KEY,
            json.dumps(paths[: cls.MAX_RECENT_PROJECTS]),
        )
        settings.sync()

    @classmethod
    def register_recent_project(cls, project_path: str) -> None:
        """
        Add a project path to local recent-history storage.

        The most recently selected project is placed first.
        """
        try:
            path = str(Path(str(project_path)).resolve())
        except Exception:
            return

        existing = cls._read_recent_paths()
        existing = [p for p in existing if p != path]
        existing.insert(0, path)

        cls._write_recent_paths(existing)

    def _existing_recent_projects(self) -> list[Path]:
        valid_paths = []

        for raw_path in self._read_recent_paths():
            path = Path(raw_path)

            if path.exists() and path.is_file() and path.suffix.lower() == ".json":
                valid_paths.append(path)

        valid_paths.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Automatically clean paths that no longer exist.
        self._write_recent_paths([str(path) for path in valid_paths])

        return valid_paths

    # ============================================================
    # Selection / opening
    # ============================================================

    def _load_recent_projects(self) -> None:
        projects = self._existing_recent_projects()

        self.table_recent.setRowCount(0)

        for row, path in enumerate(projects):
            self.table_recent.insertRow(row)
            self.table_recent.setRowHeight(row, 38)

            name_item = QTableWidgetItem(path.stem)
            date_item = QTableWidgetItem(
                datetime.fromtimestamp(path.stat().st_mtime).strftime("%d/%m/%Y  %H:%M")
            )

            name_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            date_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

            name_item.setData(Qt.UserRole, str(path))
            date_item.setData(Qt.UserRole, str(path))

            name_item.setToolTip(str(path))
            date_item.setToolTip(str(path))

            self.table_recent.setItem(row, 0, name_item)
            self.table_recent.setItem(row, 1, date_item)

        has_projects = len(projects) > 0
        self.table_recent.setVisible(has_projects)
        self.lbl_empty_history.setVisible(not has_projects)

    def _set_selected_project_frame_active(self, active: bool) -> None:
        """
        Highlight the Selected project area when a valid JSON project
        has been selected.
        """
        try:
            self.selected_frame.setProperty("projectSelected", bool(active))

            self.selected_frame.style().unpolish(self.selected_frame)
            self.selected_frame.style().polish(self.selected_frame)
            self.selected_frame.update()

        except Exception:
            pass

    def _on_recent_project_selected(self) -> None:
        selected_items = self.table_recent.selectedItems()

        if not selected_items:
            self.selected_project_path = None
            self.lbl_selected_project.setText("No project selected")
            self.lbl_selected_project.setToolTip("")
            self._set_selected_project_frame_active(False)
            self._update_mode_buttons_enabled()
            return

        path = selected_items[0].data(Qt.UserRole)

        if not path:
            return

        self._select_project_path(str(path))

    def _select_project_path(self, project_path: str) -> None:
        path = Path(str(project_path))

        if not path.exists() or path.suffix.lower() != ".json":
            self.selected_project_path = None
            self.lbl_selected_project.setText("No project selected")
            self.lbl_selected_project.setToolTip("")
            self._set_selected_project_frame_active(False)
            self._update_mode_buttons_enabled()
            return

        self.selected_project_path = str(path)
        self.lbl_selected_project.setText(path.stem)
        self.lbl_selected_project.setToolTip(str(path))

        # Same pink outline as the focused Patient ID field.
        self._set_selected_project_frame_active(True)

        self._update_mode_buttons_enabled()

    def _browse_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a NeuXelec project",
            "",
            "NeuXelec Project (*.json)",
        )

        if not path:
            return

        selected_path = Path(path)

        if not selected_path.exists():
            return

        # Display it immediately in the history table.
        self.register_recent_project(str(selected_path))
        self._load_recent_projects()

        # Select the corresponding row.
        for row in range(self.table_recent.rowCount()):
            item = self.table_recent.item(row, 0)

            if item is not None and item.data(Qt.UserRole) == str(selected_path):
                self.table_recent.selectRow(row)
                break

        self._select_project_path(str(selected_path))

    def _update_mode_buttons_enabled(self) -> None:
        enabled = bool(self.selected_project_path)

        self.btn_edit.setEnabled(enabled)
        self.btn_visualization.setEnabled(enabled)

    def _accept_with_mode(self, mode: str) -> None:
        if not self.selected_project_path:
            return

        self.selected_mode = str(mode)

        # Move opened project to the top of recent history.
        self.register_recent_project(self.selected_project_path)

        self.accept()
