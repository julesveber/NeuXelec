from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from PySide6.QtCore import (
    QEvent,
    QItemSelection,
    QItemSelectionModel,
    QPoint,
    Qt,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QGuiApplication,
    QKeySequence,
    QLinearGradient,
    QPen,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
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
        self.btn_close_window.clicked.connect(self.dialog.close)

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


class ParcellationRowDelegate(QStyledItemDelegate):
    """
    Paint hover and selection as a continuous full-row effect
    instead of a cell-by-cell fragmented effect.
    """

    def __init__(self, table, dialog):
        super().__init__(table)
        self.table = table
        self.dialog = dialog

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        row = index.row()
        col = index.column()
        model = index.model()

        selected = self.table.selectionModel().isSelected(index)
        hovered = self.dialog._hover_row == row

        first_index = model.index(row, 0)
        last_index = model.index(row, model.columnCount() - 1)

        first_rect = self.table.visualRect(first_index)
        last_rect = self.table.visualRect(last_index)
        row_rect = first_rect.united(last_rect)

        # Leave one pixel between rows so multiple selections remain readable.
        cell_rect = option.rect.adjusted(0, 0, 0, -1)

        painter.save()

        # ---------------------------------------------------------
        # Continuous full-row selection background
        # ---------------------------------------------------------
        if selected:
            gradient = QLinearGradient(
                row_rect.left(),
                row_rect.top(),
                row_rect.right(),
                row_rect.top(),
            )

            # NeuXelec gradient with 50% opacity.
            gradient.setColorAt(0.0, QColor(255, 128, 0, 128))
            gradient.setColorAt(1.0, QColor(255, 0, 160, 128))

            # Each cell uses the same row-based gradient coordinates:
            # visually, the gradient is continuous across the complete row.
            painter.fillRect(cell_rect, gradient)

            pen = QPen(QColor(255, 72, 125, 190))
            pen.setWidth(1)
            painter.setPen(pen)

            painter.drawLine(cell_rect.topLeft(), cell_rect.topRight())
            painter.drawLine(cell_rect.bottomLeft(), cell_rect.bottomRight())

            if col == 0:
                painter.drawLine(cell_rect.topLeft(), cell_rect.bottomLeft())

            if col == model.columnCount() - 1:
                painter.drawLine(cell_rect.topRight(), cell_rect.bottomRight())

        # ---------------------------------------------------------
        # Hover: outline only, without colored background
        # ---------------------------------------------------------
        elif hovered:
            pen = QPen(QColor("#FF487D"))
            pen.setWidth(1)
            painter.setPen(pen)

            painter.drawLine(cell_rect.topLeft(), cell_rect.topRight())
            painter.drawLine(cell_rect.bottomLeft(), cell_rect.bottomRight())

            if col == 0:
                painter.drawLine(cell_rect.topLeft(), cell_rect.bottomLeft())

            if col == model.columnCount() - 1:
                painter.drawLine(cell_rect.topRight(), cell_rect.bottomRight())

        painter.restore()

        # Disable the standard Qt hover and selection rendering.
        # Our delegate already paints the complete row appearance.
        opt.state &= ~QStyle.State_Selected
        opt.state &= ~QStyle.State_MouseOver
        opt.backgroundBrush = QBrush(Qt.NoBrush)

        super().paint(painter, opt, index)


class MniParcellationTableDialog(QDialog):
    """
    Floating, non-modal table showing where imported MNI electrodes fall
    in MNI parcellations.

    Expected owner:
        View3DPage instance with:
            - state.mni_electrode_sets
            - _get_mni_parcellation1_img_and_lut()
            - _get_mni_parcellation2_img_and_lut()
    """

    COLUMNS = [
        "subject",
        "source_file",
        "electrode",
        "contact",
        "hemisphere",
        "type",
        "x_mni",
        "y_mni",
        "z_mni",
        "parcellation1_atlas",
        "parcellation1_index",
        "parcellation1_label",
        "parcellation2_atlas",
        "parcellation2_index",
        "parcellation2_label",
    ]

    def __init__(self, owner, parent=None):
        super().__init__(parent)

        self.owner = owner
        self.rows: list[dict[str, Any]] = []
        self.parcellation1_name = (
            "Schaefer 2018 - 400 cortical parcels, 7 networks (MNI152NLin2009cAsym, 1mm)"
        )
        self.parcellation2_name = "Brainnetome - 246 regions cortex + subcortex, connectivity-based (MNI152NLin2009cAsym, 1mm)"
        self._positioned_once = False
        self._hover_row = -1
        # Width of the interactive resize zone around the frameless window.
        self._resize_margin = 8
        self._is_system_resizing = False
        self.setWindowTitle("MNI parcellation table")
        self.setWindowModality(Qt.NonModal)

        # Frameless rounded NeuXelec dialog.
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # Use a custom transparent grip instead of the native white corner.
        self.setSizeGripEnabled(False)

        # The window adapts to small screens but remains readable.
        # The table scrollbars preserve access to all columns and rows.
        self.setMinimumSize(560, 460)
        self._set_adapted_initial_size()

        self._build_ui()
        self._apply_style()
        self._setup_shortcuts()
        self.refresh()

    def _set_adapted_initial_size(self) -> None:
        """
        Open the parcellation table at a comfortable size while fitting inside
        the available screen geometry. The table handles smaller dimensions
        with its vertical and horizontal scrollbars.
        """
        preferred_width = 1320
        preferred_height = 720

        try:
            parent = self.parentWidget()
            screen = parent.screen() if parent is not None else None

            if screen is None:
                screen = QGuiApplication.primaryScreen()

            if screen is None:
                self.resize(preferred_width, preferred_height)
                return

            available = screen.availableGeometry()

            screen_margin = 50

            max_width = max(
                self.minimumWidth(),
                available.width() - (screen_margin * 2),
            )
            max_height = max(
                self.minimumHeight(),
                available.height() - (screen_margin * 2),
            )

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
                    min(geometry.left(), available.right() - geometry.width() + 1),
                )
                y = max(
                    available.top(),
                    min(geometry.top(), available.bottom() - geometry.height() + 1),
                )

                self.move(x, y)
            else:
                self.move(geometry.topLeft())

        except Exception:
            pass

    def _resize_edges_at_position(self, pos: QPoint):
        """
        Return the window edges associated with a pointer position situated
        near the outer rounded shell.
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
        Display the appropriate native resize cursor when the pointer is over
        one of the frameless dialog borders.
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
        Ask Qt/Windows to perform the native resize operation while retaining
        the frameless NeuXelec design.
        """
        if not edges:
            return False

        try:
            window_handle = self.windowHandle()

            if window_handle is None:
                return False

            self._hover_row = -1
            self.table.viewport().update()

            return bool(window_handle.startSystemResize(edges))

        except Exception:
            return False

    def eventFilter(self, obj, event):
        try:
            # -----------------------------------------------------
            # Table row hover rendering
            # -----------------------------------------------------
            if hasattr(self, "table") and obj is self.table.viewport():
                if event.type() == QEvent.MouseMove:
                    idx = self.table.indexAt(event.pos())
                    new_hover_row = idx.row() if idx.isValid() else -1

                    if new_hover_row != self._hover_row:
                        self._hover_row = new_hover_row
                        self.table.viewport().update()

                elif event.type() == QEvent.Leave:
                    if self._hover_row != -1:
                        self._hover_row = -1
                        self.table.viewport().update()

            # -----------------------------------------------------
            # Frameless window border resizing
            # -----------------------------------------------------
            elif hasattr(self, "dialog_shell") and obj is self.dialog_shell:
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

    def _setup_shortcuts(self) -> None:
        """
        Keyboard shortcuts for quickly locating an electrode in the table.
        QKeySequence.Find corresponds to Ctrl+F on Windows/Linux and Cmd+F
        on macOS.
        """
        self.shortcut_find = QShortcut(
            QKeySequence(QKeySequence.StandardKey.Find),
            self,
        )
        self.shortcut_find.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.shortcut_find.activated.connect(self._show_search_bar)

    def _show_search_bar(self) -> None:
        self.search_frame.show()
        self.edit_search.setFocus()
        self.edit_search.selectAll()

    def _hide_search_bar(self) -> None:
        self.search_frame.hide()
        self.table.setFocus()

    def _set_search_result(self, text: str, error: bool = False) -> None:
        self.lbl_search_result.setText(text)
        self.lbl_search_result.setProperty("error", error)
        self.lbl_search_result.setVisible(bool(text))

        self.lbl_search_result.style().unpolish(self.lbl_search_result)
        self.lbl_search_result.style().polish(self.lbl_search_result)
        self.lbl_search_result.update()

    def _clear_search(self) -> None:
        self.edit_search.clear()
        self.table.clearSelection()
        self._set_search_result("")
        self.table.viewport().update()
        self.edit_search.setFocus()

    def _find_electrode(self) -> None:
        """
        Search the electrode column.

        An exact match has priority:
            FLG -> selects all FLG contacts.

        If no exact match exists, partial matches are selected:
            FL -> may select FLG, FLH, etc.
        """
        query = self.edit_search.text().strip()

        if not query:
            self._clear_search()
            return

        try:
            electrode_column = self.COLUMNS.index("electrode")
        except ValueError:
            self._set_search_result("Electrode column not found.", error=True)
            return

        search_text = query.casefold()
        exact_rows = []
        partial_rows = []

        for row in range(self.table.rowCount()):
            item = self.table.item(row, electrode_column)
            electrode_name = item.text().strip() if item is not None else ""

            if electrode_name.casefold() == search_text:
                exact_rows.append(row)
            elif search_text in electrode_name.casefold():
                partial_rows.append(row)

        matching_rows = exact_rows if exact_rows else partial_rows

        self.table.clearSelection()

        if not matching_rows:
            self._set_search_result(
                f'No electrode found for "{query}".',
                error=True,
            )
            self.table.viewport().update()
            return

        selection = QItemSelection()

        for row in matching_rows:
            first_index = self.table.model().index(row, 0)
            last_index = self.table.model().index(
                row,
                self.table.columnCount() - 1,
            )
            selection.select(first_index, last_index)

        self.table.selectionModel().select(
            selection,
            QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QItemSelectionModel.SelectionFlag.Rows,
        )

        matched_electrodes = {
            self.table.item(row, electrode_column).text().strip()
            for row in matching_rows
            if self.table.item(row, electrode_column) is not None
        }

        first_item = self.table.item(matching_rows[0], electrode_column)

        if first_item is not None:
            self.table.scrollToItem(
                first_item,
                QAbstractItemView.ScrollHint.PositionAtCenter,
            )

        electrode_count = len(matched_electrodes)
        contact_count = len(matching_rows)

        if electrode_count == 1:
            electrode_name = next(iter(matched_electrodes))
            result_text = f"{electrode_name}: {contact_count} contact(s) selected."
        else:
            result_text = (
                f"{electrode_count} electrodes found · " f"{contact_count} contacts selected."
            )

        self._set_search_result(result_text, error=False)
        self.table.viewport().update()

    def keyPressEvent(self, event) -> None:
        if (
            event.key() == Qt.Key.Key_Escape
            and hasattr(self, "search_frame")
            and self.search_frame.isVisible()
        ):
            self._hide_search_bar()
            event.accept()
            return

        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # ============================================================
        # Transparent outer dialog and rounded NeuXelec shell
        # ============================================================
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.dialog_shell = QFrame()
        self.dialog_shell.setObjectName("dialogShell")
        self.dialog_shell.setMouseTracking(True)
        self.dialog_shell.installEventFilter(self)

        main_layout = QVBoxLayout(self.dialog_shell)
        main_layout.setContentsMargins(14, 8, 14, 14)
        main_layout.setSpacing(10)

        outer_layout.addWidget(self.dialog_shell)

        # ---------------------------------------------------------
        # Custom frameless header
        # ---------------------------------------------------------
        self.custom_header = NeuXelecDialogHeader(self)
        main_layout.addWidget(self.custom_header)

        # ---------------------------------------------------------
        # Title
        # ---------------------------------------------------------
        self.lbl_title = QLabel("MNI PARCELLATION TABLE")
        self.lbl_title.setObjectName("dialogTitle")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.lbl_title)

        self.lbl_subtitle = QLabel("Anatomical labels of imported MNI electrode contacts")
        self.lbl_subtitle.setObjectName("dialogSubtitle")
        self.lbl_subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.lbl_subtitle)

        # ---------------------------------------------------------
        # Electrode search bar - displayed with Ctrl + F
        # ---------------------------------------------------------
        self.search_frame = QFrame()
        self.search_frame.setObjectName("searchCard")
        self.search_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.search_frame.hide()

        search_layout = QVBoxLayout(self.search_frame)
        search_layout.setContentsMargins(14, 10, 14, 10)
        search_layout.setSpacing(8)

        search_header = QHBoxLayout()
        search_header.setSpacing(8)

        self.lbl_search_title = QLabel("Find an electrode")
        self.lbl_search_title.setObjectName("searchTitle")

        self.lbl_search_shortcut = QLabel("Ctrl + F  ·  Esc to hide")
        self.lbl_search_shortcut.setObjectName("searchShortcut")

        search_header.addWidget(self.lbl_search_title)
        search_header.addStretch(1)
        search_header.addWidget(self.lbl_search_shortcut)

        search_layout.addLayout(search_header)

        search_row = QHBoxLayout()
        search_row.setSpacing(8)

        self.edit_search = QLineEdit()
        self.edit_search.setObjectName("searchField")
        self.edit_search.setPlaceholderText("Electrode name, e.g. FLG")
        self.edit_search.setMinimumHeight(40)
        self.edit_search.returnPressed.connect(self._find_electrode)

        self.btn_find = QPushButton("Find")
        self.btn_find.setObjectName("searchButton")
        self.btn_find.setCursor(Qt.PointingHandCursor)
        self.btn_find.setMinimumHeight(40)
        self.btn_find.clicked.connect(self._find_electrode)

        self.btn_clear_search = QPushButton("Clear")
        self.btn_clear_search.setObjectName("searchSecondaryButton")
        self.btn_clear_search.setCursor(Qt.PointingHandCursor)
        self.btn_clear_search.setMinimumHeight(40)
        self.btn_clear_search.clicked.connect(self._clear_search)

        search_row.addWidget(self.edit_search, 1)
        search_row.addWidget(self.btn_find)
        search_row.addWidget(self.btn_clear_search)

        search_layout.addLayout(search_row)

        self.lbl_search_result = QLabel("")
        self.lbl_search_result.setObjectName("searchResultLabel")
        self.lbl_search_result.hide()
        search_layout.addWidget(self.lbl_search_result)

        main_layout.addWidget(self.search_frame)

        # ---------------------------------------------------------
        # Information card
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # Information card
        # ---------------------------------------------------------
        self.info_frame = QFrame()
        self.info_frame.setObjectName("informationCard")
        self.info_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        info_layout = QVBoxLayout(self.info_frame)
        info_layout.setContentsMargins(14, 10, 14, 10)
        info_layout.setSpacing(0)

        self.info_label = QLabel("")
        self.info_label.setObjectName("informationLabel")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)

        main_layout.addWidget(self.info_frame)

        # ---------------------------------------------------------
        # Table
        # ---------------------------------------------------------
        self.table = QTableWidget(self)
        self.table.setObjectName("parcellationTable")
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(False)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setMouseTracking(True)
        self.table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)

        # Keep the table geometry stable while the frameless window is resized.
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        # Important for smooth shrinking:
        # QTableWidget otherwise keeps an internal minimum size hint and can
        # make a frameless window jump when reducing its dimensions.
        self.table.setMinimumSize(0, 0)
        self.table.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Ignored,
        )
        self.table.setMinimumHeight(120)
        self.row_delegate = ParcellationRowDelegate(self.table, self)
        self.table.setItemDelegate(self.row_delegate)
        self.table.viewport().installEventFilter(self)
        self.table.itemSelectionChanged.connect(self.table.viewport().update)

        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(38)
        self.table.verticalHeader().setMinimumSectionSize(38)

        try:
            header = self.table.horizontalHeader()
            header.setStretchLastSection(True)
            header.setDefaultSectionSize(130)
            header.setMinimumSectionSize(90)
            header.setSectionResizeMode(QHeaderView.Interactive)
        except Exception:
            pass

        main_layout.addWidget(self.table, stretch=1)

        # ---------------------------------------------------------
        # Fixed bottom buttons
        # ---------------------------------------------------------
        buttons = QHBoxLayout()
        buttons.setContentsMargins(10, 0, 10, 4)
        buttons.setSpacing(10)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setObjectName("secondaryButton")
        self.btn_refresh.setCursor(Qt.PointingHandCursor)
        self.btn_refresh.setMinimumHeight(42)

        self.btn_export = QPushButton("Export…")
        self.btn_export.setObjectName("primaryButton")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setMinimumHeight(42)
        self.btn_export.setMinimumWidth(116)

        self.btn_close = QPushButton("Close")
        self.btn_close.setObjectName("secondaryButton")
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.setMinimumHeight(42)

        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_export.clicked.connect(self.export_table)
        self.btn_close.clicked.connect(self.close)

        buttons.addWidget(self.btn_refresh)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_close)
        buttons.addWidget(self.btn_export)

        main_layout.addLayout(buttons)

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
                color: #F2F2F5;
                font-size: 12px;
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
                padding-bottom: 4px;
            }

                        QFrame#searchCard {
                background-color: #10121A;
                border: 1px solid #FF487D;
                border-radius: 10px;
            }

            QLabel#searchTitle {
                color: #F2F2F5;
                font-size: 12px;
                font-weight: 600;
            }

            QLabel#searchShortcut {
                color: #737682;
                font-size: 11px;
                font-weight: 500;
            }

            QLineEdit#searchField {
                min-height: 40px;
                background-color: #171922;
                color: #F2F2F5;
                border: 1px solid #2B2D38;
                border-radius: 8px;
                padding-left: 12px;
                padding-right: 12px;
                selection-background-color: #FF008F;
                font-size: 12px;
            }

            QLineEdit#searchField:hover {
                border: 1px solid #3A3D4A;
            }

            QLineEdit#searchField:focus {
                border: 1px solid #FF487D;
            }

            QLabel#searchResultLabel {
                color: #A6A8B2;
                font-size: 11px;
                font-weight: 500;
                padding-top: 2px;
            }

            QLabel#searchResultLabel[error="true"] {
                color: #FF487D;
            }

            QPushButton#searchButton,
            QPushButton#searchSecondaryButton {
                min-height: 40px;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 600;
                padding-left: 16px;
                padding-right: 16px;
            }

            QPushButton#searchButton {
                color: white;
                border: none;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
            }

            QPushButton#searchButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF922B,
                    stop:1 #FF33B8
                );
            }

            QPushButton#searchSecondaryButton {
                color: white;
                background-color: #17181F;
                border: 1px solid #2B2D38;
            }

            QPushButton#searchSecondaryButton:hover {
                background-color: #20222B;
                border: 1px solid #FF487D;
            }

            QFrame#informationCard {
                background-color: #10121A;
                border: 1px solid #2B2D38;
                border-radius: 10px;
            }

            QLabel#informationLabel {
                color: #A6A8B2;
                font-size: 11px;
                font-weight: 500;
                background-color: transparent;
                border: none;
            }

            QTableWidget#parcellationTable {
                background-color: #10121A;
                alternate-background-color: #10121A;
                color: #F2F2F5;
                border: 1px solid #2B2D38;
                border-radius: 12px;
                outline: none;
                padding: 4px;
                selection-background-color: transparent;
                selection-color: white;
            }

            QTableWidget#parcellationTable::item {
                background-color: transparent;
                color: #F2F2F5;
                border: none;
                padding: 8px;
            }

            QHeaderView::section {
                background-color: #171922;
                color: #A6A8B2;
                border: none;
                border-bottom: 1px solid #2B2D38;
                padding: 10px;
                font-size: 12px;
                font-weight: 600;
            }

            QTableCornerButton::section {
                background-color: #171922;
                border: none;
                border-bottom: 1px solid #2B2D38;
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

            QTableWidget#parcellationTable QScrollBar:vertical,
            QTableWidget#parcellationTable QScrollBar:horizontal {
                background-color: #111218;
                border: none;
                border-radius: 6px;
            }

            QTableWidget#parcellationTable QScrollBar:vertical {
                width: 13px;
                margin: 5px 2px 5px 2px;
            }

            QTableWidget#parcellationTable QScrollBar:horizontal {
                height: 13px;
                margin: 2px 5px 2px 5px;
            }

            QTableWidget#parcellationTable QScrollBar::handle:vertical {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
                border-radius: 6px;
                min-height: 22px;
            }

            QTableWidget#parcellationTable QScrollBar::handle:horizontal {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
                border-radius: 6px;
                min-width: 22px;
            }

            QTableWidget#parcellationTable QScrollBar::handle:vertical:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF922B,
                    stop:1 #FF33B8
                );
            }

            QTableWidget#parcellationTable QScrollBar::handle:horizontal:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF922B,
                    stop:1 #FF33B8
                );
            }

            QTableWidget#parcellationTable QScrollBar::add-line:vertical,
            QTableWidget#parcellationTable QScrollBar::sub-line:vertical,
            QTableWidget#parcellationTable QScrollBar::add-line:horizontal,
            QTableWidget#parcellationTable QScrollBar::sub-line:horizontal {
                width: 0px;
                height: 0px;
                background: transparent;
                border: none;
            }

            QTableWidget#parcellationTable QScrollBar::add-page:vertical,
            QTableWidget#parcellationTable QScrollBar::sub-page:vertical,
            QTableWidget#parcellationTable QScrollBar::add-page:horizontal,
            QTableWidget#parcellationTable QScrollBar::sub-page:horizontal {
                background: transparent;
            }
            """)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def _mni_sets(self) -> list:
        try:
            return getattr(self.owner.state, "mni_electrode_sets", []) or []
        except Exception:
            return []

    def _get_parcellations(self):
        p1_img, p1_lut = None, {}
        p2_img, p2_lut = None, {}

        print("[MNI parcellation table] requesting parcellation 1...")
        try:
            p1_img, p1_lut = self.owner._get_mni_parcellation1_img_and_lut()
            print(
                "[MNI parcellation table] parcellation 1:",
                "loaded" if p1_img is not None else "NOT loaded",
                "| LUT size =",
                len(p1_lut) if isinstance(p1_lut, dict) else "not dict",
            )
        except Exception as e:
            print("[MNI parcellation table] parcellation 1 ERROR:", repr(e))
            p1_img, p1_lut = None, {}

        print("[MNI parcellation table] requesting parcellation 2...")
        try:
            p2_img, p2_lut = self.owner._get_mni_parcellation2_img_and_lut()
            print(
                "[MNI parcellation table] parcellation 2:",
                "loaded" if p2_img is not None else "NOT loaded",
                "| LUT size =",
                len(p2_lut) if isinstance(p2_lut, dict) else "not dict",
            )
        except Exception as e:
            print("[MNI parcellation table] parcellation 2 ERROR:", repr(e))
            p2_img, p2_lut = None, {}

        if not isinstance(p1_lut, dict):
            p1_lut = {}

        if not isinstance(p2_lut, dict):
            p2_lut = {}

        return p1_img, p1_lut, p2_img, p2_lut

    def _group_name_from_contact(self, contact: dict) -> str:
        try:
            group = str(contact.get("group", "") or "").strip()
            if group:
                return group
        except Exception:
            pass

        try:
            name = str(contact.get("name", "") or "").strip()
            base = name.rstrip("0123456789")
            return base or "contacts"
        except Exception:
            return "contacts"

    def _sample_labels_at_ras_points(
        self,
        img: sitk.Image | None,
        pts_ras: np.ndarray,
    ) -> np.ndarray:
        """
        Sample integer labels from a SimpleITK parcellation image at MNI RAS points.

        Returns:
            -1 for outside / invalid
             0 for background
            >0 for parcel labels
        """
        pts_ras = np.asarray(pts_ras, dtype=np.float64)

        if img is None or pts_ras.size == 0:
            return np.full((pts_ras.shape[0],), -1, dtype=np.int32)

        try:
            arr = sitk.GetArrayFromImage(img)  # z, y, x

            origin = np.asarray(img.GetOrigin(), dtype=np.float64)
            spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
            direction = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
            inv_direction = np.linalg.inv(direction)

            # MNI display coordinates are RAS.
            # SimpleITK physical coordinates are LPS.
            pts_lps = pts_ras.copy()
            pts_lps[:, 0] *= -1.0
            pts_lps[:, 1] *= -1.0

            rel = pts_lps - origin[None, :]
            idx_xyz = (rel @ inv_direction.T) / spacing[None, :]

            x = np.round(idx_xyz[:, 0]).astype(int)
            y = np.round(idx_xyz[:, 1]).astype(int)
            z = np.round(idx_xyz[:, 2]).astype(int)

            out = np.full((pts_ras.shape[0],), -1, dtype=np.int32)

            inside = (
                (x >= 0)
                & (x < arr.shape[2])
                & (y >= 0)
                & (y < arr.shape[1])
                & (z >= 0)
                & (z < arr.shape[0])
            )

            out[inside] = arr[z[inside], y[inside], x[inside]].astype(np.int32)

            return out

        except Exception:
            return np.full((pts_ras.shape[0],), -1, dtype=np.int32)

    def _label_name_from_lut(self, label_value: int, lut: dict) -> str:
        """
        Convert numeric parcellation index into a display label.

        Rules:
            0 -> empty cell
            17Networks_LH_XXX -> LH_XXX
            17Networks_RH_XXX -> RH_XXX
        """
        try:
            lab = int(float(label_value))
        except Exception:
            return ""

        if lab < 0:
            return "outside"

        if lab == 0:
            return ""

        if not isinstance(lut, dict):
            return f"Label {lab}"

        entry = lut.get(lab, None)

        if entry is None:
            entry = lut.get(str(lab), None)

        if entry is None:
            entry = lut.get(f"{lab}.0", None)

        if entry is None:
            return f"Label {lab}"

        if isinstance(entry, str):
            name = entry

        elif isinstance(entry, dict):
            name = str(
                entry.get("name")
                or entry.get("label")
                or entry.get("region")
                or entry.get("structure")
                or f"Label {lab}"
            )

        elif isinstance(entry, (list, tuple)) and len(entry) >= 1:
            name = str(entry[0])

        else:
            name = f"Label {lab}"

        return self._clean_parcellation_label(name)

    def _clean_parcellation_label(self, name: str) -> str:
        """
        Make TemplateFlow/Schaefer labels shorter for display.

        Example:
            17Networks_LH_VisCent_ExStr_1 -> LH_VisCent_ExStr_1
            17Networks_RH_DefaultA_PFCd_3 -> RH_DefaultA_PFCd_3
        """
        name = str(name or "").strip()

        if not name:
            return ""

        # Cas Schaefer: 17Networks_LH_... ou 7Networks_RH_...
        for hemi in ("LH_", "RH_"):
            idx = name.find(hemi)
            if idx >= 0:
                return name[idx:]

        return name

    def _compute_rows(self) -> list[dict[str, Any]]:
        sets = self._mni_sets()
        p1_img, p1_lut, p2_img, p2_lut = self._get_parcellations()
        print("[MNI parcellation table] MNI sets:", len(sets))
        for si, s in enumerate(sets):
            try:
                print(
                    f"[MNI parcellation table] set {si}:",
                    "subject =",
                    s.get("subject"),
                    "| contacts =",
                    len(s.get("contacts", []) or []),
                    "| path =",
                    s.get("path"),
                )
            except Exception as e:
                print(f"[MNI parcellation table] set {si} debug failed:", repr(e))

        print(
            "[MNI parcellation table] p1_img =",
            p1_img is not None,
            "| p1_lut =",
            len(p1_lut) if isinstance(p1_lut, dict) else "not dict",
            "| p2_img =",
            p2_img is not None,
            "| p2_lut =",
            len(p2_lut) if isinstance(p2_lut, dict) else "not dict",
        )

        all_rows: list[dict[str, Any]] = []

        for mni_set in sets:
            if not isinstance(mni_set, dict):
                continue

            subject = str(mni_set.get("subject", "") or "")
            source_file = Path(str(mni_set.get("path", "") or "")).name
            contacts = mni_set.get("contacts", []) or []

            pts = []
            valid_contacts = []

            for contact in contacts:
                try:
                    x, y, z = contact.get("mni_ras", [None, None, None])
                    x = float(x)
                    y = float(y)
                    z = float(z)
                except Exception:
                    continue

                pts.append([x, y, z])
                valid_contacts.append(contact)

            if not pts:
                continue

            pts_ras = np.asarray(pts, dtype=np.float64)

            p1_labels = self._sample_labels_at_ras_points(p1_img, pts_ras)
            p2_labels = self._sample_labels_at_ras_points(p2_img, pts_ras)

            for i, contact in enumerate(valid_contacts):
                x, y, z = pts_ras[i]
                p1_lab = int(p1_labels[i]) if i < len(p1_labels) else -1
                p2_lab = int(p2_labels[i]) if i < len(p2_labels) else -1

                contact_name = str(contact.get("name", "") or "")
                electrode_name = self._group_name_from_contact(contact)

                row = {
                    "subject": subject,
                    "source_file": source_file,
                    "electrode": electrode_name,
                    "contact": contact_name,
                    "hemisphere": str(contact.get("hemisphere", "") or ""),
                    "type": str(contact.get("type", "") or ""),
                    "x_mni": f"{float(x):.3f}",
                    "y_mni": f"{float(y):.3f}",
                    "z_mni": f"{float(z):.3f}",
                    "parcellation1_atlas": self.parcellation1_name,
                    "parcellation1_index": "" if p1_lab < 0 else str(p1_lab),
                    "parcellation1_label": self._label_name_from_lut(p1_lab, p1_lut),
                    "parcellation2_atlas": self.parcellation2_name,
                    "parcellation2_index": "" if p2_lab < 0 else str(p2_lab),
                    "parcellation2_label": self._label_name_from_lut(p2_lab, p2_lut),
                }

                all_rows.append(row)
        print("[MNI parcellation table] computed rows:", len(all_rows))
        return all_rows

    # ------------------------------------------------------------------
    # Public slots
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        try:
            if hasattr(self, "edit_search"):
                self.edit_search.clear()

            if hasattr(self, "lbl_search_result"):
                self._set_search_result("")

            self.rows = self._compute_rows()
            self._populate_table(self.rows)

            sets = self._mni_sets()
            p1_img, _p1_lut, p2_img, _p2_lut = self._get_parcellations()

            p1_status = "loaded" if p1_img is not None else "not found"
            p2_status = "loaded" if p2_img is not None else "not found"

            self.info_label.setText(
                f"{len(self.rows)} contacts | "
                f"{len(sets)} imported MNI electrode set(s)<br>"
                f"<b>Parcellation 1:</b> {self.parcellation1_name} ({p1_status})<br>"
                f"<b>Parcellation 2:</b> {self.parcellation2_name} ({p2_status})"
            )

        except Exception as e:
            QMessageBox.warning(self, "MNI parcellation table", f"Could not refresh table:\n{e}")

    def _populate_table(self, rows: list[dict[str, Any]]) -> None:
        self.table.setSortingEnabled(False)
        self.table.clearContents()
        self.table.setRowCount(len(rows))

        for r, row in enumerate(rows):
            self.table.setRowHeight(r, 38)
            for c, col in enumerate(self.COLUMNS):
                text = str(row.get(col, "") or "")
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                # Numeric sorting helper for coordinate/index columns.
                if col in (
                    "x_mni",
                    "y_mni",
                    "z_mni",
                    "parcellation1_index",
                    "parcellation2_index",
                ):
                    try:
                        item.setData(Qt.UserRole, float(text))
                    except Exception:
                        pass

                self.table.setItem(r, c, item)

        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()

    def _rows_to_export(self) -> list[dict[str, Any]]:
        """
        Return selected table rows if the user selected at least one row.
        Otherwise return all rows.

        Important:
            The table can be sorted, so we retrieve row data from the visible table
            instead of using self.rows[row_index] directly.
        """
        try:
            selected_rows = sorted(
                {idx.row() for idx in self.table.selectionModel().selectedRows()}
            )
        except Exception:
            selected_rows = []

        if not selected_rows:
            return list(self.rows)

        rows: list[dict[str, Any]] = []

        for r in selected_rows:
            row = {}

            for c, col in enumerate(self.COLUMNS):
                item = self.table.item(r, c)
                row[col] = item.text() if item is not None else ""

            rows.append(row)

        return rows

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _default_export_directory(self) -> str:
        """
        Return the folder containing the patient's MRI.
        """
        try:
            state = self.owner.state
        except Exception:
            state = None

        if state is None:
            return str(Path.home())

        candidates = (
            getattr(state, "t1_path", None),
            getattr(state, "t1_source_path", None),
            getattr(state, "last_browse_dir", None),
        )

        for candidate in candidates:
            if not candidate:
                continue

            try:
                path = Path(str(candidate))

                if path.is_file():
                    return str(path.parent)

                if path.is_dir():
                    return str(path)

                if path.suffix:
                    return str(path.parent)

            except Exception:
                continue

        return str(Path.home())

    def export_table(self) -> None:
        rows_to_export = self._rows_to_export()

        if not rows_to_export:
            QMessageBox.information(self, "Export", "No rows to export.")
            return

        default_directory = Path(self._default_export_directory())
        default_path = default_directory / "mni_electrode_parcellation.tsv"

        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export MNI parcellation table",
            str(default_path),
            "TSV table (*.tsv);;CSV table (*.csv);;Excel workbook (*.xlsx)",
        )

        if not filename:
            return
        try:
            self.owner.state.last_browse_dir = str(Path(filename).parent)
        except Exception:
            pass
        path = Path(filename)

        if selected_filter.startswith("CSV") and path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        elif selected_filter.startswith("Excel") and path.suffix.lower() != ".xlsx":
            path = path.with_suffix(".xlsx")
        elif selected_filter.startswith("TSV") and path.suffix.lower() != ".tsv":
            path = path.with_suffix(".tsv")

        try:
            if path.suffix.lower() == ".xlsx":
                self._export_xlsx(path, rows_to_export)
            elif path.suffix.lower() == ".csv":
                self._export_delimited(path, rows_to_export, delimiter=",")
            else:
                self._export_delimited(path, rows_to_export, delimiter="\t")

            QMessageBox.information(self, "Export", f"Exported:\n{path}")

        except Exception as e:
            QMessageBox.warning(self, "Export", f"Could not export table:\n{e}")

    def _export_delimited(
        self, path: Path, rows: list[dict[str, Any]], delimiter: str = "\t"
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS, delimiter=delimiter)
            writer.writeheader()
            for row in rows:
                writer.writerow({col: row.get(col, "") for col in self.COLUMNS})

    def _export_xlsx(self, path: Path, rows: list[dict[str, Any]]) -> None:
        try:
            from openpyxl import Workbook
        except Exception as e:
            raise RuntimeError(
                "openpyxl is required for Excel export.\n"
                "Install it with:\n"
                "pip install openpyxl"
            ) from e

        path.parent.mkdir(parents=True, exist_ok=True)

        wb = Workbook()
        ws = wb.active
        ws.title = "MNI parcellation"

        ws.append(self.COLUMNS)

        for row in rows:
            ws.append([row.get(col, "") for col in self.COLUMNS])

        try:
            for col_cells in ws.columns:
                max_len = 0
                col_letter = col_cells[0].column_letter
                for cell in col_cells:
                    max_len = max(max_len, len(str(cell.value or "")))
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 45)
        except Exception:
            pass

        wb.save(path)
