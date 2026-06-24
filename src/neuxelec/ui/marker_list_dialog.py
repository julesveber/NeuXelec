from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSizeGrip,
    QVBoxLayout,
    QWidget,
)


class MarkerListHeader(QFrame):
    def __init__(self, dialog: QDialog, parent=None):
        super().__init__(parent)

        self.dialog = dialog
        self._drag_offset: QPoint | None = None

        self.setObjectName("customDialogHeader")
        self.setFixedHeight(34)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.lbl_title = QLabel("MARKER LIST")
        self.lbl_title.setObjectName("markerListHeaderTitle")

        self.btn_close = QPushButton("✕")
        self.btn_close.setObjectName("closeWindowButton")
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.setFixedSize(30, 30)
        self.btn_close.clicked.connect(self.dialog.close)

        layout.addWidget(self.lbl_title, 0, Qt.AlignVCenter)
        layout.addStretch(1)
        layout.addWidget(self.btn_close, 0, Qt.AlignRight | Qt.AlignTop)

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


class MarkerListDialog(QDialog):
    """
    Non-modal floating marker list for the 3D View.

    It does not edit markers directly.
    It emits signals and View3DPage performs the real 3D actions.
    """

    showMarkerOnSlice = Signal(str, str)  # marker_id, plane
    editMarker = Signal(str)
    hideMarker = Signal(str)
    showMarker = Signal(str)
    deleteMarker = Signal(str)
    exportMarker = Signal(str)

    def __init__(
        self,
        markers_provider: Callable[[], list[dict]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._markers_provider = markers_provider
        self._positioned_once = False

        self.setWindowTitle("Marker list")
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)

        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setSizeGripEnabled(False)

        self.setMinimumSize(360, 420)
        self.resize(410, 520)

        self._build_ui()
        self._apply_style()
        self.refresh_markers()

    def showEvent(self, event) -> None:
        super().showEvent(event)

        if self._positioned_once:
            return

        self._positioned_once = True

        try:
            parent = self.parentWidget()
            if parent is None:
                return

            parent_geo = parent.frameGeometry()
            geo = self.frameGeometry()

            # Place it on the right side of the 3D page.
            x = parent_geo.right() - geo.width() - 35
            y = parent_geo.top() + 90
            self.move(x, y)

        except Exception:
            pass

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

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.dialog_shell = QFrame()
        self.dialog_shell.setObjectName("dialogShell")
        outer.addWidget(self.dialog_shell)

        main = QVBoxLayout(self.dialog_shell)
        main.setContentsMargins(14, 8, 14, 14)
        main.setSpacing(10)

        self.header = MarkerListHeader(self)
        main.addWidget(self.header)

        self.lbl_subtitle = QLabel("Right-click a marker to show, edit or manage it")
        self.lbl_subtitle.setObjectName("markerListSubtitle")
        self.lbl_subtitle.setAlignment(Qt.AlignCenter)
        main.addWidget(self.lbl_subtitle)

        self.list_markers = QListWidget()
        self.list_markers.setObjectName("markerListWidget")
        self.list_markers.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_markers.customContextMenuRequested.connect(self._open_context_menu)
        self.list_markers.itemDoubleClicked.connect(self._on_item_double_clicked)
        main.addWidget(self.list_markers, 1)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(10)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setObjectName("secondaryButton")
        self.btn_refresh.setCursor(Qt.PointingHandCursor)
        self.btn_refresh.clicked.connect(self.refresh_markers)

        self.btn_close = QPushButton("Close")
        self.btn_close.setObjectName("primaryButton")
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.clicked.connect(self.close)

        bottom.addWidget(self.btn_refresh)
        bottom.addStretch(1)
        bottom.addWidget(self.btn_close)

        main.addLayout(bottom)

        self.resize_grip = QSizeGrip(self.dialog_shell)
        self.resize_grip.setObjectName("dialogResizeGrip")
        self.resize_grip.setFixedSize(16, 16)
        self.resize_grip.raise_()

    def _markers(self) -> list[dict]:
        try:
            markers = self._markers_provider()
            if isinstance(markers, list):
                return markers
        except Exception:
            pass

        return []

    def refresh_markers(self) -> None:
        self.list_markers.clear()

        markers = self._markers()

        if not markers:
            item = QListWidgetItem("No marker")
            item.setFlags(Qt.ItemIsEnabled)
            self.list_markers.addItem(item)
            return

        for marker in markers:
            marker_id = str(marker.get("id", "")).strip()
            if not marker_id:
                continue

            name = str(marker.get("name", "Marker")).strip() or "Marker"
            marker_type = str(marker.get("type", "Lesion")).strip() or "Lesion"
            visible = bool(marker.get("visible", True))
            color = str(marker.get("color", "#FF3B30"))

            prefix = "●"
            suffix = "" if visible else "  (hidden)"
            text = f"{prefix}  {name}  -  {marker_type}{suffix}"

            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, marker_id)

            try:
                item.setForeground(QColor(color))
            except Exception:
                pass

            self.list_markers.addItem(item)

    def _marker_id_from_item(self, item: QListWidgetItem | None) -> str | None:
        if item is None:
            return None

        marker_id = item.data(Qt.UserRole)

        if marker_id is None:
            return None

        marker_id = str(marker_id).strip()

        return marker_id or None

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        marker_id = self._marker_id_from_item(item)

        if marker_id:
            self.editMarker.emit(marker_id)

    def _open_context_menu(self, pos: QPoint) -> None:
        item = self.list_markers.itemAt(pos)
        marker_id = self._marker_id_from_item(item)

        if not marker_id:
            return

        marker = None
        for m in self._markers():
            if str(m.get("id", "")) == marker_id:
                marker = m
                break

        if marker is None:
            return

        menu = QMenu(self)
        menu.setObjectName("markerListContextMenu")
        menu.setStyleSheet(self._menu_style())

        act_show_axial = menu.addAction("Show on axial slice")
        act_show_coronal = menu.addAction("Show on coronal slice")
        act_show_sagittal = menu.addAction("Show on sagittal slice")

        menu.addSeparator()

        act_edit = menu.addAction("Edit marker")

        visible = bool(marker.get("visible", True))
        act_visibility = menu.addAction("Hide marker" if visible else "Show marker")

        act_export = menu.addAction("Export marker")

        menu.addSeparator()

        act_delete = menu.addAction("Delete marker")

        choice = menu.exec(self.list_markers.viewport().mapToGlobal(pos))

        if choice == act_show_axial:
            self.showMarkerOnSlice.emit(marker_id, "axial")

        elif choice == act_show_coronal:
            self.showMarkerOnSlice.emit(marker_id, "coronal")

        elif choice == act_show_sagittal:
            self.showMarkerOnSlice.emit(marker_id, "sagittal")

        elif choice == act_edit:
            self.editMarker.emit(marker_id)

        elif choice == act_visibility:
            if visible:
                self.hideMarker.emit(marker_id)
            else:
                self.showMarker.emit(marker_id)

        elif choice == act_export:
            self.exportMarker.emit(marker_id)

        elif choice == act_delete:
            self.deleteMarker.emit(marker_id)

    def _menu_style(self) -> str:
        return """
            QMenu {
                color: #F2F2F5;
                background-color: #10121A;
                border: 1px solid #2B2D38;
                border-radius: 9px;
                padding: 6px;
            }

            QMenu::item {
                padding: 7px 24px 7px 12px;
                border-radius: 6px;
                background-color: transparent;
            }

            QMenu::item:selected {
                background-color: rgba(255, 72, 125, 45);
                color: white;
            }

            QMenu::separator {
                height: 1px;
                background-color: #2B2D38;
                margin: 6px 4px;
            }
        """

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

            QLabel#markerListHeaderTitle {
                color: #F4D9D0;
                font-size: 14px;
                font-weight: 600;
                letter-spacing: 1px;
                padding-left: 4px;
            }

            QLabel#markerListSubtitle {
                color: #8E8E98;
                font-size: 12px;
                font-weight: 400;
                padding-bottom: 4px;
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

            QListWidget#markerListWidget {
                color: #F2F2F5;
                background-color: #10121A;
                border: 1px solid #2B2D38;
                border-radius: 12px;
                padding: 6px;
                outline: none;
            }

            QListWidget#markerListWidget::item {
                min-height: 34px;
                padding: 7px 10px;
                border-radius: 8px;
                background-color: #171922;
                margin: 3px;
            }

            QListWidget#markerListWidget::item:hover {
                border: 1px solid #FF487D;
                background-color: #20222B;
            }

            QListWidget#markerListWidget::item:selected {
                border: 1px solid #FF487D;
                background-color: rgba(255, 72, 125, 32);
            }

            QPushButton#secondaryButton {
                color: white;
                background-color: #17181F;
                border: 1px solid #2B2D38;
                border-radius: 10px;
                min-height: 38px;
                padding-left: 16px;
                padding-right: 16px;
                font-size: 13px;
                font-weight: 600;
            }

            QPushButton#secondaryButton:hover {
                background-color: #20222B;
                border: 1px solid #FF487D;
            }

            QPushButton#primaryButton {
                color: white;
                border: none;
                border-radius: 10px;
                min-height: 38px;
                padding-left: 18px;
                padding-right: 18px;
                font-size: 13px;
                font-weight: 600;
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

            QSizeGrip#dialogResizeGrip {
                background-color: transparent;
                border: none;
                image: none;
            }
            """)
