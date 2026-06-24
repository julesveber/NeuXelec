from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import SimpleITK as sitk
from PySide6.QtCore import QElapsedTimer, QEvent, QPoint, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

_COORD_FIELDS = {"x", "y", "z"}


def _format_coord_value(value, decimals: int = 2):
    """
    Format exported coordinate values with a fixed number of decimals.

    - LPS/RAS/MNI coordinates are exported as strings with 2 decimals.
    - Empty values and 'n/a' are preserved.
    - Voxel integer coordinates can stay integers if they are already ints.
    """
    if value in ("", None, "n/a"):
        return value

    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return value


def _format_row_for_export(row: dict[str, Any], decimals: int = 2) -> dict[str, Any]:
    """
    Return a copy of a row where only x/y/z are formatted for export.
    """
    out = dict(row)

    for key in _COORD_FIELDS:
        if key in out:
            out[key] = _format_coord_value(out[key], decimals=decimals)

    return out


def _lps_to_ras(lps):
    x, y, z = [float(v) for v in lps]
    return (-x, -y, z)


def _parse_electrode_selection(text: str, n_electrodes: int) -> list[int]:
    text = (text or "").strip()
    if not text:
        return []

    selected = set()

    for part in text.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            try:
                a, b = part.split("-", 1)
                start = int(a.strip())
                end = int(b.strip())
            except Exception:
                continue

            if end < start:
                start, end = end, start

            for i in range(start, end + 1):
                if 1 <= i <= n_electrodes:
                    selected.add(i - 1)
        else:
            try:
                i = int(part)
            except Exception:
                continue

            if 1 <= i <= n_electrodes:
                selected.add(i - 1)

    return sorted(selected)


def _sample_parcellation_label(img: sitk.Image | None, lps_xyz):
    if img is None:
        return None

    try:
        p = tuple(float(v) for v in lps_xyz)
        idx = img.TransformPhysicalPointToIndex(p)
        size = img.GetSize()

        if not (0 <= idx[0] < size[0] and 0 <= idx[1] < size[1] and 0 <= idx[2] < size[2]):
            return None

        return int(img.GetPixel(*idx))
    except Exception:
        return None


def _lookup_lut_region(lut: Any, label: int | None) -> str:
    if label is None:
        return ""

    if not isinstance(lut, dict):
        return "Unknown"

    entry = lut.get(int(label), None)
    if entry is None:
        return "Unknown"

    try:
        name, _rgb = entry
        return str(name)
    except Exception:
        return "Unknown"


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


class BidsExportProgressDialog(QDialog):
    """
    Small NeuXelec-styled progress window displayed during BIDS export.
    It intentionally has no close button because cancelling ANTs/MNI export
    during processing could leave incomplete output files.
    """

    def __init__(self, title: str = "Export coordinates", parent=None):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.setWindowModality(Qt.ApplicationModal)

        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setSizeGripEnabled(False)

        self.setFixedSize(500, 208)

        # Time-based "smooth" progress. When active, the bar advances steadily
        # with wall-clock time toward a ceiling (98%) over a fixed duration and
        # then holds there until the real export completes and forces 100%.
        # This gives a regular, predictable bar for long operations (MNI/ANTs
        # export, ~3 min) instead of jumping with ANTs log verbosity.
        self._smooth_active = False
        self._smooth_ceiling = 98
        self._smooth_duration_ms = 180_000  # 3 minutes
        self._smooth_elapsed = QElapsedTimer()
        self._smooth_timer = QTimer(self)
        self._smooth_timer.setInterval(200)
        self._smooth_timer.timeout.connect(self._on_smooth_tick)

        self._build_ui()
        self._apply_style()
        self._center_on_parent()

    def start_smooth_progress(self, duration_seconds: float = 180.0) -> None:
        """Drive the bar smoothly to the 98% ceiling over ``duration_seconds``.

        Intermediate setValue() calls are ignored while active (the timer owns
        the bar); only setValue(100) / finish() ends it at 100%.
        """
        self._smooth_duration_ms = max(1, int(duration_seconds * 1000))
        self._smooth_active = True
        self.progress.setValue(0)
        self._smooth_elapsed.restart()
        self._smooth_timer.start()

    def _on_smooth_tick(self) -> None:
        if not self._smooth_active:
            return
        elapsed = self._smooth_elapsed.elapsed()
        fraction = min(1.0, elapsed / float(self._smooth_duration_ms))
        target = int(round(self._smooth_ceiling * fraction))
        # Never move backwards.
        if target > self.progress.value():
            self.progress.setValue(target)

    def finish(self) -> None:
        """Stop the smooth animation and complete the bar at 100%."""
        self._smooth_active = False
        self._smooth_timer.stop()
        self.progress.setValue(100)
        self.progress.repaint()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.dialog_shell = QFrame()
        self.dialog_shell.setObjectName("progressDialogShell")

        layout = QVBoxLayout(self.dialog_shell)
        layout.setContentsMargins(26, 22, 26, 24)
        layout.setSpacing(14)

        outer_layout.addWidget(self.dialog_shell)

        self.lbl_title = QLabel("BIDS EXPORT")
        self.lbl_title.setObjectName("progressDialogTitle")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_title)

        self.lbl_message = QLabel("Preparing export...")
        self.lbl_message.setObjectName("progressMessage")
        self.lbl_message.setAlignment(Qt.AlignCenter)
        self.lbl_message.setWordWrap(True)
        layout.addWidget(self.lbl_message)

        layout.addSpacing(4)

        self.progress = QProgressBar()
        self.progress.setObjectName("bidsExportProgressBar")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.progress.setTextVisible(True)
        self.progress.setMinimumHeight(24)
        layout.addWidget(self.progress)

    def _center_on_parent(self) -> None:
        try:
            parent = self.parentWidget()

            if parent is None:
                return

            geometry = self.frameGeometry()
            geometry.moveCenter(parent.frameGeometry().center())
            self.move(geometry.topLeft())

        except Exception:
            pass

    def setMaximum(self, maximum: int) -> None:
        self.progress.setMaximum(int(maximum))

    def setValue(self, value: int) -> None:
        value = int(value)
        if self._smooth_active:
            # The timer owns the bar; only completion (>=100) ends it.
            if value >= 100:
                self.finish()
            return
        self.progress.setValue(value)
        self.progress.repaint()

    def setLabelText(self, text: str) -> None:
        self.lbl_message.setText(str(text))
        self.lbl_message.repaint()

    def _apply_style(self) -> None:

        self.setStyleSheet("""
            QDialog {
                background: transparent;
            }

            QFrame#progressDialogShell {
                background-color: #06070D;
                border: 1.5px solid #FF487D;
                border-radius: 16px;
            }

            QLabel {
                background-color: transparent;
                border: none;
            }

            QLabel#progressDialogTitle {
                color: #F4D9D0;
                font-size: 18px;
                font-weight: 500;
                letter-spacing: 1px;
            }

            QLabel#progressMessage {
                color: #CFCFD6;
                font-size: 12px;
                font-weight: 500;
            }

            QProgressBar#bidsExportProgressBar {
                color: #F2F2F5;
                background-color: #171922;
                border: 1px solid #2B2D38;
                border-radius: 11px;
                text-align: center;
                font-size: 11px;
                font-weight: 600;
                min-height: 24px;
            }

            QProgressBar#bidsExportProgressBar::chunk {
                margin: 0px;
                border: none;
                border-radius: 10px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
            }
            """)


class ExportSuccessDialog(QDialog):
    """
    NeuXelec-styled confirmation dialog displayed after a successful export.

    Clicking Done validates the popup and closes Export coordinates.
    Clicking the top-right cross closes only this popup.
    """

    def __init__(self, output_directory: Path, parent=None):
        super().__init__(parent)

        self.output_directory = Path(output_directory)
        self._positioned_once = False

        self.setWindowTitle("Export completed")
        self.setModal(True)

        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setSizeGripEnabled(False)

        self.setFixedSize(540, 290)

        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.dialog_shell = QFrame()
        self.dialog_shell.setObjectName("successDialogShell")

        root = QVBoxLayout(self.dialog_shell)
        root.setContentsMargins(14, 8, 14, 14)
        root.setSpacing(10)

        outer_layout.addWidget(self.dialog_shell)

        # The shared header closes this popup with reject().
        self.custom_header = NeuXelecDialogHeader(self)
        root.addWidget(self.custom_header)

        content = QVBoxLayout()
        content.setContentsMargins(20, 0, 20, 8)
        content.setSpacing(12)

        self.lbl_title = QLabel("EXPORT COMPLETED")
        self.lbl_title.setObjectName("successTitle")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        content.addWidget(self.lbl_title)

        self.lbl_subtitle = QLabel("Your electrode coordinates have been successfully exported.")
        self.lbl_subtitle.setObjectName("successSubtitle")
        self.lbl_subtitle.setAlignment(Qt.AlignCenter)
        self.lbl_subtitle.setWordWrap(True)
        content.addWidget(self.lbl_subtitle)

        self.path_frame = QFrame()
        self.path_frame.setObjectName("outputPathCard")

        path_layout = QVBoxLayout(self.path_frame)
        path_layout.setContentsMargins(14, 10, 14, 10)
        path_layout.setSpacing(5)

        self.lbl_path_title = QLabel("Saved in")
        self.lbl_path_title.setObjectName("pathTitle")

        self.lbl_path = QLabel(str(self.output_directory))
        self.lbl_path.setObjectName("pathValue")
        self.lbl_path.setWordWrap(True)
        self.lbl_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_path.setToolTip(str(self.output_directory))

        path_layout.addWidget(self.lbl_path_title)
        path_layout.addWidget(self.lbl_path)

        content.addWidget(self.path_frame)

        root.addLayout(content)
        root.addStretch(1)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(20, 0, 20, 4)
        buttons.setSpacing(10)

        self.btn_done = QPushButton("Done")
        self.btn_done.setObjectName("successPrimaryButton")
        self.btn_done.setCursor(Qt.PointingHandCursor)
        self.btn_done.setFixedSize(118, 42)

        # Done = validate the popup.
        self.btn_done.clicked.connect(self.accept)

        buttons.addStretch(1)
        buttons.addWidget(self.btn_done)

        root.addLayout(buttons)

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

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QDialog {
                background: transparent;
            }

            QFrame#successDialogShell {
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

            QLabel#successTitle {
                color: #F4D9D0;
                font-size: 18px;
                font-weight: 500;
                letter-spacing: 1px;
            }

            QLabel#successSubtitle {
                color: #CFCFD6;
                font-size: 12px;
                font-weight: 500;
            }

            QFrame#outputPathCard {
                background-color: #10121A;
                border: 1px solid #2B2D38;
                border-radius: 10px;
            }

            QLabel#pathTitle {
                color: #8E8E98;
                font-size: 11px;
                font-weight: 500;
            }

            QLabel#pathValue {
                color: #F2F2F5;
                font-size: 12px;
                font-weight: 600;
            }

            QPushButton#successPrimaryButton {
                min-height: 42px;
                border-radius: 10px;
                color: white;
                border: none;
                font-size: 13px;
                font-weight: 600;
                padding-left: 18px;
                padding-right: 18px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
            }

            QPushButton#successPrimaryButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF922B,
                    stop:1 #FF33B8
                );
            }

            QPushButton#successPrimaryButton:pressed {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #E56F00,
                    stop:1 #E0008C
                );
            }
            """)


class ExportCoordinatesDialog(QDialog):
    def __init__(self, state, parent=None):
        super().__init__(parent)

        self.state = state
        self._positioned_once = False

        # Invisible interactive border used to resize the frameless window.
        self._resize_margin = 8

        self.setWindowTitle("Export electrode coordinates")

        # Frameless rounded NeuXelec dialog.
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # Remove native white corner grip and use a transparent internal one.
        self.setSizeGripEnabled(False)

        # The default opening size remains unchanged, but the dialog can be
        # reduced further on smaller screens. The central area will scroll.
        self.setMinimumSize(520, 420)
        self._set_adapted_initial_size()

        self._build_ui()
        self._apply_style()

    def _set_adapted_initial_size(self) -> None:
        """
        Keep the current preferred opening size whenever possible while
        ensuring that the dialog remains usable on smaller screens.
        """
        # Keep these values: this is the current default size you like.
        preferred_width = 720
        preferred_height = 760

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
        Detect whether the pointer is over one of the invisible resize borders
        or corners of the frameless rounded dialog.
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
        Display the expected resize cursor when hovering over a border
        or corner of the frameless dialog.
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
        Start Qt/Windows native resizing while preserving the custom
        NeuXelec frameless appearance.
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
        self.scroll_area.setObjectName("exportCoordinatesScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("exportCoordinatesScrollContent")
        self.scroll_content.setMinimumHeight(620)

        content = QVBoxLayout(self.scroll_content)
        content.setContentsMargins(12, 4, 12, 10)
        content.setSpacing(14)

        # ---------------------------------------------------------
        # Title
        # ---------------------------------------------------------
        self.lbl_title = QLabel("EXPORT COORDINATES")
        self.lbl_title.setObjectName("dialogTitle")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        content.addWidget(self.lbl_title)

        self.lbl_subtitle = QLabel("Choose coordinate spaces, file formats and optional metadata")
        self.lbl_subtitle.setObjectName("dialogSubtitle")
        self.lbl_subtitle.setAlignment(Qt.AlignCenter)
        content.addWidget(self.lbl_subtitle)

        # ============================================================
        # Export format and coordinates row
        # ============================================================
        top_options = QHBoxLayout()
        top_options.setSpacing(12)

        # ---------------------------------------------------------
        # Formats
        # ---------------------------------------------------------
        grp_formats = QGroupBox("Export format")
        grp_formats.setObjectName("optionGroup")

        lay_formats = QVBoxLayout(grp_formats)
        lay_formats.setContentsMargins(14, 20, 14, 12)
        lay_formats.setSpacing(8)

        self.chk_txt = QCheckBox("TXT")
        self.chk_csv = QCheckBox("CSV")
        self.chk_tsv = QCheckBox("TSV")
        self.chk_json = QCheckBox("JSON")
        self.chk_els = QCheckBox("Cartool ELS")

        formats_row_1 = QHBoxLayout()
        formats_row_1.addWidget(self.chk_txt)
        formats_row_1.addWidget(self.chk_csv)

        formats_row_2 = QHBoxLayout()
        formats_row_2.addWidget(self.chk_tsv)
        formats_row_2.addWidget(self.chk_json)

        lay_formats.addLayout(formats_row_1)
        lay_formats.addLayout(formats_row_2)

        formats_row_3 = QHBoxLayout()
        formats_row_3.addWidget(self.chk_els)
        formats_row_3.addStretch(1)

        lay_formats.addLayout(formats_row_3)

        top_options.addWidget(grp_formats, 1)

        # ---------------------------------------------------------
        # Coordinates
        # ---------------------------------------------------------
        grp_coords = QGroupBox("Coordinates to include")
        grp_coords.setObjectName("optionGroup")

        lay_coords = QVBoxLayout(grp_coords)
        lay_coords.setContentsMargins(14, 20, 14, 12)
        lay_coords.setSpacing(8)

        self.chk_lps = QCheckBox("LPS coordinates")
        self.chk_ras = QCheckBox("RAS coordinates")
        self.chk_voxel = QCheckBox("Voxel indices")

        lay_coords.addWidget(self.chk_lps)
        lay_coords.addWidget(self.chk_ras)
        lay_coords.addWidget(self.chk_voxel)

        top_options.addWidget(grp_coords, 1)

        content.addLayout(top_options)

        # ============================================================
        # BIDS options
        # ============================================================
        self.grp_bids = QGroupBox("BIDS options")
        self.grp_bids.setObjectName("optionGroup")

        lay_bids = QVBoxLayout(self.grp_bids)
        lay_bids.setContentsMargins(14, 20, 14, 14)
        lay_bids.setSpacing(10)

        self.chk_bids = QCheckBox("BIDS")
        self.chk_bids.setObjectName("emphasisCheckbox")
        lay_bids.addWidget(self.chk_bids)

        bids_options_row = QHBoxLayout()
        bids_options_row.setSpacing(20)

        # ---------------------------------------------------------
        # BIDS coordinate space
        # ---------------------------------------------------------
        space_card = QFrame()
        space_card.setObjectName("innerOptionCard")

        space_layout = QVBoxLayout(space_card)
        space_layout.setContentsMargins(12, 10, 12, 10)
        space_layout.setSpacing(8)

        lbl_bids_space = QLabel("BIDS coordinate space")
        lbl_bids_space.setObjectName("subSectionLabel")
        space_layout.addWidget(lbl_bids_space)

        self.radio_bids_native_t1 = QRadioButton("Native T1 space")
        self.radio_bids_mni = QRadioButton("MNI152 atlas space")
        self.radio_bids_native_t1.setChecked(True)

        self._bids_space_group = QButtonGroup(self)
        self._bids_space_group.addButton(self.radio_bids_native_t1)
        self._bids_space_group.addButton(self.radio_bids_mni)

        self.radio_bids_native_t1.toggled.connect(
            lambda _=None: self._update_bids_coordinate_convention_enabled()
        )
        self.radio_bids_mni.toggled.connect(
            lambda _=None: self._update_bids_coordinate_convention_enabled()
        )

        space_layout.addWidget(self.radio_bids_native_t1)
        space_layout.addWidget(self.radio_bids_mni)

        bids_options_row.addWidget(space_card, 1)

        # ---------------------------------------------------------
        # Native T1 coordinate convention
        # ---------------------------------------------------------
        convention_card = QFrame()
        convention_card.setObjectName("innerOptionCard")

        convention_layout = QVBoxLayout(convention_card)
        convention_layout.setContentsMargins(12, 10, 12, 10)
        convention_layout.setSpacing(8)

        lbl_bids_convention = QLabel("Native T1 convention")
        lbl_bids_convention.setObjectName("subSectionLabel")
        convention_layout.addWidget(lbl_bids_convention)

        self.radio_bids_lps = QRadioButton("LPS")
        self.radio_bids_ras = QRadioButton("RAS")
        self.radio_bids_vox = QRadioButton("VOX")
        self.radio_bids_lps.setChecked(True)

        self._bids_coord_group = QButtonGroup(self)
        self._bids_coord_group.addButton(self.radio_bids_lps)
        self._bids_coord_group.addButton(self.radio_bids_ras)
        self._bids_coord_group.addButton(self.radio_bids_vox)

        convention_row = QHBoxLayout()
        convention_row.setSpacing(12)
        convention_row.addWidget(self.radio_bids_lps)
        convention_row.addWidget(self.radio_bids_ras)
        convention_row.addWidget(self.radio_bids_vox)
        convention_row.addStretch(1)

        convention_layout.addLayout(convention_row)

        bids_options_row.addWidget(convention_card, 1)

        lay_bids.addLayout(bids_options_row)

        # Disable BIDS radio controls until BIDS export is enabled.
        for child in self.grp_bids.findChildren(QRadioButton):
            child.setEnabled(False)

        self.chk_bids.toggled.connect(self._toggle_bids_section)

        content.addWidget(self.grp_bids)

        # ============================================================
        # Electrodes and parcellation row
        # ============================================================
        bottom_options = QHBoxLayout()
        bottom_options.setSpacing(12)

        # ---------------------------------------------------------
        # Electrode selection
        # ---------------------------------------------------------
        grp_elec = QGroupBox("Electrodes")
        grp_elec.setObjectName("optionGroup")

        lay_elec = QVBoxLayout(grp_elec)
        lay_elec.setContentsMargins(14, 20, 14, 14)
        lay_elec.setSpacing(8)

        self.radio_all = QRadioButton("All electrodes")
        self.radio_selected = QRadioButton("Selected electrodes")
        self.radio_all.setChecked(True)

        self._elec_group = QButtonGroup(self)
        self._elec_group.addButton(self.radio_all)
        self._elec_group.addButton(self.radio_selected)

        self.edit_selection = QLineEdit()
        self.edit_selection.setObjectName("selectionField")
        self.edit_selection.setPlaceholderText("Example: 1,3,5-8")
        self.edit_selection.setMinimumHeight(38)
        self.edit_selection.setEnabled(False)

        self.radio_selected.toggled.connect(self.edit_selection.setEnabled)

        n = len(getattr(self.state, "electrodes", []) or [])

        self.lbl_available_electrodes = QLabel(f"Available electrodes: {n}")
        self.lbl_available_electrodes.setObjectName("informationLabel")

        lay_elec.addWidget(self.lbl_available_electrodes)
        lay_elec.addWidget(self.radio_all)
        lay_elec.addWidget(self.radio_selected)
        lay_elec.addWidget(self.edit_selection)

        bottom_options.addWidget(grp_elec, 1)

        # ---------------------------------------------------------
        # Parcellation
        # ---------------------------------------------------------
        grp_parc = QGroupBox("Parcellation")
        grp_parc.setObjectName("optionGroup")

        lay_parc = QVBoxLayout(grp_parc)
        lay_parc.setContentsMargins(14, 20, 14, 14)
        lay_parc.setSpacing(8)

        self.chk_include_parc = QCheckBox("Include parcellation labels")
        self.chk_parc1 = QCheckBox("Parcellation 1")
        self.chk_parc2 = QCheckBox("Parcellation 2")

        self.chk_parc1.setEnabled(False)
        self.chk_parc2.setEnabled(False)

        self.chk_include_parc.toggled.connect(self.chk_parc1.setEnabled)
        self.chk_include_parc.toggled.connect(self.chk_parc2.setEnabled)

        self.chk_parc1.setChecked(True)

        lay_parc.addWidget(self.chk_include_parc)
        lay_parc.addSpacing(4)
        lay_parc.addWidget(self.chk_parc1)
        lay_parc.addWidget(self.chk_parc2)
        lay_parc.addStretch(1)

        bottom_options.addWidget(grp_parc, 1)

        content.addLayout(bottom_options)
        content.addStretch(1)

        self.scroll_area.setWidget(self.scroll_content)
        shell_layout.addWidget(self.scroll_area, 1)

        # ============================================================
        # Fixed bottom buttons
        # ============================================================
        actions = QHBoxLayout()
        actions.setContentsMargins(12, 0, 12, 4)
        actions.setSpacing(10)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("secondaryButton")
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setMinimumHeight(42)
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_export = QPushButton("Export")
        self.btn_export.setObjectName("primaryButton")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setMinimumHeight(42)
        self.btn_export.setMinimumWidth(120)
        self.btn_export.clicked.connect(self._on_export)

        actions.addStretch(1)
        actions.addWidget(self.btn_cancel)
        actions.addWidget(self.btn_export)

        shell_layout.addLayout(actions)

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


            QWidget#exportCoordinatesScrollContent {
                background-color: transparent;
            }

            QScrollArea#exportCoordinatesScrollArea {
                background-color: transparent;
                border: none;
            }

            QScrollArea#exportCoordinatesScrollArea > QWidget > QWidget {
                background-color: transparent;
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
                padding-bottom: 3px;
            }

            QLabel#subSectionLabel {
                color: #CFCFD6;
                font-size: 12px;
                font-weight: 600;
                padding-bottom: 2px;
            }

            QLabel#informationLabel {
                color: #A6A8B2;
                background-color: #171922;
                border: 1px solid #2B2D38;
                border-radius: 7px;
                padding: 8px;
                font-size: 11px;
                font-weight: 500;
            }

            QGroupBox#optionGroup {
                color: #CFCFD6;
                font-size: 13px;
                font-weight: 600;
                background-color: #10121A;
                border: 1px solid #2B2D38;
                border-radius: 12px;
                margin-top: 10px;
                padding-top: 8px;
            }

            QGroupBox#optionGroup::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0px 6px;
                color: #CFCFD6;
                background-color: #06070D;
            }

            QFrame#innerOptionCard {
                background-color: #171922;
                border: 1px solid #2B2D38;
                border-radius: 9px;
            }

            QCheckBox,
            QRadioButton {
                color: #D8DAE4;
                spacing: 8px;
                font-size: 12px;
                font-weight: 500;
            }

            QCheckBox:disabled,
            QRadioButton:disabled {
                color: #636672;
            }

            QCheckBox::indicator,
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                background-color: #151720;
                border: 1px solid #353844;
                border-radius: 4px;
            }

            QCheckBox::indicator:hover,
            QRadioButton::indicator:hover {
                border: 1px solid #FF487D;
            }

            QCheckBox::indicator:checked,
            QRadioButton::indicator:checked {
                background-color: #151720;
                border: 1px solid #FF487D;
                border-radius: 4px;
                image: url(resources/images/neuxelec_checkbox_cross.svg);
            }

            QCheckBox::indicator:checked:hover,
            QRadioButton::indicator:checked:hover {
                background-color: #181A24;
                border: 1px solid #FF487D;
                border-radius: 4px;
                image: url(resources/images/neuxelec_checkbox_cross.svg);
            }

            QCheckBox::indicator:disabled,
            QRadioButton::indicator:disabled {
                background-color: #10121A;
                border: 1px solid #252834;
                border-radius: 4px;
            }

            QCheckBox::indicator:checked:disabled,
            QRadioButton::indicator:checked:disabled {
                background-color: #10121A;
                border: 1px solid #555967;
                border-radius: 4px;
                image: url(resources/images/neuxelec_checkbox_cross.svg);
            }

            QLineEdit#selectionField {
                min-height: 38px;
                background-color: #171922;
                color: #F2F2F5;
                border: 1px solid #2B2D38;
                border-radius: 8px;
                padding-left: 12px;
                padding-right: 12px;
                selection-background-color: #FF008F;
                font-size: 12px;
            }

            QLineEdit#selectionField:hover {
                border: 1px solid #3A3D4A;
            }

            QLineEdit#selectionField:focus {
                border: 1px solid #FF487D;
            }

            QLineEdit#selectionField:disabled {
                color: #62646E;
                background-color: #121319;
                border: 1px solid #20222A;
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

            QScrollArea#exportCoordinatesScrollArea QScrollBar:vertical {
                background-color: transparent;
                border: none;
                width: 10px;
                margin: 5px 2px 5px 2px;
                border-radius: 5px;
            }

            QScrollArea#exportCoordinatesScrollArea QScrollBar::handle:vertical {
                min-height: 28px;
                background-color: #3B3E48;
                border: 1px solid #FF487D;
                border-radius: 5px;
            }

            QScrollArea#exportCoordinatesScrollArea QScrollBar::handle:vertical:hover {
                background-color: #4A4D58;
                border: 1px solid #FF6B98;
            }

            QScrollArea#exportCoordinatesScrollArea QScrollBar::handle:vertical:pressed {
                background-color: #343743;
                border: 1px solid #FF487D;
            }

            QScrollArea#exportCoordinatesScrollArea QScrollBar::add-line:vertical,
            QScrollArea#exportCoordinatesScrollArea QScrollBar::sub-line:vertical {
                height: 0px;
                width: 0px;
                background: transparent;
                border: none;
            }

            QScrollArea#exportCoordinatesScrollArea QScrollBar::add-page:vertical,
            QScrollArea#exportCoordinatesScrollArea QScrollBar::sub-page:vertical {
                background: transparent;
                border: none;
            }
            """)

    def _selected_electrode_indices(self) -> list[int]:
        electrodes = getattr(self.state, "electrodes", []) or []
        n = len(electrodes)

        if self.radio_all.isChecked():
            return list(range(n))

        return _parse_electrode_selection(self.edit_selection.text(), n)

    def _get_selected_coordinate_systems(self) -> list[str]:
        systems = []

        if self.chk_lps.isChecked():
            systems.append("LPS")
        if self.chk_ras.isChecked():
            systems.append("RAS")
        if self.chk_voxel.isChecked():
            systems.append("VOX")

        return systems

    def _get_bids_coordinate_system(self) -> str:
        if getattr(self, "radio_bids_ras", None) is not None and self.radio_bids_ras.isChecked():
            return "RAS"
        if getattr(self, "radio_bids_vox", None) is not None and self.radio_bids_vox.isChecked():
            return "VOX"
        return "LPS"

    def _get_bids_space(self) -> str:
        """
        Return BIDS coordinate space selected by the user.
        """
        try:
            if self.radio_bids_mni.isChecked():
                return "MNI"
        except Exception:
            pass

        return "T1w"

    def _patient_filename_prefix(self) -> str:
        patient = str(getattr(self.state, "patient_id", "") or "").strip()
        patient = "".join(c for c in patient if c.isalnum() or c in ("_", "-"))

        if not patient:
            patient = "patient"

        return patient

    def _get_parcellation_images_and_luts(self):
        parcel1_img = getattr(self.state, "parcel1_img", None)
        parcel2_img = getattr(self.state, "parcel2_img", None)

        if parcel1_img is None:
            parcel1_img = getattr(self.state, "parcellation1_img", None)
        if parcel2_img is None:
            parcel2_img = getattr(self.state, "parcellation2_img", None)

        lut1 = getattr(self.state, "parcellation1_lut", {}) or {}
        lut2 = getattr(self.state, "parcellation2_lut", {}) or {}

        return parcel1_img, parcel2_img, lut1, lut2

    def _build_rows(self, coord_system: str = "LPS") -> list[dict[str, Any]]:
        coord_system = str(coord_system or "LPS").upper().strip()

        electrodes = getattr(self.state, "electrodes", []) or []
        indices = self._selected_electrode_indices()

        rows: list[dict[str, Any]] = []

        parcel1_img, parcel2_img, lut1, lut2 = self._get_parcellation_images_and_luts()
        include_parc = bool(self.chk_include_parc.isChecked())

        for elec_idx in indices:
            if elec_idx < 0 or elec_idx >= len(electrodes):
                continue

            elec = electrodes[elec_idx]
            elec_name = str(elec.get("name", f"Electrode_{elec_idx + 1}"))
            reference = str(elec.get("ref", ""))
            hemi = str(elec.get("hemisphere", ""))

            contacts_lps = elec.get("contacts_lps", []) or []
            contacts_idx = elec.get("contacts_idx", []) or []

            for ci, lps in enumerate(contacts_lps):
                try:
                    lps_tuple = tuple(float(v) for v in lps)
                except Exception:
                    continue

                row: dict[str, Any] = {
                    "electrode": elec_name,
                    "reference": reference,
                    "hemisphere": hemi,
                    "contact": f"{elec_name}{ci + 1}",
                    "type": "SEEG",
                    "coordinate_system": coord_system,
                    "units": "mm" if coord_system in ("LPS", "RAS") else "voxel",
                }

                if coord_system == "LPS":
                    row["x"] = lps_tuple[0]
                    row["y"] = lps_tuple[1]
                    row["z"] = lps_tuple[2]

                elif coord_system == "RAS":
                    ras = _lps_to_ras(lps_tuple)
                    row["x"] = ras[0]
                    row["y"] = ras[1]
                    row["z"] = ras[2]

                elif coord_system == "VOX":
                    if ci < len(contacts_idx):
                        try:
                            vx, vy, vz = contacts_idx[ci]
                            row["x"] = int(vx)
                            row["y"] = int(vy)
                            row["z"] = int(vz)
                        except Exception:
                            row["x"] = ""
                            row["y"] = ""
                            row["z"] = ""
                    else:
                        row["x"] = ""
                        row["y"] = ""
                        row["z"] = ""

                if include_parc and self.chk_parc1.isChecked():
                    label1 = _sample_parcellation_label(parcel1_img, lps_tuple)
                    row["parcel1_label"] = "" if label1 is None else int(label1)
                    row["parcel1_region"] = _lookup_lut_region(lut1, label1)

                if include_parc and self.chk_parc2.isChecked():
                    label2 = _sample_parcellation_label(parcel2_img, lps_tuple)
                    row["parcel2_label"] = "" if label2 is None else int(label2)
                    row["parcel2_region"] = _lookup_lut_region(lut2, label2)

                rows.append(row)

        return rows

    def _build_rows_mni(self, progress_callback=None) -> list[dict[str, Any]]:
        """
        Build rows in MNI coordinates.

        Native contacts are stored in NeuXelec as LPS physical coordinates
        in the patient T1 space. This method transforms them to MNI space,
        then exports them as RAS-like MNI x/y/z values in mm.
        """
        from neuxelec.utils.mni_coordinates import transform_contacts_t1_lps_to_mni_lps

        electrodes = getattr(self.state, "electrodes", []) or []

        if not electrodes:
            raise RuntimeError(
                "No electrodes are available in state.electrodes.\n"
                "Please reconstruct or load electrodes before exporting."
            )

        indices = self._selected_electrode_indices()

        # Safety fallback:
        # If the user selected 'Selected electrodes' but left the text field empty,
        # export all electrodes instead of returning no contacts.
        if not indices:
            indices = list(range(len(electrodes)))

        parcel1_img, parcel2_img, lut1, lut2 = self._get_parcellation_images_and_luts()
        include_parc = bool(self.chk_include_parc.isChecked())

        points_lps = []
        metas = []

        skipped_electrodes = []

        for elec_idx in indices:
            if elec_idx < 0 or elec_idx >= len(electrodes):
                continue

            elec = electrodes[elec_idx]

            elec_name = str(elec.get("name", f"Electrode_{elec_idx + 1}"))
            reference = str(elec.get("ref", ""))
            hemi = str(elec.get("hemisphere", ""))

            contacts_lps = elec.get("contacts_lps", []) or []

            if not contacts_lps:
                skipped_electrodes.append(elec_name)
                continue

            for ci, lps in enumerate(contacts_lps):
                try:
                    lps_tuple = tuple(float(v) for v in lps)

                    if len(lps_tuple) != 3:
                        continue

                except Exception:
                    continue

                meta = {
                    "electrode": elec_name,
                    "reference": reference,
                    "hemisphere": hemi,
                    "contact": f"{elec_name}{ci + 1}",
                    "native_lps": lps_tuple,
                }

                if include_parc and self.chk_parc1.isChecked():
                    label1 = _sample_parcellation_label(parcel1_img, lps_tuple)
                    meta["parcel1_label"] = "" if label1 is None else int(label1)
                    meta["parcel1_region"] = _lookup_lut_region(lut1, label1)

                if include_parc and self.chk_parc2.isChecked():
                    label2 = _sample_parcellation_label(parcel2_img, lps_tuple)
                    meta["parcel2_label"] = "" if label2 is None else int(label2)
                    meta["parcel2_region"] = _lookup_lut_region(lut2, label2)

                points_lps.append(lps_tuple)
                metas.append(meta)

        if not points_lps:
            msg = (
                "No contacts_lps coordinates were found for MNI export.\n\n"
                f"Number of electrodes in state: {len(electrodes)}\n"
                f"Selected electrode indices: {indices}\n"
            )

            if skipped_electrodes:
                msg += "\nElectrodes without contacts_lps:\n"
                msg += ", ".join(skipped_electrodes[:20])

            raise RuntimeError(msg)

        mni_lps_points = transform_contacts_t1_lps_to_mni_lps(
            self.state,
            points_lps,
            force_recompute_transform=False,
            progress_callback=progress_callback,
        )

        rows: list[dict[str, Any]] = []

        for meta, mni_lps in zip(metas, mni_lps_points):
            # Convert LPS to RAS-like MNI convention.
            mni_ras = _lps_to_ras(mni_lps)

            row: dict[str, Any] = {
                "electrode": meta["electrode"],
                "reference": meta["reference"],
                "hemisphere": meta["hemisphere"],
                "contact": meta["contact"],
                "type": "SEEG",
                "coordinate_system": "MNI152",
                "units": "mm",
                "x": float(mni_ras[0]),
                "y": float(mni_ras[1]),
                "z": float(mni_ras[2]),
            }

            if "parcel1_label" in meta:
                row["parcel1_label"] = meta["parcel1_label"]
                row["parcel1_region"] = meta["parcel1_region"]

            if "parcel2_label" in meta:
                row["parcel2_label"] = meta["parcel2_label"]
                row["parcel2_region"] = meta["parcel2_region"]

            rows.append(row)

        return rows

    def _create_export_progress_dialog(
        self,
        title: str = "Export coordinates",
    ) -> BidsExportProgressDialog:
        """
        Create the NeuXelec-styled BIDS export progress window.
        """
        dlg = BidsExportProgressDialog(title=title, parent=self)
        dlg.setLabelText("Preparing export...")
        dlg.setMaximum(100)
        dlg.setValue(0)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

        QApplication.processEvents()

        return dlg

    def _make_progress_callback(self, dlg: BidsExportProgressDialog):
        """
        Return a callback compatible with mni_coordinates.py and with native
        T1w BIDS export steps.
        """

        def _callback(message: str, value: int, maximum: int = 100):
            try:
                dlg.setMaximum(int(maximum))
                dlg.setLabelText(str(message))
                dlg.setValue(int(value))
                QApplication.processEvents()
            except Exception:
                pass

        return _callback

    def _default_export_directory(self) -> str:
        """
        Return the folder containing the patient's MRI.

        Priority:
            1. T1 NIfTI currently used by NeuXelec
            2. Original T1 source file or DICOM folder
            3. Last directory used in NeuXelec
            4. User home directory
        """
        candidates = (
            getattr(self.state, "t1_path", None),
            getattr(self.state, "t1_source_path", None),
            getattr(self.state, "last_browse_dir", None),
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

                # A path may refer to a file that is temporarily unavailable.
                # In that case, use its parent when it has a filename suffix.
                if path.suffix:
                    return str(path.parent)

            except Exception:
                continue

        return str(Path.home())

    def _get_cartool_t1_image(self) -> sitk.Image:
        """
        Return the native T1 image used to convert NeuXelec LPS coordinates
        to Cartool continuous image coordinates.
        """
        for attr_name in (
            "t1_sitk",
            "t1_img",
            "t1_image",
        ):
            image = getattr(self.state, attr_name, None)

            if isinstance(image, sitk.Image):
                return image

        for attr_name in (
            "t1_path",
            "t1_source_path",
        ):
            path_value = getattr(self.state, attr_name, None)

            if not path_value:
                continue

            try:
                path = Path(str(path_value))

                if path.exists():
                    return sitk.ReadImage(str(path))

            except Exception:
                pass

        raise RuntimeError(
            "The native T1 image is unavailable.\n\n"
            "Cartool ELS export requires the T1 image to convert "
            "contact LPS coordinates into continuous image coordinates."
        )

    def _build_cartool_els_groups(self) -> list[dict[str, Any]]:
        """
        Build electrode groups for a Cartool ELS file.

        NeuXelec stores contacts in physical LPS coordinates.
        Cartool ELS expects continuous image coordinates relative to the T1.
        """
        electrodes = getattr(self.state, "electrodes", []) or []
        selected_indices = self._selected_electrode_indices()

        # Same safety behavior as the other exports:
        # when the selection field is empty, export all electrodes.
        if not selected_indices and electrodes:
            selected_indices = list(range(len(electrodes)))

        t1_image = self._get_cartool_t1_image()

        groups: list[dict[str, Any]] = []

        for elec_idx in selected_indices:
            try:
                elec_idx = int(elec_idx)
            except Exception:
                continue

            if not (0 <= elec_idx < len(electrodes)):
                continue

            electrode = electrodes[elec_idx]

            electrode_name = str(
                electrode.get(
                    "name",
                    f"Electrode_{elec_idx + 1}",
                )
            ).strip()

            if not electrode_name:
                electrode_name = f"Electrode_{elec_idx + 1}"

            contacts_lps = electrode.get("contacts_lps", []) or []
            coordinates = []

            for lps in contacts_lps:
                try:
                    lps_point = (
                        float(lps[0]),
                        float(lps[1]),
                        float(lps[2]),
                    )

                    continuous_index = t1_image.TransformPhysicalPointToContinuousIndex(lps_point)

                    coordinates.append(
                        (
                            float(continuous_index[0]),
                            float(continuous_index[1]),
                            float(continuous_index[2]),
                        )
                    )

                except Exception:
                    continue

            if coordinates:
                groups.append(
                    {
                        "name": electrode_name,
                        "coordinates": coordinates,
                    }
                )

        return groups

    def _write_cartool_els(self, path: Path) -> None:
        """
        Write a Cartool ELS file.

        Structure:
            ES01
            total number of contacts
            number of electrodes
            electrode name
            number of contacts
            electrode type
            x y z
            ...
        """
        groups = self._build_cartool_els_groups()

        if not groups:
            raise RuntimeError("No electrode contacts are available for Cartool ELS export.")

        total_contacts = sum(len(group["coordinates"]) for group in groups)

        lines = [
            "ES01",
            str(total_contacts),
            str(len(groups)),
        ]

        for group in groups:
            electrode_name = str(group["name"])
            coordinates = group["coordinates"]

            lines.append(electrode_name)
            lines.append(str(len(coordinates)))

            # Cartool depth-electrode type used by your previous ELS export.
            lines.append("1")

            for x, y, z in coordinates:
                lines.append(f"{float(x):.2f}\t" f"{float(y):.2f}\t" f"{float(z):.2f}")

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Windows line endings, compatible with Cartool ELS files.
        path.write_text(
            "\r\n".join(lines) + "\r\n",
            encoding="utf-8",
        )

    def _on_export(self):
        formats = []

        if self.chk_txt.isChecked():
            formats.append("txt")

        if self.chk_csv.isChecked():
            formats.append("csv")

        if self.chk_tsv.isChecked():
            formats.append("tsv")

        if self.chk_json.isChecked():
            formats.append("json")

        if self.chk_els.isChecked():
            formats.append("els")

        if self.chk_bids.isChecked():
            formats.append("bids")

        if not formats:
            QMessageBox.warning(
                self, "Export coordinates", "Please select at least one export format."
            )
            return

        coord_systems = self._get_selected_coordinate_systems()
        classic_formats = [f for f in formats if f in ("csv", "tsv", "txt", "json")]

        # LPS/RAS/VOX checkboxes are required only for classic exports.
        # BIDS has its own coordinate-space options.
        if classic_formats and not coord_systems:
            QMessageBox.warning(
                self, "Export coordinates", "Please select at least one coordinate system."
            )
            return

        if not classic_formats:
            coord_systems = []

        default_export_dir = self._default_export_directory()

        out_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose export folder",
            default_export_dir,
            QFileDialog.ShowDirsOnly,
        )
        if not out_dir:
            return
        try:
            self.state.last_browse_dir = str(out_dir)
        except Exception:
            pass
        out_dir = Path(out_dir)
        patient = self._patient_filename_prefix()

        try:
            exported_any = False
            mni_electrodes_tsv = None

            # ---------------------------------------------------------
            # 1) Classic exports: CSV / TSV / TXT / JSON
            # ---------------------------------------------------------
            for coord_system in coord_systems:
                rows = self._build_rows(coord_system=coord_system)

                if not rows:
                    continue

                base_name = f"{patient}_electrode_coordinates_{coord_system}"

                if "csv" in formats:
                    self._write_csv(out_dir / f"{base_name}.csv", rows)
                    exported_any = True

                if "tsv" in formats:
                    self._write_tsv(out_dir / f"{base_name}.tsv", rows)
                    exported_any = True

                if "txt" in formats:
                    self._write_txt(out_dir / f"{base_name}.txt", rows)
                    exported_any = True

                if "json" in formats:
                    self._write_json(out_dir / f"{base_name}.json", rows)
                    exported_any = True

            if "els" in formats:
                els_path = out_dir / f"{patient}_electrodes.els"

                self._write_cartool_els(els_path)
                exported_any = True
            # ---------------------------------------------------------
            # 2) BIDS export
            #
            # IMPORTANT:
            # This block must be OUTSIDE the loop above.
            # Otherwise BIDS-only export never runs because coord_systems = [].
            # ---------------------------------------------------------
            if "bids" in formats:
                bids_space = self._get_bids_space()

                progress_title = (
                    "Export MNI coordinates" if bids_space == "MNI" else "Export BIDS coordinates"
                )

                progress_dlg = self._create_export_progress_dialog(progress_title)
                progress_callback = self._make_progress_callback(progress_dlg)

                # The MNI export runs ANTs normalization (~3 min). Drive the
                # bar smoothly over time to 98% and hold there until the real
                # completion forces 100%, instead of jumping with ANTs logs.
                if bids_space == "MNI":
                    progress_dlg.start_smooth_progress(180)

                try:
                    # -------------------------------------------------
                    # BIDS export in MNI space
                    # -------------------------------------------------
                    if bids_space == "MNI":
                        progress_callback(
                            "Collecting native SEEG contacts...",
                            5,
                            100,
                        )

                        bids_rows = self._build_rows_mni(progress_callback=progress_callback)

                        if bids_rows:
                            progress_callback(
                                "Writing BIDS MNI files...",
                                94,
                                100,
                            )

                            mni_electrodes_tsv = self._write_bids_export(
                                out_dir,
                                bids_rows,
                                coord_system="MNI152",
                                bids_space="MNI",
                            )

                            exported_any = True

                            progress_callback(
                                "Saving project metadata...",
                                98,
                                100,
                            )

                            try:
                                project_path = getattr(
                                    self.state,
                                    "project_path",
                                    None,
                                )
                                if project_path:
                                    from neuxelec.project_io import (
                                        save_project_json,
                                    )

                                    save_project_json(
                                        self.state,
                                        project_path,
                                    )
                            except Exception:
                                pass

                            progress_callback(
                                "MNI BIDS export completed.",
                                100,
                                100,
                            )

                    # -------------------------------------------------
                    # BIDS export in native T1w space
                    # -------------------------------------------------
                    else:
                        progress_callback(
                            "Collecting electrode contacts...",
                            15,
                            100,
                        )

                        bids_coord_system = self._get_bids_coordinate_system()
                        bids_rows = self._build_rows(coord_system=bids_coord_system)

                        if bids_rows:
                            progress_callback(
                                "Writing BIDS native T1w files...",
                                65,
                                100,
                            )

                            self._write_bids_export(
                                out_dir,
                                bids_rows,
                                coord_system=bids_coord_system,
                                bids_space="T1w",
                            )

                            exported_any = True

                            progress_callback(
                                "BIDS export completed.",
                                100,
                                100,
                            )

                    # Force one UI repaint at 100% so that the complete
                    # orange-to-pink gradient is rendered before closing.
                    if exported_any:
                        progress_dlg.setValue(100)
                        QApplication.processEvents()

                finally:
                    try:
                        progress_dlg.close()
                    except Exception:
                        pass

            if not exported_any:
                QMessageBox.warning(
                    self,
                    "Export coordinates",
                    "No contacts found to export.\n\n"
                    "Please check that electrodes are loaded and that they contain contacts_lps coordinates.",
                )
                return

            # Auto-load the just-exported MNI electrodes into the 3D MNI view,
            # so the user does not have to manually click "Load MNI
            # electrodes.tsv" for the patient they are already working on.
            # The loader checks MNI atlas mode if needed and renders the scene.
            if bids_space == "MNI" and mni_electrodes_tsv:
                try:
                    view3d_page = getattr(self.state, "view3d_page", None)
                    if view3d_page is not None and Path(mni_electrodes_tsv).exists():
                        view3d_page._load_mni_electrodes_from_paths([str(mni_electrodes_tsv)])
                except Exception:
                    pass

        except Exception as e:
            QMessageBox.critical(self, "Export coordinates", f"Export failed:\n{e}")
            return

        success_dialog = ExportSuccessDialog(
            output_directory=out_dir,
            parent=self,
        )

        result = success_dialog.exec()

        # Only the Done button closes the complete export window.
        # The top-right cross closes only the success popup.
        if result == QDialog.Accepted:
            self.accept()

    def _fieldnames(self, rows: list[dict[str, Any]]) -> list[str]:
        preferred = [
            "electrode",
            "reference",
            "hemisphere",
            "contact",
            "type",
            "coordinate_system",
            "units",
            "x",
            "y",
            "z",
            "parcel1_label",
            "parcel1_region",
            "parcel2_label",
            "parcel2_region",
        ]

        keys = set()
        for r in rows:
            keys.update(r.keys())

        return [k for k in preferred if k in keys] + sorted(k for k in keys if k not in preferred)

    def _toggle_bids_section(self, checked: bool):
        for child in self.grp_bids.findChildren(QRadioButton):
            child.setEnabled(bool(checked))

        self._update_bids_coordinate_convention_enabled()

    def _update_bids_coordinate_convention_enabled(self):
        """
        LPS/RAS/VOX convention is only meaningful for native T1 export.

        For MNI export, coordinates are exported as MNI RAS-like x/y/z in mm.
        """
        try:
            bids_on = bool(self.chk_bids.isChecked())
        except Exception:
            bids_on = False

        try:
            native_on = bool(self.radio_bids_native_t1.isChecked())
        except Exception:
            native_on = True

        enabled = bool(bids_on and native_on)

        for rb in (
            getattr(self, "radio_bids_lps", None),
            getattr(self, "radio_bids_ras", None),
            getattr(self, "radio_bids_vox", None),
        ):
            try:
                if rb is not None:
                    rb.setEnabled(enabled)
            except Exception:
                pass

    def _metadata(self):
        now = datetime.now()
        return {
            "type": "SEEG_electrode_coordinates",
            "patient": getattr(self.state, "patient_id", ""),
            "export_date": now.strftime("%Y-%m-%d"),
            "export_time": now.strftime("%H:%M:%S"),
        }

    def _write_csv(self, path: Path, rows: list[dict[str, Any]]):
        metadata = self._metadata()
        fieldnames = self._fieldnames(rows)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for k, v in metadata.items():
                writer.writerow([k, v])
            writer.writerow([])

            dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
            dict_writer.writeheader()
            for row in rows:
                dict_writer.writerow(_format_row_for_export(row, decimals=2))

    def _write_tsv(self, path: Path, rows: list[dict[str, Any]]):
        metadata = self._metadata()
        fieldnames = self._fieldnames(rows)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            for k, v in metadata.items():
                writer.writerow([k, v])
            writer.writerow([])

            dict_writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            dict_writer.writeheader()
            for row in rows:
                dict_writer.writerow(_format_row_for_export(row, decimals=2))

    def _write_json(self, path: Path, rows: list[dict[str, Any]]):
        metadata = self._metadata()

        coordinate_system = ""
        try:
            if rows:
                coordinate_system = str(rows[0].get("coordinate_system", ""))
        except Exception:
            coordinate_system = ""
        formatted_rows = [_format_row_for_export(row, decimals=2) for row in rows]
        payload = {
            **metadata,
            "coordinate_system": coordinate_system,
            "n_contacts": len(rows),
            "contacts": formatted_rows,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _write_txt(self, path: Path, rows: list[dict[str, Any]]):
        metadata = self._metadata()
        fieldnames = self._fieldnames(rows)

        coord_system = ""
        try:
            if rows:
                coord_system = str(rows[0].get("coordinate_system", ""))
        except Exception:
            coord_system = ""

        with open(path, "w", encoding="utf-8") as f:
            f.write("SEEG_electrode_coordinates\n")
            f.write("=" * 40 + "\n")
            f.write(f"Patient: {metadata.get('patient', '')}\n")
            f.write(f"Export date: {metadata.get('export_date', '')}\n")
            f.write(f"Export time: {metadata.get('export_time', '')}\n")
            if coord_system:
                f.write(f"Coordinate system: {coord_system}\n")
            f.write("=" * 40 + "\n\n")

            current_elec = None

            for row in rows:
                row = _format_row_for_export(row, decimals=2)
                elec = row.get("electrode", "")

                if elec != current_elec:
                    current_elec = elec
                    ref = row.get("reference", "")
                    hemi = row.get("hemisphere", "")
                    f.write(f"\n[{elec}]")
                    if ref:
                        f.write(f" | reference={ref}")
                    if hemi:
                        f.write(f" | hemisphere={hemi}")
                    f.write("\n")

                values = []
                for k in fieldnames:
                    if k in ("electrode", "reference", "hemisphere"):
                        continue
                    values.append(f"{k}={row.get(k, '')}")

                f.write("  " + " | ".join(values) + "\n")

    # ------------------------------------------------------------------
    # BIDS
    # ------------------------------------------------------------------
    def _safe_bids_subject(self) -> str:
        patient = str(getattr(self.state, "patient_id", "") or "unknown")
        patient = "".join(c for c in patient if c.isalnum())

        if not patient:
            patient = "unknown"

        if patient.lower().startswith("sub"):
            patient = patient[3:]

        return f"sub-{patient}"

    def _copy_if_exists(self, src, dst: Path):
        if not src:
            return

        try:
            src = Path(src)
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
        except Exception:
            pass

    def _write_bids_export(
        self,
        out_dir: Path,
        rows: list[dict[str, Any]],
        coord_system: str = "LPS",
        bids_space: str = "T1w",
    ):
        coord_system = str(coord_system or "LPS").upper().strip()

        sub = self._safe_bids_subject()

        is_mni = str(bids_space or "").upper() == "MNI"
        space_label = "MNI152NLin2009cAsym" if is_mni else "T1w"

        dataset_dir = out_dir
        sub_dir = dataset_dir / sub
        anat_dir = sub_dir / "anat"
        ieeg_dir = sub_dir / "ieeg"
        deriv_dir = dataset_dir / "derivatives" / "neuxelec" / sub / "anat"

        anat_dir.mkdir(parents=True, exist_ok=True)
        ieeg_dir.mkdir(parents=True, exist_ok=True)
        deriv_dir.mkdir(parents=True, exist_ok=True)

        metadata = self._metadata()

        dataset_description = {
            "Name": "NeuXelec SEEG electrode coordinates export",
            "BIDSVersion": "1.9.0",
            "DatasetType": "derivative",
            "GeneratedBy": [
                {
                    "Name": "NeuXelec",
                    "Description": "Multimodal SEEG electrode localization and visualization",
                }
            ],
            "ExportDate": metadata["export_date"],
            "ExportTime": metadata["export_time"],
        }

        with open(dataset_dir / "dataset_description.json", "w", encoding="utf-8") as f:
            json.dump(dataset_description, f, indent=2, ensure_ascii=False)

        self._copy_if_exists(
            getattr(self.state, "t1_path", None) or getattr(self.state, "t1_source_path", None),
            anat_dir / f"{sub}_T1w.nii.gz",
        )

        self._copy_if_exists(
            getattr(self.state, "ct_coreg_path", None) or getattr(self.state, "ct_path", None),
            deriv_dir / f"{sub}_desc-coregCT_space-T1w_ct.nii.gz",
        )

        self._copy_if_exists(
            getattr(self.state, "pet_coreg_path", None) or getattr(self.state, "pet_path", None),
            deriv_dir / f"{sub}_desc-coregPET_space-T1w_pet.nii.gz",
        )

        self._copy_if_exists(
            getattr(self.state, "siscom_coreg_path", None)
            or getattr(self.state, "siscom_path", None),
            deriv_dir / f"{sub}_desc-SISCOM_space-T1w_siscom.nii.gz",
        )

        self._copy_if_exists(
            getattr(self.state, "parcel1_path", None),
            deriv_dir / f"{sub}_desc-parcellation1_space-T1w_dseg.nii.gz",
        )

        self._copy_if_exists(
            getattr(self.state, "parcel2_path", None),
            deriv_dir / f"{sub}_desc-parcellation2_space-T1w_dseg.nii.gz",
        )

        if is_mni:
            self._copy_if_exists(
                getattr(self.state, "t1_to_mni_affine_path", None),
                deriv_dir / f"{sub}_from-T1w_to-{space_label}_xfm.mat",
            )

            self._copy_if_exists(
                getattr(self.state, "t1_to_mni_warp_path", None),
                deriv_dir / f"{sub}_from-T1w_to-{space_label}_warp.nii.gz",
            )

            self._copy_if_exists(
                getattr(self.state, "t1_to_mni_inverse_warp_path", None),
                deriv_dir / f"{sub}_from-{space_label}_to-T1w_inversewarp.nii.gz",
            )

            self._copy_if_exists(
                getattr(self.state, "t1_to_mni_warped_path", None),
                deriv_dir / f"{sub}_space-{space_label}_desc-warpedT1w.nii.gz",
            )

        electrodes_tsv = ieeg_dir / f"{sub}_space-{space_label}_electrodes.tsv"

        bids_rows = []
        for row in rows:
            formatted_row = _format_row_for_export(row, decimals=2)

            bids_rows.append(
                {
                    "name": formatted_row.get("contact", ""),
                    "x": formatted_row.get("x", "n/a"),
                    "y": formatted_row.get("y", "n/a"),
                    "z": formatted_row.get("z", "n/a"),
                    "size": 5,
                    "type": "depth SEEG",
                    "material": "Ti",
                    "manufacturer": "DIXI",
                    "group": formatted_row.get("electrode", ""),
                    "hemisphere": formatted_row.get("hemisphere", ""),
                    "reference": formatted_row.get("reference", ""),
                    "parcel1_region": formatted_row.get("parcel1_region", ""),
                    "parcel2_region": formatted_row.get("parcel2_region", ""),
                }
            )

        bids_fields = [
            "name",
            "x",
            "y",
            "z",
            "size",
            "type",
            "material",
            "manufacturer",
            "group",
            "hemisphere",
            "reference",
            "parcel1_region",
            "parcel2_region",
        ]

        with open(electrodes_tsv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=bids_fields, delimiter="\t")
            writer.writeheader()
            for r in bids_rows:
                writer.writerow(r)

        units = "voxel" if coord_system == "VOX" else "mm"

        if is_mni:
            coordsystem = {
                "IntendedFor": f"{sub}/anat/{sub}_T1w.nii.gz",
                "iEEGCoordinateSystem": "MNI152NLin2009cAsym",
                "iEEGCoordinateSystemDescription": (
                    "MNI coordinates obtained by applying the NeuXelec T1-to-MNI transform "
                    "to native T1 SEEG contact coordinates."
                ),
                "iEEGCoordinateUnits": "mm",
                "CoordinateSystemType": "MNI",
                "CoordinateSpace": "MNI152NLin2009cAsym",
                "CoordinateSpaceDescription": (
                    "Common MNI atlas space. Native T1 contact coordinates were transformed "
                    "to MNI using ANTs registration between the patient T1 MRI and the MNI template."
                ),
                "NativeCoordinateSource": "T1w",
                "NativeCoordinateConvention": "LPS",
                "ExportCoordinateConvention": "MNI RAS-like x/y/z in mm",
                "MNIExportAvailable": True,
                "MNITransformSoftware": "ANTs",
                "MNITransformType": "Affine + SyN warp",
                "T1ToMNIAffineTransform": getattr(self.state, "t1_to_mni_affine_path", ""),
                "T1ToMNIWarpTransform": getattr(self.state, "t1_to_mni_warp_path", ""),
                "T1ToMNIInverseWarpTransform": getattr(
                    self.state, "t1_to_mni_inverse_warp_path", ""
                ),
                "MNITemplate": getattr(self.state, "mni_template_path", ""),
                "MNISpaceName": getattr(self.state, "mni_space_name", "MNI152NLin2009cAsym"),
                "iEEGCoordinateProcessingDescription": (
                    "SEEG contacts were reconstructed from post-implantation CT coregistered "
                    "to the patient T1 MRI. Native T1 LPS coordinates were then transformed "
                    "to MNI space using ANTs transforms generated by NeuXelec. Coordinates in "
                    "electrodes.tsv are written as MNI RAS-like x/y/z values in millimetres."
                ),
                "iEEGCoordinateProcessingReference": "NeuXelec",
                "GeneratedBy": "NeuXelec",
                "ExportDate": metadata["export_date"],
                "ExportTime": metadata["export_time"],
            }

        else:
            coordsystem = {
                "IntendedFor": f"{sub}/anat/{sub}_T1w.nii.gz",
                "iEEGCoordinateSystem": "Other",
                "iEEGCoordinateSystemDescription": (
                    f"Patient native T1 MRI space. Coordinates are exported in {coord_system} convention."
                ),
                "iEEGCoordinateUnits": units,
                "CoordinateSystemType": coord_system,
                "CoordinateSpace": "native_T1",
                "CoordinateSpaceDescription": (
                    "Native anatomical T1 space of the individual patient, after post-implantation CT "
                    "coregistration to the T1 MRI."
                ),
                "MNIExportAvailable": False,
                "MNICoordinateSystem": "Not exported",
                "MNICoordinateSystemDescription": (
                    "MNI coordinates are not included in this export because native T1 space was selected."
                ),
                "iEEGCoordinateProcessingDescription": (
                    "SEEG contacts were reconstructed from post-implantation CT coregistered "
                    "to the patient T1 MRI. Coordinates are exported from NeuXelec in native "
                    f"T1 space using {coord_system} coordinates."
                ),
                "iEEGCoordinateProcessingReference": "NeuXelec",
                "GeneratedBy": "NeuXelec",
                "ExportDate": metadata["export_date"],
                "ExportTime": metadata["export_time"],
            }

        with open(
            ieeg_dir / f"{sub}_space-{space_label}_coordsystem.json", "w", encoding="utf-8"
        ) as f:
            json.dump(coordsystem, f, indent=2, ensure_ascii=False)

        summary = {
            "type": "SEEG_electrode_coordinates",
            "patient": getattr(self.state, "patient_id", ""),
            "subject": sub,
            "coordinate_system": coord_system,
            "bids_space": space_label,
            "export_date": metadata["export_date"],
            "export_time": metadata["export_time"],
            "n_contacts": len(rows),
            "coordinates": [_format_row_for_export(row, decimals=2) for row in rows],
        }

        summary_path = (
            dataset_dir / "derivatives" / "neuxelec" / sub / f"{sub}_desc-neuxelec_coordinates.json"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Return the written electrodes.tsv path so callers can, for example,
        # auto-load the exported MNI electrodes into the 3D MNI view.
        return electrodes_tsv
