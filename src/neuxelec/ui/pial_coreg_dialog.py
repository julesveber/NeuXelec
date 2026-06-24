import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from PySide6.QtCore import QEvent, QPoint, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from neuxelec.coregistration import rigid_coreg_to_fixed
from neuxelec.ui.neuxelec_message_dialog import NeuXelecMessageDialog


class PialCoregWorker(QThread):
    progress = Signal(int)
    finished_ok = Signal(object, object)  # (lh_data, rh_data)
    failed = Signal(str)

    def __init__(self, lh_path: str, rh_path: str, orig_mgz_path: str, t1_path: str):
        super().__init__()
        self.lh_path = lh_path
        self.rh_path = rh_path
        self.orig_mgz_path = orig_mgz_path
        self.t1_path = t1_path

    def _apply_sitk_transform_to_points(
        self, pts: np.ndarray, transform: sitk.Transform
    ) -> np.ndarray:
        out = []
        for p in pts:
            out.append(transform.TransformPoint(tuple(float(x) for x in p)))
        return np.asarray(out, dtype=np.float32)

    def run(self):
        try:
            self.progress.emit(10)

            # Read FreeSurfer surfaces (vertices in tkRAS)
            lh_v, lh_f = nib.freesurfer.read_geometry(self.lh_path)
            rh_v, rh_f = nib.freesurfer.read_geometry(self.rh_path)

            self.progress.emit(25)

            # Read orig.mgz to get FreeSurfer tkRAS -> scannerRAS conversion
            orig = nib.load(self.orig_mgz_path)
            Norig = orig.header.get_vox2ras()
            Torig = orig.header.get_vox2ras_tkr()

            # Step 1: tkRAS -> scannerRAS
            M_fs = Norig @ np.linalg.inv(Torig)
            lh_v = nib.affines.apply_affine(M_fs, lh_v)
            rh_v = nib.affines.apply_affine(M_fs, rh_v)

            self.progress.emit(40)

            # Step 2: scannerRAS -> scannerLPS
            lh_v[:, 0] *= -1.0
            lh_v[:, 1] *= -1.0
            rh_v[:, 0] *= -1.0
            rh_v[:, 1] *= -1.0

            self.progress.emit(50)

            # Step 3: convert orig.mgz to temporary NIfTI for ANTs/SimpleITK
            tmp_dir = tempfile.mkdtemp(prefix="neuxelec_pial_coreg_")
            orig_nii_path = str(Path(tmp_dir) / "orig_for_ants.nii.gz")
            nib.save(orig, orig_nii_path)

            # Step 4: ANTs rigid registration orig -> T1
            rigid_coreg_to_fixed(
                fixed_path=self.t1_path,
                moving_path=orig_nii_path,
                transforms_dir=tmp_dir,
                moving_modality="AUTO",
                use_ants=True,
            )

            affine_path = str(Path(tmp_dir) / "AUTO_to_T1_0GenericAffine.mat")
            if not Path(affine_path).exists():
                raise RuntimeError(f"ANTs affine transform not found: {affine_path}")

            self.progress.emit(75)

            tx = sitk.ReadTransform(affine_path)

            # Step 5: apply ANTs affine to vertices in LPS
            lh_v = self._apply_sitk_transform_to_points(lh_v, tx)
            rh_v = self._apply_sitk_transform_to_points(rh_v, tx)

            self.progress.emit(98)

            # IMPORTANT:
            # Save output vertices in LPS physical coordinates of the T1.
            # The dialog itself will switch to 100% only when the worker
            # has really finished.
            self.finished_ok.emit(
                (lh_v.astype(np.float32), lh_f),
                (rh_v.astype(np.float32), rh_f),
            )

        except Exception as e:
            self.failed.emit(str(e))


class NeuXelecPialHeader(QFrame):
    """
    Frameless NeuXelec header with a custom close button.
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
        self.btn_close_window.clicked.connect(self.dialog._request_close)

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


class PialCoregPromptDialog(QDialog):
    """
    Styled NeuXelec confirmation dialog displayed once both raw pial
    surfaces have been loaded.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._positioned_once = False

        self.setWindowTitle("Pial surfaces")
        self.setModal(True)

        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setSizeGripEnabled(False)

        self.setFixedSize(500, 280)

        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.dialog_shell = QFrame()
        self.dialog_shell.setObjectName("dialogShell")

        root = QVBoxLayout(self.dialog_shell)
        root.setContentsMargins(14, 8, 14, 14)
        root.setSpacing(10)

        outer_layout.addWidget(self.dialog_shell)

        self.custom_header = NeuXelecPialHeader(self)
        root.addWidget(self.custom_header)

        content = QVBoxLayout()
        content.setContentsMargins(18, 4, 18, 6)
        content.setSpacing(12)

        self.lbl_title = QLabel("PIAL SURFACES")
        self.lbl_title.setObjectName("dialogTitle")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        content.addWidget(self.lbl_title)

        self.lbl_subtitle = QLabel(
            "Do you want to coregister the loaded LH/RH pial surfaces to the T1?"
        )
        self.lbl_subtitle.setObjectName("dialogSubtitle")
        self.lbl_subtitle.setWordWrap(True)
        self.lbl_subtitle.setAlignment(Qt.AlignCenter)
        content.addWidget(self.lbl_subtitle)

        self.info_frame = QFrame()
        self.info_frame.setObjectName("informationCard")

        info_layout = QVBoxLayout(self.info_frame)
        info_layout.setContentsMargins(14, 10, 14, 10)

        self.lbl_info = QLabel(
            "Choose Coregister to align the FreeSurfer surfaces with the "
            "reference T1, or Keep original to display the raw surfaces."
        )
        self.lbl_info.setObjectName("informationLabel")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        info_layout.addWidget(self.lbl_info)
        content.addWidget(self.info_frame)

        root.addLayout(content)
        root.addStretch(1)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(12, 0, 12, 4)
        buttons.setSpacing(10)

        self.btn_keep = QPushButton("Keep original")
        self.btn_keep.setObjectName("secondaryButton")
        self.btn_keep.setCursor(Qt.PointingHandCursor)
        self.btn_keep.setMinimumHeight(42)
        self.btn_keep.clicked.connect(self.reject)

        self.btn_coregister = QPushButton("Coregister")
        self.btn_coregister.setObjectName("primaryButton")
        self.btn_coregister.setCursor(Qt.PointingHandCursor)
        self.btn_coregister.setMinimumHeight(42)
        self.btn_coregister.setMinimumWidth(126)
        self.btn_coregister.clicked.connect(self.accept)

        buttons.addStretch(1)
        buttons.addWidget(self.btn_keep)
        buttons.addWidget(self.btn_coregister)

        root.addLayout(buttons)

    def _request_close(self) -> None:
        self.reject()

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
                color: #CFCFD6;
                font-size: 12px;
                font-weight: 500;
            }

            QFrame#informationCard {
                background-color: #10121A;
                border: 1px solid #2B2D38;
                border-radius: 10px;
            }

            QLabel#informationLabel {
                color: #8E8E98;
                font-size: 11px;
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


class PialCoregDialog(QDialog):
    def __init__(self, lh_path: str, rh_path: str, t1_path: str, parent=None):
        super().__init__(parent)

        self.lh_path = lh_path
        self.rh_path = rh_path
        self.t1_path = t1_path
        self.lh_out = None
        self.rh_out = None

        self.worker = None
        self._busy = False
        self._positioned_once = False
        self._resize_margin = 8

        # Continuous estimated progress for pial coregistration.
        # Expected duration: about 2 minutes.
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(250)
        self._progress_timer.timeout.connect(self._update_estimated_progress)

        self._progress_started_at = 0.0
        self._progress_duration_s = 120.0
        self._progress_max_before_done = 98
        self._worker_reported_progress = 0

        self.setWindowTitle("Pial Coregistration")

        # Non-modal floating window: the pial coregistration runs in its
        # worker thread while the main NeuXelec window remains interactive.
        self.setModal(False)

        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setSizeGripEnabled(False)

        self.setMinimumSize(450, 320)
        self._set_adapted_initial_size()

        self._build_ui()
        self._apply_style()

        self.btn_yes.clicked.connect(self._on_yes)
        self.btn_no.clicked.connect(self.reject)

    # ============================================================
    # Window geometry
    # ============================================================

    # ============================================================
    # UI construction
    # ============================================================

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.dialog_shell = QFrame()
        self.dialog_shell.setObjectName("dialogShell")
        self.dialog_shell.setMouseTracking(True)
        self.dialog_shell.installEventFilter(self)

        root = QVBoxLayout(self.dialog_shell)
        root.setContentsMargins(14, 8, 14, 14)
        root.setSpacing(10)

        outer_layout.addWidget(self.dialog_shell)

        # ---------------------------------------------------------
        # Custom frameless header
        # ---------------------------------------------------------
        self.custom_header = NeuXelecPialHeader(self)
        root.addWidget(self.custom_header)

        # ---------------------------------------------------------
        # Scrollable central content
        # ---------------------------------------------------------
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("pialCoregScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("pialCoregScrollContent")
        self.scroll_content.setMinimumHeight(145)

        content = QVBoxLayout(self.scroll_content)
        content.setContentsMargins(12, 4, 12, 10)
        content.setSpacing(10)

        # ---------------------------------------------------------
        # Title
        # ---------------------------------------------------------
        self.lbl_title = QLabel("PIAL COREGISTRATION")
        self.lbl_title.setObjectName("dialogTitle")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        content.addWidget(self.lbl_title)

        self.lbl_subtitle = QLabel("Align FreeSurfer pial surfaces with the reference T1 volume")
        self.lbl_subtitle.setObjectName("dialogSubtitle")
        self.lbl_subtitle.setAlignment(Qt.AlignCenter)
        self.lbl_subtitle.setWordWrap(True)
        content.addWidget(self.lbl_subtitle)

        # ---------------------------------------------------------
        # Information card
        # ---------------------------------------------------------
        self.info_frame = QFrame()
        self.info_frame.setObjectName("informationCard")
        self.info_frame.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )

        info_layout = QVBoxLayout(self.info_frame)
        info_layout.setContentsMargins(16, 12, 16, 12)
        info_layout.setSpacing(8)

        self.lbl = QLabel(
            "Coregister the left and right pial surfaces to the imported T1.\n"
            "You will be asked to select the subject's FreeSurfer orig.mgz file."
        )
        self.lbl.setObjectName("informationLabel")
        self.lbl.setWordWrap(True)
        self.lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        info_layout.addWidget(self.lbl)
        content.addWidget(self.info_frame)

        # ---------------------------------------------------------
        # Progress
        # ---------------------------------------------------------
        self.progress = QProgressBar()
        self.progress.setObjectName("pialProgressBar")
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.progress.setMinimumHeight(20)

        content.addWidget(self.progress)
        content.addStretch(1)

        self.scroll_area.setWidget(self.scroll_content)
        root.addWidget(self.scroll_area, 1)

        # ---------------------------------------------------------
        # Fixed bottom actions
        # ---------------------------------------------------------
        buttons = QHBoxLayout()
        buttons.setContentsMargins(12, 0, 12, 4)
        buttons.setSpacing(10)

        self.btn_no = QPushButton("Cancel")
        self.btn_no.setObjectName("secondaryButton")
        self.btn_no.setCursor(Qt.PointingHandCursor)
        self.btn_no.setMinimumHeight(42)
        self.btn_no.setMinimumWidth(108)

        self.btn_yes = QPushButton("Coregister")
        self.btn_yes.setObjectName("primaryButton")
        self.btn_yes.setCursor(Qt.PointingHandCursor)
        self.btn_yes.setMinimumHeight(42)
        self.btn_yes.setMinimumWidth(126)

        buttons.addStretch(1)
        buttons.addWidget(self.btn_no)
        buttons.addWidget(self.btn_yes)

        root.addLayout(buttons)

    def _request_close(self) -> None:
        """
        Prevent closing the dialog while ANTs/SimpleITK processing is running.
        """
        if self._busy:
            return

        self.reject()

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)

        self.btn_yes.setEnabled(not self._busy)
        self.btn_no.setEnabled(not self._busy)
        self.custom_header.btn_close_window.setEnabled(not self._busy)

        if self._busy:
            self.lbl.setText(
                "Coregistration in progress…\n"
                "You can continue using NeuXelec while the pial surfaces are processed."
            )

        else:
            try:
                self._progress_timer.stop()
            except Exception:
                pass

            self.lbl.setText(
                "Coregister the left and right pial surfaces to the imported T1.\n"
                "You will be asked to select the subject's FreeSurfer orig.mgz file."
            )

    def _set_adapted_initial_size(self) -> None:
        """
        Open the dialog at a comfortable size while keeping it inside the
        available screen geometry. It remains manually resizable afterwards.
        """
        preferred_width = 450
        preferred_height = 320

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

            QPushButton#closeWindowButton:disabled {
                color: #62646E;
                background-color: transparent;
                border: 1px solid transparent;
            }

            QWidget#pialCoregScrollContent {
                background-color: transparent;
            }

            QScrollArea#pialCoregScrollArea {
                background-color: transparent;
                border: none;
            }

            QScrollArea#pialCoregScrollArea > QWidget > QWidget {
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

            QFrame#informationCard {
                background-color: #10121A;
                border: 1px solid #2B2D38;
                border-radius: 12px;
            }

            QLabel#informationLabel {
                color: #CFCFD6;
                font-size: 12px;
                font-weight: 500;
                line-height: 1.4;
            }

            QProgressBar#pialProgressBar {
                color: #F2F2F5;
                background-color: #171922;
                border: 1px solid #2B2D38;
                border-radius: 10px;
                text-align: center;
                font-size: 11px;
                font-weight: 600;
                min-height: 22px;
            }

            QProgressBar#pialProgressBar::chunk {
                margin: 0px;
                border: none;
                border-radius: 9px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
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

            QPushButton#secondaryButton:disabled,
            QPushButton#primaryButton:disabled {
                color: #62646E;
                background: #121319;
                border: 1px solid #20222A;
            }

            QScrollArea#pialCoregScrollArea QScrollBar:vertical {
                background-color: #111218;
                width: 13px;
                margin: 5px 2px 5px 2px;
                border: none;
                border-radius: 6px;
            }

            QScrollArea#pialCoregScrollArea QScrollBar::handle:vertical {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
                border-radius: 6px;
                min-height: 22px;
            }

            QScrollArea#pialCoregScrollArea QScrollBar::handle:vertical:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF922B,
                    stop:1 #FF33B8
                );
            }

            QScrollArea#pialCoregScrollArea QScrollBar::add-line:vertical,
            QScrollArea#pialCoregScrollArea QScrollBar::sub-line:vertical {
                height: 0px;
                background: transparent;
                border: none;
            }

            QScrollArea#pialCoregScrollArea QScrollBar::add-page:vertical,
            QScrollArea#pialCoregScrollArea QScrollBar::sub-page:vertical {
                background: transparent;
            }
            """)

    def _start_progress_animation(self) -> None:
        """
        Start a continuous estimated progression over approximately 2 minutes.

        The bar progresses smoothly up to 98%, then waits for the real worker
        to finish before displaying 100%.
        """
        self._progress_started_at = time.monotonic()
        self._progress_duration_s = 120.0
        self._progress_max_before_done = 98
        self._worker_reported_progress = 0

        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Preparing pial coregistration... %p%")

        self._progress_timer.start()

    def _progress_stage_text(self, value: int) -> str:
        """
        Human-readable text displayed inside the progress bar.
        """
        if value < 8:
            return "Preparing"

        if value < 20:
            return "Loading FreeSurfer surfaces"

        if value < 35:
            return "Converting surface coordinates"

        if value < 80:
            return "Running rigid coregistration"

        if value < 94:
            return "Transforming pial surfaces"

        return "Finalizing"

    def _update_estimated_progress(self) -> None:
        """
        Continuously update the progress bar.

        The estimated progression is linear over 120 seconds and is combined
        with any real progress reported by the worker. It never exceeds 98%
        before the worker has actually finished.
        """
        if not self._busy:
            return

        try:
            elapsed = max(
                0.0,
                time.monotonic() - self._progress_started_at,
            )

            duration = max(
                1.0,
                float(self._progress_duration_s),
            )

            fraction = min(elapsed / duration, 1.0)

            estimated = int(round(2 + (self._progress_max_before_done - 2) * fraction))

            estimated = min(
                self._progress_max_before_done,
                max(2, estimated),
            )

            real_progress = min(
                self._progress_max_before_done,
                int(self._worker_reported_progress),
            )

            value = max(
                int(self.progress.value()),
                estimated,
                real_progress,
            )

            value = min(
                self._progress_max_before_done,
                value,
            )

            self.progress.setValue(value)
            self.progress.setFormat(f"{self._progress_stage_text(value)}... %p%")

        except Exception:
            pass

    def _on_worker_progress(self, value: int) -> None:
        """
        Keep worker progress as a lower bound without allowing it to display
        100% before the worker has really completed.
        """
        try:
            value = int(value)
        except Exception:
            return

        self._worker_reported_progress = min(
            self._progress_max_before_done,
            max(
                int(self._worker_reported_progress),
                value,
            ),
        )

    def _finish_progress_animation(
        self,
        success: bool,
    ) -> None:
        """
        Stop the estimated progress animation and show the final state.
        """
        try:
            self._progress_timer.stop()
        except Exception:
            pass

        if success:
            self.progress.setValue(100)
            self.progress.setFormat("Completed · 100%")
        else:
            self.progress.setFormat("Coregistration failed")

    def _on_yes(self) -> None:
        start_dir = str(Path(self.lh_path).parent)

        orig_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select FreeSurfer orig.mgz",
            start_dir,
            "MGH/MGZ (*.mgz *.mgh);;All files (*.*)",
        )

        if not orig_path:
            return

        self._set_busy(True)
        self._start_progress_animation()

        self.worker = PialCoregWorker(
            lh_path=self.lh_path,
            rh_path=self.rh_path,
            orig_mgz_path=orig_path,
            t1_path=self.t1_path,
        )

        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished_ok.connect(self._done)
        self.worker.failed.connect(self._fail)

        self.worker.start()

    def _done(self, lh_data, rh_data) -> None:
        # The real worker has completed: the progress bar can now reach 100%.
        self._finish_progress_animation(success=True)

        self.progress.repaint()
        QApplication.processEvents()

        lh_v, lh_f = lh_data
        rh_v, rh_f = rh_data

        out_lh, _ = QFileDialog.getSaveFileName(
            self,
            "Save coregistered LH pial",
            str(Path(self.lh_path).with_name("lh_coreg.pial")),
            "FreeSurfer surface (*.pial *.surf);;All files (*.*)",
        )
        if not out_lh:
            self._set_busy(False)
            return

        out_rh, _ = QFileDialog.getSaveFileName(
            self,
            "Save coregistered RH pial",
            str(Path(self.rh_path).with_name("rh_coreg.pial")),
            "FreeSurfer surface (*.pial *.surf);;All files (*.*)",
        )
        if not out_rh:
            self._set_busy(False)
            return

        try:
            nib.freesurfer.write_geometry(out_lh, lh_v, lh_f)
            nib.freesurfer.write_geometry(out_rh, rh_v, rh_f)
        except Exception as e:
            self._finish_progress_animation(success=False)
            self._set_busy(False)

            NeuXelecMessageDialog.critical(
                self,
                "Save failed",
                ("The coregistered pial surfaces could not be saved.\n\n" f"Details:\n{e}"),
            )
            return

        self.lh_out = out_lh
        self.rh_out = out_rh

        NeuXelecMessageDialog.information(
            self,
            "Pial coregistration completed",
            (
                "The coregistered pial surfaces were saved successfully.\n\n"
                f"Left hemisphere:\n{out_lh}\n\n"
                f"Right hemisphere:\n{out_rh}"
            ),
        )

        self.accept()

    def _fail(self, msg: str) -> None:
        self._finish_progress_animation(success=False)
        self._set_busy(False)

        NeuXelecMessageDialog.critical(
            self,
            "Pial coregistration failed",
            ("The pial surfaces could not be coregistered to the T1.\n\n" f"Details:\n{msg}"),
        )

        self.reject()
