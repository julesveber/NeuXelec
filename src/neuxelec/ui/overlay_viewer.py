from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from PySide6.QtCore import QEvent, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import (
    QFont,
    QGuiApplication,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


# ----------------------------
# Helpers
# ----------------------------
def _sitk_to_np_zyx(img: sitk.Image) -> np.ndarray:
    # SimpleITK -> numpy in (z,y,x)
    return sitk.GetArrayFromImage(img).astype(np.float32, copy=False)


def _norm01(a: np.ndarray) -> np.ndarray:
    lo = np.percentile(a, 1)
    hi = np.percentile(a, 99)
    if hi <= lo:
        hi = lo + 1.0
    a = (a - lo) / (hi - lo)
    return np.clip(a, 0.0, 1.0)


def _rgb_to_qpixmap(rgb_u8: np.ndarray) -> QPixmap:
    rgb_u8 = np.ascontiguousarray(rgb_u8)
    h, w, _ = rgb_u8.shape
    qimg = QImage(rgb_u8.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _rot90_k(img2d: np.ndarray, k: int) -> np.ndarray:
    k = int(k) % 4
    if k == 0:
        return img2d
    return np.rot90(img2d, k=k)


def _flip_lr(img: np.ndarray) -> np.ndarray:
    # Works for (H,W) and (H,W,3)
    return img[:, ::-1, ...]


def _apply_flip_to_x(x: float, w: int, do_flip: bool) -> float:
    if not do_flip:
        return x
    return float((w - 1) - x)


def _draw_lr_markers(pm: QPixmap, left_text: str = "L", right_text: str = "R") -> QPixmap:
    out = QPixmap(pm)
    p = QPainter(out)
    try:
        p.setRenderHint(QPainter.Antialiasing, True)

        pen = QPen(Qt.red)
        pen.setWidth(2)
        p.setPen(pen)

        font = QFont()
        font.setBold(True)
        font.setPointSize(18)
        p.setFont(font)

        margin = 10
        baseline = margin + 18

        p.drawText(margin, baseline, left_text)

        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(right_text)
        p.drawText(max(margin, out.width() - margin - tw), baseline, right_text)

    finally:
        p.end()

    return out


def _map_display_to_base(
    px: float, py: float, base_h: int, base_w: int, label_w: int, label_h: int
) -> tuple[float, float]:
    """
    Maps a click in QLabel coordinates to pixel coordinates in the displayed pixmap,
    then to base image coords (same shape as base slice), assuming pixmap is scaled with KeepAspectRatio and centered.
    Returns (x_base, y_base) in base pixel space.
    """
    if base_h <= 0 or base_w <= 0:
        return 0.0, 0.0

    scale = min(label_w / base_w, label_h / base_h) if base_w and base_h else 1.0
    disp_w = base_w * scale
    disp_h = base_h * scale

    off_x = (label_w - disp_w) / 2.0
    off_y = (label_h - disp_h) / 2.0

    x_disp = (px - off_x) / max(scale, 1e-9)
    y_disp = (py - off_y) / max(scale, 1e-9)

    x_base = float(np.clip(x_disp, 0, base_w - 1))
    y_base = float(np.clip(y_disp, 0, base_h - 1))
    return x_base, y_base


def _rot_map_display_to_base(px: float, py: float, h0: int, w0: int, k: int) -> tuple[float, float]:
    """
    Convert click coords on a rotated image to coords in the un-rotated base slice.
    Input px/py are in the rotated image pixel space.
    Returns (x0, y0) in base image coordinates.
    """
    k = int(k) % 4
    if k == 0:
        return px, py
    if k == 1:
        return (w0 - 1 - py), px
    if k == 2:
        return (w0 - 1 - px), (h0 - 1 - py)
    return py, (h0 - 1 - px)  # k==3


def _rot_map_base_to_display(x0: float, y0: float, h0: int, w0: int, k: int) -> tuple[float, float]:
    """
    Map a point in base coords to rotated coords for drawing crosshair.
    """
    k = int(k) % 4
    if k == 0:
        return x0, y0
    if k == 1:
        return y0, (w0 - 1 - x0)
    if k == 2:
        return (w0 - 1 - x0), (h0 - 1 - y0)
    return (h0 - 1 - y0), x0  # k==3


def _fixed_center_mm(img: sitk.Image) -> tuple[float, float, float]:
    idx = [(s - 1) / 2.0 for s in img.GetSize()]  # (x,y,z)
    return tuple(float(v) for v in img.TransformContinuousIndexToPhysicalPoint(idx))


def _make_identity_like_fixed(fixed: sitk.Image) -> sitk.Euler3DTransform:
    t = sitk.Euler3DTransform()
    t.SetCenter(_fixed_center_mm(fixed))
    t.SetTranslation((0.0, 0.0, 0.0))
    t.SetRotation(0.0, 0.0, 0.0)
    return t


# ----------------------------
# Clickable image label with drag
# ----------------------------
class ClickLabel(QLabel):
    clicked = Signal(float, float)
    dragged = Signal(float, float)
    doubleClicked = Signal()
    wheeled = Signal(int, int)  # delta, modifiers

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self._dragging = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self.clicked.emit(
                float(event.position().x()),
                float(event.position().y()),
            )
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self.dragged.emit(
                float(event.position().x()),
                float(event.position().y()),
            )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self.doubleClicked.emit()
            event.accept()
            return

        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        delta = int(event.angleDelta().y())
        mods = int(event.modifiers().value)
        self.wheeled.emit(delta, mods)
        event.accept()


# ----------------------------
# Small console dialog (no images)
# ----------------------------
class ManualConsoleDialog(QDialog):
    """
    Manual console for rigid correction of the moving image in T1 space.

    The user selects a reference anatomical plane, then:
        - translates the moving image inside the selected displayed plane;
        - rotates the moving image around the normal axis of that plane.

    Depth controls were deliberately removed from this dialog.
    """

    def __init__(self, viewer: OverlayViewer, parent=None):
        super().__init__(parent)

        self.viewer = viewer
        self._positioned_once = False

        # Width of the invisible border used for manual resizing.
        self._resize_margin = 8

        self.setWindowTitle(f"Manual correction: adjust {viewer.moving_name} on MRI 1")

        # Same frameless rounded NeuXelec window as the main overlay viewer.
        # Qt.Window keeps this console floating and non-modal.
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # Remove the native white corner resize indicator.
        self.setSizeGripEnabled(False)

        # It remains manually resizable after its initial screen-adapted size.
        self.setMinimumSize(430, 420)
        self._set_adapted_initial_size()

        self._build_ui()
        self._apply_style()
        self._restore_selected_plane()
        self._connect_controls()
        self._update_controls_enabled()
        self._update_info()

    # ============================================================
    # Window geometry
    # ============================================================

    def _set_adapted_initial_size(self) -> None:
        """
        Open the manual console at a comfortable size while remaining
        visible on the screen containing the overlay viewer.
        """
        preferred_width = 550
        preferred_height = 750

        try:
            screen = self.viewer.screen()

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
            geometry = self.frameGeometry()
            geometry.moveCenter(self.viewer.frameGeometry().center())

            screen = self.viewer.screen()

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
        Detect whether the pointer is over one of the invisible resize
        borders or corners of the frameless manual console.
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
        Display the standard resize cursor over borders and corners.
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
        Start native window resizing while keeping the custom NeuXelec shell.
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

    # ============================================================
    # UI construction
    # ============================================================

    def _build_ui(self) -> None:
        # ============================================================
        # Transparent dialog and rounded NeuXelec shell
        # ============================================================
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
        self.custom_header = NeuXelecOverlayHeader(self)
        root.addWidget(self.custom_header)

        # ---------------------------------------------------------
        # Scrollable central area
        # ---------------------------------------------------------
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("manualConsoleScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("manualConsoleScrollContent")
        self.scroll_content.setMinimumHeight(590)

        content = QVBoxLayout(self.scroll_content)
        content.setContentsMargins(12, 8, 12, 12)
        content.setSpacing(14)

        # ---------------------------------------------------------
        # Title
        # ---------------------------------------------------------
        self.lbl_title = QLabel("MANUAL CORRECTION")
        self.lbl_title.setObjectName("manualDialogTitle")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        content.addWidget(self.lbl_title)

        self.lbl_subtitle = QLabel(
            f"Adjust {self.viewer.moving_name} on the MRI 1 reference volume"
        )
        self.lbl_subtitle.setObjectName("manualDialogSubtitle")
        self.lbl_subtitle.setAlignment(Qt.AlignCenter)
        content.addWidget(self.lbl_subtitle)

        # ---------------------------------------------------------
        # Current transformation information
        # ---------------------------------------------------------
        self.info_frame = QFrame()
        self.info_frame.setObjectName("manualInfoCard")
        self.info_frame.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )

        info_layout = QVBoxLayout(self.info_frame)
        info_layout.setContentsMargins(12, 10, 12, 10)

        self.lbl = QLabel()
        self.lbl.setObjectName("manualInfoLabel")
        self.lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl.setWordWrap(True)
        info_layout.addWidget(self.lbl)

        content.addWidget(self.info_frame)

        # ---------------------------------------------------------
        # Plane selection
        # ---------------------------------------------------------
        self.grp_plane = QGroupBox("Reference plane")
        self.grp_plane.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        plane_layout = QVBoxLayout(self.grp_plane)
        plane_layout.setContentsMargins(14, 20, 14, 14)
        plane_layout.setSpacing(10)

        lbl_plane_hint = QLabel("Select the anatomical view used for the manual correction.")
        lbl_plane_hint.setObjectName("hintLabel")
        lbl_plane_hint.setAlignment(Qt.AlignCenter)
        plane_layout.addWidget(lbl_plane_hint)

        plane_row = QHBoxLayout()
        plane_row.setSpacing(8)

        self.btn_plane_cor = QPushButton("Coronal")
        self.btn_plane_axi = QPushButton("Axial")
        self.btn_plane_sag = QPushButton("Sagittal")

        self.plane_group = QButtonGroup(self)
        self.plane_group.setExclusive(True)

        for button in (
            self.btn_plane_cor,
            self.btn_plane_axi,
            self.btn_plane_sag,
        ):
            button.setObjectName("planeButton")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumHeight(42)
            self.plane_group.addButton(button)
            plane_row.addWidget(button)

        plane_layout.addLayout(plane_row)
        content.addWidget(self.grp_plane)

        # ---------------------------------------------------------
        # Translation controls
        # ---------------------------------------------------------
        self.grp_translation = QGroupBox("Translation in selected plane")
        self.grp_translation.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        translation_layout = QVBoxLayout(self.grp_translation)
        translation_layout.setContentsMargins(14, 20, 14, 14)
        translation_layout.setSpacing(12)

        lbl_translation_hint = QLabel("Move the overlay within the currently selected slice view.")
        lbl_translation_hint.setObjectName("hintLabel")
        lbl_translation_hint.setAlignment(Qt.AlignCenter)
        translation_layout.addWidget(lbl_translation_hint)

        # Inline translation step setting
        self.translation_step_frame = QFrame()
        self.translation_step_frame.setObjectName("inlineStepCard")

        translation_step_layout = QHBoxLayout(self.translation_step_frame)
        translation_step_layout.setContentsMargins(12, 7, 12, 7)
        translation_step_layout.setSpacing(8)

        lbl_tstep = QLabel("Translation step")
        lbl_tstep.setObjectName("inlineStepLabel")

        self.cb_tstep = QComboBox()
        self.cb_tstep.setObjectName("stepCombo")
        self.cb_tstep.addItems(["0.5", "1", "2", "5", "10", "20"])
        self.cb_tstep.setCurrentText("2")
        self.cb_tstep.setFixedSize(84, 36)

        lbl_tunit = QLabel("mm")
        lbl_tunit.setObjectName("unitLabel")

        translation_step_layout.addWidget(lbl_tstep)
        translation_step_layout.addStretch(1)
        translation_step_layout.addWidget(self.cb_tstep)
        translation_step_layout.addWidget(lbl_tunit)

        translation_layout.addWidget(self.translation_step_frame)

        # Directional translation controls
        dpad_layout = QGridLayout()
        dpad_layout.setHorizontalSpacing(6)
        dpad_layout.setVerticalSpacing(6)
        dpad_layout.setAlignment(Qt.AlignCenter)

        self.btn_up = self._make_control_button("↑", 58, 52)
        self.btn_dn = self._make_control_button("↓", 58, 52)
        self.btn_lt = self._make_control_button("←", 58, 52)
        self.btn_rt = self._make_control_button("→", 58, 52)

        dpad_layout.addWidget(self.btn_up, 0, 1)
        dpad_layout.addWidget(self.btn_lt, 1, 0)
        dpad_layout.addWidget(self.btn_rt, 1, 2)
        dpad_layout.addWidget(self.btn_dn, 2, 1)

        translation_layout.addLayout(dpad_layout)

        content.addWidget(self.grp_translation)

        # ---------------------------------------------------------
        # Rotation controls
        # ---------------------------------------------------------
        self.grp_rotation = QGroupBox("Rotation in selected plane")
        self.grp_rotation.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        rotation_layout = QVBoxLayout(self.grp_rotation)
        rotation_layout.setContentsMargins(14, 20, 14, 14)
        rotation_layout.setSpacing(12)

        lbl_rotation_hint = QLabel("Rotate around the normal axis of the selected plane.")
        lbl_rotation_hint.setObjectName("hintLabel")
        lbl_rotation_hint.setAlignment(Qt.AlignCenter)
        rotation_layout.addWidget(lbl_rotation_hint)

        # Inline rotation step setting
        self.rotation_step_frame = QFrame()
        self.rotation_step_frame.setObjectName("inlineStepCard")

        rotation_step_layout = QHBoxLayout(self.rotation_step_frame)
        rotation_step_layout.setContentsMargins(12, 7, 12, 7)
        rotation_step_layout.setSpacing(8)

        lbl_rstep = QLabel("Rotation step")
        lbl_rstep.setObjectName("inlineStepLabel")

        self.cb_rstep = QComboBox()
        self.cb_rstep.setObjectName("stepCombo")
        self.cb_rstep.addItems(["0.5", "1", "2", "5"])
        self.cb_rstep.setCurrentText("1")
        self.cb_rstep.setFixedSize(84, 36)

        lbl_runit = QLabel("deg")
        lbl_runit.setObjectName("unitLabel")

        rotation_step_layout.addWidget(lbl_rstep)
        rotation_step_layout.addStretch(1)
        rotation_step_layout.addWidget(self.cb_rstep)
        rotation_step_layout.addWidget(lbl_runit)

        rotation_layout.addWidget(self.rotation_step_frame)

        # Rotation buttons
        rotate_row = QHBoxLayout()
        rotate_row.setSpacing(10)
        rotate_row.setAlignment(Qt.AlignCenter)

        self.btn_rot_m = self._make_control_button("↺", 88, 48)
        self.btn_rot_p = self._make_control_button("↻", 88, 48)

        rotate_row.addWidget(self.btn_rot_m)
        rotate_row.addWidget(self.btn_rot_p)

        rotation_layout.addLayout(rotate_row)

        content.addWidget(self.grp_rotation)
        content.addStretch(1)

        # ---------------------------------------------------------
        # Insert the complete console content inside the scroll area
        # ---------------------------------------------------------
        self.scroll_area.setWidget(self.scroll_content)
        root.addWidget(self.scroll_area, 1)

        # ---------------------------------------------------------
        # Fixed bottom actions
        # ---------------------------------------------------------
        bottom = QHBoxLayout()
        bottom.setContentsMargins(10, 0, 10, 2)
        bottom.setSpacing(10)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setObjectName("secondaryButton")
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.setMinimumHeight(42)
        self.btn_reset.setMinimumWidth(108)

        self.btn_close = QPushButton("Close")
        self.btn_close.setObjectName("primaryButton")
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.setMinimumHeight(42)
        self.btn_close.setMinimumWidth(108)

        bottom.addWidget(self.btn_reset)
        bottom.addStretch(1)
        bottom.addWidget(self.btn_close)

        root.addLayout(bottom)

    def _make_control_button(
        self,
        text: str,
        width: int,
        height: int,
    ) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("controlButton")
        button.setCursor(Qt.PointingHandCursor)
        button.setFixedSize(width, height)
        return button

    def _connect_controls(self) -> None:
        self.btn_plane_cor.clicked.connect(lambda: self._set_active_plane("coronal"))
        self.btn_plane_axi.clicked.connect(lambda: self._set_active_plane("axial"))
        self.btn_plane_sag.clicked.connect(lambda: self._set_active_plane("sagittal"))

        self.btn_up.clicked.connect(lambda: self._nudge_screen(0, -1))
        self.btn_dn.clicked.connect(lambda: self._nudge_screen(0, +1))
        self.btn_lt.clicked.connect(lambda: self._nudge_screen(-1, 0))
        self.btn_rt.clicked.connect(lambda: self._nudge_screen(+1, 0))

        self.btn_rot_m.clicked.connect(lambda: self._nudge_rot_selected(-1))
        self.btn_rot_p.clicked.connect(lambda: self._nudge_rot_selected(+1))

        self.btn_reset.clicked.connect(self._reset)
        self.btn_close.clicked.connect(self.accept)

    # ============================================================
    # State
    # ============================================================

    def _restore_selected_plane(self) -> None:
        active = getattr(self.viewer, "manual_active_plane", None)

        if active == "coronal":
            self.btn_plane_cor.setChecked(True)
        elif active == "axial":
            self.btn_plane_axi.setChecked(True)
        elif active == "sagittal":
            self.btn_plane_sag.setChecked(True)

    def _active_plane(self):
        plane = getattr(self.viewer, "manual_active_plane", None)

        if plane is None:
            return None

        plane = str(plane).lower().strip()

        if plane not in ("coronal", "axial", "sagittal"):
            return None

        return plane

    def _set_active_plane(self, plane: str) -> None:
        plane = str(plane or "").lower().strip()

        if plane not in ("coronal", "axial", "sagittal"):
            return

        try:
            self.viewer.set_manual_active_plane(plane)
        except Exception:
            self.viewer.manual_active_plane = plane

        if plane == "coronal":
            self.btn_plane_cor.setChecked(True)
        elif plane == "axial":
            self.btn_plane_axi.setChecked(True)
        elif plane == "sagittal":
            self.btn_plane_sag.setChecked(True)

        self._update_controls_enabled()
        self._update_info()

    def _update_controls_enabled(self) -> None:
        enabled = self._active_plane() is not None

        for button in (
            self.btn_up,
            self.btn_dn,
            self.btn_lt,
            self.btn_rt,
            self.btn_rot_m,
            self.btn_rot_p,
        ):
            button.setEnabled(enabled)

    def _update_info(self) -> None:
        tx, ty, tz = self.viewer.manual_t.GetTranslation()
        rx = np.rad2deg(self.viewer.manual_t.GetAngleX())
        ry = np.rad2deg(self.viewer.manual_t.GetAngleY())
        rz = np.rad2deg(self.viewer.manual_t.GetAngleZ())

        plane = self._active_plane()
        plane_txt = plane.capitalize() if plane is not None else "None selected"

        self.lbl.setText(
            f"Selected plane: {plane_txt}\n"
            f"Translation (mm):  Tx={tx:.1f}   Ty={ty:.1f}   Tz={tz:.1f}\n"
            f"Rotation (deg):    Rx={rx:.1f}   Ry={ry:.1f}   Rz={rz:.1f}"
        )

    def _refresh_transform_display(self) -> None:
        """
        Refresh the rendered overlay and the displayed transform values in
        both the Manual console and the main Check coregistration window.
        """
        try:
            self.viewer._update_views()
        except Exception:
            pass

        # Values displayed in the Manual console.
        try:
            self._update_info()
        except Exception:
            pass

        # Values displayed in the main OverlayViewer parameters panel.
        try:
            self.viewer._update_header()
            self.viewer.lbl_header.repaint()
        except Exception:
            pass

    def _tstep(self) -> float:
        try:
            return float(self.cb_tstep.currentText())
        except Exception:
            return 2.0

    def _rstep_deg(self) -> float:
        try:
            return float(self.cb_rstep.currentText())
        except Exception:
            return 1.0

    # ============================================================
    # Translation / rotation
    # ============================================================

    def _translation_delta_for_active_plane(
        self,
        dscreen_x: int,
        dscreen_y: int,
    ) -> tuple[float, float, float]:
        """
        Convert displayed arrows into T1 physical translation axes.

        Convention:
            - Axial:    left/right = X, up/down = Y
            - Coronal:  left/right = X, up/down = Z
            - Sagittal: left/right = Y, up/down = Z
        """
        step = self._tstep()
        plane = self._active_plane()

        if plane is None:
            return 0.0, 0.0, 0.0

        dx = dy = dz = 0.0

        if plane == "axial":
            dx = float(dscreen_x) * step
            dy = float(dscreen_y) * step

        elif plane == "coronal":
            dx = float(dscreen_x) * step
            dz = float(dscreen_y) * step

        elif plane == "sagittal":
            dy = float(dscreen_x) * step
            dz = float(dscreen_y) * step

        return dx, dy, dz

    def _nudge_screen(self, dpx: int, dpy: int) -> None:
        dx, dy, dz = self._translation_delta_for_active_plane(dpx, dpy)

        tx, ty, tz = self.viewer.manual_t.GetTranslation()
        self.viewer.manual_t.SetTranslation((tx + dx, ty + dy, tz + dz))

        self._refresh_transform_display()

    def _rotation_axis_for_active_plane(self):
        plane = self._active_plane()

        if plane is None:
            return None

        if plane == "axial":
            return "z"

        if plane == "coronal":
            return "y"

        return "x"

    def _nudge_rot_selected(self, direction: int) -> None:
        axis = self._rotation_axis_for_active_plane()

        if axis is None:
            return

        step = np.deg2rad(self._rstep_deg()) * float(direction)

        rx = self.viewer.manual_t.GetAngleX()
        ry = self.viewer.manual_t.GetAngleY()
        rz = self.viewer.manual_t.GetAngleZ()

        if axis == "x":
            rx += step
        elif axis == "y":
            ry += step
        else:
            rz += step

        self.viewer.manual_t.SetRotation(rx, ry, rz)

        self._refresh_transform_display()

    def _reset(self) -> None:
        self.viewer.manual_t = _make_identity_like_fixed(self.viewer.fixed_img)

        self._refresh_transform_display()

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

            QWidget#manualConsoleScrollContent {
                background-color: transparent;
            }

            QScrollArea#manualConsoleScrollArea {
                background-color: transparent;
                border: none;
            }

            QScrollArea#manualConsoleScrollArea > QWidget > QWidget {
                background-color: transparent;
            }

            QLabel {
                color: #F2F2F5;
                font-size: 12px;
                background-color: transparent;
                border: none;
            }

            QLabel#manualDialogTitle {
                color: #F4D9D0;
                font-size: 18px;
                font-weight: 500;
                letter-spacing: 1px;
            }

            QLabel#manualDialogSubtitle {
                color: #8E8E98;
                font-size: 12px;
                padding-bottom: 2px;
            }

            QFrame#manualInfoCard {
                background-color: #111218;
                border: 1px solid #2B2D38;
                border-radius: 9px;
            }

            QLabel#manualInfoLabel {
                color: #A6A8B2;
                border: none;
                background-color: transparent;
                font-size: 11px;
                font-weight: 500;
            }

            QLabel#hintLabel {
                color: #8E8E98;
                font-size: 11px;
                font-weight: 400;
            }

            QLabel#fieldLabel {
                color: #F2F2F5;
                font-size: 12px;
                font-weight: 500;
            }

            QLabel#unitLabel {
                color: #8E8E98;
                font-size: 12px;
                font-weight: 500;
                min-width: 25px;
            }

            QLabel#inlineStepLabel {
                color: #CFCFD6;
                font-size: 12px;
                font-weight: 600;
            }

            QFrame#inlineStepCard {
                background-color: #17181F;
                border: 1px solid #2B2D38;
                border-radius: 9px;
            }

            QFrame#inlineStepCard:hover {
                border: 1px solid #343743;
            }

            QGroupBox {
                color: #CFCFD6;
                font-size: 13px;
                font-weight: 600;
                background-color: #111218;
                border: 1px solid #2B2D38;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 8px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0px 6px;
                color: #CFCFD6;
                background-color: #06070D;
            }

            QComboBox#stepCombo {
                background-color: #111218;
                color: #F2F2F5;
                border: 1px solid #353844;
                border-radius: 7px;
                padding: 5px 28px 5px 10px;
                font-size: 12px;
                font-weight: 600;
            }

            QComboBox#stepCombo:hover {
                border: 1px solid #FF487D;
            }

            QComboBox#stepCombo:focus {
                border: 1px solid #FF487D;
            }

            QComboBox#stepCombo::drop-down {
                width: 24px;
                border: none;
            }

            QComboBox QAbstractItemView {
                background-color: #17181F;
                color: #F2F2F5;
                border: 1px solid #2B2D38;
                selection-background-color: #35202F;
                selection-color: white;
            }

            QPushButton#planeButton {
                color: #F2F2F5;
                background-color: #17181F;
                border: 1px solid #2B2D38;
                border-radius: 9px;
                font-size: 13px;
                font-weight: 600;
                padding-left: 12px;
                padding-right: 12px;
            }

            QPushButton#planeButton:hover {
                color: white;
                background-color: #20222B;
                border: 1px solid #FF487D;
            }

            QPushButton#planeButton:checked {
                color: white;
                background-color: #20222B;
                border: 1px solid #FF487D;
            }

            QPushButton#planeButton:checked:hover {
                color: white;
                background-color: #20222B;
                border: 1px solid #FF487D;
            }

            QPushButton#controlButton {
                color: #F2F2F5;
                background-color: #17181F;
                border: 1px solid #2B2D38;
                border-radius: 10px;
                font-size: 21px;
                font-weight: 600;
            }

            QPushButton#controlButton:hover {
                background-color: #20222B;
                border: 1px solid #FF487D;
            }

            QPushButton#controlButton:pressed {
                color: white;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
                border: none;
            }

            QPushButton#controlButton:disabled {
                color: #62646E;
                background-color: #121319;
                border: 1px solid #20222A;
            }

            QPushButton#secondaryButton {
                color: white;
                background-color: #17181F;
                border: 1px solid #2B2D38;
                border-radius: 10px;
                font-size: 13px;
                font-weight: 600;
                padding-left: 16px;
                padding-right: 16px;
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
                border-radius: 10px;
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

            QScrollArea#manualConsoleScrollArea QScrollBar:vertical {
                background-color: transparent;
                width: 13px;
                margin: 5px 2px 5px 2px;
                border: none;
                border-radius: 7px;
            }

            QScrollArea#manualConsoleScrollArea QScrollBar::handle:vertical {
                background-color: #3F424C;
                border: 1px solid #FF487D;
                border-radius: 6px;
                min-height: 22px;
            }

            QScrollArea#manualConsoleScrollArea QScrollBar::handle:vertical:hover {
                background-color: #4B4E59;
                border: 1px solid #FF6F9D;
                border-radius: 6px;
            }

            QScrollArea#manualConsoleScrollArea QScrollBar::handle:vertical:pressed {
                background-color: #343743;
                border: 1px solid #FF487D;
                border-radius: 6px;
            }

            QScrollArea#manualConsoleScrollArea QScrollBar::add-line:vertical,
            QScrollArea#manualConsoleScrollArea QScrollBar::sub-line:vertical {
                height: 0px;
                background: transparent;
                border: none;
            }

            QScrollArea#manualConsoleScrollArea QScrollBar::add-page:vertical,
            QScrollArea#manualConsoleScrollArea QScrollBar::sub-page:vertical {
                background: transparent;
            }
            """)


class NeuXelecOverlayHeader(QFrame):
    """
    Frameless NeuXelec header for the main coregistration viewer.
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


# ----------------------------
# OverlayViewer
# ----------------------------
class OverlayViewer(QDialog):
    """
    2x2 layout:
      [ Coronal ]   [ Sagittal ]
      [ Axial   ]   [ Parameters ]

    - Fixed (T1) in GREEN
    - Moving (CT/PET/SPECT/T2...) in RED
    - Crosshair synchronized
    - Drag to move crosshair smoothly
    - ScrollArea wrapper so it behaves on small screens

    NEW:
    - A "Manual console..." button opens a small console dialog (no images)
      which applies an additional rigid transform to the moving image in T1 space,
      updating the three views in real-time.
    """

    # Rotation settings for display.
    # IMPORTANT: Axial rotated 180° so the nose is UP (you asked for it).
    K_AXIAL = 2
    K_CORONAL = 2
    K_SAGITTAL = 2

    # Force neurological convention: Left screen = Left patient.
    # With our chosen rotations, we enforce it by a Left/Right flip AFTER rotation for each view.
    FLIP_LR_AXIAL = True
    FLIP_LR_CORONAL = True
    FLIP_LR_SAGITTAL = True

    def _get_zoom(self, plane: str) -> float:
        if plane == "axial":
            return float(self._zoom_axi)
        if plane == "coronal":
            return float(self._zoom_cor)
        return float(self._zoom_sag)

    def _reset_all_views(self) -> None:
        """
        Reset zoom and pan for all three anatomical views simultaneously.

        This does not modify:
            - the manual coregistration transform;
            - the current crosshair voxel;
            - the overlay alpha or gain settings.
        """
        self._zoom_cor = 1.0
        self._zoom_axi = 1.0
        self._zoom_sag = 1.0

        self._view_center = {
            "axial": None,
            "coronal": None,
            "sagittal": None,
        }

        self._view_state = {
            "axial": None,
            "coronal": None,
            "sagittal": None,
        }

        self._pan_drag_active = False
        self._pan_drag_plane = None
        self._pan_last_xy = None

        self._update_views()

    def _crop_for_zoom(self, img2d: np.ndarray, plane: str, cx: float, cy: float):
        if img2d.ndim < 2:
            raise ValueError(
                f"_crop_for_zoom expects at least 2 dimensions, got shape={img2d.shape}"
            )

        full_h, full_w = img2d.shape[:2]
        zoom = self._get_zoom(plane)

        if zoom <= 1.0001:
            st = {
                "full_w": float(full_w),
                "full_h": float(full_h),
                "x0": 0.0,
                "y0": 0.0,
                "cw": float(full_w),
                "ch": float(full_h),
            }
            return img2d, st

        cw = max(1, int(round(full_w / zoom)))
        ch = max(1, int(round(full_h / zoom)))

        cx = float(np.clip(cx, 0, max(0, full_w - 1)))
        cy = float(np.clip(cy, 0, max(0, full_h - 1)))

        x0 = int(round(cx - cw / 2))
        y0 = int(round(cy - ch / 2))
        x0 = int(np.clip(x0, 0, max(0, full_w - cw)))
        y0 = int(np.clip(y0, 0, max(0, full_h - ch)))

        if img2d.ndim == 2:
            crop = img2d[y0 : y0 + ch, x0 : x0 + cw]
        else:
            crop = img2d[y0 : y0 + ch, x0 : x0 + cw, ...]

        st = {
            "full_w": float(full_w),
            "full_h": float(full_h),
            "x0": float(x0),
            "y0": float(y0),
            "cw": float(cw),
            "ch": float(ch),
        }
        return crop, st

    def _label_uv(self, lbl: QLabel, pm: QPixmap, px: int, py: int):
        lw = max(1, lbl.width())
        lh = max(1, lbl.height())
        pw = pm.width()
        ph = pm.height()

        offx = max(0, (lw - pw) // 2)
        offy = max(0, (lh - ph) // 2)

        x_in = px - offx
        y_in = py - offy
        if x_in < 0 or y_in < 0 or x_in >= pw or y_in >= ph:
            return None

        u = x_in / max(1, pw - 1)
        v = y_in / max(1, ph - 1)
        return float(u), float(v)

    def _on_wheel(self, plane: str, delta: int, modifiers: int):
        ctrl = modifiers & Qt.ControlModifier.value
        step_dir = +1 if delta > 0 else -1

        # Ctrl + wheel = change slice
        if ctrl:
            if plane == "axial":
                self.iz = int(np.clip(self.iz + step_dir, 0, self.fzmax))
            elif plane == "coronal":
                self.iy = int(np.clip(self.iy + step_dir, 0, self.fymax))
            elif plane == "sagittal":
                self.ix = int(np.clip(self.ix + step_dir, 0, self.fxmax))
            self._update_views()
            return

        # wheel only = zoom
        factor = self._zoom_step if delta > 0 else (1.0 / self._zoom_step)

        if plane == "axial":
            self._zoom_axi = float(np.clip(self._zoom_axi * factor, self._zoom_min, self._zoom_max))
            if self._zoom_axi <= 1.0001:
                self._zoom_axi = 1.0
                self._view_center["axial"] = None

        elif plane == "coronal":
            self._zoom_cor = float(np.clip(self._zoom_cor * factor, self._zoom_min, self._zoom_max))
            if self._zoom_cor <= 1.0001:
                self._zoom_cor = 1.0
                self._view_center["coronal"] = None

        elif plane == "sagittal":
            self._zoom_sag = float(np.clip(self._zoom_sag * factor, self._zoom_min, self._zoom_max))
            if self._zoom_sag <= 1.0001:
                self._zoom_sag = 1.0
                self._view_center["sagittal"] = None

        self._update_views()

    def _on_click(self, plane: str, lbl: QLabel, px: float, py: float):
        pm = lbl.pixmap()
        if pm is None or pm.isNull():
            return

        shift = bool(QApplication.keyboardModifiers() & Qt.ShiftModifier)
        zoom = self._get_zoom(plane)

        if shift and zoom > 1.0001:
            self._pan_drag_active = True
            self._pan_drag_plane = plane
            self._pan_last_xy = (px, py)
            return

        self._pan_drag_active = False
        self._pan_drag_plane = None
        self._pan_last_xy = None

        uv = self._label_uv(lbl, pm, int(px), int(py))
        if uv is None:
            return
        u, v = uv

        st = self._view_state.get(plane, None)
        if not st:
            return

        x0 = float(st["x0"])
        y0 = float(st["y0"])
        cw = float(st["cw"])
        ch = float(st["ch"])
        full_w = float(st["full_w"])
        full_h = float(st["full_h"])

        x_full = x0 + u * max(1.0, (cw - 1.0))
        y_full = y0 + v * max(1.0, (ch - 1.0))

        # convert back from displayed rotated slice to underlying base slice coords
        if plane == "axial":
            h0, w0 = self.fixed_np[self.iz, :, :].shape
            x0b, y0b = _rot_map_display_to_base(x_full, y_full, h0, w0, self.K_AXIAL)
            if self.FLIP_LR_AXIAL:
                x0b = w0 - 1 - x0b
            self.ix = int(np.clip(round(x0b), 0, self.fxmax))
            self.iy = int(np.clip(round(y0b), 0, self.fymax))

        elif plane == "coronal":
            h0, w0 = self.fixed_np[:, self.iy, :].shape
            x0b, y0b = _rot_map_display_to_base(x_full, y_full, h0, w0, self.K_CORONAL)
            if self.FLIP_LR_CORONAL:
                x0b = w0 - 1 - x0b
            self.ix = int(np.clip(round(x0b), 0, self.fxmax))
            self.iz = int(np.clip(round(y0b), 0, self.fzmax))

        elif plane == "sagittal":
            h0, w0 = self.fixed_np[:, :, self.ix].shape
            x0b, y0b = _rot_map_display_to_base(x_full, y_full, h0, w0, self.K_SAGITTAL)
            if self.FLIP_LR_SAGITTAL:
                x0b = w0 - 1 - x0b
            self.iy = int(np.clip(round(x0b), 0, self.fymax))
            self.iz = int(np.clip(round(y0b), 0, self.fzmax))

        self._update_views()

    def _on_drag(self, plane: str, lbl: QLabel, px: float, py: float):
        pm = lbl.pixmap()
        if pm is None or pm.isNull():
            return

        shift = bool(QApplication.keyboardModifiers() & Qt.ShiftModifier)
        zoom = self._get_zoom(plane)

        if not shift or zoom <= 1.0001:
            self._pan_drag_active = False
            self._pan_drag_plane = None
            self._pan_last_xy = None
            self._on_click(plane, lbl, px, py)
            return

        if not self._pan_drag_active or self._pan_drag_plane != plane or self._pan_last_xy is None:
            self._pan_drag_active = True
            self._pan_drag_plane = plane
            self._pan_last_xy = (px, py)
            return

        lastx, lasty = self._pan_last_xy
        dx = float(px - lastx)
        dy = float(py - lasty)
        self._pan_last_xy = (px, py)

        st = self._view_state.get(plane, None)
        if not st:
            return

        cw = float(st["cw"])
        ch = float(st["ch"])
        pw = max(1.0, float(pm.width()))
        ph = max(1.0, float(pm.height()))

        dx_full = dx * (cw / pw)
        dy_full = dy * (ch / ph)

        vc = self._view_center.get(plane, None)
        if vc is None:
            return

        vc[0] -= dx_full
        vc[1] -= dy_full

        full_w = float(st["full_w"])
        full_h = float(st["full_h"])
        vc[0] = float(np.clip(vc[0], 0, max(0.0, full_w - 1.0)))
        vc[1] = float(np.clip(vc[1], 0, max(0.0, full_h - 1.0)))

        self._update_views()

    def _set_adapted_initial_size(self) -> None:
        """
        Open the overlay viewer at a comfortable size while remaining fully
        visible on the current screen. The user can still resize it manually
        after opening.
        """
        preferred_width = 1220
        preferred_height = 790

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
        Detect whether the mouse is located on one of the invisible resize
        borders of the frameless NeuXelec window.
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
        Display the native resize cursor when hovering over a frameless
        window border or corner.
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
        Start native resizing while retaining the custom frameless style.
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

    def __init__(
        self,
        fixed_t1: sitk.Image,
        moving_in_t1: sitk.Image,
        moving_name: str = "CT",
        parent=None,
    ):
        super().__init__(parent)

        self._ui_ready = False
        self._positioned_once = False
        self._resize_margin = 8

        # Delay expensive image redraws while the window is actively resized.
        # This avoids recomputing the SimpleITK overlay at every pixel moved.
        self._resize_redraw_timer = QTimer(self)
        self._resize_redraw_timer.setSingleShot(True)
        self._resize_redraw_timer.setInterval(35)
        self._resize_redraw_timer.timeout.connect(self._redraw_after_resize)

        self.fixed_img = fixed_t1
        self.moving_img = moving_in_t1
        self.moving_name = moving_name

        self.setWindowTitle(f"Check coregistration: MRI 1 + {self.moving_name}")

        # Frameless rounded NeuXelec window.
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setSizeGripEnabled(False)

        # The central region becomes scrollable if the screen is small.
        self.setMinimumSize(640, 470)
        self._set_adapted_initial_size()

        # Numpy
        self.fixed_np = _sitk_to_np_zyx(self.fixed_img)  # (z,y,x)

        # Manual refinement transform (extra transform in T1 physical space)
        self.manual_t = _make_identity_like_fixed(self.fixed_img)
        # Active view used by ManualConsoleDialog.
        # None means no view selected yet.
        self.manual_active_plane = None
        self._manual_console_dialog = None

        # Crosshair reference = fixed (T1) space
        self._recompute_bounds_fixed()
        self.iz = max(0, self.fzmax // 2)
        self.iy = max(0, self.fymax // 2)
        self.ix = max(0, self.fxmax // 2)

        # Display params (must exist before making params panel)
        self.alpha = 0.55
        self.g_fixed = 1.0
        self.g_moving = 1.2

        # Crosshair style
        self.cross_px = 1
        self.cross_color = Qt.white

        # Zoom state per panel
        self._zoom_cor = 1.0
        self._zoom_sag = 1.0
        self._zoom_axi = 1.0
        self._zoom_step = 1.10
        self._zoom_min = 1.0
        self._zoom_max = 8.0

        # Pan state
        self._pan_drag_active = False
        self._pan_drag_plane = None
        self._pan_last_xy = None

        # View center / crop state
        self._view_center = {"axial": None, "coronal": None, "sagittal": None}
        self._view_state = {"axial": None, "coronal": None, "sagittal": None}

        # ============================================================
        # Transparent dialog and rounded NeuXelec shell
        # ============================================================
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
        self.custom_header = NeuXelecOverlayHeader(self)
        root.addWidget(self.custom_header)

        # ============================================================
        # Scrollable central content
        # ============================================================
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("overlayScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Display the gradient scrollbar only when the available height is
        # insufficient for the complete viewer content.
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("overlayScrollContent")

        # This keeps panels readable. If the window becomes lower than this,
        # a vertical gradient scrollbar appears automatically.
        self.scroll_content.setMinimumHeight(690)

        content_layout = QVBoxLayout(self.scroll_content)
        content_layout.setContentsMargins(8, 6, 8, 8)
        content_layout.setSpacing(10)

        self.lbl_title = QLabel("CHECK COREGISTRATION")
        self.lbl_title.setObjectName("dialogTitle")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.lbl_title)

        self.lbl_subtitle = QLabel(f"MRI 1 + {self.moving_name}")
        self.lbl_subtitle.setObjectName("dialogSubtitle")
        self.lbl_subtitle.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.lbl_subtitle)

        # Header information displayed in the parameters panel.
        self.lbl_header = QLabel()
        self.lbl_header.setObjectName("overlayInfoLabel")
        self.lbl_header.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_header.setWordWrap(True)

        # ============================================================
        # Anatomical views
        # ============================================================
        # The three anatomical panels are placed in the same horizontal row.
        # This guarantees identical frame dimensions for Coronal, Sagittal
        # and Axial, independently of the size of the parameters panel.
        views_layout = QHBoxLayout()
        views_layout.setSpacing(12)

        # Coronal
        self.cor_panel = self._make_image_panel("Coronal")
        self.lbl_cor = self.cor_panel["img"]

        # Sagittal
        self.sag_panel = self._make_image_panel("Sagittal")
        self.lbl_sag = self.sag_panel["img"]

        # Axial
        self.axi_panel = self._make_image_panel("Axial")
        self.lbl_axi = self.axi_panel["img"]

        # Display order: Coronal | Axial | Sagittal
        for panel in (
            self.cor_panel,
            self.axi_panel,
            self.sag_panel,
        ):
            widget = panel["widget"]

            # Allow panels to become smaller when the window is reduced.
            # The scroll area preserves access if vertical space becomes tight.
            widget.setMinimumSize(0, 150)
            widget.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Ignored,
            )

            views_layout.addWidget(widget, 1)

        content_layout.addLayout(views_layout, 1)

        # ============================================================
        # Display parameters panel
        # ============================================================
        # The parameters panel is separated from the anatomical panels so its
        # text and controls cannot change the size of only one slice frame.
        self.params_panel = self._make_params_panel()
        self.params_panel.setMinimumSize(0, 0)
        self.params_panel.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Maximum,
        )

        content_layout.addWidget(self.params_panel, 0)

        self.scroll_area.setWidget(self.scroll_content)
        root.addWidget(self.scroll_area, 1)

        # ============================================================
        # Fixed bottom buttons
        # ============================================================
        # These buttons remain accessible even if the central content scrolls.
        btns = QHBoxLayout()
        btns.setContentsMargins(8, 0, 8, 2)
        btns.setSpacing(10)

        btns.addStretch(1)

        self.btn_close = QPushButton("Close")
        self.btn_close.setObjectName("secondaryButton")
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.setMinimumHeight(42)
        self.btn_close.setMinimumWidth(104)

        self.btn_validate = QPushButton("Validate")
        self.btn_validate.setObjectName("primaryButton")
        self.btn_validate.setCursor(Qt.PointingHandCursor)
        self.btn_validate.setMinimumHeight(42)
        self.btn_validate.setMinimumWidth(118)

        btns.addWidget(self.btn_close)
        btns.addWidget(self.btn_validate)

        root.addLayout(btns)

        self._apply_style()
        self._update_manual_plane_styles()
        self._update_slice_titles()

        self.btn_close.clicked.connect(self.reject)
        self.btn_validate.clicked.connect(self.accept)

        # Mouse -> crosshair
        self.lbl_cor.clicked.connect(lambda x, y: self._on_click("coronal", self.lbl_cor, x, y))
        self.lbl_cor.dragged.connect(lambda x, y: self._on_drag("coronal", self.lbl_cor, x, y))

        self.lbl_sag.clicked.connect(lambda x, y: self._on_click("sagittal", self.lbl_sag, x, y))
        self.lbl_sag.dragged.connect(lambda x, y: self._on_drag("sagittal", self.lbl_sag, x, y))

        self.lbl_axi.clicked.connect(lambda x, y: self._on_click("axial", self.lbl_axi, x, y))
        self.lbl_axi.dragged.connect(lambda x, y: self._on_drag("axial", self.lbl_axi, x, y))

        self.lbl_axi.wheeled.connect(lambda d, m: self._on_wheel("axial", d, m))
        self.lbl_cor.wheeled.connect(lambda d, m: self._on_wheel("coronal", d, m))
        self.lbl_sag.wheeled.connect(lambda d, m: self._on_wheel("sagittal", d, m))

        # Double-clicking on any anatomical view resets zoom and pan
        # simultaneously in all three visualizations.
        self.lbl_axi.doubleClicked.connect(self._reset_all_views)
        self.lbl_cor.doubleClicked.connect(self._reset_all_views)
        self.lbl_sag.doubleClicked.connect(self._reset_all_views)

        self._ui_ready = True
        self._update_header()
        self._update_views()

    # ----------------------------
    # External getter (so FilesPage can store the refined image on Validate)
    # ----------------------------
    def corrected_moving_image(self) -> sitk.Image:
        """
        Returns the moving image after applying the current manual refinement transform (in T1 space).
        Call this AFTER dlg.exec() returns Accepted to store the refined result in AppState.
        """
        return self._resample_manual_on_moving()

    def manual_transform(self) -> sitk.Euler3DTransform:
        return self.manual_t

    # ----------------------------
    # Panel builders
    # ----------------------------
    def _make_image_panel(self, title: str) -> dict:
        w = QWidget()
        w.setObjectName("imagePanel")
        w.setProperty("manualActive", False)
        w.setAttribute(Qt.WA_StyledBackground, True)

        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(7)

        lbl_title = QLabel(title)
        lbl_title.setObjectName("sliceTitle")
        lbl_title.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl_title, 0)

        container = QWidget()
        container.setObjectName("imageContainer")
        container.setMinimumSize(0, 0)
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        inner = QVBoxLayout(container)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(0)

        img = ClickLabel()
        img.setObjectName("overlayImage")
        img.setAlignment(Qt.AlignCenter)
        img.setMinimumSize(0, 0)
        img.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        inner.addWidget(img, 1)

        lbl_left = QLabel("L", container)
        lbl_left.setObjectName("orientationLabel")
        lbl_left.adjustSize()
        lbl_left.raise_()

        lbl_right = QLabel("R", container)
        lbl_right.setObjectName("orientationLabel")
        lbl_right.adjustSize()
        lbl_right.raise_()

        lay.addWidget(container, 1)

        return {
            "widget": w,
            "img": img,
            "title": lbl_title,
            "container": container,
            "lbl_left": lbl_left,
            "lbl_right": lbl_right,
        }

    def _panel_for_plane(self, plane: str):
        plane = str(plane or "").lower().strip()

        if plane == "coronal":
            return getattr(self, "cor_panel", None)

        if plane == "axial":
            return getattr(self, "axi_panel", None)

        if plane == "sagittal":
            return getattr(self, "sag_panel", None)

        return None

    def set_manual_active_plane(self, plane: str) -> None:
        plane = str(plane or "").lower().strip()

        if plane not in ("coronal", "axial", "sagittal"):
            self.manual_active_plane = None
            self._update_manual_plane_styles()
            return

        self.manual_active_plane = plane
        self._update_manual_plane_styles()

    def _update_manual_plane_styles(self) -> None:
        """
        Highlight the slice currently used for manual correction with the
        NeuXelec pink accent border.
        """
        active = getattr(self, "manual_active_plane", None)
        active = str(active).lower().strip() if active is not None else None

        for plane in ("coronal", "axial", "sagittal"):
            panel = self._panel_for_plane(plane)

            if not isinstance(panel, dict):
                continue

            widget = panel.get("widget", None)

            if widget is None:
                continue

            widget.setProperty("manualActive", bool(plane == active))

            try:
                widget.style().unpolish(widget)
                widget.style().polish(widget)
                widget.update()
            except Exception:
                pass

    def _apply_style(self) -> None:
        """
        Apply the NeuXelec dark theme to the coregistration overlay viewer.
        """
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

            QWidget#overlayScrollContent {
                background-color: transparent;
            }

            QScrollArea#overlayScrollArea {
                background-color: transparent;
                border: none;
            }

            QScrollArea#overlayScrollArea > QWidget > QWidget {
                background-color: transparent;
            }

            QLabel {
                color: #F2F2F5;
                font-size: 12px;
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
                padding-bottom: 5px;
            }

            QWidget#imagePanel {
                background-color: #111218;
                border: 1px solid #2B2D38;
                border-radius: 10px;
            }

            QWidget#imagePanel[manualActive="true"] {
                background-color: #131219;
                border: 2px solid #FF487D;
                border-radius: 10px;
            }

            QWidget#imageContainer {
                background-color: transparent;
                border: none;
            }

            QLabel#sliceTitle {
                color: #CFCFD6;
                font-size: 13px;
                font-weight: 600;
                border: none;
                background-color: transparent;
            }

            QLabel#overlayImage {
                background-color: #17181F;
                border: none;
                border-radius: 7px;
            }

            QLabel#orientationLabel {
                color: #FF487D;
                background-color: transparent;
                border: none;
                font-size: 18px;
                font-weight: 700;
            }

            QGroupBox#parametersPanel {
                color: #F2F2F5;
                font-size: 13px;
                font-weight: 600;
                background-color: #111218;
                border: 1px solid #2B2D38;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 8px;
            }

            QGroupBox#parametersPanel::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0px 6px;
                color: #CFCFD6;
                background-color: #06070D;
            }

            QLabel#overlayInfoLabel {
                color: #A6A8B2;
                background-color: #17181F;
                border: 1px solid #2B2D38;
                border-radius: 8px;
                padding: 8px;
                font-size: 11px;
                font-weight: 500;
            }

            QLabel#parameterLabel {
                color: #F2F2F5;
                font-size: 12px;
                font-weight: 500;
                min-width: 66px;
            }

            QSlider:horizontal {
                min-height: 24px;
                max-height: 24px;
                background-color: #111218;
                border: none;
                padding: 0px;
                margin: 0px;
            }

            QSlider::groove:horizontal {
                height: 2px;
                background-color: #111218;
                border: none;
                border-radius: 0px;
                margin: 0px;
            }

            QSlider::sub-page:horizontal {
                height: 2px;
                background-color: #F2F2F5;
                border: none;
                border-radius: 0px;
                margin: 0px;
            }

            QSlider::add-page:horizontal {
                height: 2px;
                background-color: #111218;
                border: none;
                border-radius: 0px;
                margin: 0px;
            }

            QSlider::handle:horizontal {
                background-color: #F2F2F5;
                border: none;
                width: 14px;
                height: 14px;
                margin: -6px 0px;
                border-radius: 7px;
            }

            QSlider::handle:horizontal:hover {
                background-color: white;
                border: none;
            }

            QSlider::handle:horizontal:pressed {
                background-color: white;
                border: 2px solid #FF487D;
                width: 14px;
                height: 14px;
                margin: -7px 0px;
                border-radius: 7px;
            }

            QSlider:disabled {
                background-color: #111218;
                border: none;
            }

            QSlider:disabled::groove:horizontal {
                height: 2px;
                background-color: #111218;
                border: none;
                border-radius: 0px;
                margin: 0px;
            }

            QSlider:disabled::sub-page:horizontal {
                height: 2px;
                background-color: rgba(242, 242, 245, 110);
                border: none;
                border-radius: 0px;
                margin: 0px;
            }

            QSlider:disabled::add-page:horizontal {
                height: 2px;
                background-color: #111218;
                border: none;
                border-radius: 0px;
                margin: 0px;
            }

            QSlider:disabled::handle:horizontal {
                background-color: rgba(242, 242, 245, 120);
                border: none;
                width: 14px;
                height: 14px;
                margin: -6px 0px;
                border-radius: 7px;
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

            QScrollBar:vertical {
                background-color: transparent;
                width: 13px;
                margin: 5px 2px 5px 2px;
                border: none;
                border-radius: 7px;
            }

            QScrollBar::handle:vertical {
                background-color: #3F424C;
                border: 1px solid #FF487D;
                border-radius: 6px;
                min-height: 22px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #4B4E59;
                border: 1px solid #FF6F9D;
                border-radius: 6px;
            }

            QScrollBar::handle:vertical:pressed {
                background-color: #343743;
                border: 1px solid #FF487D;
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

    def _update_slice_titles(self) -> None:
        """
        Display slice number like Analyze.
        """
        try:
            self.cor_panel["title"].setText(
                f"Coronal - Slice {int(self.iy) + 1} / {int(self.fymax) + 1}"
            )
        except Exception:
            pass

        try:
            self.axi_panel["title"].setText(
                f"Axial - Slice {int(self.iz) + 1} / {int(self.fzmax) + 1}"
            )
        except Exception:
            pass

        try:
            self.sag_panel["title"].setText(
                f"Sagittal - Slice {int(self.ix) + 1} / {int(self.fxmax) + 1}"
            )
        except Exception:
            pass

    def _make_params_panel(self) -> QWidget:
        box = QGroupBox("Display parameters")
        box.setObjectName("parametersPanel")

        main_layout = QHBoxLayout(box)
        main_layout.setContentsMargins(16, 22, 16, 16)
        main_layout.setSpacing(18)

        # ---------------------------------------------------------
        # Left: technical information and transform values
        # ---------------------------------------------------------
        self.lbl_header.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.lbl_header.setWordWrap(True)
        self.lbl_header.setMinimumWidth(285)
        self.lbl_header.setMinimumHeight(118)
        self.lbl_header.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )

        main_layout.addWidget(self.lbl_header, 1)

        # ---------------------------------------------------------
        # Right: overlay display controls
        # ---------------------------------------------------------
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(12)

        # Alpha
        row_a = QHBoxLayout()
        row_a.setSpacing(10)

        lbl_alpha = QLabel(f"{self.moving_name} alpha")
        lbl_alpha.setObjectName("parameterLabel")

        self.sl_alpha = QSlider(Qt.Horizontal)
        self.sl_alpha.setObjectName("overlayParameterSlider")
        self.sl_alpha.setRange(0, 100)
        self.sl_alpha.setValue(int(self.alpha * 100))
        self.sl_alpha.setMinimumWidth(110)

        row_a.addWidget(lbl_alpha)
        row_a.addWidget(self.sl_alpha, 1)
        controls_layout.addLayout(row_a)

        # Fixed gain
        row_fg = QHBoxLayout()
        row_fg.setSpacing(10)

        lbl_fixed = QLabel("MRI 1 gain")
        lbl_fixed.setObjectName("parameterLabel")

        self.sl_gfixed = QSlider(Qt.Horizontal)
        self.sl_gfixed.setObjectName("overlayParameterSlider")
        self.sl_gfixed.setRange(10, 300)
        self.sl_gfixed.setValue(int(self.g_fixed * 100))
        self.sl_gfixed.setMinimumWidth(110)

        row_fg.addWidget(lbl_fixed)
        row_fg.addWidget(self.sl_gfixed, 1)
        controls_layout.addLayout(row_fg)

        # Moving gain
        row_mg = QHBoxLayout()
        row_mg.setSpacing(10)

        lbl_moving = QLabel(f"{self.moving_name} gain")
        lbl_moving.setObjectName("parameterLabel")

        self.sl_gmoving = QSlider(Qt.Horizontal)
        self.sl_gmoving.setObjectName("overlayParameterSlider")
        self.sl_gmoving.setRange(10, 300)
        self.sl_gmoving.setValue(int(self.g_moving * 100))
        self.sl_gmoving.setMinimumWidth(110)

        row_mg.addWidget(lbl_moving)
        row_mg.addWidget(self.sl_gmoving, 1)
        controls_layout.addLayout(row_mg)

        controls_layout.addSpacing(4)

        self.btn_manual_console = QPushButton(f"Manual console…  Adjust {self.moving_name}")
        self.btn_manual_console.setObjectName("secondaryButton")
        self.btn_manual_console.setCursor(Qt.PointingHandCursor)
        self.btn_manual_console.setMinimumHeight(42)

        controls_layout.addWidget(self.btn_manual_console)

        main_layout.addLayout(controls_layout, 1)

        self.sl_alpha.valueChanged.connect(self._on_params_changed)
        self.sl_gfixed.valueChanged.connect(self._on_params_changed)
        self.sl_gmoving.valueChanged.connect(self._on_params_changed)
        self.btn_manual_console.clicked.connect(self._open_manual_console)

        return box

    # ----------------------------
    # Manual console
    # ----------------------------
    def _open_manual_console(self):
        # Non-modal console: the user can still interact with the Check coregistration
        # window and the rest of the application.
        try:
            if self._manual_console_dialog is not None:
                self._manual_console_dialog.show()
                self._manual_console_dialog.raise_()
                self._manual_console_dialog.activateWindow()
                return
        except Exception:
            self._manual_console_dialog = None

        dlg = ManualConsoleDialog(self, parent=self)
        dlg.setModal(False)
        dlg.finished.connect(lambda _: setattr(self, "_manual_console_dialog", None))

        self._manual_console_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    # ----------------------------
    # Bounds
    # ----------------------------
    def _recompute_bounds_fixed(self):
        self.fzmax = max(0, int(self.fixed_np.shape[0] - 1))
        self.fymax = max(0, int(self.fixed_np.shape[1] - 1))
        self.fxmax = max(0, int(self.fixed_np.shape[2] - 1))

    # ----------------------------
    # Base slices (NOT rotated)
    # ----------------------------
    def _base_axial(self, vol_zyx: np.ndarray, iz: int) -> np.ndarray:
        iz = int(np.clip(iz, 0, vol_zyx.shape[0] - 1))
        return vol_zyx[iz, :, :]

    def _base_coronal(self, vol_zyx: np.ndarray, iy: int) -> np.ndarray:
        iy = int(np.clip(iy, 0, vol_zyx.shape[1] - 1))
        return vol_zyx[:, iy, :]

    def _base_sagittal(self, vol_zyx: np.ndarray, ix: int) -> np.ndarray:
        ix = int(np.clip(ix, 0, vol_zyx.shape[2] - 1))
        return vol_zyx[:, :, ix]

    # ----------------------------
    # Manual resample (extra transform in T1 space)
    # ----------------------------
    def _resample_manual_on_moving(self) -> sitk.Image:
        """
        Applies self.manual_t to self.moving_img (which is already in T1 space).
        Output remains in T1 geometry.
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.fixed_img)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(self.manual_t)
        resampler.SetDefaultPixelValue(0)
        return resampler.Execute(self.moving_img)

    # ----------------------------
    # Rendering
    # ----------------------------
    def _compose_rgb(self, fixed2d: np.ndarray, moving2d: np.ndarray) -> np.ndarray:
        f = np.clip(_norm01(fixed2d) * self.g_fixed, 0.0, 1.0)
        m = np.clip(_norm01(moving2d) * self.g_moving, 0.0, 1.0)

        f_u8 = (f * 255.0).astype(np.uint8)
        m_u8 = (m * 255.0).astype(np.uint8)

        rgb = np.zeros((f_u8.shape[0], f_u8.shape[1], 3), dtype=np.float32)
        rgb[..., 1] = f_u8  # fixed in GREEN
        rgb[..., 0] = (1.0 - self.alpha) * rgb[..., 0] + self.alpha * m_u8  # moving in RED
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _draw_crosshair_on_pixmap(self, pm: QPixmap, x: float, y: float) -> QPixmap:
        out = QPixmap(pm)
        p = QPainter(out)
        pen = QPen(self.cross_color)
        pen.setWidth(self.cross_px)
        p.setPen(pen)

        w = out.width()
        h = out.height()
        xx = int(np.clip(round(x), 0, w - 1))
        yy = int(np.clip(round(y), 0, h - 1))

        p.drawLine(0, yy, w, yy)
        p.drawLine(xx, 0, xx, h)
        p.end()
        return out

    def _update_header(self) -> None:
        tx, ty, tz = self.manual_t.GetTranslation()
        rx = np.rad2deg(self.manual_t.GetAngleX())
        ry = np.rad2deg(self.manual_t.GetAngleY())
        rz = np.rad2deg(self.manual_t.GetAngleZ())

        self.lbl_header.setText(
            f"MRI 1 size: {self.fixed_img.GetSize()}    "
            f"spacing: {tuple(round(s, 3) for s in self.fixed_img.GetSpacing())}\n"
            f"{self.moving_name} in MRI 1 size: {self.moving_img.GetSize()}    "
            f"spacing: {tuple(round(s, 3) for s in self.moving_img.GetSpacing())}\n"
            f"Crosshair:  ix={self.ix}   iy={self.iy}   iz={self.iz}\n\n"
            f"Manual correction\n"
            f"Translation (mm):  Tx={tx:.1f}   Ty={ty:.1f}   Tz={tz:.1f}\n"
            f"Rotation (deg):    Rx={rx:.1f}   Ry={ry:.1f}   Rz={rz:.1f}"
        )

    def _update_views(self):
        # moving after manual refinement
        mov_refined = self._resample_manual_on_moving()
        moving_np = _sitk_to_np_zyx(mov_refined)

        # slices in fixed space
        f_ax0 = self._base_axial(self.fixed_np, self.iz)  # (y,x)
        f_co0 = self._base_coronal(self.fixed_np, self.iy)  # (z,x)
        f_sa0 = self._base_sagittal(self.fixed_np, self.ix)  # (z,y)

        # slices in moving (same indices because we are in T1 geometry)
        m_ax0 = self._base_axial(moving_np, self.iz)
        m_co0 = self._base_coronal(moving_np, self.iy)
        m_sa0 = self._base_sagittal(moving_np, self.ix)

        rgb_ax = self._compose_rgb(f_ax0, m_ax0)
        rgb_co = self._compose_rgb(f_co0, m_co0)
        rgb_sa = self._compose_rgb(f_sa0, m_sa0)

        rgb_ax_r = _rot90_k(rgb_ax, self.K_AXIAL)
        rgb_co_r = _rot90_k(rgb_co, self.K_CORONAL)
        rgb_sa_r = _rot90_k(rgb_sa, self.K_SAGITTAL)

        # Enforce neurological: flip LR after rotation
        if self.FLIP_LR_AXIAL:
            rgb_ax_r = _flip_lr(rgb_ax_r)
        if self.FLIP_LR_CORONAL:
            rgb_co_r = _flip_lr(rgb_co_r)
        if self.FLIP_LR_SAGITTAL:
            rgb_sa_r = _flip_lr(rgb_sa_r)

        # crosshair mapping (base coords -> rotated coords on full image)
        ax_xr, ax_yr = _rot_map_base_to_display(
            self.ix, self.iy, f_ax0.shape[0], f_ax0.shape[1], self.K_AXIAL
        )
        co_xr, co_yr = _rot_map_base_to_display(
            self.ix, self.iz, f_co0.shape[0], f_co0.shape[1], self.K_CORONAL
        )
        sa_xr, sa_yr = _rot_map_base_to_display(
            self.iy, self.iz, f_sa0.shape[0], f_sa0.shape[1], self.K_SAGITTAL
        )

        # Apply same LR flip to crosshair X in rotated space
        ax_xr = _apply_flip_to_x(ax_xr, int(rgb_ax_r.shape[1]), self.FLIP_LR_AXIAL)
        co_xr = _apply_flip_to_x(co_xr, int(rgb_co_r.shape[1]), self.FLIP_LR_CORONAL)
        sa_xr = _apply_flip_to_x(sa_xr, int(rgb_sa_r.shape[1]), self.FLIP_LR_SAGITTAL)

        # Initialize view centers when zoom starts
        if self._get_zoom("axial") > 1.0001 and self._view_center["axial"] is None:
            self._view_center["axial"] = [float(ax_xr), float(ax_yr)]
        if self._get_zoom("coronal") > 1.0001 and self._view_center["coronal"] is None:
            self._view_center["coronal"] = [float(co_xr), float(co_yr)]
        if self._get_zoom("sagittal") > 1.0001 and self._view_center["sagittal"] is None:
            self._view_center["sagittal"] = [float(sa_xr), float(sa_yr)]

        ax_cx, ax_cy = (
            self._view_center["axial"]
            if self._get_zoom("axial") > 1.0001 and self._view_center["axial"] is not None
            else [float(ax_xr), float(ax_yr)]
        )
        co_cx, co_cy = (
            self._view_center["coronal"]
            if self._get_zoom("coronal") > 1.0001 and self._view_center["coronal"] is not None
            else [float(co_xr), float(co_yr)]
        )
        sa_cx, sa_cy = (
            self._view_center["sagittal"]
            if self._get_zoom("sagittal") > 1.0001 and self._view_center["sagittal"] is not None
            else [float(sa_xr), float(sa_yr)]
        )

        # Crop according to zoom
        rgb_ax_c, ax_st = self._crop_for_zoom(rgb_ax_r, "axial", ax_cx, ax_cy)
        rgb_co_c, co_st = self._crop_for_zoom(rgb_co_r, "coronal", co_cx, co_cy)
        rgb_sa_c, sa_st = self._crop_for_zoom(rgb_sa_r, "sagittal", sa_cx, sa_cy)

        self._view_state["axial"] = ax_st
        self._view_state["coronal"] = co_st
        self._view_state["sagittal"] = sa_st

        def _cross_in_crop(pm_w: int, pm_h: int, st: dict, x_full: float, y_full: float):
            x0 = float(st["x0"])
            y0 = float(st["y0"])
            cw = float(st["cw"])
            ch = float(st["ch"])

            u = (x_full - x0) / max(1.0, (cw - 1.0))
            v = (y_full - y0) / max(1.0, (ch - 1.0))
            u = float(np.clip(u, 0.0, 1.0))
            v = float(np.clip(v, 0.0, 1.0))

            px = int(round(u * max(0, pm_w - 1)))
            py = int(round(v * max(0, pm_h - 1)))
            px = int(np.clip(px, 0, max(0, pm_w - 1)))
            py = int(np.clip(py, 0, max(0, pm_h - 1)))
            return px, py

        pm_ax = _rgb_to_qpixmap(rgb_ax_c)
        pm_co = _rgb_to_qpixmap(rgb_co_c)
        pm_sa = _rgb_to_qpixmap(rgb_sa_c)

        ax_px, ax_py = _cross_in_crop(pm_ax.width(), pm_ax.height(), ax_st, ax_xr, ax_yr)
        co_px, co_py = _cross_in_crop(pm_co.width(), pm_co.height(), co_st, co_xr, co_yr)
        sa_px, sa_py = _cross_in_crop(pm_sa.width(), pm_sa.height(), sa_st, sa_xr, sa_yr)

        pm_ax = self._draw_crosshair_on_pixmap(pm_ax, ax_px, ax_py)
        pm_co = self._draw_crosshair_on_pixmap(pm_co, co_px, co_py)
        pm_sa = self._draw_crosshair_on_pixmap(pm_sa, sa_px, sa_py)

        self.lbl_axi.setPixmap(
            pm_ax.scaled(self.lbl_axi.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.lbl_cor.setPixmap(
            pm_co.scaled(self.lbl_cor.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.lbl_sag.setPixmap(
            pm_sa.scaled(self.lbl_sag.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        self._update_lr_overlay(self.axi_panel, "L", "R")
        self._update_lr_overlay(self.cor_panel, "L", "R")

        mid_x = self.fxmax / 2.0
        hemi = "L" if self.ix <= mid_x else "R"
        self._update_lr_overlay(self.sag_panel, hemi, hemi)

        self._update_header()
        self._update_slice_titles()
        self._update_manual_plane_styles()

    def _update_lr_overlay(self, panel: dict, left_text: str, right_text: str):
        try:
            c = panel["container"]
            l = panel["lbl_left"]
            r = panel["lbl_right"]

            l.setText(left_text)
            r.setText(right_text)

            l.adjustSize()
            r.adjustSize()

            margin = 10
            top = 8

            l.move(margin, top)
            r.move(c.width() - r.width() - margin, top)

            l.raise_()
            r.raise_()

        except Exception:
            pass

    # ----------------------------
    # UI callbacks
    # ----------------------------
    def _on_params_changed(self):
        self.alpha = float(self.sl_alpha.value()) / 100.0
        self.g_fixed = float(self.sl_gfixed.value()) / 100.0
        self.g_moving = float(self.sl_gmoving.value()) / 100.0
        self._update_views()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

        if not bool(getattr(self, "_ui_ready", False)):
            return

        if not hasattr(self, "lbl_axi"):
            return

        # Debounce expensive image reconstruction while resizing.
        self._resize_redraw_timer.start()

    def _redraw_after_resize(self) -> None:
        if not bool(getattr(self, "_ui_ready", False)):
            return

        try:
            self._update_views()
        except Exception:
            pass

    # ----------------------------
    # Mouse interaction (crosshair)
    # ----------------------------
    def _on_point(self, view: str, lx: float, ly: float):
        if view == "axial":
            base = self._base_axial(self.fixed_np, self.iz)  # (y,x)
            h0, w0 = base.shape
            rot = _rot90_k(np.zeros((h0, w0), np.uint8), self.K_AXIAL)
            hr, wr = rot.shape

            x_r, y_r = _map_display_to_base(
                lx, ly, hr, wr, self.lbl_axi.width(), self.lbl_axi.height()
            )
            # undo LR flip in rotated space before inverse-rot mapping
            x_r = _apply_flip_to_x(x_r, int(wr), self.FLIP_LR_AXIAL)
            x0, y0 = _rot_map_display_to_base(x_r, y_r, h0=h0, w0=w0, k=self.K_AXIAL)

            self.ix = int(np.clip(round(x0), 0, self.fxmax))
            self.iy = int(np.clip(round(y0), 0, self.fymax))

        elif view == "coronal":
            base = self._base_coronal(self.fixed_np, self.iy)  # (z,x)
            h0, w0 = base.shape
            rot = _rot90_k(np.zeros((h0, w0), np.uint8), self.K_CORONAL)
            hr, wr = rot.shape

            x_r, y_r = _map_display_to_base(
                lx, ly, hr, wr, self.lbl_cor.width(), self.lbl_cor.height()
            )
            x_r = _apply_flip_to_x(x_r, int(wr), self.FLIP_LR_CORONAL)
            x0, y0 = _rot_map_display_to_base(x_r, y_r, h0=h0, w0=w0, k=self.K_CORONAL)

            self.ix = int(np.clip(round(x0), 0, self.fxmax))
            self.iz = int(np.clip(round(y0), 0, self.fzmax))

        else:  # sagittal
            base = self._base_sagittal(self.fixed_np, self.ix)  # (z,y)
            h0, w0 = base.shape
            rot = _rot90_k(np.zeros((h0, w0), np.uint8), self.K_SAGITTAL)
            hr, wr = rot.shape

            x_r, y_r = _map_display_to_base(
                lx, ly, hr, wr, self.lbl_sag.width(), self.lbl_sag.height()
            )
            x_r = _apply_flip_to_x(x_r, int(wr), self.FLIP_LR_SAGITTAL)
            x0, y0 = _rot_map_display_to_base(x_r, y_r, h0=h0, w0=w0, k=self.K_SAGITTAL)

            self.iy = int(np.clip(round(x0), 0, self.fymax))
            self.iz = int(np.clip(round(y0), 0, self.fzmax))

        self._update_views()
