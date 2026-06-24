from __future__ import annotations

import colorsys
import json
import math
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
    QStandardItem,
    QStandardItemModel,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QLabel,
    QLineEdit,
    QListView,
    QScrollBar,
    QVBoxLayout,
    QWidget,
)

from ..state import (
    ROLE_CONTACT_INDEX,
    ROLE_ELEC_ID,
    ROLE_KIND,
    AppState,
    Volume,
)
from ..ui.edit_contact_dialog import EditContactDialog
from ..ui.export_coordinates_dialog import ExportCoordinatesDialog
from ..ui.neuxelec_message_dialog import NeuXelecMessageDialog
from ..utils.electrode_geometry import axis_decomposition, next_contact_voxel


# -----------------------------
# Display helpers
# -----------------------------
def _norm01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo = np.percentile(a, 1)
    hi = np.percentile(a, 99)
    if hi <= lo:
        hi = lo + 1.0
    a = (a - lo) / (hi - lo)
    return np.clip(a, 0.0, 1.0)


def _to_qpixmap_gray(img2d: np.ndarray) -> QPixmap:
    """img2d float32 in [0..1] -> QPixmap grayscale (Qt requires C-contiguous buffer)."""
    u8 = (np.clip(img2d, 0.0, 1.0) * 255.0).astype(np.uint8, copy=False)
    u8 = np.ascontiguousarray(u8)  # ✅ critical for QImage
    h, w = u8.shape
    qimg = QImage(u8.data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())  # copy => Qt owns buffer


def _draw_crosshair(
    pix: QPixmap, x: int, y: int, color: Qt.GlobalColor, thickness: int = 1
) -> QPixmap:
    out = QPixmap(pix)
    painter = QPainter(out)
    painter.setRenderHint(QPainter.Antialiasing, False)
    pen = QPen(color)
    pen.setWidth(max(1, int(thickness)))
    painter.setPen(pen)
    w = out.width()
    h = out.height()
    painter.drawLine(0, y, w, y)
    painter.drawLine(x, 0, x, h)
    painter.end()
    return out


def _flip_lr(img: np.ndarray) -> np.ndarray:
    return img[:, ::-1]


def _draw_lr_markers(pm: QPixmap, left_text: str = "L", right_text: str = "R") -> QPixmap:
    out = QPixmap(pm)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen(Qt.red)
    pen.setWidth(2)
    p.setPen(pen)

    font = QFont()
    font.setBold(True)
    font.setPointSize(18)
    p.setFont(font)

    margin = 10
    p.drawText(margin, margin + 18, left_text)
    fm = p.fontMetrics()
    tw = fm.horizontalAdvance(right_text)
    p.drawText(max(margin, out.width() - margin - tw), margin + 18, right_text)

    p.end()
    return out


def _draw_lps_footer(pm: QPixmap, text: str) -> QPixmap:
    """Draw a small footer bottom-left with LPS coordinates."""
    out = QPixmap(pm)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing, True)

    font = QFont()
    font.setPointSize(10)
    font.setBold(True)
    p.setFont(font)

    fm = p.fontMetrics()
    pad = 6
    margin = 6

    tw = fm.horizontalAdvance(text)
    th = fm.height()

    box_w = tw + pad * 2
    box_h = th + pad * 2

    x0 = margin
    y0 = max(margin, out.height() - margin - box_h)

    p.fillRect(x0, y0, box_w, box_h, QColor(0, 0, 0, 160))
    p.setPen(Qt.white)
    p.drawText(x0 + pad, y0 + pad + fm.ascent(), text)

    p.end()
    return out


def _draw_points(
    pm: QPixmap, pts_xy: list[tuple[int, int]], color: QColor, radius: int = 3
) -> QPixmap:
    """Draw filled circles for electrode contacts (overlay)."""
    if not pts_xy:
        return pm
    out = QPixmap(pm)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing, True)

    pen = QPen(color)
    pen.setWidth(1)
    p.setPen(pen)
    p.setBrush(color)  # filled

    for x, y in pts_xy:
        p.drawEllipse(int(x - radius), int(y - radius), int(2 * radius), int(2 * radius))
    p.end()
    return out


def _resource_electrodes_ref_path() -> Path:
    """
    Backward-compatible default path.

    The old TXT file is still supported.
    A JSON file with the same basename is loaded in addition when present:
        electrodes_ref.txt
        electrodes_ref.json
    """
    here = Path(__file__).resolve()
    return (here.parent.parent / "utils" / "electrodes_ref.txt").resolve()


def _resource_electrodes_ref_json_path() -> Path:
    return _resource_electrodes_ref_path().with_suffix(".json")


def _sanitize_spacing_profile(profile) -> list[float]:
    """
    Return a clean list of inter-contact distances.

    Example for D08-18PIX:
        17 intervals for 18 contacts.
    """
    if not isinstance(profile, (list, tuple)):
        return []

    out = []

    for value in profile:
        try:
            d = float(str(value).replace(",", "."))
        except Exception:
            continue

        if d > 0:
            out.append(float(d))

    return out


def _normalize_ref_info(raw: dict) -> dict:
    """
    Normalize reference definitions from either TXT or JSON.

    Supported JSON keys:
        n / nb_of_contacts / num_contacts
        d / inter_contact_distance / inter_contact_distance_mm
        skip / contacts_not_connected
        spacing_profile_mm
    """
    if not isinstance(raw, dict):
        return {"n": 0, "d": 0.0, "skip": [], "spacing_profile_mm": []}

    try:
        n = int(
            raw.get("n")
            or raw.get("nb_of_contacts")
            or raw.get("num_contacts")
            or raw.get("number_of_contacts")
            or 0
        )
    except Exception:
        n = 0

    try:
        d = float(
            str(
                raw.get("d")
                or raw.get("inter_contact_distance")
                or raw.get("inter_contact_distance_mm")
                or raw.get("distance_mm")
                or 0.0
            ).replace(",", ".")
        )
    except Exception:
        d = 0.0

    skip_raw = raw.get("skip", raw.get("contacts_not_connected", []))

    if isinstance(skip_raw, str):
        skip_values = skip_raw.split(",")
    elif isinstance(skip_raw, (list, tuple)):
        skip_values = skip_raw
    else:
        skip_values = []

    skip = []

    for value in skip_values:
        try:
            skip.append(int(value))
        except Exception:
            pass

    spacing_profile = _sanitize_spacing_profile(raw.get("spacing_profile_mm", []))

    return {
        "n": int(n),
        "d": float(d),
        "skip": sorted(set(skip)),
        "spacing_profile_mm": spacing_profile,
    }


def _parse_electrode_refs_txt(txt_path: Path) -> dict:
    """
    Old TXT format:
        REF_NAME  nb_contacts  inter_contact_mm  [contacts_not_connected]
    """
    refs = {}

    if not txt_path.exists():
        return refs

    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()

        if not s or s.startswith("#"):
            continue

        parts = s.split()

        if len(parts) < 3:
            continue

        name = parts[0]

        try:
            nb = int(float(parts[1].replace(",", ".")))
            dist = float(parts[2].replace(",", "."))
        except Exception:
            continue

        skip_contacts = []

        if len(parts) >= 4:
            raw = parts[3].strip()

            if raw:
                for tok in raw.split(","):
                    tok = tok.strip()

                    if not tok:
                        continue

                    try:
                        skip_contacts.append(int(tok))
                    except Exception:
                        pass

        refs[name] = _normalize_ref_info(
            {
                "n": nb,
                "d": dist,
                "skip": skip_contacts,
            }
        )

    return refs


def _parse_electrode_refs_json(json_path: Path) -> dict:
    """
    New JSON format.

    Example:
        {
          "D08-18PIX": {
            "n": 18,
            "d": 2.0,
            "skip": [],
            "spacing_profile_mm": [2.0, ..., 3.5]
          }
        }
    """
    refs = {}

    if not json_path.exists():
        return refs

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print("[Electrode references] Could not read electrodes_ref.json:", e)
        return refs

    if not isinstance(data, dict):
        return refs

    for name, raw in data.items():
        name = str(name or "").strip()

        if not name:
            continue

        refs[name] = _normalize_ref_info(raw)

    return refs


def _parse_electrode_refs(txt_path: Path):
    """
    Load electrode references without breaking old behavior.

    Priority:
        1. TXT references are loaded first.
        2. JSON references are loaded second and can add/override refs.
        3. Other is always available.
    """
    refs = {}

    try:
        refs.update(_parse_electrode_refs_txt(txt_path))
    except Exception as e:
        print("[Electrode references] TXT loading failed:", e)

    try:
        refs.update(_parse_electrode_refs_json(txt_path.with_suffix(".json")))
    except Exception as e:
        print("[Electrode references] JSON loading failed:", e)

    refs.setdefault(
        "Other",
        {
            "n": 0,
            "d": 0.0,
            "skip": [],
            "spacing_profile_mm": [],
        },
    )

    return refs


def _random_electrode_color(existing_colors=None):
    existing_colors = existing_colors or []

    best = None
    best_score = -1.0

    for _ in range(40):
        h = random.random()
        s = random.uniform(0.65, 0.95)
        v = random.uniform(0.80, 1.00)

        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cand = (int(r * 255), int(g * 255), int(b * 255))

        if not existing_colors:
            return cand

        dmin = min(
            ((cand[0] - c[0]) ** 2 + (cand[1] - c[1]) ** 2 + (cand[2] - c[2]) ** 2) ** 0.5
            for c in existing_colors
        )

        if dmin > best_score:
            best_score = dmin
            best = cand

    return best if best is not None else (255, 165, 0)


# -----------------------------
# Clickable QLabel
# -----------------------------
class ClickableImageLabel(QLabel):
    clicked = Signal(int, int)
    doubleClicked = Signal(int, int)
    dragged = Signal(int, int)
    wheeled = Signal(int, int)  # delta, modifiers (Qt.KeyboardModifiers as int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self._drag = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag = True
            self.clicked.emit(int(event.position().x()), int(event.position().y()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag:
            self.dragged.emit(int(event.position().x()), int(event.position().y()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit(int(event.position().x()), int(event.position().y()))
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        mods = event.modifiers().value  # ✅ conversion propre en int
        self.wheeled.emit(delta, mods)
        event.accept()


# -----------------------------
# Main Page
# -----------------------------
class ReconstructionPage:
    """
    - Vue neuro (flip LR)
    - Axial rotated 180° for nose up
    - Crosshair sync
    - Ctrl+wheel: slice scroll
    - wheel: zoom (crop+rescale) without changing widget size
    - Shift+drag: pan
    - Drag without Shift: move crosshair only

    NEW:
    - Pick deepest + second, then Estimate reconstruct contacts in LPS mm
    - Display electrode + contacts coords in tv_Electrodes
    - Draw contacts as points on views
    """

    ROT_CORONAL_K = 2
    ROT_SAGITTAL_K = 2
    ROT_AXIAL_K = 2  # 180° so nose is UP

    FLIP_LR_AXIAL = True
    FLIP_LR_CORONAL = True
    FLIP_LR_SAGITTAL = True

    def __init__(self, ui_root: QWidget, state: AppState):
        self.ui = ui_root
        self.state = state

        # ---- Find frames (image containers) from your UI ----
        self.frame_cor = self._find(QFrame, ["frameReco_coronal"])
        self.frame_sag = self._find(QFrame, ["frameReco_sagittal"])
        self.frame_axi = self._find(QFrame, ["frameReco_axial"])

        # ---- Find scrollbars ----
        self.scroll_cor = self._find(
            QScrollBar, ["scrollReco_coronal", "scroll_cor", "scroll_Reco_cor"]
        )
        self.scroll_sag = self._find(
            QScrollBar, ["scrollReco_sagittal", "scroll_sag", "scroll_Reco_sag"]
        )
        self.scroll_axi = self._find(
            QScrollBar, ["scrollReco_axial", "scroll_axi", "scroll_Reco_axi"]
        )

        # ---- Ensure we have actual QLabel to draw images ----
        self.lbl_cor = self._ensure_image_label(self.frame_cor)
        self.lbl_sag = self._ensure_image_label(self.frame_sag)
        self.lbl_axi = self._ensure_image_label(self.frame_axi)

        # --- Shortcut: Ctrl+S => show current crosshair on 3D view ---
        self._shortcut_show_on_3d = QShortcut(QKeySequence("Ctrl+D"), self.ui)
        self._shortcut_show_on_3d.activated.connect(self._show_crosshair_on_3d_view)

        # --- Shortcut: Ctrl+Z => hide current 3D crosshair marker only ---
        self._shortcut_hide_3d_marker = QShortcut(QKeySequence("Ctrl+Z"), self.ui)
        self._shortcut_hide_3d_marker.activated.connect(self._hide_crosshair_on_3d_view)

        # ---- Electrode ref combo ----
        self.combo_ref = self._find(QComboBox, ["comboReco_electrodeRef"])
        self._ref_placeholder = "Select electrode reference"
        self.edit_nb_contacts = self._find(QLineEdit, ["editReco_nbContacts", "le_Reco_nbContacts"])
        self.edit_interdist = self._find(
            QLineEdit,
            ["editReco_interContact", "editReco_interContactDist", "le_Reco_interContactDist"],
        )
        for le in (self.edit_nb_contacts, self.edit_interdist):
            if le is not None:
                le.setEnabled(True)
                le.setReadOnly(True)

        # ---- Electrode name + hemisphere ----
        self.edit_elec_name = self._find(QLineEdit, ["editReco_nameElec", "le_Reco_nameElec"])
        self.chk_hemi_left = self._find(QCheckBox, ["chkReco_hemiLeft"])
        self.chk_hemi_right = self._find(QCheckBox, ["chkReco_hemiRight"])
        self._hemi_group = QButtonGroup(self.ui)
        self._hemi_group.setExclusive(True)

        self._hemi_group.addButton(self.chk_hemi_right)
        self._hemi_group.addButton(self.chk_hemi_left)

        # ---- Snapping options (Voxeloc-like) ----
        self.chk_snapping = self._find(QCheckBox, ["chkReco_snapping"])
        self.chk_localmax = self._find(QCheckBox, ["chkReco_localMax", "chkReco_localmax"])
        # Default: snapping ON (as requested)
        if self.chk_snapping is not None:
            self.chk_snapping.setChecked(True)

        self._crosshair_locked = False

        # ---- Pick buttons ----
        self.btn_pick_deep = self._find(
            QAbstractButton, ["btnReco_pickDeepest", "btn_Reco_pickDeepest"]
        )
        self.btn_pick_second = self._find(
            QAbstractButton, ["btnReco_pickSecond", "btn_Reco_pickSecond"]
        )
        self.btn_estimate = self._find(QAbstractButton, ["btnReco_estimate", "btn_Reco_estimate"])
        self.btn_new_elec = self._find(QAbstractButton, ["btnReco_newElectrode"])
        self.btn_export_coordinates = self._find(QAbstractButton, ["Export_Coordinates_1"])

        # ---- Coordinate edits (optional) ----
        # NOTE: the .ui uses "editReco_deepestX/Y/Z" and "editReco_secondtX/Y/Z" (typo secondt)
        self.edit_deep_x = self._find(
            QLineEdit, ["editReco_deepestX", "editReco_deepX", "le_Reco_deepX"]
        )
        self.edit_deep_y = self._find(
            QLineEdit, ["editReco_deepestY", "editReco_deepY", "le_Reco_deepY"]
        )
        self.edit_deep_z = self._find(
            QLineEdit, ["editReco_deepestZ", "editReco_deepZ", "le_Reco_deepZ"]
        )

        self.edit_second_x = self._find(
            QLineEdit, ["editReco_secondtX", "editReco_secondX", "le_Reco_secondX"]
        )
        self.edit_second_y = self._find(
            QLineEdit, ["editReco_secondtY", "editReco_secondY", "le_Reco_secondY"]
        )
        self.edit_second_z = self._find(
            QLineEdit, ["editReco_secondtZ", "editReco_secondZ", "le_Reco_secondZ"]
        )

        # ---- Electrode list view in UI ----

        self.lw_electrodes = self._find(QListView, ["tv_Electrodes"])
        # Shared model across pages (Reconstruction / 3D / Oblique)
        self._elec_model = getattr(self.state, "electrodes_model", QStandardItemModel())
        if self.lw_electrodes is not None:
            self.lw_electrodes.setModel(self._elec_model)
            # Voxeloc-like: vertical list only (no horizontal scrolling), wrap long coordinate text
            try:
                self.lw_electrodes.setWordWrap(True)
                self.lw_electrodes.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                self.lw_electrodes.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            except Exception:
                pass

        # React to visibility toggles (checkboxes) and selection (click electrode/contact)
        try:
            self._elec_model.itemChanged.connect(self._on_electrode_item_changed)  # type: ignore
        except Exception:
            pass
        try:
            if self.lw_electrodes is not None and self.lw_electrodes.selectionModel() is not None:
                self.lw_electrodes.selectionModel().currentChanged.connect(self._on_electrode_selection_changed)  # type: ignore
        except Exception:
            pass

        # picked points
        self._deep_idx = None
        self._second_idx = None
        self._deep_lps = None
        self._second_lps = None

        # ---- Electrode refs file ----
        self._refs = _parse_electrode_refs(_resource_electrodes_ref_path())
        self._init_ref_combo()

        # ---- Orientation mode state ----
        self._orientation_mode = "research"

        # ---- Orientation mode buttons (Radiologic / Research) ----
        # research: neurologic (L on left) => flip LR ON
        # radiologic: radiology (R on left) => flip LR OFF
        self.btn_radiologic_view = self.ui.findChild(QAbstractButton, "btn_radiologicView")
        self.btn_research_view = self.ui.findChild(QAbstractButton, "btn_researchView")

        if self.btn_radiologic_view is not None:
            self.btn_radiologic_view.clicked.connect(
                lambda: self.set_orientation_mode("radiologic")
            )
        if self.btn_research_view is not None:
            self.btn_research_view.clicked.connect(lambda: self.set_orientation_mode("research"))

        # Default
        self.set_orientation_mode(self._orientation_mode)

        # ---- Electrode reconstruction storage (shared in state) ----
        if not hasattr(self.state, "electrodes"):
            self.state.electrodes = []
        self._electrodes = self.state.electrodes
        self._active_electrode = None  # last built electrode dict
        self._editing_elec_id: int | None = None
        self._editing_contact_target: tuple[int, int] | None = None
        self._contact_edit_dialog: EditContactDialog | None = None

        # Name of a duplicated electrode explicitly accepted by the user with
        # "Continue". This prevents showing the same warning repeatedly until
        # the text is changed again.
        self._accepted_duplicate_name: str | None = None

        # picked points
        self._deep_idx = None  # (ix, iy, iz) voxel
        self._second_idx = None  # (ix, iy, iz) voxel
        self._deep_lps = None  # (x,y,z) mm
        self._second_lps = None  # (x,y,z) mm

        # ---- Picking state ----
        self._pick_mode: str | None = None  # "deepest" or "second"
        self._deep_picked: tuple[int, int, int] | None = None
        self._second_picked: tuple[int, int, int] | None = None

        # ---- Zoom state per view ----
        self._zoom_cor = 1.0
        self._zoom_sag = 1.0
        self._zoom_axi = 1.0
        self._zoom_step = 1.10
        self._zoom_min = 1.0
        self._zoom_max = 8.0

        # ---- View center (independent from crosshair) for zoom/pan ----
        self._view_center = {"axial": None, "coronal": None, "sagittal": None}

        # ---- Pan drag state (Shift+drag) ----
        self._pan_drag_active = False
        self._pan_drag_plane: str | None = None
        self._pan_last_xy: tuple[int, int] | None = None

        # ---- Crop window state (for click mapping) ----
        self._view_state = {"axial": None, "coronal": None, "sagittal": None}

        # Crosshair indices in CT space (z,y,x)
        self.iz = 0
        self.iy = 0
        self.ix = 0

        # Connect UI signals
        if self.scroll_axi is not None:
            self.scroll_axi.valueChanged.connect(lambda _: self._on_scroll_changed("axial"))
        if self.scroll_cor is not None:
            self.scroll_cor.valueChanged.connect(lambda _: self._on_scroll_changed("coronal"))
        if self.scroll_sag is not None:
            self.scroll_sag.valueChanged.connect(lambda _: self._on_scroll_changed("sagittal"))

        if self.btn_pick_deep is not None:
            self.btn_pick_deep.clicked.connect(lambda: self._pick_from_crosshair("deepest"))
        if self.btn_pick_second is not None:
            self.btn_pick_second.clicked.connect(lambda: self._pick_from_crosshair("second"))
        if self.btn_estimate is not None:
            self.btn_estimate.clicked.connect(self._reconstruct_electrode_from_two_points)
            self.btn_estimate.setEnabled(False)

        if self.btn_new_elec is not None:
            self.btn_new_elec.clicked.connect(self._reset_current_electrode_inputs)
        if self.btn_export_coordinates is not None:
            self.btn_export_coordinates.clicked.connect(self._open_export_coordinates_dialog)

        # Enable/disable estimate based on form completion
        if self.edit_elec_name is not None:
            self.edit_elec_name.textChanged.connect(lambda _: self._update_estimate_enabled())

            # Reset a previously accepted duplicate as soon as the user changes the name.
            self.edit_elec_name.textChanged.connect(self._on_electrode_name_text_changed)

            # Check for duplicate name only when the user has finished editing:
            # pressing Enter or leaving the text field.
            self.edit_elec_name.editingFinished.connect(self._check_duplicate_electrode_name)
        if self.combo_ref is not None:
            self.combo_ref.currentTextChanged.connect(lambda _: self._update_estimate_enabled())
        if self.edit_nb_contacts is not None:
            self.edit_nb_contacts.textChanged.connect(lambda _: self._update_estimate_enabled())

        if self.edit_interdist is not None:
            self.edit_interdist.textChanged.connect(lambda _: self._update_estimate_enabled())
        if self.chk_hemi_left is not None:
            self.chk_hemi_left.toggled.connect(self._on_hemi_toggled)
        if self.chk_hemi_right is not None:
            self.chk_hemi_right.toggled.connect(self._on_hemi_toggled)

        # Click/drag in each view
        self.lbl_cor.clicked.connect(lambda x, y: self._on_click("coronal", self.lbl_cor, x, y))
        self.lbl_cor.dragged.connect(lambda x, y: self._on_drag("coronal", self.lbl_cor, x, y))

        self.lbl_sag.clicked.connect(lambda x, y: self._on_click("sagittal", self.lbl_sag, x, y))
        self.lbl_sag.dragged.connect(lambda x, y: self._on_drag("sagittal", self.lbl_sag, x, y))

        self.lbl_axi.clicked.connect(lambda x, y: self._on_click("axial", self.lbl_axi, x, y))
        self.lbl_axi.dragged.connect(lambda x, y: self._on_drag("axial", self.lbl_axi, x, y))

        self.lbl_cor.doubleClicked.connect(lambda x, y: self._on_double_click_reset_view("coronal"))
        self.lbl_sag.doubleClicked.connect(
            lambda x, y: self._on_double_click_reset_view("sagittal")
        )
        self.lbl_axi.doubleClicked.connect(lambda x, y: self._on_double_click_reset_view("axial"))

        # Wheel: zoom / slice navigation
        self.lbl_cor.wheeled.connect(lambda d, m: self._on_wheel("coronal", d, m))
        self.lbl_sag.wheeled.connect(lambda d, m: self._on_wheel("sagittal", d, m))
        self.lbl_axi.wheeled.connect(lambda d, m: self._on_wheel("axial", d, m))

        # Init if CT already loaded
        self.init_from_volume()

        # Initial gate
        self._update_estimate_enabled()

        self._page_electrode_visible = {}
        self._page_contacts_visible = {}

        self._live_contact_pick_enabled: bool = False

    def _open_export_coordinates_dialog(self) -> None:
        parent = self.ui.window() if self.ui is not None else None

        try:
            dlg = ExportCoordinatesDialog(self.state, parent=parent)
            dlg.exec()

        except Exception as e:
            NeuXelecMessageDialog.critical(
                parent,
                "Export coordinates failed",
                ("The export coordinates window could not be opened.\n\n" f"Details:\n{e}"),
            )

    def _get_local_electrode_visible(self, elec_id: int) -> bool:
        return bool(self._page_electrode_visible.get(int(elec_id), True))

    def _get_local_contacts_visible(self, elec_id: int, n_contacts: int):
        vals = self._page_contacts_visible.get(int(elec_id))
        if not isinstance(vals, list) or len(vals) != int(n_contacts):
            vals = [True] * int(n_contacts)
            self._page_contacts_visible[int(elec_id)] = vals
        return vals

    def set_electrode_visible(self, elec_id: int, visible: bool) -> None:
        try:
            elec = self.state.electrodes[int(elec_id)]
            n = len(elec.get("contacts_lps", []) or [])
        except Exception:
            return
        self._page_electrode_visible[int(elec_id)] = bool(visible)
        self._page_contacts_visible[int(elec_id)] = [bool(visible)] * n
        self.render_all()

    def set_contact_visible(self, elec_id: int, contact_idx: int, visible: bool) -> None:
        try:
            elec = self.state.electrodes[int(elec_id)]
            n = len(elec.get("contacts_lps", []) or [])
        except Exception:
            return
        vals = self._get_local_contacts_visible(int(elec_id), n)
        if 0 <= int(contact_idx) < len(vals):
            vals[int(contact_idx)] = bool(visible)
        self._page_electrode_visible[int(elec_id)] = any(vals)
        self.render_all()

    def _on_hemi_toggled(self, checked: bool) -> None:
        # Keep it exclusive
        try:
            if self.chk_hemi_left is not None and self.chk_hemi_right is not None:
                if self.sender() == self.chk_hemi_left and checked:
                    self.chk_hemi_right.blockSignals(True)
                    self.chk_hemi_right.setChecked(False)
                    self.chk_hemi_right.blockSignals(False)
                if self.sender() == self.chk_hemi_right and checked:
                    self.chk_hemi_left.blockSignals(True)
                    self.chk_hemi_left.setChecked(False)
                    self.chk_hemi_left.blockSignals(False)
        except Exception:
            pass

        # If editing an existing electrode, update its hemisphere immediately
        try:
            if self._editing_elec_id is not None and 0 <= self._editing_elec_id < len(
                self._electrodes
            ):
                hemi = self._current_hemi()
                if hemi is not None:
                    self._electrodes[self._editing_elec_id]["hemisphere"] = hemi
                    if hasattr(self.state, "notify_electrodes_changed"):
                        self.state.notify_electrodes_changed()
        except Exception:
            pass

        self._update_estimate_enabled()

    def _current_hemi(self) -> str | None:
        if self.chk_hemi_left is not None and self.chk_hemi_left.isChecked():
            return "L"
        if self.chk_hemi_right is not None and self.chk_hemi_right.isChecked():
            return "R"
        return None

    def _normalize_electrode_name(self, name: str) -> str:
        """
        Normalize electrode names for duplicate detection.

        Examples considered identical:
            "A" and "a"
            " A " and "A"
            "post  P" and "post P"
        """
        return " ".join(str(name).strip().split()).casefold()

    def _on_electrode_name_text_changed(self, _text: str) -> None:
        """
        Reset the user's previous 'Continue' choice whenever the name changes.
        """
        self._accepted_duplicate_name = None

    def _check_duplicate_electrode_name(self) -> None:
        """
        Warn the user when an electrode name already exists.

        The check is triggered only after editing is finished, not on every typed
        character. When editing an existing electrode, its own current name is
        ignored so selecting or updating an electrode does not trigger a false
        duplicate warning.
        """
        if self.edit_elec_name is None:
            return

        entered_name = self.edit_elec_name.text().strip()

        if not entered_name:
            return

        normalized_name = self._normalize_electrode_name(entered_name)

        # The user already accepted this exact duplicated name.
        if normalized_name == self._accepted_duplicate_name:
            return

        duplicate_found = False

        for elec_id, elec in enumerate(self._electrodes):
            # When editing an existing electrode, do not compare it with itself.
            if self._editing_elec_id is not None and int(elec_id) == int(self._editing_elec_id):
                continue

            existing_name = str(elec.get("name", "")).strip()

            if not existing_name:
                continue

            if self._normalize_electrode_name(existing_name) == normalized_name:
                duplicate_found = True
                break

        if not duplicate_found:
            return

        parent = self.ui.window() if self.ui is not None else None

        continue_with_duplicate = NeuXelecMessageDialog.question(
            parent,
            "Electrode already created",
            (
                f'An electrode named "{entered_name}" has already been created.\n\n'
                "Do you want to continue with this name or change it?"
            ),
            accept_text="Continue",
            reject_text="Change",
        )

        if continue_with_duplicate:
            # Keep the duplicated name and do not show the warning again
            # unless the user modifies the name field.
            self._accepted_duplicate_name = normalized_name
            return

        # Change: remove the duplicated name and return focus to the field.
        self._accepted_duplicate_name = None
        self.edit_elec_name.clear()
        self.edit_elec_name.setFocus()
        self._update_estimate_enabled()

    def _spacing_profile_for_ref_info(
        self,
        ref_info: dict,
        n_total: int | None = None,
    ) -> list[float]:
        """
        Return a valid variable spacing profile for a reference.

        n_total is the total number of physical positions before removing skipped
        non-connected contacts. For N positions, we need N - 1 intervals.
        """
        if not isinstance(ref_info, dict):
            return []

        profile = _sanitize_spacing_profile(ref_info.get("spacing_profile_mm", []))

        if not profile:
            return []

        if n_total is not None:
            try:
                expected = max(0, int(n_total) - 1)
            except Exception:
                expected = 0

            if len(profile) != expected:
                return []

        return profile

    def _spacing_profile_display_text(self, profile: list[float]) -> str:
        """
        Compact UI text for variable inter-contact spacing.
        """
        profile = _sanitize_spacing_profile(profile)

        if not profile:
            return ""

        unique_values = []

        for value in profile:
            if not any(abs(float(value) - float(v)) < 1e-6 for v in unique_values):
                unique_values.append(float(value))

        if len(unique_values) == 1:
            return f"{unique_values[0]:.1f}"

        return " / ".join(f"{v:.1f}" for v in unique_values)

    def _update_estimate_enabled(self) -> None:
        """Estimate should only be clickable when all required inputs are provided."""
        if self.btn_estimate is None:
            return

        name_ok = bool(self.edit_elec_name is not None and self.edit_elec_name.text().strip())

        placeholder = getattr(
            self,
            "_ref_placeholder",
            "Select electrode reference",
        )

        ref_name = self.combo_ref.currentText().strip() if self.combo_ref is not None else ""

        ref_ok = bool(ref_name and ref_name != placeholder and ref_name in self._refs)

        hemi_ok = self._current_hemi() is not None

        pts_ok = self._deep_idx is not None and self._second_idx is not None

        params_ok = False

        try:
            n_contacts = int(float(self.edit_nb_contacts.text().strip()))

            ref_info = self._refs.get(ref_name, {})
            skip_contacts = set(ref_info.get("skip", []))
            n_total = int(n_contacts) + int(len(skip_contacts))

            spacing_profile = self._spacing_profile_for_ref_info(
                ref_info,
                n_total=n_total,
            )

            if spacing_profile:
                params_ok = bool(n_contacts > 1 and len(spacing_profile) == n_total - 1)
            else:
                d_mm = float(self.edit_interdist.text().strip().replace(",", "."))
                params_ok = bool(n_contacts > 1 and d_mm > 0)

        except Exception:
            params_ok = False

        self.btn_estimate.setEnabled(bool(name_ok and ref_ok and hemi_ok and pts_ok and params_ok))

    def _reset_current_electrode_inputs(self) -> None:
        """Prepare UI for a new electrode (does not erase already estimated electrodes)."""
        self._editing_elec_id = None
        self._editing_contact_target = None
        self._contact_edit_dialog = None
        self._accepted_duplicate_name = None
        self._pick_mode = None
        self._deep_picked = None
        self._second_picked = None
        self._deep_idx = None
        self._second_idx = None
        self._deep_lps = None
        self._second_lps = None

        for le in (
            self.edit_deep_x,
            self.edit_deep_y,
            self.edit_deep_z,
            self.edit_second_x,
            self.edit_second_y,
            self.edit_second_z,
        ):
            try:
                if le is not None:
                    le.clear()
            except Exception:
                pass

        if self.edit_elec_name is not None:
            self.edit_elec_name.clear()

        if self.combo_ref is not None:
            self.combo_ref.blockSignals(True)
            self.combo_ref.setCurrentIndex(0)
            self.combo_ref.blockSignals(False)

        if self.edit_nb_contacts is not None:
            self.edit_nb_contacts.clear()
            self.edit_nb_contacts.setEnabled(True)
            self.edit_nb_contacts.setReadOnly(True)
            self.edit_nb_contacts.setPlaceholderText("Number of contacts")

        if self.edit_interdist is not None:
            self.edit_interdist.clear()
            self.edit_interdist.setEnabled(True)
            self.edit_interdist.setReadOnly(True)
            self.edit_interdist.setPlaceholderText("Distance in mm")

        # Default behavior: snapping ON, local-max OFF (Voxeloc-like)
        if getattr(self, "chk_snapping", None) is not None:
            self.chk_snapping.setChecked(True)
        if getattr(self, "chk_localmax", None) is not None:
            self.chk_localmax.setChecked(False)

        # Update shared electrode list model immediately (no page switch needed)
        try:
            # expanded by default after estimate
            self.state.electrodes[-1]["expanded"] = True
        except Exception:
            pass
        try:
            if hasattr(self.state, "notify_electrodes_changed"):
                self.state.notify_electrodes_changed()
        except Exception:
            pass
        try:
            self._refresh_shared_electrodes_views()
        except Exception:
            pass

        # Reset pick/crosshair visual state
        self._crosshair_locked = False
        self._pick_mode = None
        self._clear_pick_button_highlight()
        # Reset hemisphere checkboxes cleanly
        try:
            if self._hemi_group is not None:
                self._hemi_group.setExclusive(False)

            for cb in (self.chk_hemi_left, self.chk_hemi_right):
                if cb is not None:
                    cb.blockSignals(True)
                    cb.setChecked(False)
                    cb.blockSignals(False)

            if self._hemi_group is not None:
                self._hemi_group.setExclusive(True)

        except Exception:
            pass
        self._update_estimate_enabled()

        self.render_all()

    # -----------------------------
    # LPS reference image for coords
    # -----------------------------
    def _ct_ref_for_lps(self) -> sitk.Image:
        """
        Image used to compute physical LPS coordinates for reconstruction.

        The coregistered CT is authorized only after it has been checked again
        in the current session.
        """
        if self._ct_blocked_for_reconstruction():
            raise RuntimeError("CT must be revalidated before Reconstruction.")
        if (
            bool(getattr(self.state, "ct_ready_for_reconstruction", False))
            and getattr(self.state, "ct_coreg_in_t1", None) is not None
        ):
            img = getattr(self.state, "ct_coreg_in_t1", None)
            if isinstance(img, sitk.Image):
                return img

        ct_path = getattr(self.state, "ct_path", None)
        if ct_path:
            return sitk.ReadImage(ct_path)

        img = self._get_ct_sitk()
        if isinstance(img, sitk.Image):
            return img

        raise RuntimeError("No CT available for LPS coordinates.")

    def _main_window_bridge(self):
        # Preferred: explicit bridge if stored in state
        mw = getattr(self.state, "main_window", None)
        if mw is not None and hasattr(mw, "set_current_page_by_name"):
            return mw

        # Fallback: search among top-level widgets
        for w in QApplication.topLevelWidgets():
            try:
                if hasattr(w, "set_current_page_by_name"):
                    return w
            except Exception:
                pass

        return None

    def _current_crosshair_lps(self) -> tuple[float, float, float] | None:
        try:
            img = self._ct_ref_for_lps()
            p = img.TransformIndexToPhysicalPoint(
                (
                    int(self.ix),
                    int(self.iy),
                    int(self.iz),
                )
            )
            return (float(p[0]), float(p[1]), float(p[2]))
        except Exception:
            return None

    def _show_crosshair_on_3d_view(self) -> None:
        lps = self._current_crosshair_lps()
        if lps is None:
            return

        vp = getattr(self.state, "view3d_page", None)
        if vp is not None and hasattr(vp, "show_crosshair_marker_lps"):
            try:
                vp.show_crosshair_marker_lps(lps)
            except Exception:
                pass

        mw = self._main_window_bridge()
        if mw is not None:
            try:
                mw.set_current_page_by_name("page3DView")
            except Exception:
                pass

    def _hide_crosshair_on_3d_view(self) -> None:
        """
        Hide the red 3D crosshair marker from Reconstruction page.

        Important:
        This only removes the visible actor. It does not erase the stored marker
        position in the 3D view, so Ctrl+D can show a new marker and Ctrl+C +
        left-click/drag in 3D can still reuse the last known depth.
        """
        vp = getattr(self.state, "view3d_page", None)

        if vp is not None and hasattr(vp, "hide_crosshair_marker"):
            try:
                vp.hide_crosshair_marker()
            except Exception:
                pass

    def _reset_all_views(self) -> None:
        """
        Reset zoom and pan in all three reconstruction views.
        """
        self._zoom_axi = 1.0
        self._zoom_cor = 1.0
        self._zoom_sag = 1.0

        self._view_center["axial"] = None
        self._view_center["coronal"] = None
        self._view_center["sagittal"] = None

    def _on_double_click_reset_view(self, _plane: str) -> None:
        """
        A double-click in any view resets all three linked views.
        """
        self._reset_all_views()
        self.render_all()

    # -----------------------------
    # Utility: robust findChild
    # -----------------------------
    def _find(self, cls, names: list[str]):
        for n in names:
            w = self.ui.findChild(cls, n)
            if w is not None:
                return w
        return None

    def _ensure_image_label(self, frame: QFrame | None) -> ClickableImageLabel:
        if frame is None:
            lbl = ClickableImageLabel()
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background-color: rgb(30,30,30); color: white;")
            lbl.setText("[No CT loaded]")
            return lbl

        for ch in frame.children():
            if isinstance(ch, ClickableImageLabel):
                return ch

        lay = frame.layout()
        if lay is None:
            lay = QVBoxLayout(frame)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)

        lbl = ClickableImageLabel(frame)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("background-color: rgb(30,30,30); color: white;")
        lbl.setText("[No CT loaded]")
        lay.addWidget(lbl)
        return lbl

    # -----------------------------
    # Electrode refs logic
    # -----------------------------

    def _is_other_ref(self, ref_name: str | None = None) -> bool:
        """
        True when the selected electrode reference is the custom/manual one.
        """
        if ref_name is None:
            if self.combo_ref is None:
                return False
            ref_name = self.combo_ref.currentText()

        return str(ref_name or "").strip().lower() == "other"

    def _init_ref_combo(self):
        if self.combo_ref is None:
            return

        self.combo_ref.blockSignals(True)
        self.combo_ref.clear()

        placeholder = getattr(self, "_ref_placeholder", "Select electrode reference")
        self.combo_ref.addItem(placeholder)

        if self._refs:
            self.combo_ref.addItems(sorted(self._refs.keys()))

        self.combo_ref.setCurrentIndex(0)
        self.combo_ref.blockSignals(False)

        self.combo_ref.currentTextChanged.connect(self._on_ref_changed)
        self.combo_ref.currentTextChanged.connect(lambda _: self._update_estimate_enabled())

        self._on_ref_changed(self.combo_ref.currentText())

    def _on_ref_changed(self, ref_name: str):
        placeholder = getattr(self, "_ref_placeholder", "Select electrode reference")

        # ---------------------------------------------------------
        # No valid reference selected:
        # fields visible but locked.
        # ---------------------------------------------------------
        if not ref_name or ref_name == placeholder or ref_name not in self._refs:
            if self.edit_nb_contacts is not None:
                self.edit_nb_contacts.clear()
                self.edit_nb_contacts.setEnabled(True)
                self.edit_nb_contacts.setReadOnly(True)
                self.edit_nb_contacts.setPlaceholderText("Number of contacts")

            if self.edit_interdist is not None:
                self.edit_interdist.clear()
                self.edit_interdist.setEnabled(True)
                self.edit_interdist.setReadOnly(True)
                self.edit_interdist.setPlaceholderText("Distance in mm")

            self._update_estimate_enabled()
            return

        # ---------------------------------------------------------
        # Other:
        # fields become editable.
        # ---------------------------------------------------------
        if self._is_other_ref(ref_name):
            if self.edit_nb_contacts is not None:
                self.edit_nb_contacts.clear()
                self.edit_nb_contacts.setEnabled(True)
                self.edit_nb_contacts.setReadOnly(False)
                self.edit_nb_contacts.setPlaceholderText("Number of contacts")

            if self.edit_interdist is not None:
                self.edit_interdist.clear()
                self.edit_interdist.setEnabled(True)
                self.edit_interdist.setReadOnly(False)
                self.edit_interdist.setPlaceholderText("Distance in mm")

            self._update_estimate_enabled()
            return

        # ---------------------------------------------------------
        # Known reference:
        # fields are filled automatically and locked.
        # ---------------------------------------------------------
        ref_info = self._refs[ref_name]
        nb = int(ref_info.get("n", 0))
        dist = float(ref_info.get("d", 0.0))
        skip_contacts = set(ref_info.get("skip", []))
        n_total = int(nb) + int(len(skip_contacts))

        spacing_profile = self._spacing_profile_for_ref_info(
            ref_info,
            n_total=n_total,
        )

        if self.edit_nb_contacts is not None:
            self.edit_nb_contacts.setReadOnly(True)
            self.edit_nb_contacts.setText(str(nb))

        if self.edit_interdist is not None:
            self.edit_interdist.setReadOnly(True)

            if spacing_profile:
                self.edit_interdist.setText(self._spacing_profile_display_text(spacing_profile))
                self.edit_interdist.setToolTip(
                    "Variable spacing profile: "
                    + ", ".join(f"{float(v):.1f} mm" for v in spacing_profile)
                )
            else:
                self.edit_interdist.setText(f"{dist:.1f}")
                self.edit_interdist.setToolTip("")

        self._update_estimate_enabled()

    def _ct_blocked_for_reconstruction(self) -> bool:
        """
        Reconstruction safety gate.

        Any loaded CT must be visually revalidated during the current session
        before it can be displayed or used in the Reconstruction page.

        This is intentional after loading a JSON project:
        - ct_validated can be True for 3D View / Oblique Slice restoration;
        - ct_ready_for_reconstruction must still be True before Reconstruction
        can display the CT.
        """
        ct_loaded = bool(
            getattr(self.state, "ct_path", None)
            or getattr(self.state, "ct_coreg_path", None)
            or getattr(self.state, "ct_coreg_in_t1", None) is not None
            or getattr(self.state, "ct_in_t1", None) is not None
        )

        return bool(ct_loaded and not getattr(self.state, "ct_ready_for_reconstruction", False))

    # -----------------------------
    # CT source: prefer SITK (same as OverlayViewer)
    # -----------------------------
    def _get_ct_sitk(self) -> sitk.Image | None:
        """
        Return the CT used by Reconstruction.

        Important:
        A validated CT restored from the JSON is available for visualization,
        but Reconstruction must use it only after the CT has been checked again
        during the current session.
        """
        if self._ct_blocked_for_reconstruction():
            return None

        if bool(getattr(self.state, "ct_ready_for_reconstruction", False)):
            img = getattr(self.state, "ct_coreg_in_t1", None)
            if isinstance(img, sitk.Image):
                return img

        ct_path = getattr(self.state, "ct_path", None)
        if isinstance(ct_path, str) and ct_path:
            try:
                return sitk.ReadImage(ct_path)
            except Exception:
                pass

        vol_obj = getattr(self.state, "volume", None)
        if isinstance(vol_obj, Volume) and vol_obj.data is not None:
            a = np.asarray(vol_obj.data)
            if a.ndim == 4:
                a = a[..., 0]
            if a.ndim == 3:
                a_zyx = np.transpose(a, (2, 1, 0))
                img = sitk.GetImageFromArray(a_zyx.astype(np.float32, copy=False))
                return img

        return None

    def _ct_np_zyx(self) -> np.ndarray | None:
        img = self._get_ct_sitk()
        if img is None:
            return None
        arr = sitk.GetArrayFromImage(img).astype(np.float32, copy=False)  # (z,y,x)
        if arr.ndim == 4:
            arr = arr[..., 0]
        if arr.ndim != 3:
            return None
        return arr

    # -----------------------------
    # Voxel utilities (Voxeloc-like)
    # -----------------------------
    def _clip_idx_xyz(
        self, idx_xyz: tuple[int, int, int], size_xyz: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        x, y, z = idx_xyz
        sx, sy, sz = size_xyz
        x = int(np.clip(int(round(x)), 0, sx - 1))
        y = int(np.clip(int(round(y)), 0, sy - 1))
        z = int(np.clip(int(round(z)), 0, sz - 1))
        return x, y, z

    def _local_max_in_kernel(
        self, center_xyz: tuple[float, float, float], radius_xyz: tuple[int, int, int]
    ) -> tuple[int, int, int] | None:
        """Return voxel index (x,y,z) of max CT intensity in a kernel around center (in voxel coordinates)."""
        arr = self._ct_np_zyx()
        if arr is None:
            return None

        img = self._get_ct_sitk()
        if img is None:
            return None
        sx, sy, sz = img.GetSize()  # (x,y,z)

        cx, cy, cz = center_xyz
        rx, ry, rz = radius_xyz

        x0 = max(0, int(math.floor(cx - rx)))
        x1 = min(sx - 1, int(math.ceil(cx + rx)))
        y0 = max(0, int(math.floor(cy - ry)))
        y1 = min(sy - 1, int(math.ceil(cy + ry)))
        z0 = max(0, int(math.floor(cz - rz)))
        z1 = min(sz - 1, int(math.ceil(cz + rz)))

        if x1 < x0 or y1 < y0 or z1 < z0:
            return None

        # arr is (z,y,x)
        sub = arr[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
        if sub.size == 0:
            return None

        flat = int(np.argmax(sub))
        dz, dy, dx = np.unravel_index(flat, sub.shape)
        return (x0 + int(dx), y0 + int(dy), z0 + int(dz))

    # -----------------------------
    # Init views
    # -----------------------------
    def init_from_volume(self):
        if self._ct_blocked_for_reconstruction():
            self._show_coreg_warning()
            return

        vol = self._ct_np_zyx()
        if vol is None:
            self._show_no_ct()
            return

        z, y, x = vol.shape

        if self.scroll_axi is not None:
            self.scroll_axi.setMinimum(0)
            self.scroll_axi.setMaximum(max(0, z - 1))
            self.scroll_axi.setValue(z // 2)

        if self.scroll_cor is not None:
            self.scroll_cor.setMinimum(0)
            self.scroll_cor.setMaximum(max(0, y - 1))
            self.scroll_cor.setValue(y // 2)

        if self.scroll_sag is not None:
            self.scroll_sag.setMinimum(0)
            self.scroll_sag.setMaximum(max(0, x - 1))
            self.scroll_sag.setValue(x // 2)

        self.iz = z // 2
        self.iy = y // 2
        self.ix = x // 2

        self._view_center["axial"] = None
        self._view_center["coronal"] = None
        self._view_center["sagittal"] = None

        self.render_all()

    def _show_no_ct(self):
        for lbl in (self.lbl_cor, self.lbl_sag, self.lbl_axi):
            if lbl is not None:
                lbl.setText("[No CT loaded]")
        if self.btn_estimate is not None:
            self.btn_estimate.setEnabled(False)

    def _show_coreg_warning(self):
        msg = "[Please validate the coregistration CT - MRI]"
        for lbl in (self.lbl_cor, self.lbl_sag, self.lbl_axi):
            if lbl is not None:
                lbl.clear()
                lbl.setPixmap(QPixmap())
                lbl.setText(msg)

    # -----------------------------
    # Picking logic
    # -----------------------------
    def _pick_from_crosshair(self, mode: str):
        """Voxeloc behavior: clicking the button captures current crosshair immediately."""
        self._pick_mode = mode
        self._set_pick_button_active(mode)
        self._apply_pick_if_needed()

    def _start_pick(self, mode: str):
        self._pick_mode = mode
        self.render_all()

    def _apply_pick_if_needed(self):
        if self._pick_mode is None:
            return

        # Current crosshair in voxel indices (x,y,z) on CT-in-T1 grid
        x = int(self.ix)
        y = int(self.iy)
        z = int(self.iz)

        # Optional: snap the picked point to local max CT intensity (Voxeloc-style)
        if getattr(self, "chk_snapping", None) is not None and self.chk_snapping.isChecked():
            snapped = self._local_max_in_kernel((x, y, z), radius_xyz=(2, 2, 2))
            if snapped is not None:
                x, y, z = snapped
                # update crosshair to snapped voxel
                self.ix, self.iy, self.iz = float(x), float(y), float(z)

        # Convert voxel -> LPS mm for display
        try:
            img = self._ct_ref_for_lps()
            lps = tuple(map(float, img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))))
        except Exception:
            lps = None

        def _set_lps_fields(le_x, le_y, le_z, lps_val):
            if lps_val is None:
                # fallback: show voxels
                if le_x is not None:
                    le_x.setText(str(int(x)))
                if le_y is not None:
                    le_y.setText(str(int(y)))
                if le_z is not None:
                    le_z.setText(str(int(z)))
                return
            if le_x is not None:
                le_x.setText(f"{lps_val[0]:.2f}")
            if le_y is not None:
                le_y.setText(f"{lps_val[1]:.2f}")
            if le_z is not None:
                le_z.setText(f"{lps_val[2]:.2f}")

        if self._pick_mode == "deepest":
            self._deep_picked = (x, y, z)
            self._deep_idx = (int(x), int(y), int(z))
            self._deep_lps = lps
            _set_lps_fields(self.edit_deep_x, self.edit_deep_y, self.edit_deep_z, lps)

        elif self._pick_mode == "second":
            self._second_picked = (x, y, z)
            self._second_idx = (int(x), int(y), int(z))
            self._second_lps = lps
            _set_lps_fields(self.edit_second_x, self.edit_second_y, self.edit_second_z, lps)

        self._crosshair_locked = True
        self._pick_mode = None
        self._update_estimate_enabled()

        # Sync scrollbars to the picked voxel so overlays update immediately (Voxeloc-like)
        if self.scroll_axi is not None:
            self.scroll_axi.blockSignals(True)
            self.scroll_axi.setValue(int(self.iz))
            self.scroll_axi.blockSignals(False)
        if self.scroll_cor is not None:
            self.scroll_cor.blockSignals(True)
            self.scroll_cor.setValue(int(self.iy))
            self.scroll_cor.blockSignals(False)
        if self.scroll_sag is not None:
            self.scroll_sag.blockSignals(True)
            self.scroll_sag.setValue(int(self.ix))
            self.scroll_sag.blockSignals(False)

        # Snapping may have slightly moved the picked voxel.
        # Recenter every zoomed view on the final selected location.
        self._recenter_zoomed_views_on_crosshair()

        self.render_all()

    def set_crosshair_from_lps(self, lps_xyz) -> None:
        """
        Called from 3D View.
        Convert an LPS physical point to CT/T1 voxel index and move crosshair.
        """
        try:
            img = self._ct_ref_for_lps()
            lps = tuple(float(v) for v in lps_xyz)

            idx = img.TransformPhysicalPointToIndex(lps)
            sx, sy, sz = img.GetSize()

            x = int(np.clip(int(idx[0]), 0, sx - 1))
            y = int(np.clip(int(idx[1]), 0, sy - 1))
            z = int(np.clip(int(idx[2]), 0, sz - 1))

            self.jump_to_voxel(x, y, z)
        except Exception:
            pass

    # -----------------------------
    # Jump helpers
    # -----------------------------
    def jump_to_voxel(self, x: int, y: int, z: int) -> None:
        """
        Move crosshair to the given voxel index (x, y, z) and refresh all linked
        zoomed views around this position.
        """
        try:
            self.ix, self.iy, self.iz = float(x), float(y), float(z)
        except Exception:
            return

        # Sync scrollbars so the displayed slices match immediately.
        if self.scroll_axi is not None:
            self.scroll_axi.blockSignals(True)
            self.scroll_axi.setValue(int(self.iz))
            self.scroll_axi.blockSignals(False)

        if self.scroll_cor is not None:
            self.scroll_cor.blockSignals(True)
            self.scroll_cor.setValue(int(self.iy))
            self.scroll_cor.blockSignals(False)

        if self.scroll_sag is not None:
            self.scroll_sag.blockSignals(True)
            self.scroll_sag.setValue(int(self.ix))
            self.scroll_sag.blockSignals(False)

        # A jump coming from a contact click or from the 3D View must center
        # the selected voxel in every zoomed reconstruction view.
        self._recenter_zoomed_views_on_crosshair()

        self.render_all()

    def jump_to_contact(self, elec_id: int, contact_index: int) -> None:
        """Jump crosshair to a contact (Voxeloc-style) when a contact row is clicked."""
        try:
            elec = self.state.electrodes[int(elec_id)]
            contacts_idx = elec.get("contacts_idx", []) or []
            if contact_index < 0 or contact_index >= len(contacts_idx):
                return
            vx, vy, vz = contacts_idx[int(contact_index)]
            self.jump_to_voxel(int(vx), int(vy), int(vz))
        except Exception:
            return

    def _update_electrodes_ui(self):
        """Refresh the shared electrode model (used by the 3 electrode lists)."""
        try:
            if hasattr(self.state, "rebuild_electrodes_model"):
                if hasattr(self.state, "notify_electrodes_changed"):
                    self.state.notify_electrodes_changed()
            else:
                # Fallback: keep previous behavior if state has no helpers
                if self._elec_model is None:
                    return
                self._elec_model.clear()
                for elec in self._electrodes:
                    header = QStandardItem(f"{elec.get('name','')} | ref={elec.get('ref','')}")
                    header.setEditable(False)
                    self._elec_model.appendRow(header)
        except Exception:
            pass

    def _on_electrode_item_changed(self, item: QStandardItem) -> None:
        # Visibility is now handled page-locally by ElectrodesController.
        # Keep this method as a no-op to avoid reintroducing global visibility state.
        return

    def _on_electrode_selection_changed(self, current, previous) -> None:
        """Click on electrode/contact in list:
        - electrode: re-fill parameters
        - contact: jump crosshair to that contact (for quick verification)
        """
        try:
            idx = current
            if idx is None or not idx.isValid():
                return
            item = self._elec_model.itemFromIndex(idx)
            if item is None:
                return
            kind = item.data(ROLE_KIND)
            elec_id = int(item.data(ROLE_ELEC_ID))
            contact_index = int(item.data(ROLE_CONTACT_INDEX))
        except Exception:
            return

        if elec_id < 0 or elec_id >= len(self._electrodes):
            return

        elec = self._electrodes[elec_id]
        # Enter edit mode for the selected electrode
        # Synchronize internal reconstruction state with the selected electrode
        self._editing_elec_id = int(elec_id)
        self._active_electrode = elec

        try:
            deep_idx = elec.get("deep_idx", None)
            second_idx = elec.get("second_idx", None)

            self._deep_idx = tuple(map(int, deep_idx)) if deep_idx is not None else None
            self._second_idx = tuple(map(int, second_idx)) if second_idx is not None else None

            self._deep_picked = self._deep_idx
            self._second_picked = self._second_idx
        except Exception:
            self._deep_idx = None
            self._second_idx = None
            self._deep_picked = None
            self._second_picked = None

        try:
            deep_lps = elec.get("deepest_lps", None)
            second_lps = elec.get("second_lps", None)

            self._deep_lps = tuple(deep_lps) if deep_lps is not None else None
            self._second_lps = tuple(second_lps) if second_lps is not None else None
        except Exception:
            self._deep_lps = None
            self._second_lps = None

        # Always show electrode parameters when clicking anywhere in the electrode group
        try:
            if self.edit_elec_name is not None:
                self.edit_elec_name.setText(str(elec.get("name", "")))
            if self.combo_ref is not None and elec.get("ref") is not None:
                # try to select matching ref in combo
                ref = str(elec.get("ref"))
                for i in range(self.combo_ref.count()):
                    if self.combo_ref.itemText(i) == ref:
                        self.combo_ref.setCurrentIndex(i)
                        break
            if self.chk_hemi_left is not None:
                self.chk_hemi_left.setChecked(
                    str(elec.get("hemisphere", "")).lower().startswith("l")
                )
            if self.chk_hemi_right is not None:
                self.chk_hemi_right.setChecked(
                    str(elec.get("hemisphere", "")).lower().startswith("r")
                )

            # deepest / second displayed as LPS in UI
            deep_lps = elec.get("deepest_lps")
            sec_lps = elec.get("second_lps")
            if deep_lps is not None:
                self._set_lps_edits(self.edit_deep_x, self.edit_deep_y, self.edit_deep_z, deep_lps)
            if sec_lps is not None:
                self._set_lps_edits(
                    self.edit_second_x, self.edit_second_y, self.edit_second_z, sec_lps
                )

            if self.edit_nb_contacts is not None and elec.get("n") is not None:
                self.edit_nb_contacts.setText(str(elec.get("n")))
            if self.edit_interdist is not None:
                spacing_profile = elec.get("spacing_profile_mm", []) or []

                if spacing_profile:
                    self.edit_interdist.setText(
                        str(
                            elec.get("spacing_label")
                            or self._spacing_profile_display_text(spacing_profile)
                        )
                    )
                    self.edit_interdist.setToolTip(
                        "Variable spacing profile: "
                        + ", ".join(f"{float(v):.1f} mm" for v in spacing_profile)
                    )

                elif elec.get("d_mm") is not None:
                    self.edit_interdist.setText(f"{float(elec.get('d_mm')):.2f}")
                    self.edit_interdist.setToolTip("")
        except Exception:
            pass

        # If clicking a contact: jump to slice and focus
        if kind == "contact":
            try:
                contacts_idx = elec.get("contacts_idx") or []
                if 0 <= contact_index < len(contacts_idx):
                    vx, vy, vz = [int(x) for x in contacts_idx[contact_index]]
                    self._set_crosshair_vox(vx, vy, vz)
            except Exception:
                pass

        # refresh estimate availability
        try:
            self._update_estimate_enabled()
        except Exception:
            pass

    def _reconstruct_electrode_from_two_points(self):
        """Voxeloc-like reconstruction:
        - Work in voxel space on CT(in T1) grid
        - Optional LocalMax refinement per contact
        - Always store/display LPS (mm) alongside voxels
        """
        # Defensive gate (button should already be disabled when incomplete)
        if self._deep_idx is None or self._second_idx is None:
            return
        if self.edit_nb_contacts is None or self.edit_interdist is None:
            return
        if self.edit_elec_name is None:
            return
        hemi = self._current_hemi()
        if hemi is None:
            return

        ref_name = self.combo_ref.currentText() if self.combo_ref is not None else "UnknownRef"
        ref_info = self._refs.get(ref_name, {})
        skip_contacts = set(ref_info.get("skip", []))

        try:
            n_connected = int(float(self.edit_nb_contacts.text().strip()))

            n_total_candidate = int(n_connected) + int(len(skip_contacts))

            spacing_profile = self._spacing_profile_for_ref_info(
                ref_info,
                n_total=n_total_candidate,
            )

            if spacing_profile:
                d_mm = float(ref_info.get("d", spacing_profile[0]))
            else:
                d_mm = float(self.edit_interdist.text().strip().replace(",", "."))

        except Exception:
            return

        if n_connected <= 1 or d_mm <= 0:
            return

        # Use CT image as geometry reference + intensity source
        try:
            img = self._ct_ref_for_lps()
        except Exception:
            return

        arr = self._ct_np_zyx()
        if arr is None:
            return

        # Points in voxel space (x,y,z)
        p0 = np.array(self._deep_idx, dtype=np.float64)
        p1 = np.array(self._second_idx, dtype=np.float64)

        v = p1 - p0
        delta_total = float(np.linalg.norm(v))
        if delta_total < 1e-6:
            return

        # Voxeloc assumes 1mm isotropic when working on CT-in-T1 grid (your case).
        # So we keep d_mm as "voxel distance".
        d = float(d_mm)

        # Axis decomposition for marching along the electrode.
        # Pure geometry, unit-tested in utils.electrode_geometry.
        delta_vec, sign_vec, _ = axis_decomposition(p0, p1)

        # LocalMax option (per-contact refinement)
        use_localmax = (
            getattr(self, "chk_localmax", None) is not None and self.chk_localmax.isChecked()
        )

        contacts_vox_f: list[tuple[float, float, float]] = []
        contacts_vox_f.append((float(p0[0]), float(p0[1]), float(p0[2])))

        sx, sy, sz = img.GetSize()

        elec_name = self.edit_elec_name.text().strip()
        if not elec_name:
            return

        # Reconstruct total physical positions:
        # connected contacts + non-connected contacts
        n_total = int(n_connected) + int(len(skip_contacts))

        for i in range(1, n_total):
            prev = np.array(contacts_vox_f[-1], dtype=np.float64)

            # ---------------------------------------------------------
            # Constant spacing for classical electrodes.
            # Variable spacing for references such as D08-18PIX.
            # spacing_profile contains N-1 intervals for N physical positions.
            # ---------------------------------------------------------
            if spacing_profile:
                step_d = float(spacing_profile[i - 1])
            else:
                step_d = float(d)

            # Predict next point (Voxeloc method), clipped to image bounds.
            # Pure geometry, unit-tested in utils.electrode_geometry.
            pred = np.array(
                next_contact_voxel(prev, delta_vec, sign_vec, step_d, bounds=(sx, sy, sz)),
                dtype=np.float64,
            )

            if use_localmax:
                # Find max CT intensity near predicted position
                mx = self._local_max_in_kernel(
                    (
                        float(pred[0]),
                        float(pred[1]),
                        float(pred[2]),
                    ),
                    radius_xyz=(1, 1, 1),
                )

                if mx is not None:
                    max_pt = np.array(mx, dtype=np.float64)
                    dv = max_pt - prev
                    nrm = float(np.linalg.norm(dv))

                    if nrm > 1e-6:
                        u = dv / nrm
                        refined = prev + u * step_d
                        refined[0] = np.clip(refined[0], 0, sx - 1)
                        refined[1] = np.clip(refined[1], 0, sy - 1)
                        refined[2] = np.clip(refined[2], 0, sz - 1)
                        pred = refined

            contacts_vox_f.append(
                (
                    float(pred[0]),
                    float(pred[1]),
                    float(pred[2]),
                )
            )

        # Round for drawing/indexing + convert to LPS for storage/display/export
        contacts_idx: list[tuple[int, int, int]] = []
        contacts_lps: list[tuple[float, float, float]] = []

        for xf, yf, zf in contacts_vox_f:
            ix = int(np.clip(int(round(xf)), 0, sx - 1))
            iy = int(np.clip(int(round(yf)), 0, sy - 1))
            iz = int(np.clip(int(round(zf)), 0, sz - 1))
            contacts_idx.append((ix, iy, iz))

            # ContinuousIndex -> PhysicalPoint (LPS mm)
            try:
                lps = img.TransformContinuousIndexToPhysicalPoint((float(xf), float(yf), float(zf)))
                contacts_lps.append((float(lps[0]), float(lps[1]), float(lps[2])))
            except Exception:
                lps = img.TransformIndexToPhysicalPoint((ix, iy, iz))
                contacts_lps.append((float(lps[0]), float(lps[1]), float(lps[2])))

        # Remove non-connected contacts according to electrodes_ref.txt
        if skip_contacts:
            filtered_idx = []
            filtered_lps = []

            for contact_number_1based, (idx, lps) in enumerate(
                zip(contacts_idx, contacts_lps), start=1
            ):
                if contact_number_1based in skip_contacts:
                    continue
                filtered_idx.append(idx)
                filtered_lps.append(lps)

            contacts_idx = filtered_idx
            contacts_lps = filtered_lps

        existing_colors = []
        for e in self._electrodes:
            c = e.get("color")
            if isinstance(c, (tuple, list)) and len(c) == 3:
                existing_colors.append((int(c[0]), int(c[1]), int(c[2])))

        new_color = _random_electrode_color(existing_colors)

        elec = {
            "name": elec_name,
            "hemisphere": hemi,
            "ref": ref_name,
            "n": len(contacts_idx),
            "n_connected": int(n_connected),
            "n_total": int(n_total),
            "skip_contacts": sorted(list(skip_contacts)),
            "d_mm": d_mm,
            "spacing_profile_mm": [float(x) for x in spacing_profile] if spacing_profile else [],
            "spacing_label": (
                self._spacing_profile_display_text(spacing_profile)
                if spacing_profile
                else f"{float(d_mm):.2f}"
            ),
            "deep_idx": tuple(map(int, self._deep_idx)),
            "second_idx": tuple(map(int, self._second_idx)),
            "contacts_lps": contacts_lps,
            "contacts_idx": contacts_idx,
            "contacts_visible": [True] * len(contacts_idx),
            "visible": True,
            "color": new_color,
            "deepest_lps": tuple(self._deep_lps) if self._deep_lps is not None else None,
            "second_lps": tuple(self._second_lps) if self._second_lps is not None else None,
        }

        # Update existing electrode if currently editing, otherwise create a new one
        if self._editing_elec_id is not None and 0 <= self._editing_elec_id < len(self._electrodes):
            old = self._electrodes[self._editing_elec_id]
            elec["color"] = old.get("color", elec["color"])
            elec["visible"] = old.get("visible", True)
            elec["expanded"] = old.get("expanded", True)

            old_cv = old.get("contacts_visible")
            if isinstance(old_cv, list):
                elec["contacts_visible"] = old_cv[: len(elec["contacts_idx"])] + [True] * max(
                    0, len(elec["contacts_idx"]) - len(old_cv)
                )

            self._electrodes[self._editing_elec_id] = elec
            self._active_electrode = elec
        else:
            self._electrodes.append(elec)
            self._active_electrode = elec
            self._editing_elec_id = len(self._electrodes) - 1

        self._update_electrodes_ui()

        # Jump to deepest contact so you immediately see at least one contact after Estimate
        try:
            ix0, iy0, iz0 = contacts_idx[0]
            self.ix, self.iy, self.iz = int(ix0), int(iy0), int(iz0)
            if self.scroll_axi is not None:
                self.scroll_axi.blockSignals(True)
                self.scroll_axi.setValue(int(self.iz))
                self.scroll_axi.blockSignals(False)
            if self.scroll_cor is not None:
                self.scroll_cor.blockSignals(True)
                self.scroll_cor.setValue(int(self.iy))
                self.scroll_cor.blockSignals(False)
            if self.scroll_sag is not None:
                self.scroll_sag.blockSignals(True)
                self.scroll_sag.setValue(int(self.ix))
                self.scroll_sag.blockSignals(False)
        except Exception:
            pass

        self.render_all()

        # If a new electrode was created, keep fields cleared for the next one.
        # If an existing electrode was edited, keep the current parameters visible.
        self._editing_elec_id = None
        self._active_electrode = None
        self._reset_current_electrode_inputs()

    def _refresh_shared_electrodes_views(self) -> None:
        """Force-refresh the shared electrodes list views (all pages) after estimate."""
        try:
            from PySide6.QtWidgets import QListView
        except Exception:
            return

        model = getattr(self.state, "electrodes_model", None)
        if model is None:
            return

        view_names = ["tv_Electrodes", "tv_Electrodes_2", "tv_Electrodes_3"]
        list_views = []
        for n in view_names:
            lv = self.ui.findChild(QListView, n)
            if lv is None:
                continue
            lv.setModel(model)
            try:
                lv.setWordWrap(True)
                lv.setHorizontalScrollBarPolicy(lv.ScrollBarAlwaysOff)
            except Exception:
                pass
            list_views.append(lv)

        rowmap = getattr(self.state, "_electrode_rowmap", {})
        if isinstance(rowmap, dict):
            for lv in list_views:
                for _, info in rowmap.items():
                    expanded = bool(info.get("expanded", True))
                    for r in info.get("meta", []) + info.get("contacts", []):
                        try:
                            lv.setRowHidden(int(r), not expanded)
                        except Exception:
                            pass
                try:
                    lv.viewport().update()
                except Exception:
                    pass

    def _on_scroll_changed(self, plane: str):
        if plane == "axial" and self.scroll_axi is not None:
            self.iz = int(self.scroll_axi.value())

        elif plane == "coronal" and self.scroll_cor is not None:
            self.iy = int(self.scroll_cor.value())

        elif plane == "sagittal" and self.scroll_sag is not None:
            self.ix = int(self.scroll_sag.value())

        if self._crosshair_locked:
            self._crosshair_locked = False
            self._clear_pick_button_highlight()

        # The manipulated plane keeps its visible crop position.
        # The two orthogonal zoomed views follow the new crosshair location.
        self._recenter_zoomed_views_on_crosshair(exclude_plane=plane)

        self.render_all()

    # -----------------------------
    # Slice extraction
    # -----------------------------
    def _slice_axial(self, vol_zyx: np.ndarray, iz: int) -> np.ndarray:
        iz = int(np.clip(iz, 0, vol_zyx.shape[0] - 1))
        sl = vol_zyx[iz, :, :]  # (y,x)
        sl = np.rot90(sl, k=self.ROT_AXIAL_K)  # nose up
        if self.FLIP_LR_AXIAL:
            sl = _flip_lr(sl)
        return sl

    def _slice_coronal(self, vol_zyx: np.ndarray, iy: int) -> np.ndarray:
        iy = int(np.clip(iy, 0, vol_zyx.shape[1] - 1))
        sl = vol_zyx[:, iy, :]  # (z,x)
        sl = np.rot90(sl, k=self.ROT_CORONAL_K)
        if self.FLIP_LR_CORONAL:
            sl = _flip_lr(sl)
        return sl

    def _slice_sagittal(self, vol_zyx: np.ndarray, ix: int) -> np.ndarray:
        ix = int(np.clip(ix, 0, vol_zyx.shape[2] - 1))
        sl = vol_zyx[:, :, ix]  # (z,y)
        sl = np.rot90(sl, k=self.ROT_SAGITTAL_K)
        if self.FLIP_LR_SAGITTAL:
            sl = _flip_lr(sl)
        return sl

    # -----------------------------
    # Zoom crop helpers
    # -----------------------------
    def _get_zoom(self, plane: str) -> float:
        if plane == "axial":
            return float(self._zoom_axi)
        if plane == "coronal":
            return float(self._zoom_cor)
        return float(self._zoom_sag)

    def _crosshair_display_position_for_plane(
        self,
        plane: str,
    ) -> tuple[float, float] | None:
        """
        Return the current crosshair position in the full displayed 2D image
        coordinates of one plane.

        The mapping is identical to the one used in render_all(), including
        radiologic/research left-right display conventions.
        """
        vol = self._ct_np_zyx()
        if vol is None:
            return None

        z, y, x = vol.shape
        plane = str(plane).lower().strip()

        if plane == "axial":
            full_h, full_w = self._slice_axial(vol, int(self.iz)).shape

            if self.FLIP_LR_AXIAL:
                px = (self.ix / max(1, x - 1)) * (full_w - 1)
            else:
                px = (1.0 - (self.ix / max(1, x - 1))) * (full_w - 1)

            py = (1.0 - (self.iy / max(1, y - 1))) * (full_h - 1)
            return float(px), float(py)

        if plane == "coronal":
            full_h, full_w = self._slice_coronal(vol, int(self.iy)).shape

            if self.FLIP_LR_CORONAL:
                px = (self.ix / max(1, x - 1)) * (full_w - 1)
            else:
                px = (1.0 - (self.ix / max(1, x - 1))) * (full_w - 1)

            py = (1.0 - (self.iz / max(1, z - 1))) * (full_h - 1)
            return float(px), float(py)

        if plane == "sagittal":
            full_h, full_w = self._slice_sagittal(vol, int(self.ix)).shape

            if self.FLIP_LR_SAGITTAL:
                px = (self.iy / max(1, y - 1)) * (full_w - 1)
            else:
                px = (1.0 - (self.iy / max(1, y - 1))) * (full_w - 1)

            py = (1.0 - (self.iz / max(1, z - 1))) * (full_h - 1)
            return float(px), float(py)

        return None

    def _recenter_zoomed_views_on_crosshair(
        self,
        exclude_plane: str | None = None,
    ) -> None:
        """
        Keep the crosshair visible in zoomed views after a navigation action.

        When exclude_plane is provided, the plane currently manipulated by the
        user keeps its existing crop position, while the other two views follow
        the new crosshair location.
        """
        exclude_plane = str(exclude_plane).lower().strip() if exclude_plane is not None else None

        for plane in ("axial", "coronal", "sagittal"):
            if plane == exclude_plane:
                continue

            if self._get_zoom(plane) <= 1.0001:
                self._view_center[plane] = None
                continue

            center = self._crosshair_display_position_for_plane(plane)

            if center is not None:
                self._view_center[plane] = [
                    float(center[0]),
                    float(center[1]),
                ]

    def _crop_for_zoom(
        self, img2d: np.ndarray, plane: str, cx: float, cy: float
    ) -> tuple[np.ndarray, dict[str, float]]:
        full_h, full_w = img2d.shape
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

        crop = img2d[y0 : y0 + ch, x0 : x0 + cw]

        st = {
            "full_w": float(full_w),
            "full_h": float(full_h),
            "x0": float(x0),
            "y0": float(y0),
            "cw": float(cw),
            "ch": float(ch),
        }
        return crop, st

    def _label_uv(self, lbl: QLabel, pm: QPixmap, px: int, py: int) -> tuple[float, float] | None:
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

    # -----------------------------
    # Mouse -> indices mapping
    # -----------------------------
    def _on_click(self, plane: str, lbl: QLabel, px: int, py: int):
        vol = self._ct_np_zyx()
        if vol is None:
            return

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

        uv = self._label_uv(lbl, pm, px, py)
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

        u_full = x_full / max(1.0, (full_w - 1.0))

        # rot90(k=2) already flips the slice, so LR compensation must be inverted
        if plane == "axial" and not self.FLIP_LR_AXIAL:
            u_full = 1.0 - u_full
        elif plane == "coronal" and not self.FLIP_LR_CORONAL:
            u_full = 1.0 - u_full
        elif plane == "sagittal" and not self.FLIP_LR_SAGITTAL:
            u_full = 1.0 - u_full

        v_full = y_full / max(1.0, (full_h - 1.0))

        zmax = vol.shape[0] - 1
        ymax = vol.shape[1] - 1
        xmax = vol.shape[2] - 1

        if plane == "axial":
            self.ix = int(np.clip(round(u_full * xmax), 0, xmax))
            self.iy = int(np.clip(round((1.0 - v_full) * ymax), 0, ymax))
        elif plane == "coronal":
            self.ix = int(np.clip(round(u_full * xmax), 0, xmax))
            self.iz = int(np.clip(round((1.0 - v_full) * zmax), 0, zmax))
        elif plane == "sagittal":
            self.iy = int(np.clip(round(u_full * ymax), 0, ymax))
            self.iz = int(np.clip(round((1.0 - v_full) * zmax), 0, zmax))

        if self.scroll_axi is not None:
            self.scroll_axi.blockSignals(True)
            self.scroll_axi.setValue(int(self.iz))
            self.scroll_axi.blockSignals(False)
        if self.scroll_cor is not None:
            self.scroll_cor.blockSignals(True)
            self.scroll_cor.setValue(int(self.iy))
            self.scroll_cor.blockSignals(False)
        if self.scroll_sag is not None:
            self.scroll_sag.blockSignals(True)
            self.scroll_sag.setValue(int(self.ix))
            self.scroll_sag.blockSignals(False)
        if self._pick_mode is None and self._crosshair_locked:
            self._crosshair_locked = False
            self._clear_pick_button_highlight()

        # Keep the active image stable under the mouse, but force the two other
        # zoomed views to follow the newly selected voxel.
        self._recenter_zoomed_views_on_crosshair(exclude_plane=plane)

        self._apply_pick_if_needed()
        self.render_all()

    def _on_drag(self, plane: str, lbl: QLabel, px: int, py: int):
        vol = self._ct_np_zyx()
        if vol is None:
            return

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

        self.render_all()

    def _on_wheel(self, plane: str, delta: int, modifiers: int):
        ctrl = modifiers & Qt.ControlModifier.value
        step_dir = +1 if delta > 0 else -1

        # Ctrl + wheel keeps the current slice navigation behavior.
        if ctrl:
            if plane == "axial" and self.scroll_axi is not None:
                self.scroll_axi.setValue(self.scroll_axi.value() + step_dir)

            elif plane == "coronal" and self.scroll_cor is not None:
                self.scroll_cor.setValue(self.scroll_cor.value() + step_dir)

            elif plane == "sagittal" and self.scroll_sag is not None:
                self.scroll_sag.setValue(self.scroll_sag.value() + step_dir)

            return

        # ---------------------------------------------------------
        # Linked zoom: scrolling over one plane applies the same zoom
        # level to axial, coronal and sagittal views.
        # ---------------------------------------------------------
        factor = self._zoom_step if delta > 0 else (1.0 / self._zoom_step)

        current_zoom = self._get_zoom(plane)
        new_zoom = float(
            np.clip(
                current_zoom * factor,
                self._zoom_min,
                self._zoom_max,
            )
        )

        if new_zoom <= 1.0001:
            new_zoom = 1.0

        self._zoom_axi = new_zoom
        self._zoom_cor = new_zoom
        self._zoom_sag = new_zoom

        if new_zoom <= 1.0001:
            self._view_center["axial"] = None
            self._view_center["coronal"] = None
            self._view_center["sagittal"] = None
        else:
            # Every view zooms around the current crosshair position.
            self._recenter_zoomed_views_on_crosshair()

        if self._crosshair_locked:
            self._crosshair_locked = False
            self._clear_pick_button_highlight()

        self.render_all()

    # -----------------------------
    # Rendering
    # -----------------------------
    def render_all(self):
        # Show warning whenever CT-MRI has not been validated in the current session.
        # This is intentional after loading a JSON project: even if a CT-in-T1 file
        # exists, the user must visually revalidate it before using Reconstruction.
        if self._ct_blocked_for_reconstruction():
            self._show_coreg_warning()
            return

        vol = self._ct_np_zyx()
        if vol is None:
            self._show_no_ct()
            return

        z, y, x = vol.shape

        self.iz = int(np.clip(self.iz, 0, z - 1))
        self.iy = int(np.clip(self.iy, 0, y - 1))
        self.ix = int(np.clip(self.ix, 0, x - 1))

        ax = self._slice_axial(vol, self.iz)
        cor = self._slice_coronal(vol, self.iy)
        sag = self._slice_sagittal(vol, self.ix)

        # Crosshair positions in displayed slice coordinates (must match display LR flip)
        ax_full_h, ax_full_w = ax.shape
        if self.FLIP_LR_AXIAL:
            ax_x_full = (self.ix / max(1, x - 1)) * (ax_full_w - 1)
        else:
            ax_x_full = (1.0 - (self.ix / max(1, x - 1))) * (ax_full_w - 1)
        ax_y_full = (1.0 - (self.iy / max(1, y - 1))) * (ax_full_h - 1)

        cor_full_h, cor_full_w = cor.shape
        if self.FLIP_LR_CORONAL:
            cor_x_full = (self.ix / max(1, x - 1)) * (cor_full_w - 1)
        else:
            cor_x_full = (1.0 - (self.ix / max(1, x - 1))) * (cor_full_w - 1)
        cor_y_full = (1.0 - (self.iz / max(1, z - 1))) * (cor_full_h - 1)

        sag_full_h, sag_full_w = sag.shape
        if self.FLIP_LR_SAGITTAL:
            sag_x_full = (self.iy / max(1, y - 1)) * (sag_full_w - 1)
        else:
            sag_x_full = (1.0 - (self.iy / max(1, y - 1))) * (sag_full_w - 1)
        sag_y_full = (1.0 - (self.iz / max(1, z - 1))) * (sag_full_h - 1)

        # Initialize view centers ONCE when zoom starts
        if self._get_zoom("axial") > 1.0001 and self._view_center["axial"] is None:
            self._view_center["axial"] = [float(ax_x_full), float(ax_y_full)]
        if self._get_zoom("coronal") > 1.0001 and self._view_center["coronal"] is None:
            self._view_center["coronal"] = [float(cor_x_full), float(cor_y_full)]
        if self._get_zoom("sagittal") > 1.0001 and self._view_center["sagittal"] is None:
            self._view_center["sagittal"] = [float(sag_x_full), float(sag_y_full)]

        ax_cx, ax_cy = (
            self._view_center["axial"]
            if self._get_zoom("axial") > 1.0001
            else [float(ax_x_full), float(ax_y_full)]
        )
        cor_cx, cor_cy = (
            self._view_center["coronal"]
            if self._get_zoom("coronal") > 1.0001
            else [float(cor_x_full), float(cor_y_full)]
        )
        sag_cx, sag_cy = (
            self._view_center["sagittal"]
            if self._get_zoom("sagittal") > 1.0001
            else [float(sag_x_full), float(sag_y_full)]
        )

        ax_crop, ax_st = self._crop_for_zoom(ax, "axial", ax_cx, ax_cy)
        cor_crop, cor_st = self._crop_for_zoom(cor, "coronal", cor_cx, cor_cy)
        sag_crop, sag_st = self._crop_for_zoom(sag, "sagittal", sag_cx, sag_cy)

        self._view_state["axial"] = ax_st
        self._view_state["coronal"] = cor_st
        self._view_state["sagittal"] = sag_st

        axn = _norm01(ax_crop)
        corn = _norm01(cor_crop)
        sagn = _norm01(sag_crop)

        pm_ax = _to_qpixmap_gray(axn)
        pm_cor = _to_qpixmap_gray(corn)
        pm_sag = _to_qpixmap_gray(sagn)

        pm_ax = pm_ax.scaled(self.lbl_axi.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pm_cor = pm_cor.scaled(self.lbl_cor.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pm_sag = pm_sag.scaled(self.lbl_sag.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        color = (
            Qt.red
            if (
                self._pick_mode is not None
                or self._live_contact_pick_enabled
                or self._crosshair_locked
            )
            else Qt.white
        )
        thick = 1

        def _cross_in_crop(
            pm: QPixmap, st: dict[str, float], x_full: float, y_full: float
        ) -> tuple[int, int]:
            x0 = float(st["x0"])
            y0 = float(st["y0"])
            cw = float(st["cw"])
            ch = float(st["ch"])

            u = (x_full - x0) / max(1.0, (cw - 1.0))
            v = (y_full - y0) / max(1.0, (ch - 1.0))
            u = float(np.clip(u, 0.0, 1.0))
            v = float(np.clip(v, 0.0, 1.0))

            px = int(round(u * max(0, pm.width() - 1)))
            py = int(round(v * max(0, pm.height() - 1)))
            px = int(np.clip(px, 0, max(0, pm.width() - 1)))
            py = int(np.clip(py, 0, max(0, pm.height() - 1)))
            return px, py

        ax_x, ax_y = _cross_in_crop(pm_ax, ax_st, ax_x_full, ax_y_full)
        cor_x, cor_y = _cross_in_crop(pm_cor, cor_st, cor_x_full, cor_y_full)
        sag_x, sag_y = _cross_in_crop(pm_sag, sag_st, sag_x_full, sag_y_full)

        pm_ax = _draw_crosshair(pm_ax, ax_x, ax_y, color=color, thickness=thick)
        pm_cor = _draw_crosshair(pm_cor, cor_x, cor_y, color=color, thickness=thick)
        pm_sag = _draw_crosshair(pm_sag, sag_x, sag_y, color=color, thickness=thick)

        # Draw reconstructed electrode points (all visible electrodes)
        try:
            for elec_id, elec in enumerate(getattr(self.state, "electrodes", [])):
                if not self._get_local_electrode_visible(elec_id):
                    continue
                color = elec.get("color", (0, 255, 0))
                qcol = QColor(int(color[0]), int(color[1]), int(color[2]))

                pts_ax: list[tuple[int, int]] = []
                pts_cor: list[tuple[int, int]] = []
                pts_sag: list[tuple[int, int]] = []

                contacts_idx = elec.get("contacts_idx", []) or []
                contacts_visible = self._get_local_contacts_visible(elec_id, len(contacts_idx))

                for ci, (ixc, iyc, izc) in enumerate(contacts_idx):
                    if not bool(contacts_visible[ci]):
                        continue

                    # Only show contacts that lie in the currently displayed slice (Voxeloc-like)
                    if int(izc) == int(self.iz):
                        # axial
                        if self.FLIP_LR_AXIAL:
                            ax_xf = (ixc / max(1, x - 1)) * (ax_full_w - 1)
                        else:
                            ax_xf = (1.0 - (ixc / max(1, x - 1))) * (ax_full_w - 1)
                        ax_yf = (1.0 - (iyc / max(1, y - 1))) * (ax_full_h - 1)
                        px, py = _cross_in_crop(pm_ax, ax_st, ax_xf, ax_yf)
                        pts_ax.append((px, py))

                    if int(iyc) == int(self.iy):
                        # coronal
                        if self.FLIP_LR_CORONAL:
                            cor_xf = (ixc / max(1, x - 1)) * (cor_full_w - 1)
                        else:
                            cor_xf = (1.0 - (ixc / max(1, x - 1))) * (cor_full_w - 1)
                        cor_yf = (1.0 - (izc / max(1, z - 1))) * (cor_full_h - 1)
                        px, py = _cross_in_crop(pm_cor, cor_st, cor_xf, cor_yf)
                        pts_cor.append((px, py))

                    if int(ixc) == int(self.ix):
                        # sagittal
                        if self.FLIP_LR_SAGITTAL:
                            sag_xf = (iyc / max(1, y - 1)) * (sag_full_w - 1)
                        else:
                            sag_xf = (1.0 - (iyc / max(1, y - 1))) * (sag_full_w - 1)
                        sag_yf = (1.0 - (izc / max(1, z - 1))) * (sag_full_h - 1)
                        px, py = _cross_in_crop(pm_sag, sag_st, sag_xf, sag_yf)
                        pts_sag.append((px, py))

                pm_ax = _draw_points(pm_ax, pts_ax, qcol)
                pm_cor = _draw_points(pm_cor, pts_cor, qcol)
                pm_sag = _draw_points(pm_sag, pts_sag, qcol)
        except Exception:
            pass

        # L/R markers depend on current orientation mode
        if getattr(self, "_orientation_mode", "research") == "radiologic":
            left_marker = "R"
            right_marker = "L"
        else:
            left_marker = "L"
            right_marker = "R"

        pm_ax = _draw_lr_markers(pm_ax, left_marker, right_marker)
        pm_cor = _draw_lr_markers(pm_cor, left_marker, right_marker)

        # On sagittal, show current hemisphere according to crosshair X position
        mid_x = (x - 1) / 2.0

        if self.ix <= mid_x:
            sag_marker = "L"
        else:
            sag_marker = "R"

        # same letter on both sides, to indicate which hemisphere the sagittal slice is in
        pm_sag = _draw_lr_markers(pm_sag, sag_marker, sag_marker)

        # LPS coordinates footer (updates with crosshair)
        ct_img = self._get_ct_sitk()
        if ct_img is not None:
            try:
                lps = ct_img.TransformIndexToPhysicalPoint(
                    (int(self.ix), int(self.iy), int(self.iz))
                )
                xL, yL, zL = float(lps[0]), float(lps[1]), float(lps[2])
                footer = f"LPS (mm): X={xL:.1f}  Y={yL:.1f}  Z={zL:.1f}"
                pm_ax = _draw_lps_footer(pm_ax, footer)
                pm_cor = _draw_lps_footer(pm_cor, footer)
                pm_sag = _draw_lps_footer(pm_sag, footer)
            except Exception:
                pass
        try:
            self._update_live_contact_dialog_from_crosshair()
        except Exception:
            pass

        self.lbl_axi.setText("")
        self.lbl_cor.setText("")
        self.lbl_sag.setText("")

        self.lbl_axi.setPixmap(pm_ax)
        self.lbl_cor.setPixmap(pm_cor)
        self.lbl_sag.setPixmap(pm_sag)

    def clear_electrode_parameters(self) -> None:
        """
        Clear the Electrode parameters panel when the selected/edited electrode
        no longer exists, for example after deleting it.
        """
        self._editing_elec_id = None
        self._active_electrode = None
        self._editing_contact_target = None
        self._accepted_duplicate_name = None

        self._pick_mode = None
        self._deep_picked = None
        self._second_picked = None
        self._deep_idx = None
        self._second_idx = None
        self._deep_lps = None
        self._second_lps = None

        for le in (
            self.edit_elec_name,
            self.edit_deep_x,
            self.edit_deep_y,
            self.edit_deep_z,
            self.edit_second_x,
            self.edit_second_y,
            self.edit_second_z,
            self.edit_nb_contacts,
            self.edit_interdist,
        ):
            try:
                if le is not None:
                    le.clear()
            except Exception:
                pass

        try:
            if self.combo_ref is not None:
                self.combo_ref.blockSignals(True)
                self.combo_ref.setCurrentIndex(0)
                self.combo_ref.blockSignals(False)
        except Exception:
            pass

        try:
            if self._hemi_group is not None:
                self._hemi_group.setExclusive(False)

            for cb in (self.chk_hemi_left, self.chk_hemi_right):
                if cb is not None:
                    cb.blockSignals(True)
                    cb.setChecked(False)
                    cb.blockSignals(False)

            if self._hemi_group is not None:
                self._hemi_group.setExclusive(True)
        except Exception:
            pass

        try:
            if self.edit_nb_contacts is not None:
                self.edit_nb_contacts.setEnabled(True)
                self.edit_nb_contacts.setReadOnly(True)
                self.edit_nb_contacts.setPlaceholderText("Number of contacts")

            if self.edit_interdist is not None:
                self.edit_interdist.setEnabled(True)
                self.edit_interdist.setReadOnly(True)
                self.edit_interdist.setPlaceholderText("Distance in mm")
        except Exception:
            pass

        try:
            if getattr(self, "chk_snapping", None) is not None:
                self.chk_snapping.setChecked(True)

            if getattr(self, "chk_localmax", None) is not None:
                self.chk_localmax.setChecked(False)
        except Exception:
            pass

        self._crosshair_locked = False
        self._clear_pick_button_highlight()
        self._update_estimate_enabled()

    def load_electrode_parameters(self, elec_id: int) -> None:
        """Populate 'Electrodes parameters' UI from a selected electrode."""
        if elec_id < 0 or elec_id >= len(self.state.electrodes):
            self.clear_electrode_parameters()
            return
        self._editing_elec_id = int(elec_id)
        elec = self.state.electrodes[elec_id]
        try:
            deep_idx = elec.get("deep_idx", None)
            second_idx = elec.get("second_idx", None)

            self._deep_idx = tuple(map(int, deep_idx)) if deep_idx is not None else None
            self._second_idx = tuple(map(int, second_idx)) if second_idx is not None else None

            self._deep_picked = self._deep_idx
            self._second_picked = self._second_idx
        except Exception:
            self._deep_idx = None
            self._second_idx = None
            self._deep_picked = None
            self._second_picked = None

        try:
            deep_lps = elec.get("deepest_lps", None)
            second_lps = elec.get("second_lps", None)

            self._deep_lps = tuple(deep_lps) if deep_lps is not None else None
            self._second_lps = tuple(second_lps) if second_lps is not None else None
        except Exception:
            self._deep_lps = None
            self._second_lps = None
        try:
            self.ui.editReco_nameElec.setText(str(elec.get("name", "")))
        except Exception:
            pass

        # hemisphere (checkboxes)
        hemi = str(elec.get("hemisphere", "")).lower()
        try:
            self.ui.chkReco_hemiLeft.blockSignals(True)
            self.ui.chkReco_hemiRight.blockSignals(True)
            self.ui.chkReco_hemiLeft.setChecked(hemi.startswith("l"))
            self.ui.chkReco_hemiRight.setChecked(hemi.startswith("r"))
        except Exception:
            pass
        finally:
            try:
                self.ui.chkReco_hemiLeft.blockSignals(False)
                self.ui.chkReco_hemiRight.blockSignals(False)
            except Exception:
                pass

        # ref combobox
        ref = elec.get("ref", "")
        try:
            idx = self.ui.comboReco_electrodeRef.findText(ref)
            if idx >= 0:
                self.ui.comboReco_electrodeRef.setCurrentIndex(idx)
        except Exception:
            pass
        try:
            is_other = self._is_other_ref(ref)

            if self.edit_nb_contacts is not None:
                self.edit_nb_contacts.setEnabled(True)
                self.edit_nb_contacts.setReadOnly(not is_other)

            if self.edit_interdist is not None:
                self.edit_interdist.setEnabled(True)
                self.edit_interdist.setReadOnly(not is_other)

        except Exception:
            pass
        # deepest/second LPS fields
        deepest_lps = elec.get("deepest_lps")
        second_lps = elec.get("second_lps")
        if deepest_lps is not None:
            try:
                self.ui.editReco_deepestX.setText(f"{float(deepest_lps[0]):.2f}")
                self.ui.editReco_deepestY.setText(f"{float(deepest_lps[1]):.2f}")
                self.ui.editReco_deepestZ.setText(f"{float(deepest_lps[2]):.2f}")
            except Exception:
                pass
        if second_lps is not None:
            try:
                self.ui.editReco_secondtX.setText(f"{float(second_lps[0]):.2f}")
                self.ui.editReco_secondtY.setText(f"{float(second_lps[1]):.2f}")
                self.ui.editReco_secondtZ.setText(f"{float(second_lps[2]):.2f}")
            except Exception:
                pass

        # enable/disable estimate button based on validation
        if hasattr(self, "_update_estimate_enabled"):
            try:
                self._update_estimate_enabled()
            except Exception:
                pass

    def delete_electrodes(self, elec_ids: list[int]) -> None:
        """
        Delete one or several electrodes from the shared state.

        Deletion is performed from the highest index to the lowest index so
        remaining indices do not shift during the operation.
        """
        valid_ids = sorted(
            {int(elec_id) for elec_id in elec_ids if 0 <= int(elec_id) < len(self._electrodes)},
            reverse=True,
        )

        if not valid_ids:
            return

        deleted_ids = set(valid_ids)

        was_editing_deleted = (
            self._editing_elec_id is not None and int(self._editing_elec_id) in deleted_ids
        )

        selected_id = getattr(self.state, "selected_electrode_id", None)
        was_selected_deleted = isinstance(selected_id, int) and int(selected_id) in deleted_ids

        for elec_id in valid_ids:
            del self._electrodes[elec_id]

        self._editing_elec_id = None
        self._editing_contact_target = None

        try:
            if was_selected_deleted or was_editing_deleted:
                self.state.selected_electrode_id = None
                self.state.selected_contact_index = None
        except Exception:
            pass

        self.clear_electrode_parameters()

        try:
            if self._contact_edit_dialog is not None:
                self._contact_edit_dialog.close()
        except Exception:
            pass

        self._contact_edit_dialog = None
        self._live_contact_pick_enabled = False

        self._update_electrodes_ui()

        try:
            if hasattr(self.state, "notify_electrodes_changed"):
                self.state.notify_electrodes_changed()
        except Exception:
            pass

        self.render_all()

    def delete_electrode(self, elec_id: int) -> None:
        """
        Backward-compatible single-electrode deletion.
        """
        self.delete_electrodes([int(elec_id)])

    def delete_contacts(self, contacts: list[tuple[int, int]]) -> None:
        """
        Delete one or several contacts, possibly belonging to different electrodes.

        Contacts are deleted from the highest index to the lowest index inside
        each electrode so the remaining contact indices stay valid.
        """
        grouped: dict[int, set[int]] = {}

        for elec_id, contact_idx in contacts:
            try:
                elec_id = int(elec_id)
                contact_idx = int(contact_idx)
            except Exception:
                continue

            if not (0 <= elec_id < len(self._electrodes)):
                continue

            contacts_lps = self._electrodes[elec_id].get("contacts_lps", []) or []

            if 0 <= contact_idx < len(contacts_lps):
                grouped.setdefault(elec_id, set()).add(contact_idx)

        if not grouped:
            return

        for elec_id, contact_indices in grouped.items():
            elec = self._electrodes[elec_id]

            for contact_idx in sorted(contact_indices, reverse=True):
                for key in (
                    "contacts_idx",
                    "contacts_lps",
                    "contacts_visible",
                    "contact_labels_visible",
                    "contact_names",
                ):
                    arr = elec.get(key)
                    if isinstance(arr, list) and 0 <= contact_idx < len(arr):
                        arr.pop(contact_idx)

            n_contacts = len(elec.get("contacts_lps", []) or [])
            elec["n"] = n_contacts
            elec["n_contacts"] = n_contacts

        self._editing_contact_target = None
        try:
            current_id = getattr(self, "_editing_elec_id", None)

            if current_id is not None and 0 <= int(current_id) < len(self._electrodes):
                self.load_electrode_parameters(int(current_id))
            else:
                self.clear_electrode_parameters()
        except Exception:
            pass

        try:
            if self._contact_edit_dialog is not None:
                self._contact_edit_dialog.close()
        except Exception:
            pass

        self._contact_edit_dialog = None
        self._live_contact_pick_enabled = False

        self._update_electrodes_ui()

        try:
            if hasattr(self.state, "notify_electrodes_changed"):
                self.state.notify_electrodes_changed()
        except Exception:
            pass

        self.render_all()

    def delete_contact(self, elec_id: int, contact_idx: int) -> None:
        """
        Backward-compatible single-contact deletion.
        """
        self.delete_contacts([(int(elec_id), int(contact_idx))])

    def open_edit_contact_dialog(self, elec_id: int, contact_idx: int) -> None:
        if elec_id < 0 or elec_id >= len(self._electrodes):
            return

        elec = self._electrodes[elec_id]
        contacts_lps = elec.get("contacts_lps", []) or []
        if not (0 <= contact_idx < len(contacts_lps)):
            return

        try:
            if self._contact_edit_dialog is not None:
                self._contact_edit_dialog.close()
        except Exception:
            pass

        parent_win = self.ui.window() if self.ui is not None else None
        dlg = EditContactDialog(parent=parent_win)
        dlg.set_lps(tuple(contacts_lps[contact_idx]))

        self._editing_contact_target = (int(elec_id), int(contact_idx))
        self._contact_edit_dialog = dlg
        self._live_contact_pick_enabled = False

        dlg.btnPick.clicked.connect(self._toggle_live_contact_pick)
        dlg.btnOk.clicked.connect(self._validate_contact_edit_from_dialog)

        dlg.finished.connect(lambda _: self._on_contact_dialog_closed())
        dlg.setModal(False)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _start_contact_pick(self, elec_id: int, contact_idx: int, dlg: EditContactDialog) -> None:
        self._editing_contact_target = (int(elec_id), int(contact_idx))
        self._contact_edit_dialog = dlg
        self._pick_mode = "edit_contact"
        self.render_all()

    def _apply_contact_lps_edit(self, elec_id: int, contact_idx: int, lps_xyz) -> None:
        if elec_id < 0 or elec_id >= len(self._electrodes):
            return

        img = self._ct_ref_for_lps()
        elec = self._electrodes[elec_id]

        try:
            idx = img.TransformPhysicalPointToIndex(
                (float(lps_xyz[0]), float(lps_xyz[1]), float(lps_xyz[2]))
            )
            idx = (int(idx[0]), int(idx[1]), int(idx[2]))
        except Exception:
            return

        contacts_idx = elec.get("contacts_idx", []) or []
        contacts_lps = elec.get("contacts_lps", []) or []

        if not (0 <= contact_idx < len(contacts_idx)):
            return

        contacts_idx[contact_idx] = idx
        contacts_lps[contact_idx] = (
            float(lps_xyz[0]),
            float(lps_xyz[1]),
            float(lps_xyz[2]),
        )

        self._pick_mode = None
        self._live_contact_pick_enabled = False

        self.jump_to_voxel(idx[0], idx[1], idx[2])

        self._update_electrodes_ui()

        try:
            if hasattr(self.state, "notify_electrodes_changed"):
                self.state.notify_electrodes_changed()
        except Exception:
            pass

        # Refresh reconstruction immediately
        self.render_all()

        try:
            for lbl in (self.lbl_cor, self.lbl_sag, self.lbl_axi):
                if lbl is not None:
                    lbl.update()
                    lbl.repaint()

            from PySide6.QtWidgets import QApplication

            QApplication.processEvents()
        except Exception:
            pass

        # Refresh 3D view immediately
        try:
            vp = getattr(self.state, "view3d_page", None)
            if vp is not None:
                if hasattr(vp, "update_electrodes"):
                    vp.update_electrodes()
                if hasattr(vp, "render_all_surface_projections"):
                    vp.render_all_surface_projections()
                if hasattr(vp, "_refresh_multiplanar_clipped_scene"):
                    vp._refresh_multiplanar_clipped_scene()
        except Exception:
            pass

        # Refresh oblique slice immediately
        try:
            op = getattr(self.state, "oblique_page", None)
            if op is not None:
                if hasattr(op, "_schedule_refresh"):
                    op._schedule_refresh(slices=True, brain=True)
                elif hasattr(op, "render_all"):
                    op.render_all()
        except Exception:
            pass

    def _update_orientation_buttons(self) -> None:
        """
        Update Radiologic / Research visual state.

        - Selected button: grey background with permanent pink border.
        - Unselected button: dark grey background and neutral border.
        - Hover on unselected button: temporary pink border only.
        """
        active_style = """
            QAbstractButton {
                background-color: rgba(80, 80, 80, 220);
                color: white;
                border: 1px solid #FF487D;
                border-radius: 9px;
            }

            QAbstractButton:hover {
                background-color: rgba(80, 80, 80, 220);
                color: white;
                border: 1px solid #FF487D;
            }

            QAbstractButton:pressed {
                background-color: rgba(70, 70, 70, 230);
                color: white;
                border: 1px solid #FF487D;
            }
        """

        inactive_style = """
            QAbstractButton {
                background-color: rgba(80, 80, 80, 80);
                color: white;
                border: 1px solid #2B2D38;
                border-radius: 9px;
            }

            QAbstractButton:hover {
                background-color: rgba(80, 80, 80, 80);
                color: white;
                border: 1px solid #FF487D;
            }

            QAbstractButton:pressed {
                background-color: rgba(80, 80, 80, 110);
                color: white;
                border: 1px solid #FF487D;
            }
        """

        research_active = self._orientation_mode == "research"
        radiologic_active = self._orientation_mode == "radiologic"

        if self.btn_research_view is not None:
            self.btn_research_view.setStyleSheet(
                active_style if research_active else inactive_style
            )

        if self.btn_radiologic_view is not None:
            self.btn_radiologic_view.setStyleSheet(
                active_style if radiologic_active else inactive_style
            )

    def set_orientation_mode(self, mode: str) -> None:
        """Set display orientation mode and keep click mapping consistent.

        - research (neurologic): L on left (flip LR ON)
        - radiologic: R on left (flip LR OFF)
        """
        mode = (mode or "").lower().strip()
        if mode not in ("research", "radiologic"):
            mode = "research"

        self._orientation_mode = mode
        flip = mode == "research"

        # Radiologic/Research changes left-right convention for axial and coronal.
        # The sagittal view should remain visually stable across both modes; otherwise
        # the modes appear swapped in sagittal and the click mapping becomes confusing.
        self.FLIP_LR_AXIAL = flip
        self.FLIP_LR_CORONAL = flip
        self.FLIP_LR_SAGITTAL = True

        self._update_orientation_buttons()

        try:
            self.render_all()
        except Exception:
            pass

    def _enable_live_contact_pick(self) -> None:
        """Enable live crosshair picking for the contact edit dialog."""
        if self._editing_contact_target is None or self._contact_edit_dialog is None:
            return

        self._pick_mode = "edit_contact_live"

        # Immediately push current crosshair coordinates into the popup
        try:
            img = self._ct_ref_for_lps()
            lps = img.TransformIndexToPhysicalPoint((int(self.ix), int(self.iy), int(self.iz)))
            self._contact_edit_dialog.set_lps((float(lps[0]), float(lps[1]), float(lps[2])))
        except Exception:
            pass

        self.render_all()

    def _toggle_live_contact_pick(self) -> None:
        if self._contact_edit_dialog is None or self._editing_contact_target is None:
            return

        self._live_contact_pick_enabled = not self._live_contact_pick_enabled

        try:
            if self._live_contact_pick_enabled:
                self._contact_edit_dialog.btnPick.setText("Picking...")
            else:
                self._contact_edit_dialog.btnPick.setText("Pick with crosshair")
        except Exception:
            pass

        # Push current crosshair immediately into the popup
        self._update_live_contact_dialog_from_crosshair()
        self.render_all()

    def _validate_contact_edit_from_dialog(self) -> None:
        if self._editing_contact_target is None or self._contact_edit_dialog is None:
            return

        elec_id, contact_idx = self._editing_contact_target

        try:
            new_lps = self._contact_edit_dialog.get_lps()
        except Exception:
            return

        self._apply_contact_lps_edit(elec_id, contact_idx, new_lps)

        # Keep the dialog open after validation.
        # Just stop live picking mode and refresh the displayed values.
        self._live_contact_pick_enabled = False

        try:
            self._contact_edit_dialog.btnPick.setText("Pick with crosshair")
        except Exception:
            pass

        # Refill dialog with the exact saved values
        try:
            elec = self._electrodes[int(elec_id)]
            contacts_lps = elec.get("contacts_lps", []) or []
            if 0 <= int(contact_idx) < len(contacts_lps):
                self._contact_edit_dialog.set_lps(tuple(contacts_lps[int(contact_idx)]))
        except Exception:
            pass

    def _cancel_contact_edit_dialog(self) -> None:
        dlg = self._contact_edit_dialog

        self._editing_contact_target = None
        self._contact_edit_dialog = None
        self._live_contact_pick_enabled = False

        try:
            if dlg is not None:
                dlg.close()
        except Exception:
            pass

        self.render_all()

    def _update_live_contact_dialog_from_crosshair(self) -> None:
        if not self._live_contact_pick_enabled:
            return
        if self._contact_edit_dialog is None:
            return

        try:
            img = self._ct_ref_for_lps()
            lps = img.TransformIndexToPhysicalPoint((int(self.ix), int(self.iy), int(self.iz)))
            self._contact_edit_dialog.set_lps((float(lps[0]), float(lps[1]), float(lps[2])))
        except Exception:
            pass

    def force_refresh_after_coreg_validation(self):
        try:
            self.init_from_volume()

            for lbl in (self.lbl_cor, self.lbl_sag, self.lbl_axi):
                if lbl is not None:
                    lbl.setText("")
                    lbl.update()
                    lbl.repaint()
        except Exception:
            pass

    def _on_contact_dialog_closed(self) -> None:
        self._editing_contact_target = None
        self._contact_edit_dialog = None
        self._live_contact_pick_enabled = False
        try:
            self.render_all()
        except Exception:
            pass

    def _set_pick_button_active(self, mode: str | None) -> None:
        active_style = """
            QAbstractButton {
                background-color: rgba(255, 0, 0, 90);
                border: 1px solid rgba(255, 0, 0, 160);
            }
        """
        normal_style = ""

        try:
            if self.btn_pick_deep is not None:
                self.btn_pick_deep.setStyleSheet(
                    active_style if mode == "deepest" else normal_style
                )
        except Exception:
            pass

        try:
            if self.btn_pick_second is not None:
                self.btn_pick_second.setStyleSheet(
                    active_style if mode == "second" else normal_style
                )
        except Exception:
            pass

    def _clear_pick_button_highlight(self) -> None:
        self._set_pick_button_active(None)
