from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import vtk
from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import map_coordinates

from neuxelec.ui.brain_render_dialog import BrainRenderDialog
from neuxelec.ui.context_menus import (
    exec_3d_view_menu,
    exec_mni_electrode_tree_menu,
)
from neuxelec.ui.export_coordinates_dialog import ExportCoordinatesDialog
from neuxelec.ui.mni_parcellation_table_dialog import MniParcellationTableDialog
from neuxelec.ui.neuxelec_color_dialog import NeuXelecColorDialog
from neuxelec.ui.neuxelec_message_dialog import (
    NeuXelecMessageDialog,
    NeuXelecSelectionDialog,
)
from neuxelec.ui.page_loading_overlay import PageLoadingOverlay
from neuxelec.ui.pyvista_quick_tools import PyVistaQuickTools
from neuxelec.utils.mni_electrodes_io import load_bids_mni_electrodes_tsv

from .view3d_camera import View3DCameraMixin
from .view3d_export import View3DExportMixin
from .view3d_fullscreen import View3DFullscreenMixin
from .view3d_markers import View3DMarkersMixin
from .view3d_mni import View3DMniMixin
from .view3d_slice_planes import View3DSlicePlanesMixin

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    from skimage.measure import marching_cubes

    _PV_OK = True
except Exception:
    _PV_OK = False

from neuxelec.utils.pet_visualization import (
    blend_pet_on_rgba,
    get_pet_window,
    normalize_pet_slice,
    pet_norm_to_colormap,
)
from neuxelec.utils.siscom_visualization import (
    blend_siscom_on_rgba,
    get_siscom_window,
    normalize_siscom_slice,
    siscom_norm_to_colormap,
)


def _lps_to_ras_points(pts: np.ndarray) -> np.ndarray:
    """Convert Nx3 LPS coordinates to RAS for VTK/PyVista display."""
    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return arr
    out = arr.copy()
    out[:, 0] *= -1.0
    out[:, 1] *= -1.0
    return out


def _top_level_window():
    aw = QApplication.activeWindow()
    if aw is not None and aw.isWindow():
        return aw

    for w in QApplication.topLevelWidgets():
        try:
            if w is not None and w.isWindow() and w.isVisible():
                return w
        except Exception:
            pass

    return None


class _CrosshairMarkerDragFilter(QObject):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def eventFilter(self, obj, event):
        # ---------------------------------------------------------
        # Fullscreen mode.
        # In fullscreen, keep PyVista camera navigation only.
        # Do not run NeuXelec custom interactions:
        # - slice Ctrl + wheel
        # - Ctrl+C marker drag
        # - hover tooltip picking
        # - parcellation picking
        # ---------------------------------------------------------
        try:
            if bool(getattr(self.owner, "_view3d_is_fullscreen", False)):
                if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape:
                    self.owner._exit_3d_fullscreen()
                    return True

                return False
        except Exception:
            pass

        # ---------------------------------------------------------
        # Keyboard + wheel controls for coronal / axial / sagittal
        # slice sliders in the 3D View.
        # ---------------------------------------------------------
        try:
            handled = self.owner._handle_slice_slider_keyboard_wheel_event(event)
            if handled:
                return True
        except Exception:
            pass

        # ---------------------------------------------------------
        # Existing Ctrl+C crosshair marker dragging behavior.
        # ---------------------------------------------------------
        try:
            handled = self.owner._handle_crosshair_marker_drag_event(event)
            if handled:
                return True
        except Exception:
            pass

        # ---------------------------------------------------------
        # Anatomical marker hover tooltip.
        # When a marker is hovered, do not show the parcellation tooltip
        # underneath it.
        # ---------------------------------------------------------
        marker_hovered = False

        try:
            marker_hovered = self.owner._handle_anatomical_marker_hover_event(
                obj,
                event,
            )
        except Exception:
            marker_hovered = False

        # ---------------------------------------------------------
        # Existing parcellation hover behavior.
        # ---------------------------------------------------------
        if not marker_hovered:
            try:
                self.owner._handle_parcellation_hover_event(obj, event)
            except Exception:
                pass

        return False


class _MniTreeBulkCheckFilter(QObject):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def eventFilter(self, obj, event):
        try:
            return self.owner._handle_mni_tree_bulk_check_event(obj, event)
        except Exception:
            return False


class View3DPage(
    View3DExportMixin,
    View3DFullscreenMixin,
    View3DMarkersMixin,
    View3DMniMixin,
    View3DSlicePlanesMixin,
    View3DCameraMixin,
):
    def __init__(self, ui_root: QObject, state: object):
        self.ui = ui_root
        self.state = state

        self.container_3d: QWidget | None = self.ui.findChild(
            QWidget,
            "frame_17",
        )
        self.container_controls: QFrame | None = self.ui.findChild(
            QFrame,
            "to_complete",
        )

        # Overlay plein écran de la page 3D.
        self._page_widget: QWidget | None = self.ui.findChild(
            QWidget,
            "page3DView",
        )

        self._loading_overlay = (
            PageLoadingOverlay(
                self._page_widget,
                "3D VIEW",
                "Preparing 3D scene",
            )
            if self._page_widget is not None
            else None
        )
        # ============================================================
        # Asynchronous 3D GIF export
        # ============================================================
        self._gif_export_active = False
        self._gif_export_state = None

        self._gif_export_timer = QTimer()
        self._gif_export_timer.setSingleShot(True)
        self._gif_export_timer.timeout.connect(self._export_next_3d_gif_frame)
        self._layout_3d: QVBoxLayout | None = None
        self._fallback_label: QLabel | None = None
        self.lbl_planes_info: QLabel | None = None

        self.interactor: QtInteractor | None = None
        self.plotter = None
        self._quick_tools = None
        self._quick_tools_resize_hook_installed = False
        self._quick_tools_original_resize_event = None
        self._quick_tools_original_show_event = None
        self._saved_camera_applied_once = False
        # Fullscreen button for the 3D viewport.
        # Important: frame_17 is never removed from the main UI layout.
        # Only the PyVista interactor is temporarily moved into a fullscreen host.
        self.btn_3d_fullscreen: QToolButton | None = None
        self._view3d_is_fullscreen = False
        self._view3d_fullscreen_host: QWidget | None = None
        self._view3d_fullscreen_layout: QVBoxLayout | None = None
        self._view3d_escape_shortcut = None
        self._view3d_host_escape_shortcut = None
        self._view3d_normal_layout_index = 0
        self._fullscreen_button_resize_filter = None
        # Data
        self._t1_img: sitk.Image | None = None
        self._t2_img: sitk.Image | None = None

        # Anatomical image used as texture on coronal/axial/sagittal planes.
        # T1 remains the geometry reference, T2 is only a display source.
        self._active_mri_source_3d = "T1"

        self._brainmask_img: sitk.Image | None = None
        self._iso_mesh_data: dict | None = None  # {"points","faces"}
        self._pet_img: sitk.Image | None = None
        self._siscom_img: sitk.Image | None = None
        self._parcel1_img: sitk.Image | None = None
        self._parcel2_img: sitk.Image | None = None
        self._parcel1_lut = {}
        self._parcel2_lut = {}

        # Color
        self._ct_color = (1.0, 1.0, 1.0)  # blanc
        self._pet_color = (1.0, 0.55, 0.0)  # orange
        self._pet_colormap_name = "jet"
        self._siscom_colormap_name = "hot"
        self.sld_pet_min: QSlider | None = None
        self.sb_pet_min: QSpinBox | None = None
        self.sld_pet_max: QSlider | None = None
        self.sb_pet_max: QSpinBox | None = None
        self.sld_pet_gamma: QSlider | None = None
        self.dsb_pet_gamma = None
        self._siscom_color = (1.0, 0.0, 0.0)  # rouge

        self._brain_color = (0.83, 0.83, 0.83)  # light gray
        self.sld_3d_brainMaskOpacity: QSlider | None = None
        self.sld_3d_PialOpacity: QSlider | None = None

        # Actors
        self._brain_actor = None
        self._ct_actor = None
        self._pet_actor = None
        self._siscom_actor = None
        self._coronal_plane_actor = None
        self._pet_scalar_bar_actor = None
        self._siscom_scalar_bar_actor = None
        self._siscom_fixed_zmax = None
        self._show_color_scales = True

        # -------------------------
        # MNI atlas mode
        # -------------------------
        self.chk_mni_atlas = None
        self._mni_atlas_actor = None
        self._mni_electrode_actors = {}
        self._mni_label_actors = {}
        self._mni_tree_connected = False
        self._mni_native_checkboxes_state = {}
        self._mni_tree_context_connected = False
        self._mni_atlas_signal_connected = False
        self._mni_tree_updating = False
        self._mni_tree_bulk_filter = None
        self._mni_tree_bulk_active = False
        self._mni_tree_bulk_target_checked = True
        self._mni_tree_bulk_pending_groups = set()
        self._mni_tree_bulk_pending_patients = set()
        self._mni_tree_bulk_last_item_key = None
        self._switching_mni_to_native_brain = False
        self._mni_template_t1_img = None
        self._mni_template_mask_img = None
        self._mni_t1_slices_visible = False
        self._mni_parcel1_img = None
        self._mni_parcel2_img = None
        self._mni_parcel1_lut = {}
        self._mni_parcel2_lut = {}
        self._mni_parcellation_table_dialog = None
        # Temporary contact label created by a MNI "Show ... slice" action.
        # Format:
        # {
        #     "set_index": int,
        #     "contact_index": int,
        #     "group_name": str,
        #     "plane": str,
        # }
        self._mni_slice_focused_contact = None
        # Parcellation hover tooltip
        self._parcellation_hover_timer = QTimer()
        self._parcellation_hover_timer.setSingleShot(True)
        self._parcellation_hover_timer.timeout.connect(self._show_parcellation_hover_tooltip)
        self._parcellation_hover_qpos = None
        self._parcellation_hover_global_pos = None

        self._elec_actors = {}
        self._elec_label_actors = {}
        self._surface_projection_actors = {}
        self._surface_projection_label_actors = {}
        self._elec_tree_widgets = []

        self._crosshair_marker_actor = None
        self._crosshair_marker_ras = None
        self._crosshair_marker_drag_armed = False
        self._crosshair_marker_drag_active = False
        self._crosshair_marker_drag_filter = None

        # Anatomical markers, e.g. lesion sites.
        # Data are stored persistently in state.markers.
        self._anatomical_marker_actors = {}  # marker_id -> PyVista actor
        self._hovered_anatomical_marker_id = None
        self._marker_list_dialog = None
        self._marker_list_page_close_connected = False
        self._connect_marker_list_auto_close_on_page_change()

        # Slice slider controlled while Ctrl + one arrow key is held:
        #     "sagittal" -> Ctrl + Left arrow + mouse wheel
        #     "coronal"  -> Ctrl + Down arrow + mouse wheel
        #     "axial"    -> Ctrl + Up arrow + mouse wheel
        self._slice_wheel_active_plane: str | None = None

        # Unclipped source meshes for planes/overlays
        self._coronal_plane_source_mesh = None
        self._axial_plane_source_mesh = None
        self._sagittal_plane_source_mesh = None

        self._coronal_pet_source_mesh = None
        self._axial_pet_source_mesh = None
        self._sagittal_pet_source_mesh = None

        self._coronal_siscom_source_mesh = None
        self._axial_siscom_source_mesh = None
        self._sagittal_siscom_source_mesh = None

        self._suspend_electrode_refresh = False

        # Controls
        self.chk_brainmask: QCheckBox | None = None
        self.chk_iso: QCheckBox | None = None
        self.chk_pial: QCheckBox | None = None  # placeholder inactive
        self._lh_pial_poly = None
        self._rh_pial_poly = None
        self._lh_pial_mask_img = None
        self._rh_pial_mask_img = None

        self.sld_brain_opacity: QSlider | None = None
        self.sld_brain_smoothing: QSlider | None = None
        self.sld_brain_iso_pct: QSlider | None = None

        self._axial_pet_actor = None
        self._axial_siscom_actor = None
        self._sagittal_pet_actor = None
        self._sagittal_siscom_actor = None

        # New CT controls
        self.chk_ct: QCheckBox | None = None
        self.sld_ct_thr: QSlider | None = None
        self.sld_ct_opacity: QSlider | None = None

        self.chk_pet: QCheckBox | None = None
        self.sld_pet_thr: QSlider | None = None
        self.sld_pet_opacity: QSlider | None = None

        self.chk_siscom: QCheckBox | None = None
        self.dsb_siscom_z: QDoubleSpinBox | None = None
        self.sld_siscom_opacity: QSlider | None = None

        self.btn_reset_camera: QPushButton | None = None

        self.chk_coronal_plane: QCheckBox | None = None
        self.sld_coronal_plane: QSlider | None = None
        self.chk_axial_plane: QCheckBox | None = None
        self.sld_axial_plane: QSlider | None = None
        self.chk_sagittal_plane: QCheckBox | None = None
        self.sld_sagittal_plane: QSlider | None = None

        self.btn_3d_StartInf: QPushButton | None = None
        self.btn_3d_StartCaudal: QPushButton | None = None
        self.btn_3d_StartLH: QPushButton | None = None

        # New electrode rendering controls
        self.btn_elec_shaft: QPushButton | None = None
        self.spin_contacts_size = None

        # Optional explicit CT image
        self._ct_img: sitk.Image | None = None

        self.btn_elec_shaft: QPushButton | None = None
        self.spin_contacts_size: QDoubleSpinBox | None = None

        self._ct_render_timer = QTimer()
        self._ct_render_timer.setSingleShot(True)
        self._ct_render_timer.timeout.connect(self._render_ct)

        self._pet_render_timer = QTimer()
        self._pet_render_timer.setSingleShot(True)
        self._pet_render_timer.timeout.connect(self._render_pet)

        self._init_ui()
        self._update_threshold_labels()

        self._coronal_plane_actor = None
        self._coronal_pet_actor = None
        self._coronal_siscom_actor = None
        self._coronal_elec_actor = None
        self._axial_plane_actor = None
        self._sagittal_plane_actor = None
        self._axial_elec_actor = None
        self._sagittal_elec_actor = None

        self._axial_from_inferior = False
        self._coronal_from_caudal = False
        self._sagittal_from_left = False

        self._coronal_outline_actor = None
        self._axial_outline_actor = None
        self._sagittal_outline_actor = None
        # Show/hide only the colored frames around coronal/axial/sagittal slices.
        # This does NOT hide the slices themselves.
        self._slice_plane_frames_visible = True

        self._coronal_render_timer = QTimer()
        self._coronal_render_timer.setSingleShot(True)
        self._coronal_render_timer.timeout.connect(self._refresh_multiplanar_clipped_scene)

        self._coronal_y_min = 0
        self._coronal_y_max = None
        self._axial_z_min = 0
        self._axial_z_max = None
        self._sagittal_x_min = 0
        self._sagittal_x_max = None

        self._is_active_page = False

        self._t1_np = None  # z, y, x
        self._brainmask_t1_np = None  # z, y, x
        self._coronal_plane_mesh = None

        # ------------------------------------------------------------------
        # Cached slice-display volumes for smooth coronal/axial/sagittal navigation.
        #
        # Arrays use SimpleITK / NumPy order:
        #     [z, y, x, rgba]
        #
        # The cache is stored at native anatomical resolution.
        # It is rebuilt only when modalities or visual parameters change,
        # never when a slice slider moves.
        # ------------------------------------------------------------------
        self._slice_base_rgba_cache = None  # T1/T2 + active parcellation
        self._slice_pet_rgba_cache = None  # PET transparent overlay
        self._slice_siscom_rgba_cache = None  # SISCOM transparent overlay

        self._slice_base_cache_ready = False
        self._slice_pet_cache_ready = False
        self._slice_siscom_cache_ready = False
        # Stable crop bounding box used by all 3D anatomical slice planes.
        # Format: (x0, x1, y0, y1, z0, z1), in voxel coordinates.
        self._slice_crop_bounds_xyz = None

        self._brain_color = (0.88, 0.85, 0.80)
        self._show_lh_pial = True
        self._show_rh_pial = True
        self._pial_assume_lps = False
        self._pial_debug_shift_ras = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self._brain_render_params = {
            "ambient": 0.10,
            "diffuse": 0.80,
            "specular": 0.30,
            "specular_power": 40.0,
            "key_light": 0.90,
            "fill_light": 0.10,
            "back_light": 0.15,
            "shadows": True,
        }

        self._brain_render_dialog = None

        self._shortcut_back_to_reco = QShortcut(QKeySequence("Ctrl+F"), self.container_3d)
        self._shortcut_back_to_reco.activated.connect(self._go_back_to_reconstruction)

        # Surface projection markers for electrodes
        self._surface_projection_defs = {}  # elec_id -> True
        self._surface_projection_actors = {}  # elec_id -> {"cross": actor, "label": actor}
        # Page-specific label visibility (must NOT be shared with other pages)
        self._page_contact_labels_visible = {}  # elec_id -> [bool, bool, ...]
        self._page_electrode_visible = {}  # elec_id -> bool
        self._page_contacts_visible = {}  # elec_id -> [bool, bool, ...]
        # Contact label temporarily displayed by "Show coronal/axial/sagittal slice".
        # Format: {"plane": str, "elec_id": int, "contact_idx": int}
        self._slice_focused_contact_label = None
        # If True, coronal/axial/sagittal slices do not hide native electrodes.
        # Electrodes remain visible with the current display mode:
        # contacts only or contacts + shaft.
        self._keep_electrodes_visible_through_slices = False

    def _get_local_electrode_visible(self, elec_id: int) -> bool:
        return bool(self._page_electrode_visible.get(int(elec_id), True))

    def _get_local_contacts_visible(self, elec_id: int, n_contacts: int):
        vals = self._page_contacts_visible.get(int(elec_id))
        if not isinstance(vals, list) or len(vals) != int(n_contacts):
            vals = [True] * int(n_contacts)
            self._page_contacts_visible[int(elec_id)] = vals
        return vals

    def _set_single_electrode_actor_visibility(self, elec_id: int, visible: bool) -> None:
        """
        Show/hide only one electrode in 3D without rebuilding all electrodes.
        """
        elec_id = int(elec_id)
        visible = bool(visible)

        # Main 3D electrode actors: points + shaft
        for key in (
            (elec_id, "points"),
            (elec_id, "line"),
        ):
            actor = getattr(self, "_elec_actors", {}).get(key)
            if actor is not None:
                try:
                    actor.SetVisibility(visible)
                except Exception:
                    pass

        # Contact labels for this electrode
        label_actor = getattr(self, "_elec_label_actors", {}).get((elec_id, "labels"))
        if label_actor is not None:
            try:
                label_actor.SetVisibility(visible)
            except Exception:
                pass

        # Surface projection actors for this electrode, if present
        proj = getattr(self, "_surface_projection_actors", {}).get(elec_id)
        if isinstance(proj, dict):
            for actor in proj.values():
                try:
                    if actor is not None:
                        actor.SetVisibility(visible)
                except Exception:
                    pass

        # If your projection labels are stored separately
        proj_label = getattr(self, "_surface_projection_label_actors", {}).get(elec_id)
        if proj_label is not None:
            try:
                proj_label.SetVisibility(visible)
            except Exception:
                pass

    def _remove_single_electrode_actors(self, elec_id: int) -> None:
        elec_id = int(elec_id)

        for key in (
            (elec_id, "points"),
            (elec_id, "line"),
        ):
            actor = getattr(self, "_elec_actors", {}).pop(key, None)
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor, reset_camera=False)
                except Exception:
                    pass

        label_actor = getattr(self, "_elec_label_actors", {}).pop((elec_id, "labels"), None)
        if label_actor is not None:
            try:
                self.plotter.remove_actor(label_actor, reset_camera=False)
            except Exception:
                pass

    def _refresh_visible_slice_electrode_overlays(self) -> None:
        """
        Refresh only the electrode overlays drawn on visible slice planes.
        This avoids rebuilding all 3D electrode sphere/shaft actors.
        """
        if bool(getattr(self, "_keep_electrodes_visible_through_slices", False)):
            try:
                self._remove_actor("coronal_elec")
                self._remove_actor("axial_elec")
                self._remove_actor("sagittal_elec")
            except Exception:
                pass
            return

        try:
            if self.chk_coronal_plane is not None and self.chk_coronal_plane.isChecked():
                self._render_coronal_electrodes_overlay()
        except Exception:
            pass

        try:
            if self.chk_axial_plane is not None and self.chk_axial_plane.isChecked():
                self._render_axial_electrodes_overlay()
        except Exception:
            pass

        try:
            if self.chk_sagittal_plane is not None and self.chk_sagittal_plane.isChecked():
                self._render_sagittal_electrodes_overlay()
        except Exception:
            pass

    def update_single_electrode_color_only(self, elec_id: int, render: bool = True) -> None:
        """
        Update only one native patient electrode color in the 3D scene.

        This does NOT call update_electrodes(), so other electrodes are not removed
        and do not flicker.
        """
        if not _PV_OK or self.plotter is None:
            return

        try:
            elec_id = int(elec_id)
            elec = self.state.electrodes[elec_id]
            rgb = tuple(elec.get("color", (0, 255, 0)))
            color = (
                float(rgb[0]) / 255.0,
                float(rgb[1]) / 255.0,
                float(rgb[2]) / 255.0,
            )
        except Exception:
            return

        # Main 3D contacts + shaft.
        for key in (
            (elec_id, "points"),
            (elec_id, "line"),
        ):
            try:
                actor = getattr(self, "_elec_actors", {}).get(key)
                self._set_existing_actor_color(actor, color)
            except Exception:
                pass

        # Contact labels, if they exist.
        try:
            label_actor = getattr(self, "_elec_label_actors", {}).get((elec_id, "labels"))
            self._set_existing_actor_color(label_actor, color)
        except Exception:
            pass

        # Surface projection cross/label, if present.
        try:
            proj = getattr(self, "_surface_projection_actors", {}).get(elec_id)
            if isinstance(proj, dict):
                for actor in proj.values():
                    self._set_existing_actor_color(actor, color)
            else:
                self._set_existing_actor_color(proj, color)
        except Exception:
            pass

        try:
            proj_label = getattr(self, "_surface_projection_label_actors", {}).get(elec_id)
            self._set_existing_actor_color(proj_label, color)
        except Exception:
            pass

        if render:
            try:
                self._render()
            except Exception:
                pass

    def _render_single_electrode(self, elec_id: int) -> None:
        """
        Rebuild only one electrode actor group: points, shaft, labels.
        Used for contact-level visibility changes.
        """
        if not _PV_OK or self.plotter is None:
            return

        try:
            elec_id = int(elec_id)
            elec = self.state.electrodes[elec_id]
        except Exception:
            return

        self._remove_single_electrode_actors(elec_id)

        if not self._get_local_electrode_visible(elec_id):
            try:
                self._render()
            except Exception:
                pass
            return

        try:
            show_shaft = True
            if self.btn_elec_shaft is not None:
                show_shaft = bool(self.btn_elec_shaft.isChecked())

            point_size = 8.0
            if self.spin_contacts_size is not None:
                try:
                    point_size = float(self.spin_contacts_size.value())
                except Exception:
                    point_size = 8.0

            rgb = tuple(elec.get("color", (0, 255, 0)))
            color = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

            contacts_lps = elec.get("contacts_lps", []) or []
            contacts_idx = elec.get("contacts_idx", []) or []
            contacts_visible = self._get_local_contacts_visible(elec_id, len(contacts_lps))

            pts = []
            for ci, p in enumerate(contacts_lps):
                if not bool(contacts_visible[ci]):
                    continue

                idx_xyz = contacts_idx[ci] if ci < len(contacts_idx) else None
                if self._contact_is_on_any_visible_slice(idx_xyz, tol=0.49):
                    continue

                try:
                    pts.append([float(p[0]), float(p[1]), float(p[2])])
                except Exception:
                    continue

            if pts:
                pts_arr = _lps_to_ras_points(np.array(pts, dtype=np.float32))
                poly = pv.PolyData(pts_arr)

                actor = self.plotter.add_points(
                    poly,
                    color=color,
                    point_size=float(point_size),
                    render_points_as_spheres=True,
                )
                try:
                    actor.PickableOff()
                except Exception:
                    pass

                self._apply_electrode_actor_depth_mode(actor)
                self._elec_actors[(elec_id, "points")] = actor

                if show_shaft and pts_arr.shape[0] >= 2:
                    line = pv.lines_from_points(pts_arr, close=False)
                    line_actor = self.plotter.add_mesh(
                        line,
                        color=color,
                        line_width=3,
                    )
                    try:
                        line_actor.PickableOff()
                    except Exception:
                        pass

                    self._apply_electrode_actor_depth_mode(line_actor)
                    self._elec_actors[(elec_id, "line")] = line_actor

            # Labels
            try:
                contact_labels_visible = self._get_local_contact_labels_visible(
                    elec_id, len(contacts_lps)
                )
                label_pts = []
                label_txt = []

                for ci, p in enumerate(contacts_lps):
                    if not bool(contacts_visible[ci]):
                        continue
                    if not bool(contact_labels_visible[ci]):
                        continue

                    ras = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float32)
                    ras[0] *= -1.0
                    ras[1] *= -1.0

                    label_pos = ras.copy()
                    label_pos[0] += 2.0
                    label_pos[2] += 2.0

                    label_pts.append(label_pos)
                    label_txt.append(f"{elec.get('name', 'E')}{ci + 1}")

                if label_pts:
                    label_actor = self.plotter.add_point_labels(
                        np.asarray(label_pts, dtype=np.float32),
                        label_txt,
                        font_size=12,
                        text_color=color,
                        shape_opacity=0.0,
                        show_points=False,
                        always_visible=True,
                    )
                    try:
                        label_actor.PickableOff()
                    except Exception:
                        pass

                    self._elec_label_actors[(elec_id, "labels")] = label_actor
            except Exception:
                pass

            self._apply_actor_clipping()
            self._refresh_visible_slice_electrode_overlays()
            self._render()

        except Exception:
            pass

    def _block_export_if_fullscreen(self, export_name: str = "export") -> bool:
        if not bool(getattr(self, "_view3d_is_fullscreen", False)):
            return False

        NeuXelecMessageDialog.warning(
            self._dialog_parent(),
            "Fullscreen mode",
            (f"{export_name} is disabled in fullscreen mode.\n\n" "Please exit fullscreen first."),
        )
        return True

    def set_electrode_visible(self, elec_id: int, visible: bool) -> None:
        try:
            elec_id = int(elec_id)
            elec = self.state.electrodes[elec_id]
            n = len(elec.get("contacts_lps", []) or [])
        except Exception:
            return

        self._page_electrode_visible[elec_id] = bool(visible)

        if not visible:
            self._page_contacts_visible[elec_id] = [False] * n
        else:
            vals = self._page_contacts_visible.get(elec_id)
            if not isinstance(vals, list) or len(vals) != n:
                vals = [True] * n

            # If all contacts were hidden because the whole electrode was hidden,
            # restore them when the electrode is re-enabled.
            if not any(vals):
                vals = [True] * n

            self._page_contacts_visible[elec_id] = vals

        # If actors already exist, just show/hide this electrode.
        has_actor = (
            (elec_id, "points") in getattr(self, "_elec_actors", {})
            or (elec_id, "line") in getattr(self, "_elec_actors", {})
            or (elec_id, "labels") in getattr(self, "_elec_label_actors", {})
        )

        if has_actor:
            self._set_single_electrode_actor_visibility(elec_id, bool(visible))
        else:
            try:
                self._render_single_electrode(elec_id)
            except Exception:
                pass

        try:
            self._refresh_visible_slice_electrode_overlays()
        except Exception:
            pass

        try:
            self._render()
        except Exception:
            pass

    def set_contact_visible(self, elec_id: int, contact_idx: int, visible: bool) -> None:
        try:
            elec_id = int(elec_id)
            contact_idx = int(contact_idx)
            elec = self.state.electrodes[elec_id]
            n = len(elec.get("contacts_lps", []) or [])
        except Exception:
            return

        vals = self._get_local_contacts_visible(elec_id, n)
        if 0 <= contact_idx < len(vals):
            vals[contact_idx] = bool(visible)

        self._page_electrode_visible[elec_id] = any(vals)

        self._render_single_electrode(elec_id)
        self._refresh_visible_slice_electrode_overlays()

        try:
            self._render()
        except Exception:
            pass

    def _build_vtk_coronal_clip_plane(self):
        geom = self._build_coronal_plane_geometry()
        if geom is None:
            return None

        plane = vtk.vtkPlane()
        n = self._get_effective_plane_normal("coronal", geom)

        plane.SetOrigin(*geom["center"])
        plane.SetNormal(*n)
        return plane

    def _schedule_coronal_refresh(self) -> None:
        self._coronal_render_timer.start(60)

    def _set_axes_widget_colored(self) -> None:
        """
        Make the small X/Y/Z orientation axes match slice-plane colors:
        X = red, Y = green, Z = blue.
        """
        try:
            if self.plotter is None:
                return

            axes_actor = getattr(self.plotter, "axes_actor", None)

            if axes_actor is None:
                try:
                    axes_actor = getattr(self.plotter.renderer, "axes_actor", None)
                except Exception:
                    axes_actor = None

            if axes_actor is None:
                return

            x_col = (1.0, 0.0, 0.0)  # X = red
            y_col = (0.0, 1.0, 0.0)  # Y = green
            z_col = (0.0, 0.35, 1.0)  # Z = blue

            def _set_caption_color(caption_actor, color):
                if caption_actor is None:
                    return

                # Method 1: CaptionTextProperty
                try:
                    prop = caption_actor.GetCaptionTextProperty()
                    prop.SetColor(*color)
                    prop.SetOpacity(1.0)
                    prop.BoldOn()
                    prop.ShadowOff()
                except Exception:
                    pass

                # Method 2: internal TextActor property
                try:
                    text_actor = caption_actor.GetTextActor()
                    if text_actor is not None:
                        prop = text_actor.GetTextProperty()
                        prop.SetColor(*color)
                        prop.SetOpacity(1.0)
                        prop.BoldOn()
                        prop.ShadowOff()
                except Exception:
                    pass

            # Axis shafts and tips
            try:
                axes_actor.GetXAxisShaftProperty().SetColor(*x_col)
                axes_actor.GetXAxisTipProperty().SetColor(*x_col)

                axes_actor.GetYAxisShaftProperty().SetColor(*y_col)
                axes_actor.GetYAxisTipProperty().SetColor(*y_col)

                axes_actor.GetZAxisShaftProperty().SetColor(*z_col)
                axes_actor.GetZAxisTipProperty().SetColor(*z_col)
            except Exception:
                pass

            # X/Y/Z labels
            try:
                _set_caption_color(axes_actor.GetXAxisCaptionActor2D(), x_col)
                _set_caption_color(axes_actor.GetYAxisCaptionActor2D(), y_col)
                _set_caption_color(axes_actor.GetZAxisCaptionActor2D(), z_col)
            except Exception:
                pass

            try:
                self.plotter.render()
            except Exception:
                try:
                    self._render()
                except Exception:
                    pass

        except Exception:
            pass

    def _init_ui(self) -> None:
        if self.container_3d is None:
            return

        self._layout_3d = QVBoxLayout(self.container_3d)
        self._layout_3d.setContentsMargins(0, 0, 0, 0)

        if not _PV_OK:
            self._fallback_label = QLabel(
                "PyVista/VTK not available.\n"
                "Install: pip install pyvista pyvistaqt vtk scikit-image"
            )
            self._fallback_label.setWordWrap(True)
            self._layout_3d.addWidget(self._fallback_label)
            return

        self.interactor = QtInteractor(self.container_3d)
        self._layout_3d.addWidget(self.interactor)
        self.plotter = self.interactor
        self._create_fullscreen_button()

        try:
            self.interactor.setFocusPolicy(Qt.StrongFocus)
            self._crosshair_marker_drag_filter = _CrosshairMarkerDragFilter(self)
            self.interactor.installEventFilter(self._crosshair_marker_drag_filter)
        except Exception:
            pass

        try:
            self.interactor.setContextMenuPolicy(Qt.CustomContextMenu)
            self.interactor.customContextMenuRequested.connect(self._show_3d_context_menu)
        except Exception:
            pass

        try:
            self.interactor.mouseDoubleClickEvent = self._on_interactor_double_click
        except Exception:
            pass

        self.lbl_planes_info = QLabel(self.container_3d)
        self.lbl_planes_info.setText("")
        self.lbl_planes_info.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0, 0, 0, 120);
                border-radius: 6px;
                padding: 6px;
                font-size: 11px;
            }
        """)
        self.lbl_planes_info.hide()

        try:
            self.plotter.set_background("black")
            self.plotter.show_axes()
            self._set_axes_widget_colored()
            self.plotter.enable_trackball_style()
        except Exception:
            pass
        try:
            self._quick_tools = PyVistaQuickTools(self.container_3d, self)
            self._quick_tools.raise_()

            self._install_quick_tools_resize_handler()
            self._update_quick_tools_geometry()

        except Exception as e:
            print("[3D Quick Tools] Failed to create toolbar:", e)

        self._hook_controls()

    def _resource_image_path(self, filename: str) -> str:
        """
        Return an absolute path to a file stored in Neuxelec/resources/images.

        This avoids fragile relative paths when the app is launched from
        scripts/, PyInstaller, or another working directory.
        """
        try:
            here = Path(__file__).resolve()

            candidates = [
                here.parents[2] / "resources" / "images" / filename,
                here.parents[1] / "resources" / "images" / filename,
                Path.cwd() / "resources" / "images" / filename,
                Path.cwd() / "Neuxelec" / "resources" / "images" / filename,
            ]

            for p in candidates:
                if p.exists():
                    return str(p)

        except Exception:
            pass

        return filename

    def _active_3d_parent_widget(self) -> QWidget | None:
        """
        Return the widget that currently contains the visible 3D viewport.

        Normal mode:
            frame_17

        Fullscreen mode:
            temporary fullscreen host

        This lets floating widgets follow the actual viewport without ever
        moving frame_17 from the main UI layout.
        """
        if bool(getattr(self, "_view3d_is_fullscreen", False)):
            host = getattr(self, "_view3d_fullscreen_host", None)
            if host is not None:
                return host

        return self.container_3d

    def _install_quick_tools_resize_handler(self) -> None:
        """
        Keep the floating 3D quick toolbar visible and adapted to the 3D viewport size.
        """
        try:
            if self.container_3d is None:
                return

            if bool(getattr(self, "_quick_tools_resize_hook_installed", False)):
                return

            self._quick_tools_resize_hook_installed = True

            self._quick_tools_original_resize_event = self.container_3d.resizeEvent
            self._quick_tools_original_show_event = self.container_3d.showEvent

            def _resize_event(event):
                try:
                    if self._quick_tools_original_resize_event is not None:
                        self._quick_tools_original_resize_event(event)
                except Exception:
                    pass

                try:
                    QTimer.singleShot(0, self._update_quick_tools_geometry)
                except Exception:
                    pass
                try:
                    self._schedule_fullscreen_button_geometry_update()
                except Exception:
                    pass

            def _show_event(event):
                try:
                    if self._quick_tools_original_show_event is not None:
                        self._quick_tools_original_show_event(event)
                except Exception:
                    pass

                try:
                    QTimer.singleShot(0, self._update_quick_tools_geometry)
                except Exception:
                    pass

                try:
                    self._schedule_fullscreen_button_geometry_update()
                except Exception:
                    pass

            self.container_3d.resizeEvent = _resize_event
            self.container_3d.showEvent = _show_event

        except Exception:
            pass

    def _update_quick_tools_geometry(self) -> None:
        """
        Adapt the floating 3D toolbar to the real visible viewport width.

        Large 3D view:
            all icons are visible.

        Narrow 3D view:
            as many icons as possible are visible and the remaining tools
            are accessible by scrolling over the toolbar.
        """
        try:
            qt = getattr(self, "_quick_tools", None)
            parent = self._active_3d_parent_widget()
            if bool(getattr(self, "_view3d_is_fullscreen", False)):
                if qt is not None:
                    qt.hide()
                    qt.setEnabled(False)
                self._update_fullscreen_button_geometry()
                return
            if qt is None or parent is None:
                return
            if qt.parentWidget() is not parent:
                qt.setParent(parent)
                qt.show()
            # Do not calculate carousel slots while the 3D page is hidden.
            # A hidden stacked-widget page can temporarily report a tiny width.
            if not bool(getattr(self, "_is_active_page", False)):
                return

            parent_w = int(parent.width())
            parent_h = int(parent.height())

            # Ignore transient invalid geometry during page switching.
            if parent_w < 120 or parent_h < 80:
                QTimer.singleShot(60, self._update_quick_tools_geometry)
                return

            if not qt.isVisible():
                qt.show()

            margin = 10

            # Keep buttons readable; the carousel handles reduced width.
            if parent_w < 520 or parent_h < 420:
                button_w = 30
                button_h = 28
                spacing_px = 3
            elif parent_w < 760 or parent_h < 560:
                button_w = 32
                button_h = 30
                spacing_px = 4
            else:
                button_w = 34
                button_h = 32
                spacing_px = 5

            available_w = max(90, parent_w - (2 * margin))

            if hasattr(qt, "set_carousel_available_width"):
                qt.set_carousel_available_width(
                    available_width=available_w,
                    button_width=button_w,
                    button_height=button_h,
                    spacing_px=spacing_px,
                )

            qt.adjustSize()

            toolbar_w = min(int(qt.sizeHint().width()), available_w)
            toolbar_h = min(
                int(qt.sizeHint().height()),
                max(38, parent_h - (2 * margin)),
            )

            qt.resize(toolbar_w, toolbar_h)

            # Position bottom-left, like the toolbars in Oblique Slice.
            x = margin
            y = max(margin, parent_h - toolbar_h - margin)

            qt.move(x, y)
            qt.raise_()
            self._update_fullscreen_button_geometry()

        except Exception:
            pass

    def _schedule_render_ct(self) -> None:
        if self._ct_render_timer is not None:
            self._ct_render_timer.start(120)

    def _schedule_render_pet(self) -> None:
        if self._pet_render_timer is not None:
            self._pet_render_timer.start(120)

    def _hook_controls(self) -> None:
        # Electrode trees are managed by ElectrodesController

        self.chk_brainmask = self.ui.findChild(QCheckBox, "chk_3d_showBrainmask")
        self.chk_iso = self.ui.findChild(QCheckBox, "chk_3d_showIsoSurface")
        self.chk_pial = self.ui.findChild(QCheckBox, "chk_3d_showPialsurface")  # inactive
        self._create_mni_atlas_checkbox()
        self.chk_mri_t1 = self.ui.findChild(QCheckBox, "checkBox_3dView_T1")
        self.chk_mri_t2 = self.ui.findChild(QCheckBox, "checkBox_3dView_T2")

        self.sld_3d_brainMaskOpacity = self.ui.findChild(QSlider, "sld_3d_brainMaskOpacity")
        self.sld_3d_PialOpacity = self.ui.findChild(QSlider, "sld_3d_PialOpacity")
        self.sld_brain_iso_pct = self.ui.findChild(QSlider, "sld_3d_brainIsoPerct")

        # Opacity values are later divided by 100.0, so these sliders must
        # explicitly use a 0–100 range.
        if self.sld_3d_brainMaskOpacity is not None:
            self.sld_3d_brainMaskOpacity.setRange(0, 100)
            self.sld_3d_brainMaskOpacity.setSingleStep(1)

            # Default opacity for a new project.
            # Without this, the MNI atlas can be created with opacity = 0
            # and therefore remain completely invisible.
            if self.sld_3d_brainMaskOpacity.value() <= 0:
                self.sld_3d_brainMaskOpacity.setValue(50)

        if self.sld_3d_PialOpacity is not None:
            self.sld_3d_PialOpacity.setRange(0, 100)
            self.sld_3d_PialOpacity.setSingleStep(1)

        if self.chk_iso is not None:
            self.chk_iso.hide()
            self.chk_iso.setEnabled(False)
        if self.sld_brain_iso_pct is not None:
            self.sld_brain_iso_pct.hide()
            self.sld_brain_iso_pct.setEnabled(False)

        self.lbl_ct_thr_info = self.ui.findChild(QLabel, "lbl_3d_ctThrInfo")
        self.lbl_siscom_info = self.ui.findChild(QLabel, "lbl_3d_siscomInfo")

        self.chk_ct = self.ui.findChild(QCheckBox, "chk_3d_showCT")
        self.sld_ct_thr = self.ui.findChild(QSlider, "sld_3d_ctTreshold")
        if self.sld_ct_thr is not None:
            self.sld_ct_thr.setMinimum(500)
            self.sld_ct_thr.setMaximum(4000)
            self.sld_ct_thr.setSingleStep(50)
            self.sld_ct_thr.setPageStep(100)
            if self.sld_ct_thr.value() < 500:
                self.sld_ct_thr.setValue(2500)
        self.sld_ct_opacity = self.ui.findChild(QSlider, "sld_3d_ctOpacity")
        if self.sld_ct_opacity is None:
            self.sld_ct_opacity = self.ui.findChild(QSlider, "sld_3d_petOpacity")

        self.chk_pet = self.ui.findChild(QCheckBox, "chk_3d_showPET")

        self.sld_pet_min = self.ui.findChild(QSlider, "sld_3d_petMin")
        self.sb_pet_min = self.ui.findChild(QSpinBox, "sb_3d_petMin")

        self.sld_pet_max = self.ui.findChild(QSlider, "sld_3d_petMax")
        self.sb_pet_max = self.ui.findChild(QSpinBox, "sb_3d_petMax")

        self.sld_pet_gamma = self.ui.findChild(QSlider, "sld_3d_petGamma")
        self.dsb_pet_gamma = self.ui.findChild(QDoubleSpinBox, "dsb_3d_petGamma")

        self.sld_pet_opacity = self.ui.findChild(QSlider, "sld_3d_petOpacity")

        if self.sld_pet_min is not None:
            self.sld_pet_min.setMinimum(0)
            self.sld_pet_min.setMaximum(100)
            if self.sld_pet_min.value() < 0:
                self.sld_pet_min.setValue(30)

        if self.sb_pet_min is not None:
            self.sb_pet_min.setMinimum(0)
            self.sb_pet_min.setMaximum(100)

        if self.sld_pet_max is not None:
            self.sld_pet_max.setMinimum(0)
            self.sld_pet_max.setMaximum(100)
            if self.sld_pet_max.value() <= 0:
                self.sld_pet_max.setValue(98)

        if self.sb_pet_max is not None:
            self.sb_pet_max.setMinimum(0)
            self.sb_pet_max.setMaximum(100)

        if self.sld_pet_gamma is not None:
            self.sld_pet_gamma.setMinimum(10)
            self.sld_pet_gamma.setMaximum(300)
            if self.sld_pet_gamma.value() < 10:
                self.sld_pet_gamma.setValue(100)

        if self.dsb_pet_gamma is not None:
            self.dsb_pet_gamma.setMinimum(0.1)
            self.dsb_pet_gamma.setMaximum(3.0)
            self.dsb_pet_gamma.setSingleStep(0.1)

        self.chk_siscom = self.ui.findChild(QCheckBox, "chk_3d_showSISCOM")
        self.dsb_siscom_z = self.ui.findChild(QDoubleSpinBox, "dsb_3d_siscomZthr")
        if self.dsb_siscom_z is not None:
            self.dsb_siscom_z.setMinimum(1.5)
            self.dsb_siscom_z.setMaximum(6.0)
            self.dsb_siscom_z.setSingleStep(0.1)
            if float(self.dsb_siscom_z.value()) < 1.5:
                self.dsb_siscom_z.setValue(2.0)
        self.sld_siscom_opacity = self.ui.findChild(QSlider, "sld_3d_siscomOpacity")

        self.chk_parcel1 = self.ui.findChild(QCheckBox, "checkBox_3dView_Parcell1")
        self.chk_parcel2 = self.ui.findChild(QCheckBox, "checkBox_3dView_Parcell2")

        self.sld_parcel1_opacity = self.ui.findChild(QSlider, "horizontalSlider_3dView_Parcell1")
        self.sld_parcel2_opacity = self.ui.findChild(QSlider, "horizontalSlider_3dView_Parcell2")

        self.spn_parcel1_opacity = self.ui.findChild(QSpinBox, "spinBox_3dView_Parcell1")
        self.spn_parcel2_opacity = self.ui.findChild(QSpinBox, "spinBox_3dView_Parcell2")

        if self.chk_parcel1 is not None:
            self.chk_parcel1.toggled.connect(self._on_chk_parcel1_toggled)

        if self.chk_parcel2 is not None:
            self.chk_parcel2.toggled.connect(self._on_chk_parcel2_toggled)

        if self.sld_parcel1_opacity is not None and self.spn_parcel1_opacity is not None:
            self.sld_parcel1_opacity.setRange(0, 100)
            self.spn_parcel1_opacity.setRange(0, 100)

            self.sld_parcel1_opacity.valueChanged.connect(self.spn_parcel1_opacity.setValue)
            self.spn_parcel1_opacity.valueChanged.connect(self.sld_parcel1_opacity.setValue)

            self.spn_parcel1_opacity.valueChanged.connect(
                lambda _: self._refresh_base_slice_cache_and_scene()
            )

        if self.sld_parcel2_opacity is not None and self.spn_parcel2_opacity is not None:
            self.sld_parcel2_opacity.setRange(0, 100)
            self.spn_parcel2_opacity.setRange(0, 100)

            self.sld_parcel2_opacity.valueChanged.connect(self.spn_parcel2_opacity.setValue)
            self.spn_parcel2_opacity.valueChanged.connect(self.sld_parcel2_opacity.setValue)

            self.spn_parcel2_opacity.valueChanged.connect(
                lambda _: self._refresh_base_slice_cache_and_scene()
            )

        try:
            if self.sld_parcel1_opacity is not None:
                self.sld_parcel1_opacity.setValue(50)
            if self.spn_parcel1_opacity is not None:
                self.spn_parcel1_opacity.setValue(50)
            if self.sld_parcel2_opacity is not None:
                self.sld_parcel2_opacity.setValue(50)
            if self.spn_parcel2_opacity is not None:
                self.spn_parcel2_opacity.setValue(50)
            if self.chk_parcel1 is not None:
                self._set_checked(self.chk_parcel1, False)
            if self.chk_parcel2 is not None:
                self._set_checked(self.chk_parcel2, False)
        except Exception:
            pass

        self.btn_elec_shaft = self.ui.findChild(QPushButton, "btn_3d_elecShaft")
        self.btn_export_coordinates = self.ui.findChild(QPushButton, "Export_Coordinates_3")
        if self.btn_export_coordinates is not None:
            self.btn_export_coordinates.clicked.connect(self._open_export_coordinates_dialog)
        self.spin_contacts_size = self.ui.findChild(QDoubleSpinBox, "spinBox_3d_sizeContacts")
        if self.spin_contacts_size is None:
            self.spin_contacts_size = self.ui.findChild(QSpinBox, "spinBox_3d_sizeContacts")

        self.chk_coronal_plane = self.ui.findChild(QCheckBox, "chk_3d_showCoronalPlane")
        self.sld_coronal_plane = self.ui.findChild(QSlider, "sld_3d_coronalPlane")
        self.chk_axial_plane = self.ui.findChild(QCheckBox, "chk_3d_showAxialPlane")
        self.sld_axial_plane = self.ui.findChild(QSlider, "sld_3d_axialPlane")
        self.chk_sagittal_plane = self.ui.findChild(QCheckBox, "chk_3d_showSagittalPlane")
        self.sld_sagittal_plane = self.ui.findChild(QSlider, "sld_3d_sagittalPlane")

        self.btn_3d_StartInf = self.ui.findChild(QPushButton, "btn_3d_StartInf")
        self.btn_3d_StartCaudal = self.ui.findChild(QPushButton, "btn_3d_StartCaudal")
        self.btn_3d_StartLH = self.ui.findChild(QPushButton, "btn_3d_StartLH")

        if self.chk_brainmask is not None:
            self.chk_brainmask.toggled.connect(lambda v: self._on_brain_source("brainmask", v))
        if self.chk_iso is not None:
            self.chk_iso.toggled.connect(lambda v: self._on_brain_source("iso", v))
        if self.chk_pial is not None:
            self.chk_pial.toggled.connect(lambda v: self._on_brain_source("pial", v))

        # --- T1/T2 anatomical source selection for 3D slice planes ---
        try:
            if self.chk_mri_t1 is not None:
                self._set_checked(self.chk_mri_t1, True)
                self.chk_mri_t1.toggled.connect(
                    lambda checked=False: self._on_3d_mri_source_toggled("T1", checked)
                )

            if self.chk_mri_t2 is not None:
                self._set_checked(self.chk_mri_t2, False)
                self.chk_mri_t2.toggled.connect(
                    lambda checked=False: self._on_3d_mri_source_toggled("T2", checked)
                )
        except Exception:
            pass

        if self.sld_3d_brainMaskOpacity is not None:
            self.sld_3d_brainMaskOpacity.valueChanged.connect(
                lambda _: self._update_brain_opacity()
            )

        if self.sld_3d_PialOpacity is not None:
            self.sld_3d_PialOpacity.valueChanged.connect(lambda _: self._update_brain_opacity())

        if self.chk_pet is not None:
            self.chk_pet.toggled.connect(self._on_pet_toggled)

        if self.sld_pet_min is not None and self.sb_pet_min is not None:
            self.sld_pet_min.valueChanged.connect(self.sb_pet_min.setValue)
            self.sb_pet_min.valueChanged.connect(self.sld_pet_min.setValue)
            self.sld_pet_min.valueChanged.connect(lambda _: self._update_threshold_labels())
            self.sld_pet_min.valueChanged.connect(lambda _: self._refresh_pet_only())

        if self.sld_pet_max is not None and self.sb_pet_max is not None:
            self.sld_pet_max.valueChanged.connect(self.sb_pet_max.setValue)
            self.sb_pet_max.valueChanged.connect(self.sld_pet_max.setValue)
            self.sld_pet_max.valueChanged.connect(lambda _: self._update_threshold_labels())
            self.sld_pet_max.valueChanged.connect(lambda _: self._refresh_pet_only())

        if self.sld_pet_gamma is not None and self.dsb_pet_gamma is not None:
            self.sld_pet_gamma.valueChanged.connect(
                lambda v: self.dsb_pet_gamma.setValue(float(v) / 100.0)
            )
            self.dsb_pet_gamma.valueChanged.connect(
                lambda v: self.sld_pet_gamma.setValue(int(round(float(v) * 100.0)))
            )
            self.sld_pet_gamma.valueChanged.connect(lambda _: self._update_threshold_labels())
            self.sld_pet_gamma.valueChanged.connect(lambda _: self._refresh_pet_only())

        if self.sld_pet_opacity is not None:
            self.sld_pet_opacity.valueChanged.connect(
                lambda _: self._update_visible_pet_overlay_opacity_only()
            )

        if self.chk_siscom is not None:
            self.chk_siscom.toggled.connect(self._on_siscom_toggled)

        if self.dsb_siscom_z is not None:
            self.dsb_siscom_z.valueChanged.connect(lambda _: self._update_threshold_labels())
            self.dsb_siscom_z.valueChanged.connect(lambda _: self._refresh_siscom_only())
        if self.sld_siscom_opacity is not None:
            self.sld_siscom_opacity.valueChanged.connect(
                lambda _: self._update_visible_siscom_overlay_opacity_only()
            )

        if self.chk_ct is not None:
            self.chk_ct.toggled.connect(lambda _: self._render_ct())
        if self.sld_ct_thr is not None:
            self.sld_ct_thr.setTracking(False)
            self.sld_ct_thr.valueChanged.connect(lambda _: self._update_threshold_labels())
            self.sld_ct_thr.sliderReleased.connect(self._render_ct)
        if self.sld_ct_opacity is not None:
            self.sld_ct_opacity.valueChanged.connect(lambda _: self._update_ct_opacity())

        if self.btn_elec_shaft is not None:
            self.btn_elec_shaft.setCheckable(True)
            self.btn_elec_shaft.setChecked(True)
            self.btn_elec_shaft.toggled.connect(
                lambda _: self._on_electrode_display_option_changed()
            )

        if self.spin_contacts_size is not None:
            try:
                if float(self.spin_contacts_size.value()) <= 0:
                    self.spin_contacts_size.setValue(8)
            except Exception:
                pass

            self.spin_contacts_size.valueChanged.connect(lambda _: self._on_contacts_size_changed())

        if self.chk_coronal_plane is not None:
            self.chk_coronal_plane.toggled.connect(
                lambda _: self._on_single_plane_changed("coronal")
            )
        if self.sld_coronal_plane is not None:
            self.sld_coronal_plane.valueChanged.connect(
                lambda _: self._on_single_plane_changed("coronal")
            )

        if self.chk_axial_plane is not None:
            self.chk_axial_plane.toggled.connect(lambda _: self._on_single_plane_changed("axial"))
        if self.sld_axial_plane is not None:
            self.sld_axial_plane.valueChanged.connect(
                lambda _: self._on_single_plane_changed("axial")
            )

        if self.chk_sagittal_plane is not None:
            self.chk_sagittal_plane.toggled.connect(
                lambda _: self._on_single_plane_changed("sagittal")
            )
        if self.sld_sagittal_plane is not None:
            self.sld_sagittal_plane.valueChanged.connect(
                lambda _: self._on_single_plane_changed("sagittal")
            )

        if self.chk_ct is not None:
            self.chk_ct.toggled.connect(lambda _: self._update_modality_controls_enabled_states())
        if self.chk_pet is not None:
            self.chk_pet.toggled.connect(lambda _: self._update_modality_controls_enabled_states())
        if self.chk_siscom is not None:
            self.chk_siscom.toggled.connect(
                lambda _: self._update_modality_controls_enabled_states()
            )

        if self.chk_coronal_plane is not None:
            self.chk_coronal_plane.toggled.connect(
                lambda _: self._update_plane_slider_enabled_states()
            )
            self.chk_coronal_plane.toggled.connect(lambda _: self._update_planes_info_label())
        if self.chk_axial_plane is not None:
            self.chk_axial_plane.toggled.connect(
                lambda _: self._update_plane_slider_enabled_states()
            )
            self.chk_axial_plane.toggled.connect(lambda _: self._update_planes_info_label())
        if self.chk_sagittal_plane is not None:
            self.chk_sagittal_plane.toggled.connect(
                lambda _: self._update_plane_slider_enabled_states()
            )
            self.chk_sagittal_plane.toggled.connect(lambda _: self._update_planes_info_label())

        if self.sld_coronal_plane is not None:
            self.sld_coronal_plane.valueChanged.connect(lambda _: self._update_planes_info_label())
        if self.sld_axial_plane is not None:
            self.sld_axial_plane.valueChanged.connect(lambda _: self._update_planes_info_label())
        if self.sld_sagittal_plane is not None:
            self.sld_sagittal_plane.valueChanged.connect(lambda _: self._update_planes_info_label())

        if self.btn_3d_StartInf is not None:
            self.btn_3d_StartInf.clicked.connect(self._toggle_axial_direction)

        if self.btn_3d_StartCaudal is not None:
            self.btn_3d_StartCaudal.clicked.connect(self._toggle_coronal_direction)

        if self.btn_3d_StartLH is not None:
            self.btn_3d_StartLH.clicked.connect(self._toggle_sagittal_direction)

        self._update_brain_opacity_slider_states()
        self._update_modality_controls_enabled_states()
        self._update_plane_slider_enabled_states()
        self._update_planes_info_label()

    def _on_electrode_display_option_changed(self) -> None:
        """
        Refresh native or MNI electrode display when shaft display changes.
        """
        try:
            if self.chk_mni_atlas is not None and self.chk_mni_atlas.isChecked():
                self._render_mni_scene(reset_camera=False)
                return
        except Exception:
            pass

        try:
            self.update_electrodes()
        except Exception:
            pass

        try:
            if self.chk_pial is not None and self.chk_pial.isChecked():
                self._render_brain(reset_camera=False)
        except Exception:
            pass

    def _on_contacts_size_changed(self) -> None:
        """
        Refresh native electrodes or MNI electrodes when contact size changes.

        In pial mode, rebuild the pial actor after the electrode actors without
        changing the camera. This avoids electrodes disappearing after resizing.
        """
        try:
            if self.chk_mni_atlas is not None and self.chk_mni_atlas.isChecked():
                self._render_mni_scene(reset_camera=False)
                return
        except Exception:
            pass

        try:
            self.update_electrodes()
        except Exception:
            pass

        try:
            if self.chk_pial is not None and self.chk_pial.isChecked():
                self._render_brain(reset_camera=False)
        except Exception:
            pass

    def _toggle_axial_direction(self):
        self._axial_from_inferior = not self._axial_from_inferior
        self._update_all_planes()

    def _toggle_coronal_direction(self):
        self._coronal_from_caudal = not self._coronal_from_caudal
        self._update_all_planes()

    def _toggle_sagittal_direction(self):
        self._sagittal_from_left = not self._sagittal_from_left
        self._update_all_planes()

    def _update_threshold_labels(self):

        # CT
        if self.sld_ct_thr is not None and self.lbl_ct_thr_info is not None:
            v = int(self.sld_ct_thr.value())
            self.lbl_ct_thr_info.setText(f"{v} HU (Hounsfield Units)")

        # SISCOM
        if self.dsb_siscom_z is not None and self.lbl_siscom_info is not None:
            v = float(self.dsb_siscom_z.value())
            self.lbl_siscom_info.setText(f"Z-score > {v:.1f}")

    def _set_checked(self, cb: QCheckBox | None, v: bool) -> None:
        if cb is None:
            return
        cb.blockSignals(True)
        cb.setChecked(bool(v))
        cb.blockSignals(False)

    def _on_chk_parcel1_toggled(self, checked: bool):
        try:
            if checked and self.chk_parcel2 is not None and self.chk_parcel2.isChecked():
                self.chk_parcel2.blockSignals(True)
                self.chk_parcel2.setChecked(False)
                self.chk_parcel2.blockSignals(False)
        except Exception:
            pass
        self._invalidate_slice_volume_cache(base=True, pet=False, siscom=False)
        self._update_modality_controls_enabled_states()

        # MNI mode: parcellation is only an overlay on MNI T1 slices.
        if self._mni_t1_slices_are_visible():
            try:
                self._refresh_all_visible_slice_planes_full()
            except Exception:
                pass
            return

        # Native mode.
        try:
            self._render_brain()
        except Exception:
            pass

        try:
            self._refresh_all_visible_slice_planes_full()
        except Exception:
            pass

    def _on_chk_parcel2_toggled(self, checked: bool):
        try:
            if checked and self.chk_parcel1 is not None and self.chk_parcel1.isChecked():
                self.chk_parcel1.blockSignals(True)
                self.chk_parcel1.setChecked(False)
                self.chk_parcel1.blockSignals(False)
        except Exception:
            pass

        self._invalidate_slice_volume_cache(base=True, pet=False, siscom=False)
        self._update_modality_controls_enabled_states()

        # MNI mode: parcellation is only an overlay on MNI T1 slices.
        if self._mni_t1_slices_are_visible():
            try:
                self._refresh_all_visible_slice_planes_full()
            except Exception:
                pass
            return

        # Native mode.
        try:
            self._render_brain()
        except Exception:
            pass

        try:
            self._refresh_all_visible_slice_planes_full()
        except Exception:
            pass

    def _on_brain_source(self, which: str, checked: bool) -> None:
        self._invalidate_slice_volume_cache()
        # If user clicks Brain mask or Pial surface while MNI is active,
        # leave MNI mode completely, then continue in native mode.
        if checked and which in ("brainmask", "pial"):
            try:
                if self.chk_mni_atlas is not None and self.chk_mni_atlas.isChecked():
                    self._leave_mni_mode_and_restore_native(
                        restore_previous_checkboxes=False,
                        keep_clicked_native_source=which,
                    )
            except Exception:
                pass
        # If turning one source ON, make it exclusive
        if checked:
            if which == "brainmask":
                self._set_checked(self.chk_iso, False)
                self._set_checked(self.chk_pial, False)
            elif which == "iso":
                self._set_checked(self.chk_brainmask, False)
                self._set_checked(self.chk_pial, False)
            elif which == "pial":
                self._set_checked(self.chk_brainmask, False)
                self._set_checked(self.chk_iso, False)

        # If nothing is checked anymore, remove brain actor immediately
        any_brain_source = False
        if self.chk_brainmask is not None and self.chk_brainmask.isChecked():
            any_brain_source = True
        if self.chk_iso is not None and self.chk_iso.isChecked():
            any_brain_source = True
        if self.chk_pial is not None and self.chk_pial.isChecked():
            any_brain_source = True

        if not any_brain_source:
            self._remove_actor("brain")
            self._render()
            return

        if which == "brainmask" and checked and self.chk_pial is not None:
            self.chk_pial.blockSignals(True)
            self.chk_pial.setChecked(False)
            self.chk_pial.blockSignals(False)

        elif which == "pial" and checked and self.chk_brainmask is not None:
            self.chk_brainmask.blockSignals(True)
            self.chk_brainmask.setChecked(False)
            self.chk_brainmask.blockSignals(False)

        # If both become unchecked, keep brainmask as fallback if available
        if (self.chk_brainmask is not None and not self.chk_brainmask.isChecked()) and (
            self.chk_pial is not None and not self.chk_pial.isChecked()
        ):
            if self._brainmask_img is not None and self.chk_brainmask is not None:
                self.chk_brainmask.blockSignals(True)
                self.chk_brainmask.setChecked(True)
                self.chk_brainmask.blockSignals(False)

        self._update_brain_opacity_slider_states()
        self._render_brain()

    def _get_t2_for_3d(self) -> sitk.Image | None:
        """
        Return T2 in T1 space for 3D View planes.

        We do not use raw T2 here, because the coronal/axial/sagittal planes
        are defined in T1 space.
        """
        if getattr(self, "_t2_img", None) is not None:
            return self._t2_img

        img = getattr(self.state, "t2_coreg_in_t1", None)
        if isinstance(img, sitk.Image):
            return img

        img = getattr(self.state, "t2_in_t1", None)
        if isinstance(img, sitk.Image):
            return img

        return None

    def _get_active_mri_for_3d(self) -> sitk.Image | None:
        """
        Image used as grayscale anatomical texture on 3D slice planes.
        Geometry still comes from _get_3d_plane_reference_img().
        """
        if self._mni_t1_slices_are_visible():
            return self._get_mni_template_t1_image()

        source = str(getattr(self, "_active_mri_source_3d", "T1"))

        if source == "T2":
            t2 = self._get_t2_for_3d()
            if t2 is not None:
                return t2

        return self._t1_img

    def _refresh_3d_mri_source_controls(self) -> None:
        """
        Refresh T1/T2 checkbox states and redraw visible slice planes.
        """
        self._invalidate_slice_volume_cache(base=True, pet=False, siscom=False)
        try:
            self._update_modality_controls_enabled_states()
        except Exception:
            pass

        try:
            self._refresh_multiplanar_clipped_scene()
        except Exception:
            pass

        try:
            self._render()
        except Exception:
            pass

    def _on_3d_mri_source_toggled(self, source: str, checked: bool) -> None:
        source = str(source).upper().strip()

        if not checked:
            # Avoid both T1 and T2 being unchecked.
            t1_checked = bool(self.chk_mri_t1 is not None and self.chk_mri_t1.isChecked())
            t2_checked = bool(self.chk_mri_t2 is not None and self.chk_mri_t2.isChecked())

            if not t1_checked and not t2_checked:
                self._active_mri_source_3d = "T1"
                self._set_checked(self.chk_mri_t1, True)
                self._set_checked(self.chk_mri_t2, False)
                self._refresh_3d_mri_source_controls()
            return

        if source == "T1":
            self._active_mri_source_3d = "T1"
            self._set_checked(self.chk_mri_t2, False)

        elif source == "T2":
            if self._get_t2_for_3d() is None:
                self._active_mri_source_3d = "T1"
                self._set_checked(self.chk_mri_t2, False)
                self._set_checked(self.chk_mri_t1, True)
                self._refresh_3d_mri_source_controls()
                return

            self._active_mri_source_3d = "T2"
            self._set_checked(self.chk_mri_t1, False)

        self._refresh_3d_mri_source_controls()

    # ---------------- Public API ----------------
    def set_t1(self, t1_img: sitk.Image | None, t1_path: str | None = None) -> None:
        if t1_img is None and t1_path:
            try:
                t1_img = sitk.ReadImage(t1_path)
            except Exception:
                t1_img = None
        self._t1_img = t1_img
        self._invalidate_slice_volume_cache()

        if self._t1_img is not None:
            self._active_mri_source_3d = "T1"
            self._set_checked(getattr(self, "chk_mri_t1", None), True)
            self._set_checked(getattr(self, "chk_mri_t2", None), False)

        try:
            self._t1_np = (
                sitk.GetArrayFromImage(self._t1_img).astype(np.float32)
                if self._t1_img is not None
                else None
            )
        except Exception:
            self._t1_np = None
        try:
            self._update_all_plane_slider_ranges()
        except Exception:
            pass
        try:
            self._render_coronal_plane()
        except Exception:
            pass
        try:
            self._refresh_multiplanar_clipped_scene()
        except Exception:
            pass

        try:
            self._render_surface_projections()
        except Exception:
            pass
        try:
            self._update_modality_controls_enabled_states()
            self._update_plane_slider_enabled_states()
            self._update_planes_info_label()
        except Exception:
            pass
        try:
            self._update_modality_controls_enabled_states()
            self._update_plane_slider_enabled_states()
            self._update_planes_info_label()
        except Exception:
            pass

    def set_t2(self, t2_img: sitk.Image | None, t2_path: str | None = None) -> None:
        """
        Set T2 image for 3D slice-plane display.

        Important:
        This T2 must already be coregistered/resampled in T1 space.
        """
        if t2_img is None and t2_path:
            try:
                t2_img = sitk.ReadImage(t2_path)
            except Exception:
                t2_img = None

        self._t2_img = t2_img
        self._invalidate_slice_volume_cache(base=True, pet=False, siscom=False)

        # If no T2 is available anymore, fallback to T1.
        if self._t2_img is None and getattr(self, "_active_mri_source_3d", "T1") == "T2":
            self._active_mri_source_3d = "T1"
            self._set_checked(self.chk_mri_t1, True)
            self._set_checked(self.chk_mri_t2, False)

        try:
            self._update_modality_controls_enabled_states()
        except Exception:
            pass

        try:
            self._refresh_multiplanar_clipped_scene()
        except Exception:
            pass

        try:
            self._render()
        except Exception:
            pass

    def set_brainmask(
        self, brainmask_img: sitk.Image | None, brainmask_path: str | None = None
    ) -> None:
        if brainmask_img is None and brainmask_path:
            try:
                brainmask_img = sitk.ReadImage(brainmask_path)
            except Exception:
                brainmask_img = None

        self._brainmask_img = brainmask_img
        self._invalidate_slice_volume_cache()

        if self.chk_brainmask is not None:
            self._set_checked(self.chk_brainmask, True)
        if self.chk_iso is not None:
            self._set_checked(self.chk_iso, False)
        if self.chk_pial is not None:
            self._set_checked(self.chk_pial, False)

        if self.sld_3d_brainMaskOpacity is not None:
            self.sld_3d_brainMaskOpacity.blockSignals(True)
            self.sld_3d_brainMaskOpacity.setValue(50)
            self.sld_3d_brainMaskOpacity.blockSignals(False)

        self._update_brain_opacity_slider_states()

        self._render_brain()

        try:
            self._update_all_plane_slider_ranges()
        except Exception:
            pass

        try:
            self._refresh_multiplanar_clipped_scene()
        except Exception:
            pass

    def set_iso_surface(self, mesh_data: object) -> None:
        try:
            self._iso_mesh_data = dict(mesh_data)
        except Exception:
            self._iso_mesh_data = None

        # Iso-surface path kept only for compatibility but hidden in UI.
        self._render_brain()

    def set_pet(
        self,
        pet_img: sitk.Image | None,
        pet_path: str | None = None,
        activate: bool = False,
    ) -> None:
        if pet_img is None and pet_path:
            try:
                pet_img = sitk.ReadImage(pet_path)
            except Exception:
                pet_img = None
        self._pet_img = pet_img
        self._invalidate_slice_volume_cache(base=False, pet=True, siscom=False)

        # Default PET controls after PET coreg is validated / loaded in 3D view
        try:
            if self.sld_pet_min is not None:
                self.sld_pet_min.blockSignals(True)
                self.sld_pet_min.setValue(15)
                self.sld_pet_min.blockSignals(False)
            if self.sb_pet_min is not None:
                self.sb_pet_min.blockSignals(True)
                self.sb_pet_min.setValue(15)
                self.sb_pet_min.blockSignals(False)

            if self.sld_pet_gamma is not None:
                self.sld_pet_gamma.blockSignals(True)
                self.sld_pet_gamma.setValue(20)  # 0.20
                self.sld_pet_gamma.blockSignals(False)

            if self.dsb_pet_gamma is not None:
                self.dsb_pet_gamma.blockSignals(True)
                self.dsb_pet_gamma.setValue(0.20)
                self.dsb_pet_gamma.blockSignals(False)
            if self.sld_pet_max is not None:
                self.sld_pet_max.blockSignals(True)
                self.sld_pet_max.setValue(75)
                self.sld_pet_max.blockSignals(False)
            if self.sb_pet_max is not None:
                self.sb_pet_max.blockSignals(True)
                self.sb_pet_max.setValue(75)
                self.sb_pet_max.blockSignals(False)

            if self.sld_pet_opacity is not None:
                self.sld_pet_opacity.blockSignals(True)
                self.sld_pet_opacity.setValue(50)
                self.sld_pet_opacity.blockSignals(False)
        except Exception:
            pass

        # By default PET stays hidden when loaded manually.
        # When restoring a project JSON, activate=True allows the validated PET
        # to come back exactly as an available/active overlay.
        if self.chk_pet is not None:
            self.chk_pet.setEnabled(bool(pet_img is not None))
            self._set_checked(
                self.chk_pet,
                bool(activate and pet_img is not None),
            )

        try:
            self._update_threshold_labels()
        except Exception:
            pass

        try:
            self._remove_pet_scalar_bar()
        except Exception:
            pass

        self._update_modality_controls_enabled_states()
        self._update_planes_info_label()
        if bool(activate and pet_img is not None):
            try:
                self._refresh_pet_only()
            except Exception:
                pass

    def set_siscom(self, siscom_img: sitk.Image | None, siscom_path: str | None = None) -> None:
        if siscom_img is None and siscom_path:
            try:
                siscom_img = sitk.ReadImage(siscom_path)
            except Exception:
                siscom_img = None

        self._siscom_img = siscom_img
        self._invalidate_slice_volume_cache(base=False, pet=False, siscom=True)
        self._siscom_fixed_zmax = None
        try:
            if self._siscom_img is not None:
                arr = sitk.GetArrayFromImage(self._siscom_img).astype(np.float32)
                vals = arr[np.isfinite(arr)]
                vals = vals[vals > 0]
                if vals.size > 0:
                    self._siscom_fixed_zmax = float(np.percentile(vals, 99.0))
        except Exception:
            self._siscom_fixed_zmax = None

        # Default SISCOM controls after SISCOM is loaded in 3D view
        try:
            if self.dsb_siscom_z is not None:
                self.dsb_siscom_z.blockSignals(True)
                self.dsb_siscom_z.setValue(2.0)
                self.dsb_siscom_z.blockSignals(False)

            if self.sld_siscom_opacity is not None:
                self.sld_siscom_opacity.blockSignals(True)
                self.sld_siscom_opacity.setValue(50)
                self.sld_siscom_opacity.blockSignals(False)
        except Exception:
            pass

        # Keep SISCOM hidden by default: user decides when to show it
        if self.chk_siscom is not None:
            self._set_checked(self.chk_siscom, False)

        try:
            self._update_threshold_labels()
        except Exception:
            pass

        try:
            self._remove_siscom_scalar_bar()
        except Exception:
            pass

        self._update_modality_controls_enabled_states()
        self._update_planes_info_label()

    def set_parcellation1(self, img: sitk.Image | None, path: str | None = None) -> None:
        if img is None and path:
            try:
                img = sitk.ReadImage(path)
            except Exception:
                img = None

        self._parcel1_img = img
        self._invalidate_slice_volume_cache(base=True, pet=False, siscom=False)

        try:
            lut = getattr(self.state, "parcellation1_lut", {})
            if isinstance(lut, dict):
                self._parcel1_lut = dict(lut)
            else:
                self._parcel1_lut = {}
        except Exception:
            self._parcel1_lut = {}

        if self.chk_parcel1 is not None:
            self._set_checked(self.chk_parcel1, False)

        self._update_modality_controls_enabled_states()
        self._refresh_multiplanar_clipped_scene()

    def set_parcellation2(self, img: sitk.Image | None, path: str | None = None) -> None:
        if img is None and path:
            try:
                img = sitk.ReadImage(path)
            except Exception:
                img = None

        self._parcel2_img = img
        self._invalidate_slice_volume_cache(base=True, pet=False, siscom=False)

        try:
            lut = getattr(self.state, "parcellation2_lut", {})
            if isinstance(lut, dict):
                self._parcel2_lut = dict(lut)
            else:
                self._parcel2_lut = {}
        except Exception:
            self._parcel2_lut = {}

        if self.chk_parcel2 is not None:
            self._set_checked(self.chk_parcel2, False)

        self._update_modality_controls_enabled_states()
        self._refresh_multiplanar_clipped_scene()

    def _get_active_parcellation_img_and_lut(self):
        p1_on = bool(self.chk_parcel1 is not None and self.chk_parcel1.isChecked())
        p2_on = bool(self.chk_parcel2 is not None and self.chk_parcel2.isChecked())

        # -------------------------
        # MNI mode
        # -------------------------
        if self._mni_t1_slices_are_visible():
            if p1_on:
                img, lut = self._get_mni_parcellation1_img_and_lut()
                return img, lut

            if p2_on:
                img, lut = self._get_mni_parcellation2_img_and_lut()
                return img, lut

            return None, {}

        # -------------------------
        # Native patient mode
        # -------------------------
        if p1_on and self._parcel1_img is not None:
            lut = getattr(self, "_parcel1_lut", None)
            if not isinstance(lut, dict):
                lut = getattr(self.state, "parcellation1_lut", {}) or {}
            return self._parcel1_img, lut

        if p2_on and self._parcel2_img is not None:
            lut = getattr(self, "_parcel2_lut", None)
            if not isinstance(lut, dict):
                lut = getattr(self.state, "parcellation2_lut", {}) or {}
            return self._parcel2_img, lut

        return None, {}

    def _sample_labels_at_ras_points(self, img: sitk.Image, pts_ras: np.ndarray):
        try:
            pts_ras = np.asarray(pts_ras, dtype=np.float64)
            if pts_ras.ndim != 2 or pts_ras.shape[1] != 3:
                return None

            pts_lps = pts_ras.copy()
            pts_lps[:, 0] *= -1.0
            pts_lps[:, 1] *= -1.0

            origin = np.asarray(img.GetOrigin(), dtype=np.float64)
            spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
            direction = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
            inv_direction = np.linalg.inv(direction)

            rel = pts_lps - origin[None, :]
            idx_xyz = (rel @ inv_direction.T) / spacing[None, :]

            x = np.round(idx_xyz[:, 0]).astype(int)
            y = np.round(idx_xyz[:, 1]).astype(int)
            z = np.round(idx_xyz[:, 2]).astype(int)

            arr = sitk.GetArrayFromImage(img)  # z,y,x
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
            return None

    def _clean_parcellation_label_for_display(self, name: str) -> str:
        """
        Clean parcellation labels for display.

        Example:
            17Networks_LH_VisCent_ExStr_1 -> LH_VisCent_ExStr_1
            17Networks_RH_DefaultA_PFCd_3 -> RH_DefaultA_PFCd_3

        Patient/native labels without LH_/RH_ are kept unchanged.
        """
        name = str(name or "").strip()

        if not name:
            return ""

        for hemi in ("LH_", "RH_"):
            idx = name.find(hemi)
            if idx >= 0:
                return name[idx:]

        return name

    def _parcellation_label_name_from_lut(self, label_value: int, lut: dict) -> str:
        try:
            lab = int(float(label_value))
        except Exception:
            return ""

        if lab <= 0:
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

        return self._clean_parcellation_label_for_display(name)

    def _handle_parcellation_hover_event(self, obj, event) -> None:
        """
        Delayed tooltip when the mouse stays over an active parcellation.

        Works for:
            - MNI parcellation on MNI T1 slices
            - patient/native parcellation on patient slices
            - patient/native parcellation displayed on the pial surface
        """
        try:
            et = event.type()

            if et == QEvent.MouseMove:
                if not self._parcellation_hover_display_active():
                    try:
                        self._parcellation_hover_timer.stop()
                    except Exception:
                        pass

                    QToolTip.hideText()
                    return

                self._parcellation_hover_qpos = (
                    event.position().toPoint() if hasattr(event, "position") else event.pos()
                )

                self._parcellation_hover_global_pos = obj.mapToGlobal(self._parcellation_hover_qpos)

                self._parcellation_hover_timer.start(650)

            elif et in (QEvent.Leave, QEvent.MouseButtonPress, QEvent.Wheel):
                try:
                    self._parcellation_hover_timer.stop()
                except Exception:
                    pass

                QToolTip.hideText()

        except Exception:
            pass

    def _show_parcellation_hover_tooltip(self) -> None:
        """
        Show active parcellation label under mouse.

        Works on:
            - MNI parcellation slices
            - patient/native parcellation slices
            - native pial surface colored with the active parcellation
        """
        try:
            if self.interactor is None or self.plotter is None:
                return

            if not self._parcellation_hover_display_active():
                QToolTip.hideText()
                return

            if self._parcellation_hover_qpos is None:
                return

            # IMPORTANT:
            # These two variables were missing in your current function.
            parcel_img, parcel_lut = self._get_active_parcellation_img_and_lut()

            if parcel_img is None or not isinstance(parcel_lut, dict) or not parcel_lut:
                QToolTip.hideText()
                return

            qpos = self._parcellation_hover_qpos
            x_qt = float(qpos.x())
            y_qt = float(qpos.y())

            ras = self._pick_world_ras_from_mouse(x_qt, y_qt)

            if ras is None:
                QToolTip.hideText()
                return

            lab = self._sample_single_label_at_ras(parcel_img, ras)

            if lab is None:
                QToolTip.hideText()
                return

            lab = int(lab)

            if lab < 0:
                QToolTip.hideText()
                return

            label_name = self._parcellation_label_name_from_lut(lab, parcel_lut)

            if not label_name:
                QToolTip.hideText()
                return

            text = label_name

            if lab > 0:
                text += f"  (index {lab})"

            global_pos = self._parcellation_hover_global_pos
            if global_pos is None:
                global_pos = self.interactor.mapToGlobal(qpos)

            QToolTip.showText(global_pos, text, self.interactor)

        except Exception as e:
            print("[Parcellation hover] Tooltip failed:", e)
            QToolTip.hideText()

    def _pick_world_ras_from_mouse(self, x_qt: float, y_qt: float):
        """
        Pick a 3D RAS point under the mouse using VTK.

        Qt mouse y is top-down.
        VTK mouse y is bottom-up.
        """
        try:
            if self.interactor is None or self.plotter is None:
                return None

            renderer = self.plotter.renderer
            h = int(self.interactor.height())

            x_vtk = int(round(float(x_qt)))
            y_vtk = int(round(float(h) - float(y_qt)))

            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.0005)

            ok = picker.Pick(x_vtk, y_vtk, 0, renderer)

            if not ok:
                return None

            pos = picker.GetPickPosition()

            if pos is None or len(pos) != 3:
                return None

            ras = np.asarray(pos, dtype=np.float64)

            if not np.all(np.isfinite(ras)):
                return None

            return ras

        except Exception:
            return None

    def _sample_single_label_at_ras(self, img: sitk.Image, ras_xyz) -> int | None:
        """
        Sample one integer parcellation label at one RAS coordinate.
        """
        try:
            ras = np.asarray(ras_xyz, dtype=np.float64).reshape(3)

            # RAS -> LPS for SimpleITK
            lps = ras.copy()
            lps[0] *= -1.0
            lps[1] *= -1.0

            idx = img.TransformPhysicalPointToIndex(tuple(float(v) for v in lps))
            size = img.GetSize()

            if (
                idx[0] < 0
                or idx[0] >= size[0]
                or idx[1] < 0
                or idx[1] >= size[1]
                or idx[2] < 0
                or idx[2] >= size[2]
            ):
                return -1

            arr = sitk.GetArrayFromImage(img)  # z, y, x
            lab = int(arr[int(idx[2]), int(idx[1]), int(idx[0])])

            return lab

        except Exception:
            return None

    def _labels_to_rgb(self, labels: np.ndarray, lut: dict):
        try:
            labels = np.asarray(labels, dtype=np.int32).ravel()
            rgb = np.zeros((labels.shape[0], 3), dtype=np.uint8)

            if not isinstance(lut, dict):
                return rgb

            for lab in np.unique(labels):
                if int(lab) <= 0:
                    continue
                entry = lut.get(int(lab), None)
                if entry is None:
                    continue
                try:
                    _name, color = entry
                    rgb[labels == int(lab), :] = np.array(color, dtype=np.uint8)
                except Exception:
                    pass
            return rgb
        except Exception:
            return np.zeros((0, 3), dtype=np.uint8)

    def _apply_brain_actor_opacity_mode(self, actor, opacity: float) -> None:
        """
        Apply the requested brain surface opacity with a stable VTK render mode.

        - below 99%: translucent surface
        - from 99% upward: completely opaque surface
        """
        if actor is None:
            return

        try:
            opacity = float(np.clip(float(opacity), 0.0, 1.0))
        except Exception:
            opacity = 1.0

        # Avoid an almost-opaque actor remaining in the translucent rendering pass.
        if opacity >= 0.99:
            opacity = 1.0

        try:
            prop = actor.GetProperty()
            if prop is not None:
                prop.SetOpacity(float(opacity))
        except Exception:
            pass

        try:
            if opacity >= 1.0:
                actor.ForceTranslucentOff()
                actor.ForceOpaqueOn()
            else:
                actor.ForceOpaqueOff()
                actor.ForceTranslucentOn()
        except Exception:
            pass

    # ---------------- Rendering ----------------
    def _render_brain(self, reset_camera: bool = True) -> None:
        if not _PV_OK or self.plotter is None:
            return

        # ---------------- Pial surface ----------------
        if self.chk_pial is not None and self.chk_pial.isChecked():
            polys = []

            # Preferred: pial masks already coregistered in T1 space
            if self._show_lh_pial:
                if self._lh_pial_mask_img is not None:
                    try:
                        poly = self._binarymask_to_polydata(self._lh_pial_mask_img)
                        if poly is not None and getattr(poly, "n_points", 0) > 0:
                            polys.append(poly.copy())
                    except Exception:
                        pass
                elif self._lh_pial_poly is not None:
                    polys.append(self._lh_pial_poly.copy())

            if self._show_rh_pial:
                if self._rh_pial_mask_img is not None:
                    try:
                        poly = self._binarymask_to_polydata(self._rh_pial_mask_img)
                        if poly is not None and getattr(poly, "n_points", 0) > 0:
                            polys.append(poly.copy())
                    except Exception:
                        pass
                elif self._rh_pial_poly is not None:
                    polys.append(self._rh_pial_poly.copy())

            if not polys:
                self._remove_actor("brain")
                self._render()
                return

            try:
                mesh = polys[0]
                for p in polys[1:]:
                    mesh = mesh.merge(p)
            except Exception:
                mesh = polys[0]

            try:
                mesh = mesh.triangulate()
            except Exception:
                pass

            try:
                mesh = mesh.clean(tolerance=1e-6)
            except Exception:
                pass

            self._remove_actor("brain")
            self._setup_brain_render_lights()

            opacity = (
                self.sld_3d_PialOpacity.value() / 100.0
                if getattr(self, "sld_3d_PialOpacity", None) is not None
                else 1.0
            )
            if opacity >= 0.99:
                opacity = 1.0

            p = getattr(self, "_brain_render_params", {})

            parcel_img, parcel_lut = self._get_active_parcellation_img_and_lut()

            scalars_name = None
            rgb_scalars = None

            if parcel_img is not None:
                try:
                    labels = self._sample_labels_at_ras_points(
                        parcel_img, np.asarray(mesh.points, dtype=np.float32)
                    )
                    if labels is not None:
                        rgb_scalars = self._labels_to_rgb(labels, parcel_lut)
                        if rgb_scalars is not None and rgb_scalars.shape[0] == mesh.n_points:
                            mesh["parcel_rgb"] = rgb_scalars
                            scalars_name = "parcel_rgb"
                except Exception:
                    scalars_name = None

            if scalars_name is not None:
                self._brain_actor = self.plotter.add_mesh(
                    mesh,
                    scalars=scalars_name,
                    rgb=True,
                    opacity=float(opacity),
                    smooth_shading=True,
                    ambient=float(p.get("ambient", 0.35)),
                    diffuse=float(p.get("diffuse", 0.60)),
                    specular=float(p.get("specular", 0.08)),
                    specular_power=float(p.get("specular_power", 12.0)),
                )
            else:
                self._brain_actor = self.plotter.add_mesh(
                    mesh,
                    color=self._brain_color,
                    opacity=float(opacity),
                    smooth_shading=True,
                    ambient=float(p.get("ambient", 0.35)),
                    diffuse=float(p.get("diffuse", 0.60)),
                    specular=float(p.get("specular", 0.08)),
                    specular_power=float(p.get("specular_power", 12.0)),
                )

            self._apply_brain_actor_opacity_mode(self._brain_actor, opacity)

            try:
                prop = self._brain_actor.GetProperty()

                # Do not cull pial back faces:
                # on some pial meshes / mask-derived surfaces this can create visible
                # holes and give a falsely transparent appearance at full opacity.
                prop.BackfaceCullingOff()
                prop.FrontfaceCullingOff()

                prop.SetInterpolationToPhong()
            except Exception:
                pass

            try:
                self._brain_actor.PickableOn()
            except Exception:
                try:
                    self._brain_actor.SetPickable(True)
                except Exception:
                    pass

            if reset_camera:
                try:
                    self.plotter.reset_camera()
                    self.plotter.camera.zoom(1.2)
                except Exception:
                    pass

            self._apply_actor_clipping()
            self._refresh_multiplanar_clipped_scene()
            self.render_all_surface_projections()
            self._render()
            return

        # ---------------- Brainmask ----------------
        if self.chk_brainmask is not None and self.chk_brainmask.isChecked():
            if self._brainmask_img is None:
                self._remove_actor("brain")
                self._render()
                return
            mesh = self._binarymask_to_polydata(self._brainmask_img)
        else:
            if self._brainmask_img is not None:
                mesh = self._binarymask_to_polydata(self._brainmask_img)
            else:
                self._remove_actor("brain")
                self._render()
                return

        if mesh is None or getattr(mesh, "n_points", 0) == 0:
            self._remove_actor("brain")
            self._render()
            return

        self._remove_actor("brain")
        self._setup_brain_render_lights()

        opacity = (
            self.sld_3d_brainMaskOpacity.value() / 100.0
            if getattr(self, "sld_3d_brainMaskOpacity", None) is not None
            else 0.35
        )

        p = getattr(self, "_brain_render_params", {})

        self._brain_actor = self.plotter.add_mesh(
            mesh,
            color=self._brain_color,
            opacity=float(opacity),
            smooth_shading=True,
            ambient=float(p.get("ambient", 0.10)),
            diffuse=float(p.get("diffuse", 0.80)),
            specular=float(p.get("specular", 0.30)),
            specular_power=float(p.get("specular_power", 40.0)),
        )
        self._apply_brain_actor_opacity_mode(self._brain_actor, opacity)
        if reset_camera:
            try:
                self.plotter.reset_camera()
                self.plotter.camera.zoom(1.2)
            except Exception:
                pass

        self._apply_actor_clipping()
        self._refresh_multiplanar_clipped_scene()
        self.render_all_surface_projections()
        self._render()

    def _update_brain_opacity(self) -> None:
        # MNI atlas opacity is controlled by the Brain mask opacity slider.
        try:
            if self.chk_mni_atlas is not None and self.chk_mni_atlas.isChecked():
                if self._mni_atlas_actor is not None:
                    op = (
                        self.sld_3d_brainMaskOpacity.value() / 100.0
                        if self.sld_3d_brainMaskOpacity is not None
                        else 0.22
                    )
                    self._mni_atlas_actor.GetProperty().SetOpacity(float(op))
                    self._render()
                return
        except Exception:
            pass
        if not _PV_OK or self.plotter is None or self._brain_actor is None:
            return
        try:
            if self.chk_pial is not None and self.chk_pial.isChecked():
                op = (
                    self.sld_3d_PialOpacity.value() / 100.0
                    if self.sld_3d_PialOpacity is not None
                    else 1.0
                )
            else:
                op = (
                    self.sld_3d_brainMaskOpacity.value() / 100.0
                    if self.sld_3d_brainMaskOpacity is not None
                    else 0.35
                )

            self._apply_brain_actor_opacity_mode(self._brain_actor, float(op))
            self._render()
        except Exception:
            pass

    def _get_pet_minmax_percentiles(self):
        pmin = 30.0
        pmax = 98.0

        try:
            if self.sld_pet_min is not None:
                pmin = float(self.sld_pet_min.value())
        except Exception:
            pass

        try:
            if self.sld_pet_max is not None:
                pmax = float(self.sld_pet_max.value())
        except Exception:
            pass

        if pmax <= pmin:
            pmax = pmin + 1.0

        return pmin, pmax

    def _get_pet_gamma_value(self):
        try:
            if self.dsb_pet_gamma is not None:
                return max(0.1, float(self.dsb_pet_gamma.value()))
        except Exception:
            pass
        try:
            if self.sld_pet_gamma is not None:
                return max(0.1, float(self.sld_pet_gamma.value()) / 100.0)
        except Exception:
            pass
        return 1.0

    def _any_slice_plane_visible(self) -> bool:
        return any(
            [
                bool(self.chk_coronal_plane is not None and self.chk_coronal_plane.isChecked()),
                bool(self.chk_axial_plane is not None and self.chk_axial_plane.isChecked()),
                bool(self.chk_sagittal_plane is not None and self.chk_sagittal_plane.isChecked()),
            ]
        )

    def _parcellation_hover_display_active(self) -> bool:
        """
        Return True when an active parcellation is visually displayed on an object
        that can be hovered:
        - a visible coronal/axial/sagittal slice
        - the native pial surface
        """
        try:
            parcel_img, parcel_lut = self._get_active_parcellation_img_and_lut()

            if parcel_img is None or not isinstance(parcel_lut, dict) or not parcel_lut:
                return False
        except Exception:
            return False

        try:
            if self._any_slice_plane_visible():
                return True
        except Exception:
            pass

        try:
            if self.chk_pial is not None and self.chk_pial.isChecked():
                return bool(self._brain_actor is not None)
        except Exception:
            pass

        return False

    def _native_surface_mode_active(self) -> bool:
        """
        True when the patient brainmask or pial surface is active.
        This option is not used in MNI mode.
        """
        try:
            mni_on = bool(self.chk_mni_atlas is not None and self.chk_mni_atlas.isChecked())
        except Exception:
            mni_on = False

        if mni_on:
            return False

        try:
            brainmask_on = bool(self.chk_brainmask is not None and self.chk_brainmask.isChecked())
        except Exception:
            brainmask_on = False

        try:
            pial_on = bool(self.chk_pial is not None and self.chk_pial.isChecked())
        except Exception:
            pial_on = False

        return bool(brainmask_on or pial_on)

    def _show_keep_electrodes_through_slices_option(self) -> bool:
        """
        Show the context-menu option only when it is useful:
        - native brainmask or pial surface is visible
        - at least one slice plane is visible
        - native electrodes are loaded
        """
        try:
            has_electrodes = bool(getattr(self.state, "electrodes", None))
        except Exception:
            has_electrodes = False

        return bool(
            self._native_surface_mode_active()
            and self._any_slice_plane_visible()
            and has_electrodes
        )

    def _apply_electrode_actor_depth_mode(self, actor) -> None:
        """
        Keep electrode contacts/shafts opaque and correctly visible through
        translucent brain or pial surfaces.

        Slice clipping remains controlled independently by
        _keep_electrodes_visible_through_slices.
        """
        if actor is None:
            return

        try:
            actor.PickableOff()
        except Exception:
            try:
                actor.SetPickable(False)
            except Exception:
                pass

        # Electrode contacts and shafts are opaque objects.
        # This must be applied every time they are recreated, including after
        # changing the contact size.
        try:
            prop = actor.GetProperty()
            if prop is not None:
                prop.SetOpacity(1.0)
        except Exception:
            pass

        try:
            actor.ForceTranslucentOff()
            actor.ForceOpaqueOn()
        except Exception:
            pass

        # Optional behavior already implemented for the slice planes:
        # do not clip electrodes when the user wants them visible through slices.
        if bool(getattr(self, "_keep_electrodes_visible_through_slices", False)):
            try:
                mapper = actor.GetMapper()
                if mapper is not None:
                    mapper.RemoveAllClippingPlanes()
            except Exception:
                pass

    def _toggle_keep_electrodes_visible_through_slices(self) -> None:
        """
        Toggle whether coronal/axial/sagittal slices are allowed to hide electrodes.

        OFF:
            current behavior, with contacts on the slice drawn as 2D discs.

        ON:
            electrodes remain as 3D actors, no 2D contact discs are drawn,
            and slice planes become semi-transparent.
        """
        self._keep_electrodes_visible_through_slices = not bool(
            getattr(self, "_keep_electrodes_visible_through_slices", False)
        )

        # Remove 2D electrode discs on slices to avoid duplicates.
        try:
            self._remove_actor("coronal_elec")
            self._remove_actor("axial_elec")
            self._remove_actor("sagittal_elec")
        except Exception:
            pass

        try:
            self.update_electrodes()
        except Exception:
            pass

        try:
            self._refresh_multiplanar_clipped_scene()
        except Exception:
            pass

        try:
            self._apply_actor_clipping()
        except Exception:
            pass

        try:
            self._render()
        except Exception:
            pass

    def _get_highres_plane_texture_size(self) -> tuple[int, int]:
        try:
            if self.container_3d is not None:
                w = max(700, int(self.container_3d.width()))
                h = max(700, int(self.container_3d.height()))
                return w, h
        except Exception:
            pass
        return 900, 900

    def _invalidate_slice_volume_cache(
        self,
        base: bool = True,
        pet: bool = True,
        siscom: bool = True,
    ) -> None:
        """
        Invalidate full-volume RGBA caches used by 3D slice planes.

        This must be called when an image, overlay, threshold, colormap,
        mask or parcellation setting changes. Slider navigation itself must
        never invalidate the cache.
        """
        if base:
            self._slice_base_rgba_cache = None
            self._slice_base_cache_ready = False
            self._slice_crop_bounds_xyz = None

        if pet:
            self._slice_pet_rgba_cache = None
            self._slice_pet_cache_ready = False

        if siscom:
            self._slice_siscom_rgba_cache = None
            self._slice_siscom_cache_ready = False

    def _same_sitk_geometry(self, a: sitk.Image | None, b: sitk.Image | None) -> bool:
        if a is None or b is None:
            return False

        try:
            return (
                tuple(a.GetSize()) == tuple(b.GetSize())
                and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-6)
                and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-6)
                and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-6)
            )
        except Exception:
            return False

    def _resample_image_for_slice_cache(
        self,
        img: sitk.Image | None,
        ref_img: sitk.Image | None,
        interpolator,
        default_value: float,
        output_pixel_type,
    ) -> sitk.Image | None:
        """
        Ensure that an image is sampled in the exact grid used by the 3D planes.
        """
        if img is None or ref_img is None:
            return None

        try:
            if self._same_sitk_geometry(img, ref_img):
                return sitk.Cast(img, output_pixel_type)

            return sitk.Resample(
                img,
                ref_img,
                sitk.Transform(),
                interpolator,
                float(default_value),
                output_pixel_type,
            )
        except Exception:
            return None

    def _get_slice_cache_mask_np(self, ref_img: sitk.Image | None) -> np.ndarray | None:
        """
        Return the mask used for slice textures as a boolean NumPy array [z, y, x].
        """
        if ref_img is None:
            return None

        try:
            mask_img = self._get_3d_plane_mask_for_slices(ref_img)

            if mask_img is None:
                size = ref_img.GetSize()  # x, y, z
                return np.ones((int(size[2]), int(size[1]), int(size[0])), dtype=bool)

            mask_img = self._resample_image_for_slice_cache(
                mask_img,
                ref_img,
                sitk.sitkNearestNeighbor,
                0.0,
                sitk.sitkUInt8,
            )

            if mask_img is None:
                return None

            mask_np = sitk.GetArrayFromImage(mask_img).astype(np.uint8)
            mask_np = mask_np > 0

            # Keep the same light dilation principle as your current slice display.
            try:
                from scipy.ndimage import binary_dilation

                mask_np = binary_dilation(mask_np, iterations=1)
            except Exception:
                pass

            return mask_np

        except Exception:
            return None

    def _get_slice_crop_bounds_xyz(
        self,
        ref_img: sitk.Image | None = None,
        padding_vox: int = 4,
    ) -> tuple[int, int, int, int, int, int]:
        """
        Return one stable brain-centered crop bounding box for all displayed
        coronal, axial and sagittal 3D slice planes.

        The crop is computed from the slice-display brain mask and cached so
        moving a slider remains smooth and the plane size does not change
        from one slice to the next.

        Returns:
            (x0, x1, y0, y1, z0, z1)
        """
        if ref_img is None:
            ref_img = self._get_3d_plane_reference_img()

        if ref_img is None:
            return (0, 0, 0, 0, 0, 0)

        size = ref_img.GetSize()  # x, y, z
        x_dim, y_dim, z_dim = int(size[0]), int(size[1]), int(size[2])

        full_bounds = (
            0,
            max(0, x_dim - 1),
            0,
            max(0, y_dim - 1),
            0,
            max(0, z_dim - 1),
        )

        cached = getattr(self, "_slice_crop_bounds_xyz", None)
        if isinstance(cached, tuple) and len(cached) == 6:
            return cached

        try:
            mask_np = self._get_slice_cache_mask_np(ref_img)  # z, y, x

            if mask_np is None or not np.any(mask_np):
                self._slice_crop_bounds_xyz = full_bounds
                return full_bounds

            coords = np.argwhere(mask_np > 0)

            z0, y0, x0 = coords.min(axis=0)
            z1, y1, x1 = coords.max(axis=0)

            pad = max(0, int(padding_vox))

            x0 = max(0, int(x0) - pad)
            x1 = min(x_dim - 1, int(x1) + pad)

            y0 = max(0, int(y0) - pad)
            y1 = min(y_dim - 1, int(y1) + pad)

            z0 = max(0, int(z0) - pad)
            z1 = min(z_dim - 1, int(z1) + pad)

            bounds = (x0, x1, y0, y1, z0, z1)
            self._slice_crop_bounds_xyz = bounds
            return bounds

        except Exception:
            self._slice_crop_bounds_xyz = full_bounds
            return full_bounds

    def _lut_color_for_label(self, lut: dict, label_value: int):
        """
        Return RGB color for a parcellation label.
        Compatible with your existing LUT format: label -> (name, (r, g, b)).
        """
        try:
            entry = lut.get(int(label_value), None)

            if entry is None:
                entry = lut.get(str(int(label_value)), None)

            if entry is None:
                return None

            if isinstance(entry, dict):
                color = entry.get("color") or entry.get("rgb")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                color = entry[1]
            else:
                return None

            if color is None or len(color) < 3:
                return None

            return np.asarray(
                [int(color[0]), int(color[1]), int(color[2])],
                dtype=np.float32,
            )

        except Exception:
            return None

    def _build_slice_base_rgba_cache(self) -> np.ndarray | None:
        """
        Build the complete anatomical slice-display volume:

            active MRI source (T1 or T2 or MNI T1)
            + active parcellation overlay

        PET and SISCOM are intentionally NOT included here because they are
        rendered as independent transparent actors in your current architecture.
        """
        ref_img = self._get_3d_plane_reference_img()
        anat_img = self._get_active_mri_for_3d()

        if ref_img is None or anat_img is None:
            return None

        anat_img = self._resample_image_for_slice_cache(
            anat_img,
            ref_img,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )

        if anat_img is None:
            return None

        try:
            anat_np = sitk.GetArrayFromImage(anat_img).astype(np.float32)
        except Exception:
            return None

        mask_np = self._get_slice_cache_mask_np(ref_img)
        if mask_np is None:
            mask_np = np.ones(anat_np.shape, dtype=bool)

        valid = np.isfinite(anat_np) & mask_np

        vals = anat_np[valid]
        if vals.size == 0:
            vals = anat_np[np.isfinite(anat_np)]

        if vals.size == 0:
            return None

        vmin = float(np.percentile(vals, 2))
        vmax = float(np.percentile(vals, 98))

        if vmax <= vmin:
            vmax = vmin + 1.0

        normalized = np.clip((anat_np - vmin) / (vmax - vmin), 0.0, 1.0)
        gray = (np.nan_to_num(normalized, nan=0.0) * 255.0).astype(np.uint8)

        rgba = np.zeros(anat_np.shape + (4,), dtype=np.uint8)
        rgba[..., 0] = gray
        rgba[..., 1] = gray
        rgba[..., 2] = gray
        rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)

        # --------------------------------------------------------------
        # Active parcellation overlay
        # --------------------------------------------------------------
        try:
            parcel_img, parcel_lut = self._get_active_parcellation_img_and_lut()

            if parcel_img is not None and isinstance(parcel_lut, dict):
                parcel_img = self._resample_image_for_slice_cache(
                    parcel_img,
                    ref_img,
                    sitk.sitkNearestNeighbor,
                    0.0,
                    sitk.sitkInt32,
                )

                if parcel_img is not None:
                    labels = sitk.GetArrayFromImage(parcel_img).astype(np.int32)
                    parcel_valid = (labels > 0) & mask_np

                    if self.chk_parcel1 is not None and self.chk_parcel1.isChecked():
                        opacity_pct = (
                            int(self.spn_parcel1_opacity.value())
                            if self.spn_parcel1_opacity is not None
                            else 50
                        )
                    else:
                        opacity_pct = (
                            int(self.spn_parcel2_opacity.value())
                            if self.spn_parcel2_opacity is not None
                            else 50
                        )

                    alpha = float(np.clip(opacity_pct, 0, 100)) / 100.0

                    for lab in np.unique(labels[parcel_valid]):
                        rgb = self._lut_color_for_label(parcel_lut, int(lab))
                        if rgb is None:
                            continue

                        m = labels == int(lab)

                        for channel in range(3):
                            rgba[..., channel][m] = (
                                (1.0 - alpha) * rgba[..., channel][m].astype(np.float32)
                                + alpha * float(rgb[channel])
                            ).astype(np.uint8)

                        rgba[..., 3][m] = 255

        except Exception:
            pass

        return np.ascontiguousarray(rgba)

    def _build_slice_pet_rgba_cache(self) -> np.ndarray | None:
        """
        Build a full transparent PET overlay volume in the 3D plane reference grid.
        Actor opacity remains controlled separately by the PET opacity slider.
        """
        if self.chk_pet is None or not self.chk_pet.isChecked():
            return None

        ref_img = self._get_3d_plane_reference_img()
        if ref_img is None or self._pet_img is None:
            return None

        pet_img = self._resample_image_for_slice_cache(
            self._pet_img,
            ref_img,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )

        if pet_img is None:
            return None

        try:
            pet_np = sitk.GetArrayFromImage(pet_img).astype(np.float32)
        except Exception:
            return None

        mask_np = self._get_slice_cache_mask_np(ref_img)
        if mask_np is None:
            mask_np = np.ones(pet_np.shape, dtype=bool)

        pet_valid = np.isfinite(pet_np) & (pet_np > 0) & mask_np
        pet_vals = pet_np[pet_valid]

        if pet_vals.size == 0:
            return None

        pmin, pmax = self._get_pet_minmax_percentiles()
        lo, hi = get_pet_window(pet_vals, pmin, pmax)
        gamma = self._get_pet_gamma_value()

        pet_norm = normalize_pet_slice(
            pet_np,
            lo,
            hi,
            gamma=gamma,
            mask=mask_np.astype(np.uint8),
        )

        pet_rgb = pet_norm_to_colormap(pet_norm, self._pet_colormap_name)

        rgba = np.zeros(pet_np.shape + (4,), dtype=np.uint8)
        rgba = blend_pet_on_rgba(
            rgba,
            pet_rgb,
            pet_norm,
            1.0,
        )

        return np.ascontiguousarray(rgba)

    def _build_slice_siscom_rgba_cache(self) -> np.ndarray | None:
        """
        Build a full transparent SISCOM overlay volume in the 3D plane reference grid.
        Actor opacity remains controlled separately by the SISCOM opacity slider.
        """
        if self.chk_siscom is None or not self.chk_siscom.isChecked():
            return None

        ref_img = self._get_3d_plane_reference_img()
        if ref_img is None or self._siscom_img is None:
            return None

        sis_img = self._resample_image_for_slice_cache(
            self._siscom_img,
            ref_img,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )

        if sis_img is None:
            return None

        try:
            sis_np = sitk.GetArrayFromImage(sis_img).astype(np.float32)
        except Exception:
            return None

        mask_np = self._get_slice_cache_mask_np(ref_img)
        if mask_np is None:
            mask_np = np.ones(sis_np.shape, dtype=bool)

        zthr = float(self.dsb_siscom_z.value()) if self.dsb_siscom_z is not None else 2.0

        sis_valid = np.isfinite(sis_np) & (sis_np >= zthr) & mask_np

        if not np.any(sis_valid):
            return None

        zmax = self._siscom_fixed_zmax
        if zmax is None or not np.isfinite(zmax) or zmax <= zthr:
            zmax = zthr + 1.0

        lo, hi = get_siscom_window(sis_np[sis_valid], zthr, zmax)

        sis_norm = normalize_siscom_slice(
            sis_np,
            lo=lo,
            hi=hi,
            gamma=1.0,
            mask=sis_valid.astype(np.uint8),
        )

        sis_rgb = siscom_norm_to_colormap(
            sis_norm,
            self._siscom_colormap_name,
        )

        sis_alpha = np.clip(
            1.8 * np.sqrt(np.clip(sis_norm, 0.0, 1.0)),
            0.0,
            1.0,
        )

        rgba = np.zeros(sis_np.shape + (4,), dtype=np.uint8)
        rgba = blend_siscom_on_rgba(
            rgba,
            sis_rgb,
            sis_alpha,
            alpha_scale=1.0,
        )

        return np.ascontiguousarray(rgba)

    def _get_or_build_slice_base_rgba_cache(self) -> np.ndarray | None:
        if self._slice_base_rgba_cache is None:
            self._slice_base_rgba_cache = self._build_slice_base_rgba_cache()
            self._slice_base_cache_ready = self._slice_base_rgba_cache is not None

        return self._slice_base_rgba_cache

    def _get_or_build_slice_pet_rgba_cache(self) -> np.ndarray | None:
        if self._slice_pet_rgba_cache is None:
            self._slice_pet_rgba_cache = self._build_slice_pet_rgba_cache()
            self._slice_pet_cache_ready = self._slice_pet_rgba_cache is not None

        return self._slice_pet_rgba_cache

    def _get_or_build_slice_siscom_rgba_cache(self) -> np.ndarray | None:
        if self._slice_siscom_rgba_cache is None:
            self._slice_siscom_rgba_cache = self._build_slice_siscom_rgba_cache()
            self._slice_siscom_cache_ready = self._slice_siscom_rgba_cache is not None

        return self._slice_siscom_rgba_cache

    def _extract_rgba_slice_from_volume_cache(
        self,
        rgba_volume: np.ndarray | None,
        geom: dict,
    ) -> np.ndarray | None:
        """
        Extract the requested cropped RGBA slice from a cached RGBA volume.

        Cache order:
            [z, y, x, rgba]

        The extracted texture must use the same crop bounds as the 3D plane
        geometry; otherwise the full slice would be compressed onto a cropped
        rectangle.
        """
        if rgba_volume is None or geom is None:
            return None

        try:
            bounds = geom.get("crop_bounds_xyz", None)

            if not isinstance(bounds, tuple) or len(bounds) != 6:
                # Compatibility fallback if an older geometry dictionary is used.
                if "y_idx" in geom:
                    idx = int(geom["y_idx"])
                    return np.ascontiguousarray(rgba_volume[:, idx, :, :])

                if "z_idx" in geom:
                    idx = int(geom["z_idx"])
                    return np.ascontiguousarray(rgba_volume[idx, :, :, :])

                if "x_idx" in geom:
                    idx = int(geom["x_idx"])
                    return np.ascontiguousarray(rgba_volume[:, :, idx, :])

                return None

            x0, x1, y0, y1, z0, z1 = [int(v) for v in bounds]

            if "y_idx" in geom:
                # Coronal plane: texture axes are z, x.
                idx = int(geom["y_idx"])
                return np.ascontiguousarray(rgba_volume[z0 : z1 + 1, idx, x0 : x1 + 1, :])

            if "z_idx" in geom:
                # Axial plane: texture axes are y, x.
                idx = int(geom["z_idx"])
                return np.ascontiguousarray(rgba_volume[idx, y0 : y1 + 1, x0 : x1 + 1, :])

            if "x_idx" in geom:
                # Sagittal plane: texture axes are z, y.
                idx = int(geom["x_idx"])
                return np.ascontiguousarray(rgba_volume[z0 : z1 + 1, y0 : y1 + 1, idx, :])

        except Exception:
            return None

        return None

    def _refresh_base_slice_cache_and_scene(self) -> None:
        """
        Rebuild the anatomical/parcellation cache and immediately redraw
        the currently visible MNI or native planes.
        """
        self._invalidate_slice_volume_cache(
            base=True,
            pet=False,
            siscom=False,
        )

        try:
            if self._mni_t1_slices_are_visible():
                self._refresh_all_visible_slice_planes_full()
            else:
                self._refresh_multiplanar_clipped_scene()
        except Exception:
            pass

        try:
            self._render()
        except Exception:
            pass

    def _sample_image_on_plane_ras(
        self,
        img: sitk.Image | None,
        plane_center_ras: np.ndarray,
        axis_u_ras: np.ndarray,
        axis_v_ras: np.ndarray,
        width_mm: float,
        height_mm: float,
        out_w: int,
        out_h: int,
        order: int = 1,
        cval: float = np.nan,
    ):
        if img is None:
            return None, None

        try:
            vol = sitk.GetArrayFromImage(img).astype(np.float32)  # z, y, x
            origin = np.asarray(img.GetOrigin(), dtype=np.float64)
            spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
            direction = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
            inv_direction = np.linalg.inv(direction)
        except Exception:
            return None, None

        center_ras = np.asarray(plane_center_ras, dtype=np.float64)
        center_lps = center_ras.copy()
        center_lps[0] *= -1.0
        center_lps[1] *= -1.0

        u_ras = np.asarray(axis_u_ras, dtype=np.float64)
        v_ras = np.asarray(axis_v_ras, dtype=np.float64)
        nu = np.linalg.norm(u_ras)
        nv = np.linalg.norm(v_ras)
        if nu <= 0 or nv <= 0:
            return None, None
        u_ras = u_ras / nu
        v_ras = v_ras / nv

        u_lps = u_ras.copy()
        v_lps = v_ras.copy()
        u_lps[0] *= -1.0
        u_lps[1] *= -1.0
        v_lps[0] *= -1.0
        v_lps[1] *= -1.0

        s_vals = np.linspace(
            -0.5 * float(height_mm), 0.5 * float(height_mm), int(out_h), dtype=np.float64
        )
        t_vals = np.linspace(
            -0.5 * float(width_mm), 0.5 * float(width_mm), int(out_w), dtype=np.float64
        )
        S, T = np.meshgrid(s_vals, t_vals, indexing="ij")

        pts = (
            center_lps[None, None, :]
            + S[..., None] * v_lps[None, None, :]
            + T[..., None] * u_lps[None, None, :]
        )

        pts_flat = pts.reshape(-1, 3)
        rel = pts_flat - origin[None, :]
        idx_xyz = (rel @ inv_direction.T) / spacing[None, :]

        x = idx_xyz[:, 0]
        y = idx_xyz[:, 1]
        z = idx_xyz[:, 2]

        size = np.asarray(img.GetSize(), dtype=np.float64)
        inside = (
            (x >= -0.5)
            & (x <= size[0] - 0.5)
            & (y >= -0.5)
            & (y <= size[1] - 0.5)
            & (z >= -0.5)
            & (z <= size[2] - 0.5)
        )

        arr = np.full((int(out_h) * int(out_w),), cval, dtype=np.float32)
        valid = np.zeros((int(out_h) * int(out_w),), dtype=np.uint8)

        if np.any(inside):
            coords = np.vstack([z[inside], y[inside], x[inside]])
            sampled = map_coordinates(
                vol,
                coords,
                order=int(order),
                mode="constant",
                cval=float(cval) if np.isfinite(cval) else np.nan,
            )
            arr[inside] = sampled.astype(np.float32)
            valid[inside] = np.isfinite(sampled).astype(np.uint8)

        return arr.reshape(int(out_h), int(out_w)), valid.reshape(int(out_h), int(out_w))

    def _build_plane_rgba_highres(self, geom: dict):
        """
        Return the anatomical/parcellation texture for one displayed plane
        by extracting it directly from the cached full RGBA volume.

        Despite the historical function name, the returned texture is now
        stored at native anatomical resolution for instantaneous navigation.
        """
        cache = self._get_or_build_slice_base_rgba_cache()
        return self._extract_rgba_slice_from_volume_cache(cache, geom)

    def _build_coronal_plane_rgba_highres(self, geom: dict):
        return self._build_plane_rgba_highres(geom)

    def _render_pet(self) -> None:
        # PET is no longer rendered as a 3D mesh.
        # It remains available only as an overlay inside the anatomical slice textures.
        self._remove_actor("pet")
        self._apply_actor_clipping()
        try:
            self._render_surface_projections()
        except Exception:
            pass
        self._render()

    def _update_pet_opacity(self) -> None:
        if self._pet_actor is None or self.plotter is None:
            return
        try:
            op = (
                (self.sld_pet_opacity.value() / 100.0) if self.sld_pet_opacity is not None else 0.55
            )
            self._pet_actor.GetProperty().SetOpacity(float(op))
            self._render()
        except Exception:
            pass

    def _render_siscom(self) -> None:
        if not _PV_OK or self.plotter is None:
            return

        show = bool(self.chk_siscom.isChecked()) if self.chk_siscom is not None else False
        if not show or self._siscom_img is None:
            self._remove_actor("siscom")
            self._remove_siscom_scalar_bar()
            self._render()
            return

        if self._any_slice_plane_visible():
            self._remove_actor("siscom")
            self._apply_actor_clipping()
            self._render()
            return

        try:
            siscom_img = self._siscom_img

            # Restrict SISCOM to active surface mask (pial if selected, else brainmask)
            active_mask = self._get_active_surface_mask_resampled_to(siscom_img)
            siscom_np = sitk.GetArrayFromImage(siscom_img).astype(np.float32)

            if active_mask is not None:
                mask_np = sitk.GetArrayFromImage(active_mask).astype(np.uint8)
                siscom_masked_np = np.zeros_like(siscom_np, dtype=np.float32)
                siscom_masked_np[mask_np > 0] = siscom_np[mask_np > 0]
            else:
                siscom_masked_np = siscom_np

            zthr = float(self.dsb_siscom_z.value()) if self.dsb_siscom_z is not None else 2.0

            # Keep only positive hyperperfusion-like values above z-threshold
            siscom_masked_np[siscom_masked_np < zthr] = 0.0

            siscom_masked_img = sitk.GetImageFromArray(siscom_masked_np)
            siscom_masked_img.CopyInformation(siscom_img)

            mesh = self._threshold_image_to_polydata(
                siscom_masked_img,
                absolute_threshold=zthr,
            )

            if mesh is None or mesh.n_points == 0:
                self._remove_actor("siscom")
                self._render()
                return

            self._remove_actor("siscom")
            opacity = (
                (self.sld_siscom_opacity.value() / 100.0)
                if self.sld_siscom_opacity is not None
                else 0.7
            )

            try:
                vals = self._sample_sitk_values_at_ras_points(
                    siscom_masked_img,
                    np.asarray(mesh.points, dtype=np.float32),
                )
            except Exception:
                vals = None

            if vals is None or vals.size == 0:
                vals = np.full((mesh.n_points,), zthr, dtype=np.float32)

            vals = np.nan_to_num(vals, nan=zthr, posinf=zthr, neginf=zthr)
            vals[vals < zthr] = zthr

            zmax = self._siscom_fixed_zmax
            if zmax is None or not np.isfinite(zmax) or zmax <= zthr:
                zmax = zthr + 1.0

            mesh["SISCOM_Z"] = vals.astype(np.float32)

            self._siscom_actor = self.plotter.add_mesh(
                mesh,
                scalars="SISCOM_Z",
                cmap=self._siscom_colormap_name,
                clim=[float(zthr), float(zmax)],
                opacity=float(opacity),
                smooth_shading=True,
                show_scalar_bar=False,
            )

            self._update_siscom_scalar_bar()
            self._apply_actor_clipping()
            self._render()

        except Exception:
            self._remove_actor("siscom")
            self._apply_actor_clipping()
            self._render()

    def _update_siscom_opacity(self) -> None:
        if self._siscom_actor is None or self.plotter is None:
            return
        try:
            op = (
                (self.sld_siscom_opacity.value() / 100.0)
                if self.sld_siscom_opacity is not None
                else 0.7
            )
            self._siscom_actor.GetProperty().SetOpacity(float(op))
            self._render()
        except Exception:
            pass

    def _get_active_surface_polydata(self):
        if not _PV_OK:
            return None

        try:
            if self.chk_pial is not None and self.chk_pial.isChecked():
                polys = []
                if self._show_lh_pial and self._lh_pial_poly is not None:
                    polys.append(self._lh_pial_poly.copy())
                if self._show_rh_pial and self._rh_pial_poly is not None:
                    polys.append(self._rh_pial_poly.copy())
                if polys:
                    surf = polys[0]
                    for p in polys[1:]:
                        try:
                            surf = surf.merge(p)
                        except Exception:
                            pass
                    try:
                        surf = surf.clean(tolerance=1e-6)
                    except Exception:
                        pass
                    try:
                        surf = surf.compute_normals(
                            cell_normals=False,
                            point_normals=True,
                            auto_orient_normals=True,
                            consistent_normals=True,
                            split_vertices=False,
                            inplace=False,
                        )
                    except Exception:
                        pass
                    return surf
        except Exception:
            pass

        try:
            if self._brainmask_img is not None:
                surf = self._binarymask_to_polydata(self._brainmask_img)
                if surf is not None and getattr(surf, "n_points", 0) > 0:
                    try:
                        surf = surf.compute_normals(
                            cell_normals=False,
                            point_normals=True,
                            auto_orient_normals=True,
                            consistent_normals=True,
                            split_vertices=False,
                            inplace=False,
                        )
                    except Exception:
                        pass
                    return surf
        except Exception:
            pass

        return None

    def _build_surface_cross_mesh(
        self, center_ras: np.ndarray, normal_ras: np.ndarray, size_mm: float = 4.0
    ):
        try:
            c = np.asarray(center_ras, dtype=np.float64)
            n = np.asarray(normal_ras, dtype=np.float64)
            nn = np.linalg.norm(n)
            if nn < 1e-6:
                n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                n = n / nn

            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(np.dot(ref, n)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

            t1 = np.cross(n, ref)
            nt1 = np.linalg.norm(t1)
            if nt1 < 1e-6:
                t1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                nt1 = np.linalg.norm(t1)
            t1 = t1 / nt1
            t2 = np.cross(n, t1)
            t2 = t2 / max(np.linalg.norm(t2), 1e-6)

            h = 0.5 * float(size_mm)
            a0, a1 = c - h * t1, c + h * t1
            b0, b1 = c - h * t2, c + h * t2
            line1 = pv.Line(a0, a1)
            line2 = pv.Line(b0, b1)
            try:
                return line1.merge(line2)
            except Exception:
                return line1
        except Exception:
            return None

    def has_surface_projection(self, elec_id: int) -> bool:
        try:
            return int(elec_id) in getattr(self, "_surface_projection_defs", {})
        except Exception:
            return False

    def remove_electrode_surface_projection(self, electrode) -> None:
        if electrode is None:
            return
        name = str(electrode.get("name", "E")).strip() or "E"
        self._remove_surface_projection_actors(name)
        electrode["surface_projection_enabled"] = False
        electrode.pop("surface_projection_point_ras", None)
        electrode.pop("surface_projection_surface", None)
        self._render()

    def _render_surface_projections(self) -> None:
        """
        Compatibility wrapper for old calls.

        Surface projection rendering must go through render_all_surface_projections(),
        because project_electrode_on_surface() now expects an electrode id, not an
        electrode dict.
        """
        self.render_all_surface_projections()

    def _open_export_coordinates_dialog(self) -> None:
        parent = self._dialog_parent()

        try:
            dlg = ExportCoordinatesDialog(self.state, parent=parent)
            dlg.exec()

        except Exception as e:
            NeuXelecMessageDialog.critical(
                parent,
                "Export coordinates failed",
                ("The export coordinates window could not be opened.\n\n" f"Details:\n{e}"),
            )

    def _get_local_contact_labels_visible(self, elec_id: int, n_contacts: int):
        vals = self._page_contact_labels_visible.get(int(elec_id))
        if not isinstance(vals, list) or len(vals) != int(n_contacts):
            vals = [False] * int(n_contacts)
            self._page_contact_labels_visible[int(elec_id)] = vals
        return vals

    def set_labels_visible(self, elec_id: int, visible: bool) -> None:
        try:
            elec = self.state.electrodes[int(elec_id)]
            n = len(elec.get("contacts_lps", []) or [])
        except Exception:
            return
        self._page_contact_labels_visible[int(elec_id)] = [bool(visible)] * n

        if not bool(getattr(self, "_suspend_electrode_refresh", False)):
            self._render_single_electrode(int(elec_id))

    def set_contact_label_visible(self, elec_id: int, contact_idx: int, visible: bool) -> None:
        try:
            elec = self.state.electrodes[int(elec_id)]
            n = len(elec.get("contacts_lps", []) or [])
        except Exception:
            return
        vals = self._get_local_contact_labels_visible(int(elec_id), n)
        if 0 <= int(contact_idx) < len(vals):
            vals[int(contact_idx)] = bool(visible)

        if not bool(getattr(self, "_suspend_electrode_refresh", False)):
            self._render_single_electrode(int(elec_id))

    def _mni_mode_is_active(self) -> bool:
        try:
            return bool(self.chk_mni_atlas is not None and self.chk_mni_atlas.isChecked())
        except Exception:
            return False

    def set_active_page(self, active: bool) -> None:
        self._is_active_page = bool(active)

        if not self._is_active_page:
            if self._loading_overlay is not None:
                self._loading_overlay.cancel()
            return

        # Le voile est posé avant le premier rendu visible du viewport.
        if self._loading_overlay is not None:
            self._loading_overlay.begin("Preparing 3D scene")

        QTimer.singleShot(0, self._activate_3d_electrodes_step)
        self._schedule_fullscreen_button_geometry_update()

    def _activate_3d_electrodes_step(self) -> None:
        if not self._is_active_page:
            return

        if self._loading_overlay is not None:
            self._loading_overlay.set_progress(
                0.22,
                "Rendering electrodes",
            )

        try:
            if self._mni_mode_is_active():
                # We are in MNI mode: never redraw native patient electrodes.
                self._clear_native_scene_for_mni_mode()
                self._render_mni_scene(reset_camera=False)
            else:
                self.update_electrodes()
        except Exception:
            pass

        try:
            QTimer.singleShot(0, self._update_quick_tools_geometry)
            QTimer.singleShot(80, self._update_quick_tools_geometry)
        except Exception:
            pass

        if self._loading_overlay is not None:
            self._loading_overlay.set_progress(
                0.48,
                "Rendering anatomy and overlays",
            )

        QTimer.singleShot(0, self._activate_3d_scene_step)

    def _activate_3d_scene_step(self) -> None:
        if not self._is_active_page:
            return

        # MNI mode: do not render native markers/projections/electrodes.
        if self._mni_mode_is_active():
            try:
                self._clear_native_scene_for_mni_mode()
            except Exception:
                pass

            try:
                self._render_mni_scene(reset_camera=False)
            except Exception:
                pass

            if self._loading_overlay is not None:
                self._loading_overlay.set_progress(
                    0.82,
                    "Finalizing MNI scene",
                )

            QTimer.singleShot(0, self._activate_3d_finish_step)
            return

        try:
            self._render_anatomical_markers()
        except Exception:
            pass

        try:
            self._refresh_multiplanar_clipped_scene()
        except Exception:
            pass

        try:
            self._render_surface_projections()
        except Exception:
            pass

        if self._loading_overlay is not None:
            self._loading_overlay.set_progress(
                0.82,
                "Finalizing camera",
            )

        QTimer.singleShot(0, self._activate_3d_finish_step)

    def _activate_3d_finish_step(self) -> None:
        if not self._is_active_page:
            return

        try:
            self._render()
        except Exception:
            pass

        try:
            if not bool(getattr(self, "_saved_camera_applied_once", False)):
                camera = getattr(self.state, "view3d_saved_camera", None)

                if camera is not None and self._apply_camera_dict(camera):
                    self._saved_camera_applied_once = True
        except Exception:
            pass

        if self._loading_overlay is not None:
            self._loading_overlay.set_progress(
                0.94,
                "Ready",
            )
            self._loading_overlay.complete()

    def update_electrodes(self) -> None:
        """Render electrodes contacts as 3D points and optional shaft + labels."""
        if not _PV_OK or self.plotter is None:
            return
        # Never draw native patient electrodes while MNI atlas is active.
        if self._mni_mode_is_active():
            try:
                self._remove_actor("electrodes")
            except Exception:
                pass

            try:
                for _, a in list(getattr(self, "_elec_label_actors", {}).items()):
                    try:
                        self.plotter.remove_actor(a, reset_camera=False)
                    except Exception:
                        pass
                self._elec_label_actors.clear()
            except Exception:
                pass

            try:
                self._remove_all_surface_projection_actors()
            except Exception:
                pass

            try:
                self._render()
            except Exception:
                pass

            return
        # Remove previous electrode actors
        self._remove_actor("electrodes")

        # Remove previous label actors
        try:
            for _, a in list(getattr(self, "_elec_label_actors", {}).items()):
                try:
                    self.plotter.remove_actor(a, reset_camera=False)
                except Exception:
                    pass
            self._elec_label_actors.clear()
        except Exception:
            pass

        try:
            electrodes = getattr(self.state, "electrodes", None)
            if not electrodes:
                self._apply_actor_clipping()
                self._render()
                return

            show_shaft = True
            if self.btn_elec_shaft is not None:
                show_shaft = bool(self.btn_elec_shaft.isChecked())

            point_size = 8.0
            if self.spin_contacts_size is not None:
                try:
                    point_size = float(self.spin_contacts_size.value())
                except Exception:
                    point_size = 8.0

            for elec_id, elec in enumerate(electrodes):
                if not self._get_local_electrode_visible(elec_id):
                    continue

                rgb = tuple(elec.get("color", (0, 255, 0)))
                color = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

                contacts_lps = elec.get("contacts_lps", []) or []
                contacts_idx = elec.get("contacts_idx", []) or []
                contacts_visible = self._get_local_contacts_visible(elec_id, len(contacts_lps))

                # ---------- points / shaft ----------
                pts = []
                kept_indices = []
                for ci, p in enumerate(contacts_lps):
                    if not bool(contacts_visible[ci]):
                        continue

                    idx_xyz = contacts_idx[ci] if ci < len(contacts_idx) else None
                    if self._contact_is_on_any_visible_slice(idx_xyz, tol=0.49):
                        # This contact will be represented as a 2D disc on the slice,
                        # so do not also draw the 3D sphere here.
                        continue

                    try:
                        pts.append([float(p[0]), float(p[1]), float(p[2])])
                        kept_indices.append(ci)
                    except Exception:
                        continue

                if pts:
                    pts_arr = _lps_to_ras_points(np.array(pts, dtype=np.float32))

                    poly = pv.PolyData(pts_arr)
                    actor = self.plotter.add_points(
                        poly,
                        color=color,
                        point_size=float(point_size),
                        render_points_as_spheres=True,
                    )
                    try:
                        actor.PickableOff()
                    except Exception:
                        try:
                            actor.SetPickable(False)
                        except Exception:
                            pass

                    self._apply_electrode_actor_depth_mode(actor)
                    self._elec_actors[(elec_id, "points")] = actor

                    if show_shaft and pts_arr.shape[0] >= 2:
                        try:
                            line = pv.lines_from_points(pts_arr, close=False)
                            line_actor = self.plotter.add_mesh(
                                line,
                                color=color,
                                line_width=3,
                            )
                            try:
                                line_actor.PickableOff()
                            except Exception:
                                try:
                                    line_actor.SetPickable(False)
                                except Exception:
                                    pass

                            self._apply_electrode_actor_depth_mode(line_actor)
                            self._elec_actors[(elec_id, "line")] = line_actor
                        except Exception:
                            pass

                # ---------- labels ----------
                try:
                    contact_labels_visible = self._get_local_contact_labels_visible(
                        elec_id, len(contacts_lps)
                    )

                    label_pts = []
                    label_txt = []

                    for ci, p in enumerate(contacts_lps):
                        if not bool(contacts_visible[ci]):
                            continue
                        if not bool(contact_labels_visible[ci]):
                            continue

                        ras = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float32)
                        ras[0] *= -1.0
                        ras[1] *= -1.0

                        # small top-right offset in display space approximation
                        label_pos = ras.copy()
                        label_pos[0] += 2.0  # right
                        label_pos[2] += 2.0  # up

                        label_pts.append(label_pos)
                        label_txt.append(f"{elec.get('name', 'E')}{ci + 1}")

                    if label_pts:
                        pts_np = np.asarray(label_pts, dtype=np.float32)

                        label_actor = self.plotter.add_point_labels(
                            pts_np,
                            label_txt,
                            font_size=12,
                            text_color=color,
                            shape_opacity=0.0,
                            show_points=False,
                            always_visible=True,
                        )

                        try:
                            label_actor.PickableOff()
                        except Exception:
                            try:
                                label_actor.SetPickable(False)
                            except Exception:
                                pass

                        self._elec_label_actors[(elec_id, "labels")] = label_actor
                except Exception:
                    pass

        except Exception:
            pass

        self._apply_actor_clipping()
        self.render_all_surface_projections()
        self._render()

    def _remove_actor(self, which: str) -> None:
        if self.plotter is None:
            return
        if which == "axial_pet" and getattr(self, "_axial_pet_actor", None) is not None:
            try:
                self.plotter.remove_actor(self._axial_pet_actor, reset_camera=False)
            except Exception:
                pass
            self._axial_pet_actor = None

        if which == "axial_siscom" and getattr(self, "_axial_siscom_actor", None) is not None:
            try:
                self.plotter.remove_actor(self._axial_siscom_actor, reset_camera=False)
            except Exception:
                pass
            self._axial_siscom_actor = None

        if which == "sagittal_pet" and getattr(self, "_sagittal_pet_actor", None) is not None:
            try:
                self.plotter.remove_actor(self._sagittal_pet_actor, reset_camera=False)
            except Exception:
                pass
            self._sagittal_pet_actor = None

        if which == "sagittal_siscom" and getattr(self, "_sagittal_siscom_actor", None) is not None:
            try:
                self.plotter.remove_actor(self._sagittal_siscom_actor, reset_camera=False)
            except Exception:
                pass
            self._sagittal_siscom_actor = None
        if which == "electrodes":
            try:
                for _, a in list(getattr(self, "_elec_actors", {}).items()):
                    try:
                        self.plotter.remove_actor(a, reset_camera=False)
                    except Exception:
                        pass
                if hasattr(self, "_elec_actors"):
                    self._elec_actors.clear()
            except Exception:
                pass
            return
        if which == "brain" and self._brain_actor is not None:
            try:
                self.plotter.remove_actor(self._brain_actor, reset_camera=False)
            except Exception:
                pass
            self._brain_actor = None
        if which == "pet" and self._pet_actor is not None:
            try:
                self.plotter.remove_actor(self._pet_actor, reset_camera=False)
            except Exception:
                pass
            self._pet_actor = None
        if which == "siscom" and self._siscom_actor is not None:
            try:
                self.plotter.remove_actor(self._siscom_actor, reset_camera=False)
            except Exception:
                pass
            self._siscom_actor = None
        if which == "ct" and self._ct_actor is not None:
            try:
                self.plotter.remove_actor(self._ct_actor, reset_camera=False)
            except Exception:
                pass
            self._ct_actor = None
        if which == "coronal_plane" and getattr(self, "_coronal_plane_actor", None) is not None:
            try:
                self.plotter.remove_actor(self._coronal_plane_actor, reset_camera=False)
            except Exception:
                pass
            self._coronal_plane_actor = None
        if which == "coronal_pet" and getattr(self, "_coronal_pet_actor", None) is not None:
            try:
                self.plotter.remove_actor(self._coronal_pet_actor, reset_camera=False)
            except Exception:
                pass
            self._coronal_pet_actor = None

        if which == "coronal_siscom" and getattr(self, "_coronal_siscom_actor", None) is not None:
            try:
                self.plotter.remove_actor(self._coronal_siscom_actor, reset_camera=False)
            except Exception:
                pass
            self._coronal_siscom_actor = None

        if which == "coronal_elec" and getattr(self, "_coronal_elec_actor", None) is not None:
            try:
                actors = self._coronal_elec_actor
                if not isinstance(actors, (list, tuple)):
                    actors = [actors]
                for a in actors:
                    try:
                        self.plotter.remove_actor(a, reset_camera=False)
                    except Exception:
                        pass
            except Exception:
                pass
            self._coronal_elec_actor = None

        if which == "axial_plane" and getattr(self, "_axial_plane_actor", None) is not None:
            try:
                self.plotter.remove_actor(self._axial_plane_actor, reset_camera=False)
            except Exception:
                pass
            self._axial_plane_actor = None

        if which == "sagittal_plane" and getattr(self, "_sagittal_plane_actor", None) is not None:
            try:
                self.plotter.remove_actor(self._sagittal_plane_actor, reset_camera=False)
            except Exception:
                pass
            self._sagittal_plane_actor = None

        if which == "axial_elec" and getattr(self, "_axial_elec_actor", None) is not None:
            try:
                actors = self._axial_elec_actor
                if not isinstance(actors, (list, tuple)):
                    actors = [actors]
                for a in actors:
                    try:
                        self.plotter.remove_actor(a, reset_camera=False)
                    except Exception:
                        pass
            except Exception:
                pass
            self._axial_elec_actor = None

        if which == "sagittal_elec" and getattr(self, "_sagittal_elec_actor", None) is not None:
            try:
                actors = self._sagittal_elec_actor
                if not isinstance(actors, (list, tuple)):
                    actors = [actors]
                for a in actors:
                    try:
                        self.plotter.remove_actor(a, reset_camera=False)
                    except Exception:
                        pass
            except Exception:
                pass
            self._sagittal_elec_actor = None

        if which == "coronal_outline":
            try:
                self.plotter.remove_actor("coronal_outline", reset_camera=False)
            except Exception:
                pass
            try:
                if getattr(self, "_coronal_outline_actor", None) is not None:
                    self.plotter.remove_actor(self._coronal_outline_actor, reset_camera=False)
            except Exception:
                pass
            self._coronal_outline_actor = None

        if which == "axial_outline":
            try:
                self.plotter.remove_actor("axial_outline", reset_camera=False)
            except Exception:
                pass
            try:
                if getattr(self, "_axial_outline_actor", None) is not None:
                    self.plotter.remove_actor(self._axial_outline_actor, reset_camera=False)
            except Exception:
                pass
            self._axial_outline_actor = None

        if which == "sagittal_outline":
            try:
                self.plotter.remove_actor("sagittal_outline", reset_camera=False)
            except Exception:
                pass
            try:
                if getattr(self, "_sagittal_outline_actor", None) is not None:
                    self.plotter.remove_actor(self._sagittal_outline_actor, reset_camera=False)
            except Exception:
                pass
            self._sagittal_outline_actor = None

    def _remove_crosshair_marker(self) -> None:
        if self.plotter is None:
            return
        if getattr(self, "_crosshair_marker_actor", None) is not None:
            try:
                self.plotter.remove_actor(self._crosshair_marker_actor, reset_camera=False)
            except Exception:
                pass
            self._crosshair_marker_actor = None

    def hide_crosshair_marker(self) -> None:
        """
        Public API called from Reconstruction page.

        Hide the visible red marker, but keep self._crosshair_marker_ras in memory.
        This allows:
            - Ctrl+D from Reconstruction to show the marker again at the current crosshair
            - Ctrl+C + left-click/drag in 3D to continue working from the last marker depth
        """
        try:
            self._remove_crosshair_marker()

            self._crosshair_marker_drag_armed = False
            self._crosshair_marker_drag_active = False

            try:
                if self.interactor is not None:
                    self.interactor.unsetCursor()
            except Exception:
                pass

            self._render()
        except Exception:
            pass

    def show_crosshair_marker_lps(self, lps_xyz) -> None:
        if not _PV_OK or self.plotter is None:
            return
        if lps_xyz is None:
            return

        try:
            lps = np.asarray(lps_xyz, dtype=np.float32).reshape(1, 3)
            ras = _lps_to_ras_points(lps)[0]
        except Exception:
            return

        self._set_crosshair_marker_ras(ras, recenter_camera=True)

    def _set_crosshair_marker_ras(self, ras_xyz, recenter_camera: bool = False) -> None:
        if not _PV_OK or self.plotter is None:
            return

        try:
            ras = np.asarray(ras_xyz, dtype=np.float32).reshape(3)
        except Exception:
            return

        self._crosshair_marker_ras = ras.copy()

        self._remove_crosshair_marker()

        try:
            marker = pv.Sphere(radius=2.0, center=tuple(ras.tolist()))
            self._crosshair_marker_actor = self.plotter.add_mesh(
                marker,
                color="red",
                opacity=0.95,
                smooth_shading=True,
                name="crosshair_marker",
            )
            try:
                self._crosshair_marker_actor.PickableOff()
            except Exception:
                pass
        except Exception:
            self._crosshair_marker_actor = None
            return

        if recenter_camera:
            try:
                cam = self.plotter.camera
                cam.focal_point = tuple(ras.tolist())
            except Exception:
                pass

        try:
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass

        self._render()

    def _crosshair_marker_lps(self):
        try:
            ras = np.asarray(self._crosshair_marker_ras, dtype=np.float64).reshape(3)
            lps = ras.copy()
            lps[0] *= -1.0
            lps[1] *= -1.0
            return tuple(float(v) for v in lps)
        except Exception:
            return None

    def _display_xy_to_world_at_marker_depth(self, x_qt: float, y_qt: float):
        """
        Convert mouse position to a RAS point on the camera plane passing through
        the current crosshair marker.
        """
        try:
            if self.plotter is None or self.interactor is None:
                return None
            if self._crosshair_marker_ras is None:
                return None

            renderer = self.plotter.renderer
            h = int(self.interactor.height())

            marker = np.asarray(self._crosshair_marker_ras, dtype=np.float64).reshape(3)

            renderer.SetWorldPoint(float(marker[0]), float(marker[1]), float(marker[2]), 1.0)
            renderer.WorldToDisplay()
            _mx, _my, marker_z = renderer.GetDisplayPoint()

            x_vtk = float(x_qt)
            y_vtk = float(h - y_qt)

            renderer.SetDisplayPoint(x_vtk, y_vtk, float(marker_z))
            renderer.DisplayToWorld()
            wx, wy, wz, ww = renderer.GetWorldPoint()

            if abs(float(ww)) < 1e-8:
                return None

            return np.array([wx / ww, wy / ww, wz / ww], dtype=np.float64)
        except Exception:
            return None

    def _event_xy(self, event):
        try:
            p = event.position()
            return float(p.x()), float(p.y())
        except Exception:
            try:
                p = event.pos()
                return float(p.x()), float(p.y())
            except Exception:
                return None

    def _slice_slider_and_checkbox(self, plane: str):
        """
        Return the slider and checkbox associated with one anatomical slice plane.
        """
        plane = str(plane).lower().strip()

        if plane == "sagittal":
            return self.sld_sagittal_plane, self.chk_sagittal_plane

        if plane == "coronal":
            return self.sld_coronal_plane, self.chk_coronal_plane

        if plane == "axial":
            return self.sld_axial_plane, self.chk_axial_plane

        return None, None

    def _plane_from_keyboard_wheel_arrow(self, key) -> str | None:
        """
        Map held arrow keys to 3D slice planes.

        User-defined controls:
            Ctrl + Left arrow + wheel -> sagittal
            Ctrl + Down arrow + wheel -> coronal
            Ctrl + Up arrow + wheel   -> axial
        """
        if key == Qt.Key_Left:
            return "sagittal"

        if key == Qt.Key_Down:
            return "coronal"

        if key == Qt.Key_Up:
            return "axial"

        return None

    def _handle_slice_slider_keyboard_wheel_event(self, event) -> bool:
        """
        Control 3D slice sliders using Ctrl + held arrow key + mouse wheel.

        Controls:
            Ctrl + Left arrow held + wheel -> sagittal slider
            Ctrl + Down arrow held + wheel -> coronal slider
            Ctrl + Up arrow held + wheel   -> axial slider

        The plane is automatically displayed on first wheel movement if it was
        not already visible.
        """
        try:
            et = event.type()

            # ---------------------------------------------------------
            # 1) Arm one plane while Ctrl + arrow is held.
            # ---------------------------------------------------------
            if et == QEvent.KeyPress:
                plane = self._plane_from_keyboard_wheel_arrow(event.key())

                if plane is not None and bool(event.modifiers() & Qt.ControlModifier):
                    self._slice_wheel_active_plane = plane

                    try:
                        if self.interactor is not None:
                            self.interactor.setCursor(Qt.SizeVerCursor)
                    except Exception:
                        pass

                    # Consume the arrow press so PyVista does not react to it.
                    return True

            # ---------------------------------------------------------
            # 2) Release the mode when arrow or Ctrl is released.
            # ---------------------------------------------------------
            if et == QEvent.KeyRelease:
                released_plane = self._plane_from_keyboard_wheel_arrow(event.key())

                if released_plane is not None:
                    if self._slice_wheel_active_plane == released_plane:
                        self._slice_wheel_active_plane = None

                        try:
                            if self.interactor is not None:
                                self.interactor.unsetCursor()
                        except Exception:
                            pass

                    return True

                if event.key() == Qt.Key_Control:
                    self._slice_wheel_active_plane = None

                    try:
                        if self.interactor is not None:
                            self.interactor.unsetCursor()
                    except Exception:
                        pass

                    return False

            # ---------------------------------------------------------
            # 3) While the mode is armed, use the mouse wheel to move
            #    the corresponding Qt slider.
            # ---------------------------------------------------------
            if et == QEvent.Wheel:
                plane = getattr(self, "_slice_wheel_active_plane", None)

                if plane is None:
                    return False

                # Safety: if Ctrl is no longer held, return to normal wheel use.
                if not bool(event.modifiers() & Qt.ControlModifier):
                    self._slice_wheel_active_plane = None

                    try:
                        if self.interactor is not None:
                            self.interactor.unsetCursor()
                    except Exception:
                        pass

                    return False

                slider, checkbox = self._slice_slider_and_checkbox(plane)

                if slider is None or checkbox is None:
                    return True

                # Make the corresponding plane visible automatically.
                if not checkbox.isChecked():
                    checkbox.setChecked(True)

                # The slider remains disabled if no compatible T1/MNI T1 volume
                # is loaded, so in that situation simply consume the wheel.
                if not slider.isEnabled():
                    return True

                delta = int(event.angleDelta().y())

                if delta == 0:
                    return True

                # Standard mouse wheel: one notch = one slice.
                # This also stays usable with high-resolution wheels.
                direction = 1 if delta > 0 else -1
                notches = max(1, int(abs(delta) / 120))
                step = int(direction * notches)

                slider.setValue(int(slider.value()) + step)

                try:
                    event.accept()
                except Exception:
                    pass

                return True

        except Exception:
            self._slice_wheel_active_plane = None

        return False

    def _handle_crosshair_marker_drag_event(self, event) -> bool:
        """
        Ctrl+C held + left mouse drag = move the red crosshair marker in 3D.
        """
        try:
            et = event.type()

            # Ctrl+C pressed: arm marker dragging
            if et == QEvent.KeyPress:
                if event.key() == Qt.Key_C and bool(event.modifiers() & Qt.ControlModifier):
                    if self._crosshair_marker_ras is not None:
                        self._crosshair_marker_drag_armed = True
                        self._crosshair_marker_drag_active = False
                        try:
                            self.interactor.setCursor(Qt.CrossCursor)
                        except Exception:
                            pass
                        return True

            # C or Ctrl released: stop marker dragging
            if et == QEvent.KeyRelease:
                if event.key() in (Qt.Key_C, Qt.Key_Control):
                    self._crosshair_marker_drag_armed = False
                    self._crosshair_marker_drag_active = False
                    try:
                        self.interactor.unsetCursor()
                    except Exception:
                        pass
                    return False

            # Left press while Ctrl+C mode is armed
            if et == QEvent.MouseButtonPress:
                if self._crosshair_marker_drag_armed and event.button() == Qt.LeftButton:
                    xy = self._event_xy(event)
                    if xy is None:
                        return True
                    world = self._display_xy_to_world_at_marker_depth(*xy)
                    if world is not None:
                        self._crosshair_marker_drag_active = True
                        self._set_crosshair_marker_ras(world, recenter_camera=False)
                    return True

            # Mouse move while dragging
            if et == QEvent.MouseMove:
                if self._crosshair_marker_drag_armed and self._crosshair_marker_drag_active:
                    xy = self._event_xy(event)
                    if xy is None:
                        return True
                    world = self._display_xy_to_world_at_marker_depth(*xy)
                    if world is not None:
                        self._set_crosshair_marker_ras(world, recenter_camera=False)
                    return True

            # Release mouse: keep marker at last position
            if et == QEvent.MouseButtonRelease:
                if self._crosshair_marker_drag_active and event.button() == Qt.LeftButton:
                    self._crosshair_marker_drag_active = False
                    return True

        except Exception:
            pass

        return False

    def _get_current_display_plane_mesh(self, which: str):
        """
        Return the exact mesh currently used as the basis for one slice plane,
        after clipping with the other active planes.
        This is used so the colored outline follows the displayed slice exactly.
        """
        which = str(which).lower().strip()

        geom = None
        source = None

        try:
            if which == "coronal":
                geom = self._build_coronal_plane_geometry()
                source = self._coronal_plane_source_mesh
                if source is None and geom is not None:
                    source = self._build_textured_plane_mesh(geom, "coronal_plane")

            elif which == "axial":
                geom = self._build_axial_plane_geometry()
                source = self._axial_plane_source_mesh
                if source is None and geom is not None:
                    source = self._build_textured_plane_mesh(geom, "axial_plane")

            elif which == "sagittal":
                geom = self._build_sagittal_plane_geometry()
                source = self._sagittal_plane_source_mesh
                if source is None and geom is not None:
                    source = self._build_textured_plane_mesh(geom, "sagittal_plane")
        except Exception:
            return None, None

        if geom is None or source is None:
            return None, None

        try:
            mesh = self._clip_plane_mesh_with_other_planes(which, source.copy())
        except Exception:
            mesh = source.copy()

        if mesh is None or mesh.n_points == 0:
            return None, None

        return mesh, geom

    def _slice_plane_frames_option_available(self) -> bool:
        """
        The context-menu action Add/Remove frame is available only when at least
        one coronal/axial/sagittal slice is currently displayed.
        """
        try:
            return bool(
                (self.chk_coronal_plane is not None and self.chk_coronal_plane.isChecked())
                or (self.chk_axial_plane is not None and self.chk_axial_plane.isChecked())
                or (self.chk_sagittal_plane is not None and self.chk_sagittal_plane.isChecked())
            )
        except Exception:
            return False

    def _remove_all_slice_plane_frame_actors(self) -> None:
        """
        Remove every actor related to the colored slice frames.

        This removes:
        - stored Python actor references
        - named PyVista actors that may still exist in the renderer

        This is necessary because old outline actors can remain visible as
        small white corner points if only the Python reference is removed.
        """
        if self.plotter is None:
            return

        for actor_name, attr_name in (
            ("coronal_outline", "_coronal_outline_actor"),
            ("axial_outline", "_axial_outline_actor"),
            ("sagittal_outline", "_sagittal_outline_actor"),
        ):
            # Remove by stored actor reference.
            try:
                actor = getattr(self, attr_name, None)
                if actor is not None:
                    self.plotter.remove_actor(actor, reset_camera=False)
            except Exception:
                pass

            # Remove by PyVista actor name.
            # This catches old actors that were added with name=actor_name.
            try:
                self.plotter.remove_actor(actor_name, reset_camera=False)
            except Exception:
                pass

            try:
                setattr(self, attr_name, None)
            except Exception:
                pass

    def _toggle_slice_plane_frames_visible(self) -> None:
        """
        Toggle only the colored frames around slices.

        Important:
        This does not remove the anatomical slice actors, PET/SISCOM overlays,
        electrode overlays, clipping, sliders, or any functional slice logic.
        It only removes or redraws:
            - coronal_outline
            - axial_outline
            - sagittal_outline
        """
        self._slice_plane_frames_visible = not bool(
            getattr(self, "_slice_plane_frames_visible", True)
        )

        if bool(getattr(self, "_slice_plane_frames_visible", True)):
            try:
                self._refresh_all_visible_plane_outlines()
            except Exception:
                pass
        else:
            try:
                self._remove_all_slice_plane_frame_actors()
            except Exception:
                pass

        try:
            self._render()
        except Exception:
            pass

    def _render_plane_outline(self, which: str, color: str, offset_mm: float = 0.15) -> None:
        """
        Draw the colored frame from the exact displayed slice mesh,
        so the frame follows the slice position and clipping precisely.

        Important:
        - only the colored frame is rendered here
        - no corner points / vertices should be visible
        - if Add frame / Remove frame is OFF, only the frame is hidden,
        the slice itself stays fully functional
        """
        if not _PV_OK or self.plotter is None:
            return

        actor_name = f"{which}_outline"
        self._remove_actor(actor_name)

        # Right-click menu option: Add frame / Remove frame
        if not bool(getattr(self, "_slice_plane_frames_visible", True)):
            return

        try:
            mesh, geom = self._get_current_display_plane_mesh(which)
            if mesh is None or geom is None or mesh.n_points == 0:
                return

            n = np.asarray(geom["normal"], dtype=np.float64)
            nn = np.linalg.norm(n)
            if nn > 0:
                n = n / nn
            else:
                n = np.array([0.0, 1.0, 0.0], dtype=np.float64)

            # Small offset only to avoid z-fighting with the slice itself
            mesh = mesh.copy()
            pts = np.asarray(mesh.points, dtype=np.float64)
            pts = pts + offset_mm * n
            mesh.points = pts

            poly = mesh.extract_feature_edges(
                boundary_edges=True,
                feature_edges=False,
                manifold_edges=False,
                non_manifold_edges=False,
            )

            if poly is None or poly.n_points == 0:
                return
            try:
                poly = poly.clean(point_merging=True, tolerance=1e-6)
            except Exception:
                pass
            # ------------------------------------------------------------------
            # IMPORTANT:
            # Rebuild a LINE-ONLY polydata so no vertex cells / corner points
            # are rendered at the 4 corners.
            # ------------------------------------------------------------------
            try:
                import pyvista as pv

                if getattr(poly, "lines", None) is not None and poly.lines.size > 0:
                    line_only = pv.PolyData()
                    line_only.points = np.asarray(poly.points, dtype=np.float64)
                    line_only.lines = poly.lines.copy()
                    poly = line_only
            except Exception:
                pass

            actor = self.plotter.add_mesh(
                poly,
                color=color,
                line_width=4,
                opacity=0.65,
                name=actor_name,
                lighting=False,
                show_scalar_bar=False,
                render_lines_as_tubes=False,
                show_vertices=False,
            )

            try:
                prop = actor.GetProperty()
                prop.SetOpacity(0.65)
                prop.SetAmbient(1.0)
                prop.SetDiffuse(0.0)
                prop.SetSpecular(0.0)
                prop.SetLineWidth(4)

                # Avoid any visible corner points / vertices
                try:
                    prop.SetRenderPointsAsSpheres(False)
                except Exception:
                    pass

                try:
                    prop.SetPointSize(1)
                except Exception:
                    pass

                try:
                    prop.SetVertexVisibility(False)
                except Exception:
                    pass

            except Exception:
                pass

            if which == "coronal":
                self._coronal_outline_actor = actor
            elif which == "axial":
                self._axial_outline_actor = actor
            elif which == "sagittal":
                self._sagittal_outline_actor = actor

        except Exception:
            self._remove_actor(actor_name)

    def _render_coronal_outline(self) -> None:
        self._remove_actor("coronal_outline")

        if self.chk_coronal_plane is None or not self.chk_coronal_plane.isChecked():
            return

        self._render_plane_outline("coronal", color="green", offset_mm=0.15)

    def _render_axial_outline(self) -> None:
        self._remove_actor("axial_outline")

        if self.chk_axial_plane is None or not self.chk_axial_plane.isChecked():
            return

        self._render_plane_outline("axial", color="blue", offset_mm=0.15)

    def _render_sagittal_outline(self) -> None:
        self._remove_actor("sagittal_outline")

        if self.chk_sagittal_plane is None or not self.chk_sagittal_plane.isChecked():
            return

        self._render_plane_outline("sagittal", color="red", offset_mm=0.15)

    def _update_plane_slider_enabled_states(self) -> None:
        mni_on = bool(self.chk_mni_atlas is not None and self.chk_mni_atlas.isChecked())

        if mni_on:
            t1_loaded = bool(self._get_mni_template_t1_image() is not None)
        else:
            t1_loaded = bool(self._get_3d_plane_reference_img() is not None)

        # --- plane checkboxes themselves ---
        try:
            if self.chk_coronal_plane is not None:
                self.chk_coronal_plane.setEnabled(t1_loaded)
                if not t1_loaded:
                    self._set_checked(self.chk_coronal_plane, False)
        except Exception:
            pass

        try:
            if self.chk_axial_plane is not None:
                self.chk_axial_plane.setEnabled(t1_loaded)
                if not t1_loaded:
                    self._set_checked(self.chk_axial_plane, False)
        except Exception:
            pass

        try:
            if self.chk_sagittal_plane is not None:
                self.chk_sagittal_plane.setEnabled(t1_loaded)
                if not t1_loaded:
                    self._set_checked(self.chk_sagittal_plane, False)
        except Exception:
            pass

        # --- sliders depend on checkbox + T1 ---
        try:
            if self.sld_coronal_plane is not None and self.chk_coronal_plane is not None:
                self.sld_coronal_plane.setEnabled(
                    bool(t1_loaded and self.chk_coronal_plane.isChecked())
                )
        except Exception:
            pass

        try:
            if self.sld_axial_plane is not None and self.chk_axial_plane is not None:
                self.sld_axial_plane.setEnabled(
                    bool(t1_loaded and self.chk_axial_plane.isChecked())
                )
        except Exception:
            pass

        try:
            if self.sld_sagittal_plane is not None and self.chk_sagittal_plane is not None:
                self.sld_sagittal_plane.setEnabled(
                    bool(t1_loaded and self.chk_sagittal_plane.isChecked())
                )
        except Exception:
            pass

        # --- direction buttons depend on checkbox + T1 ---
        try:
            if self.btn_3d_StartInf is not None and self.chk_axial_plane is not None:
                self.btn_3d_StartInf.setEnabled(
                    bool(t1_loaded and self.chk_axial_plane.isChecked())
                )
        except Exception:
            pass

        try:
            if self.btn_3d_StartCaudal is not None and self.chk_coronal_plane is not None:
                self.btn_3d_StartCaudal.setEnabled(
                    bool(t1_loaded and self.chk_coronal_plane.isChecked())
                )
        except Exception:
            pass

        try:
            if self.btn_3d_StartLH is not None and self.chk_sagittal_plane is not None:
                self.btn_3d_StartLH.setEnabled(
                    bool(t1_loaded and self.chk_sagittal_plane.isChecked())
                )
        except Exception:
            pass

    def _get_effective_plane_normal(self, which: str, geom: dict) -> np.ndarray:
        n = np.array(geom["normal"], dtype=np.float64)

        if which == "coronal":
            if self._coronal_from_caudal:
                # keep posterior/caudal side visible
                if n[1] < 0:
                    n = -n
            else:
                # default/current behavior
                if n[1] > 0:
                    n = -n

        elif which == "axial":
            if self._axial_from_inferior:
                # keep inferior side visible
                if n[2] < 0:
                    n = -n
            else:
                # default/current behavior
                if n[2] > 0:
                    n = -n

        elif which == "sagittal":
            if self._sagittal_from_left:
                # keep left hemisphere visible
                if n[0] < 0:
                    n = -n
            else:
                # default/current behavior
                if n[0] > 0:
                    n = -n

        return n

    def _update_modality_controls_enabled_states(self) -> None:

        try:
            if self.chk_mni_atlas is not None and self.chk_mni_atlas.isChecked():
                # In MNI mode, the patient T1/T2 source selector is not relevant.
                # Keep T1 visually checked but disabled to avoid confusion.
                try:
                    if self.chk_mri_t1 is not None:
                        self.chk_mri_t1.blockSignals(True)
                        self.chk_mri_t1.setChecked(True)
                        self.chk_mri_t1.setEnabled(False)
                        self.chk_mri_t1.blockSignals(False)

                    if self.chk_mri_t2 is not None:
                        self.chk_mri_t2.blockSignals(True)
                        self.chk_mri_t2.setChecked(False)
                        self.chk_mri_t2.setEnabled(False)
                        self.chk_mri_t2.blockSignals(False)
                except Exception:
                    try:
                        if self.chk_mri_t1 is not None:
                            self.chk_mri_t1.blockSignals(False)
                        if self.chk_mri_t2 is not None:
                            self.chk_mri_t2.blockSignals(False)
                    except Exception:
                        pass
                # Brain mask and Pial surface must stay clickable.
                # Clicking them will disable MNI via _on_brain_source().
                for cb in (
                    getattr(self, "chk_brainmask", None),
                    getattr(self, "chk_pial", None),
                ):
                    try:
                        if cb is not None:
                            cb.setEnabled(True)
                    except Exception:
                        pass

                # Other native-space overlays remain disabled in MNI mode.
                for cb in (
                    getattr(self, "chk_iso", None),
                    getattr(self, "chk_ct", None),
                    getattr(self, "chk_pet", None),
                    getattr(self, "chk_siscom", None),
                ):
                    try:
                        if cb is not None:
                            cb.blockSignals(True)
                            cb.setChecked(False)
                            cb.setEnabled(False)
                            cb.blockSignals(False)
                    except Exception:
                        try:
                            cb.blockSignals(False)
                        except Exception:
                            pass

                # In MNI mode, coronal/axial/sagittal planes are enabled only
                # when MNI T1 slices are visible.
                mni_t1_ready = bool(
                    getattr(self, "_mni_t1_slices_visible", False)
                    and self._get_mni_template_t1_image() is not None
                )

                for cb in (
                    getattr(self, "chk_coronal_plane", None),
                    getattr(self, "chk_axial_plane", None),
                    getattr(self, "chk_sagittal_plane", None),
                ):
                    try:
                        if cb is not None:
                            cb.setEnabled(mni_t1_ready)
                            if not mni_t1_ready:
                                cb.blockSignals(True)
                                cb.setChecked(False)
                                cb.blockSignals(False)
                    except Exception:
                        try:
                            cb.blockSignals(False)
                        except Exception:
                            pass

                # In MNI mode, parcellations are available only when MNI T1 slices are visible.
                mni_parcel1_loaded = bool(mni_t1_ready and self._mni_parcellation1_loaded())
                mni_parcel2_loaded = bool(mni_t1_ready and self._mni_parcellation2_loaded())

                try:
                    if self.chk_parcel1 is not None:
                        self.chk_parcel1.setEnabled(bool(mni_parcel1_loaded))
                        if not mni_parcel1_loaded:
                            self._set_checked(self.chk_parcel1, False)
                except Exception:
                    pass

                try:
                    if self.chk_parcel2 is not None:
                        self.chk_parcel2.setEnabled(bool(mni_parcel2_loaded))
                        if not mni_parcel2_loaded:
                            self._set_checked(self.chk_parcel2, False)
                except Exception:
                    pass

                try:
                    enabled_p1 = bool(
                        self.chk_parcel1 is not None
                        and self.chk_parcel1.isChecked()
                        and mni_parcel1_loaded
                    )

                    if self.sld_parcel1_opacity is not None:
                        self.sld_parcel1_opacity.setEnabled(enabled_p1)

                    if self.spn_parcel1_opacity is not None:
                        self.spn_parcel1_opacity.setEnabled(enabled_p1)
                except Exception:
                    pass

                try:
                    enabled_p2 = bool(
                        self.chk_parcel2 is not None
                        and self.chk_parcel2.isChecked()
                        and mni_parcel2_loaded
                    )

                    if self.sld_parcel2_opacity is not None:
                        self.sld_parcel2_opacity.setEnabled(enabled_p2)

                    if self.spn_parcel2_opacity is not None:
                        self.spn_parcel2_opacity.setEnabled(enabled_p2)
                except Exception:
                    pass

                self._update_plane_slider_enabled_states()

                self._update_brain_opacity_slider_states()
                return
        except Exception:
            pass

        ct_loaded = self._get_ct_for_3d() is not None
        # --- T1/T2 anatomical source selectors ---
        try:
            t1_loaded = bool(self._t1_img is not None)
            t2_loaded = bool(self._get_t2_for_3d() is not None)

            if self.chk_mri_t1 is not None:
                self.chk_mri_t1.setEnabled(t1_loaded)

            if self.chk_mri_t2 is not None:
                self.chk_mri_t2.setEnabled(t2_loaded)

            if not t1_loaded:
                self._set_checked(self.chk_mri_t1, False)
                self._set_checked(self.chk_mri_t2, False)
                self._active_mri_source_3d = "T1"

            elif t1_loaded and not t2_loaded:
                self._active_mri_source_3d = "T1"
                self._set_checked(self.chk_mri_t1, True)
                self._set_checked(self.chk_mri_t2, False)

            elif t1_loaded and t2_loaded:
                # Keep current source if valid, otherwise default to T1.
                if getattr(self, "_active_mri_source_3d", "T1") == "T2":
                    self._set_checked(self.chk_mri_t1, False)
                    self._set_checked(self.chk_mri_t2, True)
                else:
                    self._active_mri_source_3d = "T1"
                    self._set_checked(self.chk_mri_t1, True)
                    self._set_checked(self.chk_mri_t2, False)
        except Exception:
            pass
        try:
            if self.chk_ct is not None:
                self.chk_ct.setEnabled(bool(ct_loaded))
                if not ct_loaded:
                    self._set_checked(self.chk_ct, False)
        except Exception:
            pass

        try:
            if self.sld_ct_thr is not None:
                self.sld_ct_thr.setEnabled(
                    bool(self.chk_ct is not None and self.chk_ct.isChecked() and ct_loaded)
                )
        except Exception:
            pass
        try:
            if self.sld_ct_opacity is not None:
                self.sld_ct_opacity.setEnabled(
                    bool(self.chk_ct is not None and self.chk_ct.isChecked() and ct_loaded)
                )
        except Exception:
            pass

        pet_loaded = self._pet_img is not None
        try:
            if self.chk_pet is not None:
                self.chk_pet.setEnabled(bool(pet_loaded))
                if not pet_loaded:
                    self._set_checked(self.chk_pet, False)
        except Exception:
            pass
        try:
            enabled_pet = bool(self.chk_pet is not None and self.chk_pet.isChecked() and pet_loaded)

            if self.sld_pet_min is not None:
                self.sld_pet_min.setEnabled(enabled_pet)
            if self.sb_pet_min is not None:
                self.sb_pet_min.setEnabled(enabled_pet)

            if self.sld_pet_max is not None:
                self.sld_pet_max.setEnabled(enabled_pet)
            if self.sb_pet_max is not None:
                self.sb_pet_max.setEnabled(enabled_pet)

            if self.sld_pet_gamma is not None:
                self.sld_pet_gamma.setEnabled(enabled_pet)
            if self.dsb_pet_gamma is not None:
                self.dsb_pet_gamma.setEnabled(enabled_pet)

            if self.sld_pet_opacity is not None:
                self.sld_pet_opacity.setEnabled(enabled_pet)
        except Exception:
            pass

        parcel1_loaded = self._parcel1_img is not None
        try:
            if self.chk_parcel1 is not None:
                self.chk_parcel1.setEnabled(bool(parcel1_loaded))
                if not parcel1_loaded:
                    self._set_checked(self.chk_parcel1, False)
        except Exception:
            pass
        try:
            enabled_p1 = bool(
                self.chk_parcel1 is not None and self.chk_parcel1.isChecked() and parcel1_loaded
            )
            if self.sld_parcel1_opacity is not None:
                self.sld_parcel1_opacity.setEnabled(enabled_p1)
            if self.spn_parcel1_opacity is not None:
                self.spn_parcel1_opacity.setEnabled(enabled_p1)
        except Exception:
            pass

        parcel2_loaded = self._parcel2_img is not None
        try:
            if self.chk_parcel2 is not None:
                self.chk_parcel2.setEnabled(bool(parcel2_loaded))
                if not parcel2_loaded:
                    self._set_checked(self.chk_parcel2, False)
        except Exception:
            pass
        try:
            enabled_p2 = bool(
                self.chk_parcel2 is not None and self.chk_parcel2.isChecked() and parcel2_loaded
            )
            if self.sld_parcel2_opacity is not None:
                self.sld_parcel2_opacity.setEnabled(enabled_p2)
            if self.spn_parcel2_opacity is not None:
                self.spn_parcel2_opacity.setEnabled(enabled_p2)
        except Exception:
            pass

        sis_loaded = self._siscom_img is not None
        try:
            if self.chk_siscom is not None:
                self.chk_siscom.setEnabled(bool(sis_loaded))
                if not sis_loaded:
                    self._set_checked(self.chk_siscom, False)
        except Exception:
            pass
        try:
            if self.dsb_siscom_z is not None:
                self.dsb_siscom_z.setEnabled(
                    bool(self.chk_siscom is not None and self.chk_siscom.isChecked() and sis_loaded)
                )
        except Exception:
            pass
        try:
            if self.sld_siscom_opacity is not None:
                self.sld_siscom_opacity.setEnabled(
                    bool(self.chk_siscom is not None and self.chk_siscom.isChecked() and sis_loaded)
                )
        except Exception:
            pass
        try:
            if self.chk_pial is not None:
                pial_loaded = self._lh_pial_poly is not None and self._rh_pial_poly is not None
                self.chk_pial.setEnabled(bool(pial_loaded))
                if not pial_loaded:
                    self._set_checked(self.chk_pial, False)
        except Exception:
            pass

    def _get_plane_ras(self, plane_name: str):
        img = self._get_3d_plane_reference_img()
        if img is None:
            return None
        try:
            spacing = np.array(img.GetSpacing(), dtype=np.float64)
            origin = np.array(img.GetOrigin(), dtype=np.float64)
            direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)

            if plane_name == "coronal":
                idx = self._get_coronal_slice_index()
                if idx is None:
                    return None
                idx_xyz = np.array([0.0, float(idx), 0.0], dtype=np.float64)
            elif plane_name == "axial":
                idx = self._get_axial_slice_index()
                if idx is None:
                    return None
                idx_xyz = np.array([0.0, 0.0, float(idx)], dtype=np.float64)
            elif plane_name == "sagittal":
                idx = self._get_sagittal_slice_index()
                if idx is None:
                    return None
                idx_xyz = np.array([float(idx), 0.0, 0.0], dtype=np.float64)
            else:
                return None

            lps = origin + ((idx_xyz * spacing) @ direction.T)
            ras = lps.copy()
            ras[0] *= -1.0
            ras[1] *= -1.0
            return ras
        except Exception:
            return None

    def _update_planes_info_label(self) -> None:
        if self.lbl_planes_info is None:
            return

        parent = self._active_3d_parent_widget()

        if parent is None:
            return

        if self.lbl_planes_info.parentWidget() is not parent:
            self.lbl_planes_info.setParent(parent)

        lines = []
        if self.chk_axial_plane is not None and self.chk_axial_plane.isChecked():
            idx = self._get_axial_slice_index()
            ras = self._get_plane_ras("axial")
            if idx is not None and ras is not None:
                lines.append(
                    f'<span style="color:#4aa3ff;"><b>Axial</b>: z={idx} | RAS=({ras[0]:.1f}, {ras[1]:.1f}, {ras[2]:.1f})</span>'
                )
        if self.chk_coronal_plane is not None and self.chk_coronal_plane.isChecked():
            idx = self._get_coronal_slice_index()
            ras = self._get_plane_ras("coronal")
            if idx is not None and ras is not None:
                lines.append(
                    f'<span style="color:#66ff66;"><b>Coronal</b>: y={idx} | RAS=({ras[0]:.1f}, {ras[1]:.1f}, {ras[2]:.1f})</span>'
                )
        if self.chk_sagittal_plane is not None and self.chk_sagittal_plane.isChecked():
            idx = self._get_sagittal_slice_index()
            ras = self._get_plane_ras("sagittal")
            if idx is not None and ras is not None:
                lines.append(
                    f'<span style="color:#ff6666;"><b>Sagittal</b>: x={idx} | RAS=({ras[0]:.1f}, {ras[1]:.1f}, {ras[2]:.1f})</span>'
                )

        if not lines:
            self.lbl_planes_info.hide()
            return

        self.lbl_planes_info.setText("<br>".join(lines))
        self.lbl_planes_info.adjustSize()

        margin = 12

        # Reserve space for the fullscreen button, so the slice-info label
        # does not overlap it in the bottom-right corner.
        fullscreen_reserved_w = 0

        try:
            if self.btn_3d_fullscreen is not None and self.btn_3d_fullscreen.isVisible():
                fullscreen_reserved_w = int(self.btn_3d_fullscreen.width()) + margin
        except Exception:
            fullscreen_reserved_w = 0

        x = int(parent.width()) - int(self.lbl_planes_info.width()) - margin - fullscreen_reserved_w

        y = int(parent.height()) - int(self.lbl_planes_info.height()) - margin

        self.lbl_planes_info.move(max(0, x), max(0, y))
        self.lbl_planes_info.show()
        self.lbl_planes_info.raise_()

        # Keep the fullscreen icon above everything.
        try:
            if self.btn_3d_fullscreen is not None:
                self.btn_3d_fullscreen.raise_()
        except Exception:
            pass

    def _render(self) -> None:
        try:
            self.plotter.render()
        except Exception:
            pass

    def _on_interactor_double_click(self, event) -> None:
        """
        Double left click:
            - on a marker: open marker information dialog;
            - elsewhere: preserve the existing camera reset behavior.
        """
        try:
            if event is not None and event.button() == Qt.LeftButton:
                qpos = event.position().toPoint() if hasattr(event, "position") else event.pos()

                marker_id = self._pick_marker_id_from_qpos(qpos)

                if marker_id is not None:
                    self._edit_marker(marker_id)
                    event.accept()
                    return

                self._reset_camera_to_active_plane_view()
                event.accept()
                return

        except Exception:
            pass

        try:
            if self.interactor is not None:
                super(type(self.interactor), self.interactor).mouseDoubleClickEvent(event)
        except Exception:
            pass

    def _active_single_plane_name(self) -> str:
        """
        Return the unique visible plane if exactly one is checked.
        Otherwise return 'coronal' as default.
        """
        active = []

        try:
            if self.chk_coronal_plane is not None and self.chk_coronal_plane.isChecked():
                active.append("coronal")
        except Exception:
            pass

        try:
            if self.chk_axial_plane is not None and self.chk_axial_plane.isChecked():
                active.append("axial")
        except Exception:
            pass

        try:
            if self.chk_sagittal_plane is not None and self.chk_sagittal_plane.isChecked():
                active.append("sagittal")
        except Exception:
            pass

        if len(active) == 1:
            return active[0]

        return "coronal"

    def _load_freesurfer_surface_as_polydata(self, surf_path: str, assume_lps: bool = False):
        try:
            verts, faces = nib.freesurfer.read_geometry(surf_path)

            verts = np.asarray(verts, dtype=np.float32)
            faces = np.asarray(faces, dtype=np.int64)

            faces_vtk = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()

            # If the saved pial is in LPS (output of ANTs worker), convert it to RAS for VTK/PyVista
            if assume_lps:
                verts_plot = _lps_to_ras_points(verts)
            else:
                verts_plot = verts.copy()

            # Optional brute-force debug shift in RAS
            try:
                verts_plot = (
                    verts_plot + np.asarray(self._pial_debug_shift_ras, dtype=np.float32)[None, :]
                )
            except Exception:
                pass

            poly = pv.PolyData(verts_plot, faces_vtk)

            try:
                poly = poly.triangulate()
            except Exception:
                pass

            try:
                poly = poly.clean(tolerance=1e-6)
            except Exception:
                pass

            try:
                poly = poly.compute_normals(
                    cell_normals=False,
                    point_normals=True,
                    auto_orient_normals=True,
                    consistent_normals=True,
                    split_vertices=False,
                    inplace=False,
                )
            except Exception:
                pass

            return poly
        except Exception:
            return None

    def _render_ct(self) -> None:
        if not _PV_OK or self.plotter is None:
            return

        show = bool(self.chk_ct.isChecked()) if self.chk_ct is not None else False
        if not show:
            self._remove_actor("ct")
            self._render()
            return

        ct_img = self._get_ct_for_3d()
        if ct_img is None:
            self._remove_actor("ct")
            self._render()
            return

        active_mask = self._get_active_surface_mask_resampled_to(ct_img)

        try:
            ct_np = sitk.GetArrayFromImage(ct_img).astype(np.float32)

            if active_mask is not None:
                mask_np = sitk.GetArrayFromImage(active_mask).astype(np.uint8)

                # Keep CT values only inside the active surface mask
                ct_masked_np = np.full_like(ct_np, -1000.0, dtype=np.float32)
                ct_masked_np[mask_np > 0] = ct_np[mask_np > 0]
            else:
                ct_masked_np = ct_np

            # Rebuild a SITK image with the exact same geometry as the CT
            ct_masked_img = sitk.GetImageFromArray(ct_masked_np)
            ct_masked_img.CopyInformation(ct_img)

        except Exception:
            self._remove_actor("ct")
            self._render()
            return

        thr = float(self.sld_ct_thr.value()) if self.sld_ct_thr is not None else 2500.0

        # Reuse the same geometry-safe pipeline as PET/SISCOM/brain surfaces
        mesh = self._threshold_image_to_polydata(
            ct_masked_img,
            absolute_threshold=thr,
        )

        if mesh is None or mesh.n_points == 0:
            self._remove_actor("ct")
            self._render()
            return

        self._remove_actor("ct")

        opacity = (self.sld_ct_opacity.value() / 100.0) if self.sld_ct_opacity is not None else 0.6

        self._ct_actor = self.plotter.add_mesh(
            mesh,
            color=self._ct_color,
            opacity=float(opacity),
            smooth_shading=True,
        )

        self._apply_actor_clipping()
        self._render()

    def _update_ct_opacity(self) -> None:
        if self._ct_actor is None or self.plotter is None:
            return

        try:
            op = (self.sld_ct_opacity.value() / 100.0) if self.sld_ct_opacity is not None else 0.75
            self._ct_actor.GetProperty().SetOpacity(float(op))
            self._render()
        except Exception:
            pass

    def _get_coronal_slice_index(self) -> int | None:
        img = self._get_3d_plane_reference_img()
        if img is None or self.sld_coronal_plane is None:
            return None

        try:
            size = img.GetSize()  # x, y, z
            full_y_dim = int(size[1])

            y_min = int(self._coronal_y_min) if self._coronal_y_min is not None else 0
            y_max = (
                int(self._coronal_y_max) if self._coronal_y_max is not None else (full_y_dim - 1)
            )

            if y_max < y_min:
                y_min, y_max = 0, full_y_dim - 1

            slider_max = max(0, y_max - y_min)
            slider_val = int(np.clip(self.sld_coronal_plane.value(), 0, slider_max))

            if self._coronal_from_caudal:
                y_idx = y_min + slider_val
            else:
                y_idx = y_max - slider_val

            return int(np.clip(y_idx, 0, full_y_dim - 1))

        except Exception:
            return None

    def _offset_coronal_quad_points(self, geom: dict, offset_mm: float = 0.4):
        """
        Return the 4 coronal quad points slightly shifted along plane normal
        to avoid z-fighting with the T1 coronal face.
        """
        n = np.asarray(geom["normal"], dtype=np.float64)
        nn = np.linalg.norm(n)
        if nn <= 0:
            n = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            n = n / nn

        d = float(offset_mm)

        p00 = np.asarray(geom["p00"], dtype=np.float64) + d * n
        p10 = np.asarray(geom["p10"], dtype=np.float64) + d * n
        p11 = np.asarray(geom["p11"], dtype=np.float64) + d * n
        p01 = np.asarray(geom["p01"], dtype=np.float64) + d * n

        return p00, p10, p11, p01

    def _get_coronal_plane_origin_ras(self) -> np.ndarray | None:
        if self._t1_img is None:
            return None

        y_idx = self._get_coronal_slice_index()
        if y_idx is None:
            return None

        spacing = np.array(self._t1_img.GetSpacing(), dtype=np.float64)  # x,y,z
        origin = np.array(self._t1_img.GetOrigin(), dtype=np.float64)
        direction = np.array(self._t1_img.GetDirection(), dtype=np.float64).reshape(3, 3)

        # Any point with this fixed y is on the coronal plane
        idx_xyz = np.array([0.0, float(y_idx), 0.0], dtype=np.float64)
        lps = origin + ((idx_xyz * spacing) @ direction.T)

        ras = lps.copy()
        ras[0] *= -1.0
        ras[1] *= -1.0
        return ras

    def _apply_coronal_clip(self, mesh):
        if mesh is None:
            return None

        if self.chk_coronal_plane is None or not self.chk_coronal_plane.isChecked():
            return mesh

        geom = self._build_coronal_plane_geometry()
        if geom is None:
            return mesh

        try:
            clipped = mesh.clip(
                normal=tuple(geom["normal"]),
                origin=tuple(geom["center"]),
                invert=False,  # <-- CHANGEMENT ICI
            )
            return clipped
        except Exception:
            return mesh

    def _refresh_coronal_clipped_scene(self) -> None:
        self._refresh_multiplanar_clipped_scene()

    def _render_coronal_plane(self) -> None:
        self._remove_actor("coronal_plane")

        if not _PV_OK or self.plotter is None:
            return

        show = (
            bool(self.chk_coronal_plane.isChecked())
            if self.chk_coronal_plane is not None
            else False
        )
        if not show or self._get_3d_plane_reference_img() is None or self.sld_coronal_plane is None:
            self._render()
            return

        try:
            import pyvista as pv

            geom = self._build_coronal_plane_geometry()
            if geom is None:
                self._render()
                return

            rgba = self._build_plane_rgba_highres(geom)
            if rgba is None:
                self._render()
                return

            texture = pv.numpy_to_texture(rgba)

            source_quad = self._build_textured_plane_mesh(geom, "coronal_plane")
            if source_quad is None or source_quad.n_points == 0:
                self._render()
                return

            self._coronal_plane_source_mesh = source_quad.copy()

            quad = self._clip_plane_mesh_with_other_planes("coronal", source_quad.copy())
            if quad is None or quad.n_points == 0:
                self._render()
                return
            if quad is None or quad.n_points == 0:
                self._render()
                return

            self._coronal_plane_actor = self.plotter.add_mesh(
                quad,
                texture=texture,
                opacity=1.0,
                lighting=False,
                show_scalar_bar=False,
                name="coronal_plane",
            )
            try:
                prop = self._coronal_plane_actor.GetProperty()
                prop.SetAmbient(1.0)
                prop.SetDiffuse(0.0)
                prop.SetSpecular(0.0)
            except Exception:
                pass

            self._render()
        except Exception:
            self._remove_actor("coronal_plane")
            self._render()

    def _get_active_surface_mask_img(self) -> sitk.Image | None:
        """
        Return the mask that should define the visible brain volume for slices/overlays:
        - if pial is selected and pial masks exist -> union(LH, RH)
        - otherwise -> brainmask
        """

        def _same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
            try:
                return (
                    tuple(a.GetSize()) == tuple(b.GetSize())
                    and tuple(np.round(a.GetSpacing(), 6)) == tuple(np.round(b.GetSpacing(), 6))
                    and tuple(np.round(a.GetOrigin(), 6)) == tuple(np.round(b.GetOrigin(), 6))
                    and tuple(np.round(a.GetDirection(), 6)) == tuple(np.round(b.GetDirection(), 6))
                )
            except Exception:
                return False

        def _to_uint8_mask(
            img: sitk.Image | None, ref: sitk.Image | None = None
        ) -> sitk.Image | None:
            if img is None:
                return None
            try:
                out = img
                if ref is not None and not _same_geometry(out, ref):
                    out = sitk.Resample(
                        out,
                        ref,
                        sitk.Transform(),
                        sitk.sitkNearestNeighbor,
                        0,
                        sitk.sitkUInt8,
                    )
                return sitk.Cast(out > 0, sitk.sitkUInt8)
            except Exception:
                return None

        # Preferred: pial masks if pial mode is active
        if self.chk_pial is not None and self.chk_pial.isChecked():
            ref = self._t1_img if self._t1_img is not None else self._brainmask_img

            lh = self._lh_pial_mask_img if getattr(self, "_show_lh_pial", True) else None
            rh = self._rh_pial_mask_img if getattr(self, "_show_rh_pial", True) else None

            lh = _to_uint8_mask(lh, ref=ref)
            rh = _to_uint8_mask(rh, ref=ref)

            if lh is not None and rh is not None:
                try:
                    return sitk.Cast((lh > 0) | (rh > 0), sitk.sitkUInt8)
                except Exception:
                    pass
            if lh is not None:
                return lh
            if rh is not None:
                return rh

        # Fallback: classic brainmask
        return self._brainmask_img

    def _get_active_surface_mask_resampled_to(
        self, ref_img: sitk.Image | None
    ) -> sitk.Image | None:
        if ref_img is None:
            return None

        mask = self._get_active_surface_mask_img()
        if mask is None:
            return None

        try:
            return sitk.Resample(
                mask,
                ref_img,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                sitk.sitkUInt8,
            )
        except Exception:
            return None

    def _clip_plane_mesh_with_other_planes(self, which: str, mesh):
        """
        Clip a slice plane mesh with the other active slice planes so that
        intersecting planes do not extend beyond each other.
        """
        if mesh is None:
            return mesh

        planes = []

        if (
            which != "coronal"
            and self.chk_coronal_plane is not None
            and self.chk_coronal_plane.isChecked()
        ):
            geom = self._build_coronal_plane_geometry()
            if geom is not None:
                n = self._get_effective_plane_normal("coronal", geom)
                planes.append((geom["center"], n))

        if (
            which != "axial"
            and self.chk_axial_plane is not None
            and self.chk_axial_plane.isChecked()
        ):
            geom = self._build_axial_plane_geometry()
            if geom is not None:
                n = self._get_effective_plane_normal("axial", geom)
                planes.append((geom["center"], n))

        if (
            which != "sagittal"
            and self.chk_sagittal_plane is not None
            and self.chk_sagittal_plane.isChecked()
        ):
            geom = self._build_sagittal_plane_geometry()
            if geom is not None:
                n = self._get_effective_plane_normal("sagittal", geom)
                planes.append((geom["center"], n))

        clipped = mesh
        for origin, normal in planes:
            try:
                clipped = clipped.clip(
                    normal=tuple(np.asarray(normal, dtype=np.float64)),
                    origin=tuple(np.asarray(origin, dtype=np.float64)),
                    invert=False,
                )
            except Exception:
                pass

        return clipped

    # ---------- helpers to build meshes ----------
    def _meshdata_to_polydata(self, mesh_data: dict) -> pv.PolyData | None:
        try:
            pts = np.asarray(mesh_data["points"], dtype=np.float32)
            faces = np.asarray(mesh_data["faces"], dtype=np.int32)  # (M,3)
            faces_vtk = np.hstack(
                [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]
            ).ravel()
            return pv.PolyData(_lps_to_ras_points(pts), faces_vtk)
        except Exception:
            return None

    def _binarymask_to_polydata(self, img: sitk.Image) -> pv.PolyData | None:
        arr = sitk.GetArrayFromImage(img)  # z,y,x
        vol = (arr > 0).astype(np.uint8)
        if vol.max() == 0:
            return None
        try:
            verts_zyx, faces, _, _ = marching_cubes(vol, level=0.5)
        except Exception:
            return None
        return self._verts_faces_to_polydata(img, verts_zyx, faces)

    def _threshold_image_to_polydata(
        self,
        img: sitk.Image,
        threshold_percentile: int | None = None,
        absolute_threshold: float | None = None,
    ) -> pv.PolyData | None:
        arr = sitk.GetArrayFromImage(img).astype(np.float32, copy=False)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return None
        if absolute_threshold is not None:
            thr = float(absolute_threshold)
        else:
            pct = int(np.clip(int(threshold_percentile or 90), 1, 99))
            thr = float(np.percentile(arr[finite], pct))
        mask = (arr > thr) & finite
        if np.count_nonzero(mask) < 50:
            return None
        vol = mask.astype(np.uint8)
        try:
            verts_zyx, faces, _, _ = marching_cubes(vol, level=0.5)
        except Exception:
            return None
        return self._verts_faces_to_polydata(img, verts_zyx, faces)

    def _verts_faces_to_polydata(
        self, img: sitk.Image, verts_zyx: np.ndarray, faces: np.ndarray
    ) -> pv.PolyData | None:
        try:
            verts_xyz = verts_zyx[:, ::-1].astype(np.float64)  # x,y,z voxel
            origin = np.array(img.GetOrigin(), dtype=np.float64)
            spacing = np.array(img.GetSpacing(), dtype=np.float64)
            direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
            phys = origin[None, :] + ((verts_xyz * spacing[None, :]) @ direction.T)

            faces_vtk = np.hstack(
                [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]
            ).ravel()
            mesh = pv.PolyData(_lps_to_ras_points(phys.astype(np.float32)), faces_vtk)
            try:
                mesh = mesh.clean(tolerance=1e-6)
            except Exception:
                pass
            return mesh
        except Exception:
            return None

    def _get_ct_for_3d(self) -> sitk.Image | None:
        """
        Return only a validated CT coregistered into T1 space.

        Raw CT must remain available for Reconstruction, but it must never be
        displayed as a validated 3D overlay.
        """
        if not bool(getattr(self.state, "ct_validated", False)):
            return None

        if self._ct_img is not None:
            return self._ct_img

        img = getattr(self.state, "ct_coreg_in_t1", None)
        if isinstance(img, sitk.Image):
            return img

        img = getattr(self.state, "ct_in_t1", None)
        if isinstance(img, sitk.Image):
            return img

        return None

    def set_ct(self, ct_img: sitk.Image | None, ct_path: str | None = None) -> None:
        if ct_img is None and ct_path:
            try:
                ct_img = sitk.ReadImage(ct_path)
            except Exception:
                ct_img = None

        self._ct_img = ct_img
        try:
            self._render_ct()
        except Exception:
            pass
        try:
            self._update_modality_controls_enabled_states()
        except Exception:
            pass

    def _apply_actor_clipping(self) -> None:
        if self.plotter is None:
            return

        planes = []

        # Coronal
        if self.chk_coronal_plane is not None and self.chk_coronal_plane.isChecked():
            p = self._build_vtk_coronal_clip_plane()
            if p is not None:
                planes.append(p)

        # Axial
        if self.chk_axial_plane is not None and self.chk_axial_plane.isChecked():
            p = self._build_vtk_axial_clip_plane()
            if p is not None:
                planes.append(p)

        # Sagittal
        if self.chk_sagittal_plane is not None and self.chk_sagittal_plane.isChecked():
            p = self._build_vtk_sagittal_clip_plane()
            if p is not None:
                planes.append(p)

        actors = [
            self._brain_actor,
            self._ct_actor,
            self._pet_actor,
            self._siscom_actor,
            self._mni_atlas_actor,
        ]

        keep_electrodes = bool(getattr(self, "_keep_electrodes_visible_through_slices", False))

        # Native electrodes:
        # normal mode -> clipped by slice planes
        # keep-through-slices mode -> not clipped
        try:
            if keep_electrodes:
                for actor in list(getattr(self, "_elec_actors", {}).values()):
                    self._apply_electrode_actor_depth_mode(actor)
            else:
                actors.extend(list(getattr(self, "_elec_actors", {}).values()))
        except Exception:
            pass

        # Keep MNI electrodes with the current behavior.
        try:
            actors.extend(list(getattr(self, "_mni_electrode_actors", {}).values()))
        except Exception:
            pass

        for actor in actors:
            if actor is None:
                continue

            try:
                mapper = actor.GetMapper()
                if mapper is None:
                    continue

                mapper.RemoveAllClippingPlanes()

                for plane in planes:
                    mapper.AddClippingPlane(plane)

            except Exception:
                continue

        self._render()

    def _get_pet_slice_overlay_alpha(self) -> int:
        try:
            if self.sld_pet_opacity is not None:
                return int(np.clip(float(self.sld_pet_opacity.value()) * 255.0 / 100.0, 0, 255))
        except Exception:
            pass
        return 170

    def _get_siscom_slice_overlay_alpha(self) -> float:
        try:
            if self.sld_siscom_opacity is not None:
                return float(np.clip(float(self.sld_siscom_opacity.value()) / 100.0, 0.0, 1.0))
        except Exception:
            pass
        return 0.75

    def _blend_overlay_on_rgba(
        self, rgba: np.ndarray, mask: np.ndarray, color_rgb, alpha: int
    ) -> np.ndarray:
        """
        Alpha-blend a binary overlay onto an existing RGBA image.
        rgba: H x W x 4 uint8
        mask: H x W bool
        color_rgb: tuple/list of 3 ints in [0,255]
        alpha: int in [0,255]
        """
        if rgba is None or mask is None or not np.any(mask):
            return rgba

        out = rgba.astype(np.float32, copy=True)
        a = float(np.clip(alpha, 0, 255)) / 255.0

        for c in range(3):
            out[..., c][mask] = (1.0 - a) * out[..., c][mask] + a * float(color_rgb[c])

        # garde la face visible partout où elle l'était déjà
        out[..., 3][mask] = np.maximum(out[..., 3][mask], 255.0)

        return np.clip(out, 0, 255).astype(np.uint8)

    def _get_contact_overlay_point_size(self) -> float:
        try:
            if self.spin_contacts_size is not None:
                return max(1.0, float(self.spin_contacts_size.value()))
        except Exception:
            pass
        return 8.0

    def _get_overlay_contact_radius_mm(self) -> float:
        try:
            if self.spin_contacts_size is not None:
                v = float(self.spin_contacts_size.value())
                return max(0.4, 0.12 * v)
        except Exception:
            pass
        return 1.0

    def _project_point_to_plane(
        self,
        point_ras: np.ndarray,
        plane_center_ras: np.ndarray,
        plane_normal_ras: np.ndarray,
    ) -> np.ndarray:
        try:
            p = np.asarray(point_ras, dtype=np.float64)
            c = np.asarray(plane_center_ras, dtype=np.float64)
            n = np.asarray(plane_normal_ras, dtype=np.float64)

            nn = np.linalg.norm(n)
            if nn < 1e-6:
                return p.copy()

            n = n / nn
            signed_dist = np.dot(p - c, n)
            return p - signed_dist * n
        except Exception:
            return np.asarray(point_ras, dtype=np.float64)

    def _build_contact_cylinder_mesh(
        self,
        center_ras: np.ndarray,
        axis_ras: np.ndarray,
        radius_mm: float = 1.0,
        length_mm: float = 2.0,
    ):
        """
        Build a small cylinder centered on contact, oriented along electrode axis.
        """
        try:
            c = np.asarray(center_ras, dtype=np.float64)
            a = np.asarray(axis_ras, dtype=np.float64)

            n = np.linalg.norm(a)
            if n < 1e-6:
                a = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                a = a / n

            cyl = pv.Cylinder(
                center=c,
                direction=a,
                radius=float(radius_mm),
                height=float(length_mm),
                resolution=24,
            )
            return cyl
        except Exception:
            return None

    def _get_contact_axis_ras(self, contacts_lps, ci: int) -> np.ndarray:
        """
        Estimate local electrode axis at contact ci from neighboring contacts.
        """
        try:
            pts = np.asarray(contacts_lps, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
                return np.array([0.0, 0.0, 1.0], dtype=np.float64)

            pts_ras = pts.copy()
            pts_ras[:, 0] *= -1.0
            pts_ras[:, 1] *= -1.0

            if pts_ras.shape[0] == 1:
                return np.array([0.0, 0.0, 1.0], dtype=np.float64)

            if ci <= 0:
                axis = pts_ras[1] - pts_ras[0]
            elif ci >= pts_ras.shape[0] - 1:
                axis = pts_ras[-1] - pts_ras[-2]
            else:
                axis = pts_ras[ci + 1] - pts_ras[ci - 1]

            n = np.linalg.norm(axis)
            if n < 1e-6:
                return np.array([0.0, 0.0, 1.0], dtype=np.float64)

            return axis / n
        except Exception:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def _get_active_surface_polydata_for_projection(self):
        if not _PV_OK:
            return None
        try:
            if self.chk_pial is not None and self.chk_pial.isChecked():
                polys = []
                if self._show_lh_pial:
                    if self._lh_pial_poly is not None:
                        polys.append(self._lh_pial_poly.copy())
                    elif self._lh_pial_mask_img is not None:
                        poly = self._binarymask_to_polydata(self._lh_pial_mask_img)
                        if poly is not None and getattr(poly, "n_points", 0) > 0:
                            polys.append(poly.copy())
                if self._show_rh_pial:
                    if self._rh_pial_poly is not None:
                        polys.append(self._rh_pial_poly.copy())
                    elif self._rh_pial_mask_img is not None:
                        poly = self._binarymask_to_polydata(self._rh_pial_mask_img)
                        if poly is not None and getattr(poly, "n_points", 0) > 0:
                            polys.append(poly.copy())
                if polys:
                    mesh = polys[0]
                    for p in polys[1:]:
                        try:
                            mesh = mesh.merge(p)
                        except Exception:
                            pass
                    try:
                        mesh = mesh.triangulate().clean(tolerance=1e-6)
                    except Exception:
                        pass
                    return mesh

            if self._brainmask_img is not None:
                return self._binarymask_to_polydata(self._brainmask_img)
        except Exception:
            return None
        return None

    def _remove_all_surface_projection_actors(self) -> None:
        """
        Remove every native surface projection actor from the PyVista scene.

        These projections are in native patient space and must never stay visible
        when MNI atlas mode is active.
        """
        if self.plotter is None:
            return

        # Remove actors stored in the current dict format:
        # _surface_projection_actors[elec_id] = {"cross": actor, "label": actor}
        try:
            for elec_id, actors in list(getattr(self, "_surface_projection_actors", {}).items()):
                if isinstance(actors, dict):
                    for actor in actors.values():
                        if actor is not None:
                            try:
                                self.plotter.remove_actor(actor, reset_camera=False)
                            except Exception:
                                pass
                elif actors is not None:
                    try:
                        self.plotter.remove_actor(actors, reset_camera=False)
                    except Exception:
                        pass

            self._surface_projection_actors.clear()
        except Exception:
            pass

        # Compatibility with older separate label storage.
        try:
            for _, actor in list(getattr(self, "_surface_projection_label_actors", {}).items()):
                if actor is not None:
                    try:
                        self.plotter.remove_actor(actor, reset_camera=False)
                    except Exception:
                        pass

            self._surface_projection_label_actors.clear()
        except Exception:
            pass

        # Extra safety: remove named actors.
        try:
            keys = set()

            try:
                keys.update(int(k) for k in getattr(self, "_surface_projection_defs", {}).keys())
            except Exception:
                pass

            try:
                keys.update(int(k) for k in getattr(self, "_surface_projection_actors", {}).keys())
            except Exception:
                pass

            # Also try a broad range, because old actors may have lost their Python refs.
            try:
                n_elec = len(getattr(self.state, "electrodes", []) or [])
                keys.update(range(n_elec))
            except Exception:
                pass

            for elec_id in keys:
                for actor_name in (
                    f"surface_proj_cross_{int(elec_id)}",
                    f"surface_proj_label_{int(elec_id)}",
                ):
                    try:
                        self.plotter.remove_actor(actor_name, reset_camera=False)
                    except Exception:
                        pass

        except Exception:
            pass

    def _remove_surface_projection_actors(self, elec_id=None) -> None:
        """
        Compatibility wrapper.

        Old code calls _remove_surface_projection_actors().
        New code mostly uses _remove_surface_projection_actor(elec_id).
        """
        if elec_id is None:
            self._remove_all_surface_projection_actors()
            return

        try:
            self._remove_surface_projection_actor(int(elec_id))
        except Exception:
            self._remove_all_surface_projection_actors()

    def _remove_surface_projection_actor(self, elec_id: int) -> None:
        if self.plotter is None:
            return
        actors = self._surface_projection_actors.pop(int(elec_id), None)
        if not actors:
            return
        for key in ("cross", "label"):
            a = actors.get(key)
            if a is None:
                continue
            try:
                self.plotter.remove_actor(a, reset_camera=False)
            except Exception:
                pass

    def _build_surface_projection_cross(
        self, center_ras: np.ndarray, normal_ras: np.ndarray, size_mm: float = 3.0
    ):
        try:
            c = np.asarray(center_ras, dtype=np.float64)
            n = np.asarray(normal_ras, dtype=np.float64)
            nn = np.linalg.norm(n)
            if nn < 1e-6:
                n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                n = n / nn

            ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(np.dot(ref, n)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

            u = np.cross(n, ref)
            nu = np.linalg.norm(u)
            if nu < 1e-6:
                u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                u = u / nu

            v = np.cross(n, u)
            nv = np.linalg.norm(v)
            if nv < 1e-6:
                v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            else:
                v = v / nv

            h = float(size_mm) * 0.5
            p1 = c - h * u
            p2 = c + h * u
            p3 = c - h * v
            p4 = c + h * v
            line1 = pv.Line(p1, p2)
            line2 = pv.Line(p3, p4)
            try:
                return line1.merge(line2)
            except Exception:
                return line1
        except Exception:
            return None

    def _project_single_electrode_on_surface(self, elec_id: int) -> None:
        if not _PV_OK or self.plotter is None:
            return
        try:
            electrodes = getattr(self.state, "electrodes", None) or []
            if not (0 <= int(elec_id) < len(electrodes)):
                return
            elec = electrodes[int(elec_id)]
            contacts_lps = elec.get("contacts_lps", []) or []
            contacts_visible = elec.get("contacts_visible")
            if contacts_visible is None or len(contacts_visible) != len(contacts_lps):
                contacts_visible = [True] * len(contacts_lps)

            pts = []
            for ci, p in enumerate(contacts_lps):
                if ci < len(contacts_visible) and not bool(contacts_visible[ci]):
                    continue
                try:
                    ras = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float64)
                    ras[0] *= -1.0
                    ras[1] *= -1.0
                    pts.append(ras)
                except Exception:
                    pass
            if not pts:
                return

            surf = self._get_active_surface_polydata_for_projection()
            if surf is None or getattr(surf, "n_points", 0) == 0:
                return

            surf_pts = np.asarray(surf.points, dtype=np.float64)
            if surf_pts.size == 0:
                return

            # choose the contact closest to the active surface
            # Use the implantation axis: from previous contact toward the last contact,
            # then continue outward to the external surface.
            pts_arr = np.asarray(pts, dtype=np.float64)
            if pts_arr.shape[0] == 1:
                origin = pts_arr[0]
                direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                origin = pts_arr[-1]
                direction = pts_arr[-1] - pts_arr[-2]

            hit, best_pid = self._ray_surface_intersection(surf, origin, direction)
            if hit is None:
                return

            best_src = origin
            best_proj = np.asarray(hit, dtype=np.float64)

            try:
                normals_mesh = surf.compute_normals(
                    cell_normals=False,
                    point_normals=True,
                    auto_orient_normals=True,
                    consistent_normals=True,
                    split_vertices=False,
                    inplace=False,
                )
                nrm_arr = np.asarray(normals_mesh.point_normals, dtype=np.float64)
                normal = (
                    nrm_arr[int(best_pid)]
                    if best_pid is not None and nrm_arr.size
                    else (best_proj - best_src)
                )
            except Exception:
                normal = best_proj - best_src

            nlen = np.linalg.norm(normal)
            if nlen < 1e-6:
                normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                normal = normal / nlen

            offset_mm = 1.0
            offset_center = np.asarray(best_proj, dtype=np.float64) + offset_mm * normal

            rgb255 = tuple(elec.get("color", (0, 255, 0)))
            color = (rgb255[0] / 255.0, rgb255[1] / 255.0, rgb255[2] / 255.0)
            name = str(elec.get("name", f"Elec{int(elec_id)+1}"))

            self._remove_surface_projection_actor(int(elec_id))

            cross = self._build_surface_projection_cross(offset_center, normal, size_mm=4.0)
            cross_actor = None
            if cross is not None and getattr(cross, "n_points", 0) > 0:
                cross_actor = self.plotter.add_mesh(
                    cross,
                    color=color,
                    line_width=4,
                    name=f"surface_proj_cross_{int(elec_id)}",
                )
                try:
                    cross_actor.PickableOff()
                except Exception:
                    pass

            # build a small tangential "top-right" offset on the surface
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(np.dot(ref, normal)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

            u = np.cross(normal, ref)
            nu = np.linalg.norm(u)
            if nu < 1e-6:
                u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                u = u / nu

            v = np.cross(normal, u)
            nv = np.linalg.norm(v)
            if nv < 1e-6:
                v = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                v = v / nv

            label_pos = (
                np.asarray(offset_center, dtype=np.float64)
                + 1.2 * normal  # lift a bit above surface
                + 2.0 * u  # right
                + 2.0 * v  # up
            )

            label_actor = self.plotter.add_point_labels(
                np.asarray([label_pos], dtype=np.float32),
                [name],
                font_size=12,
                text_color=color,
                shape_opacity=0.0,
                show_points=False,
                always_visible=True,
                name=f"surface_proj_label_{int(elec_id)}",
            )
            try:
                label_actor.PickableOff()
            except Exception:
                pass

            self._surface_projection_actors[int(elec_id)] = {
                "cross": cross_actor,
                "label": label_actor,
            }
        except Exception:
            pass

    def render_all_surface_projections(self) -> None:
        if not _PV_OK or self.plotter is None:
            return

        # Never render native surface projections in MNI space.
        if self._mni_mode_is_active():
            self._remove_all_surface_projection_actors()
            return

        for elec_id in list(getattr(self, "_surface_projection_defs", {}).keys()):
            self._project_single_electrode_on_surface(int(elec_id))

        self._render()

    def project_electrode_on_surface(self, elec_id: int) -> None:
        try:
            self._surface_projection_defs[int(elec_id)] = True
        except Exception:
            return
        self._project_single_electrode_on_surface(int(elec_id))
        self._render()

    def remove_surface_projection(self, elec_id: int) -> None:
        try:
            self._surface_projection_defs.pop(int(elec_id), None)
        except Exception:
            pass
        self._remove_surface_projection_actor(int(elec_id))
        self._render()

    def _toggle_color_scales(self) -> None:
        self._show_color_scales = not bool(getattr(self, "_show_color_scales", True))

        if not self._show_color_scales:
            try:
                self._remove_pet_scalar_bar()
            except Exception:
                pass
            try:
                self._remove_siscom_scalar_bar()
            except Exception:
                pass
            self._render()
            return

        try:
            self._update_pet_scalar_bar()
        except Exception:
            pass
        try:
            self._update_siscom_scalar_bar()
        except Exception:
            pass
        self._render()

    def _is_pial_surface_checked(self) -> bool:
        """
        True only when the Pial Surface checkbox is checked in the 3D View.
        Used to show/hide the LH/RH context-menu options.
        """
        try:
            return bool(self.chk_pial is not None and self.chk_pial.isChecked())
        except Exception:
            return False

    def _is_pet_or_siscom_checked(self) -> bool:
        """
        True only when PET or SISCOM is checked in the 3D View.
        Used to show/hide the Remove/Add color scale context-menu option.
        """
        try:
            pet_on = bool(self.chk_pet is not None and self.chk_pet.isChecked())
        except Exception:
            pet_on = False

        try:
            siscom_on = bool(self.chk_siscom is not None and self.chk_siscom.isChecked())
        except Exception:
            siscom_on = False

        return bool(pet_on or siscom_on)

    def _is_mni_atlas_checked(self) -> bool:
        """
        True only when the MNI atlas checkbox is checked in the 3D View.
        Used to show/hide the 'Load MNI electrodes.tsv…' context-menu option.
        """
        try:
            return bool(self.chk_mni_atlas is not None and self.chk_mni_atlas.isChecked())
        except Exception:
            return False

    def _set_mni_t1_slices_visible(self, visible: bool) -> None:
        """
        Enable/disable MNI T1 template slices.

        This reuses the existing coronal/axial/sagittal checkboxes and sliders,
        but the image source becomes the MNI template T1.
        """
        visible = bool(visible)

        try:
            if visible and not self._is_mni_atlas_checked():
                return

            if visible and self._get_mni_template_t1_image() is None:
                NeuXelecMessageDialog.warning(
                    self._dialog_parent(),
                    "MNI T1 slices",
                    (
                        "Could not load template_T1w.nii.gz.\n\n"
                        "Expected file:\n"
                        "tools/templates/template_T1w.nii.gz"
                    ),
                )
                return

            self._mni_t1_slices_visible = visible

            try:
                # Preload MNI parcellations if they exist, so controls can update immediately.
                if visible:
                    self._get_mni_parcellation1_img_and_lut()
                    self._get_mni_parcellation2_img_and_lut()
            except Exception:
                pass

            if not visible:
                # Remove visible slice actors.
                self._set_checked(self.chk_coronal_plane, False)
                self._set_checked(self.chk_axial_plane, False)
                self._set_checked(self.chk_sagittal_plane, False)

                self._set_checked(self.chk_parcel1, False)
                self._set_checked(self.chk_parcel2, False)

                self._remove_actor("coronal_plane")
                self._remove_actor("axial_plane")
                self._remove_actor("sagittal_plane")

                self._remove_actor("coronal_outline")
                self._remove_actor("axial_outline")
                self._remove_actor("sagittal_outline")

            else:
                # MNI T1 is available, but the user decides which slice to display.
                self._set_checked(self.chk_coronal_plane, False)
                self._set_checked(self.chk_axial_plane, False)
                self._set_checked(self.chk_sagittal_plane, False)

            self._update_all_plane_slider_ranges()
            self._update_modality_controls_enabled_states()
            self._update_plane_slider_enabled_states()
            self._update_planes_info_label()
            self._refresh_multiplanar_clipped_scene()
            self._render()

        except Exception as e:
            print("[MNI T1 slices] toggle failed:", e)

    def _toggle_mni_t1_slices(self) -> None:
        self._set_mni_t1_slices_visible(not bool(getattr(self, "_mni_t1_slices_visible", False)))

    def _open_mni_parcellation_table(self) -> None:
        """
        Open a floating, non-modal table showing MNI contacts and parcellation labels.
        """
        print("[MNI parcellation table] open requested")
        print("[MNI parcellation table] MNI atlas checked:", self._is_mni_atlas_checked())
        print(
            "[MNI parcellation table] MNI sets in state:",
            len(getattr(self.state, "mni_electrode_sets", []) or []),
        )
        try:
            self._mni_parcel1_img = None
            self._mni_parcel2_img = None
            self._mni_parcel1_lut = {}
            self._mni_parcel2_lut = {}
        except Exception:
            pass

        try:
            if not self._is_mni_atlas_checked():
                NeuXelecMessageDialog.information(
                    self._dialog_parent(),
                    "MNI parcellation table",
                    "Please enable MNI atlas first.",
                )
                return

            sets = getattr(self.state, "mni_electrode_sets", []) or []
            if not sets:
                NeuXelecMessageDialog.information(
                    self._dialog_parent(),
                    "MNI parcellation table",
                    (
                        "No MNI electrodes.tsv file is loaded yet.\n\n"
                        "Right-click in the 3D view and choose:\n"
                        "Load MNI electrodes.tsv…"
                    ),
                )
                return

            # Reuse existing dialog if it already exists.
            dlg = getattr(self, "_mni_parcellation_table_dialog", None)

            if dlg is not None:
                try:
                    if dlg.isVisible():
                        dlg.refresh()
                        dlg.raise_()
                        dlg.activateWindow()
                        return
                except Exception:
                    pass

            parent = self._dialog_parent()
            dlg = MniParcellationTableDialog(self, parent=parent)

            # Non-modal: do NOT use exec().
            dlg.setWindowModality(Qt.NonModal)

            self._mni_parcellation_table_dialog = dlg

            dlg.show()
            dlg.raise_()
            dlg.activateWindow()

        except Exception as e:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(),
                "MNI parcellation table",
                f"Could not open parcellation table:\n{e}",
            )

    def _show_3d_context_menu(self, pos) -> None:
        if bool(getattr(self, "_view3d_is_fullscreen", False)):
            return

        if self.interactor is None:
            return

        try:
            lh_loaded = bool(self._lh_pial_poly is not None or self._lh_pial_mask_img is not None)

            rh_loaded = bool(self._rh_pial_poly is not None or self._rh_pial_mask_img is not None)

            global_pos = self.interactor.mapToGlobal(pos)

            pial_checked = self._is_pial_surface_checked()
            pet_or_siscom_checked = self._is_pet_or_siscom_checked()
            mni_checked = self._is_mni_atlas_checked()

            clicked_marker_id = self._pick_marker_id_from_qpos(pos)

            clicked_slice_ras = None

            if clicked_marker_id is None:
                clicked_slice_ras = self._pick_visible_slice_ras_from_qpos(pos)

            has_hidden_markers = any(
                not bool(marker.get("visible", True)) for marker in self._markers()
            )

            choice = exec_3d_view_menu(
                global_pos,
                has_lh=bool(lh_loaded and pial_checked),
                has_rh=bool(rh_loaded and pial_checked),
                show_lh=getattr(self, "_show_lh_pial", True),
                show_rh=getattr(self, "_show_rh_pial", True),
                color_scale_visible=bool(getattr(self, "_show_color_scales", True)),
                show_pial_options=bool(pial_checked),
                show_color_scale_option=bool(pet_or_siscom_checked),
                show_mni_load_option=bool(mni_checked),
                show_mni_t1_option=False,
                mni_t1_visible=bool(getattr(self, "_mni_t1_slices_visible", False)),
                native_actions_enabled=not bool(mni_checked),
                show_mni_parcellation_table_option=bool(mni_checked),
                show_keep_electrodes_through_slices_option=self._show_keep_electrodes_through_slices_option(),
                keep_electrodes_through_slices=bool(
                    getattr(self, "_keep_electrodes_visible_through_slices", False)
                ),
                show_slice_plane_frames_option=self._slice_plane_frames_option_available(),
                slice_plane_frames_visible=bool(getattr(self, "_slice_plane_frames_visible", True)),
                can_add_marker=bool(clicked_slice_ras is not None),
                marker_under_cursor=bool(clicked_marker_id is not None),
                has_hidden_markers=bool(has_hidden_markers),
            )
            if choice == "marker_list":
                self._open_marker_list_dialog()

            elif choice == "add_marker" and clicked_slice_ras is not None:
                self._create_marker_at_ras(clicked_slice_ras)

            elif choice == "edit_marker" and clicked_marker_id is not None:
                self._edit_marker(clicked_marker_id)

            elif choice == "hide_marker" and clicked_marker_id is not None:
                self._hide_marker(clicked_marker_id)

            elif choice == "export_marker" and clicked_marker_id is not None:
                self._export_marker_text(clicked_marker_id)

            elif choice == "delete_marker" and clicked_marker_id is not None:
                self._delete_marker(clicked_marker_id)

            elif choice == "show_hidden_markers":
                self._show_hidden_markers()

            elif choice == "toggle_slice_plane_frames":
                self._toggle_slice_plane_frames_visible()

            elif choice == "pet":
                self._choose_pet_colormap()

            elif choice == "siscom":
                self._choose_siscom_colormap()

            elif choice == "ct":
                self._choose_modality_color("ct")

            elif choice == "render_brain":
                self._open_brain_render_dialog()

            elif choice == "load_mni_electrodes":
                self.open_mni_electrodes_files()

            elif choice == "mni_parcellation_table":
                self._open_mni_parcellation_table()

            elif choice == "toggle_lh":
                self._toggle_lh_pial_visibility()

            elif choice == "toggle_rh":
                self._toggle_rh_pial_visibility()

            elif choice == "toggle_color_scale":
                self._toggle_color_scales()

            elif choice == "toggle_keep_electrodes_through_slices":
                self._toggle_keep_electrodes_visible_through_slices()

        except Exception as e:
            print("[3D context menu] failed:", e)

    def _choose_modality_color(self, which: str) -> None:
        try:
            if which == "pet":
                init = self._tuple_to_qcolor(self._pet_color)
                title = "Choose PET color"

            elif which == "siscom":
                init = self._tuple_to_qcolor(self._siscom_color)
                title = "Choose SISCOM color"

            elif which == "ct":
                init = self._tuple_to_qcolor(self._ct_color)
                title = "Choose CT color"

            elif which == "brain":
                init = self._tuple_to_qcolor(self._brain_color)
                title = "Choose Brain color"

            else:
                return

            color_hex = NeuXelecColorDialog.get_color(
                initial_color=init,
                parent=self._dialog_parent(),
                title=title,
            )

            if color_hex is None:
                return

            color = QColor(color_hex)

            if not color.isValid():
                return

            rgb = self._qcolor_to_tuple(color)

            if which == "pet":
                self._pet_color = rgb
                self._render_pet()

            elif which == "siscom":
                self._siscom_color = rgb
                self._render_siscom()

            elif which == "ct":
                self._ct_color = rgb
                self._render_ct()

            elif which == "brain":
                self._brain_color = rgb
                self._render_brain()

            self._refresh_multiplanar_clipped_scene()

        except Exception:
            pass

    def _tuple_to_qcolor(self, rgb) -> QColor:
        r = int(np.clip(rgb[0] * 255.0, 0, 255))
        g = int(np.clip(rgb[1] * 255.0, 0, 255))
        b = int(np.clip(rgb[2] * 255.0, 0, 255))
        return QColor(r, g, b)

    def _qcolor_to_tuple(self, color: QColor):
        return (
            float(color.red()) / 255.0,
            float(color.green()) / 255.0,
            float(color.blue()) / 255.0,
        )

    def _tuple01_to_rgb255(self, rgb):
        return (
            int(np.clip(rgb[0] * 255.0, 0, 255)),
            int(np.clip(rgb[1] * 255.0, 0, 255)),
            int(np.clip(rgb[2] * 255.0, 0, 255)),
        )

    def set_pial_surfaces(
        self, lh_path: str | None, rh_path: str | None, assume_lps: bool = False
    ) -> None:
        self._pial_assume_lps = bool(assume_lps)
        self._invalidate_slice_volume_cache()

        self._lh_pial_poly = (
            self._load_freesurfer_surface_as_polydata(
                lh_path,
                assume_lps=self._pial_assume_lps,
            )
            if lh_path
            else None
        )

        self._rh_pial_poly = (
            self._load_freesurfer_surface_as_polydata(
                rh_path,
                assume_lps=self._pial_assume_lps,
            )
            if rh_path
            else None
        )

        self._show_lh_pial = True
        self._show_rh_pial = True

        if self.chk_pial is not None:
            self._set_checked(self.chk_pial, True)
        if self.chk_brainmask is not None:
            self._set_checked(self.chk_brainmask, False)
        if self.chk_iso is not None:
            self._set_checked(self.chk_iso, False)

        if self.sld_3d_PialOpacity is not None:
            self.sld_3d_PialOpacity.blockSignals(True)
            self.sld_3d_PialOpacity.setValue(50)
            self.sld_3d_PialOpacity.blockSignals(False)

        self._update_brain_opacity_slider_states()
        self._render_brain()

        try:
            self._update_modality_controls_enabled_states()
        except Exception:
            pass

        try:
            if self.chk_pial is not None and self.chk_pial.isChecked():
                self._render_brain()
        except Exception:
            pass

        try:
            self._render_surface_projections()
        except Exception:
            pass

    def set_pial_masks(self, lh_img=None, rh_img=None) -> None:
        self._lh_pial_mask_img = lh_img
        self._rh_pial_mask_img = rh_img
        self._invalidate_slice_volume_cache()

        try:
            self._update_all_plane_slider_ranges()
        except Exception:
            pass

        try:
            self._refresh_multiplanar_clipped_scene()
        except Exception:
            pass

        try:
            if self.chk_pial is not None and self.chk_pial.isChecked():
                self._render_brain()
                self._render_coronal_plane()
                self._render_axial_plane()
                self._render_sagittal_plane()
                self._render_pet()
                self._render_ct()
                self._render_siscom()
                self.render_all_surface_projections()
        except Exception:
            pass

        try:
            self._render_surface_projections()
        except Exception:
            pass

    def _open_brain_render_dialog(self):
        try:
            if self._brain_render_dialog is None:
                self._brain_render_dialog = BrainRenderDialog(
                    self, self.ui.window() if self.ui is not None else None
                )
            self._brain_render_dialog.show()
            self._brain_render_dialog.raise_()
            self._brain_render_dialog.activateWindow()
        except Exception:
            pass

    def _setup_brain_render_lights(self) -> None:
        if not _PV_OK or self.plotter is None:
            return

        p = getattr(self, "_brain_render_params", {})

        try:
            self.plotter.set_background("#2b2d31")
        except Exception:
            pass

        try:
            self.plotter.remove_all_lights()
        except Exception:
            pass

        try:
            # Main global light tied to camera: illuminates the whole visible brain
            head = pv.Light(light_type="headlight")
            head.intensity = float(p.get("key_light", 1.0))
            self.plotter.add_light(head)
        except Exception:
            pass

        try:
            # Gentle symmetric fills to avoid strong shadows
            fill1 = pv.Light(
                position=(250, 250, 250),
                focal_point=(0, 0, 0),
                intensity=float(p.get("fill_light", 0.25)),
            )
            fill2 = pv.Light(
                position=(-250, -250, 250),
                focal_point=(0, 0, 0),
                intensity=float(p.get("back_light", 0.20)),
            )
            self.plotter.add_light(fill1)
            self.plotter.add_light(fill2)
        except Exception:
            pass

        # For a global uniform render, shadows usually hurt more than help
        # So do not enable shadows here

    def _toggle_lh_pial_visibility(self) -> None:
        """
        Show or hide the left pial hemisphere without changing the current camera.
        """
        self._show_lh_pial = not bool(getattr(self, "_show_lh_pial", True))

        # The active pial mask changes when a hemisphere is hidden/shown.
        self._invalidate_slice_volume_cache()

        try:
            if self.chk_pial is not None and self.chk_pial.isChecked():
                self._render_brain(reset_camera=False)
            else:
                self._render()
        except Exception as e:
            print("[3D View] Failed to toggle LH pial visibility:", e)

    def _toggle_rh_pial_visibility(self) -> None:
        """
        Show or hide the right pial hemisphere without changing the current camera.
        """
        self._show_rh_pial = not bool(getattr(self, "_show_rh_pial", True))

        # The active pial mask changes when a hemisphere is hidden/shown.
        self._invalidate_slice_volume_cache()

        try:
            if self.chk_pial is not None and self.chk_pial.isChecked():
                self._render_brain(reset_camera=False)
            else:
                self._render()
        except Exception as e:
            print("[3D View] Failed to toggle RH pial visibility:", e)

    def _update_brain_opacity_slider_states(self) -> None:
        mni_on = bool(self.chk_mni_atlas is not None and self.chk_mni_atlas.isChecked())
        brainmask_on = bool(self.chk_brainmask is not None and self.chk_brainmask.isChecked())
        pial_on = bool(self.chk_pial is not None and self.chk_pial.isChecked())

        # In MNI mode, the MNI atlas opacity is controlled by the Brain mask opacity slider.
        if getattr(self, "sld_3d_brainMaskOpacity", None) is not None:
            self.sld_3d_brainMaskOpacity.setEnabled(bool(mni_on or brainmask_on))

        # Pial opacity is only for native pial surface, not MNI atlas.
        if getattr(self, "sld_3d_PialOpacity", None) is not None:
            self.sld_3d_PialOpacity.setEnabled(bool((not mni_on) and pial_on))

    def _choose_pet_colormap(self) -> None:
        options = [
            "hot",
            "inferno",
            "plasma",
            "jet",
            "turbo",
            "viridis",
            "gray",
        ]

        current_index = (
            options.index(self._pet_colormap_name) if self._pet_colormap_name in options else 0
        )

        cmap = NeuXelecSelectionDialog.select_item(
            self._dialog_parent(),
            "PET colormap",
            "Choose the color scale used for the PET overlay:",
            options=options,
            current_index=current_index,
            accept_text="Apply",
            reject_text="Cancel",
        )

        if not cmap:
            return

        self._pet_colormap_name = str(cmap)
        self._remove_pet_scalar_bar()
        self._refresh_pet_only()

    def _choose_siscom_colormap(self) -> None:
        options = [
            "hot",
            "inferno",
            "plasma",
            "jet",
            "turbo",
            "viridis",
            "gray",
        ]

        current_index = (
            options.index(self._siscom_colormap_name)
            if self._siscom_colormap_name in options
            else 0
        )

        cmap = NeuXelecSelectionDialog.select_item(
            self._dialog_parent(),
            "SISCOM colormap",
            "Choose the color scale used for the SISCOM overlay:",
            options=options,
            current_index=current_index,
            accept_text="Apply",
            reject_text="Cancel",
        )

        if not cmap:
            return

        self._siscom_colormap_name = str(cmap)
        self._remove_siscom_scalar_bar()
        self._refresh_siscom_only()

    def _ray_surface_intersection(self, surf, origin_ras: np.ndarray, direction_ras: np.ndarray):
        try:
            o = np.asarray(origin_ras, dtype=np.float64)
            d = np.asarray(direction_ras, dtype=np.float64)
            nd = np.linalg.norm(d)
            if nd < 1e-6:
                return None, None

            d = d / nd
            p1 = o
            p2 = o + 200.0 * d  # 200 mm, largement suffisant pour sortir du cerveau

            points, ind = surf.ray_trace(p1, p2, first_point=False)
            if points is None or len(points) == 0:
                return None, None

            pts = np.asarray(points, dtype=np.float64)
            vecs = pts - o[None, :]
            proj = vecs @ d
            valid = proj > 0.0
            if not np.any(valid):
                return None, None

            pts = pts[valid]
            proj = proj[valid]
            k = int(np.argmin(proj))
            hit = pts[k]

            pid = None
            try:
                pid = int(surf.find_closest_point(hit))
            except Exception:
                pid = None

            return hit, pid
        except Exception:
            return None, None

    def _slider_value_for_plane_index(self, plane_name: str, voxel_index: int) -> int | None:
        try:
            idx = int(voxel_index)
        except Exception:
            return None

        if plane_name == "coronal":
            ymin = int(self._coronal_y_min) if self._coronal_y_min is not None else 0
            ymax = int(self._coronal_y_max) if self._coronal_y_max is not None else ymin
            idx = int(np.clip(idx, ymin, ymax))
            return int(idx - ymin) if self._coronal_from_caudal else int(ymax - idx)

        if plane_name == "axial":
            zmin = int(self._axial_z_min) if self._axial_z_min is not None else 0
            zmax = int(self._axial_z_max) if self._axial_z_max is not None else zmin
            idx = int(np.clip(idx, zmin, zmax))
            return int(idx - zmin) if self._axial_from_inferior else int(zmax - idx)

        if plane_name == "sagittal":
            xmin = int(self._sagittal_x_min) if self._sagittal_x_min is not None else 0
            xmax = int(self._sagittal_x_max) if self._sagittal_x_max is not None else xmin
            idx = int(np.clip(idx, xmin, xmax))
            return int(idx - xmin) if self._sagittal_from_left else int(xmax - idx)

        return None

    def _clear_all_slice_plane_actors(self) -> None:
        """
        Remove every slice-related actor from the current 3D scene.

        This is required when switching between MNI and native space because
        the plane checkboxes are often changed with signals blocked.
        """
        for actor_name in (
            # Anatomical slices
            "coronal_plane",
            "axial_plane",
            "sagittal_plane",
            # PET overlays
            "coronal_pet",
            "axial_pet",
            "sagittal_pet",
            # SISCOM overlays
            "coronal_siscom",
            "axial_siscom",
            "sagittal_siscom",
            # Electrode overlays drawn on slices
            "coronal_elec",
            "axial_elec",
            "sagittal_elec",
            # Colored slice frames
            "coronal_outline",
            "axial_outline",
            "sagittal_outline",
        ):
            try:
                self._remove_actor(actor_name)
            except Exception:
                pass

        # Remove any stale source meshes built from the MNI template.
        self._coronal_plane_source_mesh = None
        self._axial_plane_source_mesh = None
        self._sagittal_plane_source_mesh = None

        self._coronal_pet_source_mesh = None
        self._axial_pet_source_mesh = None
        self._sagittal_pet_source_mesh = None

        self._coronal_siscom_source_mesh = None
        self._axial_siscom_source_mesh = None
        self._sagittal_siscom_source_mesh = None

        try:
            self._remove_all_slice_plane_frame_actors()
        except Exception:
            pass

    def _hide_plane_and_overlays(self, plane_name: str) -> None:
        plane_name = str(plane_name).lower().strip()
        try:
            self._clear_slice_focused_contact_label(plane_name)
        except Exception:
            pass
        try:
            if plane_name == "coronal":
                if self.chk_coronal_plane is not None:
                    self.chk_coronal_plane.blockSignals(True)
                    self.chk_coronal_plane.setChecked(False)
                    self.chk_coronal_plane.blockSignals(False)

                self._remove_actor("coronal_plane")
                self._remove_actor("coronal_pet")
                self._remove_actor("coronal_siscom")
                self._remove_actor("coronal_outline")
                self._remove_actor("coronal_elec")

            elif plane_name == "axial":
                if self.chk_axial_plane is not None:
                    self.chk_axial_plane.blockSignals(True)
                    self.chk_axial_plane.setChecked(False)
                    self.chk_axial_plane.blockSignals(False)

                self._remove_actor("axial_plane")
                self._remove_actor("axial_pet")
                self._remove_actor("axial_siscom")
                self._remove_actor("axial_outline")
                self._remove_actor("axial_elec")

            elif plane_name == "sagittal":
                if self.chk_sagittal_plane is not None:
                    self.chk_sagittal_plane.blockSignals(True)
                    self.chk_sagittal_plane.setChecked(False)
                    self.chk_sagittal_plane.blockSignals(False)

                self._remove_actor("sagittal_plane")
                self._remove_actor("sagittal_pet")
                self._remove_actor("sagittal_siscom")
                self._remove_actor("sagittal_outline")
                self._remove_actor("sagittal_elec")
        except Exception:
            pass

    def _show_only_requested_plane(self, plane_name: str) -> None:
        plane_name = str(plane_name).lower().strip()

        for other in ("coronal", "axial", "sagittal"):
            if other != plane_name:
                self._hide_plane_and_overlays(other)

        try:
            if plane_name == "coronal" and self.chk_coronal_plane is not None:
                self.chk_coronal_plane.blockSignals(True)
                self.chk_coronal_plane.setChecked(True)
                self.chk_coronal_plane.blockSignals(False)

            elif plane_name == "axial" and self.chk_axial_plane is not None:
                self.chk_axial_plane.blockSignals(True)
                self.chk_axial_plane.setChecked(True)
                self.chk_axial_plane.blockSignals(False)

            elif plane_name == "sagittal" and self.chk_sagittal_plane is not None:
                self.chk_sagittal_plane.blockSignals(True)
                self.chk_sagittal_plane.setChecked(True)
                self.chk_sagittal_plane.blockSignals(False)
        except Exception:
            pass

    def _clear_slice_focused_contact_label(self, plane_name: str | None = None) -> None:
        """
        Remove only the temporary contact label created by
        'Show coronal/axial/sagittal slice'.

        Manually added labels remain untouched.
        """
        focused = getattr(self, "_slice_focused_contact_label", None)

        if not isinstance(focused, dict):
            return

        try:
            focused_plane = str(focused.get("plane", "")).lower().strip()

            if plane_name is not None:
                requested_plane = str(plane_name).lower().strip()
                if focused_plane != requested_plane:
                    return

            elec_id = int(focused["elec_id"])
            contact_idx = int(focused["contact_idx"])

            elec = self.state.electrodes[elec_id]
            n_contacts = len(elec.get("contacts_lps", []) or [])

            vals = self._get_local_contact_labels_visible(elec_id, n_contacts)

            if 0 <= contact_idx < len(vals):
                vals[contact_idx] = False

            self._slice_focused_contact_label = None

            if not bool(getattr(self, "_suspend_electrode_refresh", False)):
                self._render_single_electrode(elec_id)

        except Exception:
            self._slice_focused_contact_label = None

    def show_contact_in_slice(self, elec_id: int, contact_idx: int, plane_name: str) -> None:
        self._suspend_electrode_refresh = True
        try:
            electrodes = getattr(self.state, "electrodes", None) or []
            if not (0 <= int(elec_id) < len(electrodes)):
                return
            elec = electrodes[int(elec_id)]

            contacts_idx = elec.get("contacts_idx", []) or []
            if not (0 <= int(contact_idx) < len(contacts_idx)):
                return

            idx_xyz = contacts_idx[int(contact_idx)]
            if idx_xyz is None or len(idx_xyz) < 3:
                return

            plane_name = str(plane_name).lower().strip()

            if plane_name == "coronal":
                target_idx = int(idx_xyz[1])
                slider = self.sld_coronal_plane
            elif plane_name == "axial":
                target_idx = int(idx_xyz[2])
                slider = self.sld_axial_plane
            elif plane_name == "sagittal":
                target_idx = int(idx_xyz[0])
                slider = self.sld_sagittal_plane
            else:
                return

            self._show_only_requested_plane(plane_name)

            if slider is not None:
                slider_value = self._slider_value_for_plane_index(plane_name, target_idx)
                if slider_value is not None:
                    slider_value = int(np.clip(slider_value, slider.minimum(), slider.maximum()))
                    slider.blockSignals(True)
                    slider.setValue(slider_value)
                    slider.blockSignals(False)

            # Remove the previous temporary slice-focus label, if any.
            self._clear_slice_focused_contact_label()

            # Keep your current behavior: show only the selected contact label.
            self._clear_all_contact_labels()
            self.set_contact_label_visible(int(elec_id), int(contact_idx), True)

            # Remember that this label belongs to the slice-focus action.
            self._slice_focused_contact_label = {
                "plane": str(plane_name),
                "elec_id": int(elec_id),
                "contact_idx": int(contact_idx),
            }

        except Exception:
            pass
        finally:
            self._suspend_electrode_refresh = False

        try:
            self._update_plane_slider_enabled_states()
            self._refresh_single_plane_full(plane_name)
            self.update_electrodes()
            self._apply_actor_clipping()
            self._update_planes_info_label()
            self._render()
        except Exception:
            pass

    def show_mni_contact_in_slice(
        self,
        set_index: int,
        contact_index: int,
        plane_name: str,
    ) -> None:
        """
        Display only one requested MNI slice through one MNI contact.

        Any previously displayed contact-focused slice is replaced.
        """
        try:
            set_index = int(set_index)
            contact_index = int(contact_index)
            plane_name = str(plane_name).lower().strip()

            if plane_name not in (
                "coronal",
                "axial",
                "sagittal",
            ):
                return

            sets = (
                getattr(
                    self.state,
                    "mni_electrode_sets",
                    [],
                )
                or []
            )

            if not (0 <= set_index < len(sets)):
                return

            mni_set = sets[set_index]
            self._ensure_mni_visibility_fields(mni_set)

            contacts = mni_set.get("contacts", []) or []

            if not (0 <= contact_index < len(contacts)):
                return

            contact = contacts[contact_index]

            try:
                x_ras, y_ras, z_ras = contact.get(
                    "mni_ras",
                    [None, None, None],
                )

                ras = np.asarray(
                    [
                        float(x_ras),
                        float(y_ras),
                        float(z_ras),
                    ],
                    dtype=np.float64,
                )
            except Exception:
                return

            template_t1 = self._get_mni_template_t1_image()

            if template_t1 is None:
                NeuXelecMessageDialog.warning(
                    self._dialog_parent(),
                    "MNI slice",
                    "The MNI template T1 could not be loaded.",
                )
                return

            # Ensure the MNI T1 is the active slice reference.
            self._mni_t1_slices_visible = True
            self._invalidate_slice_volume_cache(
                base=True,
                pet=True,
                siscom=True,
            )
            self._update_all_plane_slider_ranges()

            # SimpleITK uses LPS physical coordinates.
            lps = ras.copy()
            lps[0] *= -1.0
            lps[1] *= -1.0

            try:
                idx_xyz = template_t1.TransformPhysicalPointToIndex(tuple(float(v) for v in lps))
            except Exception:
                return

            size = template_t1.GetSize()

            inside = (
                0 <= idx_xyz[0] < size[0]
                and 0 <= idx_xyz[1] < size[1]
                and 0 <= idx_xyz[2] < size[2]
            )

            if not inside:
                NeuXelecMessageDialog.warning(
                    self._dialog_parent(),
                    "MNI slice",
                    "The selected contact is outside the MNI template.",
                )
                return

            if plane_name == "coronal":
                target_idx = int(idx_xyz[1])
                slider = self.sld_coronal_plane

            elif plane_name == "axial":
                target_idx = int(idx_xyz[2])
                slider = self.sld_axial_plane

            else:
                target_idx = int(idx_xyz[0])
                slider = self.sld_sagittal_plane

            # Remove the previous contact-focused label.
            self._clear_mni_slice_focused_contact()

            # Display only the newly requested slice.
            self._show_only_requested_plane(plane_name)

            if slider is not None:
                slider_value = self._slider_value_for_plane_index(
                    plane_name,
                    target_idx,
                )

                if slider_value is not None:
                    slider_value = int(
                        np.clip(
                            slider_value,
                            slider.minimum(),
                            slider.maximum(),
                        )
                    )

                    slider.blockSignals(True)
                    slider.setValue(slider_value)
                    slider.blockSignals(False)

            # Display only this temporary contact label.
            group_name = self._mni_group_name_from_contact(contact)

            mni_set["contact_label_visible"][str(contact_index)] = True

            self._mni_slice_focused_contact = {
                "set_index": set_index,
                "contact_index": contact_index,
                "group_name": group_name,
                "plane": plane_name,
            }

            self._render_single_mni_group(
                set_index,
                group_name,
            )

            self._update_plane_slider_enabled_states()
            self._refresh_single_plane_full(plane_name)
            self._apply_actor_clipping()
            self._update_planes_info_label()
            self._render()

        except Exception as e:
            print("[MNI contact slice] failed:", e)

    def _clear_all_contact_labels(self) -> None:
        try:
            electrodes = getattr(self.state, "electrodes", None) or []
            for elec_id, elec in enumerate(electrodes):
                contacts_idx = elec.get("contacts_idx", []) or []
                for contact_idx in range(len(contacts_idx)):
                    try:
                        self.set_contact_label_visible(elec_id, contact_idx, False)
                    except Exception:
                        pass
        except Exception:
            pass

    def _remove_pet_scalar_bar_name(self) -> None:
        if self.plotter is None:
            return

        try:
            keys = list(getattr(self.plotter, "scalar_bars", {}).keys())
        except Exception:
            keys = []

        for key in keys:
            try:
                k = str(key)
                if k.startswith("PET"):
                    self.plotter.remove_scalar_bar(k)
            except Exception:
                pass

        try:
            self.plotter.remove_scalar_bar("pet_scalar_bar_dummy")
        except Exception:
            pass

    def _remove_siscom_scalar_bar_name(self) -> None:
        if self.plotter is None:
            return

        try:
            keys = list(getattr(self.plotter, "scalar_bars", {}).keys())
        except Exception:
            keys = []

        for key in keys:
            try:
                k = str(key)
                if k.startswith("SISCOM"):
                    self.plotter.remove_scalar_bar(k)
            except Exception:
                pass

        try:
            self.plotter.remove_scalar_bar("siscom_scalar_bar_dummy")
        except Exception:
            pass

    def _remove_pet_scalar_bar(self) -> None:
        if self.plotter is None:
            return

        try:
            if self._pet_scalar_bar_actor is not None:
                self.plotter.remove_actor(self._pet_scalar_bar_actor, reset_camera=False)
        except Exception:
            pass

        self._pet_scalar_bar_actor = None
        self._remove_pet_scalar_bar_name()

    def _remove_siscom_scalar_bar(self) -> None:
        if self.plotter is None:
            return

        try:
            if self._siscom_scalar_bar_actor is not None:
                self.plotter.remove_actor(self._siscom_scalar_bar_actor, reset_camera=False)
        except Exception:
            pass

        self._siscom_scalar_bar_actor = None
        self._remove_siscom_scalar_bar_name()

    def _get_siscom_display_range(self):
        zmin = float(self.dsb_siscom_z.value()) if self.dsb_siscom_z is not None else 2.0

        zmax = self._siscom_fixed_zmax
        if zmax is None or not np.isfinite(zmax) or zmax <= zmin:
            zmax = zmin + 1.0

        return float(zmin), float(zmax)

    def _update_siscom_scalar_bar(self) -> None:
        if not _PV_OK or self.plotter is None:
            return
        if not bool(getattr(self, "_show_color_scales", True)):
            self._remove_pet_scalar_bar()
            self._render()
            return
        show = bool(self.chk_siscom is not None and self.chk_siscom.isChecked())
        if not show or self._siscom_img is None:
            self._remove_siscom_scalar_bar()
            self._render()
            return

        try:
            zmin, zmax = self._get_siscom_display_range()

            layout = self._scalar_bar_layout()
            sis_layout = layout["siscom"]
            if sis_layout is None:
                self._remove_siscom_scalar_bar()
                self._render()
                return

            sb = None
            try:
                for key, val in getattr(self.plotter, "scalar_bars", {}).items():
                    if str(key).startswith("SISCOM"):
                        sb = val
                        break
            except Exception:
                sb = None

            if self._siscom_scalar_bar_actor is None or sb is None:
                self._remove_siscom_scalar_bar()

                dummy = pv.PolyData(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
                dummy["SISCOM_Z"] = np.array([zmin], dtype=np.float32)

                self._siscom_scalar_bar_actor = self.plotter.add_mesh(
                    dummy,
                    scalars="SISCOM_Z",
                    cmap=self._siscom_colormap_name,
                    clim=[float(zmin), float(zmax)],
                    opacity=0.0,
                    show_scalar_bar=True,
                    scalar_bar_args={
                        "title": "SISCOM Z",
                        "vertical": True,
                        "position_x": sis_layout["position_x"],
                        "position_y": sis_layout["position_y"],
                        "width": sis_layout["width"],
                        "height": sis_layout["height"],
                        "fmt": "%.2f",
                        "title_font_size": 12,
                        "label_font_size": 10,
                        "color": "white",
                        "n_labels": 5,
                    },
                    name="siscom_scalar_bar_dummy",
                )
            else:
                try:
                    mapper = self._siscom_scalar_bar_actor.GetMapper()
                    if mapper is not None:
                        mapper.SetScalarRange(float(zmin), float(zmax))
                except Exception:
                    pass

                try:
                    sb.SetTitle("SISCOM Z")
                except Exception:
                    pass

                try:
                    sb.SetNumberOfLabels(5)
                except Exception:
                    pass

        except Exception:
            pass

        self._render()

    def _update_pet_scalar_bar(self) -> None:
        if not _PV_OK or self.plotter is None:
            return

        if not bool(getattr(self, "_show_color_scales", True)):
            self._remove_pet_scalar_bar()
            self._render()
            return

        show = bool(self.chk_pet is not None and self.chk_pet.isChecked())
        if not show or self._pet_img is None:
            self._remove_pet_scalar_bar()
            self._render()
            return

        try:
            pet_np = sitk.GetArrayFromImage(self._pet_img).astype(np.float32)
            finite = np.isfinite(pet_np)
            vals = pet_np[finite]
            vals = vals[vals > 0]

            if vals.size == 0:
                self._remove_pet_scalar_bar()
                self._render()
                return

            pmin, pmax = self._get_pet_minmax_percentiles()
            gamma = self._get_pet_gamma_value()
            lo, hi = get_pet_window(vals, pmin, pmax)

            layout = self._scalar_bar_layout()
            pet_layout = layout["pet"]
            if pet_layout is None:
                self._remove_pet_scalar_bar()
                self._render()
                return

            if hi <= lo:
                hi = lo + 1.0

            # check whether the visible scalar bar still exists
            sb = None
            try:
                for key, val in getattr(self.plotter, "scalar_bars", {}).items():
                    if str(key).startswith("PET"):
                        sb = val
                        break
            except Exception:
                sb = None

            if self._pet_scalar_bar_actor is None or sb is None:
                self._remove_pet_scalar_bar()

                dummy = pv.PolyData(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
                dummy["PET"] = np.array([lo], dtype=np.float32)

                self._pet_scalar_bar_actor = self.plotter.add_mesh(
                    dummy,
                    scalars="PET",
                    cmap=self._pet_colormap_name,
                    clim=[float(lo), float(hi)],
                    opacity=0.0,
                    show_scalar_bar=True,
                    scalar_bar_args={
                        "title": f"PET (γ={gamma:.2f})",
                        "vertical": True,
                        "position_x": pet_layout["position_x"],
                        "position_y": pet_layout["position_y"],
                        "width": pet_layout["width"],
                        "height": pet_layout["height"],
                        "fmt": "%.2f",
                        "title_font_size": 12,
                        "label_font_size": 10,
                        "color": "white",
                        "n_labels": 5,
                    },
                    name="pet_scalar_bar_dummy",
                )
            else:
                try:
                    mapper = self._pet_scalar_bar_actor.GetMapper()
                    if mapper is not None:
                        mapper.SetScalarRange(float(lo), float(hi))
                except Exception:
                    pass

                try:
                    sb.SetTitle(f"PET (γ={gamma:.2f})")
                except Exception:
                    pass

                try:
                    sb.SetNumberOfLabels(5)
                except Exception:
                    pass

        except Exception:
            pass

        self._render()

    def _scalar_bar_layout(self):
        show_pet = bool(
            self.chk_pet is not None and self.chk_pet.isChecked() and self._pet_img is not None
        )
        show_sis = bool(
            self.chk_siscom is not None
            and self.chk_siscom.isChecked()
            and self._siscom_img is not None
        )

        pet_x = 0.84
        sis_x = 0.91
        y = 0.12
        w = 0.055
        h = 0.76

        return {
            "pet": (
                {"position_x": pet_x, "position_y": y, "width": w, "height": h}
                if show_pet
                else None
            ),
            "siscom": (
                {"position_x": sis_x, "position_y": y, "width": w, "height": h}
                if show_sis
                else None
            ),
        }

    def _render_visible_slice_planes_only(self) -> None:
        try:
            if self.chk_coronal_plane is not None and self.chk_coronal_plane.isChecked():
                self._render_coronal_plane()
        except Exception:
            pass

        try:
            if self.chk_axial_plane is not None and self.chk_axial_plane.isChecked():
                self._render_axial_plane()
        except Exception:
            pass

        try:
            if self.chk_sagittal_plane is not None and self.chk_sagittal_plane.isChecked():
                self._render_sagittal_plane()
        except Exception:
            pass

    def _on_pet_toggled(self, checked: bool) -> None:
        if bool(checked):
            self._show_color_scales = True
        self._refresh_pet_only()

    def _on_siscom_toggled(self, checked: bool) -> None:
        if bool(checked):
            self._show_color_scales = True
        self._refresh_siscom_only()

    def _refresh_pet_only(self) -> None:
        self._invalidate_slice_volume_cache(base=False, pet=True, siscom=False)

        try:
            self._render_pet()
        except Exception:
            pass

        try:
            self._render_visible_pet_overlays_only()
        except Exception:
            pass

        try:
            self._update_pet_scalar_bar()
        except Exception:
            pass

        try:
            self._render()
        except Exception:
            pass

    def _refresh_siscom_only(self) -> None:
        self._invalidate_slice_volume_cache(base=False, pet=False, siscom=True)

        try:
            self._render_siscom()
        except Exception:
            pass

        try:
            self._render_visible_siscom_overlays_only()
        except Exception:
            pass

        try:
            self._update_siscom_scalar_bar()
        except Exception:
            pass

        try:
            self._render()
        except Exception:
            pass

    def _update_visible_pet_overlay_opacity_only(self) -> None:
        try:
            op = 1.0
            if self.sld_pet_opacity is not None:
                op = float(np.clip(float(self.sld_pet_opacity.value()) / 100.0, 0.0, 1.0))

            for actor in (
                getattr(self, "_coronal_pet_actor", None),
                getattr(self, "_axial_pet_actor", None),
                getattr(self, "_sagittal_pet_actor", None),
            ):
                if actor is not None:
                    try:
                        actor.GetProperty().SetOpacity(op)
                    except Exception:
                        pass

            self._render()
        except Exception:
            pass

    def _update_visible_siscom_overlay_opacity_only(self) -> None:
        try:
            op = 1.0
            if self.sld_siscom_opacity is not None:
                op = float(np.clip(float(self.sld_siscom_opacity.value()) / 100.0, 0.0, 1.0))

            for actor in (
                getattr(self, "_coronal_siscom_actor", None),
                getattr(self, "_axial_siscom_actor", None),
                getattr(self, "_sagittal_siscom_actor", None),
            ):
                if actor is not None:
                    try:
                        actor.GetProperty().SetOpacity(op)
                    except Exception:
                        pass

            self._render()
        except Exception:
            pass

    def _refresh_pet_scalar_bar_only(self) -> None:
        try:
            self._update_pet_scalar_bar()
        except Exception:
            pass
        try:
            self._render()
        except Exception:
            pass

    def _refresh_siscom_scalar_bar_only(self) -> None:
        try:
            self._update_siscom_scalar_bar()
        except Exception:
            pass
        try:
            self._render()
        except Exception:
            pass

    def _build_contact_disc_mesh(
        self,
        center_ras: np.ndarray,
        plane_normal_ras: np.ndarray,
        radius_mm: float = 1.0,
        n_sides: int = 40,
        offset_mm: float = 0.1,
    ):
        """
        Build a flat circular disc lying in the slice plane.
        The disc is slightly offset along the plane normal to avoid z-fighting
        with the texture slice.
        """
        try:
            c = np.asarray(center_ras, dtype=np.float64)
            n = np.asarray(plane_normal_ras, dtype=np.float64)

            nn = np.linalg.norm(n)
            if nn < 1e-6:
                n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                n = n / nn

            c = c + float(offset_mm) * n

            disc = pv.Disc(
                center=c,
                inner=0.0,
                outer=float(radius_mm),
                normal=n,
                r_res=1,
                c_res=int(max(12, n_sides)),
            )
            return disc
        except Exception:
            return None

    def _contact_is_on_any_visible_slice(self, idx_xyz, tol: float = 0.49) -> bool:
        try:
            # In keep-through-slices mode, contacts stay as true 3D actors.
            # They are not replaced by 2D discs on the slice planes.
            if bool(getattr(self, "_keep_electrodes_visible_through_slices", False)):
                return False

            if idx_xyz is None or len(idx_xyz) < 3:
                return False

            ix, iy, iz = float(idx_xyz[0]), float(idx_xyz[1]), float(idx_xyz[2])

            if self.chk_coronal_plane is not None and self.chk_coronal_plane.isChecked():
                y_idx = self._get_coronal_slice_index()
                if y_idx is not None and abs(iy - float(y_idx)) <= tol:
                    return True

            if self.chk_axial_plane is not None and self.chk_axial_plane.isChecked():
                z_idx = self._get_axial_slice_index()
                if z_idx is not None and abs(iz - float(z_idx)) <= tol:
                    return True

            if self.chk_sagittal_plane is not None and self.chk_sagittal_plane.isChecked():
                x_idx = self._get_sagittal_slice_index()
                if x_idx is not None and abs(ix - float(x_idx)) <= tol:
                    return True

        except Exception:
            return False

        return False

    def _sample_sitk_values_at_ras_points(self, img: sitk.Image, pts_ras: np.ndarray):
        try:
            pts_ras = np.asarray(pts_ras, dtype=np.float64)
            if pts_ras.ndim != 2 or pts_ras.shape[1] != 3:
                return None

            pts_lps = pts_ras.copy()
            pts_lps[:, 0] *= -1.0
            pts_lps[:, 1] *= -1.0

            origin = np.asarray(img.GetOrigin(), dtype=np.float64)
            spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
            direction = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
            inv_direction = np.linalg.inv(direction)

            rel = pts_lps - origin[None, :]
            idx_xyz = (rel @ inv_direction.T) / spacing[None, :]

            x = idx_xyz[:, 0]
            y = idx_xyz[:, 1]
            z = idx_xyz[:, 2]

            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            vals = map_coordinates(
                arr,
                np.vstack([z, y, x]),
                order=1,
                mode="constant",
                cval=np.nan,
            )
            return vals.astype(np.float32)
        except Exception:
            return None

    def _build_t1_plane_rgba_highres(self, geom: dict):
        # T1 remains the geometry reference.
        # The displayed anatomical texture can be T1 or T2.
        if self._t1_img is None:
            return None

        mri_img = self._get_active_mri_for_3d()
        if mri_img is None:
            return None

        p00 = np.asarray(geom["p00"], dtype=np.float64)
        p10 = np.asarray(geom["p10"], dtype=np.float64)
        p01 = np.asarray(geom["p01"], dtype=np.float64)
        p11 = np.asarray(geom["p11"], dtype=np.float64)

        u = p10 - p00
        v = p01 - p00
        width_mm = float(np.linalg.norm(u))
        height_mm = float(np.linalg.norm(v))
        if width_mm <= 0 or height_mm <= 0:
            return None

        center = 0.25 * (p00 + p10 + p01 + p11)
        out_w, out_h = self._get_highres_plane_texture_size()

        arr_t1, valid_t1 = self._sample_image_on_plane_ras(
            mri_img, center, u, v, width_mm, height_mm, out_w, out_h, order=1, cval=np.nan
        )
        if arr_t1 is None:
            return None

        brainmask_for_slices = None
        try:
            if self._brainmask_img is not None and self._t1_img is not None:
                brainmask_for_slices = sitk.Resample(
                    self._brainmask_img,
                    self._t1_img,
                    sitk.Transform(),
                    sitk.sitkNearestNeighbor,
                    0,
                    sitk.sitkUInt8,
                )
        except Exception:
            brainmask_for_slices = None

        if brainmask_for_slices is not None:
            arr_mask, _ = self._sample_image_on_plane_ras(
                brainmask_for_slices,
                center,
                u,
                v,
                width_mm,
                height_mm,
                out_w,
                out_h,
                order=0,
                cval=0.0,
            )
            msl = (np.isfinite(arr_mask) & (arr_mask > 0.5)).astype(np.uint8)

            try:
                from scipy.ndimage import binary_dilation

                msl = binary_dilation(msl.astype(bool), iterations=1).astype(np.uint8)

                normal = np.asarray(geom.get("normal", [0.0, 0.0, 0.0]), dtype=np.float64)
                if abs(normal[1]) > 0.8:
                    h, w = msl.shape
                    y0 = int(0.30 * h)
                    inferior = msl[y0:, :].astype(bool)
                    inferior = binary_dilation(inferior, iterations=3)
                    msl[y0:, :] = inferior.astype(np.uint8)
            except Exception:
                pass
        else:
            msl = np.ones_like(arr_t1, dtype=np.uint8)

        valid = np.isfinite(arr_t1)
        if valid_t1 is not None:
            valid &= valid_t1 > 0
        valid &= msl > 0

        vals = arr_t1[valid]
        if vals.size == 0:
            vals = arr_t1[np.isfinite(arr_t1)]
        if vals.size == 0:
            return None

        vmin = float(np.percentile(vals, 2))
        vmax = float(np.percentile(vals, 98))
        if vmax <= vmin:
            vmax = vmin + 1.0

        sln = np.clip((arr_t1 - vmin) / (vmax - vmin), 0.0, 1.0)
        rgba = np.zeros((int(out_h), int(out_w), 4), dtype=np.uint8)
        gray = (np.nan_to_num(sln, nan=0.0) * 255.0).astype(np.uint8)
        rgba[..., 0] = gray
        rgba[..., 1] = gray
        rgba[..., 2] = gray
        rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)

        # ---------------- Parcellation overlay on slices ----------------
        # ---------------- Parcellation overlay on slices only ----------------
        try:
            parcel_img, parcel_lut = self._get_active_parcellation_img_and_lut()

            if parcel_img is not None:
                arr_parc, valid_parc = self._sample_image_on_plane_ras(
                    parcel_img,
                    center,
                    u,
                    v,
                    width_mm,
                    height_mm,
                    out_w,
                    out_h,
                    order=0,
                    cval=np.nan,
                )

                if arr_parc is not None:
                    parc_valid = np.isfinite(arr_parc) & (arr_parc > 0) & (msl > 0)
                    if valid_parc is not None:
                        parc_valid &= valid_parc > 0

                    if np.any(parc_valid):
                        labels = np.full(arr_parc.shape, -1, dtype=np.int32)
                        labels[parc_valid] = np.round(arr_parc[parc_valid]).astype(np.int32)

                        rgb_parc = np.zeros(
                            (arr_parc.shape[0], arr_parc.shape[1], 3), dtype=np.float32
                        )

                        if isinstance(parcel_lut, dict) and len(parcel_lut) > 0:
                            unique_labels = np.unique(labels[parc_valid])

                            for lab in unique_labels:
                                if int(lab) <= 0:
                                    continue

                                entry = parcel_lut.get(int(lab), None)
                                if entry is None:
                                    continue

                                try:
                                    _name, color = entry
                                    r, g, b = color
                                    m = labels == int(lab)
                                    rgb_parc[m, 0] = float(r)
                                    rgb_parc[m, 1] = float(g)
                                    rgb_parc[m, 2] = float(b)
                                except Exception:
                                    pass

                        # choose the opacity of the active parcellation
                        if self.chk_parcel1 is not None and self.chk_parcel1.isChecked():
                            a_pct = (
                                int(self.spn_parcel1_opacity.value())
                                if self.spn_parcel1_opacity is not None
                                else 50
                            )
                        else:
                            a_pct = (
                                int(self.spn_parcel2_opacity.value())
                                if self.spn_parcel2_opacity is not None
                                else 50
                            )

                        a = float(np.clip(a_pct, 0, 100)) / 100.0

                        # blend only on the slice texture
                        for c in range(3):
                            base = rgba[..., c].astype(np.float32)
                            base[parc_valid] = (1.0 - a) * base[parc_valid] + a * rgb_parc[..., c][
                                parc_valid
                            ]
                            rgba[..., c] = np.clip(base, 0, 255).astype(np.uint8)

                        rgba[..., 3][parc_valid] = 255
        except Exception:
            pass

        return rgba

    def _build_pet_plane_rgba_highres(self, geom: dict):
        """
        Return the PET overlay slice from the cached full PET RGBA volume.
        """
        cache = self._get_or_build_slice_pet_rgba_cache()
        return self._extract_rgba_slice_from_volume_cache(cache, geom)

    def _build_siscom_plane_rgba_highres(self, geom: dict):
        """
        Return the SISCOM overlay slice from the cached full SISCOM RGBA volume.
        """
        cache = self._get_or_build_slice_siscom_rgba_cache()
        return self._extract_rgba_slice_from_volume_cache(cache, geom)

    def _build_textured_plane_mesh(self, geom: dict, actor_name: str):
        if not _PV_OK or geom is None:
            return None

        try:
            pts = np.array([geom["p00"], geom["p10"], geom["p11"], geom["p01"]], dtype=np.float64)

            try:
                # Keep T1 planes exactly on the anatomical slice.
                # Only PET/SISCOM overlays should be offset.
                if actor_name.endswith("_pet") or actor_name.endswith("_siscom"):
                    if actor_name.startswith("coronal"):
                        n = self._get_effective_plane_normal("coronal", geom)
                    elif actor_name.startswith("axial"):
                        n = self._get_effective_plane_normal("axial", geom)
                    elif actor_name.startswith("sagittal"):
                        n = self._get_effective_plane_normal("sagittal", geom)
                    else:
                        n = np.asarray(geom["normal"], dtype=np.float64)

                    n = np.asarray(n, dtype=np.float64)
                    nn = np.linalg.norm(n)
                    if nn > 1e-6:
                        n = n / nn
                        pts = pts - 0.15 * n[None, :]
            except Exception:
                pass

            quad = pv.PolyData(pts.astype(np.float32))
            quad.faces = np.array([4, 0, 1, 2, 3], dtype=np.int64)
            quad.active_texture_coordinates = np.array(
                [[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32
            )
            return quad
        except Exception:
            return None

    def _update_actor_mesh_from_source(self, actor, source_mesh, which: str) -> None:
        if actor is None or source_mesh is None:
            return
        try:
            clipped = self._clip_plane_mesh_with_other_planes(which, source_mesh.copy())
            if clipped is None or clipped.n_points == 0:
                try:
                    actor.SetVisibility(False)
                except Exception:
                    pass
                return

            mapper = actor.GetMapper()
            if mapper is not None:
                mapper.SetInputData(clipped)
                try:
                    actor.SetVisibility(True)
                except Exception:
                    pass
        except Exception:
            pass

    def _render_textured_plane_actor(
        self,
        actor_attr_name: str,
        actor_name: str,
        source_mesh_attr_name: str,
        which_plane: str,
        geom: dict,
        rgba: np.ndarray,
    ):
        if not _PV_OK or self.plotter is None:
            return

        self._remove_actor(actor_name)
        setattr(self, source_mesh_attr_name, None)

        if rgba is None:
            return

        try:
            texture = pv.numpy_to_texture(rgba)

            source_quad = self._build_textured_plane_mesh(geom, actor_name)
            if source_quad is None or source_quad.n_points == 0:
                return

            setattr(self, source_mesh_attr_name, source_quad.copy())

            quad = self._clip_plane_mesh_with_other_planes(which_plane, source_quad.copy())
            if quad is None or quad.n_points == 0:
                return

            actor_opacity = 1.0
            try:
                if actor_name.endswith("_pet"):
                    actor_opacity = float(
                        np.clip(self._get_pet_slice_overlay_alpha() / 255.0, 0.0, 1.0)
                    )
                elif actor_name.endswith("_siscom"):
                    actor_opacity = float(np.clip(self._get_siscom_slice_overlay_alpha(), 0.0, 1.0))
            except Exception:
                actor_opacity = 1.0

            actor = self.plotter.add_mesh(
                quad,
                texture=texture,
                opacity=float(actor_opacity),
                lighting=False,
                show_scalar_bar=False,
                name=actor_name,
            )

            try:
                prop = actor.GetProperty()
                prop.SetAmbient(1.0)
                prop.SetDiffuse(0.0)
                prop.SetSpecular(0.0)
            except Exception:
                pass

            setattr(self, actor_attr_name, actor)
        except Exception:
            setattr(self, actor_attr_name, None)
            setattr(self, source_mesh_attr_name, None)

    def _render_coronal_pet_overlay(self) -> None:
        self._remove_actor("coronal_pet")
        if self.chk_coronal_plane is None or not self.chk_coronal_plane.isChecked():
            return
        if self.chk_pet is None or not self.chk_pet.isChecked():
            return
        geom = self._build_coronal_plane_geometry()
        if geom is None:
            return
        rgba = self._build_pet_plane_rgba_highres(geom)
        self._render_textured_plane_actor(
            "_coronal_pet_actor",
            "coronal_pet",
            "_coronal_pet_source_mesh",
            "coronal",
            geom,
            rgba,
        )

    def _render_axial_pet_overlay(self) -> None:
        self._remove_actor("axial_pet")
        if self.chk_axial_plane is None or not self.chk_axial_plane.isChecked():
            return
        if self.chk_pet is None or not self.chk_pet.isChecked():
            return
        geom = self._build_axial_plane_geometry()
        if geom is None:
            return
        rgba = self._build_pet_plane_rgba_highres(geom)
        self._render_textured_plane_actor(
            "_axial_pet_actor",
            "axial_pet",
            "_axial_pet_source_mesh",
            "axial",
            geom,
            rgba,
        )

    def _render_sagittal_pet_overlay(self) -> None:
        self._remove_actor("sagittal_pet")
        if self.chk_sagittal_plane is None or not self.chk_sagittal_plane.isChecked():
            return
        if self.chk_pet is None or not self.chk_pet.isChecked():
            return
        geom = self._build_sagittal_plane_geometry()
        if geom is None:
            return
        rgba = self._build_pet_plane_rgba_highres(geom)
        self._render_textured_plane_actor(
            "_sagittal_pet_actor",
            "sagittal_pet",
            "_sagittal_pet_source_mesh",
            "sagittal",
            geom,
            rgba,
        )

    def _render_coronal_siscom_overlay(self) -> None:
        self._remove_actor("coronal_siscom")
        if self.chk_coronal_plane is None or not self.chk_coronal_plane.isChecked():
            return
        if self.chk_siscom is None or not self.chk_siscom.isChecked():
            return
        geom = self._build_coronal_plane_geometry()
        if geom is None:
            return
        rgba = self._build_siscom_plane_rgba_highres(geom)
        self._render_textured_plane_actor(
            "_coronal_siscom_actor",
            "coronal_siscom",
            "_coronal_siscom_source_mesh",
            "coronal",
            geom,
            rgba,
        )

    def _render_axial_siscom_overlay(self) -> None:
        self._remove_actor("axial_siscom")
        if self.chk_axial_plane is None or not self.chk_axial_plane.isChecked():
            return
        if self.chk_siscom is None or not self.chk_siscom.isChecked():
            return
        geom = self._build_axial_plane_geometry()
        if geom is None:
            return
        rgba = self._build_siscom_plane_rgba_highres(geom)
        self._render_textured_plane_actor(
            "_axial_siscom_actor",
            "axial_siscom",
            "_axial_siscom_source_mesh",
            "axial",
            geom,
            rgba,
        )

    def _render_sagittal_siscom_overlay(self) -> None:
        self._remove_actor("sagittal_siscom")
        if self.chk_sagittal_plane is None or not self.chk_sagittal_plane.isChecked():
            return
        if self.chk_siscom is None or not self.chk_siscom.isChecked():
            return
        geom = self._build_sagittal_plane_geometry()
        if geom is None:
            return
        rgba = self._build_siscom_plane_rgba_highres(geom)
        self._render_textured_plane_actor(
            "_sagittal_siscom_actor",
            "sagittal_siscom",
            "_sagittal_siscom_source_mesh",
            "sagittal",
            geom,
            rgba,
        )

    def _render_visible_pet_overlays_only(self) -> None:
        try:
            self._render_coronal_pet_overlay()
        except Exception:
            pass
        try:
            self._render_axial_pet_overlay()
        except Exception:
            pass
        try:
            self._render_sagittal_pet_overlay()
        except Exception:
            pass

    def _render_visible_siscom_overlays_only(self) -> None:
        try:
            self._render_coronal_siscom_overlay()
        except Exception:
            pass
        try:
            self._render_axial_siscom_overlay()
        except Exception:
            pass
        try:
            self._render_sagittal_siscom_overlay()
        except Exception:
            pass

    def _reclip_existing_plane_actors_only(self, changed_plane: str) -> None:
        changed_plane = str(changed_plane).lower().strip()

        try:
            if changed_plane != "coronal":
                self._update_actor_mesh_from_source(
                    self._coronal_plane_actor,
                    self._coronal_plane_source_mesh,
                    "coronal",
                )
                self._update_actor_mesh_from_source(
                    self._coronal_pet_actor,
                    self._coronal_pet_source_mesh,
                    "coronal",
                )
                self._update_actor_mesh_from_source(
                    self._coronal_siscom_actor,
                    self._coronal_siscom_source_mesh,
                    "coronal",
                )
        except Exception:
            pass

        try:
            if changed_plane != "axial":
                self._update_actor_mesh_from_source(
                    self._axial_plane_actor,
                    self._axial_plane_source_mesh,
                    "axial",
                )
                self._update_actor_mesh_from_source(
                    self._axial_pet_actor,
                    self._axial_pet_source_mesh,
                    "axial",
                )
                self._update_actor_mesh_from_source(
                    self._axial_siscom_actor,
                    self._axial_siscom_source_mesh,
                    "axial",
                )
        except Exception:
            pass

        try:
            if changed_plane != "sagittal":
                self._update_actor_mesh_from_source(
                    self._sagittal_plane_actor,
                    self._sagittal_plane_source_mesh,
                    "sagittal",
                )
                self._update_actor_mesh_from_source(
                    self._sagittal_pet_actor,
                    self._sagittal_pet_source_mesh,
                    "sagittal",
                )
                self._update_actor_mesh_from_source(
                    self._sagittal_siscom_actor,
                    self._sagittal_siscom_source_mesh,
                    "sagittal",
                )
        except Exception:
            pass

    def _refresh_all_visible_plane_outlines(self) -> None:
        """
        Rebuild every visible colored slice frame from the current position
        of all active planes.

        This is required because moving one plane changes the clipping boundary
        of the two orthogonal planes as well.
        """
        if not bool(getattr(self, "_slice_plane_frames_visible", True)):
            try:
                self._remove_all_slice_plane_frame_actors()
            except Exception:
                pass
            return
        try:
            if self.chk_coronal_plane is not None and self.chk_coronal_plane.isChecked():
                self._render_coronal_outline()
            else:
                self._remove_actor("coronal_outline")
        except Exception:
            pass

        try:
            if self.chk_axial_plane is not None and self.chk_axial_plane.isChecked():
                self._render_axial_outline()
            else:
                self._remove_actor("axial_outline")
        except Exception:
            pass

        try:
            if self.chk_sagittal_plane is not None and self.chk_sagittal_plane.isChecked():
                self._render_sagittal_outline()
            else:
                self._remove_actor("sagittal_outline")
        except Exception:
            pass

    def _main_window_bridge(self):
        mw = getattr(self.state, "main_window", None)
        if mw is not None and hasattr(mw, "set_current_page_by_name"):
            return mw

        from PySide6.QtWidgets import QApplication

        for w in QApplication.topLevelWidgets():
            try:
                if hasattr(w, "set_current_page_by_name"):
                    return w
            except Exception:
                pass
        return None

    def _go_back_to_reconstruction(self):
        # 1) récupérer la position actuelle du marker 3D
        lps = self._crosshair_marker_lps()

        mw = self._main_window_bridge()

        # 2) retrouver la vraie page Reconstruction
        reco = None

        # Nom utilisé dans ton code actuel
        try:
            reco = getattr(self.state, "reco_page", None)
        except Exception:
            reco = None

        # Ancien nom possible
        if reco is None:
            try:
                reco = getattr(self.state, "reconstruction_page", None)
            except Exception:
                reco = None

        # Depuis MainWindow
        if reco is None and mw is not None:
            try:
                reco = getattr(mw, "reco_page", None)
            except Exception:
                reco = None

        if reco is None and mw is not None:
            try:
                reco = getattr(mw, "reconstruction_page", None)
            except Exception:
                reco = None

        # 3) envoyer le point à Reconstruction AVANT de changer de page
        if lps is not None and reco is not None and hasattr(reco, "set_crosshair_from_lps"):
            try:
                reco.set_crosshair_from_lps(lps)
            except Exception as e:
                print("[3D->Reco] Failed to update reconstruction crosshair:", e)
        else:
            print(
                "[3D->Reco] No marker or no reconstruction page found.", "lps=", lps, "reco=", reco
            )

        # 4) revenir sur la page Reconstruction
        if mw is not None:
            try:
                mw.set_current_page_by_name("pageReconstruction")
            except Exception:
                pass

    def _quick_tools_no_background(self) -> bool:
        try:
            if hasattr(self, "quick_tools") and self.quick_tools is not None:
                if hasattr(self.quick_tools, "transparent_background_enabled"):
                    return bool(self.quick_tools.transparent_background_enabled())
                return bool(getattr(self.quick_tools, "_transparent_background", False))
        except Exception:
            pass
        return False

    def _create_mni_atlas_checkbox(self) -> None:
        """
        Connect the MNI atlas checkbox from the .ui file if it exists.
        If it does not exist, create it dynamically as fallback.

        Important:
        Do not call signal.disconnect() here. PySide can emit RuntimeWarning
        even when the exception is caught. Instead, connect only once using a flag.
        """
        try:
            existing = self.ui.findChild(QCheckBox, "chk_3d_showMNIAtlas")

            if existing is not None:
                self.chk_mni_atlas = existing
                self.chk_mni_atlas.setText("MNI atlas")
                self.chk_mni_atlas.setToolTip(
                    "Display the MNI atlas brain and allow loading BIDS MNI electrodes.tsv files."
                )

                if not bool(getattr(self, "_mni_atlas_signal_connected", False)):
                    self.chk_mni_atlas.toggled.connect(self._on_mni_atlas_toggled)
                    self._mni_atlas_signal_connected = True

                return

            if self.chk_mni_atlas is not None:
                return

            parent = None
            layout = None

            if self.chk_pial is not None:
                parent = self.chk_pial.parentWidget()
                layout = parent.layout() if parent is not None else None

            if parent is None:
                parent = self.ui

            self.chk_mni_atlas = QCheckBox("MNI atlas")
            self.chk_mni_atlas.setObjectName("chk_3d_showMNIAtlas")
            self.chk_mni_atlas.setToolTip(
                "Display the MNI atlas brain and allow loading BIDS MNI electrodes.tsv files."
            )

            if layout is not None:
                try:
                    idx = layout.indexOf(self.chk_pial)
                    if idx >= 0:
                        layout.insertWidget(idx + 1, self.chk_mni_atlas)
                    else:
                        layout.addWidget(self.chk_mni_atlas)
                except Exception:
                    layout.addWidget(self.chk_mni_atlas)
            else:
                self.chk_mni_atlas.setParent(self.container_3d)
                self.chk_mni_atlas.move(12, 12)
                self.chk_mni_atlas.show()
                self.chk_mni_atlas.raise_()

            if not bool(getattr(self, "_mni_atlas_signal_connected", False)):
                self.chk_mni_atlas.toggled.connect(self._on_mni_atlas_toggled)
                self._mni_atlas_signal_connected = True

        except Exception as e:
            print("[MNI atlas] Could not create/connect checkbox:", e)

    def _load_mni_electrodes_from_paths(self, paths) -> None:
        """
        Load one or several BIDS MNI electrodes.tsv files.

        The files must contain columns:
            name, x, y, z

        Coordinates are assumed to be MNI RAS-like mm values,
        as exported by NeuXelec BIDS MNI export.
        """
        if not paths:
            return

        try:
            if not hasattr(self.state, "mni_electrode_sets"):
                self.state.mni_electrode_sets = []
        except Exception:
            return

        loaded = 0
        errors = []

        existing_paths = set()
        try:
            for s in getattr(self.state, "mni_electrode_sets", []) or []:
                if isinstance(s, dict):
                    existing_paths.add(str(s.get("path", "")))
        except Exception:
            existing_paths = set()

        for path in paths:
            try:
                p = str(Path(path))

                if p in existing_paths:
                    continue

                mni_set = load_bids_mni_electrodes_tsv(p)

                # Default MNI list style:
                # white background, black text, like the native white electrode row.
                mni_set["color"] = [255, 255, 255]
                mni_set.setdefault("group_color", {})

                self.state.mni_electrode_sets.append(mni_set)
                existing_paths.add(p)
                loaded += 1

            except Exception as e:
                errors.append(f"{Path(path).name}: {e}")

        if loaded > 0:
            try:
                if self.chk_mni_atlas is not None and not self.chk_mni_atlas.isChecked():
                    self.chk_mni_atlas.setChecked(True)
                else:
                    self._render_mni_scene()
            except Exception:
                pass

            try:
                self._refresh_mni_tree_items()
            except Exception:
                pass

        if errors:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(),
                "MNI electrodes",
                "Some files could not be loaded:\n\n" + "\n\n".join(errors[:10]),
            )

    def open_mni_electrodes_files(self) -> None:
        """
        Manual loader for BIDS MNI electrodes.tsv files.
        More reliable than drag-and-drop on PyVista/QtInteractor.
        """
        files, _ = QFileDialog.getOpenFileNames(
            self._dialog_parent(),
            "Load BIDS MNI electrodes.tsv",
            "",
            "BIDS electrodes TSV (*electrodes.tsv *.tsv);;TSV files (*.tsv);;All files (*)",
        )

        if files:
            self._load_mni_electrodes_from_paths(files)

    def _dialog_parent(self):
        """
        Return a safe top-level parent for QFileDialog/QColorDialog/QMessageBox.
        Avoids: QWidgetWindow(...) must be a top level window.
        """
        try:
            w = self.ui.window() if self.ui is not None else None
            if w is not None and w.isWindow():
                return w
        except Exception:
            pass

        try:
            w = _top_level_window()
            if w is not None and w.isWindow():
                return w
        except Exception:
            pass

        return None

    def _refresh_electrode_tree_for_current_3d_mode(self) -> None:
        """
        Rebuild the 3D electrode list according to the current mode.

        MNI mode:
            tv_Electrodes_3 shows only MNI electrodes.

        Native mode:
            tv_Electrodes_3 shows patient electrodes.
        """
        try:
            ctrl = getattr(self.state, "electrodes_controller", None)

            if ctrl is not None and hasattr(ctrl, "refresh_all"):
                ctrl.refresh_all()
                return

        except Exception:
            pass

        # Fallback: if controller is not available, at least remove MNI rows
        # when returning to native mode. Native rows need the controller.
        try:
            tree = self.ui.findChild(QTreeWidget, "tv_Electrodes_3")

            if tree is None:
                return

            if self.is_mni_atlas_active():
                self._refresh_mni_tree_items()
            else:
                for i in reversed(range(tree.topLevelItemCount())):
                    item = tree.topLevelItem(i)
                    try:
                        if item.data(0, Qt.UserRole + 50) in (
                            "mni_set",
                            "mni_group",
                            "mni_contact",
                        ):
                            tree.takeTopLevelItem(i)
                    except Exception:
                        pass

        except Exception:
            pass

    def _mni_group_name_from_contact(self, contact: dict) -> str:
        """
        Infer the electrode/group name for one MNI contact.
        Priority:
        1. BIDS 'group' column
        2. contact name without trailing digits
        """
        try:
            g = str(contact.get("group", "") or "").strip()
            if g:
                return g
        except Exception:
            pass

        try:
            name = str(contact.get("name", "") or "").strip()
            base = name.rstrip("0123456789")
            return base or "contacts"
        except Exception:
            return "contacts"

    def _ensure_mni_visibility_fields(self, mni_set: dict) -> None:
        """
        Create visibility dictionaries used by the MNI tree and renderer.
        """
        if not isinstance(mni_set, dict):
            return
        if not isinstance(mni_set.get("group_color"), dict):
            mni_set["group_color"] = {}

        if "electrode_names_visible" not in mni_set:
            mni_set["electrode_names_visible"] = False

        contacts = mni_set.get("contacts", []) or []

        if "visible" not in mni_set:
            mni_set["visible"] = True

        if not isinstance(mni_set.get("group_visible"), dict):
            mni_set["group_visible"] = {}

        if not isinstance(mni_set.get("contact_visible"), dict):
            mni_set["contact_visible"] = {}

        if not isinstance(mni_set.get("contact_label_visible"), dict):
            mni_set["contact_label_visible"] = {}

        for ci, c in enumerate(contacts):
            group = self._mni_group_name_from_contact(c)
            key = str(ci)

            if group not in mni_set["group_visible"]:
                mni_set["group_visible"][group] = True

            if key not in mni_set["contact_visible"]:
                mni_set["contact_visible"][key] = True

            if key not in mni_set["contact_label_visible"]:
                mni_set["contact_label_visible"][key] = False

    def _mni_contact_is_visible(self, mni_set: dict, ci: int, contact: dict) -> bool:
        try:
            if not bool(mni_set.get("visible", True)):
                return False

            group = self._mni_group_name_from_contact(contact)
            if not bool(mni_set.get("group_visible", {}).get(group, True)):
                return False

            if not bool(mni_set.get("contact_visible", {}).get(str(int(ci)), True)):
                return False

            return True
        except Exception:
            return True

    def _mni_contact_label_is_visible(self, mni_set: dict, ci: int) -> bool:
        try:
            return bool(mni_set.get("contact_label_visible", {}).get(str(int(ci)), False))
        except Exception:
            return False

    def is_mni_atlas_active(self) -> bool:
        """
        Public helper used by ElectrodesController.
        """
        try:
            return bool(self.chk_mni_atlas is not None and self.chk_mni_atlas.isChecked())
        except Exception:
            return False

    def _refresh_electrode_tree_for_current_3d_mode(self) -> None:
        """
        Ask the shared electrode controller to rebuild the 3D electrode list.

        In native mode: tv_Electrodes_3 shows patient electrodes.
        In MNI mode: tv_Electrodes_3 shows only MNI electrodes.
        """
        try:
            ctrl = getattr(self.state, "electrodes_controller", None)

            if ctrl is not None and hasattr(ctrl, "refresh_all"):
                ctrl.refresh_all()
                return

        except Exception:
            pass

        # Fallback if the controller is not available.
        try:
            if self.is_mni_atlas_active():
                self._refresh_mni_tree_items()
        except Exception:
            pass

    def _mni_tree_text_color_for_background(self, rgb) -> QColor:
        """
        Choose black or white text depending on the MNI row background.
        """
        try:
            r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

            if luminance >= 148:
                return QColor("#080A10")

            return QColor("#F2F2F5")

        except Exception:
            return QColor("#080A10")

    def _apply_mni_tree_row_style(
        self,
        item: QTreeWidgetItem,
        rgb,
        alpha: int = 255,
        kind: str = "electrode",
    ) -> None:
        """
        Make MNI rows use the same coloured-row delegate as native electrodes.

        ROLEs used by ElectrodeRowColorDelegate in controllers/electrodes.py:
            Qt.UserRole + 1 = kind
            Qt.UserRole + 4 = row RGB
            Qt.UserRole + 5 = row alpha
        """
        if item is None:
            return

        try:
            rgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        except Exception:
            rgb = (255, 255, 255)

        try:
            item.setData(0, Qt.UserRole + 1, f"mni_{kind}")
            item.setData(0, Qt.UserRole + 4, tuple(rgb))
            item.setData(0, Qt.UserRole + 5, int(alpha))
            item.setForeground(0, QBrush(self._mni_tree_text_color_for_background(rgb)))
        except Exception:
            pass

    def _refresh_mni_tree_items(self) -> None:
        """
        Add imported MNI electrode sets to tv_Electrodes_3.

        Structure:
            [MNI] subject
                electrode/group
                    contact
        """
        tree = None

        try:
            tree = self.ui.findChild(QTreeWidget, "tv_Electrodes_3")
            if tree is None:
                return

            self._mni_tree_updating = True
            tree.blockSignals(True)

            # In MNI mode, the 3D list must contain only MNI electrodes.
            # Therefore clear everything, including patient-native electrodes.
            if self.is_mni_atlas_active():
                tree.clear()
            else:
                # In MNI mode, tv_Electrodes_3 must contain ONLY MNI electrodes.
                # In native mode, this method should only remove old MNI rows.
                if self.is_mni_atlas_active():
                    tree.clear()
                else:
                    for i in reversed(range(tree.topLevelItemCount())):
                        item = tree.topLevelItem(i)
                        try:
                            if item.data(0, Qt.UserRole + 50) in (
                                "mni_set",
                                "mni_group",
                                "mni_contact",
                            ):
                                tree.takeTopLevelItem(i)
                        except Exception:
                            pass

            sets = getattr(self.state, "mni_electrode_sets", []) or []

            for si, mni_set in enumerate(sets):
                if not isinstance(mni_set, dict):
                    continue

                self._ensure_mni_visibility_fields(mni_set)

                subject = str(mni_set.get("subject", f"MNI set {si + 1}"))
                contacts = mni_set.get("contacts", []) or []
                n_contacts = len(contacts)

                color = mni_set.get("color", (255, 255, 255))

                try:
                    qcolor = QColor(int(color[0]), int(color[1]), int(color[2]))
                except Exception:
                    qcolor = QColor(255, 255, 255)

                root_rgb = (qcolor.red(), qcolor.green(), qcolor.blue())

                root = QTreeWidgetItem([f"[MNI] {subject} ({n_contacts} contacts)"])
                root.setData(0, Qt.UserRole + 50, "mni_set")
                root.setData(0, Qt.UserRole + 51, int(si))
                root.setFlags(
                    root.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled
                )
                root.setCheckState(
                    0, Qt.Checked if bool(mni_set.get("visible", True)) else Qt.Unchecked
                )
                self._apply_mni_tree_row_style(
                    root,
                    root_rgb,
                    alpha=255,
                    kind="set",
                )
                root.setToolTip(
                    0,
                    "Imported MNI electrodes.tsv\n"
                    "Right-click to change color or remove this MNI set.",
                )

                tree.addTopLevelItem(root)

                # Group contacts by electrode name.
                groups = {}
                for ci, c in enumerate(contacts):
                    group = self._mni_group_name_from_contact(c)
                    groups.setdefault(group, []).append((ci, c))

                for group_name, group_contacts in sorted(groups.items()):
                    group_item = QTreeWidgetItem([f"{group_name} ({len(group_contacts)} contacts)"])
                    group_item.setData(0, Qt.UserRole + 50, "mni_group")
                    group_item.setData(0, Qt.UserRole + 51, int(si))
                    group_item.setData(0, Qt.UserRole + 52, str(group_name))
                    group_item.setFlags(
                        group_item.flags()
                        | Qt.ItemIsUserCheckable
                        | Qt.ItemIsSelectable
                        | Qt.ItemIsEnabled
                    )

                    group_visible = bool(mni_set.get("group_visible", {}).get(group_name, True))
                    group_item.setCheckState(0, Qt.Checked if group_visible else Qt.Unchecked)
                    group_rgb = self._mni_group_color_rgb(mni_set, group_name)
                    try:
                        group_qcolor = QColor(
                            int(group_rgb[0]), int(group_rgb[1]), int(group_rgb[2])
                        )
                    except Exception:
                        group_qcolor = qcolor
                    group_rgb_tuple = (
                        int(group_qcolor.red()),
                        int(group_qcolor.green()),
                        int(group_qcolor.blue()),
                    )

                    self._apply_mni_tree_row_style(
                        root,
                        root_rgb,
                        alpha=255,
                        kind="set",
                    )

                    root.addChild(group_item)

                    for ci, c in group_contacts:
                        name = str(
                            c.get("name", f"{group_name}{ci + 1}") or f"{group_name}{ci + 1}"
                        )
                        contact_item = QTreeWidgetItem([name])
                        contact_item.setData(0, Qt.UserRole + 50, "mni_contact")
                        contact_item.setData(0, Qt.UserRole + 51, int(si))
                        contact_item.setData(0, Qt.UserRole + 52, str(group_name))
                        contact_item.setData(0, Qt.UserRole + 53, int(ci))
                        contact_item.setFlags(
                            contact_item.flags()
                            | Qt.ItemIsUserCheckable
                            | Qt.ItemIsSelectable
                            | Qt.ItemIsEnabled
                        )

                        contact_visible = bool(
                            mni_set.get("contact_visible", {}).get(str(ci), True)
                        )
                        contact_item.setCheckState(
                            0, Qt.Checked if contact_visible else Qt.Unchecked
                        )
                        self._apply_mni_tree_row_style(
                            contact_item,
                            group_rgb_tuple,
                            alpha=148,
                            kind="contact",
                        )

                        group_item.addChild(contact_item)

                    group_item.setExpanded(False)

                root.setExpanded(True)

            tree.blockSignals(False)
            self._mni_tree_updating = False

            if not bool(getattr(self, "_mni_tree_connected", False)):
                tree.itemChanged.connect(self._on_mni_tree_item_changed)
                self._mni_tree_connected = True

            if not bool(getattr(self, "_mni_tree_context_connected", False)):
                tree.setContextMenuPolicy(Qt.CustomContextMenu)
                tree.customContextMenuRequested.connect(self._on_mni_tree_context_menu)
                self._mni_tree_context_connected = True

            try:
                if self._mni_tree_bulk_filter is None:
                    self._mni_tree_bulk_filter = _MniTreeBulkCheckFilter(self)
                    tree.viewport().installEventFilter(self._mni_tree_bulk_filter)
            except Exception:
                pass

        except Exception as e:
            print("[MNI electrodes] Tree refresh failed:", e)

            try:
                if tree is not None:
                    tree.blockSignals(False)
            except Exception:
                pass

            self._mni_tree_updating = False

    def _mni_rgb_to_float_color(self, rgb):
        try:
            return (
                float(rgb[0]) / 255.0,
                float(rgb[1]) / 255.0,
                float(rgb[2]) / 255.0,
            )
        except Exception:
            return (0.4, 0.7, 1.0)

    def _mni_set_actor_color(self, actor, color) -> None:
        """
        Update an existing VTK/PyVista actor color without rebuilding the scene.
        """
        self._set_existing_actor_color(actor, color)

    def _mni_set_label_color(self, actor, color) -> None:
        """
        Update MNI label/text actor color without removing/rebuilding it.
        """
        if actor is None:
            return

        # Case 1: regular actor property
        try:
            prop = actor.GetProperty()
            if prop is not None:
                prop.SetColor(float(color[0]), float(color[1]), float(color[2]))
        except Exception:
            pass

        # Case 2: vtkTextActor-like
        try:
            prop = actor.GetTextProperty()
            if prop is not None:
                prop.SetColor(float(color[0]), float(color[1]), float(color[2]))
        except Exception:
            pass

        # Case 3: PyVista label actor / vtkActor2D wrappers
        try:
            text_actor = actor.GetTextActor()
            if text_actor is not None:
                prop = text_actor.GetTextProperty()
                if prop is not None:
                    prop.SetColor(float(color[0]), float(color[1]), float(color[2]))
        except Exception:
            pass

        # Case 4: some VTK assemblies expose parts
        try:
            parts = actor.GetParts()
            parts.InitTraversal()

            for _ in range(parts.GetNumberOfItems()):
                part = parts.GetNextProp()

                try:
                    prop = part.GetProperty()
                    if prop is not None:
                        prop.SetColor(float(color[0]), float(color[1]), float(color[2]))
                except Exception:
                    pass

                try:
                    prop = part.GetTextProperty()
                    if prop is not None:
                        prop.SetColor(float(color[0]), float(color[1]), float(color[2]))
                except Exception:
                    pass

        except Exception:
            pass

    def _set_existing_actor_color(self, actor, color) -> None:
        """
        Update an existing PyVista/VTK actor color without rebuilding it.
        Works for electrode contacts, shafts, projection crosses and most labels.
        """
        if actor is None:
            return

        try:
            prop = actor.GetProperty()
            if prop is not None:
                prop.SetColor(float(color[0]), float(color[1]), float(color[2]))
        except Exception:
            pass

        # Some text/label actors expose text properties instead of regular mesh property.
        try:
            prop = actor.GetTextProperty()
            if prop is not None:
                prop.SetColor(float(color[0]), float(color[1]), float(color[2]))
        except Exception:
            pass

        try:
            text_actor = actor.GetTextActor()
            if text_actor is not None:
                prop = text_actor.GetTextProperty()
                if prop is not None:
                    prop.SetColor(float(color[0]), float(color[1]), float(color[2]))
        except Exception:
            pass

    def _remove_mni_group_actors(self, si: int, group_name: str) -> None:
        """
        Remove only one MNI electrode/group from the 3D scene.
        Does not touch the MNI atlas brain or other electrodes.
        """
        if self.plotter is None:
            return

        si = int(si)
        group_name = str(group_name)

        for key in [
            (si, group_name, "points"),
            (si, group_name, "line"),
        ]:
            actor = self._mni_electrode_actors.pop(key, None)
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor, reset_camera=False)
                except Exception:
                    pass

        for key in [
            (si, group_name, "label"),
            (si, group_name, "contact_labels"),
        ]:
            actor = self._mni_label_actors.pop(key, None)
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor, reset_camera=False)
                except Exception:
                    pass

    def _render_single_mni_group(self, si: int, group_name: str) -> None:
        """
        Rebuild only one MNI electrode/group.
        Used when one contact visibility or labels change.
        """
        if not _PV_OK or self.plotter is None:
            return

        try:
            si = int(si)
            group_name = str(group_name)

            sets = getattr(self.state, "mni_electrode_sets", []) or []
            if not (0 <= si < len(sets)):
                return

            mni_set = sets[si]
            self._ensure_mni_visibility_fields(mni_set)

            self._remove_mni_group_actors(si, group_name)

            if not bool(mni_set.get("visible", True)):
                self._render()
                return

            if not bool(mni_set.get("group_visible", {}).get(group_name, True)):
                self._render()
                return

            contacts = mni_set.get("contacts", []) or []

            group_contacts = []
            for ci, c in enumerate(contacts):
                if self._mni_group_name_from_contact(c) != group_name:
                    continue

                if not self._mni_contact_is_visible(mni_set, ci, c):
                    continue

                try:
                    x, y, z = c.get("mni_ras", [None, None, None])
                    p = [float(x), float(y), float(z)]
                except Exception:
                    continue

                group_contacts.append((ci, c, p))

            if not group_contacts:
                self._render()
                return

            pts_arr = np.asarray([p for _ci, _c, p in group_contacts], dtype=np.float32)

            group_rgb = self._mni_group_color_rgb(mni_set, group_name)
            color = self._mni_rgb_to_float_color(group_rgb)

            point_size = 8.0
            try:
                if self.spin_contacts_size is not None:
                    point_size = float(self.spin_contacts_size.value())
            except Exception:
                pass

            show_shaft = True
            try:
                if self.btn_elec_shaft is not None:
                    show_shaft = bool(self.btn_elec_shaft.isChecked())
            except Exception:
                pass

            # Points
            try:
                poly = pv.PolyData(pts_arr)
                actor = self.plotter.add_points(
                    poly,
                    color=color,
                    point_size=float(point_size),
                    render_points_as_spheres=True,
                    name=f"mni_contacts_{si}_{group_name}",
                )

                try:
                    actor.PickableOff()
                except Exception:
                    pass

                self._mni_electrode_actors[(si, group_name, "points")] = actor

            except Exception as e:
                print("[MNI electrodes] Failed to render single group points:", e)

            # Shaft
            if show_shaft and pts_arr.shape[0] >= 2:
                try:
                    line = pv.lines_from_points(pts_arr, close=False)
                    line_actor = self.plotter.add_mesh(
                        line,
                        color=color,
                        line_width=3,
                        name=f"mni_line_{si}_{group_name}",
                    )

                    try:
                        line_actor.PickableOff()
                    except Exception:
                        pass

                    self._mni_electrode_actors[(si, group_name, "line")] = line_actor

                except Exception:
                    pass

            # Electrode name label: AD, AG, etc.
            try:
                if bool(mni_set.get("electrode_names_visible", False)):
                    label_pt = pts_arr[0].copy()
                    label_pt[0] += 3.0
                    label_pt[2] += 3.0

                    label_actor = self.plotter.add_point_labels(
                        np.asarray([label_pt], dtype=np.float32),
                        [str(group_name)],
                        font_size=12,
                        text_color=color,
                        shape_opacity=0.0,
                        show_points=False,
                        always_visible=True,
                        name=f"mni_label_{si}_{group_name}",
                    )

                    try:
                        label_actor.PickableOff()
                    except Exception:
                        pass

                    self._mni_label_actors[(si, group_name, "label")] = label_actor
            except Exception:
                pass

            # Contact labels: AD1, AD2, etc.
            try:
                label_pts = []
                label_txt = []

                for ci, c, p in group_contacts:
                    if not self._mni_contact_label_is_visible(mni_set, ci):
                        continue

                    lp = np.asarray(p, dtype=np.float32)
                    lp[0] += 2.0
                    lp[2] += 2.0

                    label_pts.append(lp)
                    label_txt.append(str(c.get("name", f"{group_name}{ci + 1}")))

                if label_pts:
                    contact_label_actor = self.plotter.add_point_labels(
                        np.asarray(label_pts, dtype=np.float32),
                        label_txt,
                        font_size=11,
                        text_color=color,
                        shape_opacity=0.0,
                        show_points=False,
                        always_visible=True,
                        name=f"mni_contact_labels_{si}_{group_name}",
                    )

                    try:
                        contact_label_actor.PickableOff()
                    except Exception:
                        pass

                    self._mni_label_actors[(si, group_name, "contact_labels")] = contact_label_actor

            except Exception:
                pass

            self._render()

        except Exception as e:
            print("[MNI electrodes] render single group failed:", e)

    def _set_mni_group_visibility_only(self, si: int, group_name: str, visible: bool) -> None:
        """
        Show/hide one MNI electrode group without rebuilding.
        """
        si = int(si)
        group_name = str(group_name)
        visible = bool(visible)

        for key in [
            (si, group_name, "points"),
            (si, group_name, "line"),
            (si, group_name, "label"),
            (si, group_name, "contact_labels"),
        ]:
            actor = self._mni_electrode_actors.get(key)
            if actor is None:
                actor = self._mni_label_actors.get(key)

            if actor is not None:
                try:
                    actor.SetVisibility(visible)
                except Exception:
                    pass

        self._render()

    def _set_mni_patient_visibility_only(self, si: int, visible: bool) -> None:
        """
        Show/hide all actors belonging to one MNI patient without rebuilding.
        """
        si = int(si)
        visible = bool(visible)

        for key, actor in list(getattr(self, "_mni_electrode_actors", {}).items()):
            try:
                if isinstance(key, tuple) and len(key) >= 1 and int(key[0]) == si:
                    actor.SetVisibility(visible)
            except Exception:
                pass

        for key, actor in list(getattr(self, "_mni_label_actors", {}).items()):
            try:
                if isinstance(key, tuple) and len(key) >= 1 and int(key[0]) == si:
                    actor.SetVisibility(visible)
            except Exception:
                pass

        self._render()

    def _update_mni_group_color_only(
        self,
        si: int,
        group_name: str,
        render: bool = True,
    ) -> None:
        """
        Update color of one MNI electrode/group without rebuilding anything.

        Important:
        Do NOT call _render_single_mni_group() here.
        Do NOT remove label actors here.
        Otherwise the MNI electrode visually jumps/flickers.
        """
        try:
            si = int(si)
            group_name = str(group_name)

            sets = getattr(self.state, "mni_electrode_sets", []) or []
            if not (0 <= si < len(sets)):
                return

            mni_set = sets[si]
            color = self._mni_rgb_to_float_color(self._mni_group_color_rgb(mni_set, group_name))

            # Contacts + shaft.
            for key in (
                (si, group_name, "points"),
                (si, group_name, "line"),
            ):
                actor = self._mni_electrode_actors.get(key)
                self._mni_set_actor_color(actor, color)

            # Labels updated in place.
            for key in (
                (si, group_name, "label"),
                (si, group_name, "contact_labels"),
            ):
                actor = self._mni_label_actors.get(key)
                self._mni_set_label_color(actor, color)

            if render:
                self._render()

        except Exception as e:
            print("[MNI electrodes] update group color failed:", e)

    def _update_mni_patient_color_only(self, si: int) -> None:
        """
        Update color of all visible electrodes for one MNI patient.
        Does not rebuild the atlas or other patients.
        """
        try:
            si = int(si)

            sets = getattr(self.state, "mni_electrode_sets", []) or []
            if not (0 <= si < len(sets)):
                return

            mni_set = sets[si]
            contacts = mni_set.get("contacts", []) or []

            groups = sorted({self._mni_group_name_from_contact(c) for c in contacts})

            for group_name in groups:
                self._update_mni_group_color_only(
                    si,
                    group_name,
                    render=False,
                )

            self._render()

        except Exception as e:
            print("[MNI electrodes] update patient color failed:", e)

    def _mni_tree_item_key(self, item):
        try:
            kind = item.data(0, Qt.UserRole + 50)
            si = int(item.data(0, Qt.UserRole + 51))

            if kind == "mni_set":
                return ("mni_set", si)

            if kind == "mni_group":
                group_name = str(item.data(0, Qt.UserRole + 52))
                return ("mni_group", si, group_name)

            if kind == "mni_contact":
                ci = int(item.data(0, Qt.UserRole + 53))
                return ("mni_contact", si, ci)

        except Exception:
            pass

        return None

    def _apply_mni_tree_item_check_without_render(self, item, checked: bool) -> None:
        """
        Apply checkbox changes to the MNI tree + state without rendering immediately.
        The render is flushed later on mouse release.
        """
        try:
            kind = item.data(0, Qt.UserRole + 50)

            if kind not in ("mni_set", "mni_group", "mni_contact"):
                return

            si = int(item.data(0, Qt.UserRole + 51))
            sets = getattr(self.state, "mni_electrode_sets", []) or []

            if not (0 <= si < len(sets)):
                return

            mni_set = sets[si]
            self._ensure_mni_visibility_fields(mni_set)

            checked = bool(checked)
            tree = self.ui.findChild(QTreeWidget, "tv_Electrodes_3")

            if tree is not None:
                tree.blockSignals(True)

            item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)

            if kind == "mni_set":
                mni_set["visible"] = checked
                self._mni_tree_bulk_pending_patients.add(si)

                for g in list(mni_set.get("group_visible", {}).keys()):
                    mni_set["group_visible"][g] = checked

                for k in list(mni_set.get("contact_visible", {}).keys()):
                    mni_set["contact_visible"][k] = checked

                for gi in range(item.childCount()):
                    group_item = item.child(gi)
                    group_name = str(group_item.data(0, Qt.UserRole + 52))
                    group_item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                    self._mni_tree_bulk_pending_groups.add((si, group_name))

                    for ci in range(group_item.childCount()):
                        contact_item = group_item.child(ci)
                        contact_item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)

            elif kind == "mni_group":
                group_name = str(item.data(0, Qt.UserRole + 52))
                mni_set["group_visible"][group_name] = checked
                self._mni_tree_bulk_pending_groups.add((si, group_name))

                contacts = mni_set.get("contacts", []) or []
                for ci, c in enumerate(contacts):
                    if self._mni_group_name_from_contact(c) == group_name:
                        mni_set["contact_visible"][str(ci)] = checked

                for child_i in range(item.childCount()):
                    contact_item = item.child(child_i)
                    contact_item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)

                parent = item.parent()
                if parent is not None:
                    any_group_on = False
                    for gi in range(parent.childCount()):
                        if parent.child(gi).checkState(0) == Qt.Checked:
                            any_group_on = True
                            break

                    parent.setCheckState(0, Qt.Checked if any_group_on else Qt.Unchecked)
                    mni_set["visible"] = bool(any_group_on)

            elif kind == "mni_contact":
                ci = int(item.data(0, Qt.UserRole + 53))
                mni_set["contact_visible"][str(ci)] = checked

                contacts = mni_set.get("contacts", []) or []
                group_name = None

                if 0 <= ci < len(contacts):
                    group_name = self._mni_group_name_from_contact(contacts[ci])
                    self._mni_tree_bulk_pending_groups.add((si, group_name))

                group_item = item.parent()
                if group_item is not None:
                    any_contact_on = False
                    for child_i in range(group_item.childCount()):
                        if group_item.child(child_i).checkState(0) == Qt.Checked:
                            any_contact_on = True
                            break

                    group_item.setCheckState(0, Qt.Checked if any_contact_on else Qt.Unchecked)

                    gname = str(group_item.data(0, Qt.UserRole + 52))
                    mni_set["group_visible"][gname] = bool(any_contact_on)

                    root = group_item.parent()
                    if root is not None:
                        any_group_on = False
                        for gi in range(root.childCount()):
                            if root.child(gi).checkState(0) == Qt.Checked:
                                any_group_on = True
                                break

                        root.setCheckState(0, Qt.Checked if any_group_on else Qt.Unchecked)
                        mni_set["visible"] = bool(any_group_on)

            if tree is not None:
                tree.blockSignals(False)

        except Exception as e:
            print("[MNI bulk check] apply failed:", e)

            try:
                tree = self.ui.findChild(QTreeWidget, "tv_Electrodes_3")
                if tree is not None:
                    tree.blockSignals(False)
            except Exception:
                pass

    def _flush_mni_tree_bulk_render(self) -> None:
        """
        Render only once after a bulk checkbox drag operation.
        """
        try:
            if getattr(self, "_mni_tree_bulk_pending_patients", None):
                for si in list(self._mni_tree_bulk_pending_patients):
                    sets = getattr(self.state, "mni_electrode_sets", []) or []
                    if 0 <= int(si) < len(sets):
                        visible = bool(sets[int(si)].get("visible", True))
                        self._set_mni_patient_visibility_only(int(si), visible)

            if getattr(self, "_mni_tree_bulk_pending_groups", None):
                for si, group_name in list(self._mni_tree_bulk_pending_groups):
                    sets = getattr(self.state, "mni_electrode_sets", []) or []
                    if not (0 <= int(si) < len(sets)):
                        continue

                    mni_set = sets[int(si)]
                    group_visible = bool(
                        mni_set.get("group_visible", {}).get(str(group_name), True)
                    )

                    # If the whole group was toggled, simple visibility is enough.
                    # If individual contacts changed, rebuilding this one group updates the point cloud.
                    self._render_single_mni_group(int(si), str(group_name))

                    if not group_visible:
                        self._set_mni_group_visibility_only(int(si), str(group_name), False)

            self._mni_tree_bulk_pending_patients.clear()
            self._mni_tree_bulk_pending_groups.clear()

            try:
                self._render()
            except Exception:
                pass

        except Exception as e:
            print("[MNI bulk check] flush failed:", e)

    def _mni_tree_click_is_on_expand_arrow(
        self,
        tree: QTreeWidget,
        item: QTreeWidgetItem,
        pos,
    ) -> bool:
        """
        Return True when the click is on the expand/collapse arrow.

        This must be left to QTreeWidget so clicking the arrow opens the details
        without toggling the checkbox.
        """
        if tree is None or item is None:
            return False

        try:
            kind = item.data(0, Qt.UserRole + 50)

            if kind not in ("mni_set", "mni_group"):
                return False

            if item.childCount() <= 0:
                return False

            rect = tree.visualItemRect(item)

            if not rect.isValid():
                return False

            if pos.y() < rect.top() or pos.y() > rect.bottom():
                return False

            branch_zone_right = max(
                int(tree.indentation()) + 10,
                int(rect.left()),
            )

            return bool(0 <= int(pos.x()) < branch_zone_right)

        except Exception:
            return False

    def _mni_tree_click_is_on_checkbox(
        self,
        tree: QTreeWidget,
        item: QTreeWidgetItem,
        pos,
    ) -> bool:
        """
        Same behavior as native electrodes:
        only the checkbox zone toggles visibility.

        Click on text:
            select item / keep normal tree behavior

        Click on arrow:
            expand/collapse item

        Click + hold + hover over checkboxes:
            bulk check/uncheck
        """
        if tree is None or item is None:
            return False

        try:
            rect = tree.visualItemRect(item)

            if not rect.isValid():
                return False

            if pos.y() < rect.top() or pos.y() > rect.bottom():
                return False

            # Same principle as native electrodes.
            # The checkbox is in the left part of the visual item rect.
            x0 = int(rect.left())
            x1 = int(rect.left()) + 34

            return bool(x0 <= int(pos.x()) <= x1)

        except Exception:
            return False

    def _handle_mni_tree_bulk_check_event(self, obj, event) -> bool:
        """
        Allow click-and-drag over MNI tree checkboxes.
        The state changes while dragging, but 3D rendering is done only on mouse release.
        """
        try:
            tree = self.ui.findChild(QTreeWidget, "tv_Electrodes_3")
            if tree is None:
                return False

            et = event.type()

            if et == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
                item = tree.itemAt(pos)

                if item is None:
                    return False

                kind = item.data(0, Qt.UserRole + 50)
                if kind not in ("mni_set", "mni_group", "mni_contact"):
                    return False

                # Same as native electrodes:
                # clicking the arrow must only expand/collapse, not toggle visibility.
                if self._mni_tree_click_is_on_expand_arrow(tree, item, pos):
                    return False

                # Only checkbox clicks start the check/drag behavior.
                if not self._mni_tree_click_is_on_checkbox(tree, item, pos):
                    return False

                # On first click, choose the target state as the opposite of current state.
                target_checked = item.checkState(0) != Qt.Checked

                self._mni_tree_bulk_active = True
                self._mni_tree_bulk_target_checked = bool(target_checked)
                self._mni_tree_bulk_last_item_key = None
                self._mni_tree_bulk_pending_groups.clear()
                self._mni_tree_bulk_pending_patients.clear()

                key = self._mni_tree_item_key(item)
                self._mni_tree_bulk_last_item_key = key
                self._apply_mni_tree_item_check_without_render(item, target_checked)

                event.accept()
                return True

            if et == QEvent.MouseMove and bool(getattr(self, "_mni_tree_bulk_active", False)):
                pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
                item = tree.itemAt(pos)

                try:
                    buttons = event.buttons()
                except Exception:
                    buttons = Qt.NoButton

                if not bool(buttons & Qt.LeftButton):
                    self._mni_tree_bulk_active = False
                    self._mni_tree_bulk_last_item_key = None
                    return False

                if item is None:
                    event.accept()
                    return True

                kind = item.data(0, Qt.UserRole + 50)
                if kind not in ("mni_set", "mni_group", "mni_contact"):
                    event.accept()
                    return True

                if not self._mni_tree_click_is_on_checkbox(tree, item, pos):
                    event.accept()
                    return True

                key = self._mni_tree_item_key(item)
                if key != getattr(self, "_mni_tree_bulk_last_item_key", None):
                    self._mni_tree_bulk_last_item_key = key
                    self._apply_mni_tree_item_check_without_render(
                        item,
                        bool(getattr(self, "_mni_tree_bulk_target_checked", True)),
                    )

                event.accept()
                return True

            if et == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                if bool(getattr(self, "_mni_tree_bulk_active", False)):
                    self._mni_tree_bulk_active = False
                    self._mni_tree_bulk_last_item_key = None
                    self._flush_mni_tree_bulk_render()

                    event.accept()
                    return True

        except Exception as e:
            print("[MNI bulk check] event failed:", e)

            try:
                self._mni_tree_bulk_active = False
            except Exception:
                pass

        return False

    def _on_mni_tree_item_changed(self, item, column) -> None:
        try:
            if bool(getattr(self, "_mni_tree_updating", False)):
                return

            kind = item.data(0, Qt.UserRole + 50)
            si = int(item.data(0, Qt.UserRole + 51))
            sets = getattr(self.state, "mni_electrode_sets", []) or []

            if not (0 <= si < len(sets)):
                return

            mni_set = sets[si]
            self._ensure_mni_visibility_fields(mni_set)

            checked = item.checkState(0) == Qt.Checked

            tree = self.ui.findChild(QTreeWidget, "tv_Electrodes_3")
            if tree is not None:
                tree.blockSignals(True)

            groups_to_rebuild = set()
            groups_to_visibility = {}

            if kind == "mni_set":
                mni_set["visible"] = bool(checked)

                for g in list(mni_set.get("group_visible", {}).keys()):
                    mni_set["group_visible"][g] = bool(checked)
                    groups_to_visibility[g] = bool(checked)

                for k in list(mni_set.get("contact_visible", {}).keys()):
                    mni_set["contact_visible"][k] = bool(checked)

                for gi in range(item.childCount()):
                    group_item = item.child(gi)
                    group_item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)

                    for ci in range(group_item.childCount()):
                        contact_item = group_item.child(ci)
                        contact_item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)

                # Fast path: show/hide the whole patient actors.
                self._set_mni_patient_visibility_only(si, checked)

            elif kind == "mni_group":
                group_name = str(item.data(0, Qt.UserRole + 52))
                mni_set["group_visible"][group_name] = bool(checked)

                contacts = mni_set.get("contacts", []) or []

                for ci, c in enumerate(contacts):
                    if self._mni_group_name_from_contact(c) == group_name:
                        mni_set["contact_visible"][str(ci)] = bool(checked)

                for child_i in range(item.childCount()):
                    contact_item = item.child(child_i)
                    contact_item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)

                parent = item.parent()
                if parent is not None:
                    any_group_on = False
                    for gi in range(parent.childCount()):
                        if parent.child(gi).checkState(0) == Qt.Checked:
                            any_group_on = True
                            break

                    parent.setCheckState(0, Qt.Checked if any_group_on else Qt.Unchecked)
                    mni_set["visible"] = bool(any_group_on)

                # Fast path: show/hide the whole electrode actor.
                self._set_mni_group_visibility_only(si, group_name, checked)

            elif kind == "mni_contact":
                ci = int(item.data(0, Qt.UserRole + 53))
                mni_set["contact_visible"][str(ci)] = bool(checked)

                contacts = mni_set.get("contacts", []) or []
                group_name = None

                if 0 <= ci < len(contacts):
                    group_name = self._mni_group_name_from_contact(contacts[ci])
                    groups_to_rebuild.add(group_name)

                group_item = item.parent()
                if group_item is not None:
                    any_contact_on = False
                    for child_i in range(group_item.childCount()):
                        if group_item.child(child_i).checkState(0) == Qt.Checked:
                            any_contact_on = True
                            break

                    group_item.setCheckState(0, Qt.Checked if any_contact_on else Qt.Unchecked)

                    gname = str(group_item.data(0, Qt.UserRole + 52))
                    mni_set["group_visible"][gname] = bool(any_contact_on)

                    root = group_item.parent()
                    if root is not None:
                        any_group_on = False
                        for gi in range(root.childCount()):
                            if root.child(gi).checkState(0) == Qt.Checked:
                                any_group_on = True
                                break

                        root.setCheckState(0, Qt.Checked if any_group_on else Qt.Unchecked)
                        mni_set["visible"] = bool(any_group_on)

                # Contact-level change needs rebuilding only this electrode,
                # because the point cloud geometry changes.
                for g in groups_to_rebuild:
                    self._render_single_mni_group(si, g)

            if tree is not None:
                tree.blockSignals(False)

        except Exception as e:
            print("[MNI tree] item changed failed:", e)
            try:
                tree = self.ui.findChild(QTreeWidget, "tv_Electrodes_3")
                if tree is not None:
                    tree.blockSignals(False)
            except Exception:
                pass

    def _clear_mni_slice_focused_contact(self) -> None:
        """
        Remove the temporary MNI contact label created by a previous
        Show coronal/axial/sagittal slice action.
        """
        focused = getattr(
            self,
            "_mni_slice_focused_contact",
            None,
        )

        if not isinstance(focused, dict):
            return

        try:
            si = int(focused["set_index"])
            ci = int(focused["contact_index"])
            group_name = str(focused["group_name"])

            sets = (
                getattr(
                    self.state,
                    "mni_electrode_sets",
                    [],
                )
                or []
            )

            if 0 <= si < len(sets):
                mni_set = sets[si]
                self._ensure_mni_visibility_fields(mni_set)

                mni_set["contact_label_visible"][str(ci)] = False

                self._render_single_mni_group(
                    si,
                    group_name,
                )

        except Exception:
            pass

        self._mni_slice_focused_contact = None

    def _on_mni_tree_context_menu(self, pos) -> None:
        """
        Context menu for imported MNI electrodes in tv_Electrodes_3.
        The menu UI is centralized in context_menus.py.
        """
        try:
            tree = self.ui.findChild(QTreeWidget, "tv_Electrodes_3")
            if tree is None:
                return

            item = tree.itemAt(pos)
            if item is None:
                return

            kind = item.data(0, Qt.UserRole + 50)
            if kind not in ("mni_set", "mni_group", "mni_contact"):
                return

            si = int(item.data(0, Qt.UserRole + 51))
            sets = getattr(self.state, "mni_electrode_sets", []) or []

            if not (0 <= si < len(sets)):
                return

            mni_set = sets[si]
            self._ensure_mni_visibility_fields(mni_set)

            labels_on = False
            electrode_names_on = False
            patient_name_on = False

            if kind == "mni_set":
                electrode_names_on = self._mni_electrode_names_are_visible(mni_set)
                patient_name_on = bool(mni_set.get("patient_name_visible", False))

            elif kind == "mni_group":
                group_name = str(item.data(0, Qt.UserRole + 52))
                labels_on = self._mni_group_labels_are_visible(mni_set, group_name)

            elif kind == "mni_contact":
                ci = int(item.data(0, Qt.UserRole + 53))
                labels_on = bool(mni_set.get("contact_label_visible", {}).get(str(ci), False))

            choice = exec_mni_electrode_tree_menu(
                tree.viewport().mapToGlobal(pos),
                kind=str(kind),
                labels_on=bool(labels_on),
                electrode_names_on=bool(electrode_names_on),
                patient_name_on=bool(patient_name_on),
            )

            if choice is None:
                return

            if kind == "mni_set":
                if choice == "color":
                    self._set_mni_patient_color(si)

                elif choice == "toggle_electrode_names":
                    mni_set["electrode_names_visible"] = not bool(electrode_names_on)

                    contacts = mni_set.get("contacts", []) or []
                    groups = sorted({self._mni_group_name_from_contact(c) for c in contacts})

                    for group_name in groups:
                        self._render_single_mni_group(si, group_name)

                elif choice == "toggle_patient_name":
                    mni_set["patient_name_visible"] = not bool(patient_name_on)
                    self._render_mni_patient_name_label_only(si)

            elif kind == "mni_group":
                group_name = str(item.data(0, Qt.UserRole + 52))

                if choice == "color":
                    self._set_mni_group_color(si, group_name)

                elif choice == "toggle_labels":
                    self._set_mni_labels_visible_from_item(item, not labels_on)

            elif kind == "mni_contact":
                ci = int(item.data(0, Qt.UserRole + 53))

                if choice == "toggle_labels":
                    mni_set["contact_label_visible"][str(ci)] = not bool(labels_on)

                    contacts = mni_set.get("contacts", []) or []

                    if 0 <= ci < len(contacts):
                        group_name = self._mni_group_name_from_contact(contacts[ci])

                        self._render_single_mni_group(
                            si,
                            group_name,
                        )

                elif choice in (
                    "show_coronal_slice",
                    "show_axial_slice",
                    "show_sagittal_slice",
                ):
                    plane_name = {
                        "show_coronal_slice": "coronal",
                        "show_axial_slice": "axial",
                        "show_sagittal_slice": "sagittal",
                    }[choice]

                    self.show_mni_contact_in_slice(
                        set_index=si,
                        contact_index=ci,
                        plane_name=plane_name,
                    )

        except Exception as e:
            print("[MNI tree menu] failed:", e)

    def _set_mni_labels_visible_from_item(self, item, visible: bool) -> None:
        try:
            kind = item.data(0, Qt.UserRole + 50)
            si = int(item.data(0, Qt.UserRole + 51))
            sets = getattr(self.state, "mni_electrode_sets", []) or []

            if not (0 <= si < len(sets)):
                return

            mni_set = sets[si]
            self._ensure_mni_visibility_fields(mni_set)

            contacts = mni_set.get("contacts", []) or []
            groups_to_update = set()

            if kind == "mni_set":
                for ci, c in enumerate(contacts):
                    mni_set["contact_label_visible"][str(ci)] = bool(visible)
                    groups_to_update.add(self._mni_group_name_from_contact(c))

            elif kind == "mni_group":
                group_name = str(item.data(0, Qt.UserRole + 52))

                for ci, c in enumerate(contacts):
                    if self._mni_group_name_from_contact(c) == group_name:
                        mni_set["contact_label_visible"][str(ci)] = bool(visible)

                groups_to_update.add(group_name)

            elif kind == "mni_contact":
                ci = int(item.data(0, Qt.UserRole + 53))
                mni_set["contact_label_visible"][str(ci)] = bool(visible)

                if 0 <= ci < len(contacts):
                    groups_to_update.add(self._mni_group_name_from_contact(contacts[ci]))

            for group_name in groups_to_update:
                self._render_single_mni_group(si, group_name)

        except Exception as e:
            print("[MNI labels] failed:", e)

    def _change_mni_set_color(self, si: int) -> None:
        """
        Change the color of one imported MNI electrode set.
        """
        try:
            sets = getattr(self.state, "mni_electrode_sets", []) or []

            if not (0 <= int(si) < len(sets)):
                return

            mni_set = sets[int(si)]
            color = mni_set.get("color", (100, 180, 255))

            try:
                current = QColor(int(color[0]), int(color[1]), int(color[2]))
            except Exception:
                current = QColor(100, 180, 255)

            color_hex = NeuXelecColorDialog.get_color(
                initial_color=current,
                parent=self._dialog_parent(),
                title="Choose MNI electrode color",
            )

            if color_hex is None:
                return

            qcolor = QColor(color_hex)

            if not qcolor.isValid():
                return

            mni_set["color"] = [
                int(qcolor.red()),
                int(qcolor.green()),
                int(qcolor.blue()),
            ]

            self._refresh_mni_tree_items()
            self._render_mni_scene(reset_camera=False)

        except Exception as e:
            print("[MNI electrodes] color change failed:", e)

    def _remove_mni_set(self, si: int) -> None:
        """
        Remove one imported MNI electrode set from the scene and tree.
        """
        try:
            sets = getattr(self.state, "mni_electrode_sets", []) or []

            if not (0 <= int(si) < len(sets)):
                return

            sets.pop(int(si))

            self._refresh_mni_tree_items()
            self._render_mni_scene(reset_camera=False)

        except Exception as e:
            print("[MNI electrodes] remove failed:", e)

    def _get_mni_template_t1_image(self) -> sitk.Image | None:
        """
        Return the MNI template T1 image used for MNI slice planes.

        Expected filename:
            tools/templates/template_T1w.nii.gz

        It should match the MNI brain mask:
            tools/templates/template_brain_mask.nii.gz
        """
        try:
            if self._mni_template_t1_img is not None:
                return self._mni_template_t1_img

            from neuxelec.coregistration import _default_brainmask_template_paths

            template_t1, _template_mask = _default_brainmask_template_paths()

            if template_t1 is not None and Path(template_t1).exists():
                self._mni_template_t1_img = sitk.ReadImage(str(template_t1))
                return self._mni_template_t1_img

        except Exception as e:
            print("[MNI atlas] Could not load template T1:", e)

        return None

    def _get_mni_template_mask_image(self) -> sitk.Image | None:
        """
        Cached MNI brain mask image used for both MNI atlas mesh and MNI T1 slice cropping.
        """
        try:
            if self._mni_template_mask_img is not None:
                return self._mni_template_mask_img

            from neuxelec.coregistration import _default_brainmask_template_paths

            _template_t1, template_mask = _default_brainmask_template_paths()

            if template_mask is not None and Path(template_mask).exists():
                self._mni_template_mask_img = sitk.ReadImage(str(template_mask))
                return self._mni_template_mask_img

        except Exception as e:
            print("[MNI atlas] Could not load template mask:", e)

        return None

    def _mni_templates_dir(self) -> Path | None:
        """
        Return tools/templates directory, based on the current MNI template location.
        """
        try:
            from neuxelec.coregistration import _default_brainmask_template_paths

            template_t1, _template_mask = _default_brainmask_template_paths()

            if template_t1 is not None:
                return Path(template_t1).parent

        except Exception:
            pass

        return None

    def _load_mni_lut_tsv(self, lut_path: Path) -> dict:
        """
        Load MNI parcellation LUT from TSV.

        Expected format:
            index    name    color
        """
        try:
            import csv

            lut_path = Path(lut_path)

            if not lut_path.exists():
                print(f"[MNI parcellation] TSV LUT not found: {lut_path}")
                return {}

            lut = {}

            def _hex_to_rgb(hex_color: str, lab: int):
                s = str(hex_color or "").strip()

                if s.startswith("#") and len(s) >= 7:
                    try:
                        return [
                            int(s[1:3], 16),
                            int(s[3:5], 16),
                            int(s[5:7], 16),
                        ]
                    except Exception:
                        pass

                return [
                    int((37 * lab + 53) % 255),
                    int((91 * lab + 101) % 255),
                    int((173 * lab + 17) % 255),
                ]

            with open(lut_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")

                if reader.fieldnames is None:
                    print(f"[MNI parcellation] Empty TSV LUT: {lut_path}")
                    return {}

                for row in reader:
                    try:
                        lab = int(float(str(row.get("index", "")).strip()))
                    except Exception:
                        continue

                    if lab <= 0:
                        continue

                    name = str(row.get("name", "") or "").strip()
                    if not name:
                        name = f"Label {lab}"

                    color = _hex_to_rgb(row.get("color", ""), lab)
                    lut[lab] = (name, color)

            return lut

        except Exception as e:
            print("[MNI parcellation] Could not load TSV LUT:", e)
            return {}

    def _generate_simple_lut_from_image(self, img: sitk.Image | None) -> dict:
        """
        Fallback LUT if no JSON is provided.
        Generates deterministic colors for labels.
        """
        if img is None:
            return {}

        try:
            arr = sitk.GetArrayFromImage(img)
            labels = np.unique(arr.astype(np.int32))
            labels = labels[labels > 0]

            lut = {}

            for lab in labels:
                lab = int(lab)

                # deterministic pseudo-random color from label id
                r = (37 * lab + 53) % 255
                g = (91 * lab + 101) % 255
                b = (173 * lab + 17) % 255

                lut[lab] = (
                    f"Label {lab}",
                    [int(r), int(g), int(b)],
                )

            return lut

        except Exception:
            return {}

    def _get_mni_parcellation1_img_and_lut(self):
        """
        Load first MNI parcellation from:
            tools/templates/template_parcellation1.nii.gz
            tools/templates/template_parcellation1_lut.tsv
        """
        try:
            if self._mni_parcel1_img is not None and self._mni_parcel1_lut:
                return self._mni_parcel1_img, self._mni_parcel1_lut

            d = self._mni_templates_dir()
            if d is None:
                return None, {}

            img_path = d / "template_parcellation1.nii.gz"
            lut_path = d / "template_parcellation1_lut.tsv"

            if not img_path.exists():
                print(f"[MNI parcellation] Parcellation 1 NIfTI not found: {img_path}")
                return None, {}

            if not lut_path.exists():
                print(f"[MNI parcellation] Parcellation 1 TSV not found: {lut_path}")

            img = sitk.ReadImage(str(img_path))
            lut = self._load_mni_lut_tsv(lut_path)

            if not lut:
                print("[MNI parcellation] Parcellation 1 TSV LUT empty. Using fallback.")
                lut = self._generate_simple_lut_from_image(img)

            self._mni_parcel1_img = img
            self._mni_parcel1_lut = lut

            return self._mni_parcel1_img, self._mni_parcel1_lut

        except Exception as e:
            print("[MNI parcellation] Could not load parcellation 1:", e)
            return None, {}

    def _get_mni_parcellation2_img_and_lut(self):
        """
        Load second MNI parcellation from:
            tools/templates/template_parcellation2.nii.gz
            tools/templates/template_parcellation2_lut.tsv
        """
        try:
            if self._mni_parcel2_img is not None and self._mni_parcel2_lut:
                return self._mni_parcel2_img, self._mni_parcel2_lut

            d = self._mni_templates_dir()
            if d is None:
                return None, {}

            img_path = d / "template_parcellation2.nii.gz"
            lut_path = d / "template_parcellation2_lut.tsv"

            if not img_path.exists():
                print(f"[MNI parcellation] Parcellation 2 NIfTI not found: {img_path}")
                return None, {}

            if not lut_path.exists():
                print(f"[MNI parcellation] Parcellation 2 TSV not found: {lut_path}")

            img = sitk.ReadImage(str(img_path))
            lut = self._load_mni_lut_tsv(lut_path)

            if not lut:
                print("[MNI parcellation] Parcellation 2 TSV LUT empty. Using fallback.")
                lut = self._generate_simple_lut_from_image(img)

            self._mni_parcel2_img = img
            self._mni_parcel2_lut = lut

            return self._mni_parcel2_img, self._mni_parcel2_lut

        except Exception as e:
            print("[MNI parcellation] Could not load parcellation 2:", e)
            return None, {}

    def _load_mni_lut_json(self, lut_path: Path) -> dict:
        """
        Load a LUT json for MNI parcellation.

        Accepted JSON formats:
        1)
        {
            "1": ["Region name", [255, 0, 0]],
            "2": ["Region name", [0, 255, 0]]
        }

        2)
        {
            "1": {"name": "Region name", "color": [255, 0, 0]},
            "2": {"name": "Region name", "rgb": [0, 255, 0]}
        }
        """
        try:
            import json

            if lut_path is None or not Path(lut_path).exists():
                return {}

            with open(lut_path, encoding="utf-8") as f:
                raw = json.load(f)

            lut = {}

            if not isinstance(raw, dict):
                return {}

            for k, v in raw.items():
                try:
                    lab = int(k)
                except Exception:
                    continue

                name = f"Label {lab}"
                color = None

                if isinstance(v, dict):
                    name = str(v.get("name", name))
                    color = v.get("color", v.get("rgb", None))

                elif isinstance(v, (list, tuple)):
                    if len(v) >= 2:
                        name = str(v[0])
                        color = v[1]

                if color is None:
                    continue

                try:
                    r, g, b = color[:3]
                    lut[lab] = (
                        name,
                        [int(r), int(g), int(b)],
                    )
                except Exception:
                    continue

            return lut

        except Exception as e:
            print("[MNI parcellation] Could not load LUT:", e)
            return {}

    def _load_mni_lut_json(self, lut_path: Path) -> dict:
        """
        Load a LUT json for MNI parcellation.

        Accepted JSON formats:
        1)
        {
            "1": ["Region name", [255, 0, 0]],
            "2": ["Region name", [0, 255, 0]]
        }

        2)
        {
            "1": {"name": "Region name", "color": [255, 0, 0]},
            "2": {"name": "Region name", "rgb": [0, 255, 0]}
        }
        """
        try:
            import json

            if lut_path is None or not Path(lut_path).exists():
                return {}

            with open(lut_path, encoding="utf-8") as f:
                raw = json.load(f)

            lut = {}

            if not isinstance(raw, dict):
                return {}

            for k, v in raw.items():
                try:
                    lab = int(k)
                except Exception:
                    continue

                name = f"Label {lab}"
                color = None

                if isinstance(v, dict):
                    name = str(v.get("name", name))
                    color = v.get("color", v.get("rgb", None))

                elif isinstance(v, (list, tuple)):
                    if len(v) >= 2:
                        name = str(v[0])
                        color = v[1]

                if color is None:
                    continue

                try:
                    r, g, b = color[:3]
                    lut[lab] = (
                        name,
                        [int(r), int(g), int(b)],
                    )
                except Exception:
                    continue

            return lut

        except Exception as e:
            print("[MNI parcellation] Could not load LUT:", e)
            return {}

    def _mni_parcellation1_loaded(self) -> bool:
        img, _lut = self._get_mni_parcellation1_img_and_lut()
        return img is not None

    def _mni_parcellation2_loaded(self) -> bool:
        img, _lut = self._get_mni_parcellation2_img_and_lut()
        return img is not None

    def _mni_t1_slices_are_visible(self) -> bool:
        try:
            return bool(
                self.chk_mni_atlas is not None
                and self.chk_mni_atlas.isChecked()
                and bool(getattr(self, "_mni_t1_slices_visible", False))
                and self._get_mni_template_t1_image() is not None
            )
        except Exception:
            return False

    def _get_3d_plane_reference_img(self) -> sitk.Image | None:
        """
        Geometry reference for coronal/axial/sagittal planes.

        Native mode:
            patient T1

        MNI mode + MNI T1 slices:
            MNI template T1
        """
        if self._mni_t1_slices_are_visible():
            return self._get_mni_template_t1_image()

        return self._t1_img

    def _get_3d_plane_mask_for_slices(self, ref_img: sitk.Image | None) -> sitk.Image | None:
        """
        Mask used to crop slice texture.

        Native mode:
            active native surface mask / brainmask

        MNI mode:
            MNI template brain mask
        """
        if ref_img is None:
            return None

        try:
            if self._mni_t1_slices_are_visible():
                mask = self._get_mni_template_mask_image()
                if mask is None:
                    return None

                return sitk.Resample(
                    mask,
                    ref_img,
                    sitk.Transform(),
                    sitk.sitkNearestNeighbor,
                    0,
                    sitk.sitkUInt8,
                )
        except Exception:
            return None

        try:
            return self._get_active_surface_mask_resampled_to(ref_img)
        except Exception:
            return None
