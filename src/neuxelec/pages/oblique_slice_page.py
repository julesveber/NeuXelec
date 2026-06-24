from __future__ import annotations

import numpy as np
import pyvista as pv
import scipy.io as sio
import SimpleITK as sitk
from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollBar,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from scipy.ndimage import map_coordinates

from neuxelec.ui.context_menus import exec_oblique_slice_menu
from neuxelec.ui.export_coordinates_dialog import ExportCoordinatesDialog
from neuxelec.ui.neuxelec_message_dialog import (
    NeuXelecMessageDialog,
    NeuXelecSelectionDialog,
)
from neuxelec.ui.page_loading_overlay import PageLoadingOverlay
from neuxelec.ui.pyvista_quick_tools import PyVistaQuickTools
from neuxelec.ui.slice_quick_tools import SliceQuickTools

from ..utils.pet_visualization import (
    blend_pet_on_rgb,
    get_pet_window,
    normalize_pet_slice,
    normalize_threshold_map,
    pet_norm_to_colormap,
)
from ..utils.siscom_visualization import (
    get_siscom_window,
)


def _top_level_window():
    from PySide6.QtWidgets import QApplication

    for w in QApplication.topLevelWidgets():
        try:
            if w.isVisible():
                return w
        except Exception:
            pass
    return None


class ObliqueSlicePage(QObject):
    """
    First implementation of the Oblique Slice page.

    Current scope:
      - use the EXISTING blue checkboxes in tv_Electrodes_2
      - 1st checked electrode -> 1st oblique slice
      - 2nd checked electrode -> 2nd oblique slice
      - oblique slice rotates around the electrode axis with the side scrollbar

    Notes:
      - images / electrodes are assumed to be in the same LPS physical space
        (CT in T1 space, contacts_lps in LPS mm)
      - frame_visuBrain is left as a placeholder for the next step
    """

    def __init__(self, ui, state):
        super().__init__()
        self.ui = ui
        self.state = state

        self._is_active_page = False
        self.rot1 = None
        self.rot2 = None
        self._rot_dragging_slot = None
        self._rot_drag_pending_slot = None

        # ---------------------------------------------------------
        # Display orientation of the two oblique 2D slices.
        #
        # Same convention as Reconstruction:
        # - research   = neurologic convention, L displayed on the left
        #                -> left/right mirror ON
        # - radiologic = radiologic convention, R displayed on the left
        #                -> left/right mirror OFF
        # ---------------------------------------------------------
        self._orientation_mode = "research"
        self._flip_lr_oblique = True

        self.btn_radiologic_view: QAbstractButton | None = ui.findChild(
            QAbstractButton,
            "btn_ObliqueSlice_radiologicView",
        )
        self.btn_research_view: QAbstractButton | None = ui.findChild(
            QAbstractButton,
            "btn_ObliqueSlice_ResearchView",
        )

        self._orientation_button_group = QButtonGroup(self.ui)
        self._orientation_button_group.setExclusive(True)

        if self.btn_radiologic_view is not None:
            self.btn_radiologic_view.setCheckable(True)
            self._orientation_button_group.addButton(
                self.btn_radiologic_view,
            )
            self.btn_radiologic_view.clicked.connect(
                lambda: self.set_orientation_mode("radiologic")
            )

        if self.btn_research_view is not None:
            self.btn_research_view.setCheckable(True)
            self._orientation_button_group.addButton(
                self.btn_research_view,
            )
            self.btn_research_view.clicked.connect(lambda: self.set_orientation_mode("research"))

        # -------- Widgets --------
        self.frame1: QFrame | None = ui.findChild(
            QFrame,
            "frame_1_ObliqueSlice",
        )
        self.frame2: QFrame | None = ui.findChild(
            QFrame,
            "frame_2_ObliqueSlice",
        )
        self.frame_brain: QFrame | None = ui.findChild(
            QFrame,
            "frame_visuBrain",
        )

        # Overlay qui masque toute la page pendant le rendu des coupes
        # et de la prévisualisation 3D.
        self._page_widget: QWidget | None = ui.findChild(
            QWidget,
            "pageObliqueSlices",
        )

        self._loading_overlay = (
            PageLoadingOverlay(
                self._page_widget,
                "OBLIQUE SLICE",
                "Preparing oblique slices",
            )
            if self._page_widget is not None
            else None
        )

        # ============================================================
        # Asynchronous oblique-slice GIF export
        # ============================================================
        self._gif_export_active = False
        self._gif_export_state = None

        self._gif_export_timer = QTimer(self)
        self._gif_export_timer.setSingleShot(True)
        self._gif_export_timer.timeout.connect(self._export_next_oblique_gif_frame)

        # image labels
        self.image1 = QLabel(self.frame1) if self.frame1 is not None else None
        self.image2 = QLabel(self.frame2) if self.frame2 is not None else None

        self._ct_img_cache = None

        for lbl in (self.image1, self.image2):
            if lbl is None:
                continue

            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background-color: black;")
            lbl.setText("No slice")

            # Important for hover tooltip:
            # without this, MouseMove is not emitted unless a mouse button is pressed.
            lbl.setMouseTracking(True)
            lbl.setAttribute(Qt.WA_Hover, True)

            lbl.installEventFilter(self)
            lbl.setContextMenuPolicy(Qt.CustomContextMenu)
            lbl.customContextMenuRequested.connect(self._show_slice_context_menu)

        if self.frame1 is not None:
            self.frame1.setStyleSheet("")
            self.frame1.setMouseTracking(True)
            self.frame1.setAttribute(Qt.WA_Hover, True)
            self.frame1.installEventFilter(self)

        if self.frame2 is not None:
            self.frame2.setStyleSheet("")
            self.frame2.setMouseTracking(True)
            self.frame2.setAttribute(Qt.WA_Hover, True)
            self.frame2.installEventFilter(self)

        # name badges
        self.badge1 = QLabel(self.frame1) if self.frame1 is not None else None
        self.badge2 = QLabel(self.frame2) if self.frame2 is not None else None

        for badge in (self.badge1, self.badge2):
            if badge is None:
                continue
            badge.setAlignment(Qt.AlignCenter)
            badge.setText("No electrode")
            badge.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            badge.setStyleSheet(
                "QLabel { background-color: black; color: white; padding: 3px 8px; border-radius: 4px; }"
            )
            badge.adjustSize()
            badge.raise_()

        if self.frame1 is not None:
            self.frame1.setStyleSheet("")
            self.frame1.installEventFilter(self)
        if self.frame2 is not None:
            self.frame2.setStyleSheet("")
            self.frame2.installEventFilter(self)
        if self.frame_brain is not None:
            self.frame_brain.setStyleSheet("""
                QFrame {
                    background-color: transparent;
                    border: none;
                }
            """)

        # -------- Modalities --------
        self.chk_ct: QCheckBox | None = ui.findChild(QCheckBox, "chk_obliqueSlice_CT")
        self.chk_mri: QCheckBox | None = ui.findChild(QCheckBox, "chk_obliqueSlice_MRI")
        self.chk_pet: QCheckBox | None = ui.findChild(QCheckBox, "chk_obliqueSlice_PET")

        # MRI source selection inside Oblique Slice
        self.chk_mri_t1: QCheckBox | None = ui.findChild(QCheckBox, "checkBox_ObliqueSlice_T1")
        self.chk_mri_t2: QCheckBox | None = ui.findChild(QCheckBox, "checkBox_ObliqueSlice_T2")

        self.sld_ct: QSlider | None = ui.findChild(QSlider, "horizontalSlider_obliqueSlice_CT")
        self.sld_mri: QSlider | None = ui.findChild(QSlider, "horizontalSlider_obliqueSlice_MRI")

        self.spn_ct: QSpinBox | None = ui.findChild(QSpinBox, "spinBox_obliqueSlice_CT")
        self.spn_mri: QSpinBox | None = ui.findChild(QSpinBox, "spinBox_obliqueSlice_MRI")

        self.chk_siscom: QCheckBox | None = ui.findChild(QCheckBox, "chk_obliqueSlice_SISCOM")
        if self.chk_siscom is None:
            self.chk_siscom = ui.findChild(QCheckBox, "chk_obliqueSlice_SPECT")

        self.sld_siscom: QSlider | None = ui.findChild(
            QSlider, "horizontalSlider_obliqueSlice_SISCOM"
        )
        if self.sld_siscom is None:
            self.sld_siscom = ui.findChild(QSlider, "horizontalSlider_obliqueSlice_SPECT")

        self.spn_siscom: QSpinBox | None = ui.findChild(QSpinBox, "sb_obliqueSlice_SISCOM")
        if self.spn_siscom is None:
            self.spn_siscom = ui.findChild(QSpinBox, "spinBox_obliqueSlice_SPECT")

        self.dsb_siscom_z: QDoubleSpinBox | None = ui.findChild(
            QDoubleSpinBox, "doubleSpinBox_ObliqueSlices_zScoreSISCOM"
        )

        self.chk_parcel1: QCheckBox | None = ui.findChild(
            QCheckBox, "chk_obliqueSlice_Parcellation1"
        )
        self.chk_parcel2: QCheckBox | None = ui.findChild(
            QCheckBox, "chk_obliqueSlice_Parcellation2"
        )

        self.sld_parcel1: QSlider | None = ui.findChild(
            QSlider, "horizontalSlider_obliqueSlice_Parcellation1"
        )
        self.sld_parcel2: QSlider | None = ui.findChild(
            QSlider, "horizontalSlider_obliqueSlice_Parcellation2"
        )

        self.spn_parcel1: QSpinBox | None = ui.findChild(
            QSpinBox, "spinBox_obliqueSlice_Parcellation1"
        )
        self.spn_parcel2: QSpinBox | None = ui.findChild(
            QSpinBox, "spinBox_obliqueSlice_Parcellation2"
        )

        self._connect_opacity(self.sld_parcel1, self.spn_parcel1)
        self._connect_opacity(self.sld_parcel2, self.spn_parcel2)

        if self.chk_parcel1 is not None:
            self.chk_parcel1.toggled.connect(self._on_chk_parcel1_toggled)

        if self.chk_parcel2 is not None:
            self.chk_parcel2.toggled.connect(self._on_chk_parcel2_toggled)

        self._parcel1_img = None
        self._parcel2_img = None
        self._parcel1_lut = {}
        self._parcel2_lut = {}

        self.tbl_parcellation_contacts: QTableWidget | None = ui.findChild(
            QTableWidget, "tableWidget_oblique8parcellationContacts"
        )

        if self.tbl_parcellation_contacts is not None:
            try:
                self.tbl_parcellation_contacts.setColumnCount(3)
                self.tbl_parcellation_contacts.setHorizontalHeaderLabels(
                    ["Contact", "Label", "Region"]
                )
                self.tbl_parcellation_contacts.setRowCount(0)

                header = self.tbl_parcellation_contacts.horizontalHeader()
                header.setStretchLastSection(True)
                header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
                header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
                header.setSectionResizeMode(2, QHeaderView.Stretch)

                self.tbl_parcellation_contacts.verticalHeader().setVisible(False)
            except Exception:
                pass

        self._siscom_colormap_name = "hot"

        self._pet_colormap_name = "jet"
        self._show_color_scales = True

        # Current MRI source for oblique slices: "T1" or "T2"
        self._oblique_mri_source = "T1"

        self.sld_pet_min: QSlider | None = ui.findChild(
            QSlider, "horizontalSlider_obliqueSlice_petMin"
        )
        self.spn_pet_min: QSpinBox | None = ui.findChild(QSpinBox, "sb_obliqueSlice_petMin")

        self.sld_pet_max: QSlider | None = ui.findChild(
            QSlider, "horizontalSlider_obliqueSlice_petMax"
        )
        self.spn_pet_max: QSpinBox | None = ui.findChild(QSpinBox, "sb_obliqueSlice_petMax")

        self.sld_pet_gamma: QSlider | None = ui.findChild(
            QSlider, "horizontalSlider_obliqueSlice_petGamma"
        )
        self.spn_pet_gamma: QDoubleSpinBox | None = ui.findChild(
            QDoubleSpinBox, "dsb_obliqueSlice_petGamma"
        )

        self.sld_pet_opacity: QSlider | None = ui.findChild(
            QSlider, "horizontalSlider_obliqueSlice_petOpacity"
        )
        self.spn_pet_opacity: QSpinBox | None = ui.findChild(
            QSpinBox, "spinBox_obliqueSlice_petOpacity"
        )

        # -------- Rotation sliders --------
        self.rot1: QScrollBar | None = ui.findChild(QScrollBar, "verticalScrollBar_1st")
        self.rot2: QScrollBar | None = ui.findChild(QScrollBar, "verticalScrollBar_2nd")

        # -------- Electrode tree --------
        self.tree: QTreeWidget | None = ui.findChild(QTreeWidget, "tv_Electrodes_2")
        self.btn_export_coordinates: QPushButton | None = ui.findChild(
            QPushButton, "Export_Coordinates_2"
        )
        if self.btn_export_coordinates is not None:
            self.btn_export_coordinates.clicked.connect(self._open_export_coordinates_dialog)
        # -------- Parameters --------
        self.slice_h_px = 520
        self.slice_w_px = 340
        self.slice_length_mm = 80.0
        self.slice_width_mm = 50.0

        self._last_pm1 = None
        self._last_pm2 = None

        # Last projected contacts for GIF export.
        self._last_gif_contacts_1 = None
        self._last_gif_contacts_2 = None

        # Display-only settings for the two 2D slice images.
        # These do not change the anatomical oblique plane.
        # They only rotate the final QPixmap around its centre and optionally
        # make black pixels transparent.
        self._display_rotation1 = 0.0
        self._display_rotation2 = 0.0
        self._slice_background_removed1 = False
        self._slice_background_removed2 = False
        self._rotation_dragging_slot = None
        self._rotation_drag_last_angle = None

        self._zoom1 = 1.0
        self._zoom2 = 1.0

        self._slice_cache_1 = None
        self._slice_cache_2 = None

        self._render_cache_1 = None
        self._render_cache_2 = None
        self._base_cache_1 = None
        self._base_cache_2 = None
        self._pet_cache_1 = None
        self._pet_cache_2 = None

        self.spn_contact_size: QSpinBox | None = ui.findChild(
            QSpinBox, "spinBox_obliqueSlice_SizeContacts"
        )

        self._pan1 = [0.0, 0.0]  # x, y in source-pixmap pixels
        self._pan2 = [0.0, 0.0]

        self._dragging_slot = None
        self._drag_last_pos = None

        # Floating quick-tools for the two 2D oblique slice images.
        self._slice_tools_1 = None
        self._slice_tools_2 = None

        try:
            if self.frame1 is not None:
                self._slice_tools_1 = SliceQuickTools(self.frame1, self, slot_index=1)
                self._slice_tools_1.raise_()

            if self.frame2 is not None:
                self._slice_tools_2 = SliceQuickTools(self.frame2, self, slot_index=2)
                self._slice_tools_2.raise_()

            self._update_oblique_toolbars_geometry()

        except Exception as e:
            print("[ObliqueSlice Slice Quick Tools] Failed to create toolbar:", e)

        self._brain_plotter = None
        self._brain_actor = None
        self._quick_tools = None
        self._oblique_saved_camera = None
        # Adaptive floating toolbars
        self._oblique_toolbars_resize_hook_installed = False

        self._plane_actor_1 = None
        self._plane_actor_2 = None
        self._plane_outline_actor_1 = None
        self._plane_outline_actor_2 = None

        self._fmh_mesh = None

        # Size of the oblique planes shown in the small 3D preview.
        # This does not change the actual oblique slice extraction.
        self._oblique_preview_plane_length_mm = 200.0
        self._oblique_preview_plane_width_mm = 200.0

        self._is_rendering_brain = False
        self._last_brain_key = None
        self._last_plane1_key = None
        self._last_plane2_key = None
        self._last_brain_kind = None

        # Coalesced refresh to avoid repeated renders
        self._pending_refresh_slices = False
        self._pending_refresh_brain = False
        self._freeze_electrode_visibility_refresh = False
        self._freeze_pending_slices = False
        self._freeze_pending_brain = False

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._flush_pending_refresh)

        self._last_checked_electrode_names = None
        self._last_displayed_electrode_slot1 = None
        self._last_displayed_electrode_slot2 = None
        # Page-specific label visibility (must NOT be shared with other pages)
        self._page_contact_labels_visible = {}  # elec_id -> [bool, bool, ...]
        self._page_electrode_visible = {}  # elec_id -> bool
        self._page_contacts_visible = {}  # elec_id -> [bool, bool, ...]

        # -------- Connections --------
        self._connect_opacity(self.sld_ct, self.spn_ct)
        self._connect_opacity(self.sld_mri, self.spn_mri)
        self._connect_opacity(self.sld_pet_opacity, self.spn_pet_opacity)
        self._connect_opacity(self.sld_pet_min, self.spn_pet_min)
        self._connect_opacity(self.sld_pet_max, self.spn_pet_max)
        self._connect_opacity(self.sld_siscom, self.spn_siscom)

        if self.sld_pet_gamma is not None:
            try:
                self.sld_pet_gamma.setRange(10, 300)  # gamma 0.10 -> 3.00
                if self.sld_pet_gamma.value() <= 0:
                    self.sld_pet_gamma.setValue(100)  # default gamma = 1.00
            except Exception:
                pass

        if self.spn_pet_gamma is not None:
            try:
                self.spn_pet_gamma.setRange(0.10, 3.00)
                self.spn_pet_gamma.setSingleStep(0.05)
                if self.spn_pet_gamma.value() <= 0:
                    self.spn_pet_gamma.setValue(1.00)
            except Exception:
                pass

        if self.sld_pet_gamma is not None and self.spn_pet_gamma is not None:
            self.sld_pet_gamma.valueChanged.connect(
                lambda v: self.spn_pet_gamma.setValue(v / 100.0)
            )
            self.spn_pet_gamma.valueChanged.connect(
                lambda v: self.sld_pet_gamma.setValue(int(round(v * 100)))
            )
            self.sld_pet_gamma.valueChanged.connect(
                lambda _=None: self._schedule_refresh(slices=True, brain=False)
            )
            self.spn_pet_gamma.valueChanged.connect(
                lambda _=None: self._schedule_refresh(slices=True, brain=False)
            )

        elif self.sld_pet_gamma is not None:
            self.sld_pet_gamma.valueChanged.connect(
                lambda _=None: self._schedule_refresh(slices=True, brain=False)
            )

        for cb in (self.chk_ct, self.chk_mri, self.chk_pet, self.chk_siscom):
            if cb is not None:
                cb.toggled.connect(
                    lambda _=None: self._turn_off_parcellations_if_other_overlay_selected()
                )
                cb.toggled.connect(lambda _=None: self._update_modality_controls_enabled_states())
                cb.toggled.connect(
                    lambda checked=False, _cb=cb: self._on_oblique_modality_toggled(_cb, checked)
                )

        if self.chk_mri_t1 is not None:
            self.chk_mri_t1.toggled.connect(
                lambda checked=False: self._on_oblique_mri_source_toggled("T1", checked)
            )

        if self.chk_mri_t2 is not None:
            self.chk_mri_t2.toggled.connect(
                lambda checked=False: self._on_oblique_mri_source_toggled("T2", checked)
            )

        if self.rot1 is not None:
            self.rot1.setRange(0, 359)
            self.rot1.setTracking(False)
            self.rot1.installEventFilter(self)

        if self.rot2 is not None:
            self.rot2.setRange(0, 359)
            self.rot2.setTracking(False)
            self.rot2.installEventFilter(self)

        if self.dsb_siscom_z is not None:
            self.dsb_siscom_z.valueChanged.connect(
                lambda _=None: self._schedule_refresh(slices=True, brain=False)
            )

        if self.spn_contact_size is not None:
            try:
                self.spn_contact_size.setRange(1, 20)
                if self.spn_contact_size.value() <= 0:
                    self.spn_contact_size.setValue(4)
            except Exception:
                pass
            self.spn_contact_size.valueChanged.connect(
                lambda _=None: self._schedule_refresh(slices=True, brain=False)
            )

        try:
            if self.chk_ct is not None:
                self.chk_ct.blockSignals(True)
                self.chk_ct.setChecked(True)
                self.chk_ct.blockSignals(False)
        except Exception:
            pass

        try:
            if self.chk_mri is not None:
                self.chk_mri.blockSignals(True)
                self.chk_mri.setChecked(True)
                self.chk_mri.blockSignals(False)
        except Exception:
            pass
        try:
            if self.chk_mri_t1 is not None:
                self.chk_mri_t1.blockSignals(True)
                self.chk_mri_t1.setChecked(True)
                self.chk_mri_t1.blockSignals(False)

            if self.chk_mri_t2 is not None:
                self.chk_mri_t2.blockSignals(True)
                self.chk_mri_t2.setChecked(False)
                self.chk_mri_t2.blockSignals(False)
        except Exception:
            pass
        try:
            if self.sld_ct is not None:
                self.sld_ct.blockSignals(True)
                self.sld_ct.setValue(50)
                self.sld_ct.blockSignals(False)
            if self.spn_ct is not None:
                self.spn_ct.blockSignals(True)
                self.spn_ct.setValue(50)
                self.spn_ct.blockSignals(False)

            if self.sld_mri is not None:
                self.sld_mri.blockSignals(True)
                self.sld_mri.setValue(50)
                self.sld_mri.blockSignals(False)
            if self.spn_mri is not None:
                self.spn_mri.blockSignals(True)
                self.spn_mri.setValue(50)
                self.spn_mri.blockSignals(False)
        except Exception:
            pass
        self._update_modality_controls_enabled_states()

        self._init_brain_view()
        self._apply_oblique_slice_initial_saved_views()
        self._relayout_labels()
        self.apply_default_pet_siscom_settings()

        # Same default orientation as Reconstruction.
        self.set_orientation_mode("research", refresh=False)

    def _dialog_parent(self):
        """
        Return the main NeuXelec window used as parent for styled dialogs.
        """
        try:
            if self.ui is not None:
                return self.ui.window()
        except Exception:
            pass

        return _top_level_window()

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

    def _update_orientation_buttons(self) -> None:
        """
        Keep Radiologic / Research buttons visually synchronized with
        the active oblique slice orientation mode.
        """
        try:
            if self.btn_research_view is not None:
                self.btn_research_view.blockSignals(True)
                self.btn_research_view.setChecked(self._orientation_mode == "research")
                self.btn_research_view.blockSignals(False)

            if self.btn_radiologic_view is not None:
                self.btn_radiologic_view.blockSignals(True)
                self.btn_radiologic_view.setChecked(self._orientation_mode == "radiologic")
                self.btn_radiologic_view.blockSignals(False)

        except Exception:
            pass

    def set_orientation_mode(
        self,
        mode: str,
        refresh: bool = True,
    ) -> None:
        """
        Set the left/right display convention of both oblique 2D slices.

        Same convention as Reconstruction:
        - research   / neurologic view: L displayed on the left
          -> horizontal flip enabled
        - radiologic view: R displayed on the left
          -> horizontal flip disabled

        The mini 3D preview is not changed because this setting only
        concerns the convention of the two displayed 2D slices.
        """
        mode = str(mode or "").lower().strip()

        if mode not in ("research", "radiologic"):
            mode = "research"

        self._orientation_mode = mode
        self._flip_lr_oblique = mode == "research"

        self._update_orientation_buttons()

        if refresh:
            try:
                self.render_slices_only()
            except Exception:
                pass

    def set_active_page(self, active: bool):
        self._is_active_page = bool(active)

        if not self._is_active_page:
            if self._loading_overlay is not None:
                self._loading_overlay.cancel()

            try:
                self._refresh_timer.stop()
            except Exception:
                pass

            self._pending_refresh_brain = False

            if self._brain_plotter is not None:
                try:
                    self._brain_plotter.disable()
                except Exception:
                    pass

                try:
                    self._brain_plotter.hide()
                except Exception:
                    pass

            return

        # Le loader apparaît immédiatement. Le rendu commence seulement
        # au cycle Qt suivant, afin que l’utilisateur voie d’abord le logo.
        if self._loading_overlay is not None:
            self._loading_overlay.begin("Preparing oblique slices")

        QTimer.singleShot(0, self._activate_oblique_slices_step)

    def _activate_oblique_slices_step(self) -> None:
        if not self._is_active_page:
            return

        self._render_cache_1 = None
        self._render_cache_2 = None
        self._base_cache_1 = None
        self._base_cache_2 = None
        self._pet_cache_1 = None
        self._pet_cache_2 = None

        self._slice_cache_1 = None
        self._slice_cache_2 = None

        self._last_brain_key = None
        self._last_plane1_key = None
        self._last_plane2_key = None
        self._last_brain_kind = None

        current = tuple(self._get_checked_electrode_names())

        self._last_checked_electrode_names = current
        self._last_displayed_electrode_slot1 = current[0] if len(current) > 0 else None
        self._last_displayed_electrode_slot2 = current[1] if len(current) > 1 else None

        if self._loading_overlay is not None:
            self._loading_overlay.set_progress(
                0.28,
                "Rendering oblique slices",
            )

        self.render_slices_only()

        if self._loading_overlay is not None:
            self._loading_overlay.set_progress(
                0.62,
                "Building 3D orientation preview",
            )

        QTimer.singleShot(0, self._activate_oblique_brain_step)

    def _activate_oblique_brain_step(self) -> None:
        if not self._is_active_page:
            return

        if self._brain_plotter is not None:
            try:
                self._brain_plotter.show()
            except Exception:
                pass

            try:
                self._brain_plotter.enable()
            except Exception:
                pass

        self.render_brain_only()

        if self._loading_overlay is not None:
            self._loading_overlay.set_progress(
                0.94,
                "Ready",
            )
            self._loading_overlay.complete()

    def apply_default_pet_siscom_settings(self) -> None:
        # PET defaults
        try:
            if self.sld_pet_min is not None:
                self.sld_pet_min.blockSignals(True)
                self.sld_pet_min.setValue(15)
                self.sld_pet_min.blockSignals(False)
            if self.spn_pet_min is not None:
                self.spn_pet_min.blockSignals(True)
                self.spn_pet_min.setValue(15)
                self.spn_pet_min.blockSignals(False)

            if self.sld_pet_max is not None:
                self.sld_pet_max.blockSignals(True)
                self.sld_pet_max.setValue(75)
                self.sld_pet_max.blockSignals(False)
            if self.spn_pet_max is not None:
                self.spn_pet_max.blockSignals(True)
                self.spn_pet_max.setValue(75)
                self.spn_pet_max.blockSignals(False)

            if self.sld_pet_gamma is not None:
                self.sld_pet_gamma.blockSignals(True)
                self.sld_pet_gamma.setValue(20)  # gamma = 0.20
                self.sld_pet_gamma.blockSignals(False)
            if self.spn_pet_gamma is not None:
                self.spn_pet_gamma.blockSignals(True)
                self.spn_pet_gamma.setValue(0.20)
                self.spn_pet_gamma.blockSignals(False)

            if self.sld_pet_opacity is not None:
                self.sld_pet_opacity.blockSignals(True)
                self.sld_pet_opacity.setValue(50)
                self.sld_pet_opacity.blockSignals(False)
            if self.spn_pet_opacity is not None:
                self.spn_pet_opacity.blockSignals(True)
                self.spn_pet_opacity.setValue(50)
                self.spn_pet_opacity.blockSignals(False)
        except Exception:
            pass

        # SISCOM defaults
        try:
            if self.dsb_siscom_z is not None:
                self.dsb_siscom_z.blockSignals(True)
                self.dsb_siscom_z.setValue(2.0)
                self.dsb_siscom_z.blockSignals(False)

            if self.sld_siscom is not None:
                self.sld_siscom.blockSignals(True)
                self.sld_siscom.setValue(50)
                self.sld_siscom.blockSignals(False)
            if self.spn_siscom is not None:
                self.spn_siscom.blockSignals(True)
                self.spn_siscom.setValue(50)
                self.spn_siscom.blockSignals(False)
        except Exception:
            pass

        # Keep PET/SISCOM hidden by default
        try:
            if self.chk_pet is not None:
                self.chk_pet.blockSignals(True)
                self.chk_pet.setChecked(False)
                self.chk_pet.blockSignals(False)
        except Exception:
            pass

        try:
            if self.chk_siscom is not None:
                self.chk_siscom.blockSignals(True)
                self.chk_siscom.setChecked(False)
                self.chk_siscom.blockSignals(False)
        except Exception:
            pass

        self._slice_cache_1 = None
        self._slice_cache_2 = None
        self._render_cache_1 = None
        self._render_cache_2 = None
        self._base_cache_1 = None
        self._base_cache_2 = None
        self._pet_cache_1 = None
        self._pet_cache_2 = None
        self._schedule_refresh(slices=True, brain=False)

    def _rotation_scrollbar_slot(self, obj):
        rot1 = getattr(self, "rot1", None)
        rot2 = getattr(self, "rot2", None)

        if rot1 is not None and obj is rot1:
            return 1
        if rot2 is not None and obj is rot2:
            return 2
        return None

    def _set_rotation_scrollbar_from_mouse(self, sb: QScrollBar, event) -> None:
        """
        Set the scrollbar value directly from mouse position.

        For these vertical QScrollBars:
        top    -> minimum value
        bottom -> maximum value
        """
        if sb is None:
            return

        try:
            try:
                y = float(event.position().y())
            except Exception:
                y = float(event.pos().y())

            h = max(1.0, float(sb.height() - 1))
            y = max(0.0, min(y, h))

            vmin = int(sb.minimum())
            vmax = int(sb.maximum())

            value = int(round(vmin + (y / h) * float(vmax - vmin)))
            value = max(vmin, min(value, vmax))

            sb.blockSignals(True)
            sb.setValue(value)
            sb.blockSignals(False)

        except Exception:
            pass

    def _display_delta_to_source_delta(self, slot_index: int, delta) -> tuple[float, float]:
        """
        Convert mouse movement in the fixed QLabel canvas into source-pixmap pixels.

        The slice content is visually rotated around the fixed canvas centre.
        To make dragging feel natural after rotation, we convert the mouse movement
        back through the inverse display rotation.
        """
        try:
            if int(slot_index) == 1:
                pm = self._last_pm1
                zoom = float(self._zoom1)
                label = self.image1
            else:
                pm = self._last_pm2
                zoom = float(self._zoom2)
                label = self.image2

            if pm is None or pm.isNull() or label is None:
                return float(delta.x()), float(delta.y())

            src_w = float(pm.width())
            src_h = float(pm.height())

            zoom = max(1.0, zoom)

            crop_w = max(1.0, src_w / zoom)
            crop_h = max(1.0, src_h / zoom)

            target_w = max(1.0, float(label.width()))
            target_h = max(1.0, float(label.height()))

            # Same fit logic used in _render_oblique_slice_display_canvas.
            _draw_x, _draw_y, _draw_w, _draw_h, scale = self._fit_source_rect_to_target(
                crop_w,
                crop_h,
                target_w,
                target_h,
            )

            dx_display = float(delta.x())
            dy_display = float(delta.y())

            # Inverse of the visual display rotation.
            angle = float(self._get_oblique_slice_rotation(slot_index)) % 360.0
            theta = np.deg2rad(-angle)

            cos_t = float(np.cos(theta))
            sin_t = float(np.sin(theta))

            dx_unrotated = cos_t * dx_display - sin_t * dy_display
            dy_unrotated = sin_t * dx_display + cos_t * dy_display

            dx_src = dx_unrotated / max(scale, 1e-6)
            dy_src = dy_unrotated / max(scale, 1e-6)

            return dx_src, dy_src

        except Exception:
            return float(delta.x()), float(delta.y())

    # ---------- UI helpers ----------
    def eventFilter(self, obj, event):
        try:
            rot_slot = self._rotation_scrollbar_slot(obj)
        except Exception:
            rot_slot = None

        # ---------------------------------------------------------
        # Rotation vertical scrollbars:
        # - left click anywhere -> jump directly to that position
        # - drag -> update cursor position only
        # - refresh only when mouse is released
        # ---------------------------------------------------------

        if rot_slot is not None:
            if event.type() == QEvent.MouseButtonPress:
                try:
                    if event.button() == Qt.LeftButton:
                        self._rot_dragging_slot = int(rot_slot)
                        self._rot_drag_pending_slot = int(rot_slot)
                        self._set_rotation_scrollbar_from_mouse(obj, event)
                        return True
                except Exception:
                    pass

            elif event.type() == QEvent.MouseMove:
                try:
                    if self._rot_dragging_slot == int(rot_slot):
                        self._rot_drag_pending_slot = int(rot_slot)
                        self._set_rotation_scrollbar_from_mouse(obj, event)
                        return True
                except Exception:
                    pass

            elif event.type() == QEvent.MouseButtonRelease:
                try:
                    if self._rot_dragging_slot == int(rot_slot):
                        self._set_rotation_scrollbar_from_mouse(obj, event)

                        pending_slot = self._rot_drag_pending_slot
                        self._rot_dragging_slot = None
                        self._rot_drag_pending_slot = None

                        if pending_slot is not None:
                            self._on_rotation_changed(slot_index=int(pending_slot))

                        return True
                except Exception:
                    self._rot_dragging_slot = None
                    self._rot_drag_pending_slot = None
                    return True

            return False

        if event.type() == QEvent.Resize:
            if obj in (
                self.frame1,
                self.frame2,
                self.image1,
                self.image2,
                self.frame_brain,
                getattr(self, "_brain_plotter", None),
            ):
                self._relayout_labels()
                self._update_oblique_toolbars_geometry()

        elif event.type() == QEvent.Wheel:
            delta = event.angleDelta().y()

            # ONLY handle wheel for the 2 slice views
            if obj in (self.frame1, self.image1):
                if delta > 0:
                    self._zoom1 *= 1.15
                else:
                    self._zoom1 /= 1.15
                self._zoom1 = max(1.0, min(8.0, self._zoom1))
                self._schedule_refresh(slices=True, brain=False)
                return True

            elif obj in (self.frame2, self.image2):
                if delta > 0:
                    self._zoom2 *= 1.15
                else:
                    self._zoom2 /= 1.15
                self._zoom2 = max(1.0, min(8.0, self._zoom2))
                self._schedule_refresh(slices=True, brain=False)
                return True

            # DO NOT intercept wheel for the brain viewer
            return False

        elif event.type() == QEvent.MouseButtonPress:
            # ONLY handle mouse press for the 2 slice views
            if obj in (self.frame1, self.image1):
                try:
                    ctrl_down = bool(event.modifiers() & Qt.ControlModifier)
                except Exception:
                    ctrl_down = False

                if ctrl_down:
                    self._rotation_dragging_slot = 1
                    self._rotation_drag_last_angle = self._slice_local_angle_from_global_pos(
                        1, event.globalPosition().toPoint()
                    )
                    return True

                self._dragging_slot = 1
                self._drag_last_pos = event.globalPosition().toPoint()
                return True

            elif obj in (self.frame2, self.image2):
                try:
                    ctrl_down = bool(event.modifiers() & Qt.ControlModifier)
                except Exception:
                    ctrl_down = False

                if ctrl_down:
                    self._rotation_dragging_slot = 2
                    self._rotation_drag_last_angle = self._slice_local_angle_from_global_pos(
                        2, event.globalPosition().toPoint()
                    )
                    return True

                self._dragging_slot = 2
                self._drag_last_pos = event.globalPosition().toPoint()
                return True

            # DO NOT intercept mouse press for the brain viewer
            return False

        elif event.type() == QEvent.MouseMove:
            if self._rotation_dragging_slot is not None:
                try:
                    slot = int(self._rotation_dragging_slot)
                    new_angle = self._slice_local_angle_from_global_pos(
                        slot, event.globalPosition().toPoint()
                    )
                    old_angle = self._rotation_drag_last_angle

                    if new_angle is not None and old_angle is not None:
                        delta_angle = float(new_angle) - float(old_angle)

                        # Keep the shortest angular path when crossing +/-180°.
                        if delta_angle > 180.0:
                            delta_angle -= 360.0
                        elif delta_angle < -180.0:
                            delta_angle += 360.0

                        self._rotation_drag_last_angle = new_angle
                        self._rotate_oblique_slice_display(slot, delta_angle)
                        return True

                    self._rotation_drag_last_angle = new_angle
                    return True

                except Exception:
                    return True

            if self._dragging_slot is not None and self._drag_last_pos is not None:
                new_pos = event.globalPosition().toPoint()
                delta = new_pos - self._drag_last_pos
                self._drag_last_pos = new_pos

                if self._dragging_slot == 1 and self._zoom1 > 1.0:
                    dx_src, dy_src = self._display_delta_to_source_delta(1, delta)

                    self._pan1[0] -= dx_src
                    self._pan1[1] -= dy_src

                    # Refresh only slice 1, not both slices
                    self.render_slice_slot(1)
                    return True

                elif self._dragging_slot == 2 and self._zoom2 > 1.0:
                    dx_src, dy_src = self._display_delta_to_source_delta(2, delta)

                    self._pan2[0] -= dx_src
                    self._pan2[1] -= dy_src

                    # Refresh only slice 2, not both slices
                    self.render_slice_slot(2)
                    return True

            return False

        elif event.type() == QEvent.MouseButtonRelease:
            if self._rotation_dragging_slot is not None:
                self._rotation_dragging_slot = None
                self._rotation_drag_last_angle = None
                return True

            # only consume release if we were dragging one of the slices
            if self._dragging_slot is not None:
                self._dragging_slot = None
                self._drag_last_pos = None
                return True
            return False

        elif event.type() == QEvent.MouseButtonDblClick:
            try:
                is_left_double_click = event.button() == Qt.LeftButton
            except Exception:
                is_left_double_click = True

            # Mini 3D brain view reset
            if is_left_double_click and obj in (
                self.frame_brain,
                getattr(self, "_brain_plotter", None),
                getattr(getattr(self, "_brain_plotter", None), "interactor", None),
            ):
                self._reset_oblique_brain_camera_coronal()
                return True

            # Slice 1 reset
            if obj in (self.frame1, self.image1):
                self._reset_oblique_slice_display_to_default(
                    1,
                    refresh=True,
                    reset_background=True,
                )
                return True

            # Slice 2 reset
            elif obj in (self.frame2, self.image2):
                self._reset_oblique_slice_display_to_default(
                    2,
                    refresh=True,
                    reset_background=True,
                )
                return True

            return False

        return False

    def begin_electrode_visibility_freeze(self) -> None:
        """
        Freeze oblique slice rendering while the user drags across electrode checkboxes.
        The UI checkboxes can still change, but the 2 oblique slices and mini 3D view
        are not refreshed until end_electrode_visibility_freeze().
        """
        self._freeze_electrode_visibility_refresh = True
        self._freeze_pending_slices = False
        self._freeze_pending_brain = False

        try:
            if self._refresh_timer is not None:
                self._refresh_timer.stop()
        except Exception:
            pass

    def end_electrode_visibility_freeze(self, refresh: bool = True) -> None:
        """
        Unfreeze oblique slice rendering and refresh once.
        """
        self._freeze_electrode_visibility_refresh = False

        do_slices = bool(self._freeze_pending_slices)
        do_brain = bool(self._freeze_pending_brain)

        self._freeze_pending_slices = False
        self._freeze_pending_brain = False

        if refresh:
            # For electrode visibility changes, both slices and mini 3D planes may change.
            self._slice_cache_1 = None
            self._slice_cache_2 = None
            self._base_cache_1 = None
            self._base_cache_2 = None
            self._pet_cache_1 = None
            self._pet_cache_2 = None
            self._last_plane1_key = None
            self._last_plane2_key = None

            self._schedule_refresh(
                slices=True or do_slices,
                brain=True or do_brain,
            )

    def _schedule_refresh(self, slices: bool = False, brain: bool = False):
        if bool(getattr(self, "_freeze_electrode_visibility_refresh", False)):
            if slices:
                self._freeze_pending_slices = True
            if brain:
                self._freeze_pending_brain = True
            return

        if slices:
            self._pending_refresh_slices = True

        # Brain preview only when Oblique page is active.
        if brain and self._is_active_page:
            self._pending_refresh_brain = True

        # If page is inactive, do not start timer only for brain preview.
        if not self._is_active_page and not slices:
            return

        self._refresh_timer.start(20)

    def _flush_pending_refresh(self):
        do_slices = self._pending_refresh_slices
        do_brain = self._pending_refresh_brain

        self._pending_refresh_slices = False
        self._pending_refresh_brain = False

        if do_slices:
            self.render_slices_only()

        if do_brain and self._is_active_page:
            self.render_brain_only()
        else:
            self._pending_refresh_brain = False

    def _adapt_single_floating_toolbar(self, toolbar, parent_frame) -> None:
        """
        Adapt one floating toolbar to its frame.

        If the frame is too narrow to show every icon, the toolbar activates its
        internal circular carousel. Icons can then be browsed with the mouse wheel
        while the cursor is above the toolbar.
        """
        try:
            if toolbar is None or parent_frame is None:
                return

            parent_w = max(1, int(parent_frame.width()))
            parent_h = max(1, int(parent_frame.height()))
            margin = 8

            # Do not reduce tools too much: on small images, switch to carousel
            # while keeping buttons readable.
            if parent_w < 360 or parent_h < 260:
                button_w = 38
                button_h = 28
                spacing_px = 2
            elif parent_w < 520 or parent_h < 380:
                button_w = 38
                button_h = 30
                spacing_px = 3
            elif parent_w < 760 or parent_h < 520:
                button_w = 40
                button_h = 32
                spacing_px = 4
            else:
                button_w = 42
                button_h = 34
                spacing_px = 5

            available_w = max(70, parent_w - 2 * margin)

            if hasattr(toolbar, "set_carousel_available_width"):
                toolbar.set_carousel_available_width(
                    available_width=available_w,
                    button_width=button_w,
                    button_height=button_h,
                    spacing_px=spacing_px,
                )

            toolbar.adjustSize()

            toolbar_w = min(int(toolbar.sizeHint().width()), available_w)
            toolbar_h = min(
                int(toolbar.sizeHint().height()),
                max(34, parent_h - 2 * margin),
            )

            toolbar.resize(toolbar_w, toolbar_h)

            # Keep each toolbar at the bottom-left of its own view.
            x = margin
            y = max(margin, parent_h - toolbar_h - margin)

            toolbar.move(x, y)
            toolbar.raise_()

        except Exception:
            pass

    def _update_oblique_toolbars_geometry(self) -> None:
        """
        Keep the three Oblique Slice toolbars adapted to their current frames:
            - slice 1 toolbar
            - slice 2 toolbar
            - mini 3D brain toolbar
        """
        try:
            self._adapt_single_floating_toolbar(
                getattr(self, "_slice_tools_1", None),
                getattr(self, "frame1", None),
            )

            self._adapt_single_floating_toolbar(
                getattr(self, "_slice_tools_2", None),
                getattr(self, "frame2", None),
            )

            self._adapt_single_floating_toolbar(
                getattr(self, "_quick_tools", None),
                getattr(self, "frame_brain", None),
            )

        except Exception:
            pass

    def _relayout_labels(self):
        for frame, img, badge in (
            (self.frame1, self.image1, self.badge1),
            (self.frame2, self.image2, self.badge2),
        ):
            if frame is None:
                continue
            if img is not None:
                img.setGeometry(frame.rect())
            if badge is not None:
                badge.adjustSize()
                x = max(6, (frame.width() - badge.width()) // 2)
                badge.move(x, 6)
                badge.raise_()
        try:
            self._update_oblique_toolbars_geometry()
        except Exception:
            pass

    def _connect_opacity(self, slider, spinbox):
        if slider is None or spinbox is None:
            return

        try:
            slider.setRange(0, 100)
        except Exception:
            pass
        try:
            spinbox.setRange(0, 100)
        except Exception:
            pass

        slider.valueChanged.connect(spinbox.setValue)
        spinbox.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(lambda _=None: self._schedule_refresh(slices=True, brain=False))
        spinbox.valueChanged.connect(
            lambda _=None: self._schedule_refresh(slices=True, brain=False)
        )

    def _set_checkbox_checked_silent(self, cb, checked: bool) -> None:
        if cb is None:
            return
        try:
            cb.blockSignals(True)
            cb.setChecked(bool(checked))
            cb.blockSignals(False)
        except Exception:
            pass

    def _get_t2_image(self):
        """
        T2 displayed in Oblique Slice must be in T1 space.
        Preferred source: validated/coregistered T2.
        """
        img = getattr(self.state, "t2_coreg_in_t1", None)
        if img is not None:
            return img

        # Backward-compatible alias if used somewhere else
        img = getattr(self.state, "t2_in_t1", None)
        if img is not None:
            return img

        return None

    def _get_active_mri_image(self):
        """
        Return the MRI image selected in the Oblique Slice page.
        T1 is default. T2 is used only if selected and available.
        """
        source = getattr(self, "_oblique_mri_source", "T1")

        if source == "T2":
            t2 = self._get_t2_image()
            if t2 is not None:
                return t2

        return self._get_t1_image()

    def _invalidate_oblique_image_caches(self):
        self._slice_cache_1 = None
        self._slice_cache_2 = None
        self._render_cache_1 = None
        self._render_cache_2 = None
        self._base_cache_1 = None
        self._base_cache_2 = None
        self._pet_cache_1 = None
        self._pet_cache_2 = None

    def _on_oblique_mri_source_toggled(self, source: str, checked: bool):
        if not checked:
            # Prevent both T1 and T2 from being unchecked while MRI is active.
            try:
                if self.chk_mri is not None and self.chk_mri.isChecked():
                    if source == "T1" and not (
                        self.chk_mri_t2 is not None and self.chk_mri_t2.isChecked()
                    ):
                        self._set_checkbox_checked_silent(self.chk_mri_t1, True)
                        self._oblique_mri_source = "T1"
                    elif source == "T2" and not (
                        self.chk_mri_t1 is not None and self.chk_mri_t1.isChecked()
                    ):
                        self._set_checkbox_checked_silent(self.chk_mri_t1, True)
                        self._set_checkbox_checked_silent(self.chk_mri_t2, False)
                        self._oblique_mri_source = "T1"
            except Exception:
                pass
            return

        if source == "T1":
            self._oblique_mri_source = "T1"
            self._set_checkbox_checked_silent(self.chk_mri_t2, False)

        elif source == "T2":
            # Only allow T2 if a T2 in T1 space exists.
            if self._get_t2_image() is None:
                self._set_checkbox_checked_silent(self.chk_mri_t2, False)
                self._set_checkbox_checked_silent(self.chk_mri_t1, True)
                self._oblique_mri_source = "T1"
                return

            self._oblique_mri_source = "T2"
            self._set_checkbox_checked_silent(self.chk_mri_t1, False)

        self._invalidate_oblique_image_caches()
        self._update_modality_controls_enabled_states()
        self._schedule_refresh(slices=True, brain=False)

    def _update_modality_controls_enabled_states(self):
        self._enforce_parcellation_overlay_mode()
        try:
            ct_on = bool(self.chk_ct is not None and self.chk_ct.isChecked())
            if self.sld_ct is not None:
                self.sld_ct.setEnabled(ct_on)
            if self.spn_ct is not None:
                self.spn_ct.setEnabled(ct_on)
        except Exception:
            pass

        try:
            mri_on = bool(self.chk_mri is not None and self.chk_mri.isChecked())

            t1_loaded = bool(self._get_t1_image() is not None)
            t2_loaded = bool(self._get_t2_image() is not None)

            # T1/T2 selectors are clickable only when MRI is checked
            if self.chk_mri_t1 is not None:
                self.chk_mri_t1.setEnabled(bool(mri_on and t1_loaded))

            if self.chk_mri_t2 is not None:
                self.chk_mri_t2.setEnabled(bool(mri_on and t2_loaded))

            # If MRI is off, keep source selection visually stable but disabled
            if not mri_on:
                pass

            # If T2 selected but no T2 available anymore, fallback to T1
            if mri_on and getattr(self, "_oblique_mri_source", "T1") == "T2" and not t2_loaded:
                self._oblique_mri_source = "T1"
                self._set_checkbox_checked_silent(self.chk_mri_t1, True)
                self._set_checkbox_checked_silent(self.chk_mri_t2, False)

            # If MRI is on and neither source is checked, default to T1
            if mri_on:
                t1_checked = bool(self.chk_mri_t1 is not None and self.chk_mri_t1.isChecked())
                t2_checked = bool(self.chk_mri_t2 is not None and self.chk_mri_t2.isChecked())

                if not t1_checked and not t2_checked:
                    self._oblique_mri_source = "T1"
                    self._set_checkbox_checked_silent(self.chk_mri_t1, True)
                    self._set_checkbox_checked_silent(self.chk_mri_t2, False)

            # MRI opacity slider controls whichever MRI source is active: T1 or T2
            active_mri_available = bool(self._get_active_mri_image() is not None)
            mri_controls_on = bool(mri_on and active_mri_available)

            if self.sld_mri is not None:
                self.sld_mri.setEnabled(mri_controls_on)
            if self.spn_mri is not None:
                self.spn_mri.setEnabled(mri_controls_on)

        except Exception:
            pass

        try:
            pet_on = bool(self.chk_pet is not None and self.chk_pet.isChecked())
            if self.sld_pet_min is not None:
                self.sld_pet_min.setEnabled(pet_on)
            if self.spn_pet_min is not None:
                self.spn_pet_min.setEnabled(pet_on)

            if self.sld_pet_max is not None:
                self.sld_pet_max.setEnabled(pet_on)
            if self.spn_pet_max is not None:
                self.spn_pet_max.setEnabled(pet_on)

            if self.sld_pet_gamma is not None:
                self.sld_pet_gamma.setEnabled(pet_on)
            if self.spn_pet_gamma is not None:
                self.spn_pet_gamma.setEnabled(pet_on)

            if self.sld_pet_opacity is not None:
                self.sld_pet_opacity.setEnabled(pet_on)
            if self.spn_pet_opacity is not None:
                self.spn_pet_opacity.setEnabled(pet_on)
        except Exception:
            pass

        try:
            p1_loaded = bool(self._parcel1_img is not None)
            if self.chk_parcel1 is not None:
                self.chk_parcel1.setEnabled(p1_loaded)
                if not p1_loaded:
                    self.chk_parcel1.setChecked(False)

            p1_on = bool(
                self.chk_parcel1 is not None and self.chk_parcel1.isChecked() and p1_loaded
            )
            if self.sld_parcel1 is not None:
                self.sld_parcel1.setEnabled(p1_on)
            if self.spn_parcel1 is not None:
                self.spn_parcel1.setEnabled(p1_on)
        except Exception:
            pass

        try:
            p2_loaded = bool(self._parcel2_img is not None)
            if self.chk_parcel2 is not None:
                self.chk_parcel2.setEnabled(p2_loaded)
                if not p2_loaded:
                    self.chk_parcel2.setChecked(False)
            p2_on = bool(
                self.chk_parcel2 is not None and self.chk_parcel2.isChecked() and p2_loaded
            )
            if self.sld_parcel2 is not None:
                self.sld_parcel2.setEnabled(p2_on)
            if self.spn_parcel2 is not None:
                self.spn_parcel2.setEnabled(p2_on)
        except Exception:
            pass

        try:
            sis_on = bool(self.chk_siscom is not None and self.chk_siscom.isChecked())
            if self.sld_siscom is not None:
                self.sld_siscom.setEnabled(sis_on)
            if self.spn_siscom is not None:
                self.spn_siscom.setEnabled(sis_on)
            if self.dsb_siscom_z is not None:
                self.dsb_siscom_z.setEnabled(sis_on)
        except Exception:
            pass

    def refresh_available_modalities(
        self,
        refresh: bool = True,
        activate_validated: bool = False,
    ) -> None:
        """
        Refresh modality availability after validation, invalidation or
        restoration from the project JSON.

        This prevents a modality from remaining disabled until the user clicks
        another checkbox.
        """
        try:
            for checkbox, image in (
                (self.chk_ct, self._get_ct_image()),
                (self.chk_pet, self._get_pet_image()),
                (self.chk_siscom, self._get_siscom_image()),
            ):
                if checkbox is not None and image is None:
                    self._set_checkbox_checked_silent(checkbox, False)
        except Exception:
            pass
        if activate_validated:
            try:
                pet_ready = bool(self._get_pet_image() is not None)

                if self.chk_pet is not None:
                    self.chk_pet.setEnabled(pet_ready)
                    self._set_checkbox_checked_silent(
                        self.chk_pet,
                        pet_ready,
                    )

                # PET is an overlay mode, so if it is restored active,
                # parcellation overlays should not remain active at the same time.
                if pet_ready:
                    self._set_checkbox_checked_silent(self.chk_parcel1, False)
                    self._set_checkbox_checked_silent(self.chk_parcel2, False)

            except Exception:
                pass
        try:
            self._invalidate_oblique_image_caches()
        except Exception:
            self._slice_cache_1 = None
            self._slice_cache_2 = None
            self._render_cache_1 = None
            self._render_cache_2 = None
            self._base_cache_1 = None
            self._base_cache_2 = None
            self._pet_cache_1 = None
            self._pet_cache_2 = None

        self._update_modality_controls_enabled_states()

        if refresh:
            self._schedule_refresh(
                slices=True,
                brain=False,
            )

    def _enforce_parcellation_overlay_mode(self):
        """
        Rules:
          - Parcellation 1 and 2 are mutually exclusive
            (handled in _on_chk_parcel1_toggled / _on_chk_parcel2_toggled)
          - If any parcellation is checked:
                * MRI is forced on
                * CT / PET / SISCOM are turned off
          - If CT / PET / SISCOM is checked manually later,
            parcellations can be unchecked elsewhere.
          - All checkboxes remain clickable.
        """
        p1_on = bool(self.chk_parcel1 is not None and self.chk_parcel1.isChecked())
        p2_on = bool(self.chk_parcel2 is not None and self.chk_parcel2.isChecked())

        any_parcel_on = p1_on or p2_on

        if any_parcel_on:
            # force MRI on
            try:
                if self.chk_mri is not None:
                    self.chk_mri.blockSignals(True)
                    self.chk_mri.setChecked(True)
                    self.chk_mri.blockSignals(False)
            except Exception:
                pass

            # turn off competing overlays
            for cb in (self.chk_ct, self.chk_pet, self.chk_siscom):
                try:
                    if cb is not None and cb.isChecked():
                        cb.blockSignals(True)
                        cb.setChecked(False)
                        cb.blockSignals(False)
                except Exception:
                    pass

        # keep everything clickable according to availability only
        try:
            if self.chk_ct is not None:
                self.chk_ct.setEnabled(bool(self._get_ct_image() is not None))
            if self.chk_mri is not None:
                self.chk_mri.setEnabled(True)
            if self.chk_pet is not None:
                self.chk_pet.setEnabled(bool(self._get_pet_image() is not None))
            if self.chk_siscom is not None:
                self.chk_siscom.setEnabled(bool(self._get_siscom_image() is not None))
            if self.chk_parcel1 is not None:
                self.chk_parcel1.setEnabled(bool(self._parcel1_img is not None))
            if self.chk_parcel2 is not None:
                self.chk_parcel2.setEnabled(bool(self._parcel2_img is not None))
        except Exception:
            pass

    # ---------- Electrode selection ----------
    def _get_checked_electrode_names(self):
        result = []
        electrodes = getattr(self.state, "electrodes", []) or []

        for elec_id, elec in enumerate(electrodes):
            name = elec.get("name")
            if not name:
                continue
            if self._get_local_electrode_visible(elec_id):
                result.append(name)

        return result

    def _get_electrode_by_name(self, name: str | None):
        if not name:
            return None
        for elec in getattr(self.state, "electrodes", []) or []:
            if isinstance(elec, dict) and elec.get("name") == name:
                return elec
        return None

    # ---------- Geometry ----------
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return np.zeros_like(v, dtype=np.float64)
        return v / n

    @staticmethod
    def _rodrigues_rotate(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
        axis = ObliqueSlicePage._normalize(axis.astype(np.float64))
        v = v.astype(np.float64)
        c = float(np.cos(angle_rad))
        s = float(np.sin(angle_rad))
        return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)

    @staticmethod
    def _lps_to_ras_point(p):
        p = np.asarray(p, dtype=np.float64).copy()
        p[0] *= -1.0
        p[1] *= -1.0
        return p

    @staticmethod
    def _lps_to_ras_vec(v):
        v = np.asarray(v, dtype=np.float64).copy()
        v[0] *= -1.0
        v[1] *= -1.0
        return v

    def _electrode_axis_and_center(self, elec: dict):
        # Best: use deepest/second points if available for the direction
        p0 = elec.get("deepest_lps", None)
        p1 = elec.get("second_lps", None)

        contacts = np.asarray(elec.get("contacts_lps", []) or [], dtype=np.float64)
        if contacts.ndim != 2 or contacts.shape[1] != 3 or contacts.shape[0] < 2:
            return None, None

        if p0 is not None and p1 is not None:
            p0 = np.asarray(p0, dtype=np.float64)
            p1 = np.asarray(p1, dtype=np.float64)
            axis = self._normalize(p1 - p0)
        else:
            axis = self._normalize(contacts[1] - contacts[0])

        if float(np.linalg.norm(axis)) < 1e-9:
            return None, None

        center = contacts.mean(axis=0)
        return axis, center

    def _electrode_center_only(self, elec: dict):
        contacts = np.asarray(elec.get("contacts_lps", []) or [], dtype=np.float64)
        if contacts.ndim != 2 or contacts.shape[1] != 3 or contacts.shape[0] == 0:
            return None
        return contacts.mean(axis=0)

    def _make_plane_basis(self, axis: np.ndarray, angle_deg: float):
        axis = self._normalize(axis)

        # Start from a "vertical" anatomical direction
        vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # If electrode is almost parallel to Z, use Y instead
        if abs(float(np.dot(axis, vertical))) > 0.95:
            vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # In-plane lateral direction
        w0 = self._normalize(np.cross(axis, vertical))
        if float(np.linalg.norm(w0)) < 1e-9:
            vertical = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            w0 = self._normalize(np.cross(axis, vertical))

        angle_rad = np.deg2rad(float(angle_deg))
        w = self._rodrigues_rotate(w0, axis, angle_rad)
        w = self._normalize(w)

        # Electrode axis is the long direction of the slice
        u = axis
        return u, w

    # ---------- Slice extraction ----------
    def _extract_electrode_plane_slice(
        self, elec: dict, angle_deg: float, image_label: QLabel | None = None
    ):
        axis, center = self._electrode_axis_and_center(elec)

        if axis is None or center is None:
            return None

        u, w = self._make_plane_basis(axis, angle_deg)

        # Use T1 as reference for FOV
        ref_img = self._get_t1_image()
        if ref_img is None:
            return None

        # Match sampling grid to widget size
        if image_label is not None:
            H = max(200, int(image_label.height()))
            W = max(200, int(image_label.width()))
        else:
            H = int(self.slice_h_px)
            W = int(self.slice_w_px)

        # Full FOV in this plane
        s_min, s_max, t_min, t_max = self._compute_plane_fov_from_image(ref_img, center, u, w)

        # Adapt lateral FOV to widget aspect ratio
        s_span = float(s_max - s_min)
        target_ratio = float(W) / max(1.0, float(H))
        desired_t_span = s_span * target_ratio

        current_t_span = float(t_max - t_min)
        if desired_t_span > current_t_span:
            extra = 0.5 * (desired_t_span - current_t_span)
            t_min -= extra
            t_max += extra

        s_vals = np.linspace(s_min, s_max, H, dtype=np.float64)
        t_vals = np.linspace(t_min, t_max, W, dtype=np.float64)

        # Build the whole plane in vectorized form
        S, T = np.meshgrid(s_vals, t_vals, indexing="ij")  # (H, W)
        pts = (
            center[None, None, :]
            + S[..., None] * u[None, None, :]
            + T[..., None] * w[None, None, :]
        )  # (H, W, 3) in LPS physical coordinates

        def _sample_image(img, interp_order=1):
            if img is None:
                return None, None

            try:
                vol = sitk.GetArrayFromImage(img).astype(np.float32)  # [z, y, x]
            except Exception:
                return None, None

            try:
                origin = np.asarray(img.GetOrigin(), dtype=np.float64)  # (x, y, z)
                spacing = np.asarray(img.GetSpacing(), dtype=np.float64)  # (x, y, z)
                direction = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
                inv_direction = np.linalg.inv(direction)
            except Exception:
                return None, None

            # Convert physical LPS points -> continuous image indices (x, y, z)
            pts_flat = pts.reshape(-1, 3)  # (N, 3)
            rel = pts_flat - origin[None, :]  # (N, 3)
            idx_xyz = (rel @ inv_direction.T) / spacing[None, :]  # (N, 3)

            x = idx_xyz[:, 0]
            y = idx_xyz[:, 1]
            z = idx_xyz[:, 2]

            size = np.asarray(img.GetSize(), dtype=np.float64)  # (x, y, z)

            inside = (
                (x >= -0.5)
                & (x <= size[0] - 0.5)
                & (y >= -0.5)
                & (y <= size[1] - 0.5)
                & (z >= -0.5)
                & (z <= size[2] - 0.5)
            )

            arr = np.full((H * W,), np.nan, dtype=np.float32)
            valid = np.zeros((H * W,), dtype=np.uint8)

            if np.any(inside):
                coords = np.vstack(
                    [
                        z[inside],  # map_coordinates expects z, y, x
                        y[inside],
                        x[inside],
                    ]
                )

                sampled = map_coordinates(
                    vol,
                    coords,
                    order=int(interp_order),
                    mode="constant",
                    cval=np.nan,
                )

                arr[inside] = sampled.astype(np.float32)
                valid[inside] = np.isfinite(sampled).astype(np.uint8)

            return arr.reshape(H, W), valid.reshape(H, W)

        arr_t1, valid_t1 = _sample_image(self._get_active_mri_image(), interp_order=1)
        arr_ct, valid_ct = _sample_image(self._get_ct_image(), interp_order=1)
        arr_pet, valid_pet = _sample_image(self._get_pet_image(), interp_order=1)
        arr_sis, valid_sis = _sample_image(self._get_siscom_image(), interp_order=1)
        arr_parcel1, valid_parcel1 = _sample_image(self._parcel1_img, interp_order=0)
        arr_parcel2, valid_parcel2 = _sample_image(self._parcel2_img, interp_order=0)

        # Project contacts into the oblique plane
        contacts = np.asarray(elec.get("contacts_lps", []) or [], dtype=np.float64)
        contact_rows = []
        contact_cols = []
        contact_names = []

        if contacts.ndim == 2 and contacts.shape[0] > 0:
            contacts_visible = elec.get("contacts_visible")
            if contacts_visible is None or len(contacts_visible) != contacts.shape[0]:
                contacts_visible = [True] * contacts.shape[0]
                elec["contacts_visible"] = contacts_visible

            contact_labels_visible = elec.get("contact_labels_visible")
            if contact_labels_visible is None or len(contact_labels_visible) != contacts.shape[0]:
                contact_labels_visible = [False] * contacts.shape[0]
                elec["contact_labels_visible"] = contact_labels_visible

            rel = contacts - center[None, :]
            s_proj = rel @ u
            t_proj = rel @ w

            rows = np.round((s_proj - s_min) / max(1e-9, (s_max - s_min)) * (H - 1)).astype(int)
            cols = np.round((t_proj - t_min) / max(1e-9, (t_max - t_min)) * (W - 1)).astype(int)

            elec_name = str(elec.get("name", "E"))

            for ci, (row, col) in enumerate(zip(rows, cols)):
                if not bool(contacts_visible[ci]):
                    continue

                if 0 <= row < H and 0 <= col < W:
                    contact_rows.append(int(row))
                    contact_cols.append(int(col))

                    # same behavior as 3D view: only show label if contact_labels_visible[ci] is True
                    if bool(contact_labels_visible[ci]):
                        contact_names.append(f"{elec_name}{ci + 1}")
                    else:
                        contact_names.append("")

        return (
            arr_t1,
            valid_t1,
            arr_ct,
            valid_ct,
            arr_pet,
            valid_pet,
            arr_sis,
            valid_sis,
            arr_parcel1,
            valid_parcel1,
            arr_parcel2,
            valid_parcel2,
            center,
            u,
            w,
            s_min,
            s_max,
            t_min,
            t_max,
            H,
            W,
        )

    def _ct_slice_to_qpixmap(
        self,
        arr_t1,
        arr_ct,
        arr_pet,
        arr_sis,
        arr_parcel1,
        contact_rows,
        contact_cols,
        contact_names,
        elec_name: str | None,
        t1_opacity_pct: int,
        ct_opacity_pct: int,
        pet_opacity_pct: int,
        sis_opacity_pct: int,
        parcel1_opacity_pct: int,
    ):
        # choose shape from first available image
        base = None
        for arr in (arr_t1, arr_ct, arr_pet, arr_sis, arr_parcel1):
            if arr is not None:
                base = arr
                break

        if base is None:
            return None

        H, W = base.shape
        img = np.zeros((H, W, 3), dtype=np.float32)

        def _norm(arr):
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return None
            lo = float(np.percentile(finite, 1.0))
            hi = float(np.percentile(finite, 99.0))
            if hi <= lo:
                lo, hi = float(np.min(finite)), float(np.max(finite) + 1.0)
            out = (arr - lo) / max(1e-6, (hi - lo))
            return np.clip(out, 0.0, 1.0)

        # MRI/T1 base in grayscale
        if arr_t1 is not None:
            n = _norm(arr_t1)
            if n is not None:
                a = float(np.clip(t1_opacity_pct, 0, 100)) / 100.0
                gray = n[..., None] * 255.0
                img += gray * a

        # CT overlay in grayscale
        if arr_ct is not None:
            n = _norm(arr_ct)
            if n is not None:
                a = float(np.clip(ct_opacity_pct, 0, 100)) / 100.0
                gray = n[..., None] * 255.0
                img = img * (1.0 - 0.5 * a) + gray * a

        # PET overlay in orange
        if arr_pet is not None:
            finite = arr_pet[np.isfinite(arr_pet)]
            finite = finite[finite > 0]

            if finite.size > 0:
                pmin = float(self.spn_pet_min.value()) if self.spn_pet_min is not None else 15.0
                pmax = float(self.spn_pet_max.value()) if self.spn_pet_max is not None else 75.0
                gamma = 1.0
                try:
                    if self.spn_pet_gamma is not None:
                        gamma = float(self.spn_pet_gamma.value())
                    elif self.sld_pet_gamma is not None:
                        gamma = float(self.sld_pet_gamma.value()) / 100.0
                except Exception:
                    gamma = 1.0

                gamma = max(0.1, gamma)

                lo, hi = get_pet_window(finite, pmin, pmax)
                pet_norm = normalize_pet_slice(arr_pet, lo, hi, gamma=gamma)
                pet_rgb = pet_norm_to_colormap(pet_norm, self._pet_colormap_name)

                img = blend_pet_on_rgb(
                    img,
                    pet_rgb,
                    pet_norm,
                    alpha_scale=float(np.clip(pet_opacity_pct, 0, 100)) / 100.0,
                )

        # Parcellation 1 overlay in cyan
        if arr_parcel1 is not None:
            try:
                mask = np.isfinite(arr_parcel1) & (arr_parcel1 > 0)
                if np.any(mask):
                    a = float(np.clip(parcel1_opacity_pct, 0, 100)) / 100.0
                    color = np.zeros((H, W, 3), dtype=np.float32)
                    color[..., 1] = 255.0  # green
                    color[..., 2] = 255.0  # blue
                    img[mask] = img[mask] * (1.0 - a) + color[mask] * a
            except Exception:
                pass

        # SISCOM overlay in red
        if arr_sis is not None:
            n = _norm(arr_sis)
            if n is not None:
                a = float(np.clip(sis_opacity_pct, 0, 100)) / 100.0
                color = np.zeros((H, W, 3), dtype=np.float32)
                color[..., 0] = n * 255.0
                img = img * (1.0 - 0.5 * a) + color * a

        # Replace NaNs and infs before casting
        img = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)

        img = np.clip(img, 0, 255).astype(np.uint8)

        # draw contacts as small round points using the electrode color
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    # ---------- Rendering ----------
    def render_slices_only(self):
        selected = self._get_checked_electrode_names()

        elec1_name = selected[0] if len(selected) > 0 else None
        elec2_name = selected[1] if len(selected) > 1 else None

        angle1 = self.rot1.value() if self.rot1 is not None else 0
        angle2 = self.rot2.value() if self.rot2 is not None else 0

        self._render_slice(self.image1, self.badge1, elec1_name, angle1, slot_index=1)
        self._render_slice(self.image2, self.badge2, elec2_name, angle2, slot_index=2)
        self._update_parcellation_contacts_table()

    def render_slice_slot(self, slot_index: int):
        selected = self._get_checked_electrode_names()

        if int(slot_index) == 1:
            elec_name = selected[0] if len(selected) > 0 else None
            angle = self.rot1.value() if self.rot1 is not None else 0
            self._render_slice(self.image1, self.badge1, elec_name, angle, slot_index=1)

        elif int(slot_index) == 2:
            elec_name = selected[1] if len(selected) > 1 else None
            angle = self.rot2.value() if self.rot2 is not None else 0
            self._render_slice(self.image2, self.badge2, elec_name, angle, slot_index=2)

        self._update_parcellation_contacts_table()

    def render_brain_only(self):
        if not self._is_active_page:
            return

        if self._brain_plotter is None:
            return

        try:
            if self._brain_plotter.interactor is None:
                return
        except Exception:
            return

        try:
            if not self.ui.window().isVisible():
                return
            if not self.frame_brain.isVisible():
                return
            if not self._brain_plotter.isVisible():
                return
        except Exception:
            return

        selected = self._get_checked_electrode_names()

        elec1_name = selected[0] if len(selected) > 0 else None
        elec2_name = selected[1] if len(selected) > 1 else None

        angle1 = self.rot1.value() if self.rot1 is not None else 0
        angle2 = self.rot2.value() if self.rot2 is not None else 0

        brain_key = (
            bool(getattr(self.state, "brainmask_sitk", None) is not None),
            bool(getattr(self.state, "view3d_page", None) is not None),
            bool(
                getattr(getattr(self.state, "view3d_page", None), "_lh_pial_poly", None) is not None
            ),
            bool(
                getattr(getattr(self.state, "view3d_page", None), "_rh_pial_poly", None) is not None
            ),
        )

        if brain_key != self._last_brain_key:
            self._last_brain_key = brain_key
            self._last_brain_kind = None

        self._render_brain_planes(elec1_name, elec2_name, angle1, angle2)

    def render_all(self):
        self.render_slices_only()
        if self._is_active_page:
            self.render_brain_only()

    def refresh_after_electrode_color_change(self) -> None:
        """
        Refresh Oblique Slice after an electrode color change.

        Important:
        - 2D slices need to be redrawn because contact dots use electrode colors.
        - Mini 3D preview planes need to be rebuilt because their actors keep
        the previous color until the plane actor is recreated.
        """
        try:
            self._last_plane1_key = None
            self._last_plane2_key = None
        except Exception:
            pass

        try:
            self._render_cache_1 = None
            self._render_cache_2 = None
        except Exception:
            pass

        self._schedule_refresh(slices=True, brain=True)

    def _render_slice(
        self,
        image_label: QLabel | None,
        badge: QLabel | None,
        elec_name: str | None,
        angle_deg: float,
        slot_index: int,
    ):
        if image_label is None or badge is None:
            return

        if elec_name is None:
            badge.setText("No electrode")
            badge.adjustSize()
            image_label.setText("No slice")
            image_label.setPixmap(QPixmap())
            self._relayout_labels()
            if slot_index == 1:
                self._slice_cache_1 = None
                self._base_cache_1 = None
                self._pet_cache_1 = None
            else:
                self._slice_cache_2 = None
                self._base_cache_2 = None
                self._pet_cache_2 = None
            return

        badge.setText(elec_name)
        badge.adjustSize()

        elec = self._get_electrode_by_name(elec_name)
        if elec is None:
            image_label.setText("Electrode not found")
            image_label.setPixmap(QPixmap())
            self._relayout_labels()
            return

        t1_enabled = bool(self.chk_mri is not None and self.chk_mri.isChecked())
        ct_enabled = bool(self.chk_ct is not None and self.chk_ct.isChecked())
        pet_enabled = bool(self.chk_pet is not None and self.chk_pet.isChecked())
        sis_enabled = bool(self.chk_siscom is not None and self.chk_siscom.isChecked())
        parcel1_enabled = bool(self.chk_parcel1 is not None and self.chk_parcel1.isChecked())
        parcel2_enabled = bool(self.chk_parcel2 is not None and self.chk_parcel2.isChecked())

        if not (
            t1_enabled
            or ct_enabled
            or pet_enabled
            or sis_enabled
            or parcel1_enabled
            or parcel2_enabled
        ):
            image_label.setText("No modality checked")
            image_label.setPixmap(QPixmap())
            self._relayout_labels()
            return

        # -----------------------------
        # 1) Geometry / sampling cache
        # -----------------------------
        cache_key = self._make_slice_cache_key(elec_name, angle_deg)

        cache = self._slice_cache_1 if slot_index == 1 else self._slice_cache_2
        if cache is None or cache.get("key") != cache_key:
            extracted = self._extract_electrode_plane_slice(
                elec, angle_deg, image_label=image_label
            )
            if extracted is None:
                image_label.setText("Slice failed")
                image_label.setPixmap(QPixmap())
                self._relayout_labels()
                return

            cache = {
                "key": cache_key,
                "data": extracted,
            }
            if slot_index == 1:
                self._slice_cache_1 = cache
                self._base_cache_1 = None
                self._pet_cache_1 = None
            else:
                self._slice_cache_2 = cache
                self._base_cache_2 = None
                self._pet_cache_2 = None
        else:
            extracted = cache["data"]

        (
            arr_t1,
            valid_t1,
            arr_ct,
            valid_ct,
            arr_pet,
            valid_pet,
            arr_sis,
            valid_sis,
            arr_parcel1,
            valid_parcel1,
            arr_parcel2,
            valid_parcel2,
            center,
            u,
            w,
            s_min,
            s_max,
            t_min,
            t_max,
            H,
            W,
        ) = extracted

        if not t1_enabled:
            arr_t1, valid_t1 = None, None
        if not ct_enabled:
            arr_ct, valid_ct = None, None
        if not pet_enabled:
            arr_pet, valid_pet = None, None
        if not sis_enabled:
            arr_sis, valid_sis = None, None
        if not parcel1_enabled:
            arr_parcel1, valid_parcel1 = None, None
        if not parcel2_enabled:
            arr_parcel2, valid_parcel2 = None, None

        contact_rows, contact_cols, contact_names = self._project_contacts_for_cached_slice(
            elec,
            center,
            u,
            w,
            s_min,
            s_max,
            t_min,
            t_max,
            H,
            W,
        )

        gif_contacts = {
            "rows": list(contact_rows),
            "cols": list(contact_cols),
            "names": list(contact_names),
            "elec_name": elec_name,
        }

        if slot_index == 1:
            self._last_gif_contacts_1 = gif_contacts
        else:
            self._last_gif_contacts_2 = gif_contacts

        t1_opacity = int(self.spn_mri.value()) if self.spn_mri is not None else 100
        ct_opacity = int(self.spn_ct.value()) if self.spn_ct is not None else 100
        pet_opacity = int(self.spn_pet_opacity.value()) if self.spn_pet_opacity is not None else 100
        sis_opacity = int(self.spn_siscom.value()) if self.spn_siscom is not None else 100
        parcel1_opacity = int(self.spn_parcel1.value()) if self.spn_parcel1 is not None else 50
        parcel2_opacity = int(self.spn_parcel2.value()) if self.spn_parcel2 is not None else 50

        # -----------------------------
        # 2) Base cache: T1 + CT + SIS + Parcell1
        # -----------------------------
        base_key = self._make_base_cache_key(
            elec_name,
            angle_deg,
            t1_enabled,
            ct_enabled,
            sis_enabled,
            parcel1_enabled,
            parcel2_enabled,
            t1_opacity,
            ct_opacity,
            sis_opacity,
            parcel1_opacity,
            parcel2_opacity,
        )

        base_cache = self._base_cache_1 if slot_index == 1 else self._base_cache_2
        if base_cache is None or base_cache.get("key") != base_key:
            base_rgb = self._build_base_rgb(
                arr_t1,
                arr_ct,
                arr_sis,
                arr_parcel1,
                arr_parcel2,
                t1_opacity,
                ct_opacity,
                sis_opacity,
                parcel1_opacity,
                parcel2_opacity,
            )
            base_cache = {
                "key": base_key,
                "rgb": base_rgb,
            }
            if slot_index == 1:
                self._base_cache_1 = base_cache
            else:
                self._base_cache_2 = base_cache
        else:
            base_rgb = base_cache["rgb"]

        # -----------------------------
        # 3) PET cache: recalculated
        #    ONLY if PET params changed
        # -----------------------------
        pet_key = self._make_pet_cache_key(
            elec_name,
            angle_deg,
            pet_enabled,
            pet_opacity,
        )

        pet_cache = self._pet_cache_1 if slot_index == 1 else self._pet_cache_2
        if pet_cache is None or pet_cache.get("key") != pet_key:
            if pet_enabled:
                pet_rgb, pet_norm = self._build_pet_overlay(arr_pet)
            else:
                pet_rgb, pet_norm = None, None

            pet_cache = {
                "key": pet_key,
                "pet_rgb": pet_rgb,
                "pet_norm": pet_norm,
            }
            if slot_index == 1:
                self._pet_cache_1 = pet_cache
            else:
                self._pet_cache_2 = pet_cache
        else:
            pet_rgb = pet_cache["pet_rgb"]
            pet_norm = pet_cache["pet_norm"]

        # -----------------------------
        # 4) Compose final image
        # -----------------------------
        img_rgb = self._compose_rgb_with_pet(base_rgb, pet_rgb, pet_norm, pet_opacity)

        if img_rgb is None:
            pm = None
        else:
            qimg = QImage(
                img_rgb.data,
                img_rgb.shape[1],
                img_rgb.shape[0],
                img_rgb.strides[0],
                QImage.Format_RGB888,
            )
            pm = QPixmap.fromImage(qimg.copy())

        if pm is None:
            image_label.setText("No image")
            image_label.setPixmap(QPixmap())
            self._relayout_labels()
            return

        # -----------------------------
        # 5) Display-only operations:
        #    zoom / pan / labels / PET scale
        #    -> no PET recomputation
        # -----------------------------
        if slot_index == 1:
            self._last_pm1 = pm
            zoom = self._zoom1
            pan = self._pan1
        else:
            self._last_pm2 = pm
            zoom = self._zoom2
            pan = self._pan2

        target_size = image_label.size()

        if target_size.width() > 10 and target_size.height() > 10:
            pm_display = self._render_oblique_slice_display_canvas(
                pm_source=pm,
                target_size=target_size,
                slot_index=slot_index,
                zoom=zoom,
                pan_xy=pan,
                contact_rows=contact_rows,
                contact_cols=contact_cols,
                contact_names=contact_names,
                elec_name=elec_name,
            )
        else:
            pm_display = pm

        # Color scale is drawn AFTER the display rotation,
        # so it always stays fixed in the final canvas.
        pm_display = self._overlay_scalar_bars_on_pixmap(pm_display)

        image_label.setText("")
        image_label.setPixmap(pm_display)
        self._relayout_labels()

    def _reset_oblique_slice_display_to_default(
        self,
        slot_index: int,
        refresh: bool = True,
        reset_background: bool = True,
    ) -> None:
        """
        Reset only the display state of an oblique slice.

        This does NOT change the anatomical oblique plane scrollbar.
        It only resets:
        - zoom
        - pan
        - visual 2D rotation
        - optional background removal
        """
        slot = int(slot_index)

        if slot == 1:
            self._zoom1 = 1.0
            self._pan1 = [0.0, 0.0]
            self._display_rotation1 = 0.0

            if reset_background:
                self._slice_background_removed1 = False
                try:
                    if self.image1 is not None:
                        self.image1.setStyleSheet("background-color: black;")
                except Exception:
                    pass
                try:
                    if getattr(self, "_slice_tools_1", None) is not None:
                        self._slice_tools_1.set_background_removed_checked(False)
                except Exception:
                    pass

        elif slot == 2:
            self._zoom2 = 1.0
            self._pan2 = [0.0, 0.0]
            self._display_rotation2 = 0.0

            if reset_background:
                self._slice_background_removed2 = False
                try:
                    if self.image2 is not None:
                        self.image2.setStyleSheet("background-color: black;")
                except Exception:
                    pass
                try:
                    if getattr(self, "_slice_tools_2", None) is not None:
                        self._slice_tools_2.set_background_removed_checked(False)
                except Exception:
                    pass

        if refresh:
            self.render_slice_slot(slot)

    def _get_oblique_slice_display_pixmap_for_export(
        self, slot_index: int, remove_background: bool | None = None
    ):
        label = self.image1 if int(slot_index) == 1 else self.image2
        if label is None:
            return None

        try:
            pm = label.pixmap()
        except Exception:
            pm = None

        if pm is None or pm.isNull():
            return None

        out = QPixmap(pm)

        if remove_background is None:
            remove_background = self._get_oblique_slice_background_removed(slot_index)

        if remove_background:
            out = self._remove_black_background_from_pixmap(out)

        return out

    def _get_oblique_slice_raw_pixmap(self, slot_index: int):
        return self._last_pm1 if int(slot_index) == 1 else self._last_pm2

    def _get_oblique_slice_rotation(self, slot_index: int) -> float:
        return float(self._display_rotation1 if int(slot_index) == 1 else self._display_rotation2)

    def _set_oblique_slice_rotation(
        self, slot_index: int, angle_deg: float, refresh: bool = True
    ) -> None:
        angle = float(angle_deg) % 360.0

        if int(slot_index) == 1:
            self._display_rotation1 = angle
        else:
            self._display_rotation2 = angle

        if refresh:
            self.render_slice_slot(int(slot_index))

    def _rotate_oblique_slice_display(self, slot_index: int, delta_deg: float) -> None:
        current = self._get_oblique_slice_rotation(slot_index)
        self._set_oblique_slice_rotation(slot_index, current + float(delta_deg), refresh=True)

    def _get_oblique_slice_background_removed(self, slot_index: int) -> bool:
        return bool(
            self._slice_background_removed1
            if int(slot_index) == 1
            else self._slice_background_removed2
        )

    def _set_oblique_slice_background_removed(
        self, slot_index: int, checked: bool, refresh: bool = True
    ) -> None:
        slot = int(slot_index)

        if slot == 1:
            self._slice_background_removed1 = bool(checked)
            label = self.image1
        else:
            self._slice_background_removed2 = bool(checked)
            label = self.image2

        try:
            if label is not None:
                label.setStyleSheet(
                    "background-color: transparent;" if checked else "background-color: black;"
                )
        except Exception:
            pass

        if refresh:
            self.render_slice_slot(slot)

    def _remove_black_background_from_pixmap(self, pixmap: QPixmap, threshold: int = 8) -> QPixmap:
        if pixmap is None or pixmap.isNull():
            return pixmap

        try:
            img = pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
            w, h = img.width(), img.height()

            for y in range(h):
                for x in range(w):
                    c = QColor(img.pixel(x, y))
                    if c.red() <= threshold and c.green() <= threshold and c.blue() <= threshold:
                        c.setAlpha(0)
                        img.setPixelColor(x, y, c)

            return QPixmap.fromImage(img)

        except Exception:
            return pixmap

    def _fit_source_rect_to_target(
        self, src_w: float, src_h: float, target_w: float, target_h: float
    ):
        """
        Return the QRect-like geometry used to fit a source pixmap into a target canvas
        with Qt.KeepAspectRatio behavior.

        Returns:
            draw_x, draw_y, draw_w, draw_h, scale
        """
        try:
            src_w = max(1.0, float(src_w))
            src_h = max(1.0, float(src_h))
            target_w = max(1.0, float(target_w))
            target_h = max(1.0, float(target_h))

            scale = min(target_w / src_w, target_h / src_h)
            draw_w = src_w * scale
            draw_h = src_h * scale

            draw_x = (target_w - draw_w) / 2.0
            draw_y = (target_h - draw_h) / 2.0

            return draw_x, draw_y, draw_w, draw_h, scale

        except Exception:
            return 0.0, 0.0, float(target_w), float(target_h), 1.0

    def _crop_source_pixmap_for_zoom(
        self,
        pm_source: QPixmap,
        zoom: float,
        pan_xy=None,
    ):
        """
        Crop the source pixmap according to zoom and pan.

        This returns:
            cropped_pixmap, x0, y0, crop_w, crop_h

        x0/y0/crop_w/crop_h are source-space values and are needed to project
        contacts correctly into the cropped image.
        """
        if pm_source is None or pm_source.isNull():
            return pm_source, 0, 0, 1, 1

        try:
            zoom = max(1.0, float(zoom))

            src_w = int(pm_source.width())
            src_h = int(pm_source.height())

            crop_w = max(1, int(round(float(src_w) / zoom)))
            crop_h = max(1, int(round(float(src_h) / zoom)))

            cx = float(src_w) / 2.0
            cy = float(src_h) / 2.0

            if pan_xy is not None:
                cx += float(pan_xy[0])
                cy += float(pan_xy[1])

            x0 = int(round(cx - float(crop_w) / 2.0))
            y0 = int(round(cy - float(crop_h) / 2.0))

            x0 = max(0, min(x0, src_w - crop_w))
            y0 = max(0, min(y0, src_h - crop_h))

            cropped = pm_source.copy(x0, y0, crop_w, crop_h)

            return cropped, x0, y0, crop_w, crop_h

        except Exception:
            return pm_source, 0, 0, max(1, pm_source.width()), max(1, pm_source.height())

    def _draw_contacts_on_transformed_canvas(
        self,
        painter: QPainter,
        contact_rows,
        contact_cols,
        contact_names,
        elec_name: str | None,
        x0: int,
        y0: int,
        crop_w: int,
        crop_h: int,
        draw_x: float,
        draw_y: float,
        draw_w: float,
        draw_h: float,
        zoom: float,
    ) -> None:
        """
        Draw contacts in the same painter coordinate system as the slice.

        Important:
        This function is called while the painter is already rotated.
        Therefore, contacts rotate with the slice, but the final QLabel canvas does not.
        """
        try:
            contact_color = self._get_electrode_rgb(elec_name)

            radius = 4
            if self.spn_contact_size is not None:
                try:
                    radius = max(1, int(self.spn_contact_size.value()))
                except Exception:
                    pass

            scale_x = float(draw_w) / max(1.0, float(crop_w))
            scale_y = float(draw_h) / max(1.0, float(crop_h))

            rr = max(2, int(radius * min(scale_x, scale_y)))

            pen = QPen(QColor(int(contact_color[0]), int(contact_color[1]), int(contact_color[2])))
            painter.setPen(pen)
            painter.setBrush(
                QColor(int(contact_color[0]), int(contact_color[1]), int(contact_color[2]))
            )

            font = QFont()
            font.setPointSize(max(8, int(8 + 2 * float(zoom))))
            painter.setFont(font)

            for r, c, txt in zip(contact_rows, contact_cols, contact_names):
                dx = float(draw_x) + (float(c) - float(x0)) * scale_x
                dy = float(draw_y) + (float(r) - float(y0)) * scale_y

                if dx < draw_x or dx >= draw_x + draw_w or dy < draw_y or dy >= draw_y + draw_h:
                    continue

                painter.drawEllipse(QPoint(int(round(dx)), int(round(dy))), rr, rr)

                if txt:
                    painter.drawText(
                        int(round(dx + rr + 4)),
                        int(round(dy - rr - 2)),
                        str(txt),
                    )

        except Exception:
            pass

    def _render_oblique_slice_display_canvas(
        self,
        pm_source: QPixmap,
        target_size,
        slot_index: int,
        zoom: float,
        pan_xy,
        contact_rows,
        contact_cols,
        contact_names,
        elec_name: str | None,
    ) -> QPixmap:
        """
        Render the oblique slice into the final QLabel canvas.

        Key behavior:
        - the final canvas always has the full QLabel size;
        - only the anatomical slice content rotates;
        - the canvas itself never rotates;
        - contacts rotate with the slice;
        - color scales are NOT drawn here, so they can be added afterward
        and stay fixed.
        """
        if pm_source is None or pm_source.isNull():
            return pm_source

        try:
            target_w = max(2, int(target_size.width()))
            target_h = max(2, int(target_size.height()))

            remove_background = bool(self._get_oblique_slice_background_removed(slot_index))
            angle = float(self._get_oblique_slice_rotation(slot_index)) % 360.0
            zoom = max(1.0, float(zoom))

            # -----------------------------------------------------
            # Left/right display convention.
            #
            # Flip the anatomical source before zoom/pan and before
            # drawing the contacts. Contact columns are mirrored too,
            # while text labels and PET/SISCOM scalar bars remain readable.
            # -----------------------------------------------------
            source_for_display = QPixmap(pm_source)
            display_contact_cols = list(contact_cols)

            if bool(getattr(self, "_flip_lr_oblique", True)):
                flip_transform = QTransform()
                flip_transform.scale(-1.0, 1.0)

                source_for_display = source_for_display.transformed(
                    flip_transform,
                    Qt.SmoothTransformation,
                )

                source_width = int(source_for_display.width())
                display_contact_cols = [source_width - 1 - int(col) for col in contact_cols]

            # 1) Crop source for zoom/pan BEFORE visual rotation.
            cropped, x0, y0, crop_w, crop_h = self._crop_source_pixmap_for_zoom(
                source_for_display,
                zoom=zoom,
                pan_xy=pan_xy,
            )

            if remove_background:
                cropped = self._remove_black_background_from_pixmap(cropped)

            # 2) Final fixed canvas = full QLabel.
            canvas = QPixmap(target_w, target_h)
            canvas.fill(Qt.transparent if remove_background else QColor(0, 0, 0))

            # 3) Fit the cropped source into the final canvas.
            draw_x, draw_y, draw_w, draw_h, _scale = self._fit_source_rect_to_target(
                cropped.width(),
                cropped.height(),
                target_w,
                target_h,
            )

            # Make the displayed slice bigger when it is visually rotated,
            # so the diagonal view fills more of the canvas.
            if abs(angle) > 1e-6:
                rotation_cover_scale = 1.35  # increase to 1.45 or 1.55 if needed

                center_x = draw_x + draw_w / 2.0
                center_y = draw_y + draw_h / 2.0

                draw_w *= rotation_cover_scale
                draw_h *= rotation_cover_scale

                draw_x = center_x - draw_w / 2.0
                draw_y = center_y - draw_h / 2.0

            painter = QPainter(canvas)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.TextAntialiasing, True)

            # 4) Rotate ONLY the drawing coordinate system around the canvas centre.
            # The canvas remains the full rectangular QLabel.
            cx = float(target_w) / 2.0
            cy = float(target_h) / 2.0

            painter.translate(cx, cy)
            painter.rotate(angle)
            painter.translate(-cx, -cy)

            # 5) Draw the slice in the rotated coordinate system.
            painter.drawPixmap(
                int(round(draw_x)),
                int(round(draw_y)),
                int(round(draw_w)),
                int(round(draw_h)),
                cropped,
            )

            # 6) Draw contacts in the same rotated coordinate system.
            self._draw_contacts_on_transformed_canvas(
                painter=painter,
                contact_rows=contact_rows,
                contact_cols=display_contact_cols,
                contact_names=contact_names,
                elec_name=elec_name,
                x0=x0,
                y0=y0,
                crop_w=crop_w,
                crop_h=crop_h,
                draw_x=draw_x,
                draw_y=draw_y,
                draw_w=draw_w,
                draw_h=draw_h,
                zoom=zoom,
            )

            painter.end()
            return canvas

        except Exception:
            return pm_source

    def _apply_display_rotation_and_background(
        self,
        pixmap: QPixmap,
        slot_index: int,
        remove_background: bool = False,
    ) -> QPixmap:
        """
        Legacy helper.

        The main oblique slice display now uses
        _render_oblique_slice_display_canvas(), which keeps the QLabel canvas fixed
        and rotates only the anatomical content.

        Keep this function simple for exports or fallback calls.
        """
        if pixmap is None or pixmap.isNull():
            return pixmap

        try:
            out = QPixmap(pixmap)

            if remove_background:
                out = self._remove_black_background_from_pixmap(out)

            angle = float(self._get_oblique_slice_rotation(slot_index)) % 360.0

            if abs(angle) <= 1e-6:
                return out

            transform = QTransform()
            transform.rotate(angle)

            return out.transformed(transform, Qt.SmoothTransformation)

        except Exception:
            return pixmap

    def _save_oblique_slice_screenshot(
        self,
        slot_index: int,
        filename: str,
        remove_background: bool = False,
    ) -> None:
        try:
            filename = str(filename)

            if remove_background and not filename.lower().endswith(".png"):
                filename = filename.rsplit(".", 1)[0] + ".png"

            pm = self._get_oblique_slice_display_pixmap_for_export(
                slot_index,
                remove_background=remove_background,
            )

            if pm is None or pm.isNull():
                return

            ok = pm.save(filename)

            if not ok:
                NeuXelecMessageDialog.warning(
                    self._dialog_parent(),
                    "Screenshot",
                    "Could not save the oblique slice screenshot.",
                )

        except Exception as e:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(),
                "Screenshot",
                f"Failed to save screenshot:\n{e}",
            )

    def _qimage_to_rgba_array(self, qimg: QImage):
        img = qimg.convertToFormat(QImage.Format_RGBA8888)
        width = img.width()
        height = img.height()

        ptr = img.bits()

        try:
            ptr.setsize(height * width * 4)
        except Exception:
            pass

        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4)).copy()
        return arr

    def _get_oblique_slice_gif_contacts(self, slot_index: int):
        if int(slot_index) == 1:
            return getattr(self, "_last_gif_contacts_1", None)
        return getattr(self, "_last_gif_contacts_2", None)

    def _build_oblique_slice_source_for_plane_angle(
        self,
        slot_index: int,
        elec_name: str,
        plane_angle_deg: float,
    ):
        """
        Build the anatomical oblique slice source pixmap for a given electrode-plane angle.

        This recomputes the oblique plane around the electrode axis and composes
        the currently enabled modalities (T1 / CT / PET / SISCOM / parcellations).
        It also projects the currently visible contacts and labels into that plane.
        """
        try:
            image_label = self.image1 if int(slot_index) == 1 else self.image2
            elec = self._get_electrode_by_name(elec_name)
            if elec is None:
                return None, None

            extracted = self._extract_electrode_plane_slice(
                elec,
                float(plane_angle_deg),
                image_label=image_label,
            )
            if extracted is None:
                return None, None

            (
                arr_t1,
                valid_t1,
                arr_ct,
                valid_ct,
                arr_pet,
                valid_pet,
                arr_sis,
                valid_sis,
                arr_parcel1,
                valid_parcel1,
                arr_parcel2,
                valid_parcel2,
                center,
                u,
                w,
                s_min,
                s_max,
                t_min,
                t_max,
                H,
                W,
            ) = extracted

            # Current modality checkboxes
            t1_enabled = bool(self.chk_mri is not None and self.chk_mri.isChecked())
            ct_enabled = bool(self.chk_ct is not None and self.chk_ct.isChecked())
            pet_enabled = bool(self.chk_pet is not None and self.chk_pet.isChecked())
            sis_enabled = bool(self.chk_siscom is not None and self.chk_siscom.isChecked())
            parcel1_enabled = bool(self.chk_parcel1 is not None and self.chk_parcel1.isChecked())
            parcel2_enabled = bool(self.chk_parcel2 is not None and self.chk_parcel2.isChecked())

            if not t1_enabled:
                arr_t1, valid_t1 = None, None
            if not ct_enabled:
                arr_ct, valid_ct = None, None
            if not pet_enabled:
                arr_pet, valid_pet = None, None
            if not sis_enabled:
                arr_sis, valid_sis = None, None
            if not parcel1_enabled:
                arr_parcel1, valid_parcel1 = None, None
            if not parcel2_enabled:
                arr_parcel2, valid_parcel2 = None, None

            # Contacts / labels currently visible in this oblique view
            contact_rows, contact_cols, contact_names = self._project_contacts_for_cached_slice(
                elec,
                center,
                u,
                w,
                s_min,
                s_max,
                t_min,
                t_max,
                H,
                W,
            )

            gif_contacts = {
                "rows": list(contact_rows),
                "cols": list(contact_cols),
                "names": list(contact_names),
                "elec_name": str(elec_name),
            }

            # Current opacities
            t1_opacity = int(self.spn_mri.value()) if self.spn_mri is not None else 100
            ct_opacity = int(self.spn_ct.value()) if self.spn_ct is not None else 100
            pet_opacity = (
                int(self.spn_pet_opacity.value()) if self.spn_pet_opacity is not None else 100
            )
            sis_opacity = int(self.spn_siscom.value()) if self.spn_siscom is not None else 100
            parcel1_opacity = int(self.spn_parcel1.value()) if self.spn_parcel1 is not None else 50
            parcel2_opacity = int(self.spn_parcel2.value()) if self.spn_parcel2 is not None else 50

            # Compose anatomical base
            base_rgb = self._build_base_rgb(
                arr_t1,
                arr_ct,
                arr_sis,
                arr_parcel1,
                arr_parcel2,
                t1_opacity,
                ct_opacity,
                sis_opacity,
                parcel1_opacity,
                parcel2_opacity,
            )

            # PET overlay
            if pet_enabled:
                pet_rgb, pet_norm = self._build_pet_overlay(arr_pet)
            else:
                pet_rgb, pet_norm = None, None

            img_rgb = self._compose_rgb_with_pet(
                base_rgb,
                pet_rgb,
                pet_norm,
                pet_opacity,
            )

            if img_rgb is None:
                return None, None

            qimg = QImage(
                img_rgb.data,
                img_rgb.shape[1],
                img_rgb.shape[0],
                img_rgb.strides[0],
                QImage.Format_RGB888,
            )

            pm_source = QPixmap.fromImage(qimg.copy())
            return pm_source, gif_contacts

        except Exception:
            return None, None

    def _export_oblique_slice_gif(
        self,
        slot_index: int,
        filename: str,
        remove_background: bool = False,
    ) -> None:
        """
        Start a non-blocking oblique-slice GIF export.

        One QPixmap frame is produced per Qt event-loop iteration so the
        PageLoadingOverlay remains animated and blocks additional user input.
        """
        if bool(getattr(self, "_gif_export_active", False)):
            NeuXelecMessageDialog.information(
                self._dialog_parent(),
                "GIF export",
                "A GIF export is already in progress.",
            )
            return

        try:
            filename = str(filename)

            if not filename.lower().endswith(".gif"):
                filename += ".gif"

            slot = int(slot_index)

            selected = self._get_checked_electrode_names()

            if slot == 1:
                elec_name = selected[0] if len(selected) >= 1 else None
                image_label = self.image1
                start_plane_angle = float(self.rot1.value()) if self.rot1 is not None else 0.0
                zoom = float(self._zoom1)
                pan = list(self._pan1)

            else:
                elec_name = selected[1] if len(selected) >= 2 else None
                image_label = self.image2
                start_plane_angle = float(self.rot2.value()) if self.rot2 is not None else 0.0
                zoom = float(self._zoom2)
                pan = list(self._pan2)

            if not elec_name:
                NeuXelecMessageDialog.warning(
                    self._dialog_parent(),
                    "GIF export",
                    ("No electrode is currently assigned " f"to oblique slice {slot}."),
                )
                return

            target_size = None

            if image_label is not None and image_label.width() > 10 and image_label.height() > 10:
                target_size = image_label.size()

            self._gif_export_active = True

            self._gif_export_state = {
                "slot": slot,
                "filename": filename,
                "remove_background": bool(remove_background),
                "elec_name": elec_name,
                "start_plane_angle": start_plane_angle,
                "zoom": zoom,
                "pan": pan,
                "target_size": target_size,
                "n_frames": 72,
                "frame_index": 0,
                "frames": [],
            }

            if self._loading_overlay is not None:
                self._loading_overlay.begin(f"Preparing GIF for oblique slice {slot}")
                self._loading_overlay.set_progress(
                    0.06,
                    "Preparing animation frames",
                )

            QTimer.singleShot(
                0,
                self._export_next_oblique_gif_frame,
            )

        except Exception as e:
            self._fail_oblique_gif_export(str(e))

    def _export_next_oblique_gif_frame(self) -> None:
        """
        Generate one oblique-slice GIF frame and then return control to Qt.
        """
        state = getattr(self, "_gif_export_state", None)

        if not self._gif_export_active or not isinstance(state, dict):
            return

        try:
            frame_index = int(state["frame_index"])
            n_frames = int(state["n_frames"])
            slot = int(state["slot"])

            plane_angle = float(state["start_plane_angle"]) + (
                360.0 * float(frame_index) / float(n_frames)
            )

            pm_source, gif_contacts = self._build_oblique_slice_source_for_plane_angle(
                slot_index=slot,
                elec_name=state["elec_name"],
                plane_angle_deg=plane_angle,
            )

            if pm_source is not None and not pm_source.isNull():
                target_size = state.get("target_size")

                final_target_size = pm_source.size() if target_size is None else target_size

                pm_display = self._render_oblique_slice_display_canvas(
                    pm_source=pm_source,
                    target_size=final_target_size,
                    slot_index=slot,
                    zoom=float(state["zoom"]),
                    pan_xy=state["pan"],
                    contact_rows=gif_contacts.get("rows", []),
                    contact_cols=gif_contacts.get("cols", []),
                    contact_names=gif_contacts.get("names", []),
                    elec_name=gif_contacts.get(
                        "elec_name",
                        None,
                    ),
                )

                if pm_display is not None and not pm_display.isNull():
                    if bool(state["remove_background"]):
                        pm_display = self._remove_black_background_from_pixmap(pm_display)

                    pm_display = self._overlay_scalar_bars_on_pixmap(pm_display)

                    frame_array = self._qimage_to_rgba_array(pm_display.toImage())

                    state["frames"].append(frame_array)

            frame_index += 1
            state["frame_index"] = frame_index

            progress = 0.08 + (0.76 * float(frame_index) / float(n_frames))

            if self._loading_overlay is not None:
                self._loading_overlay.set_progress(
                    min(0.84, progress),
                    (f"Generating frame " f"{frame_index} of {n_frames}"),
                )

            if frame_index >= n_frames:
                QTimer.singleShot(
                    0,
                    self._encode_oblique_gif,
                )
                return

            self._gif_export_timer.start(0)

        except Exception as e:
            self._fail_oblique_gif_export(str(e))

    def _encode_oblique_gif(self) -> None:
        """
        Encode all prepared RGBA frames into the final GIF.
        """
        state = getattr(self, "_gif_export_state", None)

        if not self._gif_export_active or not isinstance(state, dict):
            return

        try:
            frames = state.get("frames", [])

            if not frames:
                raise RuntimeError("No valid image could be generated for the GIF.")

            if self._loading_overlay is not None:
                self._loading_overlay.set_progress(
                    0.88,
                    "Encoding GIF animation",
                )

            filename = str(state["filename"])

            try:
                import imageio.v2 as imageio

                imageio.mimsave(
                    filename,
                    frames,
                    duration=0.05,
                    loop=0,
                )

            except Exception:
                from PIL import Image

                pil_frames = [Image.fromarray(frame, mode="RGBA") for frame in frames]

                pil_frames[0].save(
                    filename,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=50,
                    loop=0,
                    disposal=2,
                )

            self._finish_oblique_gif_export()

        except Exception as e:
            self._fail_oblique_gif_export(str(e))

    def _finish_oblique_gif_export(self) -> None:
        """
        Complete a successful oblique-slice GIF export.
        """
        state = getattr(self, "_gif_export_state", None)

        if not isinstance(state, dict):
            return

        filename = str(state.get("filename", ""))
        slot = int(state.get("slot", 1))

        if self._loading_overlay is not None:
            self._loading_overlay.set_progress(
                0.94,
                "GIF export completed",
            )
            self._loading_overlay.complete()

        self._gif_export_active = False
        self._gif_export_state = None

        NeuXelecMessageDialog.information(
            self._dialog_parent(),
            "GIF export completed",
            (
                f"The animation for oblique slice {slot} "
                "was exported successfully.\n\n"
                f"{filename}"
            ),
        )

    def _fail_oblique_gif_export(
        self,
        error_message: str,
    ) -> None:
        """
        Cancel the overlay and reset the GIF state after an error.
        """
        try:
            self._gif_export_timer.stop()
        except Exception:
            pass

        if self._loading_overlay is not None:
            self._loading_overlay.cancel()

        self._gif_export_active = False
        self._gif_export_state = None

        NeuXelecMessageDialog.warning(
            self._dialog_parent(),
            "GIF export failed",
            ("The oblique-slice GIF could not be exported.\n\n" f"Details:\n{error_message}"),
        )

    def _current_oblique_slice_view_dict(self, slot_index: int) -> dict:
        slot = int(slot_index)

        if slot == 1:
            zoom = self._zoom1
            pan = self._pan1
        else:
            zoom = self._zoom2
            pan = self._pan2

        return {
            "zoom": float(zoom),
            "pan": [float(pan[0]), float(pan[1])],
            "display_rotation_deg": float(self._get_oblique_slice_rotation(slot)),
            "background_removed": bool(self._get_oblique_slice_background_removed(slot)),
        }

    def _apply_oblique_slice_view_dict(
        self, slot_index: int, view: dict, refresh: bool = True
    ) -> bool:
        if not isinstance(view, dict):
            return False

        slot = int(slot_index)

        try:
            zoom = max(1.0, min(8.0, float(view.get("zoom", 1.0))))

            pan = view.get("pan", [0.0, 0.0])
            if not isinstance(pan, (list, tuple)) or len(pan) != 2:
                pan = [0.0, 0.0]

            pan = [float(pan[0]), float(pan[1])]
            rotation = float(view.get("display_rotation_deg", 0.0)) % 360.0
            background = bool(view.get("background_removed", False))

            if slot == 1:
                self._zoom1 = zoom
                self._pan1 = pan
                self._display_rotation1 = rotation
                self._slice_background_removed1 = background
                tools = getattr(self, "_slice_tools_1", None)

            else:
                self._zoom2 = zoom
                self._pan2 = pan
                self._display_rotation2 = rotation
                self._slice_background_removed2 = background
                tools = getattr(self, "_slice_tools_2", None)

            try:
                if tools is not None:
                    tools.set_background_removed_checked(background)
            except Exception:
                pass

            try:
                label = self.image1 if slot == 1 else self.image2
                if label is not None:
                    label.setStyleSheet(
                        "background-color: transparent;"
                        if background
                        else "background-color: black;"
                    )
            except Exception:
                pass

            if refresh:
                self.render_slice_slot(slot)

            return True

        except Exception:
            return False

    def _save_current_oblique_slice_view_to_state(self, slot_index: int) -> None:
        slot = int(slot_index)

        try:
            saved = getattr(self.state, "oblique_slice_saved_views", None)

            if not isinstance(saved, dict):
                saved = {}

            saved[str(slot)] = self._current_oblique_slice_view_dict(slot)
            self.state.oblique_slice_saved_views = saved

            project_path = getattr(self.state, "project_path", None)

            if project_path:
                from neuxelec.project_io import save_project_json

                save_project_json(self.state, project_path)

        except Exception as e:
            print("[ObliqueSlice] Could not save slice view to project JSON:", e)

    def _apply_saved_oblique_slice_view_from_state(self, slot_index: int) -> None:
        try:
            saved = getattr(self.state, "oblique_slice_saved_views", {}) or {}
            view = saved.get(str(int(slot_index))) or saved.get(int(slot_index))
            self._apply_oblique_slice_view_dict(int(slot_index), view, refresh=True)

        except Exception:
            pass

    def _apply_oblique_slice_initial_saved_views(self) -> None:
        try:
            saved = getattr(self.state, "oblique_slice_saved_views", {}) or {}

            for slot in (1, 2):
                view = saved.get(str(slot)) or saved.get(slot)

                if isinstance(view, dict):
                    self._apply_oblique_slice_view_dict(slot, view, refresh=False)

        except Exception:
            pass

    def _slice_local_angle_from_global_pos(
        self, slot_index: int, global_pos: QPoint
    ) -> float | None:
        try:
            label = self.image1 if int(slot_index) == 1 else self.image2

            if label is None:
                return None

            pos = label.mapFromGlobal(global_pos)

            cx = float(label.width()) / 2.0
            cy = float(label.height()) / 2.0

            dx = float(pos.x()) - cx
            dy = float(pos.y()) - cy

            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                return None

            return float(np.degrees(np.arctan2(dy, dx)))

        except Exception:
            return None

    def _get_ct_image(self):
        if not bool(getattr(self.state, "ct_validated", False)):
            return None
        return getattr(self.state, "ct_coreg_in_t1", None)

    def _get_t1_image(self):
        return getattr(self.state, "t1_sitk", None)

    def _get_pet_image(self):
        if not bool(getattr(self.state, "pet_validated", False)):
            return None
        return getattr(self.state, "pet_coreg_in_t1", None)

    def _get_siscom_image(self):
        if not bool(getattr(self.state, "siscom_validated", False)):
            return None

        img = getattr(self.state, "siscom_thr_in_t1", None)
        if img is not None:
            return img

        return getattr(self.state, "siscom_z_in_t1", None)

    def _make_slice_cache_key(self, elec_name: str | None, angle_deg: float):
        return (
            elec_name,
            float(angle_deg),
            str(getattr(self, "_oblique_mri_source", "T1")),
            bool(self._get_t1_image() is not None),
            bool(self._get_t2_image() is not None),
            bool(self._get_ct_image() is not None),
            bool(self._get_pet_image() is not None),
            bool(self._get_siscom_image() is not None),
        )

    def _apply_zoom_to_pixmap(self, pm: QPixmap, target_size, zoom: float, pan_xy=None) -> QPixmap:
        if pm is None or pm.isNull() or target_size.width() <= 1 or target_size.height() <= 1:
            return pm

        zoom = max(1.0, float(zoom))

        src_w = pm.width()
        src_h = pm.height()

        # zoom=1 -> whole image
        crop_w = max(1, int(round(src_w / zoom)))
        crop_h = max(1, int(round(src_h / zoom)))

        # default center = middle of image
        cx = src_w / 2.0
        cy = src_h / 2.0

        if pan_xy is not None:
            cx += float(pan_xy[0])
            cy += float(pan_xy[1])

        x0 = int(round(cx - crop_w / 2.0))
        y0 = int(round(cy - crop_h / 2.0))

        x0 = max(0, min(x0, src_w - crop_w))
        y0 = max(0, min(y0, src_h - crop_h))

        cropped = pm.copy(x0, y0, crop_w, crop_h)

        return cropped.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

    def _get_slice_geometry_for_label(self, image_label: QLabel | None):
        # fallback
        H = int(self.slice_h_px)
        W = int(self.slice_w_px)
        length_mm = float(self.slice_length_mm)

        if image_label is not None:
            tw = max(50, image_label.width())
            th = max(50, image_label.height())

            # use frame ratio
            W = int(tw)
            H = int(th)

            # keep the long physical dimension fixed and adapt width to frame ratio
            ratio = float(W) / max(1.0, float(H))
            width_mm = length_mm * ratio
        else:
            width_mm = float(self.slice_width_mm)

        return H, W, length_mm, width_mm

    def _compute_plane_fov_from_image(
        self, img: sitk.Image, center: np.ndarray, u: np.ndarray, w: np.ndarray
    ):
        """
        Compute full field-of-view in the oblique plane by projecting the 8 corners
        of the reference image onto the plane axes (u, w), relative to center.
        Returns: s_min, s_max, t_min, t_max
        """
        size = img.GetSize()  # x, y, z

        corners_idx = [
            (0, 0, 0),
            (size[0] - 1, 0, 0),
            (0, size[1] - 1, 0),
            (0, 0, size[2] - 1),
            (size[0] - 1, size[1] - 1, 0),
            (size[0] - 1, 0, size[2] - 1),
            (0, size[1] - 1, size[2] - 1),
            (size[0] - 1, size[1] - 1, size[2] - 1),
        ]

        s_vals = []
        t_vals = []

        for idx in corners_idx:
            try:
                p = np.asarray(
                    img.TransformIndexToPhysicalPoint(tuple(int(v) for v in idx)), dtype=np.float64
                )
                rel = p - center
                s_vals.append(float(np.dot(rel, u)))
                t_vals.append(float(np.dot(rel, w)))
            except Exception:
                continue

        if len(s_vals) == 0 or len(t_vals) == 0:
            # fallback
            return -100.0, 100.0, -100.0, 100.0

        s_min = min(s_vals)
        s_max = max(s_vals)
        t_min = min(t_vals)
        t_max = max(t_vals)

        # small margin
        margin = 5.0
        return s_min - margin, s_max + margin, t_min - margin, t_max + margin

    def _get_electrode_rgb(self, elec_name: str | None):
        if not elec_name:
            return np.array([255, 255, 0], dtype=np.uint8)  # fallback yellow

        elec = self._get_electrode_by_name(elec_name)
        if not isinstance(elec, dict):
            return np.array([255, 255, 0], dtype=np.uint8)

        color = elec.get("color", None)
        if color is None:
            return np.array([255, 255, 0], dtype=np.uint8)

        try:
            # case 1: QColor-like
            if hasattr(color, "red") and hasattr(color, "green") and hasattr(color, "blue"):
                return np.array(
                    [int(color.red()), int(color.green()), int(color.blue())], dtype=np.uint8
                )

            # case 2: tuple/list rgb
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                vals = [float(color[0]), float(color[1]), float(color[2])]

                # normalize if stored in 0..1
                if max(vals) <= 1.0:
                    vals = [v * 255.0 for v in vals]

                return np.array([int(vals[0]), int(vals[1]), int(vals[2])], dtype=np.uint8)
        except Exception:
            pass

        return np.array([255, 255, 0], dtype=np.uint8)

    def _reset_oblique_brain_camera_coronal(self):
        """
        Reset mini 3D preview to a coronal/anterior view.

        Display convention:
        - Meshes are displayed in RAS coordinates.
        - RAS +Y = anterior.
        - Camera placed on +Y looks at the brain from the front.
        - Z is kept upward.
        """
        plotter = getattr(self, "_brain_plotter", None)
        if plotter is None:
            return

        try:
            mesh, _kind = self._get_brain_mesh_for_oblique_view()
        except Exception:
            mesh = None

        try:
            if mesh is not None and getattr(mesh, "n_points", 0) > 0:
                center = np.asarray(mesh.center, dtype=np.float64)
                bounds = mesh.bounds  # xmin, xmax, ymin, ymax, zmin, zmax

                dx = float(bounds[1] - bounds[0])
                dy = float(bounds[3] - bounds[2])
                dz = float(bounds[5] - bounds[4])
                dist = max(dx, dy, dz) * 2.4

                camera_position = (
                    float(center[0]),
                    float(center[1] + dist),
                    float(center[2]),
                )
                focal_point = (
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                )
                view_up = (0.0, 0.0, 1.0)

                plotter.camera_position = [camera_position, focal_point, view_up]
            else:
                # Fallback if no mesh is available yet
                plotter.view_yz()
        except Exception:
            try:
                plotter.view_yz()
            except Exception:
                pass

        try:
            plotter.reset_camera()
        except Exception:
            pass

        try:
            plotter.render()
        except Exception:
            pass

    def _set_brain_axes_widget_colored(self) -> None:
        """
        Make the small X/Y/Z orientation axes match slice-plane colors:
        X = red, Y = green, Z = blue.
        """
        try:
            plotter = getattr(self, "_brain_plotter", None)
            if plotter is None:
                return

            axes_actor = getattr(plotter, "axes_actor", None)

            if axes_actor is None:
                try:
                    axes_actor = getattr(plotter.renderer, "axes_actor", None)
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

                try:
                    prop = caption_actor.GetCaptionTextProperty()
                    prop.SetColor(*color)
                    prop.SetOpacity(1.0)
                    prop.BoldOn()
                    prop.ShadowOff()
                except Exception:
                    pass

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
                plotter.render()
            except Exception:
                pass

        except Exception:
            pass

    def _init_brain_view(self):
        if self.frame_brain is None:
            return

        try:
            # Clean old layout/widgets if something was already inside frame_visuBrain
            old_layout = self.frame_brain.layout()
            if old_layout is not None:
                while old_layout.count():
                    item = old_layout.takeAt(0)
                    w = item.widget()
                    if w is not None:
                        try:
                            w.hide()
                        except Exception:
                            pass
                        try:
                            w.setParent(None)
                        except Exception:
                            pass
                        try:
                            w.deleteLater()
                        except Exception:
                            pass

                layout = old_layout
            else:
                layout = QVBoxLayout()
                self.frame_brain.setLayout(layout)

            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            self._brain_layout = layout

            self.frame_brain.setStyleSheet("""
                QFrame {
                    background-color: transparent;
                    border: none;
                }
            """)

            self._brain_plotter = QtInteractor(self.frame_brain)
            self._brain_layout.addWidget(self._brain_plotter)
            try:
                self.frame_brain.installEventFilter(self)
            except Exception:
                pass

            try:
                self._brain_plotter.installEventFilter(self)
            except Exception:
                pass

            try:
                if (
                    hasattr(self._brain_plotter, "interactor")
                    and self._brain_plotter.interactor is not None
                ):
                    self._brain_plotter.interactor.installEventFilter(self)
            except Exception:
                pass

            self._brain_plotter.set_background("#2b2d31")
            self._brain_plotter.setFocusPolicy(Qt.StrongFocus)
            self._brain_plotter.show_axes()
            try:
                self._quick_tools = PyVistaQuickTools(self.frame_brain, self)
                self._quick_tools.raise_()
                self._update_oblique_toolbars_geometry()
            except Exception as e:
                print("[ObliqueSlice Quick Tools] Failed to create toolbar:", e)
            self._set_brain_axes_widget_colored()
            self._brain_camera_initialized = False

            try:
                import vtk

                style = vtk.vtkInteractorStyleTrackballCamera()
                self._brain_plotter.interactor.SetInteractorStyle(style)
            except Exception:
                try:
                    self._brain_plotter.enable_trackball_style()
                except Exception:
                    pass

            self._brain_plotter.enable()
            self._brain_plotter.show()

        except Exception as e:
            print("[ObliqueSlice] Failed to init brain view:", e)
            self._brain_plotter = None

    def _oblique_plotter(self):
        return getattr(self, "_brain_plotter", None)

    def _save_3d_view_screenshot(
        self,
        filename: str,
        transparent_background: bool = False,
        camera_state=None,
    ) -> None:
        """
        Screenshot of the Oblique Slice mini 3D view only.
        """
        plotter = self._oblique_plotter()
        if plotter is None:
            return

        try:
            filename = str(filename)

            if transparent_background and not filename.lower().endswith(".png"):
                filename = filename.rsplit(".", 1)[0] + ".png"

            old_camera = camera_state if camera_state is not None else self._camera_state_tuple()

            axes_hidden = False
            if transparent_background:
                try:
                    plotter.hide_axes()
                    axes_hidden = True
                except Exception:
                    pass

            try:
                plotter.render()
            except Exception:
                pass

            try:
                plotter.screenshot(
                    filename,
                    transparent_background=bool(transparent_background),
                    return_img=False,
                )
            except TypeError:
                plotter.screenshot(filename, return_img=False)

            if axes_hidden:
                try:
                    plotter.show_axes()
                    if hasattr(self, "_set_brain_axes_widget_colored"):
                        self._set_brain_axes_widget_colored()
                except Exception:
                    pass

            try:
                self._restore_camera_state_tuple(old_camera)
            except Exception:
                pass

            try:
                plotter.render()
            except Exception:
                pass

        except Exception as e:
            print("[ObliqueSlice] Screenshot failed:", e)

    def _capture_camera_state(self):
        return self._camera_state_tuple()

    def _restore_camera_state(self, state):
        return self._restore_camera_state_tuple(state)

    def _camera_state_tuple(self):
        plotter = self._oblique_plotter()
        if plotter is None:
            return None

        try:
            cam = plotter.camera
            return (
                tuple(float(v) for v in cam.position),
                tuple(float(v) for v in cam.focal_point),
                tuple(float(v) for v in cam.up),
                tuple(float(v) for v in cam.clipping_range),
            )
        except Exception:
            return None

    def _restore_camera_state_tuple(self, state_tuple) -> None:
        plotter = self._oblique_plotter()
        if plotter is None or state_tuple is None:
            return

        try:
            pos, focal, up, clip = state_tuple
            cam = plotter.camera
            cam.position = pos
            cam.focal_point = focal
            cam.up = up

            try:
                cam.clipping_range = clip
            except Exception:
                try:
                    plotter.reset_camera_clipping_range()
                except Exception:
                    pass

            plotter.render()
        except Exception:
            pass

    def _get_camera_scene_center_and_distance(self):
        plotter = self._oblique_plotter()

        center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        dist = 400.0

        try:
            actor = getattr(self, "_brain_actor", None)
            if actor is None:
                actor = getattr(self, "_brain_mesh_actor", None)

            if actor is not None:
                bounds = actor.GetBounds()
                if bounds is not None and len(bounds) == 6:
                    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in bounds]
                    center = np.array(
                        [
                            0.5 * (xmin + xmax),
                            0.5 * (ymin + ymax),
                            0.5 * (zmin + zmax),
                        ],
                        dtype=np.float64,
                    )

                    dist = max(xmax - xmin, ymax - ymin, zmax - zmin) * 2.4
                    if not np.isfinite(dist) or dist <= 1:
                        dist = 400.0
        except Exception:
            pass

        return center, float(dist)

    def _set_camera_quick_view(self, view_name: str) -> None:
        """
        Same camera presets as 3D View, applied to the Oblique Slice mini 3D view.
        """
        plotter = self._oblique_plotter()
        if plotter is None:
            return

        try:
            view_name = str(view_name).lower().strip()
            center, dist = self._get_camera_scene_center_and_distance()

            if view_name == "front":
                pos = center + np.array([0.0, dist, 0.0])
                up = (0.0, 0.0, 1.0)

            elif view_name == "back":
                pos = center + np.array([0.0, -dist, 0.0])
                up = (0.0, 0.0, 1.0)

            elif view_name == "left":
                pos = center + np.array([-dist, 0.0, 0.0])
                up = (0.0, 0.0, 1.0)

            elif view_name == "right":
                pos = center + np.array([dist, 0.0, 0.0])
                up = (0.0, 0.0, 1.0)

            elif view_name == "top":
                pos = center + np.array([0.0, 0.0, dist])
                up = (0.0, 1.0, 0.0)

            elif view_name == "beauty_left":
                pos = center + np.array([-0.65 * dist, 0.95 * dist, 0.35 * dist])
                up = (0.0, 0.0, 1.0)

            elif view_name == "beauty_right":
                pos = center + np.array([0.65 * dist, 0.95 * dist, 0.35 * dist])
                up = (0.0, 0.0, 1.0)

            else:
                return

            cam = plotter.camera
            cam.position = tuple(float(v) for v in pos)
            cam.focal_point = tuple(float(v) for v in center)
            cam.up = up

            try:
                plotter.reset_camera_clipping_range()
            except Exception:
                pass

            plotter.render()

        except Exception:
            pass

    def _save_current_3d_camera_to_state(self) -> None:
        """
        Save current mini 3D camera only in memory.
        Later we can persist it in JSON if needed.
        """
        try:
            self._oblique_saved_camera = self._camera_state_tuple()
        except Exception:
            pass

    def _apply_saved_3d_camera_from_state(self) -> None:
        try:
            self._restore_camera_state_tuple(getattr(self, "_oblique_saved_camera", None))
        except Exception:
            pass

    def _rotation_axis_from_user(self) -> str:
        try:
            selected_axis = NeuXelecSelectionDialog.select_item(
                self._dialog_parent(),
                "GIF rotation axis",
                "Choose the anatomical axis used to rotate the mini 3D brain:",
                options=[
                    "Z axis - axial / vertical rotation",
                    "Y axis - coronal rotation",
                    "X axis - sagittal rotation",
                ],
                current_index=0,
                accept_text="Continue",
                reject_text="Cancel",
            )

            if not selected_axis:
                return ""

            normalized_axis = str(selected_axis).strip().upper()

            if normalized_axis.startswith("X"):
                return "X"

            if normalized_axis.startswith("Y"):
                return "Y"

            return "Z"

        except Exception:
            return ""

    def _rotate_camera_for_gif_frame(self, axis: str, step_deg: float) -> None:
        plotter = self._oblique_plotter()
        if plotter is None:
            return

        try:
            cam = plotter.camera

            pos = np.asarray(cam.position, dtype=np.float64)
            focal = np.asarray(cam.focal_point, dtype=np.float64)
            up = np.asarray(cam.up, dtype=np.float64)

            vec = pos - focal

            axis = str(axis).upper().strip()
            if axis == "X":
                rot_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            elif axis == "Y":
                rot_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            else:
                rot_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            theta = np.deg2rad(float(step_deg))
            k = rot_axis / max(np.linalg.norm(rot_axis), 1e-9)

            def _rotate(v):
                v = np.asarray(v, dtype=np.float64)
                return (
                    v * np.cos(theta)
                    + np.cross(k, v) * np.sin(theta)
                    + k * np.dot(k, v) * (1.0 - np.cos(theta))
                )

            cam.position = tuple(float(v) for v in (focal + _rotate(vec)))
            cam.up = tuple(float(v) for v in _rotate(up))

            try:
                plotter.reset_camera_clipping_range()
            except Exception:
                pass

        except Exception:
            pass

    def _export_3d_view_gif(self) -> None:
        plotter = self._oblique_plotter()
        if plotter is None:
            return

        axis = self._rotation_axis_from_user()
        if not axis:
            return

        filename, _ = QFileDialog.getSaveFileName(
            _top_level_window(),
            "Export rotating mini 3D GIF",
            f"neuxelec_oblique_3d_rotation_{axis}.gif",
            "GIF animation (*.gif)",
        )

        if not filename:
            return

        if not filename.lower().endswith(".gif"):
            filename += ".gif"

        n_frames = 72
        step_deg = 360.0 / float(n_frames)
        original_camera = self._camera_state_tuple()

        try:
            plotter.open_gif(filename)

            plotter.render()
            plotter.write_frame()

            for _ in range(n_frames):
                self._rotate_camera_for_gif_frame(axis, step_deg)
                plotter.render()
                plotter.write_frame()

        except Exception as e:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(),
                "GIF export",
                f"Failed to export rotating mini 3D GIF:\n{e}",
            )

        finally:
            try:
                writer = getattr(plotter, "mwriter", None)
                if writer is not None:
                    writer.close()
            except Exception:
                pass

            try:
                plotter.mwriter = None
            except Exception:
                pass

            try:
                self._restore_camera_state_tuple(original_camera)
            except Exception:
                pass

            try:
                plotter.render()
            except Exception:
                pass

    def _get_brain_mesh_for_oblique_view(self):
        # Priority 1: pial from 3D view if available
        vp = getattr(self.state, "view3d_page", None)
        if vp is not None:
            try:
                polys = []
                if getattr(vp, "_lh_pial_poly", None) is not None:
                    polys.append(vp._lh_pial_poly.copy())
                if getattr(vp, "_rh_pial_poly", None) is not None:
                    polys.append(vp._rh_pial_poly.copy())

                if polys:
                    mesh = polys[0]
                    for p in polys[1:]:
                        mesh = mesh.merge(p)
                    try:
                        mesh = mesh.triangulate()
                    except Exception:
                        pass
                    try:
                        mesh = mesh.clean(tolerance=1e-6)
                    except Exception:
                        pass
                    return mesh, "pial"
            except Exception:
                pass

        # Priority 2: brainmask
        mask = getattr(self.state, "brainmask_sitk", None)
        if mask is not None and vp is not None and hasattr(vp, "_binarymask_to_polydata"):
            try:
                mesh = vp._binarymask_to_polydata(mask)
                return mesh, "brainmask"
            except Exception:
                pass

        # Priority 3: fallback fmh.mat
        if self._fmh_mesh is None:
            try:
                data = sio.loadmat("/mnt/data/fmh.mat")
                fmh = data["fmh"][0, 0]
                verts = np.asarray(fmh["vertices"], dtype=np.float32)
                faces = np.asarray(fmh["faces"], dtype=np.int64) - 1

                faces_vtk = np.hstack(
                    [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
                ).ravel()

                self._fmh_mesh = pv.PolyData(verts, faces_vtk)
            except Exception:
                self._fmh_mesh = None

        if self._fmh_mesh is not None:
            return self._fmh_mesh, "fallback"

        return None, None

    def _get_preview_brain_center_lps(self):
        """
        Return the brain center in LPS coordinates for the small oblique 3D preview.

        Priority:
        1) visible brain mesh center from the mini 3D preview source
        2) T1 physical center
        3) None
        """
        # 1) Try brain mesh center.
        try:
            mesh, _kind = self._get_brain_mesh_for_oblique_view()
            if mesh is not None and getattr(mesh, "n_points", 0) > 0:
                center_ras = np.asarray(mesh.center, dtype=np.float64)
                center_lps = center_ras.copy()
                center_lps[0] *= -1.0
                center_lps[1] *= -1.0
                return center_lps
        except Exception:
            pass

        # 2) Fallback: T1 physical center.
        try:
            img = self._get_t1_image()
            if img is not None:
                size = img.GetSize()
                idx_center = (
                    int(size[0] // 2),
                    int(size[1] // 2),
                    int(size[2] // 2),
                )
                return np.asarray(
                    img.TransformIndexToPhysicalPoint(idx_center),
                    dtype=np.float64,
                )
        except Exception:
            pass

        return None

    def _make_plane_mesh_for_electrode(self, elec: dict, angle_deg: float):
        axis, electrode_center = self._electrode_axis_and_center(elec)
        if axis is None or electrode_center is None:
            return None

        u, w = self._make_plane_basis(axis, angle_deg)

        # ---------------------------------------------------------
        # Small 3D preview plane.
        # Important:
        # The real oblique slice extraction still uses the electrode.
        # Here we only change the displayed quad center for a nicer preview.
        # ---------------------------------------------------------
        length_mm = float(getattr(self, "_oblique_preview_plane_length_mm", 95.0))
        width_mm = float(getattr(self, "_oblique_preview_plane_width_mm", 70.0))

        s_min = -0.5 * length_mm
        s_max = 0.5 * length_mm
        t_min = -0.5 * width_mm
        t_max = 0.5 * width_mm

        # Use brain center for visual centering, but project it onto the electrode-defined plane.
        # This keeps the plane parallel/oriented exactly like the oblique slice and still belonging
        # to the same mathematical plane family defined by the electrode.
        brain_center_lps = self._get_preview_brain_center_lps()

        if brain_center_lps is not None:
            try:
                rel = np.asarray(brain_center_lps, dtype=np.float64) - np.asarray(
                    electrode_center, dtype=np.float64
                )

                # Projection of brain center onto the oblique plane spanned by u and w.
                preview_center_lps = (
                    np.asarray(electrode_center, dtype=np.float64)
                    + float(np.dot(rel, u)) * u
                    + float(np.dot(rel, w)) * w
                )
            except Exception:
                preview_center_lps = np.asarray(electrode_center, dtype=np.float64)
        else:
            preview_center_lps = np.asarray(electrode_center, dtype=np.float64)

        c = self._lps_to_ras_point(preview_center_lps)
        u_r = self._lps_to_ras_vec(u)
        w_r = self._lps_to_ras_vec(w)

        p0 = c + s_min * u_r + t_min * w_r
        p1 = c + s_min * u_r + t_max * w_r
        p2 = c + s_max * u_r + t_max * w_r
        p3 = c + s_max * u_r + t_min * w_r

        pts = np.vstack([p0, p1, p2, p3]).astype(np.float32)
        faces = np.array([4, 0, 1, 2, 3], dtype=np.int64)

        return pv.PolyData(pts, faces)

    def _make_plane_outline_mesh(self, plane_mesh):
        if plane_mesh is None:
            return None

        try:
            return plane_mesh.extract_feature_edges(
                boundary_edges=True,
                feature_edges=False,
                manifold_edges=False,
                non_manifold_edges=False,
            )
        except Exception:
            return None

    def _render_brain_planes(
        self, elec1_name: str | None, elec2_name: str | None, angle1: float, angle2: float
    ):
        if not self._is_active_page:
            return

        if self._brain_plotter is None:
            return

        if self.frame_brain is None:
            return

        try:
            if not self.ui.window().isVisible():
                return
            if not self.frame_brain.isVisible():
                return
            if not self._brain_plotter.isVisible():
                return
        except Exception:
            return

        if self.frame_brain.width() <= 0 or self.frame_brain.height() <= 0:
            return

        if self._is_rendering_brain:
            return

        self._is_rendering_brain = True
        try:
            self._setup_oblique_brain_lights()

            # ---------- Brain actor: only create/update if source changed ----------
            brain_mesh, brain_kind = self._get_brain_mesh_for_oblique_view()

            if self._brain_actor is None or brain_kind != self._last_brain_kind:
                if self._brain_actor is not None:
                    try:
                        self._brain_plotter.remove_actor(self._brain_actor, reset_camera=False)
                    except Exception:
                        pass
                    self._brain_actor = None

                if brain_mesh is not None:
                    try:
                        if brain_kind == "pial":
                            self._brain_actor = self._brain_plotter.add_mesh(
                                brain_mesh,
                                color="lightgray",
                                opacity=1.0,
                                smooth_shading=True,
                                ambient=0.35,
                                diffuse=0.60,
                                specular=0.08,
                                specular_power=12.0,
                            )
                            try:
                                self._brain_actor.ForceOpaqueOn()
                            except Exception:
                                pass
                            try:
                                prop = self._brain_actor.GetProperty()
                                prop.BackfaceCullingOn()
                                prop.SetInterpolationToPhong()
                            except Exception:
                                pass
                        else:
                            self._brain_actor = self._brain_plotter.add_mesh(
                                brain_mesh,
                                color="lightgray",
                                opacity=1.0,
                                smooth_shading=True,
                                ambient=0.10,
                                diffuse=0.80,
                                specular=0.30,
                                specular_power=40.0,
                            )
                    except Exception:
                        self._brain_actor = None

                self._last_brain_kind = brain_kind

            # ---------- Plane 1 ----------
            try:
                color1_key = (
                    tuple(int(v) for v in self._get_electrode_rgb(elec1_name))
                    if elec1_name
                    else None
                )
            except Exception:
                color1_key = None

            plane1_key = (elec1_name, float(angle1), color1_key)
            if plane1_key != self._last_plane1_key:
                # remove old plane surface
                if self._plane_actor_1 is not None:
                    try:
                        self._brain_plotter.remove_actor(self._plane_actor_1, reset_camera=False)
                    except Exception:
                        pass
                    self._plane_actor_1 = None

                # remove old plane outline
                if self._plane_outline_actor_1 is not None:
                    try:
                        self._brain_plotter.remove_actor(
                            self._plane_outline_actor_1, reset_camera=False
                        )
                    except Exception:
                        pass
                    self._plane_outline_actor_1 = None

                if elec1_name:
                    elec1 = self._get_electrode_by_name(elec1_name)
                    if elec1 is not None:
                        plane1 = self._make_plane_mesh_for_electrode(elec1, angle1)
                        if plane1 is not None:
                            color1 = self._get_electrode_rgb(elec1_name)
                            color1_list = [int(color1[0]), int(color1[1]), int(color1[2])]

                            # Semi-transparent plane surface
                            try:
                                self._plane_actor_1 = self._brain_plotter.add_mesh(
                                    plane1,
                                    color=color1_list,
                                    opacity=0.28,
                                    show_edges=False,
                                    lighting=False,
                                    name="oblique_plane_1_surface",
                                )

                                try:
                                    prop = self._plane_actor_1.GetProperty()
                                    prop.SetAmbient(1.0)
                                    prop.SetDiffuse(0.0)
                                    prop.SetSpecular(0.0)
                                except Exception:
                                    pass

                            except Exception:
                                self._plane_actor_1 = None

                            # Fully opaque outline
                            try:
                                outline1 = self._make_plane_outline_mesh(plane1)
                                if outline1 is not None and outline1.n_points > 0:
                                    self._plane_outline_actor_1 = self._brain_plotter.add_mesh(
                                        outline1,
                                        color=color1_list,
                                        opacity=1.0,
                                        line_width=4,
                                        name="oblique_plane_1_outline",
                                    )

                                    try:
                                        prop = self._plane_outline_actor_1.GetProperty()
                                        prop.SetAmbient(1.0)
                                        prop.SetDiffuse(0.0)
                                        prop.SetSpecular(0.0)
                                    except Exception:
                                        pass

                            except Exception:
                                self._plane_outline_actor_1 = None

                self._last_plane1_key = plane1_key

            # ---------- Plane 2 ----------
            try:
                color2_key = (
                    tuple(int(v) for v in self._get_electrode_rgb(elec2_name))
                    if elec2_name
                    else None
                )
            except Exception:
                color2_key = None

            plane2_key = (elec2_name, float(angle2), color2_key)
            if plane2_key != self._last_plane2_key:
                # remove old plane surface
                if self._plane_actor_2 is not None:
                    try:
                        self._brain_plotter.remove_actor(self._plane_actor_2, reset_camera=False)
                    except Exception:
                        pass
                    self._plane_actor_2 = None

                # remove old plane outline
                if self._plane_outline_actor_2 is not None:
                    try:
                        self._brain_plotter.remove_actor(
                            self._plane_outline_actor_2, reset_camera=False
                        )
                    except Exception:
                        pass
                    self._plane_outline_actor_2 = None

                if elec2_name:
                    elec2 = self._get_electrode_by_name(elec2_name)
                    if elec2 is not None:
                        plane2 = self._make_plane_mesh_for_electrode(elec2, angle2)
                        if plane2 is not None:
                            color2 = self._get_electrode_rgb(elec2_name)
                            color2_list = [int(color2[0]), int(color2[1]), int(color2[2])]

                            # Semi-transparent plane surface
                            try:
                                self._plane_actor_2 = self._brain_plotter.add_mesh(
                                    plane2,
                                    color=color2_list,
                                    opacity=0.28,
                                    show_edges=False,
                                    lighting=False,
                                    name="oblique_plane_2_surface",
                                )

                                try:
                                    prop = self._plane_actor_2.GetProperty()
                                    prop.SetAmbient(1.0)
                                    prop.SetDiffuse(0.0)
                                    prop.SetSpecular(0.0)
                                except Exception:
                                    pass

                            except Exception:
                                self._plane_actor_2 = None

                            # Fully opaque outline
                            try:
                                outline2 = self._make_plane_outline_mesh(plane2)
                                if outline2 is not None and outline2.n_points > 0:
                                    self._plane_outline_actor_2 = self._brain_plotter.add_mesh(
                                        outline2,
                                        color=color2_list,
                                        opacity=1.0,
                                        line_width=4,
                                        name="oblique_plane_2_outline",
                                    )

                                    try:
                                        prop = self._plane_outline_actor_2.GetProperty()
                                        prop.SetAmbient(1.0)
                                        prop.SetDiffuse(0.0)
                                        prop.SetSpecular(0.0)
                                    except Exception:
                                        pass

                            except Exception:
                                self._plane_outline_actor_2 = None

                self._last_plane2_key = plane2_key

        finally:
            self._is_rendering_brain = False

    def _setup_oblique_brain_lights(self):
        if self._brain_plotter is None:
            return

        try:
            self._brain_plotter.set_background("#2b2d31")
        except Exception:
            pass

        try:
            self._brain_plotter.remove_all_lights()
        except Exception:
            pass

        try:
            head = pv.Light(light_type="headlight")
            head.intensity = 1.0
            self._brain_plotter.add_light(head)
        except Exception:
            pass

        try:
            fill1 = pv.Light(
                position=(250, 250, 250),
                focal_point=(0, 0, 0),
                intensity=0.25,
            )
            fill2 = pv.Light(
                position=(-250, -250, 250),
                focal_point=(0, 0, 0),
                intensity=0.20,
            )
            self._brain_plotter.add_light(fill1)
            self._brain_plotter.add_light(fill2)
        except Exception:
            pass

    def _on_rotation_changed(self, slot_index: int):
        slot_index = int(slot_index)

        if slot_index == 1:
            self._slice_cache_1 = None
            self._base_cache_1 = None
            self._pet_cache_1 = None
            self._render_cache_1 = None
            self._last_plane1_key = None

            self.render_slice_slot(1)

        elif slot_index == 2:
            self._slice_cache_2 = None
            self._base_cache_2 = None
            self._pet_cache_2 = None
            self._render_cache_2 = None
            self._last_plane2_key = None

            self.render_slice_slot(2)

        # Mini 3D preview:
        # render_brain_only() calls _render_brain_planes(),
        # but because only one _last_planeX_key is reset,
        # only that plane actor will be rebuilt.
        if self._is_active_page:
            self.render_brain_only()

    def _on_electrode_selection_changed(self):
        current = tuple(self._get_checked_electrode_names())

        if current == self._last_checked_electrode_names:
            return

        old_slot1 = getattr(self, "_last_displayed_electrode_slot1", None)
        old_slot2 = getattr(self, "_last_displayed_electrode_slot2", None)

        new_slot1 = current[0] if len(current) > 0 else None
        new_slot2 = current[1] if len(current) > 1 else None

        # If a new electrode appears in a slot, do not inherit the previous
        # zoom/pan/visual rotation/background state.
        if new_slot1 != old_slot1:
            self._reset_oblique_slice_display_to_default(
                1,
                refresh=False,
                reset_background=True,
            )

        if new_slot2 != old_slot2:
            self._reset_oblique_slice_display_to_default(
                2,
                refresh=False,
                reset_background=True,
            )

        self._last_displayed_electrode_slot1 = new_slot1
        self._last_displayed_electrode_slot2 = new_slot2
        self._last_checked_electrode_names = current

        self._slice_cache_1 = None
        self._slice_cache_2 = None
        self._last_brain_key = None
        self._last_plane1_key = None
        self._last_plane2_key = None
        self._last_brain_kind = None

        self._render_cache_1 = None
        self._render_cache_2 = None
        self._base_cache_1 = None
        self._base_cache_2 = None
        self._pet_cache_1 = None
        self._pet_cache_2 = None

        self._schedule_refresh(slices=True, brain=True)

    def _on_oblique_modality_toggled(self, cb, checked: bool):
        # If user turns PET/SISCOM on, show color scales automatically again.
        try:
            if checked and cb in (self.chk_pet, self.chk_siscom):
                self._show_color_scales = True
        except Exception:
            pass

        # If MRI is turned on, default to T1 unless T2 is already selected and available.
        try:
            if cb is self.chk_mri:
                if checked:
                    if (
                        getattr(self, "_oblique_mri_source", "T1") == "T2"
                        and self._get_t2_image() is not None
                    ):
                        self._set_checkbox_checked_silent(self.chk_mri_t1, False)
                        self._set_checkbox_checked_silent(self.chk_mri_t2, True)
                    else:
                        self._oblique_mri_source = "T1"
                        self._set_checkbox_checked_silent(self.chk_mri_t1, True)
                        self._set_checkbox_checked_silent(self.chk_mri_t2, False)
        except Exception:
            pass

        self._invalidate_oblique_image_caches()
        self._update_modality_controls_enabled_states()
        self._schedule_refresh(slices=True, brain=False)

    def _is_pet_or_siscom_checked(self) -> bool:
        try:
            pet_on = bool(self.chk_pet is not None and self.chk_pet.isChecked())
        except Exception:
            pet_on = False

        try:
            siscom_on = bool(self.chk_siscom is not None and self.chk_siscom.isChecked())
        except Exception:
            siscom_on = False

        return bool(pet_on or siscom_on)

    def _show_slice_context_menu(self, pos):
        sender = self.sender()
        if sender is None:
            return

        try:
            global_pos = sender.mapToGlobal(pos)
            choice = exec_oblique_slice_menu(
                global_pos,
                color_scale_visible=bool(getattr(self, "_show_color_scales", True)),
                show_color_scale_option=bool(self._is_pet_or_siscom_checked()),
            )

            if choice == "pet":
                self._choose_pet_colormap()
            elif choice == "siscom":
                self._choose_siscom_colormap()
            elif choice == "toggle_color_scale":
                self._toggle_color_scales()
        except Exception:
            pass

    def _toggle_color_scales(self):
        self._show_color_scales = not bool(getattr(self, "_show_color_scales", True))
        self._schedule_refresh(slices=True, brain=False)

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

        self._slice_cache_1 = None
        self._slice_cache_2 = None
        self._render_cache_1 = None
        self._render_cache_2 = None
        self._base_cache_1 = None
        self._base_cache_2 = None

        self._schedule_refresh(slices=True, brain=False)

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

        self._slice_cache_1 = None
        self._slice_cache_2 = None
        self._render_cache_1 = None
        self._render_cache_2 = None
        self._base_cache_1 = None
        self._base_cache_2 = None
        self._pet_cache_1 = None
        self._pet_cache_2 = None

        self._schedule_refresh(slices=True, brain=False)

    def _get_pet_gamma_value(self) -> float:
        try:
            if self.spn_pet_gamma is not None:
                return max(0.1, float(self.spn_pet_gamma.value()))
        except Exception:
            pass
        try:
            if self.sld_pet_gamma is not None:
                return max(0.1, float(self.sld_pet_gamma.value()) / 100.0)
        except Exception:
            pass
        return 1.0

    def _get_pet_scalar_bar_range(self):
        pet_img = self._get_pet_image()
        if pet_img is None:
            return None

        try:
            pet_np = sitk.GetArrayFromImage(pet_img).astype(np.float32)
            finite = np.isfinite(pet_np)
            vals = pet_np[finite]
            vals = vals[vals > 0]
            if vals.size == 0:
                return None

            pmin = float(self.spn_pet_min.value()) if self.spn_pet_min is not None else 15.0
            pmax = float(self.spn_pet_max.value()) if self.spn_pet_max is not None else 75.0
            gamma = self._get_pet_gamma_value()

            lo, hi = get_pet_window(vals, pmin, pmax)
            if hi <= lo:
                hi = lo + 1.0

            return {
                "lo": float(lo),
                "hi": float(hi),
                "gamma": float(gamma),
            }
        except Exception:
            return None

    def _get_siscom_scalar_bar_range(self):
        sis_img = self._get_siscom_image()
        if sis_img is None:
            return None

        try:
            zmin = float(self.dsb_siscom_z.value()) if self.dsb_siscom_z is not None else 2.0

            # Try to use the same fixed zmax as 3D View if available
            zmax = None
            try:
                vp = getattr(self.state, "view3d_page", None)
                zmax = getattr(vp, "_siscom_fixed_zmax", None)
            except Exception:
                zmax = None

            if zmax is None or not np.isfinite(float(zmax)) or float(zmax) <= zmin:
                sis_np = sitk.GetArrayFromImage(sis_img).astype(np.float32)
                valid = np.isfinite(sis_np) & (sis_np >= zmin)
                vals = sis_np[valid]

                if vals.size == 0:
                    return None

                zmax = float(np.percentile(vals, 99.0))
                if zmax <= zmin:
                    zmax = zmin + 1.0

            lo, hi = get_siscom_window(None, zmin, float(zmax))

            return {
                "lo": float(lo),
                "hi": float(hi),
            }

        except Exception:
            return None

    def _draw_vertical_scalar_bar_on_pixmap(
        self,
        pm: QPixmap,
        title: str,
        lo: float,
        hi: float,
        cmap_name: str,
        x_frac: float,
    ) -> QPixmap:
        if pm is None or pm.isNull():
            return pm

        out = pm.copy()

        painter = QPainter(out)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        w = out.width()
        h = out.height()

        hi_txt = f"{float(hi):.2f}"
        lo_txt = f"{float(lo):.2f}"

        font = QFont()
        font.setPointSize(max(8, int(round(0.026 * h))))
        painter.setFont(font)
        fm = painter.fontMetrics()

        title_w = fm.horizontalAdvance(title)
        label_w = max(fm.horizontalAdvance(hi_txt), fm.horizontalAdvance(lo_txt))
        text_h = fm.height()

        bar_w = max(14, int(round(0.038 * w)))
        bar_h = max(120, int(round(0.68 * h)))

        text_pad = 4
        box_margin = 8
        inner_pad = 6

        box_w = max(title_w + 2 * inner_pad, bar_w + label_w + text_pad + 3 * inner_pad)
        box_h = bar_h + text_h + 3 * inner_pad

        box_x = int(round(float(x_frac) * w))
        box_y = int(round(0.10 * h))

        if box_x + box_w > w - box_margin:
            box_x = max(box_margin, w - box_w - box_margin)
        if box_y + box_h > h - box_margin:
            box_y = max(box_margin, h - box_h - box_margin)

        title_x = box_x + inner_pad
        title_y = box_y + fm.ascent() + 2

        bar_x = box_x + inner_pad
        bar_y = title_y + 6 + max(4, int(0.35 * text_h))
        bar_rect_h = min(bar_h, h - bar_y - box_margin)
        bar_rect_h = max(80, bar_rect_h)

        box_h = (bar_y - box_y) + bar_rect_h + inner_pad

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 155))
        painter.drawRoundedRect(box_x, box_y, box_w, box_h, 8, 8)

        grad = QLinearGradient(bar_x, bar_y + bar_rect_h, bar_x, bar_y)

        sample_vals = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(256, 1)

        try:
            # PET and SISCOM use same matplotlib-like colormap logic
            sample_rgb = pet_norm_to_colormap(sample_vals, cmap_name)

            if np.issubdtype(sample_rgb.dtype, np.floating):
                sample_rgb = np.clip(sample_rgb * 255.0, 0.0, 255.0)

            sample_rgb = sample_rgb.astype(np.uint8)

            for i in range(sample_rgb.shape[0]):
                rr = int(sample_rgb[i, 0, 0])
                gg = int(sample_rgb[i, 0, 1])
                bb = int(sample_rgb[i, 0, 2])
                grad.setColorAt(
                    float(i) / float(max(1, sample_rgb.shape[0] - 1)),
                    QColor(rr, gg, bb),
                )

        except Exception:
            grad.setColorAt(0.0, QColor(0, 0, 0))
            grad.setColorAt(1.0, QColor(255, 140, 0))

        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawRect(bar_x, bar_y, bar_w, bar_rect_h)

        painter.setPen(QColor(255, 255, 255))
        painter.setFont(font)

        label_x = bar_x + bar_w + text_pad
        hi_y = bar_y + fm.ascent()
        lo_y = bar_y + bar_rect_h

        painter.drawText(title_x, title_y, title)
        painter.drawText(label_x, hi_y, hi_txt)
        painter.drawText(label_x, lo_y, lo_txt)

        painter.end()
        return out

    def _overlay_pet_scalar_bar_on_pixmap(self, pm: QPixmap) -> QPixmap:
        if pm is None or pm.isNull():
            return pm

        show_pet = bool(self.chk_pet is not None and self.chk_pet.isChecked())
        if not show_pet:
            return pm

        info = self._get_pet_scalar_bar_range()
        if info is None:
            return pm

        gamma = float(info["gamma"])
        lo = float(info["lo"])
        hi = float(info["hi"])

        return self._draw_vertical_scalar_bar_on_pixmap(
            pm=pm,
            title=f"PET (γ={gamma:.2f})",
            lo=lo,
            hi=hi,
            cmap_name=self._pet_colormap_name,
            x_frac=0.70,
        )

    def _overlay_siscom_scalar_bar_on_pixmap(self, pm: QPixmap) -> QPixmap:
        if pm is None or pm.isNull():
            return pm

        show_siscom = bool(self.chk_siscom is not None and self.chk_siscom.isChecked())
        if not show_siscom:
            return pm

        info = self._get_siscom_scalar_bar_range()
        if info is None:
            return pm

        lo = float(info["lo"])
        hi = float(info["hi"])

        return self._draw_vertical_scalar_bar_on_pixmap(
            pm=pm,
            title="SISCOM Z",
            lo=lo,
            hi=hi,
            cmap_name=self._siscom_colormap_name,
            x_frac=0.86,
        )

    def _overlay_scalar_bars_on_pixmap(self, pm: QPixmap) -> QPixmap:
        if pm is None or pm.isNull():
            return pm

        if not bool(getattr(self, "_show_color_scales", True)):
            return pm

        out = pm
        out = self._overlay_pet_scalar_bar_on_pixmap(out)
        out = self._overlay_siscom_scalar_bar_on_pixmap(out)
        return out

    def _overlay_contacts_and_labels_on_display_pixmap(
        self,
        pm_display: QPixmap,
        pm_source: QPixmap,
        contact_rows,
        contact_cols,
        contact_names,
        elec_name: str | None,
        zoom,
        pan_xy,
    ):
        if pm_display is None or pm_display.isNull():
            return pm_display
        if pm_source is None or pm_source.isNull():
            return pm_display

        out = pm_display.copy()

        src_w = pm_source.width()
        src_h = pm_source.height()
        dst_w = out.width()
        dst_h = out.height()

        zoom = max(1.0, float(zoom))

        crop_w = max(1, int(round(src_w / zoom)))
        crop_h = max(1, int(round(src_h / zoom)))

        cx = src_w / 2.0
        cy = src_h / 2.0

        if pan_xy is not None:
            cx += float(pan_xy[0])
            cy += float(pan_xy[1])

        x0 = int(round(cx - crop_w / 2.0))
        y0 = int(round(cy - crop_h / 2.0))

        x0 = max(0, min(x0, src_w - crop_w))
        y0 = max(0, min(y0, src_h - crop_h))

        scale_x = dst_w / max(1.0, float(crop_w))
        scale_y = dst_h / max(1.0, float(crop_h))

        contact_color = self._get_electrode_rgb(elec_name)

        radius = 4
        if self.spn_contact_size is not None:
            try:
                radius = max(1, int(self.spn_contact_size.value()))
            except Exception:
                pass

        painter = QPainter(out)

        pen = QPen(QColor(int(contact_color[0]), int(contact_color[1]), int(contact_color[2])))
        painter.setPen(pen)

        font = QFont()
        font.setPointSize(max(8, int(8 + 2 * zoom)))
        painter.setFont(font)

        brush_color = QColor(int(contact_color[0]), int(contact_color[1]), int(contact_color[2]))
        painter.setBrush(brush_color)

        for r, c, txt in zip(contact_rows, contact_cols, contact_names):
            dx = (float(c) - x0) * scale_x
            dy = (float(r) - y0) * scale_y

            if dx < 0 or dx >= dst_w or dy < 0 or dy >= dst_h:
                continue

            rr = max(2, int(radius * min(scale_x, scale_y)))
            painter.drawEllipse(QPoint(int(dx), int(dy)), rr, rr)

            if txt:
                painter.drawText(int(dx + rr + 4), int(dy - rr - 2), str(txt))

        painter.end()
        return out

    def cleanup(self):
        """
        Stop Oblique Slice rendering safely.

        Important:
        Do NOT manually close/delete the QtInteractor on Windows.
        Let Qt destroy the widget with the main UI, like the main 3D View.
        """
        try:
            if self._refresh_timer is not None:
                self._refresh_timer.stop()
        except Exception:
            pass

        self._pending_refresh_slices = False
        self._pending_refresh_brain = False
        self._is_active_page = False
        self._is_rendering_brain = False

        plotter = getattr(self, "_brain_plotter", None)
        if plotter is None:
            return

        try:
            plotter.disable()
        except Exception:
            pass

        try:
            plotter.hide()
        except Exception:
            pass

        self._brain_actor = None
        self._plane_actor_1 = None
        self._plane_actor_2 = None
        self._plane_outline_actor_1 = None
        self._plane_outline_actor_2 = None
        self._last_brain_key = None
        self._last_plane1_key = None
        self._last_plane2_key = None
        self._last_brain_kind = None

    def _make_render_cache_key(
        self,
        elec_name: str | None,
        angle_deg: float,
        t1_enabled: bool,
        ct_enabled: bool,
        pet_enabled: bool,
        sis_enabled: bool,
        t1_opacity: int,
        ct_opacity: int,
        pet_opacity: int,
        sis_opacity: int,
    ):
        pmin = int(self.spn_pet_min.value()) if self.spn_pet_min is not None else 15
        pmax = int(self.spn_pet_max.value()) if self.spn_pet_max is not None else 75
        gamma = round(self._get_pet_gamma_value(), 3)

        return (
            elec_name,
            float(angle_deg),
            bool(t1_enabled),
            bool(ct_enabled),
            bool(pet_enabled),
            bool(sis_enabled),
            int(t1_opacity),
            int(ct_opacity),
            int(pet_opacity),
            int(sis_opacity),
            int(pmin),
            int(pmax),
            float(gamma),
            str(self._pet_colormap_name),
        )

    def refresh_mri_source_controls(self):
        self._update_modality_controls_enabled_states()
        self._invalidate_oblique_image_caches()
        self._schedule_refresh(slices=True, brain=False)

    def _make_base_cache_key(
        self,
        elec_name: str | None,
        angle_deg: float,
        t1_enabled: bool,
        ct_enabled: bool,
        sis_enabled: bool,
        parcel1_enabled: bool,
        parcel2_enabled: bool,
        t1_opacity: int,
        ct_opacity: int,
        sis_opacity: int,
        parcel1_opacity: int,
        parcel2_opacity: int,
    ):
        zthr = round(float(self.dsb_siscom_z.value()), 3) if self.dsb_siscom_z is not None else 2.0

        return (
            elec_name,
            float(angle_deg),
            str(getattr(self, "_oblique_mri_source", "T1")),
            bool(t1_enabled),
            bool(ct_enabled),
            bool(sis_enabled),
            bool(parcel1_enabled),
            bool(parcel2_enabled),
            int(t1_opacity),
            int(ct_opacity),
            int(sis_opacity),
            int(parcel1_opacity),
            int(parcel2_opacity),
            float(zthr),
            str(self._siscom_colormap_name),
        )

    def _make_pet_cache_key(
        self,
        elec_name: str | None,
        angle_deg: float,
        pet_enabled: bool,
        pet_opacity: int,
    ):
        pmin = int(self.spn_pet_min.value()) if self.spn_pet_min is not None else 15
        pmax = int(self.spn_pet_max.value()) if self.spn_pet_max is not None else 75
        gamma = round(self._get_pet_gamma_value(), 3)

        return (
            elec_name,
            float(angle_deg),
            bool(pet_enabled),
            int(pet_opacity),
            int(pmin),
            int(pmax),
            float(gamma),
            str(self._pet_colormap_name),
        )

    def _build_base_rgb(
        self,
        arr_t1,
        arr_ct,
        arr_sis,
        arr_parcel1,
        arr_parcel2,
        t1_opacity_pct: int,
        ct_opacity_pct: int,
        sis_opacity_pct: int,
        parcel1_opacity_pct: int,
        parcel2_opacity_pct: int,
    ):
        base = None
        for arr in (arr_t1, arr_ct, arr_sis, arr_parcel1, arr_parcel2):
            if arr is not None:
                base = arr
                break

        if base is None:
            return None

        H, W = base.shape
        img = np.zeros((H, W, 3), dtype=np.float32)

        def _norm(arr):
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return None
            lo = float(np.percentile(finite, 1.0))
            hi = float(np.percentile(finite, 99.0))
            if hi <= lo:
                lo, hi = float(np.min(finite)), float(np.max(finite) + 1.0)
            out = (arr - lo) / max(1e-6, (hi - lo))
            return np.clip(out, 0.0, 1.0)

        if arr_t1 is not None:
            n = _norm(arr_t1)
            if n is not None:
                a = float(np.clip(t1_opacity_pct, 0, 100)) / 100.0
                gray = n[..., None] * 255.0
                img += gray * a

        if arr_ct is not None:
            n = _norm(arr_ct)
            if n is not None:
                a = float(np.clip(ct_opacity_pct, 0, 100)) / 100.0
                gray = n[..., None] * 255.0
                img = img * (1.0 - 0.5 * a) + gray * a

        if arr_sis is not None:
            zthr = float(self.dsb_siscom_z.value()) if self.dsb_siscom_z is not None else 2.0

            sis_valid = np.isfinite(arr_sis) & (arr_sis >= zthr)
            sis_vals = arr_sis[sis_valid]

            if sis_vals.size > 0:
                zmax = float(np.percentile(sis_vals, 99.0))
                if zmax <= zthr:
                    zmax = zthr + 1.0

                sis_norm = normalize_threshold_map(
                    arr_sis,
                    lo=zthr,
                    hi=zmax,
                    gamma=1.0,
                    mask=sis_valid.astype(np.uint8),
                )
                sis_rgb = pet_norm_to_colormap(sis_norm, self._siscom_colormap_name)
                a = float(np.clip(sis_opacity_pct, 0, 100)) / 100.0

                img = blend_pet_on_rgb(
                    img,
                    sis_rgb,
                    sis_norm,
                    alpha_scale=a,
                )
        if arr_parcel1 is not None:
            try:
                valid_mask = np.isfinite(arr_parcel1) & (arr_parcel1 > 0)

                if np.any(valid_mask):
                    labels = np.zeros(arr_parcel1.shape, dtype=np.int32)
                    labels[valid_mask] = np.round(arr_parcel1[valid_mask]).astype(np.int32)

                    lut = self._get_parcellation1_lut()
                    rgb_parc = np.zeros((H, W, 3), dtype=np.float32)

                    unique_labels = np.unique(labels[valid_mask])
                    for lab in unique_labels:
                        entry = lut.get(int(lab), None)
                        if entry is None:
                            continue
                        try:
                            _, (r, g, b) = entry
                        except Exception:
                            continue

                        m = labels == int(lab)
                        rgb_parc[m, 0] = float(r)
                        rgb_parc[m, 1] = float(g)
                        rgb_parc[m, 2] = float(b)

                    a = float(np.clip(parcel1_opacity_pct, 0, 100)) / 100.0
                    img[valid_mask] = img[valid_mask] * (1.0 - a) + rgb_parc[valid_mask] * a
            except Exception:
                pass
        if arr_parcel2 is not None:
            try:
                valid_mask = np.isfinite(arr_parcel2) & (arr_parcel2 > 0)

                if np.any(valid_mask):
                    labels = np.zeros(arr_parcel2.shape, dtype=np.int32)
                    labels[valid_mask] = np.round(arr_parcel2[valid_mask]).astype(np.int32)

                    lut = self._get_parcellation2_lut()
                    rgb_parc = np.zeros((H, W, 3), dtype=np.float32)

                    unique_labels = np.unique(labels[valid_mask])
                    for lab in unique_labels:
                        entry = lut.get(int(lab), None)
                        if entry is None:
                            continue
                        try:
                            _, (r, g, b) = entry
                        except Exception:
                            continue

                        m = labels == int(lab)
                        rgb_parc[m, 0] = float(r)
                        rgb_parc[m, 1] = float(g)
                        rgb_parc[m, 2] = float(b)

                    a = float(np.clip(parcel2_opacity_pct, 0, 100)) / 100.0
                    img[valid_mask] = img[valid_mask] * (1.0 - a) + rgb_parc[valid_mask] * a
            except Exception:
                pass
        return np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)

    def _build_pet_overlay(self, arr_pet):
        if arr_pet is None:
            return None, None

        finite = arr_pet[np.isfinite(arr_pet)]
        finite = finite[finite > 0]

        if finite.size == 0:
            return None, None

        pmin = float(self.spn_pet_min.value()) if self.spn_pet_min is not None else 15.0
        pmax = float(self.spn_pet_max.value()) if self.spn_pet_max is not None else 75.0
        gamma = self._get_pet_gamma_value()
        gamma = max(0.1, float(gamma))

        lo, hi = get_pet_window(finite, pmin, pmax)
        pet_norm = normalize_pet_slice(arr_pet, lo, hi, gamma=gamma)
        pet_rgb = pet_norm_to_colormap(pet_norm, self._pet_colormap_name)

        return pet_rgb, pet_norm

    def _compose_rgb_with_pet(self, base_rgb, pet_rgb, pet_norm, pet_opacity_pct: int):
        if base_rgb is None:
            return None

        img = base_rgb.copy()

        if pet_rgb is not None and pet_norm is not None:
            img = blend_pet_on_rgb(
                img,
                pet_rgb,
                pet_norm,
                alpha_scale=float(np.clip(pet_opacity_pct, 0, 100)) / 100.0,
            )

        img = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _get_local_contact_labels_visible(self, elec_id: int, n_contacts: int):
        vals = self._page_contact_labels_visible.get(int(elec_id))
        if not isinstance(vals, list) or len(vals) != int(n_contacts):
            vals = [False] * int(n_contacts)
            self._page_contact_labels_visible[int(elec_id)] = vals
        return vals

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
        self._schedule_refresh(slices=True, brain=False)

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
        self._schedule_refresh(slices=True, brain=False)

    def set_labels_visible(self, elec_id: int, visible: bool) -> None:
        try:
            elec = self.state.electrodes[int(elec_id)]
            n = len(elec.get("contacts_lps", []) or [])
        except Exception:
            return
        self._page_contact_labels_visible[int(elec_id)] = [bool(visible)] * n
        self._schedule_refresh(slices=True, brain=False)

    def set_contact_label_visible(self, elec_id: int, contact_idx: int, visible: bool) -> None:
        try:
            elec = self.state.electrodes[int(elec_id)]
            n = len(elec.get("contacts_lps", []) or [])
        except Exception:
            return
        vals = self._get_local_contact_labels_visible(int(elec_id), n)
        if 0 <= int(contact_idx) < len(vals):
            vals[int(contact_idx)] = bool(visible)
        self._schedule_refresh(slices=True, brain=False)

    def _project_contacts_for_cached_slice(
        self,
        elec: dict,
        center: np.ndarray,
        u: np.ndarray,
        w: np.ndarray,
        s_min: float,
        s_max: float,
        t_min: float,
        t_max: float,
        H: int,
        W: int,
    ):
        contacts = np.asarray(elec.get("contacts_lps", []) or [], dtype=np.float64)
        contact_rows = []
        contact_cols = []
        contact_names = []

        if contacts.ndim != 2 or contacts.shape[0] == 0:
            return contact_rows, contact_cols, contact_names

        try:
            elec_id = int(getattr(self.state, "electrodes", []).index(elec))
        except Exception:
            elec_id = None

        if elec_id is not None:
            contacts_visible = self._get_local_contacts_visible(elec_id, contacts.shape[0])
        else:
            contacts_visible = [True] * contacts.shape[0]

        try:
            elec_id = int(getattr(self.state, "electrodes", []).index(elec))
        except Exception:
            elec_id = None

        if elec_id is not None:
            contact_labels_visible = self._get_local_contact_labels_visible(
                elec_id, contacts.shape[0]
            )
        else:
            contact_labels_visible = [False] * contacts.shape[0]

        rel = contacts - center[None, :]
        s_proj = rel @ u
        t_proj = rel @ w

        rows = np.round((s_proj - s_min) / max(1e-9, (s_max - s_min)) * (H - 1)).astype(int)
        cols = np.round((t_proj - t_min) / max(1e-9, (t_max - t_min)) * (W - 1)).astype(int)

        elec_name = str(elec.get("name", "E"))

        for ci, (row, col) in enumerate(zip(rows, cols)):
            if not bool(contacts_visible[ci]):
                continue

            if 0 <= row < H and 0 <= col < W:
                contact_rows.append(int(row))
                contact_cols.append(int(col))
                if bool(contact_labels_visible[ci]):
                    contact_names.append(f"{elec_name}{ci + 1}")
                else:
                    contact_names.append("")

        return contact_rows, contact_cols, contact_names

    def _update_parcellation_contacts_table(self):
        tbl = getattr(self, "tbl_parcellation_contacts", None)
        if tbl is None:
            return

        try:
            from PySide6.QtWidgets import QTableWidgetItem
        except Exception:
            return

        p1_on = bool(self.chk_parcel1 is not None and self.chk_parcel1.isChecked())
        p2_on = bool(self.chk_parcel2 is not None and self.chk_parcel2.isChecked())

        # ---------------------------------------------------------
        # Choose the active parcellation.
        # Only ONE parcellation is allowed at a time in your UI.
        # The table always has exactly 3 columns:
        #   Contact | Label | Region
        # ---------------------------------------------------------
        if p1_on:
            parcel = getattr(self, "_parcel1_img", None)
            lookup_region = self._lookup_parcellation1_region
            header_label = "P1 label"
            header_region = "P1 region"

        elif p2_on:
            parcel = getattr(self, "_parcel2_img", None)
            lookup_region = self._lookup_parcellation2_region
            header_label = "P2 label"
            header_region = "P2 region"

        else:
            parcel = None
            lookup_region = None
            header_label = "Label"
            header_region = "Region"

        try:
            tbl.setColumnCount(3)
            tbl.setHorizontalHeaderLabels(["Contact", header_label, header_region])
        except Exception:
            pass

        # If no parcellation is checked/loaded, empty the table.
        if parcel is None or lookup_region is None:
            try:
                tbl.setRowCount(0)
            except Exception:
                pass
            return

        rows = []

        # Only electrodes currently displayed in the two oblique slices.
        selected = self._get_checked_electrode_names()
        selected_names = set(selected[:2])

        electrodes = getattr(self.state, "electrodes", []) or []

        for elec_id, elec in enumerate(electrodes):
            elec_name = str(elec.get("name", "") or "")
            if not elec_name:
                continue

            if elec_name not in selected_names:
                continue

            contacts = elec.get("contacts_lps", []) or []
            contacts_visible = self._get_local_contacts_visible(
                elec_id,
                len(contacts),
            )

            for i, p in enumerate(contacts):
                if i >= len(contacts_visible) or not bool(contacts_visible[i]):
                    continue

                try:
                    idx = parcel.TransformPhysicalPointToIndex(tuple(float(v) for v in p))
                    label = int(parcel.GetPixel(*idx))
                    _label_txt, region, region_rgb = lookup_region(label)
                except Exception:
                    label = -1
                    region = "Out of bounds"
                    region_rgb = (255, 255, 255)

                try:
                    elec_rgb = elec.get("color", (255, 255, 0))

                    if max(elec_rgb) <= 1.0:
                        elec_rgb = (
                            int(elec_rgb[0] * 255),
                            int(elec_rgb[1] * 255),
                            int(elec_rgb[2] * 255),
                        )
                    else:
                        elec_rgb = (
                            int(elec_rgb[0]),
                            int(elec_rgb[1]),
                            int(elec_rgb[2]),
                        )
                except Exception:
                    elec_rgb = (255, 255, 0)

                rows.append(
                    (
                        f"{elec_name}{i + 1}",
                        str(label),
                        str(region),
                        elec_rgb,
                        region_rgb,
                    )
                )

        try:
            tbl.setRowCount(len(rows))

            for r, row_data in enumerate(rows):
                contact, label, region, elec_rgb, region_rgb = row_data

                item_contact = QTableWidgetItem(contact)
                item_label = QTableWidgetItem(label)
                item_region = QTableWidgetItem(region)

                # Contact column background = electrode color.
                try:
                    ec = QColor(
                        int(elec_rgb[0]),
                        int(elec_rgb[1]),
                        int(elec_rgb[2]),
                    )
                    item_contact.setBackground(QBrush(ec))

                    if (int(elec_rgb[0]) + int(elec_rgb[1]) + int(elec_rgb[2])) < 382:
                        item_contact.setForeground(QBrush(QColor(255, 255, 255)))
                    else:
                        item_contact.setForeground(QBrush(QColor(0, 0, 0)))
                except Exception:
                    pass

                # Region column background = parcellation LUT color.
                try:
                    rc = QColor(
                        int(region_rgb[0]),
                        int(region_rgb[1]),
                        int(region_rgb[2]),
                    )
                    item_region.setBackground(QBrush(rc))

                    if (int(region_rgb[0]) + int(region_rgb[1]) + int(region_rgb[2])) < 382:
                        item_region.setForeground(QBrush(QColor(255, 255, 255)))
                    else:
                        item_region.setForeground(QBrush(QColor(0, 0, 0)))
                except Exception:
                    pass

                tbl.setItem(r, 0, item_contact)
                tbl.setItem(r, 1, item_label)
                tbl.setItem(r, 2, item_region)

            header = tbl.horizontalHeader()
            header.setStretchLastSection(True)
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.Stretch)

        except Exception:
            pass

    def set_parcellation1(self, img: sitk.Image | None, path: str | None = None) -> None:
        if img is None and path:
            try:
                img = sitk.ReadImage(path)
            except Exception:
                img = None

        self._parcel1_img = img

        # Keep a local copy of the LUT so it does not disappear if state changes later
        try:
            lut = getattr(self.state, "parcellation1_lut", {})
            if isinstance(lut, dict):
                self._parcel1_lut = dict(lut)
            else:
                self._parcel1_lut = {}
        except Exception:
            self._parcel1_lut = {}

        try:
            if self.chk_parcel1 is not None:
                self.chk_parcel1.blockSignals(True)
                self.chk_parcel1.setChecked(False)
                self.chk_parcel1.blockSignals(False)
        except Exception:
            pass

        try:
            if self.sld_parcel1 is not None:
                self.sld_parcel1.blockSignals(True)
                self.sld_parcel1.setValue(50)
                self.sld_parcel1.blockSignals(False)
            if self.spn_parcel1 is not None:
                self.spn_parcel1.blockSignals(True)
                self.spn_parcel1.setValue(50)
                self.spn_parcel1.blockSignals(False)
        except Exception:
            pass

        self._update_modality_controls_enabled_states()
        self._schedule_refresh(slices=True, brain=False)

    def set_parcellation2(self, img: sitk.Image | None, path: str | None = None) -> None:
        if img is None and path:
            try:
                img = sitk.ReadImage(path)
            except Exception:
                img = None

        self._parcel2_img = img

        try:
            lut = getattr(self.state, "parcellation2_lut", {})
            if isinstance(lut, dict):
                self._parcel2_lut = dict(lut)
            else:
                self._parcel2_lut = {}
        except Exception:
            self._parcel2_lut = {}

        try:
            if self.chk_parcel2 is not None:
                self.chk_parcel2.blockSignals(True)
                self.chk_parcel2.setChecked(False)
                self.chk_parcel2.blockSignals(False)
        except Exception:
            pass

        try:
            if self.sld_parcel2 is not None:
                self.sld_parcel2.blockSignals(True)
                self.sld_parcel2.setValue(50)
                self.sld_parcel2.blockSignals(False)
            if self.spn_parcel2 is not None:
                self.spn_parcel2.blockSignals(True)
                self.spn_parcel2.setValue(50)
                self.spn_parcel2.blockSignals(False)
        except Exception:
            pass

        self._update_modality_controls_enabled_states()
        self._schedule_refresh(slices=True, brain=False)

    def _get_parcellation1_lut(self):
        lut = getattr(self, "_parcel1_lut", None)
        if isinstance(lut, dict) and len(lut) > 0:
            return lut

        lut = getattr(self.state, "parcellation1_lut", None)
        if isinstance(lut, dict):
            return lut

        return {}

    def _lookup_parcellation1_region(self, label: int):
        lut = self._get_parcellation1_lut()

        entry = lut.get(int(label), None)
        if entry is None:
            # debug
            # print(f"[Parcellation LUT] missing label {label} | LUT size={len(lut)}")
            return str(label), "Unknown", (255, 255, 255)

        try:
            name, rgb = entry
            return str(label), str(name), rgb
        except Exception:
            return str(label), "Unknown", (255, 255, 255)

    def _get_parcellation2_lut(self):
        lut = getattr(self, "_parcel2_lut", None)
        if isinstance(lut, dict) and len(lut) > 0:
            return lut

        lut = getattr(self.state, "parcellation2_lut", None)
        if isinstance(lut, dict):
            return lut

        return {}

    def _lookup_parcellation2_region(self, label: int):
        lut = self._get_parcellation2_lut()

        entry = lut.get(int(label), None)
        if entry is None:
            return str(label), "Unknown", (255, 255, 255)

        try:
            name, rgb = entry
            return str(label), str(name), rgb
        except Exception:
            return str(label), "Unknown", (255, 255, 255)

    def _on_chk_parcel1_toggled(self, checked: bool):
        try:
            if checked and self.chk_parcel2 is not None and self.chk_parcel2.isChecked():
                self.chk_parcel2.blockSignals(True)
                self.chk_parcel2.setChecked(False)
                self.chk_parcel2.blockSignals(False)
        except Exception:
            pass

        self._enforce_parcellation_overlay_mode()
        self._update_modality_controls_enabled_states()
        self._schedule_refresh(slices=True, brain=False)

    def _on_chk_parcel2_toggled(self, checked: bool):
        try:
            if checked and self.chk_parcel1 is not None and self.chk_parcel1.isChecked():
                self.chk_parcel1.blockSignals(True)
                self.chk_parcel1.setChecked(False)
                self.chk_parcel1.blockSignals(False)
        except Exception:
            pass

        self._enforce_parcellation_overlay_mode()
        self._update_modality_controls_enabled_states()
        self._schedule_refresh(slices=True, brain=False)

    def _turn_off_parcellations_if_other_overlay_selected(self):
        ct_on = bool(self.chk_ct is not None and self.chk_ct.isChecked())
        pet_on = bool(self.chk_pet is not None and self.chk_pet.isChecked())
        sis_on = bool(self.chk_siscom is not None and self.chk_siscom.isChecked())

        if ct_on or pet_on or sis_on:
            try:
                if self.chk_parcel1 is not None and self.chk_parcel1.isChecked():
                    self.chk_parcel1.blockSignals(True)
                    self.chk_parcel1.setChecked(False)
                    self.chk_parcel1.blockSignals(False)
            except Exception:
                pass

            try:
                if self.chk_parcel2 is not None and self.chk_parcel2.isChecked():
                    self.chk_parcel2.blockSignals(True)
                    self.chk_parcel2.setChecked(False)
                    self.chk_parcel2.blockSignals(False)
            except Exception:
                pass
