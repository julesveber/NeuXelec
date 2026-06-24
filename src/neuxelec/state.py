from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt


@dataclass
class Volume:
    """
    3D volume used by the Reconstruction page.
    Convention: data is (X, Y, Z) float32.
    """

    data: np.ndarray
    path: str


# Custom data roles for electrode items in the shared Qt model
ROLE_KIND = Qt.UserRole + 1  # "electrode" | "meta" | "contact"
ROLE_ELEC_ID = Qt.UserRole + 2  # int
ROLE_CONTACT_INDEX = Qt.UserRole + 3  # int
ROLE_ROW_TYPE = Qt.UserRole + 4  # int (0-based) for contacts


class AppState:
    """
    Single source of truth for loaded file paths and derived data.

    Naming:
    - T1 is the FIXED image for all registrations.
    - "*_coreg_in_t1" are SimpleITK images resampled into T1 space.
    - We also keep a few backward-compatible aliases (ct_in_t1, etc.)
      to avoid breaking older code paths.
    """

    def __init__(self) -> None:

        # -------------------------
        # Project/session
        # -------------------------
        self.project_path: str | None = None
        self.patient_id: str = ""
        self.app_mode: str = "edit"
        self.view3d_saved_camera = None
        self.mni_electrode_sets = []

        # Anatomical 3D markers created from native patient slices.
        # Visibility is controlled from the 3D context menu, not from the edit dialog.
        self.markers: list[dict[str, Any]] = []

        # -------------------------
        # MNI / atlas space
        # -------------------------
        self.mni_space_name: str = "MNI152NLin2009cAsym"
        self.mni_template_path: str | None = None
        self.t1_to_mni_affine_path: str | None = None
        self.t1_to_mni_warp_path: str | None = None
        self.t1_to_mni_inverse_warp_path: str | None = None
        self.t1_to_mni_warped_path: str | None = None

        self.oblique_slice_saved_views = {}

        # Saved coreg file paths on disk
        self.t2_coreg_path: str | None = None
        self.ct_coreg_path: str | None = None
        self.pet_coreg_path: str | None = None
        self.ictal_spect_coreg_path: str | None = None
        self.interictal_spect_coreg_path: str | None = None
        self.siscom_coreg_path: str | None = None
        self.siscom_coreg_in_t1 = None

        self.siscom_path: str | None = None
        self.siscom_validated: bool = False
        self.siscom_diff_in_t1: Any | None = None
        self.siscom_z_in_t1: Any | None = None
        self.siscom_thr_in_t1: Any | None = None
        self.t1_source_path: str | None = None
        # T1 conform / resolution check
        # t1_path is the actual T1 used by NeuXelec.
        # If the original T1 was not 1 mm isotropic and the user accepted conversion,
        # t1_path points to the conformed 1 mm isotropic NIfTI.
        self.t1_was_conformed: bool = False
        self.t1_conformed_path: str | None = None
        self.t1_original_spacing: list[float] | None = None
        self.t1_conformed_spacing: list[float] | None = None
        self.t2_source_path: str | None = None
        self.ct_source_path: str | None = None
        self.pet_source_path: str | None = None
        self.ictal_spect_source_path: str | None = None
        self.interictal_spect_source_path: str | None = None

        self.brainmask_generated: bool = False
        self.brainmask_saved: bool = False
        self.brainmask_generated_path: str | None = None

        # -------------------------
        # Paths (as loaded from UI)
        # -------------------------
        self.t1_path: str | None = None
        self.t2_path: str | None = None
        self.ct_path: str | None = None
        self.pet_path: str | None = None
        self.ictal_spect_path: str | None = None
        self.interictal_spect_path: str | None = None

        self.parcel1_path: str | None = None
        self.parcel2_path: str | None = None

        self.lh_pial_path: str | None = None
        self.rh_pial_path: str | None = None

        # True once the user has answered the pial popup:
        # - No = use the selected surfaces as available
        # - Yes + successful coregistration = use the coregistered surfaces
        self.pial_surfaces_available: bool = False
        self.pial_surfaces_assume_lps: bool = True

        self.parcellation1_lut = None
        self.parcellation2_lut = None

        self.last_browse_dir = str(Path.home())

        # Global output directory for ANTs transforms / warped NIfTI (optional)
        self.transforms_dir: str | None = None

        # Brain mask (optional, generated on demand)
        self.brainmask_path: str | None = None
        self.brainmask_sitk: Any | None = None

        # -------------------------
        # Reconstruction
        # -------------------------
        self.volume: Volume | None = None  # CT volume shown in Reconstruction (native or coreg)

        self.parcel1_img = None
        self.parcel2_img = None
        # -------------------------
        # Electrodes (shared across pages)
        # -------------------------
        # Each electrode is a dict with at least:
        #  - name, hemisphere, ref, n, d_mm, contacts_lps, contacts_idx
        self.electrodes: list[dict[str, Any]] = []
        # Callbacks to notify GUI when electrodes list/visibility/colors change
        self._electrodes_changed_callbacks: list[Any] = []
        self.selected_electrode_id: int | None = None
        self.selected_contact_index: int | None = None

        # -------------------------
        # SimpleITK objects (kept as Any to avoid import cycles)
        # -------------------------
        self.t1_sitk: Any | None = None

        self.t2_coreg_in_t1: Any | None = None
        self.ct_coreg_in_t1: Any | None = None
        self.pet_coreg_in_t1: Any | None = None
        self.ictal_spect_coreg_in_t1: Any | None = None
        self.interictal_spect_coreg_in_t1: Any | None = None

        # Backward-compatible aliases (older names used in some modules)
        self.t2_in_t1: Any | None = None
        self.ct_in_t1: Any | None = None
        self.pet_in_t1: Any | None = None
        self.ictal_spect_in_t1: Any | None = None
        self.interictal_spect_in_t1: Any | None = None

        # -------------------------
        # Validation flags
        # -------------------------
        self.t2_validated: bool = False

        # Persistent CT validation:
        # True when a CT coregistered in T1 space has been visually validated
        # and can be restored for 3D View / Oblique Slice.
        self.ct_validated: bool = False

        # Session-only safety gate for Reconstruction:
        # even if a validated CT is restored from the project JSON,
        # Reconstruction remains blocked until the user checks the CT again
        # in the current NeuXelec session.
        self.ct_ready_for_reconstruction: bool = False

        self.pet_validated: bool = False
        self.ictal_spect_validated: bool = False
        self.interictal_spect_validated: bool = False

    def sync_aliases_from_new_names(self) -> None:
        """Optional helper if some legacy code reads ct_in_t1 instead of ct_coreg_in_t1."""
        self.t2_in_t1 = self.t2_coreg_in_t1
        self.ct_in_t1 = self.ct_coreg_in_t1
        self.pet_in_t1 = self.pet_coreg_in_t1
        self.ictal_spect_in_t1 = self.ictal_spect_coreg_in_t1
        self.interictal_spect_in_t1 = self.interictal_spect_coreg_in_t1

    # -------------------------
    # Electrodes model helpers
    # -------------------------

    def rebuild_electrodes_model(self) -> None:
        """Deprecated: GUI now uses QTreeWidget and rebuilds itself."""
        return

    def set_electrode_color(self, elec_id: int, rgb: tuple[int, int, int]) -> None:
        if elec_id < 0 or elec_id >= len(self.electrodes):
            return
        self.electrodes[elec_id]["color"] = tuple(int(c) for c in rgb)
        self.notify_electrodes_changed()

    def set_electrode_visibility(self, elec_id: int, visible: bool) -> None:
        if elec_id < 0 or elec_id >= len(self.electrodes):
            return
        self.electrodes[elec_id]["visible"] = bool(visible)
        # If hiding electrode, also hide all contacts; if showing, keep contacts_visible as is.
        if not visible:
            cv = self.electrodes[elec_id].get("contacts_visible")
            if isinstance(cv, list):
                self.electrodes[elec_id]["contacts_visible"] = [False] * len(cv)
        self.notify_electrodes_changed()

    def set_contact_visibility(self, elec_id: int, contact_index: int, visible: bool) -> None:
        if elec_id < 0 or elec_id >= len(self.electrodes):
            return
        elec = self.electrodes[elec_id]
        cv = elec.get("contacts_visible")
        if not isinstance(cv, list):
            return
        if contact_index < 0 or contact_index >= len(cv):
            return
        cv[contact_index] = bool(visible)
        # If any contact visible, electrode visible too
        if any(cv):
            elec["visible"] = True
        self.notify_electrodes_changed()

    def register_electrodes_changed(self, callback) -> None:
        """Register a callback called whenever electrodes data changes."""
        if callback not in self._electrodes_changed_callbacks:
            self._electrodes_changed_callbacks.append(callback)

    def notify_electrodes_changed(self) -> None:
        for cb in list(self._electrodes_changed_callbacks):
            try:
                cb()
            except Exception:
                # Never crash the app due to a UI refresh callback
                pass
