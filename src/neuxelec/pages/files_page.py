from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QLabel,
    QLineEdit,
    QProgressBar,
    QWidget,
)

from ..coregistration import (
    CoregResult,
    save_nifti,
)
from ..state import AppState, Volume
from ..ui.mri_filename_label_dialog import MRIFilenameLabelDialog
from ..ui.neuxelec_file_assignment_dialog import FileAssignmentDialog
from ..ui.neuxelec_message_dialog import NeuXelecMessageDialog
from ..ui.overlay_viewer import OverlayViewer
from ..ui.page_loading_overlay import PageLoadingOverlay
from ..ui.pial_coreg_dialog import PialCoregDialog
from ..utils.format_convert import convert_to_nifti_if_needed
from ..utils.t1_conform import ask_user_to_conform_t1_if_needed

Modality = Literal["T2", "CT", "PET", "ictalSPECT", "interictalSPECT"]


from ..workers import BrainMaskWorker, CoregWorker, IsoSurfaceWorker, SISCOMWorker


class FilesPage:
    """Files/Coreg page.

    Design choice (requested):
      - Coregistration is ALWAYS pairwise: T1 (fixed) + exactly ONE moving modality.
      - No preview when clicking "Perform" (fully automatic).
      - Visual check + optional manual refinement happens in "Check coregistration".
    """

    def __init__(
        self,
        ui_root: QObject,
        state: AppState,
        on_ct_loaded: Callable[[], None] | None = None,
        on_ct_updated: Callable[[], None] | None = None,
    ):
        self.ui = ui_root
        self.state = state
        self.on_ct_loaded = on_ct_loaded
        self.on_ct_updated = on_ct_updated

        # Same animated loading overlay as 3D View / Oblique Slice.
        # Used during DICOM/NIfTI conversion and bulk import.
        self._page_widget = self.ui.findChild(QWidget, "pageFiles")
        self._loading_overlay = (
            PageLoadingOverlay(
                self._page_widget,
                "FILES / COREGISTRATION",
                "Preparing import",
            )
            if self._page_widget is not None
            else None
        )

        # --- UI widgets (paths) ---
        self.le_t1 = self.ui.findChild(QLineEdit, "le_FilesCoreg_loadT1")
        self.le_t2 = self.ui.findChild(QLineEdit, "le_FilesCoreg_loadT2")
        self.le_ct = self.ui.findChild(QLineEdit, "le_FilesCoreg_loadCT")
        self.le_pet = self.ui.findChild(QLineEdit, "le_FilesCoreg_loadPET")
        self.le_ictal = self.ui.findChild(QLineEdit, "le_FilesCoreg_loadictalSPECT")
        self.le_interictal = self.ui.findChild(QLineEdit, "le_FilesCoreg_loadinterictalSPECT")
        self.le_siscom = self.ui.findChild(QLineEdit, "le_FilesCoreg_loadSISCOM")  # NEW
        self.le_parcel1 = self.ui.findChild(QLineEdit, "le_FilesCoreg_loadParcel1")
        self.le_parcel2 = self.ui.findChild(QLineEdit, "le_FilesCoreg_loadParcel2")
        self.le_lh_pial = self.ui.findChild(QLineEdit, "le_FilesCoreg_lhpial")
        self.le_rh_pial = self.ui.findChild(QLineEdit, "le_FilesCoreg_rhpial")

        for le in (
            self.le_t1,
            self.le_t2,
            self.le_ct,
            self.le_pet,
            self.le_ictal,
            self.le_interictal,
            self.le_siscom,  # NEW
            self.le_parcel1,
            self.le_parcel2,
            self.le_lh_pial,
            self.le_rh_pial,
        ):
            if le is not None:
                le.setReadOnly(True)
                le.setPlaceholderText("No file selected")

        # --- Status pills / compact workflow labels (new UI) ---
        self._status_pills: dict[str, QLabel] = {
            "T1": self.ui.findChild(QLabel, "pill_FilesCoreg_T1"),
            "T2": self.ui.findChild(QLabel, "pill_FilesCoreg_T2"),
            "CT": self.ui.findChild(QLabel, "pill_FilesCoreg_CT"),
            "PET": self.ui.findChild(QLabel, "pill_FilesCoreg_PET"),
            "ictalSPECT": self.ui.findChild(QLabel, "pill_FilesCoreg_ictalSPECT"),
            "interictalSPECT": self.ui.findChild(QLabel, "pill_FilesCoreg_interictalSPECT"),
            "SISCOM": self.ui.findChild(QLabel, "pill_FilesCoreg_SISCOM"),
            "Parcel1": self.ui.findChild(QLabel, "pill_FilesCoreg_Parcel1"),
            "Parcel2": self.ui.findChild(QLabel, "pill_FilesCoreg_Parcel2"),
            "LHPial": self.ui.findChild(QLabel, "pill_FilesCoreg_LHPial"),
            "RHPial": self.ui.findChild(QLabel, "pill_FilesCoreg_RHPial"),
            "BrainMask": self.ui.findChild(QLabel, "pill_FilesCoreg_BrainMask"),
        }
        self._status_texts: dict[str, QLabel] = {
            "T1": self.ui.findChild(QLabel, "status_FilesCoreg_T1"),
            "T2": self.ui.findChild(QLabel, "status_FilesCoreg_T2"),
            "CT": self.ui.findChild(QLabel, "status_FilesCoreg_CT"),
            "PET": self.ui.findChild(QLabel, "status_FilesCoreg_PET"),
            "ictalSPECT": self.ui.findChild(QLabel, "status_FilesCoreg_ictalSPECT"),
            "interictalSPECT": self.ui.findChild(QLabel, "status_FilesCoreg_interictalSPECT"),
            "SISCOM": self.ui.findChild(QLabel, "status_FilesCoreg_SISCOM"),
            "Parcel1": self.ui.findChild(QLabel, "status_FilesCoreg_Parcel1"),
            "Parcel2": self.ui.findChild(QLabel, "status_FilesCoreg_Parcel2"),
            "LHPial": self.ui.findChild(QLabel, "status_FilesCoreg_LHPial"),
            "RHPial": self.ui.findChild(QLabel, "status_FilesCoreg_RHPial"),
            "BrainMask": self.ui.findChild(QLabel, "status_FilesCoreg_BrainMask"),
        }

        # --- Load buttons ---
        self.btn_load_t1 = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_loadT1")
        self.btn_load_t2 = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_loadT2")
        self.btn_load_ct = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_loadCT")
        self.btn_load_pet = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_loadPET")
        self.btn_load_ictal = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_loadictalSPECT")
        self.btn_load_interictal = self.ui.findChild(
            QAbstractButton, "btn_FilesCoreg_loadinterictalSPECT"
        )
        self.btn_load_siscom = self.ui.findChild(
            QAbstractButton, "btn_FilesCoreg_loadSISCOM"
        )  # NEW
        self.btn_load_parcel1 = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_loadParcel1")
        self.btn_load_parcel2 = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_loadParcel2")
        self.btn_load_lhpial = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_lhpial")
        self.btn_load_rhpial = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_rhpial")

        # --- Bulk load buttons (new workflow UI) ---
        self.btn_load_imaging = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_loadImaging")
        self.btn_load_parcellations = self.ui.findChild(
            QAbstractButton, "btn_FilesCoreg_loadParcellations"
        )
        self.btn_load_surfaces = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_loadSurfaces")

        if self.btn_load_imaging:
            self.btn_load_imaging.clicked.connect(self.load_imaging_bundle)
        if self.btn_load_parcellations:
            self.btn_load_parcellations.clicked.connect(self.load_parcellation_bundle)
        if self.btn_load_surfaces:
            self.btn_load_surfaces.clicked.connect(self.load_surfaces_bundle)

        if self.btn_load_t1:
            self.btn_load_t1.clicked.connect(self.load_t1)
        if self.btn_load_t2:
            self.btn_load_t2.clicked.connect(self.load_t2)
        if self.btn_load_ct:
            self.btn_load_ct.clicked.connect(self.load_ct)
        if self.btn_load_pet:
            self.btn_load_pet.clicked.connect(self.load_pet)
        if self.btn_load_ictal:
            self.btn_load_ictal.clicked.connect(self.load_ictal_spect)
        if self.btn_load_interictal:
            self.btn_load_interictal.clicked.connect(self.load_interictal_spect)
        if self.btn_load_siscom:
            self.btn_load_siscom.clicked.connect(self.load_siscom)  # NEW
        if self.btn_load_parcel1:
            self.btn_load_parcel1.clicked.connect(self.load_parcellation1)
        if self.btn_load_parcel2:
            self.btn_load_parcel2.clicked.connect(self.load_parcellation2)
        if self.btn_load_lhpial:
            self.btn_load_lhpial.clicked.connect(self.load_lh_pial)

        if self.btn_load_rhpial:
            self.btn_load_rhpial.clicked.connect(self.load_rh_pial)

        # --- Checkboxes ---
        self.chk_t1 = self.ui.findChild(QCheckBox, "chk_FilesCoreg_T1")
        self.chk_t2 = self.ui.findChild(QCheckBox, "chk_FilesCoreg_T2")
        self.chk_ct = self.ui.findChild(QCheckBox, "chk_FilesCoreg_CT")
        self.chk_pet = self.ui.findChild(QCheckBox, "chk_FilesCoreg_PET")
        self.chk_ictal = self.ui.findChild(QCheckBox, "chk_FilesCoreg_ictalSPECT")
        self.chk_interictal = self.ui.findChild(QCheckBox, "chk_FilesCoreg_interictalSPECT")

        self._moving_modality_group = QButtonGroup(self.ui)
        self._moving_modality_group.setExclusive(True)

        for cb in (self.chk_t2, self.chk_ct, self.chk_pet, self.chk_ictal, self.chk_interictal):
            if cb is not None:
                self._moving_modality_group.addButton(cb)

        for cb in (self.chk_t2, self.chk_ct, self.chk_pet, self.chk_ictal, self.chk_interictal):
            if cb is not None:
                cb.toggled.connect(self._update_buttons)

        if self.chk_t1 is not None:
            self.chk_t1.toggled.connect(self._force_t1_checked)

        # --- Progress bars ---
        self.coreg_bar = self.ui.findChild(QProgressBar, "progressBarCoreg")
        if self.coreg_bar is not None:
            self.coreg_bar.setVisible(False)
            self.coreg_bar.setRange(0, 100)
            self.coreg_bar.setValue(0)
        # Smooth estimated progress for ANTs coregistration.
        # ANTs itself only reports start/end in the current implementation,
        # so we animate the bar during the expected ~60 s runtime.
        self._coreg_progress_timer = QTimer()
        self._coreg_progress_timer.setInterval(250)
        self._coreg_progress_timer.timeout.connect(self._tick_coreg_progress)

        self._coreg_progress_running = False
        self._coreg_progress_start_time = 0.0
        self._coreg_progress_modality = None
        self._coreg_progress_duration_s = 180.0
        self._coreg_progress_max_before_done = 98

        self.brainmask_bar = self.ui.findChild(QProgressBar, "progressBarBrainmask")
        if self.brainmask_bar is not None:
            self.brainmask_bar.setVisible(False)
            self.brainmask_bar.setRange(0, 100)
            self.brainmask_bar.setValue(0)
        # Smooth estimated progress for BrainMask generation.
        # The real worker may not provide continuous progress, so we animate
        # linearly during the expected ~4 min runtime.
        self._brainmask_progress_timer = QTimer()
        self._brainmask_progress_timer.setInterval(250)
        self._brainmask_progress_timer.timeout.connect(self._tick_brainmask_progress)

        self._brainmask_progress_running = False
        self._brainmask_progress_start_time = 0.0
        self._brainmask_progress_duration_s = 240.0
        self._brainmask_progress_max_before_done = 98
        self._brainmask_progress_label = "Brain mask"
        self.siscom_bar = self.ui.findChild(QProgressBar, "progressBarSISCOM")
        if self.siscom_bar is not None:
            self.siscom_bar.setVisible(False)
            self.siscom_bar.setRange(0, 100)
            self.siscom_bar.setValue(0)

        # --- Coreg buttons ---
        self.btn_perform = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_performCoreg")
        self.btn_check_coreg = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_checkCoreg")
        if self.btn_perform:
            self.btn_perform.clicked.connect(self.perform_coreg)
        if self.btn_check_coreg:
            self.btn_check_coreg.clicked.connect(self.check_coreg)

        # --- Brain mask button ---
        self.btn_brainmask = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_Brainmask")
        if self.btn_brainmask:
            self.btn_brainmask.clicked.connect(self.generate_brain_mask)
            self.btn_brainmask.setEnabled(False)

        # --- Save Brainmask button ---
        self.btn_save_brainmask = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_SaveBrainmask")
        if self.btn_save_brainmask is not None:
            self.btn_save_brainmask.clicked.connect(self.save_brainmask)
            self.btn_save_brainmask.setEnabled(False)

        # --- Iso-surface buttons (NEW) ---
        self.btn_iso_surface = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_isoSurface")
        self.btn_save_iso_surface = self.ui.findChild(
            QAbstractButton, "btn_FilesCoreg_SaveIsoSurface"
        )

        if self.btn_iso_surface is not None:
            self.btn_iso_surface.clicked.connect(self.generate_iso_surface)
            self.btn_iso_surface.setEnabled(False)

        if self.btn_save_iso_surface is not None:
            self.btn_save_iso_surface.clicked.connect(self.save_iso_surface)
            self.btn_save_iso_surface.setEnabled(False)

        # --- NEW: Load Brain Mask / Load Iso-Surface (UI vNext) ---
        self.btn_load_brainmask = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_loadBrainMask")
        self.le_load_brainmask = self.ui.findChild(QLineEdit, "le_FilesCoreg_loadBrainMask")

        self.btn_load_isosurface = self.ui.findChild(
            QAbstractButton, "btn_FilesCoreg_loadIsoSurface"
        )
        self.le_load_isosurface = self.ui.findChild(QLineEdit, "le_FilesCoreg_loadIsoSurface")

        if self.btn_load_brainmask is not None:
            self.btn_load_brainmask.clicked.connect(self.load_brainmask)

        if self.btn_load_isosurface is not None:
            self.btn_load_isosurface.clicked.connect(self.load_iso_surface)

        # --- SISCOM buttons ---
        self.btn_perf_siscom = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_PerfSISCOM")
        self.btn_check_siscom = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_CheckSISCOM")
        self.btn_save_siscom = self.ui.findChild(
            QAbstractButton, "btn_FilesCoreg_SaveSISCOM"
        )  # NEW
        if self.btn_perf_siscom:
            self.btn_perf_siscom.clicked.connect(self.perform_siscom)
            self.btn_perf_siscom.setEnabled(False)
        if self.btn_check_siscom:
            self.btn_check_siscom.clicked.connect(self.check_siscom)
            self.btn_check_siscom.setEnabled(False)
        if self.btn_save_siscom:
            self.btn_save_siscom.clicked.connect(self.save_siscom)  # NEW
            self.btn_save_siscom.setEnabled(False)

        # --- Save buttons ---
        self.btn_save_t2 = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_saveT2")
        self.btn_save_ct = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_saveCT")
        self.btn_save_pet = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_savePET")
        self.btn_save_ictal = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_saveIctalSPECT")
        self.btn_save_interictal = self.ui.findChild(
            QAbstractButton, "btn_FilesCoreg_saveInterIctalSPECT"
        )
        self.btn_save_all = self.ui.findChild(QAbstractButton, "btn_FilesCoreg_saveAll")

        if self.btn_save_t2:
            self.btn_save_t2.clicked.connect(lambda: self.save_coreg("T2"))
        if self.btn_save_ct:
            self.btn_save_ct.clicked.connect(lambda: self.save_coreg("CT"))
        if self.btn_save_pet:
            self.btn_save_pet.clicked.connect(lambda: self.save_coreg("PET"))
        if self.btn_save_ictal:
            self.btn_save_ictal.clicked.connect(lambda: self.save_coreg("ictalSPECT"))
        if self.btn_save_interictal:
            self.btn_save_interictal.clicked.connect(lambda: self.save_coreg("interictalSPECT"))
        if self.btn_save_all:
            self.btn_save_all.clicked.connect(self.save_all_coreg_validated)

        # --- State init (if missing) ---
        for name in (
            "t1_path",
            "t2_path",
            "ct_path",
            "pet_path",
            "ictal_spect_path",
            "interictal_spect_path",
            "t1_source_path",
            "t2_source_path",
            "ct_source_path",
            "pet_source_path",
            "ictal_spect_source_path",
            "interictal_spect_source_path",
            "parcel1_path",
            "parcel2_path",
            "parcel1_img",
            "parcellation1_lut",
            "parcellation2_lut",
            "t1_sitk",
            "t2_coreg_in_t1",
            "ct_coreg_in_t1",
            "pet_coreg_in_t1",
            "ictal_spect_coreg_in_t1",
            "interictal_spect_coreg_in_t1",
            "t2_validated",
            "ct_validated",
            "pet_validated",
            "ictal_spect_validated",
            "interictal_spect_validated",
            "brainmask_path",
            "brainmask_generated",
            "brainmask_saved",
            "brainmask_generated_path",
            "brainmask_sitk",
            "siscom_diff_in_t1",
            "siscom_z_in_t1",
            "siscom_thr_in_t1",
            "siscom_path",
            "siscom_validated",
            "iso_surface_mesh",
            "lh_pial_path",
            "rh_pial_path",
            "pial_surfaces_available",
            "iso_surface_saved_path",
            "mri1_filename_label",
            "mri2_filename_label",
        ):
            if not hasattr(self.state, name):
                setattr(self.state, name, None)

        if getattr(self.state, "parcellation1_lut", None) is None:
            self.state.parcellation1_lut = {}

        if getattr(self.state, "parcellation2_lut", None) is None:
            self.state.parcellation2_lut = {}

        if self.state.siscom_validated is None:
            self.state.siscom_validated = False

        if getattr(self.state, "iso_surface_mesh", None) is None:
            self.state.iso_surface_mesh = None
        if getattr(self.state, "iso_surface_saved_path", None) is None:
            self.state.iso_surface_saved_path = None

        if getattr(self.state, "pial_surfaces_available", None) is None:
            self.state.pial_surfaces_available = False

        if getattr(self.state, "mri1_filename_label", None) is None:
            self.state.mri1_filename_label = "MRI1"

        if getattr(self.state, "mri2_filename_label", None) is None:
            self.state.mri2_filename_label = "MRI2"
        self._last_result: CoregResult | None = None
        self._last_modality: Modality | None = None
        self._pending_matrix_save_path: str | None = None

        self._update_buttons()

    def _display_modality_name(self, modality: str | None) -> str:
        """
        User-facing modality names.

        Important:
        Keep internal keys as T1 / T2 to avoid breaking the code.
        Only labels/messages shown to the user become MRI 1 / MRI 2.
        """
        names = {
            "T1": "MRI 1",
            "T2": "MRI 2",
            "CT": "CT",
            "PET": "PET",
            "ictalSPECT": "Ictal SPECT",
            "interictalSPECT": "Interictal SPECT",
            "SISCOM": "SISCOM",
        }

        return names.get(str(modality), str(modality))

    def _safe_filename_part(self, value: object, fallback: str = "Unknown") -> str:
        """
        Convert any user text into a safe filename component.
        """
        text = str(value or "").strip()

        if not text:
            text = fallback

        safe = []
        for ch in text:
            if ch.isalnum() or ch in ("-", "_"):
                safe.append(ch)
            elif ch in (" ", ".", "/", "\\", ":", ";", ","):
                safe.append("_")

        out = "".join(safe).strip("_")

        while "__" in out:
            out = out.replace("__", "_")

        return out or fallback

    def _patient_id_for_filename(self) -> str:
        """
        Return patient ID used in exported filenames.

        Tries common state attributes first, then falls back to PatientID.
        """
        for attr in (
            "patient_id",
            "patientID",
            "current_patient_id",
            "subject_id",
            "project_patient_id",
        ):
            value = getattr(self.state, attr, None)
            if value:
                return self._safe_filename_part(value, fallback="PatientID")

        return "PatientID"

    def _mri1_filename_label(self) -> str:
        return self._safe_filename_part(
            getattr(self.state, "mri1_filename_label", None) or "MRI1",
            fallback="MRI1",
        )

    def _mri2_filename_label(self) -> str:
        return self._safe_filename_part(
            getattr(self.state, "mri2_filename_label", None) or "MRI2",
            fallback="MRI2",
        )

    def _filename_label_for_modality(self, modality: str) -> str:
        modality = str(modality)

        if modality == "T1":
            return self._mri1_filename_label()

        if modality == "T2":
            return self._mri2_filename_label()

        return self._safe_filename_part(
            self._display_modality_name(modality).replace(" ", ""),
            fallback=modality,
        )

    def _default_filename(self, name: str, suffix: str) -> str:
        patient_id = self._patient_id_for_filename()
        name = self._safe_filename_part(name, fallback="output")
        return f"{patient_id}_{name}{suffix}"

    def _guess_mri_label_from_path(self, path: str | None, fallback: str) -> str:
        """
        Guess a clean filename label from the loaded MRI filename.

        Examples:
            *_T1_* / *_MPRAGE_* -> T1 or MPRAGE
            *_T2_* / *_FLAIR_*  -> T2 or FLAIR
        """
        try:
            name = Path(str(path)).name.lower()
        except Exception:
            return fallback

        if "mprage" in name:
            return "MPRAGE"
        if "bravo" in name:
            return "BRAVO"
        if "3dt1" in name or "3d_t1" in name:
            return "3DT1"
        if "flair" in name:
            return "FLAIR"
        if "t2" in name:
            return "T2"
        if "t1" in name:
            return "T1"

        return fallback

    def _ask_mri_filename_labels(
        self,
        role_paths: list[tuple[str, str]],
    ) -> None:
        """
        Ask MRI filename labels in a single NeuXelec popup.

        Examples:
            [(\"T1\", path)]              -> one MRI 1 row
            [(\"T2\", path)]              -> one MRI 2 row
            [(\"T1\", path), (\"T2\", path)] -> one popup with two rows
        """
        items = []
        seen = set()

        for role, source_path in role_paths:
            role = str(role)

            if role not in ("T1", "T2"):
                continue

            if role in seen:
                continue

            seen.add(role)

            if role == "T1":
                attr = "mri1_filename_label"
                fallback = "MRI1"
                display_name = "MRI 1"
            else:
                attr = "mri2_filename_label"
                fallback = "MRI2"
                display_name = "MRI 2"

            current = getattr(self.state, attr, None)
            guessed = self._guess_mri_label_from_path(source_path, fallback)

            default = current if current not in (None, "", fallback) else guessed

            try:
                source_name = Path(str(source_path)).name
            except Exception:
                source_name = str(source_path or "")

            items.append(
                {
                    "role": role,
                    "display_name": display_name,
                    "default": str(default),
                    "source_name": source_name,
                }
            )

        if not items:
            return

        values = MRIFilenameLabelDialog.get_labels(
            items,
            parent=self._dialog_parent(),
        )

        if values is None:
            return

        for role, text in values.items():
            if role == "T1":
                clean = self._safe_filename_part(text, fallback="MRI1")
                self.state.mri1_filename_label = clean

            elif role == "T2":
                clean = self._safe_filename_part(text, fallback="MRI2")
                self.state.mri2_filename_label = clean

    def _ask_mri_filename_label(
        self,
        role: str,
        source_path: str | None = None,
    ) -> None:
        """
        Compatibility helper for single MRI loading.
        Internally uses the one-window multi-row dialog.
        """
        self._ask_mri_filename_labels([(str(role), str(source_path or ""))])

    # ------------------------
    # Helper: 3D view bridge
    # ------------------------
    def _view3d(self):
        return getattr(self.state, "view3d_page", None)

    def _dialog_parent(self):
        """
        Return the main NeuXelec window as parent for all dialogs opened
        from the Files / Coregistration page.
        """
        try:
            return self.ui.window()
        except Exception:
            return None

    def _show_files_loading(
        self,
        message: str = "Preparing import",
        progress: float = 0.10,
    ) -> None:
        """
        Show the same animated NeuXelec logo overlay used by 3D View.

        This is displayed during DICOM/NIfTI conversions and bulk imports.
        """
        try:
            if self._loading_overlay is None:
                return

            if not self._loading_overlay.isVisible():
                self._loading_overlay.begin(message)
            else:
                self._loading_overlay.set_progress(progress, message)

            QApplication.processEvents()

        except Exception:
            pass

    def _update_files_loading(
        self,
        progress: float,
        message: str,
    ) -> None:
        try:
            if self._loading_overlay is not None:
                self._loading_overlay.set_progress(progress, message)
                QApplication.processEvents()
        except Exception:
            pass

    def _complete_files_loading(self) -> None:
        try:
            if self._loading_overlay is not None:
                self._loading_overlay.complete()
                QApplication.processEvents()
        except Exception:
            pass

    def _cancel_files_loading(self) -> None:
        try:
            if self._loading_overlay is not None:
                self._loading_overlay.cancel()
                QApplication.processEvents()
        except Exception:
            pass

    def _load_freesurfer_lut_dict(
        self, lut_path: Path
    ) -> dict[int, tuple[str, tuple[int, int, int]]]:
        lut: dict[int, tuple[str, tuple[int, int, int]]] = {}

        try:
            with open(lut_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue

                    parts = s.split()
                    if len(parts) < 6:
                        continue

                    try:
                        idx = int(parts[0])
                        name = parts[1]
                        r = int(parts[2])
                        g = int(parts[3])
                        b = int(parts[4])
                        lut[idx] = (name, (r, g, b))
                    except Exception:
                        continue
        except Exception:
            return {}

        return lut

    def _enable_3d_checkbox(self, object_name: str, enabled: bool) -> None:
        cb = self.ui.findChild(QCheckBox, object_name)
        if cb is None:
            return
        cb.setEnabled(bool(enabled))
        if not enabled:
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)

    def _pick_multiple_dicom_folders(self, caption: str) -> list[str]:
        """
        Select one or several DICOM folders while keeping the native Windows
        folder picker.

        The native Windows folder picker allows one folder at a time, so NeuXelec
        reopens it until the user clicks Done.
        """
        folders: list[str] = []
        browse_dir = self.state.last_browse_dir or str(Path.home())

        while True:
            folder = QFileDialog.getExistingDirectory(
                self._dialog_parent(),
                caption if not folders else "Select another DICOM folder",
                browse_dir,
                QFileDialog.ShowDirsOnly,
            )

            if not folder:
                break

            folder = str(folder)

            if folder not in folders:
                folders.append(folder)

            try:
                self.state.last_browse_dir = folder
                browse_dir = folder
            except Exception:
                pass

            add_another = NeuXelecMessageDialog.question(
                self._dialog_parent(),
                "DICOM folders",
                "Do you want to add another DICOM folder?",
                accept_text="Add another",
                reject_text="Done",
            )

            if not add_another:
                break

        return folders

    # ------------------------
    # Bulk import workflow (new Files/Coreg UI)
    # ------------------------
    def _pick_multiple_files(
        self,
        caption: str,
        file_filter: str,
        allow_dicom_folders: bool = True,
        files_button_text: str = "Files (.nii / .mgz / .dcm)",
    ) -> list[str]:
        """
        Bulk picker for the redesigned Files/Coreg page.

        Parameters
        ----------
        allow_dicom_folders:
            True for imaging imports.
            False for parcellations and surfaces.

        files_button_text:
            Text displayed on the file-selection button.
        """
        browse_dir = self.state.last_browse_dir or str(Path.home())

        choices = [
            (
                "files",
                files_button_text,
                True,
            ),
        ]

        if allow_dicom_folders:
            choices.append(
                (
                    "dicom_folders",
                    "DICOM folder(s)",
                    False,
                )
            )

        input_choice = NeuXelecMessageDialog.choice(
            self._dialog_parent(),
            "Select Input Type",
            "What do you want to load?",
            choices=choices,
            cancel_text="Cancel",
        )

        if input_choice == "files":
            paths, _ = QFileDialog.getOpenFileNames(
                self._dialog_parent(),
                caption,
                browse_dir,
                file_filter,
            )

            if paths:
                try:
                    self.state.last_browse_dir = str(Path(paths[0]).parent)
                except Exception:
                    pass

            return [str(p) for p in paths]

        if input_choice == "dicom_folders":
            return self._pick_multiple_dicom_folders(
                caption + " - select one or several DICOM folders"
            )

        return []

    def _is_dicom_source(self, path: str) -> bool:
        """
        Return True if the selected source is probably a DICOM input.

        In NeuXelec bulk import, DICOM input is mainly a folder.
        Single .dcm / .ima files are also treated as DICOM sources.
        """
        try:
            p = Path(path)

            if p.is_dir():
                return True

            return p.suffix.lower() in (".dcm", ".ima")

        except Exception:
            return False

    def _ask_common_nifti_output_dir(self, assignments: list[dict]) -> Path | None:
        """
        Ask only once where all converted NIfTI files should be saved.

        Only needed if at least one assigned source is DICOM.
        """
        has_dicom = any(
            self._is_dicom_source(str(item.get("path", "")))
            for item in assignments
            if item.get("path", "") and item.get("role", "IGNORE") != "IGNORE"
        )

        if not has_dicom:
            return None

        start_dir = self.state.last_browse_dir or str(Path.home())

        out_dir = QFileDialog.getExistingDirectory(
            self._dialog_parent(),
            "Choose folder where all converted NIfTI files will be saved",
            start_dir,
            QFileDialog.ShowDirsOnly,
        )

        if not out_dir:
            return None

        try:
            self.state.last_browse_dir = str(out_dir)
        except Exception:
            pass

        return Path(out_dir)

    def _nifti_filename_for_role(self, role: str) -> str:
        """
        Return the NIfTI filename used for bulk DICOM conversion.
        """
        role = str(role)

        names = {
            "T1": self._default_filename(self._mri1_filename_label(), ".nii"),
            "T2": self._default_filename(self._mri2_filename_label(), ".nii"),
            "CT": self._default_filename("CT", ".nii"),
            "PET": self._default_filename("PET", ".nii"),
            "ictalSPECT": self._default_filename("ictalSPECT", ".nii"),
            "interictalSPECT": self._default_filename("interictalSPECT", ".nii"),
            "SISCOM": self._default_filename("SISCOM", ".nii"),
            "PARCEL1": self._default_filename("Parcellation1", ".nii"),
            "PARCEL2": self._default_filename("Parcellation2", ".nii"),
        }

        return names.get(role, self._default_filename(role, ".nii"))

    def _unique_output_path(self, output_dir: Path, filename: str) -> Path:
        """
        Avoid overwriting if the same modality is accidentally assigned twice.
        First file keeps modality.nii, next ones become modality_2.nii, etc.
        """
        output_path = output_dir / filename

        if not output_path.exists():
            return output_path

        stem = output_path.stem
        suffix = output_path.suffix

        i = 2
        while True:
            candidate = output_dir / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                return candidate
            i += 1

    def _convert_to_nifti_for_import(
        self,
        source_path: str,
        role: str,
        output_dir: Path | None = None,
    ) -> str:
        """
        Convert source_path to NIfTI.

        If output_dir is provided and source_path is DICOM, the NIfTI is saved
        automatically as modality.nii in output_dir without asking again.
        """
        if output_dir is not None and self._is_dicom_source(source_path):
            output_path = self._unique_output_path(
                output_dir,
                self._nifti_filename_for_role(role),
            )

            nifti_path = convert_to_nifti_if_needed(
                source_path,
                parent=self._dialog_parent(),
                output_path=str(output_path),
            )

            return str(nifti_path)

        nifti_path = convert_to_nifti_if_needed(
            source_path,
            parent=self._dialog_parent(),
        )

        return str(nifti_path)

    @staticmethod
    def _guess_imaging_role(path: str) -> str:
        name = Path(path).name.lower()
        full = str(path).lower()
        if "siscom" in name:
            return "SISCOM"
        if "interictal" in name or "inter_ictal" in name or "baseline" in name:
            return "interictalSPECT"
        if "ictal" in name or "spect_ict" in name:
            return "ictalSPECT"
        if "pet" in name:
            return "PET"
        if (
            "ct" in name
            or "scanner" in name
            or "postop" in name
            or "post-op" in name
            or "post" in name
        ):
            return "CT"
        if "t2" in name or "flair" in name:
            return "T2"

        if "t1" in name or "mprage" in name or "bravo" in name or "3dt1" in full:
            return "T1"
        return "IGNORE"

    @staticmethod
    def _guess_parcellation_role(path: str) -> str:
        """
        Suggest only Parcellation 1 or Parcellation 2.

        LUT files are no longer assigned manually in the bulk-import popup.
        FreeSurfer LUTs are detected and loaded automatically when possible.
        """
        name = Path(path).name.lower()

        if "parcel2" in name or "parcellation2" in name or "_p2" in name or "p2_" in name:
            return "PARCEL2"

        if "parcel1" in name or "parcellation1" in name or "_p1" in name or "p1_" in name:
            return "PARCEL1"

        if "aparc" in name or "aseg" in name or "wmparc" in name:
            return "PARCEL1"

        return "PARCEL1"

    @staticmethod
    def _guess_surface_role(path: str) -> str:
        name = Path(path).name.lower()
        if "brainmask" in name or "brain_mask" in name or "mask" in name:
            return "BRAINMASK"
        if name.startswith("lh.") or "left" in name or "_lh" in name:
            return "LH_PIAL"
        if name.startswith("rh.") or "right" in name or "_rh" in name:
            return "RH_PIAL"
        return "IGNORE"

    def load_imaging_bundle(self) -> None:
        paths = self._pick_multiple_files(
            "Load imaging files",
            "Medical images (*.nii *.nii.gz *.mgz *.mgh *.dcm *.ima);;All files (*.*)",
        )
        if not paths:
            return

        roles = [
            ("T1", "MRI 1 / fixed reference"),
            ("T2", "MRI 2"),
            ("CT", "CT post-implant"),
            ("PET", "PET"),
            ("ictalSPECT", "Ictal SPECT"),
            ("interictalSPECT", "Interictal SPECT"),
            ("SISCOM", "SISCOM already in MRI 1 space"),
            ("IGNORE", "Ignore"),
        ]
        suggestions = {p: self._guess_imaging_role(p) for p in paths}
        assignments = FileAssignmentDialog.get_assignments(
            paths,
            roles,
            suggestions=suggestions,
            parent=self._dialog_parent(),
            title="Assign imaging files",
            subtitle="Select the correct modality for each file. NeuXelec will keep the existing coregistration logic behind the scenes.",
        )
        if assignments is None:
            return

        valid_assignments = [
            item
            for item in assignments
            if item.get("path", "") and item.get("role", "IGNORE") != "IGNORE"
        ]

        if not valid_assignments:
            self._update_buttons()
            return
        mri_label_role_paths = [
            (
                str(item.get("role", "")),
                str(item.get("path", "")),
            )
            for item in valid_assignments
            if str(item.get("role", "")) in ("T1", "T2")
        ]

        self._ask_mri_filename_labels(mri_label_role_paths)
        shared_output_dir = self._ask_common_nifti_output_dir(valid_assignments)

        if (
            any(self._is_dicom_source(str(item.get("path", ""))) for item in valid_assignments)
            and shared_output_dir is None
        ):
            self._update_buttons()
            return

        try:
            self._show_files_loading(
                "Preparing imaging import",
                progress=0.10,
            )

            total = len(valid_assignments)

            for i, item in enumerate(valid_assignments, start=1):
                role = item.get("role", "IGNORE")
                source_path = item.get("path", "")

                try:
                    file_name = Path(source_path).name
                except Exception:
                    file_name = str(source_path)

                base_progress = 0.10 + 0.75 * ((i - 1) / max(1, total))

                try:
                    self._update_files_loading(
                        base_progress,
                        f"Converting {i}/{total} · {file_name}",
                    )

                    nifti_path = self._convert_to_nifti_for_import(
                        source_path,
                        role=role,
                        output_dir=shared_output_dir,
                    )

                    self._update_files_loading(
                        min(0.92, base_progress + 0.12),
                        f"Importing {i}/{total} · {file_name}",
                    )

                    self._import_imaging_file(
                        role,
                        source_path,
                        str(nifti_path),
                    )

                except Exception as e:
                    NeuXelecMessageDialog.critical(
                        self._dialog_parent(),
                        "Import failed",
                        f"Could not import:\n{source_path}\n\nDetails:\n{e}",
                    )

            self._complete_files_loading()

        finally:
            QTimer.singleShot(350, self._cancel_files_loading)
            self._update_buttons()

    def load_parcellation_bundle(self) -> None:
        paths = self._pick_multiple_files(
            "Load parcellation / atlas files",
            ("Parcellations (*.nii *.nii.gz *.mgz *.mgh);;" "All files (*.*)"),
            allow_dicom_folders=False,
            files_button_text=("Files (.nii / .nii.gz / .mgz / .mgh)"),
        )
        if not paths:
            return

        roles = [
            ("PARCEL1", "Parcellation 1"),
            ("PARCEL2", "Parcellation 2"),
        ]
        suggestions = {p: self._guess_parcellation_role(p) for p in paths}
        assignments = FileAssignmentDialog.get_assignments(
            paths,
            roles,
            suggestions=suggestions,
            parent=self._dialog_parent(),
            title="Assign parcellation files",
            subtitle=(
                "Assign each atlas image to Parcellation 1 or Parcellation 2. "
                "Geometry checks are performed against MRI 1."
            ),
        )
        if assignments is None:
            return

        valid_assignments = [
            item
            for item in assignments
            if item.get("path", "") and item.get("role", "IGNORE") != "IGNORE"
        ]

        if not valid_assignments:
            self._update_buttons()
            return

        convertible_assignments = [
            item for item in valid_assignments if item.get("role", "") in ("PARCEL1", "PARCEL2")
        ]

        shared_output_dir = self._ask_common_nifti_output_dir(convertible_assignments)

        if (
            any(
                self._is_dicom_source(str(item.get("path", ""))) for item in convertible_assignments
            )
            and shared_output_dir is None
        ):
            self._update_buttons()
            return

        try:
            self._show_files_loading(
                "Preparing parcellation import",
                progress=0.10,
            )

            total = len(valid_assignments)

            for i, item in enumerate(valid_assignments, start=1):
                role = item.get("role", "PARCEL1")
                source_path = item.get("path", "")

                try:
                    file_name = Path(source_path).name
                except Exception:
                    file_name = str(source_path)

                base_progress = 0.10 + 0.75 * ((i - 1) / max(1, total))

                try:
                    self._update_files_loading(
                        base_progress,
                        f"Converting {i}/{total} · {file_name}",
                    )

                    nifti_path = self._convert_to_nifti_for_import(
                        source_path,
                        role=role,
                        output_dir=shared_output_dir,
                    )

                    self._update_files_loading(
                        min(0.92, base_progress + 0.12),
                        f"Importing {i}/{total} · {file_name}",
                    )

                    self._import_parcellation_file(
                        role,
                        source_path,
                        str(nifti_path),
                    )

                except Exception as e:
                    NeuXelecMessageDialog.critical(
                        self._dialog_parent(),
                        "Import failed",
                        (f"Could not import:\n{source_path}\n\n" f"Details:\n{e}"),
                    )

            self._complete_files_loading()

        finally:
            QTimer.singleShot(350, self._cancel_files_loading)
            self._update_buttons()

    def load_surfaces_bundle(self) -> None:
        paths = self._pick_multiple_files(
            "Load surfaces / derived masks",
            (
                "Surfaces and masks "
                "(*.pial *.surf *.stl *.ply *.vtp *.npz *.nii *.nii.gz);;"
                "FreeSurfer pial surfaces (*.pial);;"
                "All files (*.*)"
            ),
            allow_dicom_folders=False,
            files_button_text=("Files (.pial / .surf / .stl / .ply / .vtp / .npz / .nii)"),
        )
        if not paths:
            return

        roles = [
            ("LH_PIAL", "Left pial surface"),
            ("RH_PIAL", "Right pial surface"),
            ("BRAINMASK", "Brain mask"),
            ("IGNORE", "Ignore"),
        ]
        suggestions = {p: self._guess_surface_role(p) for p in paths}
        assignments = FileAssignmentDialog.get_assignments(
            paths,
            roles,
            suggestions=suggestions,
            parent=self._dialog_parent(),
            title="Assign surfaces and derived files",
            subtitle="Assign FreeSurfer pial surfaces or a brain mask. Pial coregistration is triggered only when both LH and RH are available.",
        )
        if assignments is None:
            return

        pial_changed = False
        for item in assignments:
            role = item.get("role", "IGNORE")
            path = item.get("path", "")
            if not path or role == "IGNORE":
                continue
            try:
                if role == "LH_PIAL":
                    self._import_lh_pial_file(path, trigger_coreg=False)
                    pial_changed = True
                elif role == "RH_PIAL":
                    self._import_rh_pial_file(path, trigger_coreg=False)
                    pial_changed = True
                elif role == "BRAINMASK":
                    self._import_brainmask_file(path)
            except Exception as e:
                NeuXelecMessageDialog.critical(
                    self._dialog_parent(),
                    "Import failed",
                    f"Could not import:\n{path}\n\nDetails:\n{e}",
                )

        if pial_changed:
            self._maybe_trigger_pial_coreg()

        self._update_buttons()

    def _import_imaging_file(self, role: str, source_path: str, path: str) -> None:
        role = str(role)
        if role == "T1":
            try:
                final_path, final_img, conform_info = ask_user_to_conform_t1_if_needed(
                    path,
                    parent=self._dialog_parent(),
                )
            except Exception as e:
                NeuXelecMessageDialog.critical(
                    self._dialog_parent(),
                    "T1 resolution check failed",
                    f"The T1 resolution could not be checked.\n\nDetails:\n{e}",
                )
                return

            self.state.t1_source_path = source_path
            self.state.t1_path = final_path
            self.state.t1_sitk = final_img

            self.state.t1_was_conformed = bool(conform_info.get("was_conformed", False))

            self.state.t1_conformed_path = (
                str(conform_info.get("final_path"))
                if bool(conform_info.get("was_conformed", False))
                else None
            )
            self.state.t1_original_spacing = conform_info.get("original_spacing", None)
            self.state.t1_conformed_spacing = conform_info.get("final_spacing", None)
            if self.le_t1:
                self.le_t1.setText(final_path)
                self.le_t1.setCursorPosition(0)
                self.le_t1.setToolTip(final_path)
            if self.chk_t1 is not None:
                self.chk_t1.setEnabled(True)
            self._force_t1_checked()
            self._invalidate_all_coreg()
            vp = self._view3d()
            if vp is not None:
                try:
                    vp.set_t1(self.state.t1_sitk, t1_path=final_path)
                except Exception:
                    pass
            op = getattr(self.state, "oblique_page", None)
            if op is not None:
                try:
                    if hasattr(op, "refresh_mri_source_controls"):
                        op.refresh_mri_source_controls()
                    if hasattr(op, "refresh_available_modalities"):
                        op.refresh_available_modalities(refresh=False)
                except Exception:
                    pass
            return

        if role == "T2":
            self.state.t2_source_path = source_path
            self.state.t2_path = path

            if self.le_t2:
                self.le_t2.setText(path)
                self.le_t2.setCursorPosition(0)
                self.le_t2.setToolTip(path)
            if self.chk_t2 is not None:
                self.chk_t2.setEnabled(True)
            self._invalidate_modality("T2")
            vp = self._view3d()
            if vp is not None and hasattr(vp, "_refresh_3d_mri_source_controls"):
                try:
                    vp._refresh_3d_mri_source_controls()
                except Exception:
                    pass
            op = getattr(self.state, "oblique_page", None)
            if op is not None and hasattr(op, "refresh_mri_source_controls"):
                try:
                    op.refresh_mri_source_controls()
                except Exception:
                    pass
            return

        if role == "CT":
            self.state.ct_source_path = source_path
            self.state.ct_path = path
            if self.le_ct:
                self.le_ct.setText(path)
                self.le_ct.setCursorPosition(0)
                self.le_ct.setToolTip(path)
            if self.chk_ct is not None:
                self.chk_ct.setEnabled(True)
            img = nib.load(path)
            data = img.get_fdata(dtype=np.float32)
            if data.ndim == 4:
                data = data[..., 0]
            self.state.volume = Volume(data=data, path=path)
            self._invalidate_modality("CT")
            try:
                reco = getattr(self.state, "reco_page", None)
                if reco is not None and hasattr(reco, "_show_coreg_warning"):
                    reco._show_coreg_warning()
            except Exception:
                pass
            if self.on_ct_loaded:
                self.on_ct_loaded()
            elif self.on_ct_updated:
                self.on_ct_updated()
            return

        if role == "PET":
            self.state.pet_source_path = source_path
            self.state.pet_path = path
            if self.le_pet:
                self.le_pet.setText(path)
                self.le_pet.setCursorPosition(0)
                self.le_pet.setToolTip(path)
            if self.chk_pet is not None:
                self.chk_pet.setEnabled(True)
            self._invalidate_modality("PET")
            return

        if role == "ictalSPECT":
            self.state.ictal_spect_source_path = source_path
            self.state.ictal_spect_path = path
            if self.le_ictal:
                self.le_ictal.setText(path)
                self.le_ictal.setCursorPosition(0)
                self.le_ictal.setToolTip(path)
            if self.chk_ictal is not None:
                self.chk_ictal.setEnabled(True)
            self._invalidate_modality("ictalSPECT")
            return

        if role == "interictalSPECT":
            self.state.interictal_spect_source_path = source_path
            self.state.interictal_spect_path = path
            if self.le_interictal:
                self.le_interictal.setText(path)
                self.le_interictal.setCursorPosition(0)
                self.le_interictal.setToolTip(path)
            if self.chk_interictal is not None:
                self.chk_interictal.setEnabled(True)
            self._invalidate_modality("interictalSPECT")
            return

        if role == "SISCOM":
            self._import_siscom_file(source_path, path)
            return

    def _import_siscom_file(self, source_path: str, path: str) -> None:
        self.state.siscom_path = path
        if self.le_siscom:
            self.le_siscom.setText(path)
            self.le_siscom.setCursorPosition(0)
            self.le_siscom.setToolTip(path)

        if getattr(self.state, "t1_sitk", None) is None and getattr(self.state, "t1_path", None):
            try:
                self.state.t1_sitk = sitk.ReadImage(self.state.t1_path)
            except Exception:
                self.state.t1_sitk = None

        img = sitk.ReadImage(path)
        self._warn_geometry_mismatch_against_t1(img, "SISCOM")
        self.state.siscom_coreg_path = None
        self.state.siscom_coreg_in_t1 = img
        self.state.siscom_z_in_t1 = img
        self.state.siscom_thr_in_t1 = None
        self.state.siscom_validated = False
        vp = self._view3d()
        if vp is not None:
            try:
                vp.set_siscom(img, siscom_path=path)
            except Exception:
                pass

    def _warn_geometry_mismatch_against_t1(self, img: sitk.Image, label: str) -> None:
        t1 = getattr(self.state, "t1_sitk", None)
        if t1 is None:
            return
        try:
            mismatch = []
            if tuple(img.GetSize()) != tuple(t1.GetSize()):
                mismatch.append(f"Size: {label} {img.GetSize()} vs T1 {t1.GetSize()}")
            if tuple(np.round(img.GetSpacing(), 6)) != tuple(np.round(t1.GetSpacing(), 6)):
                mismatch.append(f"Spacing: {label} {img.GetSpacing()} vs T1 {t1.GetSpacing()}")
            if tuple(np.round(img.GetOrigin(), 6)) != tuple(np.round(t1.GetOrigin(), 6)):
                mismatch.append(f"Origin: {label} {img.GetOrigin()} vs T1 {t1.GetOrigin()}")
            if tuple(np.round(img.GetDirection(), 6)) != tuple(np.round(t1.GetDirection(), 6)):
                mismatch.append(f"Direction differs ({label} vs T1)")
            if mismatch:
                NeuXelecMessageDialog.warning(
                    None,
                    f"{label} geometry mismatch",
                    f"Warning: {label} does not match the T1 geometry.\n"
                    "Overlay may be incorrect unless it is already in T1 space.\n\n"
                    + "\n".join(mismatch),
                )
        except Exception:
            pass

    def _auto_load_freesurfer_lut_from_image(self, img: sitk.Image, filename: str) -> dict:
        lut_path = Path(__file__).resolve().parent.parent / "utils" / "FreeSurferColorLUT.txt"
        if not lut_path.exists():
            if "aparc" in filename or "aseg" in filename or "wmparc" in filename:
                NeuXelecMessageDialog.warning(
                    None,
                    "FreeSurfer LUT not found",
                    "This parcellation looks like a FreeSurfer atlas, but FreeSurferColorLUT.txt "
                    f"was not found at:\n{lut_path}\n\n"
                    "Please place FreeSurferColorLUT.txt in:\n"
                    "Neuxelec/src/neuxelec/utils/",
                )
            return {}

        if "aparc" in filename or "aseg" in filename or "wmparc" in filename:
            return self._load_freesurfer_lut_dict(lut_path)

        try:
            arr = sitk.GetArrayViewFromImage(img)
            uniq = np.unique(arr)
            if np.any((uniq >= 1000) & (uniq < 5000)):
                return self._load_freesurfer_lut_dict(lut_path)
        except Exception:
            pass

        return {}

    def _import_parcellation_file(self, role: str, source_path: str, path: str) -> None:
        target = "1" if role == "PARCEL1" else "2"
        img = sitk.ReadImage(path)
        filename = Path(path).name.lower()
        lut = self._auto_load_freesurfer_lut_from_image(img, filename)

        if target == "1":
            self.state.parcel1_path = path
            self.state.parcellation1_lut = lut
            if self.le_parcel1:
                self.le_parcel1.setText(path)
                self.le_parcel1.setCursorPosition(0)
                self.le_parcel1.setToolTip(path)
            self._warn_geometry_mismatch_against_t1(img, "Parcellation 1")
            self.state.parcel1_img = img
            vp = self._view3d()
            if vp is not None and hasattr(vp, "set_parcellation1"):
                try:
                    vp.set_parcellation1(img, path)
                except Exception:
                    pass
            op = getattr(self.state, "oblique_page", None)
            if op is not None and hasattr(op, "set_parcellation1"):
                try:
                    op.set_parcellation1(img, path)
                except Exception:
                    pass
        else:
            self.state.parcel2_path = path
            self.state.parcellation2_lut = lut
            if self.le_parcel2:
                self.le_parcel2.setText(path)
                self.le_parcel2.setCursorPosition(0)
                self.le_parcel2.setToolTip(path)
            self._warn_geometry_mismatch_against_t1(img, "Parcellation 2")
            self.state.parcel2_img = img
            vp = self._view3d()
            if vp is not None and hasattr(vp, "set_parcellation2"):
                try:
                    vp.set_parcellation2(img, path)
                except Exception:
                    pass
            op = getattr(self.state, "oblique_page", None)
            if op is not None and hasattr(op, "set_parcellation2"):
                try:
                    op.set_parcellation2(img, path)
                except Exception:
                    pass

    def _import_parcellation_lut(self, role: str, path: str) -> None:
        lut = self._load_freesurfer_lut_dict(Path(path))
        if not lut:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(),
                "Lookup table",
                f"No valid FreeSurfer-style LUT entries were found in:\n{path}",
            )
            return
        if role == "PARCEL2_LUT":
            self.state.parcellation2_lut = lut
        else:
            self.state.parcellation1_lut = lut

    def _import_lh_pial_file(self, path: str, trigger_coreg: bool = True) -> None:
        self.state.lh_pial_path = path
        self.state.pial_surfaces_available = False
        if getattr(self, "le_lh_pial", None) is not None:
            self.le_lh_pial.setText(path)
            self.le_lh_pial.setCursorPosition(0)
            self.le_lh_pial.setToolTip(path)
        try:
            self.state.last_browse_dir = str(Path(path).parent)
        except Exception:
            pass
        vp = self._view3d()
        if vp is not None:
            try:
                vp.set_pial_surfaces(
                    lh_path=self.state.lh_pial_path,
                    rh_path=getattr(self.state, "rh_pial_path", None),
                    assume_lps=False,
                )
            except Exception:
                pass
        if trigger_coreg:
            self._maybe_trigger_pial_coreg()

    def _import_rh_pial_file(self, path: str, trigger_coreg: bool = True) -> None:
        self.state.rh_pial_path = path
        self.state.pial_surfaces_available = False
        if getattr(self, "le_rh_pial", None) is not None:
            self.le_rh_pial.setText(path)
            self.le_rh_pial.setCursorPosition(0)
            self.le_rh_pial.setToolTip(path)
        try:
            self.state.last_browse_dir = str(Path(path).parent)
        except Exception:
            pass
        vp = self._view3d()
        if vp is not None:
            try:
                vp.set_pial_surfaces(
                    lh_path=getattr(self.state, "lh_pial_path", None),
                    rh_path=self.state.rh_pial_path,
                    assume_lps=False,
                )
            except Exception:
                pass
        if trigger_coreg:
            self._maybe_trigger_pial_coreg()

    def _import_brainmask_file(self, path: str) -> None:
        self.state.brainmask_path = path
        self.state.brainmask_generated = True
        self.state.brainmask_saved = True
        self.state.brainmask_generated_path = None
        self.state.brainmask_sitk = sitk.ReadImage(path)
        if getattr(self, "le_load_brainmask", None) is not None:
            self.le_load_brainmask.setText(path)
            self.le_load_brainmask.setCursorPosition(0)
            self.le_load_brainmask.setToolTip(path)
        vp = self._view3d()
        if vp is not None:
            try:
                vp.set_brainmask(self.state.brainmask_sitk, brainmask_path=path)
            except Exception:
                pass
        op = getattr(self.state, "oblique_page", None)
        if op is not None:
            try:
                if hasattr(op, "_last_brain_key"):
                    op._last_brain_key = None
                if hasattr(op, "_last_brain_kind"):
                    op._last_brain_kind = None
                if hasattr(op, "_schedule_refresh"):
                    op._schedule_refresh(slices=True, brain=True)
            except Exception:
                pass
        if getattr(self, "btn_save_brainmask", None) is not None:
            self.btn_save_brainmask.setEnabled(True)

    def _import_iso_surface_file(self, path: str) -> None:
        suffix = Path(path).suffix.lower()
        if suffix == ".npz":
            mesh = np.load(path, allow_pickle=True)
            if "mesh" in mesh:
                mesh_obj = mesh["mesh"].item()
            else:
                points = mesh["points"] if "points" in mesh else mesh.get("verts")
                faces = mesh["faces"] if "faces" in mesh else None
                mesh_obj = {"points": points, "faces": faces}
        else:
            import pyvista as pv

            poly = pv.read(path)
            mesh_obj = {
                "points": np.asarray(poly.points, dtype=np.float32),
                "faces": (
                    np.asarray(poly.faces.reshape(-1, 4)[:, 1:], dtype=np.int32)
                    if getattr(poly, "faces", None) is not None and poly.faces.size
                    else None
                ),
            }
        self.state.iso_surface_mesh = mesh_obj
        self.state.iso_surface_saved_path = path
        if getattr(self, "le_load_isosurface", None) is not None:
            self.le_load_isosurface.setText(path)
            self.le_load_isosurface.setCursorPosition(0)
            self.le_load_isosurface.setToolTip(path)
        vp = self._view3d()
        if vp is not None and hasattr(vp, "set_iso_surface"):
            try:
                vp.set_iso_surface(mesh_obj)
            except Exception:
                pass
        if getattr(self, "btn_save_iso_surface", None) is not None:
            self.btn_save_iso_surface.setEnabled(True)

    # ------------------------
    # Helper: pick file/folder + convert
    # ------------------------
    def _pick_path_or_convert(
        self,
        caption: str,
        start_dir: str | None = None,
    ) -> tuple[str, str] | None:
        browse_dir = start_dir or self.state.last_browse_dir

        input_choice = NeuXelecMessageDialog.choice(
            self._dialog_parent(),
            "Select Input Type",
            "What do you want to load?",
            choices=[
                (
                    "single_file",
                    "Single file (.nii / .mgz)",
                    True,
                ),
                (
                    "dicom_folder",
                    "DICOM folder",
                    False,
                ),
            ],
            cancel_text="Cancel",
        )

        if input_choice == "single_file":
            path, _ = QFileDialog.getOpenFileName(
                self._dialog_parent(),
                caption,
                browse_dir,
                "Medical images (*.nii *.nii.gz *.mgz *.mgh *.dcm *.ima);;All files (*.*)",
            )

            if path:
                self.state.last_browse_dir = str(Path(path).parent)

            if not path:
                return None

        elif input_choice == "dicom_folder":
            path = QFileDialog.getExistingDirectory(
                self._dialog_parent(),
                caption + " (DICOM folder)",
                browse_dir,
            )

            if path:
                self.state.last_browse_dir = str(Path(path))

            if not path:
                return None

        else:
            return None

        try:
            self._show_files_loading(
                f"Converting · {Path(path).name}",
                progress=0.20,
            )

            nifti_path = convert_to_nifti_if_needed(
                path,
                parent=self._dialog_parent(),
            )

            self._complete_files_loading()

            return str(path), str(nifti_path)

        except Exception as e:
            self._cancel_files_loading()
            NeuXelecMessageDialog.critical(
                self._dialog_parent(),
                "Conversion failed",
                str(e),
            )
            return None

        finally:
            QTimer.singleShot(350, self._cancel_files_loading)

    # ------------------------
    # Loaders
    # ------------------------
    def load_t1(self):
        picked = self._pick_path_or_convert("Select MRI 1 (fixed)")
        if not picked:
            return

        source_path, path = picked

        try:
            final_path, final_img, conform_info = ask_user_to_conform_t1_if_needed(
                path,
                parent=self._dialog_parent(),
            )
        except Exception as e:
            NeuXelecMessageDialog.critical(
                self._dialog_parent(),
                "MRI 1 resolution check failed",
                ("The MRI 1 resolution could not be checked.\n\n" f"Details:\n{e}"),
            )
            return

        self.state.t1_source_path = source_path
        self.state.t1_path = final_path
        self.state.t1_sitk = final_img
        self._ask_mri_filename_label("T1", source_path=source_path)

        self.state.t1_was_conformed = bool(conform_info.get("was_conformed", False))
        self.state.t1_conformed_path = (
            str(conform_info.get("final_path"))
            if bool(conform_info.get("was_conformed", False))
            else None
        )
        self.state.t1_original_spacing = conform_info.get(
            "original_spacing",
            None,
        )
        self.state.t1_conformed_spacing = conform_info.get(
            "final_spacing",
            None,
        )

        if self.le_t1:
            self.le_t1.setText(final_path)
            self.le_t1.setCursorPosition(0)
            self.le_t1.setToolTip(final_path)

        if self.chk_t1 is not None:
            self.chk_t1.setEnabled(True)

        self._force_t1_checked()

        # New T1 reference space: invalidate every derived coregistration.
        self._invalidate_all_coreg()

        # Push the final T1 actually used by NeuXelec to 3D View.
        vp = self._view3d()
        if vp is not None:
            try:
                vp.set_t1(
                    self.state.t1_sitk,
                    t1_path=final_path,
                )
            except Exception:
                pass

        self._update_buttons()

        op = getattr(self.state, "oblique_page", None)
        if op is not None:
            try:
                if hasattr(op, "refresh_mri_source_controls"):
                    op.refresh_mri_source_controls()
                if hasattr(op, "refresh_available_modalities"):
                    op.refresh_available_modalities(refresh=False)
            except Exception:
                pass

    def load_t2(self):
        picked = self._pick_path_or_convert("Select MRI 2")
        if not picked:
            return
        source_path, path = picked
        self.state.t2_source_path = source_path
        self.state.t2_path = path
        self._ask_mri_filename_label("T2", source_path=source_path)
        if self.le_t2:
            self.le_t2.setText(path)
            self.le_t2.setCursorPosition(0)
        if self.chk_t2 is not None:
            self.chk_t2.setEnabled(True)
        self._invalidate_modality("T2")
        self._update_buttons()
        vp = self._view3d()
        if vp is not None and hasattr(vp, "_refresh_3d_mri_source_controls"):
            try:
                vp._refresh_3d_mri_source_controls()
            except Exception:
                pass
        op = getattr(self.state, "oblique_page", None)
        if op is not None and hasattr(op, "refresh_mri_source_controls"):
            try:
                op.refresh_mri_source_controls()
            except Exception:
                pass

    def load_ct(self):
        picked = self._pick_path_or_convert("Select CT")
        if not picked:
            return
        source_path, path = picked
        self.state.ct_source_path = source_path
        self.state.ct_path = path

        if self.le_ct:
            self.le_ct.setText(path)
            self.le_ct.setCursorPosition(0)
        if self.chk_ct is not None:
            self.chk_ct.setEnabled(True)

        try:
            img = nib.load(path)
            data = img.get_fdata(dtype=np.float32)
            if data.ndim == 4:
                data = data[..., 0]
            self.state.volume = Volume(data=data, path=path)
        except Exception as e:
            NeuXelecMessageDialog.critical(self._dialog_parent(), "Load CT failed", str(e))
            return

        self._invalidate_modality("CT")
        self._update_buttons()

        # Force Reconstruction page to immediately show the
        # "Please check the coregistration" message
        try:
            reco = getattr(self.state, "reco_page", None)
            if reco is not None and hasattr(reco, "_show_coreg_warning"):
                reco._show_coreg_warning()
        except Exception:
            pass

        if self.on_ct_loaded:
            self.on_ct_loaded()
        elif self.on_ct_updated:
            self.on_ct_updated()

    def load_pet(self):
        picked = self._pick_path_or_convert("Select PET")
        if not picked:
            return
        source_path, path = picked
        self.state.pet_source_path = source_path
        self.state.pet_path = path
        if self.le_pet:
            self.le_pet.setText(path)
            self.le_pet.setCursorPosition(0)
        if self.chk_pet is not None:
            self.chk_pet.setEnabled(True)
        self._invalidate_modality("PET")
        self._update_buttons()

    def load_ictal_spect(self):
        picked = self._pick_path_or_convert("Select ictal SPECT")
        if not picked:
            return
        source_path, path = picked
        self.state.ictal_spect_source_path = source_path
        self.state.ictal_spect_path = path
        if self.le_ictal:
            self.le_ictal.setText(path)
            self.le_ictal.setCursorPosition(0)
        if self.chk_ictal is not None:
            self.chk_ictal.setEnabled(True)
        self._invalidate_modality("ictalSPECT")
        self._update_buttons()

    def load_interictal_spect(self):
        picked = self._pick_path_or_convert("Select interictal SPECT")
        if not picked:
            return
        source_path, path = picked
        self.state.interictal_spect_source_path = source_path
        self.state.interictal_spect_path = path
        if self.le_interictal:
            self.le_interictal.setText(path)
            self.le_interictal.setCursorPosition(0)
        if self.chk_interictal is not None:
            self.chk_interictal.setEnabled(True)
        self._invalidate_modality("interictalSPECT")
        self._update_buttons()

    def load_siscom(self):
        """Load a SISCOM map (expected already in T1 space). Adds a geometry mismatch warning vs T1."""
        picked = self._pick_path_or_convert(
            "Select SISCOM (already in T1 space)",
            start_dir=self._t1_preferred_dir(),
        )
        if not picked:
            return

        source_path, path = picked
        self.state.siscom_path = path

        if self.le_siscom:
            self.le_siscom.setText(path)
            self.le_siscom.setCursorPosition(0)

        # Ensure T1 loaded for geometry check (warning only)
        if getattr(self.state, "t1_sitk", None) is None and getattr(self.state, "t1_path", None):
            try:
                self.state.t1_sitk = sitk.ReadImage(self.state.t1_path)
            except Exception:
                self.state.t1_sitk = None

        try:
            img = sitk.ReadImage(path)
        except Exception as e:
            NeuXelecMessageDialog.critical(self._dialog_parent(), "Load SISCOM failed", str(e))
            return

        # Geometry warning vs T1
        t1 = getattr(self.state, "t1_sitk", None)
        if t1 is not None:
            try:
                mismatch = []
                if tuple(img.GetSize()) != tuple(t1.GetSize()):
                    mismatch.append(f"Size: SISCOM {img.GetSize()} vs T1 {t1.GetSize()}")
                if tuple(np.round(img.GetSpacing(), 6)) != tuple(np.round(t1.GetSpacing(), 6)):
                    mismatch.append(f"Spacing: SISCOM {img.GetSpacing()} vs T1 {t1.GetSpacing()}")
                if tuple(np.round(img.GetOrigin(), 6)) != tuple(np.round(t1.GetOrigin(), 6)):
                    mismatch.append(f"Origin: SISCOM {img.GetOrigin()} vs T1 {t1.GetOrigin()}")
                if tuple(np.round(img.GetDirection(), 6)) != tuple(np.round(t1.GetDirection(), 6)):
                    mismatch.append("Direction differs (SISCOM vs T1)")
                if mismatch:
                    NeuXelecMessageDialog.warning(
                        None,
                        "SISCOM geometry mismatch",
                        "Warning: The SISCOM geometry does not match the T1 geometry.\n"
                        "Overlay may be incorrect unless SISCOM is already in T1 space.\n\n"
                        + "\n".join(mismatch),
                    )
            except Exception:
                pass

        # Same logic as PET/CT: raw path loaded, but no validated saved coreg yet
        self.state.siscom_coreg_path = None
        self.state.siscom_coreg_in_t1 = img
        self.state.siscom_z_in_t1 = img
        self.state.siscom_thr_in_t1 = None
        self.state.siscom_validated = False

        # Push to 3D
        vp = self._view3d()
        if vp is not None:
            try:
                vp.set_siscom(img, siscom_path=path)
            except Exception:
                pass

        self._update_buttons()

    def load_parcellation1(self):
        picked = self._pick_path_or_convert(
            "Select Parcellation 1",
            start_dir=self._t1_preferred_dir(),
        )
        if not picked:
            return

        source_path, path = picked
        self.state.parcel1_path = path

        filename = Path(path).name.lower()

        # Reset LUT by default
        self.state.parcellation1_lut = {}

        if "aparc+aseg" in filename:
            lut_path = Path(__file__).resolve().parent.parent / "utils" / "FreeSurferColorLUT.txt"

            if lut_path.exists():
                self.state.parcellation1_lut = self._load_freesurfer_lut_dict(lut_path)
                print("LUT size =", len(self.state.parcellation1_lut))
                print("41 =>", self.state.parcellation1_lut.get(41))
                print("54 =>", self.state.parcellation1_lut.get(54))
                print("2015 =>", self.state.parcellation1_lut.get(2015))
                print(f"✅ FreeSurfer LUT automatically loaded for aparc+aseg: {lut_path}")
            else:
                NeuXelecMessageDialog.warning(
                    None,
                    "FreeSurfer LUT not found",
                    "Parcellation 1 looks like aparc+aseg, but FreeSurferColorLUT.txt "
                    f"was not found at:\n{lut_path}\n\n"
                    "Please place FreeSurferColorLUT.txt in:\n"
                    "Neuxelec/src/neuxelec/utils/",
                )

        if self.le_parcel1:
            self.le_parcel1.setText(path)
            self.le_parcel1.setCursorPosition(0)

        try:
            img = sitk.ReadImage(path)
        except Exception as e:
            NeuXelecMessageDialog.critical(
                self._dialog_parent(), "Load Parcellation 1 failed", str(e)
            )
            return
        # Fallback detection from labels if filename does not contain aparc+aseg
        try:
            if not getattr(self.state, "parcellation1_lut", {}):
                arr = sitk.GetArrayViewFromImage(img)
                uniq = np.unique(arr)
                if np.any((uniq >= 1000) & (uniq < 5000)):
                    lut_path = (
                        Path(__file__).resolve().parent.parent / "utils" / "FreeSurferColorLUT.txt"
                    )
                    if lut_path.exists():
                        self.state.parcellation1_lut = self._load_freesurfer_lut_dict(lut_path)
                        print(f"✅ FreeSurfer LUT automatically loaded from labels: {lut_path}")
        except Exception:
            pass

        t1 = getattr(self.state, "t1_sitk", None)
        if t1 is not None:
            try:
                mismatch = []
                if tuple(img.GetSize()) != tuple(t1.GetSize()):
                    mismatch.append(f"Size: Parcellation 1 {img.GetSize()} vs T1 {t1.GetSize()}")
                if tuple(np.round(img.GetSpacing(), 6)) != tuple(np.round(t1.GetSpacing(), 6)):
                    mismatch.append(
                        f"Spacing: Parcellation 1 {img.GetSpacing()} vs T1 {t1.GetSpacing()}"
                    )
                if tuple(np.round(img.GetOrigin(), 6)) != tuple(np.round(t1.GetOrigin(), 6)):
                    mismatch.append(
                        f"Origin: Parcellation 1 {img.GetOrigin()} vs T1 {t1.GetOrigin()}"
                    )
                if tuple(np.round(img.GetDirection(), 6)) != tuple(np.round(t1.GetDirection(), 6)):
                    mismatch.append("Direction differs (Parcellation 1 vs T1)")
                if mismatch:
                    NeuXelecMessageDialog.warning(
                        None,
                        "Parcellation 1 geometry mismatch",
                        "Warning: Parcellation 1 does not match the T1 geometry.\n"
                        "Overlay may be incorrect unless it is already in T1 space.\n\n"
                        + "\n".join(mismatch),
                    )
            except Exception:
                pass

        self.state.parcel1_img = img

        vp = self._view3d()
        if vp is not None and hasattr(vp, "set_parcellation1"):
            try:
                vp.set_parcellation1(img, path)
            except Exception:
                pass

        op = getattr(self.state, "oblique_page", None)
        if op is not None and hasattr(op, "set_parcellation1"):
            try:
                op.set_parcellation1(img, path)
            except Exception:
                pass

        self._update_buttons()

    def load_parcellation2(self):
        picked = self._pick_path_or_convert(
            "Select Parcellation 2",
            start_dir=self._t1_preferred_dir(),
        )
        if not picked:
            return
        source_path, path = picked
        self.state.parcel2_path = path
        filename = Path(path).name.lower()

        # Reset LUT by default
        self.state.parcellation2_lut = {}

        if "aparc+aseg" in filename:
            lut_path = Path(__file__).resolve().parent.parent / "utils" / "FreeSurferColorLUT.txt"

            if lut_path.exists():
                self.state.parcellation2_lut = self._load_freesurfer_lut_dict(lut_path)
            else:
                NeuXelecMessageDialog.warning(
                    None,
                    "FreeSurfer LUT not found",
                    "Parcellation 2 looks like aparc+aseg, but FreeSurferColorLUT.txt "
                    f"was not found at:\n{lut_path}\n\n"
                    "Please place FreeSurferColorLUT.txt in:\n"
                    "Neuxelec/src/neuxelec/utils/",
                )

        if self.le_parcel2:
            self.le_parcel2.setText(path)
            self.le_parcel2.setCursorPosition(0)

        try:
            img = sitk.ReadImage(path)
        except Exception as e:
            NeuXelecMessageDialog.critical(
                self._dialog_parent(), "Load Parcellation 2 failed", str(e)
            )
            return
        try:
            if not getattr(self.state, "parcellation2_lut", {}):
                arr = sitk.GetArrayViewFromImage(img)
                uniq = np.unique(arr)
                if np.any((uniq >= 1000) & (uniq < 5000)):
                    lut_path = (
                        Path(__file__).resolve().parent.parent / "utils" / "FreeSurferColorLUT.txt"
                    )
                    if lut_path.exists():
                        self.state.parcellation2_lut = self._load_freesurfer_lut_dict(lut_path)
        except Exception:
            pass

        t1 = getattr(self.state, "t1_sitk", None)
        if t1 is not None:
            try:
                mismatch = []
                if tuple(img.GetSize()) != tuple(t1.GetSize()):
                    mismatch.append(f"Size: Parcellation 2 {img.GetSize()} vs T1 {t1.GetSize()}")
                if tuple(np.round(img.GetSpacing(), 6)) != tuple(np.round(t1.GetSpacing(), 6)):
                    mismatch.append(
                        f"Spacing: Parcellation 2 {img.GetSpacing()} vs T1 {t1.GetSpacing()}"
                    )
                if tuple(np.round(img.GetOrigin(), 6)) != tuple(np.round(t1.GetOrigin(), 6)):
                    mismatch.append(
                        f"Origin: Parcellation 2 {img.GetOrigin()} vs T1 {t1.GetOrigin()}"
                    )
                if tuple(np.round(img.GetDirection(), 6)) != tuple(np.round(t1.GetDirection(), 6)):
                    mismatch.append("Direction differs (Parcellation 2 vs T1)")
                if mismatch:
                    NeuXelecMessageDialog.warning(
                        None,
                        "Parcellation 2 geometry mismatch",
                        "Warning: Parcellation 2 does not match the T1 geometry.\n"
                        "Overlay may be incorrect unless it is already in T1 space.\n\n"
                        + "\n".join(mismatch),
                    )
            except Exception:
                pass

        self.state.parcel2_img = img

        vp = self._view3d()
        if vp is not None and hasattr(vp, "set_parcellation2"):
            try:
                vp.set_parcellation2(img, path)
            except Exception:
                pass

        op = getattr(self.state, "oblique_page", None)
        if op is not None and hasattr(op, "set_parcellation2"):
            try:
                op.set_parcellation2(img, path)
            except Exception:
                pass

        self._update_buttons()

    def load_lh_pial(self):
        start_dir = self.state.last_browse_dir or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self._dialog_parent(),
            "Select left hemisphere pial surface",
            start_dir,
            "FreeSurfer surface (*.pial *.surf);;All files (*.*)",
        )
        if not path:
            return

        self.state.lh_pial_path = path
        self.state.pial_surfaces_available = False

        if getattr(self, "le_lh_pial", None) is not None:
            self.le_lh_pial.setText(path)
            self.le_lh_pial.setCursorPosition(0)
            self.le_lh_pial.setToolTip(path)

        try:
            self.state.last_browse_dir = str(Path(path).parent)
        except Exception:
            pass

        vp = self._view3d()
        if vp is not None:
            try:
                vp.set_pial_surfaces(
                    lh_path=self.state.lh_pial_path,
                    rh_path=getattr(self.state, "rh_pial_path", None),
                    assume_lps=False,  # raw FreeSurfer pial
                )
            except Exception:
                pass

        self._update_buttons()
        self._maybe_trigger_pial_coreg()

    def load_rh_pial(self):
        start_dir = self.state.last_browse_dir or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self._dialog_parent(),
            "Select right hemisphere pial surface",
            start_dir,
            "FreeSurfer surface (*.pial *.surf);;All files (*.*)",
        )
        if not path:
            return

        self.state.rh_pial_path = path
        self.state.pial_surfaces_available = False

        if getattr(self, "le_rh_pial", None) is not None:
            self.le_rh_pial.setText(path)
            self.le_rh_pial.setCursorPosition(0)
            self.le_rh_pial.setToolTip(path)

        try:
            self.state.last_browse_dir = str(Path(path).parent)
        except Exception:
            pass

        vp = self._view3d()
        if vp is not None:
            try:
                vp.set_pial_surfaces(
                    lh_path=getattr(self.state, "lh_pial_path", None),
                    rh_path=self.state.rh_pial_path,
                    assume_lps=False,  # raw FreeSurfer pial
                )
            except Exception:
                pass

        self._update_buttons()
        self._maybe_trigger_pial_coreg()

    # ------------------------
    # Invalidate helpers
    # ------------------------
    def _invalidate_all_coreg(self):
        """
        Invalidate every object derived from the previous T1 reference space.

        Important:
        do not clear state.t1_sitk here because this method is called just
        after a newly selected T1 has already been loaded.
        """
        self.state.t2_coreg_in_t1 = None
        self.state.ct_coreg_in_t1 = None
        self.state.pet_coreg_in_t1 = None
        self.state.ictal_spect_coreg_in_t1 = None
        self.state.interictal_spect_coreg_in_t1 = None

        self.state.t2_in_t1 = None
        self.state.ct_in_t1 = None
        self.state.pet_in_t1 = None
        self.state.ictal_spect_in_t1 = None
        self.state.interictal_spect_in_t1 = None

        self.state.t2_coreg_path = None
        self.state.ct_coreg_path = None
        self.state.pet_coreg_path = None
        self.state.ictal_spect_coreg_path = None
        self.state.interictal_spect_coreg_path = None

        self.state.t2_validated = False
        self.state.ct_validated = False
        self.state.ct_ready_for_reconstruction = False
        self.state.pet_validated = False
        self.state.ictal_spect_validated = False
        self.state.interictal_spect_validated = False

        self.state.siscom_coreg_in_t1 = None
        self.state.siscom_coreg_path = None
        self.state.siscom_diff_in_t1 = None
        self.state.siscom_z_in_t1 = None
        self.state.siscom_thr_in_t1 = None
        self.state.siscom_path = None
        self.state.siscom_validated = False

        self._last_result = None
        self._last_modality = None

        self.state.iso_surface_mesh = None
        self.state.iso_surface_saved_path = None

        vp = self._view3d()
        if vp is not None:
            try:
                if hasattr(vp, "set_ct"):
                    vp.set_ct(None)
                if hasattr(vp, "set_t2"):
                    vp.set_t2(None)
                if hasattr(vp, "set_pet"):
                    vp.set_pet(None)
                if hasattr(vp, "set_siscom"):
                    vp.set_siscom(None)
            except Exception:
                pass

        op = getattr(self.state, "oblique_page", None)
        if op is not None and hasattr(op, "refresh_available_modalities"):
            try:
                op.refresh_available_modalities(refresh=False)
            except Exception:
                pass

    def _invalidate_modality(self, modality: Modality):
        """
        Invalidate a modality when a new source file is loaded.

        A previously saved coregistered path must never remain associated with
        a newly selected raw image.
        """
        vp = self._view3d()

        if modality == "T2":
            self.state.t2_coreg_in_t1 = None
            self.state.t2_in_t1 = None
            self.state.t2_coreg_path = None
            self.state.t2_validated = False

            if vp is not None and hasattr(vp, "set_t2"):
                try:
                    vp.set_t2(None)
                except Exception:
                    pass

        elif modality == "CT":
            self.state.ct_coreg_in_t1 = None
            self.state.ct_in_t1 = None
            self.state.ct_coreg_path = None
            self.state.ct_validated = False
            self.state.ct_ready_for_reconstruction = False

            if vp is not None and hasattr(vp, "set_ct"):
                try:
                    vp.set_ct(None)
                except Exception:
                    pass

        elif modality == "PET":
            self.state.pet_coreg_in_t1 = None
            self.state.pet_in_t1 = None
            self.state.pet_coreg_path = None
            self.state.pet_validated = False

            if vp is not None and hasattr(vp, "set_pet"):
                try:
                    vp.set_pet(None)
                except Exception:
                    pass

        elif modality == "ictalSPECT":
            self.state.ictal_spect_coreg_in_t1 = None
            self.state.ictal_spect_in_t1 = None
            self.state.ictal_spect_coreg_path = None
            self.state.ictal_spect_validated = False

        elif modality == "interictalSPECT":
            self.state.interictal_spect_coreg_in_t1 = None
            self.state.interictal_spect_in_t1 = None
            self.state.interictal_spect_coreg_path = None
            self.state.interictal_spect_validated = False

        if modality in ("ictalSPECT", "interictalSPECT"):
            self.state.siscom_diff_in_t1 = None
            self.state.siscom_coreg_in_t1 = None
            self.state.siscom_z_in_t1 = None
            self.state.siscom_thr_in_t1 = None
            self.state.siscom_coreg_path = None
            self.state.siscom_validated = False

            if vp is not None and hasattr(vp, "set_siscom"):
                try:
                    vp.set_siscom(None)
                except Exception:
                    pass

        if self._last_modality == modality:
            self._last_modality = None
            self._last_result = None

        op = getattr(self.state, "oblique_page", None)
        if op is not None and hasattr(op, "refresh_available_modalities"):
            try:
                op.refresh_available_modalities(refresh=False)
            except Exception:
                pass

    # ------------------------
    # Selection logic
    # ------------------------
    def _force_t1_checked(self):
        if getattr(self.state, "t1_path", None) and self.chk_t1 is not None:
            self.chk_t1.blockSignals(True)
            self.chk_t1.setChecked(True)
            self.chk_t1.blockSignals(False)
            self.chk_t1.setEnabled(False)

    def _selected_modality(self) -> Modality | None:
        selected: list[Modality] = []
        if self.chk_t2 is not None and self.chk_t2.isChecked():
            selected.append("T2")
        if self.chk_ct is not None and self.chk_ct.isChecked():
            selected.append("CT")
        if self.chk_pet is not None and self.chk_pet.isChecked():
            selected.append("PET")
        if self.chk_ictal is not None and self.chk_ictal.isChecked():
            selected.append("ictalSPECT")
        if self.chk_interictal is not None and self.chk_interictal.isChecked():
            selected.append("interictalSPECT")
        return selected[0] if len(selected) == 1 else None

    def _moving_path_for(self, modality: Modality) -> str | None:
        return {
            "T2": getattr(self.state, "t2_path", None),
            "CT": getattr(self.state, "ct_path", None),
            "PET": getattr(self.state, "pet_path", None),
            "ictalSPECT": getattr(self.state, "ictal_spect_path", None),
            "interictalSPECT": getattr(self.state, "interictal_spect_path", None),
        }[modality]

    def _dir_from_path_or_dir(self, p: str | None) -> Path | None:
        if not p:
            return None
        try:
            pp = Path(p)
            return pp.parent if pp.is_file() else pp
        except Exception:
            return None

    def _t1_preferred_dir(self) -> str:
        # Prefer the saved T1 NIfTI location; otherwise the original T1 source location
        t1_path = getattr(self.state, "t1_path", None)
        t1_source = getattr(self.state, "t1_source_path", None)

        d = self._dir_from_path_or_dir(t1_path)
        if d is None:
            d = self._dir_from_path_or_dir(t1_source)
        if d is None:
            d = Path(self.state.last_browse_dir or Path.home())

        return str(d)

    def _ictal_preferred_dir(self) -> str:
        # Prefer the saved ictal SPECT NIfTI location; otherwise the original ictal source location
        ictal_path = getattr(self.state, "ictal_spect_path", None)
        ictal_source = getattr(self.state, "ictal_spect_source_path", None)

        d = self._dir_from_path_or_dir(ictal_path)
        if d is None:
            d = self._dir_from_path_or_dir(ictal_source)
        if d is None:
            d = Path(self.state.last_browse_dir or Path.home())

        return str(d)

    # ------------------------------------------------------------------
    # Brain mask (on demand)
    # ------------------------------------------------------------------

    def _start_brainmask_progress_animation(
        self,
        label: str = "Brain mask",
        duration_s: float = 240.0,
    ) -> None:
        """
        Start a linear estimated progress animation for brain mask generation.

        The bar reaches 98% in about 4 minutes, then waits for the real worker
        to finish before displaying 100%.
        """
        self._brainmask_progress_running = True
        self._brainmask_progress_start_time = time.monotonic()
        self._brainmask_progress_duration_s = float(duration_s)
        self._brainmask_progress_max_before_done = 98
        self._brainmask_progress_label = str(label)

        if self.brainmask_bar is not None:
            self.brainmask_bar.setVisible(True)
            self.brainmask_bar.setRange(0, 100)
            self.brainmask_bar.setValue(0)
            self.brainmask_bar.setFormat(f"{label}: Preparing... %p%")

        try:
            self._brainmask_progress_timer.start()
        except Exception:
            pass

    def _brainmask_progress_stage_text(self, value: int) -> str:
        if value < 8:
            return "Preparing"
        if value < 30:
            return "Loading T1"
        if value < 75:
            return "Generating mask"
        if value < 95:
            return "Refining mask"
        if value < 99:
            return "Finalizing"
        return "Done"

    def _tick_brainmask_progress(self) -> None:
        """
        Linear BrainMask progress animation.

        It progresses steadily up to 98%, then waits for the worker to finish.
        """
        if not bool(getattr(self, "_brainmask_progress_running", False)):
            return

        if self.brainmask_bar is None:
            return

        try:
            label = str(getattr(self, "_brainmask_progress_label", "Brain mask"))
            elapsed = max(
                0.0,
                time.monotonic() - float(self._brainmask_progress_start_time),
            )
            duration = max(
                1.0,
                float(getattr(self, "_brainmask_progress_duration_s", 240.0)),
            )

            t = min(elapsed / duration, 1.0)

            # Linear progression.
            start_value = 2
            max_before_done = int(getattr(self, "_brainmask_progress_max_before_done", 98))

            estimated_value = int(round(start_value + (max_before_done - start_value) * t))

            estimated_value = max(
                start_value,
                min(max_before_done, estimated_value),
            )

            current_value = int(self.brainmask_bar.value())
            value = max(current_value, estimated_value)

            self.brainmask_bar.setValue(value)
            self.brainmask_bar.setFormat(
                f"{label}: {self._brainmask_progress_stage_text(value)}... %p%"
            )

        except Exception:
            pass

    def _finish_brainmask_progress_animation(
        self,
        label: str = "Brain mask",
        success: bool = True,
    ) -> None:
        """
        Stop the BrainMask progress animation and display the final state.
        """
        self._brainmask_progress_running = False

        try:
            self._brainmask_progress_timer.stop()
        except Exception:
            pass

        if self.brainmask_bar is None:
            return

        if success:
            self.brainmask_bar.setValue(100)
            self.brainmask_bar.setFormat(f"{label}: Done")
        else:
            self.brainmask_bar.setFormat(f"{label}: Failed")

    def generate_brain_mask(self) -> None:
        """Generate a brain mask for the currently loaded T1 (on demand)."""
        if not self.state.t1_path:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(), "Brain Mask", "Please load a MRI 1 first."
            )
            return

        self._start_brainmask_progress_animation(
            label="Brain mask",
            duration_s=240.0,
        )

        if not getattr(self.state, "transforms_dir", None):
            out_dir = QFileDialog.getExistingDirectory(
                self._dialog_parent(),
                "Select output folder for transforms / brain mask",
                self.state.last_browse_dir or str(Path.home()),
            )
            if not out_dir:
                return
            self.state.transforms_dir = out_dir
            try:
                self.state.last_browse_dir = str(Path(out_dir))
            except Exception:
                pass

        self._bm_worker = BrainMaskWorker(
            self.state.t1_path, transforms_dir=self.state.transforms_dir
        )
        self._bm_worker.progress.connect(self._on_brainmask_progress)

        def _done(out_path: str):
            """Brainmask worker success handler."""
            try:
                # Brainmask was generated but not explicitly saved by the user yet.
                # Keep the generated temporary/output path separately so we know it exists,
                # but keep brainmask_path = None to mean "not saved by user".
                self.state.brainmask_generated = True
                self.state.brainmask_saved = False
                self.state.brainmask_generated_path = str(out_path)
                self.state.brainmask_path = None

                # Do not show it as a saved/loaded file.
                # The line edit stays empty until the user clicks Save Brainmask or Load Brain Mask.
                if getattr(self, "le_load_brainmask", None) is not None:
                    try:
                        self.le_load_brainmask.setText("")
                    except Exception:
                        pass

                # Load brainmask image in memory
                try:
                    self.state.brainmask_sitk = sitk.ReadImage(out_path)
                except Exception:
                    self.state.brainmask_sitk = None

                # Push to 3D view immediately
                vp = self._view3d()
                if vp is not None:
                    try:
                        vp.set_brainmask(self.state.brainmask_sitk, brainmask_path=out_path)
                    except Exception:
                        pass

                # Push to Oblique Slice immediately as well.
                op = getattr(self.state, "oblique_page", None)
                if op is not None:
                    try:
                        # In Oblique Slice, the mini 3D brain preview reads state.brainmask_sitk.
                        # We just invalidate the mini-brain cache and refresh.
                        if hasattr(op, "_last_brain_key"):
                            op._last_brain_key = None
                        if hasattr(op, "_last_brain_kind"):
                            op._last_brain_kind = None

                        if hasattr(op, "_schedule_refresh"):
                            op._schedule_refresh(slices=True, brain=True)
                        elif hasattr(op, "render_all"):
                            op.render_all()
                    except Exception:
                        pass

                # Enable 3D checkbox (brainmask)
                self._enable_3d_checkbox("chk_3d_showBrainmask", True)

                NeuXelecMessageDialog.information(
                    self._dialog_parent(), "Brain Mask", f"Brain mask generated:\n{out_path}"
                )

                self._finish_brainmask_progress_animation(
                    label="Brain mask",
                    success=True,
                )

                if getattr(self, "brainmask_bar", None) is not None:
                    QTimer.singleShot(800, self.brainmask_bar.hide)

                if getattr(self, "btn_save_brainmask", None) is not None:
                    self.btn_save_brainmask.setEnabled(True)

            finally:
                self._update_buttons()

        def _fail(msg: str):
            NeuXelecMessageDialog.critical(self._dialog_parent(), "Brain Mask Error", msg)
            self._finish_brainmask_progress_animation(
                label="Brain mask",
                success=False,
            )

            if getattr(self, "brainmask_bar", None) is not None:
                QTimer.singleShot(1200, self.brainmask_bar.hide)
            self._update_buttons()

        self._bm_worker.finished_ok.connect(_done)
        self._bm_worker.failed.connect(_fail)
        self._bm_worker.start()

    def _on_brainmask_progress(self, v: int):
        """
        Real BrainMask worker progress callback.

        While the smooth UI animation is running, keep the bar linear and cap
        completion at 98% until the worker really finishes.
        """
        if getattr(self, "brainmask_bar", None) is None:
            return

        try:
            v = int(v)
        except Exception:
            return

        if bool(getattr(self, "_brainmask_progress_running", False)):
            if v >= 100:
                self.brainmask_bar.setValue(
                    max(
                        int(self.brainmask_bar.value()),
                        int(getattr(self, "_brainmask_progress_max_before_done", 98)),
                    )
                )
                self.brainmask_bar.setFormat("Brain mask: Finalizing... %p%")
            elif v > 0:
                self.brainmask_bar.setValue(max(int(self.brainmask_bar.value()), v))
            return

        self.brainmask_bar.setValue(v)

    def save_brainmask(self) -> None:
        bm_img = getattr(self.state, "brainmask_sitk", None)
        bm_path = getattr(self.state, "brainmask_path", None)

        if bm_img is None and bm_path:
            try:
                bm_img = sitk.ReadImage(bm_path)
                self.state.brainmask_sitk = bm_img
            except Exception:
                bm_img = None

        if bm_img is None:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(),
                "Save Brain Mask",
                "No brain mask available. Please generate it first.",
            )
            return

        start_dir = self._t1_preferred_dir()
        fixed_label = self._mri1_filename_label()

        default_name = str(
            Path(start_dir)
            / self._default_filename(
                f"{fixed_label}_brainmask",
                ".nii.gz",
            )
        )

        out_path, _ = QFileDialog.getSaveFileName(
            self._dialog_parent(),
            "Save Brain Mask",
            default_name,
            "NIfTI (*.nii *.nii.gz);;All files (*.*)",
        )
        if not out_path:
            return

        try:
            sitk.WriteImage(bm_img, out_path)
        except Exception as e:
            NeuXelecMessageDialog.critical(self._dialog_parent(), "Save Brain Mask failed", str(e))
            return

        try:
            self.state.last_browse_dir = str(Path(out_path).parent)
        except Exception:
            pass
        self.state.brainmask_path = out_path
        self.state.brainmask_generated = True
        self.state.brainmask_saved = True
        self.state.brainmask_generated_path = (
            getattr(self.state, "brainmask_generated_path", None) or out_path
        )

        try:
            if getattr(self, "le_load_brainmask", None) is not None:
                self.le_load_brainmask.setText(out_path)
        except Exception:
            pass

        self._update_buttons()

        NeuXelecMessageDialog.information(
            None,
            "Save Brain Mask",
            "Saved:\n" + out_path,
        )

    # ------------------------------------------------------------------
    # Iso-surface (NEW)
    # ------------------------------------------------------------------
    def generate_iso_surface(self) -> None:
        """Generate an iso-surface from the loaded T1, using progressBarBrainmask."""
        # Ensure T1 sitk loaded
        if getattr(self.state, "t1_sitk", None) is None and getattr(self.state, "t1_path", None):
            try:
                self.state.t1_sitk = sitk.ReadImage(self.state.t1_path)
            except Exception:
                self.state.t1_sitk = None

        if getattr(self.state, "t1_sitk", None) is None:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(), "Iso-Surface", "Please load a T1 first."
            )
            return

        if getattr(self, "brainmask_bar", None) is not None:
            self.brainmask_bar.setVisible(True)
            self.brainmask_bar.setRange(0, 100)
            self.brainmask_bar.setValue(0)
            self.brainmask_bar.setFormat("Generating iso-surface... %p%")

        # You can tune percentile later from UI if you want; keep a safe default now
        self._iso_worker = IsoSurfaceWorker(self.state.t1_sitk, iso_percentile=60)
        self._iso_worker.progress.connect(self._on_brainmask_progress)

        def _done(mesh: object):
            try:
                self.state.iso_surface_mesh = mesh

                # Push to 3D view if method exists
                vp = self._view3d()
                if vp is not None:
                    try:
                        if hasattr(vp, "set_iso_surface"):
                            vp.set_iso_surface(mesh)
                    except Exception:
                        pass

                # Enable 3D checkbox (iso-surface)
                self._enable_3d_checkbox("chk_3d_showIsoSurface", True)

                if getattr(self, "btn_save_iso_surface", None) is not None:
                    self.btn_save_iso_surface.setEnabled(True)

                # Auto-save a lightweight NPZ so the UI can display a concrete path immediately
                out_path = None
                try:
                    base_dir = getattr(self.state, "transforms_dir", None) or str(
                        Path(getattr(self.state, "last_browse_dir", "") or Path.home())
                    )
                    out_path = str(Path(base_dir) / "iso_surface.npz")
                    pts = np.asarray(mesh.get("points", None), dtype=np.float32)
                    faces = np.asarray(mesh.get("faces", None), dtype=np.int32)
                    if pts is not None and faces is not None and pts.size and faces.size:
                        np.savez_compressed(out_path, points=pts, faces=faces)
                        self.state.iso_surface_saved_path = out_path
                        if getattr(self, "le_load_isosurface", None) is not None:
                            try:
                                self.le_load_isosurface.setText(out_path)
                            except Exception:
                                pass
                except Exception:
                    pass

                NeuXelecMessageDialog.information(
                    self._dialog_parent(),
                    "Iso-Surface",
                    "Iso-surface generated. You can now display it in 3D view.",
                )

            finally:
                if getattr(self, "brainmask_bar", None) is not None:
                    self.brainmask_bar.setValue(100)
                    self.brainmask_bar.setVisible(False)
                self._update_buttons()

        def _fail(msg: str):
            NeuXelecMessageDialog.critical(self._dialog_parent(), "Iso-Surface Error", msg)
            if getattr(self, "brainmask_bar", None) is not None:
                self.brainmask_bar.setVisible(False)
            self._update_buttons()

        self._iso_worker.finished_ok.connect(_done)
        self._iso_worker.failed.connect(_fail)
        self._iso_worker.start()

    def save_iso_surface(self) -> None:
        """Save iso-surface mesh to disk (STL/PLY/VTP if pyvista available; else NPZ)."""
        mesh = getattr(self.state, "iso_surface_mesh", None)
        if mesh is None:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(),
                "Save Iso-Surface",
                "No iso-surface available. Please generate it first.",
            )
            return

        start_dir = getattr(self.state, "last_browse_dir", "") or str(Path.home())
        fixed_label = self._mri1_filename_label()

        default_name = str(
            Path(start_dir)
            / self._default_filename(
                f"{fixed_label}_isosurface",
                ".stl",
            )
        )

        out_path, _ = QFileDialog.getSaveFileName(
            self._dialog_parent(),
            "Save Iso-Surface",
            default_name,
            "STL (*.stl);;PLY (*.ply);;VTP (*.vtp);;NPZ (*.npz);;All files (*.*)",
        )
        if not out_path:
            return

        try:
            pts = np.asarray(mesh.get("points", None), dtype=np.float32)
            faces = np.asarray(mesh.get("faces", None), dtype=np.int32)
            if pts is None or faces is None or pts.size == 0 or faces.size == 0:
                raise RuntimeError("Invalid iso-surface mesh in memory.")

            suffix = Path(out_path).suffix.lower()

            if suffix in (".stl", ".ply", ".vtp"):
                try:
                    import pyvista as pv
                except Exception:
                    raise RuntimeError(
                        "pyvista is required to save STL/PLY/VTP. Install pyvista (and vtk)."
                    )

                faces_vtk = np.hstack(
                    [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]
                ).ravel()
                poly = pv.PolyData(pts, faces_vtk)
                poly.save(out_path)
            else:
                # fallback: save as npz
                if suffix != ".npz":
                    out_path = str(Path(out_path).with_suffix(".npz"))
                np.savez_compressed(out_path, points=pts, faces=faces)

        except Exception as e:
            NeuXelecMessageDialog.critical(self._dialog_parent(), "Save Iso-Surface failed", str(e))
            return

        try:
            self.state.iso_surface_saved_path = out_path
            self.state.last_browse_dir = str(Path(out_path).parent)
        except Exception:
            pass

        NeuXelecMessageDialog.information(
            self._dialog_parent(), "Save Iso-Surface", "Saved:\n" + out_path
        )

    # ------------------------------------------------------------------
    # SISCOM
    # ------------------------------------------------------------------
    def perform_siscom(self) -> None:
        if getattr(self.state, "t1_sitk", None) is None and getattr(self.state, "t1_path", None):
            try:
                self.state.t1_sitk = sitk.ReadImage(self.state.t1_path)
            except Exception:
                self.state.t1_sitk = None

        if getattr(self.state, "t1_sitk", None) is None:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(), "SISCOM", "Please load a MRI 1 first."
            )
            return
        brainmask = getattr(self.state, "brainmask_sitk", None)

        if brainmask is None:
            choice = NeuXelecMessageDialog.choice(
                self._dialog_parent(),
                "Brain mask required",
                (
                    "A brain mask is required before performing SISCOM.\n\n"
                    "Would you like to generate a new brain mask from MRI 1 "
                    "or load an existing brain mask?"
                ),
                choices=[
                    (
                        "generate",
                        "Generate brain mask",
                        True,
                    ),
                    (
                        "load",
                        "Load brain mask",
                        False,
                    ),
                ],
                cancel_text="Cancel",
            )

            if choice == "generate":
                self.generate_brain_mask()
                return

            if choice == "load":
                self.load_brainmask()
                return

            return
        if not bool(getattr(self.state, "ictal_spect_validated", False)) or not bool(
            getattr(self.state, "interictal_spect_validated", False)
        ):
            NeuXelecMessageDialog.warning(
                None,
                "SISCOM",
                "Please validate BOTH ictal and interictal SPECT coregistrations to T1 first.",
            )
            return

        if getattr(self, "siscom_bar", None) is not None:
            self.siscom_bar.setVisible(True)
            self.siscom_bar.setRange(0, 100)
            self.siscom_bar.setValue(0)
            self.siscom_bar.setFormat("Computing SISCOM... %p%")

        if getattr(self, "btn_perf_siscom", None) is not None:
            self.btn_perf_siscom.setEnabled(False)
        if getattr(self, "btn_check_siscom", None) is not None:
            self.btn_check_siscom.setEnabled(False)
        if getattr(self, "btn_save_siscom", None) is not None:
            self.btn_save_siscom.setEnabled(False)

        brainmask = getattr(self.state, "brainmask_sitk", None)

        self._siscom_worker = SISCOMWorker(
            t1_img=self.state.t1_sitk,
            ictal_img=self.state.ictal_spect_coreg_in_t1,
            interictal_img=self.state.interictal_spect_coreg_in_t1,
            brainmask_img=brainmask,
            z_threshold=2.0,
            smooth_fwhm_mm=6.0,
        )
        self._siscom_worker.progress.connect(self._on_siscom_progress)

        def _done(out: dict):
            try:
                self.state.siscom_diff_in_t1 = out.get("diff", None)
                self.state.siscom_z_in_t1 = out.get("z", None)
                self.state.siscom_thr_in_t1 = out.get("thr", None)
                self.state.siscom_validated = False
                NeuXelecMessageDialog.information(
                    self._dialog_parent(),
                    "SISCOM",
                    "SISCOM computed. Click 'Check SISCOM' to visualize.",
                )
            finally:
                if getattr(self, "siscom_bar", None) is not None:
                    self.siscom_bar.setValue(100)
                    self.siscom_bar.setVisible(False)
                if getattr(self, "btn_perf_siscom", None) is not None:
                    self.btn_perf_siscom.setEnabled(True)
                self._update_buttons()

        def _fail(msg: str):
            if getattr(self, "siscom_bar", None) is not None:
                self.siscom_bar.setVisible(False)
            if getattr(self, "btn_perf_siscom", None) is not None:
                self.btn_perf_siscom.setEnabled(True)
            NeuXelecMessageDialog.critical(self._dialog_parent(), "SISCOM Error", msg)
            self._update_buttons()

        self._siscom_worker.finished_ok.connect(_done)
        self._siscom_worker.failed.connect(_fail)
        self._siscom_worker.start()

    def _on_siscom_progress(self, v: int):
        if getattr(self, "siscom_bar", None) is not None:
            self.siscom_bar.setValue(int(v))

    def check_siscom(self) -> None:
        if getattr(self.state, "t1_sitk", None) is None and getattr(self.state, "t1_path", None):
            try:
                self.state.t1_sitk = sitk.ReadImage(self.state.t1_path)
            except Exception:
                self.state.t1_sitk = None

        if getattr(self.state, "t1_sitk", None) is None:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(), "Check SISCOM", "Please load a MRI 1 first."
            )
            return

        siscom_img = getattr(self.state, "siscom_thr_in_t1", None) or getattr(
            self.state, "siscom_z_in_t1", None
        )
        if siscom_img is None:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(),
                "Check SISCOM",
                "No SISCOM map available. Load one or click 'Perform SISCOM' first.",
            )
            return

        dlg = OverlayViewer(
            fixed_t1=self.state.t1_sitk,
            moving_in_t1=siscom_img,
            moving_name="SISCOM",
            parent=self._dialog_parent(),
        )
        result = dlg.exec()

        if result == QDialog.Accepted:
            try:
                refined = dlg.corrected_moving_image()
            except Exception:
                refined = siscom_img

            self.state.siscom_coreg_in_t1 = refined
            self.state.siscom_z_in_t1 = refined
            self.state.siscom_thr_in_t1 = None
            self.state.siscom_validated = True

            op = getattr(self.state, "oblique_page", None)
            if op is not None:
                try:
                    if hasattr(op, "refresh_available_modalities"):
                        op.refresh_available_modalities()
                    elif hasattr(op, "render_all"):
                        op.render_all()
                except Exception:
                    pass

            # Push to 3D
            vp = self._view3d()
            if vp is not None:
                try:
                    vp.set_siscom(refined)
                except Exception:
                    pass

            # Enable 3D checkbox (siscom) only after validation
            self._enable_3d_checkbox("chk_3d_showSISCOM", True)

            self._update_buttons()
            NeuXelecMessageDialog.information(
                self._dialog_parent(), "Validated", "SISCOM validated."
            )

    def save_siscom(self) -> None:
        if not bool(getattr(self.state, "siscom_validated", False)):
            NeuXelecMessageDialog.warning(
                self._dialog_parent(),
                "Save SISCOM",
                "Please validate SISCOM first (Check SISCOM → OK).",
            )
            return

        siscom_img = getattr(self.state, "siscom_coreg_in_t1", None)
        if siscom_img is None:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(), "Save SISCOM", "No SISCOM image to save."
            )
            return

        start_dir = self._ictal_preferred_dir()
        fixed_label = self._mri1_filename_label()

        default_path = str(
            Path(start_dir)
            / self._default_filename(
                f"SISCOM_to_{fixed_label}",
                ".nii.gz",
            )
        )

        out_path, _ = QFileDialog.getSaveFileName(
            self._dialog_parent(),
            "Save SISCOM",
            default_path,
            "NIfTI (*.nii *.nii.gz);;All files (*.*)",
        )
        if not out_path:
            return

        try:
            save_nifti(siscom_img, out_path)
        except Exception as e:
            NeuXelecMessageDialog.critical(self._dialog_parent(), "Save failed", str(e))
            return
        self.state.siscom_coreg_path = out_path
        self.state.siscom_path = out_path
        try:
            self.state.last_browse_dir = str(Path(out_path).parent)
        except Exception:
            pass

        self._update_buttons()

        NeuXelecMessageDialog.information(
            None,
            "Save SISCOM",
            "Saved:\n" + out_path,
        )

    # ------------------------------------------------------------------
    # Coreg actions (PAIRWISE)
    # ------------------------------------------------------------------

    def _start_coreg_progress_animation(self, modality: str) -> None:
        """
        Start a smooth estimated progress animation for coregistration.

        The real ANTs process currently reports only 0 and 100, so this timer
        gives the user continuous feedback during the average ~1 min runtime.
        The bar intentionally slows down and stops around 96% until the worker
        really finishes.
        """
        self._coreg_progress_running = True
        self._coreg_progress_start_time = time.monotonic()
        self._coreg_progress_modality = str(modality)
        self._coreg_progress_duration_s = 180.0
        self._coreg_progress_max_before_done = 96

        if self.coreg_bar is not None:
            self.coreg_bar.setVisible(True)
            self.coreg_bar.setRange(0, 100)
            self.coreg_bar.setValue(0)
            self.coreg_bar.setFormat(
                f"{self._display_modality_name(modality)} → MRI 1: Preparing ANTs registration... %p%"
            )

        try:
            self._coreg_progress_timer.start()
        except Exception:
            pass

    def _coreg_progress_stage_text(self, value: int) -> str:
        """
        Human-readable status shown inside the progress bar.
        """
        if value < 8:
            return "Preparing ANTs registration"
        if value < 20:
            return "Initializing alignment"
        if value < 72:
            return "Optimizing transform"
        if value < 90:
            return "Resampling in T1 space"
        if value < 97:
            return "Finalizing registration"
        return "Done"

    def _tick_coreg_progress(self) -> None:
        """
        Smoothly animate the coregistration bar.

        Curve:
        - fast enough at the beginning to reassure the user;
        - progressively slower near the end;
        - capped at 96% until the real worker finishes.
        """
        if not bool(getattr(self, "_coreg_progress_running", False)):
            return

        if self.coreg_bar is None:
            return

        try:
            modality = self._display_modality_name(
                getattr(self, "_coreg_progress_modality", None) or "Image"
            )
            elapsed = max(0.0, time.monotonic() - float(self._coreg_progress_start_time))
            duration = max(1.0, float(getattr(self, "_coreg_progress_duration_s", 60.0)))

            t = min(elapsed / duration, 1.0)

            # Ease-out curve: quick at start, slow near the end.
            # Reaches max_before_done at around 60 seconds.
            eased = t

            start_value = 2
            max_before_done = int(getattr(self, "_coreg_progress_max_before_done", 96))
            estimated_value = int(round(start_value + (max_before_done - start_value) * eased))

            estimated_value = max(start_value, min(max_before_done, estimated_value))

            # Never go backwards if the real callback already moved the bar.
            current_value = int(self.coreg_bar.value())
            value = max(current_value, estimated_value)

            self.coreg_bar.setValue(value)
            self.coreg_bar.setFormat(
                f"{modality} → MRI 1: {self._coreg_progress_stage_text(value)}... %p%"
            )

        except Exception:
            pass

    def _finish_coreg_progress_animation(
        self,
        modality: str,
        success: bool = True,
    ) -> None:
        """
        Stop the estimated animation and display final status.
        """
        self._coreg_progress_running = False

        try:
            self._coreg_progress_timer.stop()
        except Exception:
            pass

        if self.coreg_bar is None:
            return

        if success:
            self.coreg_bar.setValue(100)
            self.coreg_bar.setFormat(f"{self._display_modality_name(modality)} → MRI 1: Done")
        else:
            # Keep the last progress value visible, but show failure.
            self.coreg_bar.setFormat(f"{self._display_modality_name(modality)} → MRI 1: Failed")

    def perform_coreg(self):
        fixed_path = getattr(self.state, "t1_path", None)
        if not fixed_path:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(), "Coregistration", "Please load MRI 1 first (fixed image)."
            )
            return

        modality = self._selected_modality()
        if modality is None:
            NeuXelecMessageDialog.warning(
                None,
                "Coregistration",
                "Please check EXACTLY ONE modality to coregister (MRI 2 or CT or PET or ictal SPECT or interictal SPECT).",
            )
            return

        moving_path = self._moving_path_for(modality)
        if not moving_path:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(),
                "Coregistration",
                f"Please load {self._display_modality_name(modality)} first.",
            )
            return

        # Ask where to save ONLY the transform matrix
        matrix_save_path = self._ask_transform_save_path(modality)
        if not matrix_save_path:
            return

        self._pending_matrix_save_path = matrix_save_path

        self._start_coreg_progress_animation(modality)

        self._set_busy(True)

        # Important: let ANTs work in a temporary directory
        # so the warped image is not user-saved automatically
        self.worker = CoregWorker(
            modality,
            fixed_path=fixed_path,
            moving_path=moving_path,
            transforms_dir=None,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_coreg_ok)
        self.worker.failed.connect(self._on_coreg_fail)
        self.worker.start()

    def _refresh_coreg_progress_ui(self) -> None:
        if self.coreg_bar is None:
            return

        modality = self._selected_modality()
        if modality is None:
            self.coreg_bar.setVisible(False)
            self.coreg_bar.setValue(0)
            self.coreg_bar.setFormat("")
            return

        is_done = self._coreg_image_for(modality) is not None

        if is_done:
            self.coreg_bar.setVisible(True)
            self.coreg_bar.setValue(100)
            self.coreg_bar.setFormat(f"{self._display_modality_name(modality)} → MRI 1: Done")
        else:
            self.coreg_bar.setVisible(True)
            self.coreg_bar.setValue(0)
            self.coreg_bar.setFormat(f"{self._display_modality_name(modality)} → MRI 1: Ready")

    def _on_progress(self, v: int):
        """
        Real worker progress callback.

        In the current ANTs backend, this usually emits only 0 and 100.
        While the smooth UI animation is running, we ignore 0 and cap 100 at 96
        until _on_coreg_ok() confirms that everything is really finished.
        """
        if self.coreg_bar is None:
            return

        try:
            v = int(v)
        except Exception:
            return

        if bool(getattr(self, "_coreg_progress_running", False)):
            if v >= 100:
                self.coreg_bar.setValue(
                    max(
                        int(self.coreg_bar.value()),
                        int(getattr(self, "_coreg_progress_max_before_done", 96)),
                    )
                )
                modality = self._display_modality_name(
                    getattr(self, "_coreg_progress_modality", None) or "Image"
                )
                self.coreg_bar.setFormat(f"{modality} → MRI 1: Finalizing registration... %p%")
            elif v > 0:
                self.coreg_bar.setValue(max(int(self.coreg_bar.value()), v))
            return

        self.coreg_bar.setValue(v)

    def _on_coreg_ok(self, modality: str, res: object):
        self.state.t1_sitk = res.fixed
        self._last_result = res  # type: ignore
        self._last_modality = modality  # type: ignore

        # Save only the transform matrix to the user-selected location
        try:
            src_mat = getattr(res, "affine_mat_path", None)
            dst_mat = getattr(self, "_pending_matrix_save_path", None)
            if src_mat and dst_mat:
                shutil.copy2(src_mat, dst_mat)
        except Exception as e:
            NeuXelecMessageDialog.warning(
                None,
                "Transform matrix",
                f"Coregistration succeeded, but the transform matrix could not be saved:\n{e}",
            )

        self._pending_matrix_save_path = None

        if modality == "T2":
            self.state.t2_coreg_in_t1 = res.moving_in_fixed
            self.state.t2_validated = False
        elif modality == "CT":
            self.state.ct_coreg_in_t1 = res.moving_in_fixed
            self.state.ct_validated = False
        elif modality == "PET":
            self.state.pet_coreg_in_t1 = res.moving_in_fixed
            self.state.pet_validated = False
        elif modality == "ictalSPECT":
            self.state.ictal_spect_coreg_in_t1 = res.moving_in_fixed
            self.state.ictal_spect_validated = False
        elif modality == "interictalSPECT":
            self.state.interictal_spect_coreg_in_t1 = res.moving_in_fixed
            self.state.interictal_spect_validated = False

        self._finish_coreg_progress_animation(
            modality,
            success=True,
        )

        self._set_busy(False)
        self._update_buttons()
        NeuXelecMessageDialog.information(
            self._dialog_parent(),
            "Coregistration",
            f"{self._display_modality_name(modality)} → MRI 1 done. Click 'Check coregistration'.",
        )

    def _on_coreg_fail(self, modality: str, msg: str):
        self._finish_coreg_progress_animation(
            modality,
            success=False,
        )

        self._set_busy(False)
        self._update_buttons()
        NeuXelecMessageDialog.critical(
            self._dialog_parent(), f"Coregistration failed ({modality})", msg
        )

    # ------------------------------------------------------------------
    # Check coregistration
    # ------------------------------------------------------------------
    def _coreg_image_for(self, modality: Modality) -> sitk.Image | None:
        return {
            "T2": getattr(self.state, "t2_coreg_in_t1", None),
            "CT": getattr(self.state, "ct_coreg_in_t1", None),
            "PET": getattr(self.state, "pet_coreg_in_t1", None),
            "ictalSPECT": getattr(self.state, "ictal_spect_coreg_in_t1", None),
            "interictalSPECT": getattr(self.state, "interictal_spect_coreg_in_t1", None),
        }[modality]

    def _apply_brainmask_to_image(self, img: sitk.Image) -> sitk.Image:
        brainmask = getattr(self.state, "brainmask_sitk", None)
        if brainmask is None:
            return img

        mask = sitk.Resample(
            brainmask,
            img,
            sitk.Transform(3, sitk.sitkIdentity),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )

        img_arr = sitk.GetArrayFromImage(img).astype(np.float32, copy=False)
        mask_arr = sitk.GetArrayFromImage(mask) > 0

        out_arr = np.zeros_like(img_arr, dtype=np.float32)
        out_arr[mask_arr] = img_arr[mask_arr]

        out = sitk.GetImageFromArray(out_arr)
        out.CopyInformation(img)
        return out

    def check_coreg(self):
        if getattr(self.state, "t1_sitk", None) is None and getattr(self.state, "t1_path", None):
            try:
                self.state.t1_sitk = sitk.ReadImage(self.state.t1_path)
            except Exception:
                self.state.t1_sitk = None

        if self.state.t1_sitk is None:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(), "Check coregistration", "No T1 available."
            )
            return

        modality = self._selected_modality()
        if modality is None:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(),
                "Check coregistration",
                "Please select exactly one modality to check.",
            )
            return

        moving_img = self._coreg_image_for(modality)
        moving_name = modality

        if moving_img is None:
            moving_path = self._moving_path_for(modality)
            if not moving_path:
                NeuXelecMessageDialog.warning(
                    self._dialog_parent(),
                    "Check coregistration",
                    "No image loaded for this modality.",
                )
                return
            try:
                raw = sitk.ReadImage(moving_path)
                moving_img = sitk.Resample(
                    raw,
                    self.state.t1_sitk,
                    sitk.Transform(3, sitk.sitkIdentity),
                    sitk.sitkLinear,
                    0.0,
                    sitk.sitkFloat32,
                )
                moving_name = f"{modality} (not coregistered)"
            except Exception as e:
                NeuXelecMessageDialog.critical(
                    self._dialog_parent(), "Check coregistration", f"Failed to load image:\n{e}"
                )
                return

        dlg = OverlayViewer(
            fixed_t1=self.state.t1_sitk,
            moving_in_t1=moving_img,
            moving_name=moving_name,
            parent=self._dialog_parent(),
        )

        result = dlg.exec()

        if result == QDialog.Accepted:
            refined_img = dlg.corrected_moving_image()

            if modality == "CT":
                self.state.ct_coreg_in_t1 = refined_img
                self.state.ct_in_t1 = refined_img
                self.state.ct_validated = True

                # CT can now be used for electrode reconstruction during
                # the current session only.
                self.state.ct_ready_for_reconstruction = True

                # Push the validated CT to 3D View.
                vp = self._view3d()
                if vp is not None and hasattr(vp, "set_ct"):
                    try:
                        vp.set_ct(refined_img)
                    except Exception:
                        pass

                # Refresh reconstruction immediately so the warning message
                reco = getattr(self.state, "reco_page", None)
                if reco is not None:
                    try:
                        if hasattr(reco, "force_refresh_after_coreg_validation"):
                            reco.force_refresh_after_coreg_validation()
                        else:
                            reco.init_from_volume()
                            reco.render_all()
                    except Exception:
                        pass

                op = getattr(self.state, "oblique_page", None)
                if op is not None:
                    try:
                        if hasattr(op, "refresh_available_modalities"):
                            op.refresh_available_modalities()
                        elif hasattr(op, "render_all"):
                            op.render_all()
                    except Exception:
                        pass
            elif modality == "T2":
                self.state.t2_coreg_in_t1 = refined_img
                self.state.t2_validated = True

                # Push T2 to Oblique Slice
                op = getattr(self.state, "oblique_page", None)
                if op is not None:
                    try:
                        if hasattr(op, "refresh_mri_source_controls"):
                            op.refresh_mri_source_controls()
                        elif hasattr(op, "render_all"):
                            op.render_all()
                    except Exception:
                        pass

                # Push T2 to 3D View
                vp = self._view3d()
                if vp is not None:
                    try:
                        if hasattr(vp, "set_t2"):
                            vp.set_t2(refined_img)
                        elif hasattr(vp, "_refresh_3d_mri_source_controls"):
                            vp._refresh_3d_mri_source_controls()
                    except Exception:
                        pass
            elif modality == "PET":
                pet_img_to_store = refined_img

                apply_brainmask = NeuXelecMessageDialog.question(
                    self._dialog_parent(),
                    "Apply brain mask",
                    "Do you want to apply the brain mask to the validated PET?",
                    accept_text="Yes",
                    reject_text="No",
                )

                if apply_brainmask:
                    brainmask = getattr(self.state, "brainmask_sitk", None)

                    if brainmask is None:
                        NeuXelecMessageDialog.information(
                            None,
                            "Brain mask required",
                            "No brain mask is currently available.\n\n"
                            "Please generate or load a brain mask first, then validate the PET coregistration again.",
                        )
                        return
                    else:
                        try:
                            pet_img_to_store = self._apply_brainmask_to_image(refined_img)
                        except Exception as e:
                            NeuXelecMessageDialog.warning(
                                None,
                                "Brain mask",
                                f"Failed to apply brain mask to PET:\n{e}\n\n"
                                "The validated PET will be kept without brain masking.",
                            )
                            pet_img_to_store = refined_img

                self.state.pet_coreg_in_t1 = pet_img_to_store
                self.state.pet_validated = True
                self.state.pet_in_t1 = pet_img_to_store

                op = getattr(self.state, "oblique_page", None)
                if op is not None:
                    try:
                        if hasattr(op, "refresh_available_modalities"):
                            op.refresh_available_modalities()
                        elif hasattr(op, "render_all"):
                            op.render_all()
                    except Exception:
                        pass

                # Push PET to 3D (now that it is validated)
                vp = self._view3d()
                if vp is not None:
                    try:
                        vp.set_pet(pet_img_to_store)
                    except Exception:
                        pass

                # Enable 3D checkbox (pet) only after validation
                self._enable_3d_checkbox("chk_3d_showPET", True)

            elif modality == "ictalSPECT":
                self.state.ictal_spect_coreg_in_t1 = refined_img
                self.state.ictal_spect_validated = True

            elif modality == "interictalSPECT":
                self.state.interictal_spect_coreg_in_t1 = refined_img
                self.state.interictal_spect_validated = True

            self._update_buttons()

            try:
                from PySide6.QtWidgets import QApplication

                QApplication.processEvents()
            except Exception:
                pass

            NeuXelecMessageDialog.information(
                self._dialog_parent(), "Validated", f"{modality} validated."
            )

    # ------------------------------------------------------------------
    # Save (coreg modalities)
    # ------------------------------------------------------------------
    def save_coreg(self, modality: Modality):
        validated = {
            "T2": bool(getattr(self.state, "t2_validated", False)),
            "CT": bool(getattr(self.state, "ct_validated", False)),
            "PET": bool(getattr(self.state, "pet_validated", False)),
            "ictalSPECT": bool(getattr(self.state, "ictal_spect_validated", False)),
            "interictalSPECT": bool(getattr(self.state, "interictal_spect_validated", False)),
        }[modality]

        coreg_img = self._coreg_image_for(modality)
        moving_path = self._moving_path_for(modality)

        if not moving_path:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(), f"Save {modality}", f"No {modality} loaded."
            )
            return

        if (not validated) or (coreg_img is None):
            save_anyway = NeuXelecMessageDialog.question(
                self._dialog_parent(),
                "Not coregistered",
                (
                    "This image has not been coregistered.\n\n"
                    "Are you sure you want to save it anyway?"
                ),
                accept_text="Save anyway",
                reject_text="Cancel",
            )

            if not save_anyway:
                return

            start_dir = self._preferred_save_start_path(modality)
            raw_label = self._filename_label_for_modality(modality)

            default_path = str(
                Path(start_dir)
                / self._default_filename(
                    f"{raw_label}_raw",
                    ".nii.gz",
                )
            )

            out_path, _ = QFileDialog.getSaveFileName(
                self._dialog_parent(),
                f"Save {modality} (raw, not coregistered)",
                default_path,
                "NIfTI (*.nii *.nii.gz);;All files (*.*)",
            )
            if not out_path:
                return

            try:
                raw_img = sitk.ReadImage(moving_path)
                save_nifti(raw_img, out_path)
            except Exception as e:
                NeuXelecMessageDialog.critical(self._dialog_parent(), "Save failed", str(e))
                return

            NeuXelecMessageDialog.information(
                self._dialog_parent(), f"Save {modality}", "Saved (raw):\n" + out_path
            )
            return

        start_dir = self._preferred_save_start_path(modality)
        moving_label = self._filename_label_for_modality(modality)
        fixed_label = self._mri1_filename_label()

        default_path = str(
            Path(start_dir)
            / self._default_filename(
                f"{moving_label}_to_{fixed_label}",
                ".nii.gz",
            )
        )

        out_path, _ = QFileDialog.getSaveFileName(
            self._dialog_parent(),
            f"Save {self._display_modality_name(modality)} in MRI 1",
            default_path,
            "NIfTI (*.nii *.nii.gz);;All files (*.*)",
        )
        if not out_path:
            return

        try:
            save_nifti(coreg_img, out_path)
        except Exception as e:
            NeuXelecMessageDialog.critical(self._dialog_parent(), "Save failed", str(e))
            return
        # IMPORTANT: store saved coreg path in state
        if modality == "T2":
            self.state.t2_coreg_path = out_path
        elif modality == "CT":
            self.state.ct_coreg_path = out_path
        elif modality == "PET":
            self.state.pet_coreg_path = out_path
        elif modality == "ictalSPECT":
            self.state.ictal_spect_coreg_path = out_path
        elif modality == "interictalSPECT":
            self.state.interictal_spect_coreg_path = out_path
        try:
            self.state.last_browse_dir = str(Path(out_path).parent)
        except Exception:
            pass

        self._update_buttons()

        NeuXelecMessageDialog.information(
            None,
            f"Save {modality}",
            "Saved:\n" + out_path,
        )

    def save_all_coreg_validated(self) -> None:
        """
        Save all currently available outputs in one destination:
        - validated coregistered/resampled images in T1 space;
        - validated SISCOM map, when available;
        - generated or loaded brain mask, when available.
        """
        fixed_label = self._mri1_filename_label()

        modalities = [
            (
                "T2",
                "t2_validated",
                "t2_coreg_path",
                self._default_filename(
                    f"{self._mri2_filename_label()}_to_{fixed_label}", ".nii.gz"
                ),
            ),
            (
                "CT",
                "ct_validated",
                "ct_coreg_path",
                self._default_filename(f"CT_to_{fixed_label}", ".nii.gz"),
            ),
            (
                "PET",
                "pet_validated",
                "pet_coreg_path",
                self._default_filename(f"PET_to_{fixed_label}", ".nii.gz"),
            ),
            (
                "ictalSPECT",
                "ictal_spect_validated",
                "ictal_spect_coreg_path",
                self._default_filename(f"ictalSPECT_to_{fixed_label}", ".nii.gz"),
            ),
            (
                "interictalSPECT",
                "interictal_spect_validated",
                "interictal_spect_coreg_path",
                self._default_filename(f"interictalSPECT_to_{fixed_label}", ".nii.gz"),
            ),
        ]

        to_save = []

        for modality, validated_attr, path_attr, filename in modalities:
            validated = bool(getattr(self.state, validated_attr, False))
            img = self._coreg_image_for(modality)

            if validated and img is not None:
                to_save.append((modality, img, path_attr, filename))

        # Optional: also save validated SISCOM if available.
        siscom_validated = bool(getattr(self.state, "siscom_validated", False))
        siscom_img = getattr(self.state, "siscom_coreg_in_t1", None)

        if siscom_validated and siscom_img is not None:
            to_save.append(
                (
                    "SISCOM",
                    siscom_img,
                    "siscom_coreg_path",
                    self._default_filename(f"SISCOM_to_{fixed_label}", ".nii.gz"),
                )
            )

        brainmask_img = getattr(self.state, "brainmask_sitk", None)
        if brainmask_img is None:
            brainmask_source_path = getattr(self.state, "brainmask_path", None) or getattr(
                self.state, "brainmask_generated_path", None
            )
            if brainmask_source_path:
                try:
                    brainmask_img = sitk.ReadImage(str(brainmask_source_path))
                except Exception:
                    brainmask_img = None

        if brainmask_img is not None:
            to_save.append(
                (
                    "Brain mask",
                    brainmask_img,
                    "brainmask_path",
                    self._default_filename(f"{fixed_label}_brainmask", ".nii.gz"),
                )
            )

        if not to_save:
            NeuXelecMessageDialog.warning(
                None,
                "Save all",
                "No validated image, SISCOM map or brain mask is currently available for export.",
            )
            return

        start_dir = self._t1_preferred_dir()

        out_dir = QFileDialog.getExistingDirectory(
            self._dialog_parent(),
            "Select folder to save all validated coregistered images",
            start_dir,
        )

        if not out_dir:
            return

        out_dir = str(out_dir)
        saved = []
        failed = []

        for modality, img, path_attr, filename in to_save:
            out_path = str(Path(out_dir) / filename)

            try:
                save_nifti(img, out_path)
                setattr(self.state, path_attr, out_path)

                # Keep linked state synchronized with exported derived outputs.
                if modality == "SISCOM":
                    self.state.siscom_path = out_path

                elif modality == "Brain mask":
                    self.state.brainmask_path = out_path
                    self.state.brainmask_generated = True
                    self.state.brainmask_saved = True
                    self.state.brainmask_generated_path = out_path

                    if getattr(self, "le_load_brainmask", None) is not None:
                        self.le_load_brainmask.setText(out_path)
                        self.le_load_brainmask.setCursorPosition(0)
                        self.le_load_brainmask.setToolTip(out_path)

                saved.append(f"{modality}: {out_path}")

            except Exception as e:
                failed.append(f"{modality}: {e}")

        try:
            self.state.last_browse_dir = out_dir
        except Exception:
            pass

        self._update_buttons()

        msg = ""

        if saved:
            msg += "Saved:\n" + "\n".join(saved)

        if failed:
            if msg:
                msg += "\n\n"
            msg += "Failed:\n" + "\n".join(failed)

        if failed:
            NeuXelecMessageDialog.warning(self._dialog_parent(), "Save all", msg)
        else:
            NeuXelecMessageDialog.information(self._dialog_parent(), "Save all", msg)

    # ------------------------------------------------------------------
    # Enable / disable rules
    # ------------------------------------------------------------------
    def _set_busy(self, busy: bool):
        if self.btn_perform is not None:
            self.btn_perform.setEnabled(not busy)
        if self.btn_check_coreg is not None:
            self.btn_check_coreg.setEnabled(not busy and self.btn_check_coreg.isEnabled())

        for b in (
            self.btn_save_t2,
            self.btn_save_ct,
            self.btn_save_pet,
            self.btn_save_ictal,
            self.btn_save_interictal,
            self.btn_save_siscom,
            getattr(self, "btn_save_all", None),
            self.btn_save_brainmask,
            self.btn_save_iso_surface,
        ):
            if b is not None:
                b.setEnabled((not busy) and b.isEnabled())

    def load_brainmask(self) -> None:
        """Load an existing brain mask file and push it to state + 3D view."""
        start_dir = self._t1_preferred_dir()
        path, _ = QFileDialog.getOpenFileName(
            self._dialog_parent(),
            "Select Brain Mask file",
            start_dir,
            "NIfTI files (*.nii *.nii.gz);;All files (*.*)",
        )
        if not path:
            return
        self.state.brainmask_path = path
        self.state.brainmask_generated = True
        self.state.brainmask_saved = True
        self.state.brainmask_generated_path = None
        try:
            self.state.brainmask_sitk = sitk.ReadImage(path)
        except Exception:
            self.state.brainmask_sitk = None

        if getattr(self, "le_load_brainmask", None) is not None:
            try:
                self.le_load_brainmask.setText(path)
            except Exception:
                pass

        vp = self._view3d()
        if vp is not None:
            try:
                vp.set_brainmask(self.state.brainmask_sitk, brainmask_path=path)
            except Exception:
                pass
        op = getattr(self.state, "oblique_page", None)
        if op is not None:
            try:
                if hasattr(op, "_last_brain_key"):
                    op._last_brain_key = None
                if hasattr(op, "_last_brain_kind"):
                    op._last_brain_kind = None

                if hasattr(op, "_schedule_refresh"):
                    op._schedule_refresh(slices=True, brain=True)
                elif hasattr(op, "render_all"):
                    op.render_all()
            except Exception:
                pass
        if getattr(self, "btn_save_brainmask", None) is not None:
            self.btn_save_brainmask.setEnabled(True)
        self._update_buttons()

    def load_iso_surface(self) -> None:
        """Load an iso-surface mesh previously saved by Neuxelec (.npz) and push to 3D view."""
        start_dir = self.state.last_browse_dir or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self._dialog_parent(),
            "Select Iso-Surface file",
            start_dir,
            "NPZ files (*.npz);;All files (*.*)",
        )
        if not path:
            return
        try:
            mesh = np.load(path, allow_pickle=True)
            if "mesh" in mesh:
                mesh_obj = mesh["mesh"].item()
            else:
                points = mesh["points"] if "points" in mesh else mesh.get("verts")
                faces = mesh["faces"] if "faces" in mesh else None
                mesh_obj = {"points": points, "faces": faces}
        except Exception:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(), "Iso-Surface", "Failed to load iso-surface file."
            )
            return

        self.state.iso_surface_mesh = mesh_obj
        self.state.iso_surface_saved_path = path

        if getattr(self, "le_load_isosurface", None) is not None:
            try:
                self.le_load_isosurface.setText(path)
            except Exception:
                pass

        vp = self._view3d()
        if vp is not None:
            try:
                if hasattr(vp, "set_iso_surface"):
                    vp.set_iso_surface(mesh_obj)
            except Exception:
                pass

        if getattr(self, "btn_save_iso_surface", None) is not None:
            self.btn_save_iso_surface.setEnabled(True)
        self._update_buttons()

    def restore_from_state(self) -> None:
        # ---------- paths in line edits ----------
        try:
            if self.le_t1 is not None:
                self.le_t1.setText(str(getattr(self.state, "t1_path", "") or ""))
                self.le_t1.setCursorPosition(0)
        except Exception:
            pass

        try:
            if self.le_t2 is not None:
                txt = (
                    getattr(self.state, "t2_coreg_path", None)
                    or getattr(self.state, "t2_path", None)
                    or ""
                )
                self.le_t2.setText(str(txt))
                self.le_t2.setCursorPosition(0)
        except Exception:
            pass

        try:
            if self.le_ct is not None:
                txt = (
                    getattr(self.state, "ct_coreg_path", None)
                    or getattr(self.state, "ct_path", None)
                    or ""
                )
                self.le_ct.setText(str(txt))
                self.le_ct.setCursorPosition(0)
        except Exception:
            pass

        try:
            if self.le_pet is not None:
                txt = (
                    getattr(self.state, "pet_coreg_path", None)
                    or getattr(self.state, "pet_path", None)
                    or ""
                )
                self.le_pet.setText(str(txt))
                self.le_pet.setCursorPosition(0)
        except Exception:
            pass
        try:
            pet_path = getattr(self.state, "pet_coreg_path", None) or getattr(
                self.state, "pet_path", None
            )

            if (
                pet_path
                and bool(getattr(self.state, "pet_validated", False))
                and getattr(self.state, "pet_coreg_in_t1", None) is None
            ):
                img = sitk.ReadImage(pet_path)

                self.state.pet_coreg_in_t1 = img
                self.state.pet_in_t1 = img

                vp = self._view3d()
                if vp is not None:
                    try:
                        vp.set_pet(
                            img,
                            pet_path=pet_path,
                            activate=True,
                        )
                    except TypeError:
                        vp.set_pet(img, pet_path=pet_path)
                        try:
                            if getattr(vp, "chk_pet", None) is not None:
                                vp.chk_pet.blockSignals(True)
                                vp.chk_pet.setEnabled(True)
                                vp.chk_pet.setChecked(True)
                                vp.chk_pet.blockSignals(False)
                            if hasattr(vp, "_refresh_pet_only"):
                                vp._refresh_pet_only()
                        except Exception:
                            pass
                    except Exception:
                        pass

                op = getattr(self.state, "oblique_page", None)
                if op is not None:
                    try:
                        op.refresh_available_modalities(
                            refresh=False,
                            activate_validated=True,
                        )
                    except TypeError:
                        op.refresh_available_modalities(refresh=False)
                        try:
                            if getattr(op, "chk_pet", None) is not None:
                                op.chk_pet.blockSignals(True)
                                op.chk_pet.setEnabled(True)
                                op.chk_pet.setChecked(True)
                                op.chk_pet.blockSignals(False)
                            if hasattr(op, "_invalidate_oblique_image_caches"):
                                op._invalidate_oblique_image_caches()
                        except Exception:
                            pass
                    except Exception:
                        pass

        except Exception:
            pass
        try:
            if self.le_ictal is not None:
                txt = (
                    getattr(self.state, "ictal_spect_coreg_path", None)
                    or getattr(self.state, "ictal_spect_path", None)
                    or ""
                )
                self.le_ictal.setText(str(txt))
                self.le_ictal.setCursorPosition(0)
        except Exception:
            pass

        try:
            if self.le_interictal is not None:
                txt = (
                    getattr(self.state, "interictal_spect_coreg_path", None)
                    or getattr(self.state, "interictal_spect_path", None)
                    or ""
                )
                self.le_interictal.setText(str(txt))
                self.le_interictal.setCursorPosition(0)
        except Exception:
            pass

        try:
            if self.le_siscom is not None:
                txt = (
                    getattr(self.state, "siscom_coreg_path", None)
                    or getattr(self.state, "siscom_path", None)
                    or ""
                )
                self.le_siscom.setText(str(txt))
                self.le_siscom.setCursorPosition(0)
        except Exception:
            pass
        try:
            siscom_path = getattr(self.state, "siscom_coreg_path", None) or getattr(
                self.state, "siscom_path", None
            )
            if siscom_path and getattr(self.state, "siscom_coreg_in_t1", None) is None:
                img = sitk.ReadImage(siscom_path)
                self.state.siscom_coreg_in_t1 = img
                self.state.siscom_z_in_t1 = img
                self.state.siscom_thr_in_t1 = None

                vp = self._view3d()
                if vp is not None:
                    try:
                        vp.set_siscom(img, siscom_path=siscom_path)
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            if self.le_parcel1 is not None:
                self.le_parcel1.setText(str(getattr(self.state, "parcel1_path", "") or ""))
                self.le_parcel1.setCursorPosition(0)
        except Exception:
            pass
        try:
            parcel1_path = getattr(self.state, "parcel1_path", None)
            if parcel1_path:
                img = sitk.ReadImage(parcel1_path)
                self.state.parcel1_img = img

                # Rebuild LUT exactly like load_parcellation1()
                self.state.parcellation1_lut = {}

                filename = Path(parcel1_path).name.lower()
                lut_path = (
                    Path(__file__).resolve().parent.parent / "utils" / "FreeSurferColorLUT.txt"
                )

                if "aparc+aseg" in filename and lut_path.exists():
                    self.state.parcellation1_lut = self._load_freesurfer_lut_dict(lut_path)

                # Fallback detection from labels
                if not getattr(self.state, "parcellation1_lut", {}):
                    arr = sitk.GetArrayViewFromImage(img)
                    uniq = np.unique(arr)
                    if np.any((uniq >= 1000) & (uniq < 5000)) and lut_path.exists():
                        self.state.parcellation1_lut = self._load_freesurfer_lut_dict(lut_path)

                vp = self._view3d()
                if vp is not None and hasattr(vp, "set_parcellation1"):
                    try:
                        vp.set_parcellation1(img, parcel1_path)
                    except Exception:
                        pass

                op = getattr(self.state, "oblique_page", None)
                if op is not None and hasattr(op, "set_parcellation1"):
                    try:
                        op.set_parcellation1(img, parcel1_path)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            if self.le_parcel2 is not None:
                self.le_parcel2.setText(str(getattr(self.state, "parcel2_path", "") or ""))
                self.le_parcel2.setCursorPosition(0)
        except Exception:
            pass

        try:
            if getattr(self, "le_lh_pial", None) is not None:
                value = str(getattr(self.state, "lh_pial_path", "") or "")
                self.le_lh_pial.setText(value)
                self.le_lh_pial.setCursorPosition(0)
                self.le_lh_pial.setToolTip(value)

            if getattr(self, "le_rh_pial", None) is not None:
                value = str(getattr(self.state, "rh_pial_path", "") or "")
                self.le_rh_pial.setText(value)
                self.le_rh_pial.setCursorPosition(0)
                self.le_rh_pial.setToolTip(value)
        except Exception:
            pass
        try:
            lh = getattr(self.state, "lh_pial_path", None)
            rh = getattr(self.state, "rh_pial_path", None)

            if lh and rh and bool(getattr(self.state, "pial_surfaces_available", False)):
                vp = self._view3d()

                if vp is not None and hasattr(vp, "set_pial_surfaces"):
                    vp.set_pial_surfaces(
                        lh_path=lh,
                        rh_path=rh,
                        assume_lps=bool(getattr(self.state, "pial_surfaces_assume_lps", True)),
                    )

        except Exception:
            pass
        try:
            if getattr(self, "le_load_brainmask", None) is not None:
                self.le_load_brainmask.setText(str(getattr(self.state, "brainmask_path", "") or ""))
        except Exception:
            pass

        try:
            if getattr(self, "le_load_isosurface", None) is not None:
                self.le_load_isosurface.setText(
                    str(getattr(self.state, "iso_surface_saved_path", "") or "")
                )
        except Exception:
            pass

        # ---------- enable checkboxes based on loaded files ----------
        try:
            self._enable_checkbox(self.chk_t2, bool(getattr(self.state, "t2_path", None)))
            self._enable_checkbox(self.chk_ct, bool(getattr(self.state, "ct_path", None)))
            self._enable_checkbox(self.chk_pet, bool(getattr(self.state, "pet_path", None)))
            self._enable_checkbox(
                self.chk_ictal, bool(getattr(self.state, "ictal_spect_path", None))
            )
            self._enable_checkbox(
                self.chk_interictal, bool(getattr(self.state, "interictal_spect_path", None))
            )
        except Exception:
            pass

        try:
            if self.chk_t1 is not None:
                self.chk_t1.setEnabled(bool(getattr(self.state, "t1_path", None)))
            self._force_t1_checked()
        except Exception:
            pass

        # ---------- restore the selected modality checkbox ----------
        # If a coreg image exists, prefer showing that modality as selected.
        selected_modality = None
        if getattr(self.state, "ct_coreg_in_t1", None) is not None or getattr(
            self.state, "ct_validated", False
        ):
            selected_modality = "CT"
        elif getattr(self.state, "t2_coreg_in_t1", None) is not None or getattr(
            self.state, "t2_validated", False
        ):
            selected_modality = "T2"
        elif getattr(self.state, "pet_coreg_in_t1", None) is not None or getattr(
            self.state, "pet_validated", False
        ):
            selected_modality = "PET"
        elif getattr(self.state, "ictal_spect_coreg_in_t1", None) is not None or getattr(
            self.state, "ictal_spect_validated", False
        ):
            selected_modality = "ictalSPECT"
        elif getattr(self.state, "interictal_spect_coreg_in_t1", None) is not None or getattr(
            self.state, "interictal_spect_validated", False
        ):
            selected_modality = "interictalSPECT"

        try:
            for cb in (self.chk_t2, self.chk_ct, self.chk_pet, self.chk_ictal, self.chk_interictal):
                if cb is not None:
                    cb.blockSignals(True)
                    cb.setChecked(False)
                    cb.blockSignals(False)

            if selected_modality == "T2" and self.chk_t2 is not None and self.chk_t2.isEnabled():
                self.chk_t2.blockSignals(True)
                self.chk_t2.setChecked(True)
                self.chk_t2.blockSignals(False)
            elif selected_modality == "CT" and self.chk_ct is not None and self.chk_ct.isEnabled():
                self.chk_ct.blockSignals(True)
                self.chk_ct.setChecked(True)
                self.chk_ct.blockSignals(False)
            elif (
                selected_modality == "PET" and self.chk_pet is not None and self.chk_pet.isEnabled()
            ):
                self.chk_pet.blockSignals(True)
                self.chk_pet.setChecked(True)
                self.chk_pet.blockSignals(False)
            elif (
                selected_modality == "ictalSPECT"
                and self.chk_ictal is not None
                and self.chk_ictal.isEnabled()
            ):
                self.chk_ictal.blockSignals(True)
                self.chk_ictal.setChecked(True)
                self.chk_ictal.blockSignals(False)
            elif (
                selected_modality == "interictalSPECT"
                and self.chk_interictal is not None
                and self.chk_interictal.isEnabled()
            ):
                self.chk_interictal.blockSignals(True)
                self.chk_interictal.setChecked(True)
                self.chk_interictal.blockSignals(False)
        except Exception:
            pass

        # ---------- refresh button states ----------
        self._update_buttons()

    def _update_buttons(self):
        t1_loaded = bool(getattr(self.state, "t1_path", None))

        if getattr(self, "btn_brainmask", None) is not None:
            self.btn_brainmask.setEnabled(t1_loaded)

        if getattr(self, "btn_iso_surface", None) is not None:
            self.btn_iso_surface.setEnabled(t1_loaded)

        if self.chk_t1 is not None:
            self.chk_t1.setEnabled(t1_loaded)
            if t1_loaded:
                self._force_t1_checked()
            else:
                self.chk_t1.setChecked(False)

        self._enable_checkbox(self.chk_t2, bool(getattr(self.state, "t2_path", None)))
        self._enable_checkbox(self.chk_ct, bool(getattr(self.state, "ct_path", None)))
        self._enable_checkbox(self.chk_pet, bool(getattr(self.state, "pet_path", None)))
        self._enable_checkbox(self.chk_ictal, bool(getattr(self.state, "ictal_spect_path", None)))
        self._enable_checkbox(
            self.chk_interictal, bool(getattr(self.state, "interictal_spect_path", None))
        )

        can_perform = t1_loaded and (self._selected_modality() is not None)
        if self.btn_perform is not None:
            self.btn_perform.setEnabled(can_perform)

        modality = self._selected_modality()
        moving_path = self._moving_path_for(modality) if modality else None
        can_check = bool(t1_loaded and modality and moving_path)
        if self.btn_check_coreg is not None:
            self.btn_check_coreg.setEnabled(can_check)

        if self.btn_save_t2 is not None:
            self.btn_save_t2.setEnabled(bool(getattr(self.state, "t2_validated", False)))
        if self.btn_save_ct is not None:
            self.btn_save_ct.setEnabled(bool(getattr(self.state, "ct_validated", False)))
        if self.btn_save_pet is not None:
            self.btn_save_pet.setEnabled(bool(getattr(self.state, "pet_validated", False)))
        if self.btn_save_ictal is not None:
            self.btn_save_ictal.setEnabled(
                bool(getattr(self.state, "ictal_spect_validated", False))
            )
        if self.btn_save_interictal is not None:
            self.btn_save_interictal.setEnabled(
                bool(getattr(self.state, "interictal_spect_validated", False))
            )
        if getattr(self, "btn_save_all", None) is not None:
            has_any_validated_modality = any(
                [
                    bool(getattr(self.state, "t2_validated", False))
                    and self._coreg_image_for("T2") is not None,
                    bool(getattr(self.state, "ct_validated", False))
                    and self._coreg_image_for("CT") is not None,
                    bool(getattr(self.state, "pet_validated", False))
                    and self._coreg_image_for("PET") is not None,
                    bool(getattr(self.state, "ictal_spect_validated", False))
                    and self._coreg_image_for("ictalSPECT") is not None,
                    bool(getattr(self.state, "interictal_spect_validated", False))
                    and self._coreg_image_for("interictalSPECT") is not None,
                ]
            )

            has_validated_siscom = bool(
                getattr(self.state, "siscom_validated", False)
                and getattr(self.state, "siscom_coreg_in_t1", None) is not None
            )

            has_brainmask_available = bool(
                getattr(self.state, "brainmask_sitk", None) is not None
                or getattr(self.state, "brainmask_path", None)
                or getattr(self.state, "brainmask_generated_path", None)
            )

            has_any_saveable_output = bool(
                has_any_validated_modality or has_validated_siscom or has_brainmask_available
            )

            self.btn_save_all.setEnabled(has_any_saveable_output)

        can_perf_siscom = bool(getattr(self.state, "ictal_spect_validated", False)) and bool(
            getattr(self.state, "interictal_spect_validated", False)
        )

        if getattr(self, "btn_perf_siscom", None) is not None:
            self.btn_perf_siscom.setEnabled(bool(can_perf_siscom))

        can_check_siscom = bool(
            getattr(self.state, "siscom_thr_in_t1", None) is not None
            or getattr(self.state, "siscom_z_in_t1", None) is not None
        )

        if getattr(self, "btn_check_siscom", None) is not None:
            self.btn_check_siscom.setEnabled(can_check_siscom)

        if getattr(self, "btn_save_siscom", None) is not None:
            self.btn_save_siscom.setEnabled(bool(getattr(self.state, "siscom_validated", False)))

        if getattr(self, "btn_save_brainmask", None) is not None:
            self.btn_save_brainmask.setEnabled(
                bool(
                    getattr(self.state, "brainmask_path", None)
                    or getattr(self.state, "brainmask_sitk", None)
                )
            )

        if getattr(self, "btn_save_iso_surface", None) is not None:
            self.btn_save_iso_surface.setEnabled(
                bool(getattr(self.state, "iso_surface_mesh", None))
            )

        self._enable_3d_checkbox(
            "chk_3d_showBrainmask",
            bool(
                getattr(self.state, "brainmask_sitk", None)
                or getattr(self.state, "brainmask_path", None)
            ),
        )
        self._enable_3d_checkbox(
            "chk_3d_showIsoSurface", bool(getattr(self.state, "iso_surface_mesh", None))
        )
        self._enable_3d_checkbox(
            "chk_3d_showPET", bool(getattr(self.state, "pet_validated", False))
        )
        self._enable_3d_checkbox(
            "chk_3d_showSISCOM", bool(getattr(self.state, "siscom_validated", False))
        )
        self._enable_3d_checkbox(
            "chk_3d_showPialsurface",
            bool(
                getattr(self.state, "lh_pial_path", None)
                and getattr(self.state, "rh_pial_path", None)
            ),
        )

        # Keep the page visual state synchronized with the workflow state.
        # Styling is handled by dynamic Qt properties declared in MainWindow.ui.
        self._sync_files_workflow_visual_states()
        self._sync_files_status_overview()

        self._refresh_coreg_progress_ui()

    @staticmethod
    def _set_dynamic_property(widget, name: str, value) -> None:
        """Apply a Qt dynamic property and refresh only the affected widget."""
        if widget is None:
            return

        if widget.property(name) == value:
            return

        widget.setProperty(name, value)

        try:
            style = widget.style()
            style.unpolish(widget)
            style.polish(widget)
        except Exception:
            pass

        widget.update()

    def _set_status_item(self, key: str, loaded: bool, text: str, state: str = "missing") -> None:
        pill = getattr(self, "_status_pills", {}).get(key)
        label = getattr(self, "_status_texts", {}).get(key)

        if pill is not None:
            pill.setText("●")
            self._set_dynamic_property(pill, "status", state)

        if label is not None:
            label.setText(text)
            self._set_dynamic_property(label, "status", state)

    def _sync_files_status_overview(self) -> None:
        """Update compact status pills in the redesigned Files/Coreg page.

        Color convention:
        - blue = loaded / needs coregistration
        - green = coregistered / available
        - grey = missing
        """
        mri1_loaded = bool(getattr(self.state, "t1_path", None))

        self._set_status_item(
            "T1",
            mri1_loaded,
            "Valid" if mri1_loaded else "Missing",
            "validated" if mri1_loaded else "missing",
        )

        self._set_status_item(
            "T2",
            bool(getattr(self.state, "t2_path", None)),
            (
                "Coregistered"
                if bool(getattr(self.state, "t2_validated", False))
                else (
                    "Needs coregistration"
                    if bool(getattr(self.state, "t2_path", None))
                    else "Missing"
                )
            ),
            (
                "validated"
                if bool(getattr(self.state, "t2_validated", False))
                else ("loaded" if bool(getattr(self.state, "t2_path", None)) else "missing")
            ),
        )

        self._set_status_item(
            "CT",
            bool(getattr(self.state, "ct_path", None)),
            (
                "Coregistered"
                if bool(getattr(self.state, "ct_validated", False))
                else (
                    "Needs coregistration"
                    if bool(getattr(self.state, "ct_path", None))
                    else "Missing"
                )
            ),
            (
                "validated"
                if bool(getattr(self.state, "ct_validated", False))
                else ("loaded" if bool(getattr(self.state, "ct_path", None)) else "missing")
            ),
        )

        self._set_status_item(
            "PET",
            bool(getattr(self.state, "pet_path", None)),
            (
                "Coregistered"
                if bool(getattr(self.state, "pet_validated", False))
                else (
                    "Needs coregistration"
                    if bool(getattr(self.state, "pet_path", None))
                    else "Missing"
                )
            ),
            (
                "validated"
                if bool(getattr(self.state, "pet_validated", False))
                else ("loaded" if bool(getattr(self.state, "pet_path", None)) else "missing")
            ),
        )

        self._set_status_item(
            "ictalSPECT",
            bool(getattr(self.state, "ictal_spect_path", None)),
            (
                "Coregistered"
                if bool(getattr(self.state, "ictal_spect_validated", False))
                else (
                    "Needs coregistration"
                    if bool(getattr(self.state, "ictal_spect_path", None))
                    else "Missing"
                )
            ),
            (
                "validated"
                if bool(getattr(self.state, "ictal_spect_validated", False))
                else (
                    "loaded" if bool(getattr(self.state, "ictal_spect_path", None)) else "missing"
                )
            ),
        )

        self._set_status_item(
            "interictalSPECT",
            bool(getattr(self.state, "interictal_spect_path", None)),
            (
                "Coregistered"
                if bool(getattr(self.state, "interictal_spect_validated", False))
                else (
                    "Needs coregistration"
                    if bool(getattr(self.state, "interictal_spect_path", None))
                    else "Missing"
                )
            ),
            (
                "validated"
                if bool(getattr(self.state, "interictal_spect_validated", False))
                else (
                    "loaded"
                    if bool(getattr(self.state, "interictal_spect_path", None))
                    else "missing"
                )
            ),
        )
        self._set_status_item(
            "SISCOM",
            bool(
                getattr(self.state, "siscom_path", None)
                or getattr(self.state, "siscom_z_in_t1", None)
            ),
            (
                "Validated"
                if getattr(self.state, "siscom_validated", False)
                else (
                    "Loaded/computed"
                    if (
                        getattr(self.state, "siscom_path", None)
                        or getattr(self.state, "siscom_z_in_t1", None)
                    )
                    else "Missing"
                )
            ),
            (
                "validated"
                if getattr(self.state, "siscom_validated", False)
                else (
                    "loaded"
                    if (
                        getattr(self.state, "siscom_path", None)
                        or getattr(self.state, "siscom_z_in_t1", None)
                    )
                    else "missing"
                )
            ),
        )

        parcel1_loaded = bool(getattr(self.state, "parcel1_path", None))
        parcel2_loaded = bool(getattr(self.state, "parcel2_path", None))
        self._set_status_item(
            "Parcel1",
            parcel1_loaded,
            "Available" if parcel1_loaded else "Missing",
            "validated" if parcel1_loaded else "missing",
        )
        self._set_status_item(
            "Parcel2",
            parcel2_loaded,
            "Available" if parcel2_loaded else "Missing",
            "validated" if parcel2_loaded else "missing",
        )

        lh_loaded = bool(getattr(self.state, "lh_pial_path", None))
        rh_loaded = bool(getattr(self.state, "rh_pial_path", None))
        pials_available = bool(getattr(self.state, "pial_surfaces_available", False))
        lh_available = bool(lh_loaded and rh_loaded and pials_available)
        rh_available = bool(lh_loaded and rh_loaded and pials_available)
        self._set_status_item(
            "LHPial",
            lh_loaded,
            "Available" if lh_available else ("Loaded" if lh_loaded else "Missing"),
            "validated" if lh_available else ("loaded" if lh_loaded else "missing"),
        )
        self._set_status_item(
            "RHPial",
            rh_loaded,
            "Available" if rh_available else ("Loaded" if rh_loaded else "Missing"),
            "validated" if rh_available else ("loaded" if rh_loaded else "missing"),
        )

        brainmask_available = bool(
            getattr(self.state, "brainmask_sitk", None)
            or getattr(self.state, "brainmask_path", None)
        )
        self._set_status_item(
            "BrainMask",
            brainmask_available,
            "Available" if brainmask_available else "Missing",
            "validated" if brainmask_available else "missing",
        )

    def _sync_files_workflow_visual_states(self) -> None:
        """
        Update the visual feedback of Files / Coregistration without changing
        the functional workflow or the enabled/disabled rules.
        """
        load_states = (
            (
                getattr(self, "btn_load_t1", None),
                getattr(self.state, "t1_path", None),
                getattr(self, "le_t1", None),
            ),
            (
                getattr(self, "btn_load_t2", None),
                getattr(self.state, "t2_path", None),
                getattr(self, "le_t2", None),
            ),
            (
                getattr(self, "btn_load_ct", None),
                getattr(self.state, "ct_path", None),
                getattr(self, "le_ct", None),
            ),
            (
                getattr(self, "btn_load_pet", None),
                getattr(self.state, "pet_path", None),
                getattr(self, "le_pet", None),
            ),
            (
                getattr(self, "btn_load_ictal", None),
                getattr(self.state, "ictal_spect_path", None),
                getattr(self, "le_ictal", None),
            ),
            (
                getattr(self, "btn_load_interictal", None),
                getattr(self.state, "interictal_spect_path", None),
                getattr(self, "le_interictal", None),
            ),
            (
                getattr(self, "btn_load_siscom", None),
                getattr(self.state, "siscom_path", None),
                getattr(self, "le_siscom", None),
            ),
            (
                getattr(self, "btn_load_parcel1", None),
                getattr(self.state, "parcel1_path", None),
                getattr(self, "le_parcel1", None),
            ),
            (
                getattr(self, "btn_load_parcel2", None),
                getattr(self.state, "parcel2_path", None),
                getattr(self, "le_parcel2", None),
            ),
            (
                getattr(self, "btn_load_lhpial", None),
                getattr(self.state, "lh_pial_path", None),
                getattr(self, "le_lh_pial", None),
            ),
            (
                getattr(self, "btn_load_rhpial", None),
                getattr(self.state, "rh_pial_path", None),
                getattr(self, "le_rh_pial", None),
            ),
            (
                getattr(self, "btn_load_brainmask", None),
                getattr(self.state, "brainmask_path", None),
                getattr(self, "le_load_brainmask", None),
            ),
        )

        for button, path, line_edit in load_states:
            is_loaded = bool(path)
            self._set_dynamic_property(button, "loaded", is_loaded)
            self._set_dynamic_property(line_edit, "hasFile", is_loaded)

            if line_edit is not None and is_loaded:
                try:
                    line_edit.setToolTip(str(path))
                except Exception:
                    pass

        # Bulk import buttons show an orange outline when at least one file in
        # their family has been loaded.
        self._set_dynamic_property(
            getattr(self, "btn_load_imaging", None),
            "loaded",
            bool(
                getattr(self.state, "t1_path", None)
                or getattr(self.state, "t2_path", None)
                or getattr(self.state, "ct_path", None)
                or getattr(self.state, "pet_path", None)
                or getattr(self.state, "ictal_spect_path", None)
                or getattr(self.state, "interictal_spect_path", None)
                or getattr(self.state, "siscom_path", None)
            ),
        )
        self._set_dynamic_property(
            getattr(self, "btn_load_parcellations", None),
            "loaded",
            bool(
                getattr(self.state, "parcel1_path", None)
                or getattr(self.state, "parcel2_path", None)
            ),
        )
        self._set_dynamic_property(
            getattr(self, "btn_load_surfaces", None),
            "loaded",
            bool(
                getattr(self.state, "lh_pial_path", None)
                or getattr(self.state, "rh_pial_path", None)
                or getattr(self.state, "brainmask_path", None)
            ),
        )

        # Outputs receive a permanent pink outline only after they
        # have actually been written to disk.
        save_states = (
            (
                getattr(self, "btn_save_t2", None),
                bool(getattr(self.state, "t2_coreg_path", None)),
            ),
            (
                getattr(self, "btn_save_ct", None),
                bool(getattr(self.state, "ct_coreg_path", None)),
            ),
            (
                getattr(self, "btn_save_pet", None),
                bool(getattr(self.state, "pet_coreg_path", None)),
            ),
            (
                getattr(self, "btn_save_ictal", None),
                bool(getattr(self.state, "ictal_spect_coreg_path", None)),
            ),
            (
                getattr(self, "btn_save_interictal", None),
                bool(getattr(self.state, "interictal_spect_coreg_path", None)),
            ),
            (
                getattr(self, "btn_save_siscom", None),
                bool(getattr(self.state, "siscom_coreg_path", None)),
            ),
            (
                getattr(self, "btn_save_brainmask", None),
                bool(getattr(self.state, "brainmask_path", None)),
            ),
            (
                getattr(self, "btn_save_all", None),
                bool(
                    getattr(self.state, "t2_coreg_path", None)
                    or getattr(self.state, "ct_coreg_path", None)
                    or getattr(self.state, "pet_coreg_path", None)
                    or getattr(self.state, "ictal_spect_coreg_path", None)
                    or getattr(self.state, "interictal_spect_coreg_path", None)
                    or getattr(self.state, "siscom_coreg_path", None)
                    or getattr(self.state, "brainmask_path", None)
                ),
            ),
        )

        for button, is_saved in save_states:
            self._set_dynamic_property(button, "saved", is_saved)

        t1_loaded = bool(getattr(self.state, "t1_path", None))
        self._set_dynamic_property(getattr(self, "chk_t1", None), "referenceLoaded", t1_loaded)

        selected_modality = self._selected_modality()

        validated_by_modality = {
            "T2": bool(getattr(self.state, "t2_validated", False)),
            "CT": bool(getattr(self.state, "ct_validated", False)),
            "PET": bool(getattr(self.state, "pet_validated", False)),
            "ictalSPECT": bool(getattr(self.state, "ictal_spect_validated", False)),
            "interictalSPECT": bool(getattr(self.state, "interictal_spect_validated", False)),
        }

        coregistration_validated = bool(
            selected_modality and validated_by_modality.get(selected_modality, False)
        )

        self._set_dynamic_property(
            getattr(self, "btn_check_coreg", None),
            "completed",
            coregistration_validated,
        )

        siscom_validated = bool(getattr(self.state, "siscom_validated", False))

        self._set_dynamic_property(
            getattr(self, "btn_check_siscom", None),
            "completed",
            siscom_validated,
        )

    def _enable_checkbox(self, cb: QCheckBox | None, ok: bool):
        if cb is None:
            return
        cb.setEnabled(ok)
        if not ok:
            cb.setChecked(False)

    def _maybe_trigger_pial_coreg(self):
        lh = getattr(self.state, "lh_pial_path", None)
        rh = getattr(self.state, "rh_pial_path", None)

        # Show the popup as soon as both pial surfaces are loaded,
        # even if T1 is not loaded yet.
        if not (lh and rh):
            return

        should_coregister_pial = NeuXelecMessageDialog.question(
            self._dialog_parent(),
            "Pial surfaces",
            "Do you want to coregister LH/RH pial surfaces to the MRI 1?",
            accept_text="Yes",
            reject_text="No",
        )

        vp = self._view3d()

        # -------------------------
        # NO = keep current behavior
        # -------------------------
        if not should_coregister_pial:
            self.state.pial_surfaces_available = True
            self.state.pial_surfaces_assume_lps = True
            if vp is not None:
                try:
                    vp.set_pial_surfaces(
                        lh_path=self.state.lh_pial_path,
                        rh_path=self.state.rh_pial_path,
                        assume_lps=True,
                    )
                except Exception:
                    pass

            self._update_buttons()
            return

        # -------------------------
        # YES = need T1
        # -------------------------
        t1 = getattr(self.state, "t1_path", None)
        if not t1:
            NeuXelecMessageDialog.information(
                None,
                "MRI 1 required",
                "Please load the MRI 1 first. You will then be asked again to coregister the pial surfaces.",
            )
            self.load_t1()

            t1 = getattr(self.state, "t1_path", None)
            if not t1:
                # User cancelled T1 loading
                self._update_buttons()
                return

        dlg = PialCoregDialog(
            lh,
            rh,
            t1,
            parent=self._dialog_parent(),
        )
        result = dlg.exec()

        if result == QDialog.Accepted:
            self.state.pial_surfaces_available = True
            # Newly coregistered files produced by the dialog
            self.state.lh_pial_path = dlg.lh_out
            self.state.rh_pial_path = dlg.rh_out
            self.state.pial_surfaces_assume_lps = True

            if getattr(self, "le_lh_pial", None) is not None:
                self.le_lh_pial.setText(str(dlg.lh_out or ""))
                self.le_lh_pial.setCursorPosition(0)
                self.le_lh_pial.setToolTip(str(dlg.lh_out or ""))

            if getattr(self, "le_rh_pial", None) is not None:
                self.le_rh_pial.setText(str(dlg.rh_out or ""))
                self.le_rh_pial.setCursorPosition(0)
                self.le_rh_pial.setToolTip(str(dlg.rh_out or ""))

            if vp is not None:
                try:
                    vp.set_pial_surfaces(
                        lh_path=self.state.lh_pial_path,
                        rh_path=self.state.rh_pial_path,
                        assume_lps=True,
                    )
                except Exception:
                    pass

            self._update_buttons()
            return

        # If the coreg dialog is cancelled after choosing "Yes",
        # do nothing special and keep buttons refreshed.
        self._update_buttons()

    def _ask_transform_save_path(self, modality: Modality) -> str | None:
        start_dir = self.state.last_browse_dir or str(Path.home())
        moving_label = self._filename_label_for_modality(modality)
        fixed_label = self._mri1_filename_label()

        default_name = str(
            Path(start_dir)
            / self._default_filename(
                f"{moving_label}_to_{fixed_label}_0GenericAffine",
                ".mat",
            )
        )

        out_path, _ = QFileDialog.getSaveFileName(
            self._dialog_parent(),
            f"Save {modality} transform matrix",
            default_name,
            "ANTs affine matrix (*.mat);;All files (*.*)",
        )
        if not out_path:
            return None

        try:
            self.state.last_browse_dir = str(Path(out_path).parent)
        except Exception:
            pass

        return out_path

    def _preferred_save_start_path(self, modality: Modality) -> str:
        nifti_path = {
            "T2": getattr(self.state, "t2_path", None),
            "CT": getattr(self.state, "ct_path", None),
            "PET": getattr(self.state, "pet_path", None),
            "ictalSPECT": getattr(self.state, "ictal_spect_path", None),
            "interictalSPECT": getattr(self.state, "interictal_spect_path", None),
        }.get(modality)

        source_path = {
            "T2": getattr(self.state, "t2_source_path", None),
            "CT": getattr(self.state, "ct_source_path", None),
            "PET": getattr(self.state, "pet_source_path", None),
            "ictalSPECT": getattr(self.state, "ictal_spect_source_path", None),
            "interictalSPECT": getattr(self.state, "interictal_spect_source_path", None),
        }.get(modality)

        base_dir = None
        if nifti_path:
            try:
                base_dir = Path(nifti_path).parent
            except Exception:
                base_dir = None

        if base_dir is None and source_path:
            try:
                sp = Path(source_path)
                base_dir = sp.parent if sp.is_file() else sp
            except Exception:
                base_dir = None

        if base_dir is None:
            base_dir = Path(self.state.last_browse_dir or Path.home())

        return str(base_dir)
