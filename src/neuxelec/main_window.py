from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PySide6.QtCore import QEvent, QObject, QSize, Qt, Signal
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractButton,
    QLabel,
    QPushButton,
    QStackedWidget,
    QWidget,
)

from .controllers.electrodes import ElectrodesController
from .controllers.menu import connect_menu_navigation_by_page_name
from .pages.files_page import FilesPage
from .pages.oblique_slice_page import ObliqueSlicePage
from .pages.reconstruction_page import ReconstructionPage
from .pages.view3d import View3DPage
from .project_io import (
    apply_project_dict_to_state,
    get_unsaved_validated_modalities,
    save_project_json,
)
from .state import AppState, Volume
from .ui.neuxelec_message_dialog import NeuXelecMessageDialog
from .utils.resources import resource_path
from .utils.ui_loader import load_ui


class NeuxelecWindow(QWidget):
    """Top-level window wrapper around the Qt Designer UI."""

    restart_requested = Signal()

    def __init__(self, ui_path: str | Path | None = None):
        super().__init__()

        if ui_path is None:
            ui_path = resource_path("resources/ui/MainWindow.ui")

        self.ui = load_ui(ui_path, self)
        self.ui.setParent(self)

        # App title (top-left)
        self.setWindowTitle("Neuxelec")

        self.state = AppState()
        try:
            self.state.main_window = self
        except Exception:
            pass
        self._is_closing = False
        self._returning_to_startup = False
        self._skip_close_prompt = False

        # Pages
        self.reco_page = ReconstructionPage(self.ui, self.state)
        self.state.reco_page = self.reco_page
        self.files_page = FilesPage(
            self.ui,
            self.state,
            on_ct_loaded=lambda: self._on_ct_volume_updated(switch_to_reco=False),
            on_ct_updated=lambda: self._on_ct_volume_updated(switch_to_reco=False),
        )

        # 3D View page (brainmask display)
        self.view3d_page = View3DPage(self.ui, self.state)
        self.state.view3d_page = self.view3d_page

        # Oblique Slice page
        self.oblique_page = ObliqueSlicePage(self.ui, self.state)
        self.state.oblique_page = self.oblique_page

        # Shared electrode list + color buttons across pages
        self.electrodes_controller = ElectrodesController(
            self.ui, self.state, reco_page=self.reco_page
        )

        try:
            self.state.electrodes_controller = self.electrodes_controller
        except Exception:
            pass

        try:
            if hasattr(self.view3d_page, "enable_mni_drag_and_drop"):
                self.view3d_page.enable_mni_drag_and_drop()
        except Exception:
            pass

        # Central menu navigation (button -> page objectName)
        self._menu_page_by_button = {
            "btn_menu_fileCoreg": "pageFiles",
            "btn_menu_reconstruction": "pageReconstruction",
            "btn_menu_obliqueSlice": "pageObliqueSlices",
            "btn_menu_3Dview": "page3DView",
            "btn_menu_save": "pageSaveExport",
        }

        connect_menu_navigation_by_page_name(
            self.ui,
            "stackedWidget",
            self._menu_page_by_button,
        )

        self._init_left_menu_selection_state()

        sw = self.ui.findChild(QObject, "stackedWidget")
        if sw is not None:
            sw.currentChanged.connect(self._on_page_changed)

        # Force startup on Files page
        self.set_current_page_by_name("pageFiles")
        self._on_page_changed(self.ui.findChild(QObject, "stackedWidget").currentIndex())

        # Activate the black NeuXelec frame only once all heavy page widgets
        # have been created. This avoids interfering with VTK/PyVista startup.
        self._setup_custom_window_frame()

    def _make_padded_logo_pixmap(
        self,
        logo: QPixmap,
        target_size: QSize,
        padding: int = 4,
    ) -> QPixmap:
        """
        Scale the NeuXelec logo into a transparent canvas with internal padding.

        This prevents the logo from being visually clipped because the PNG
        artwork touches the image boundaries.
        """
        if logo.isNull():
            return logo

        canvas_w = int(target_size.width())
        canvas_h = int(target_size.height())

        canvas = QPixmap(canvas_w, canvas_h)
        canvas.fill(Qt.GlobalColor.transparent)

        inner_w = max(1, canvas_w - 2 * int(padding))
        inner_h = max(1, canvas_h - 2 * int(padding))

        scaled = logo.scaled(
            QSize(inner_w, inner_h),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        x = (canvas_w - scaled.width()) // 2
        y = (canvas_h - scaled.height()) // 2

        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(x, y, scaled)
        painter.end()

        return canvas

    def _setup_custom_window_frame(self) -> None:
        """Activate the title bar already present in the current .ui file."""
        # Keep the normal Qt window behavior, only remove the native white frame.
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._title_bar = self.ui.findChild(QWidget, "customTitleBar")
        self._title_drag_label = self.ui.findChild(QLabel, "lbl_windowTitle")
        self._btn_window_minimize = self.ui.findChild(QPushButton, "btn_window_minimize")
        self._btn_window_maximize = self.ui.findChild(QPushButton, "btn_window_maximize")
        self._btn_window_close = self.ui.findChild(QPushButton, "btn_window_close")

        # Your PNG logo is loaded here without changing the click behavior.
        self._logo_label = self.ui.findChild(QLabel, "appTitle")
        if self._logo_label is not None:
            logo_path = resource_path("resources/images/neuxelec_logo.png")
            logo = QPixmap(str(logo_path))

            if not logo.isNull():
                self._logo_label.setText("")
                self._logo_label.setMinimumHeight(72)
                self._logo_label.setMaximumHeight(72)
                self._logo_label.setMinimumWidth(188)
                self._logo_label.setScaledContents(False)
                self._logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

                padded_logo = self._make_padded_logo_pixmap(
                    logo,
                    QSize(188, 66),
                    padding=5,
                )

                self._logo_label.setPixmap(padded_logo)

            self._logo_label.setCursor(Qt.CursorShape.PointingHandCursor)
            self._logo_label.setToolTip("Return to project start menu")
            self._logo_label.installEventFilter(self)

        if self._btn_window_minimize is not None:
            self._btn_window_minimize.clicked.connect(self.showMinimized)
        if self._btn_window_maximize is not None:
            self._btn_window_maximize.clicked.connect(self._toggle_maximized)
        if self._btn_window_close is not None:
            self._btn_window_close.clicked.connect(self.close)

        for drag_widget in (self._title_bar, self._title_drag_label):
            if drag_widget is not None:
                drag_widget.installEventFilter(self)

        self._sync_maximize_button()

    def _toggle_maximized(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
        self._sync_maximize_button()

    def _sync_maximize_button(self) -> None:
        maximized = bool(self.isMaximized())

        if getattr(self, "_btn_window_maximize", None) is not None:
            self._btn_window_maximize.setText("❐" if maximized else "□")
            self._btn_window_maximize.setToolTip(
                "Restore window" if maximized else "Maximize window"
            )

        try:
            self.ui.setProperty("maximized", maximized)
            self.ui.style().unpolish(self.ui)
            self.ui.style().polish(self.ui)
            self.ui.update()
        except Exception:
            pass

    def eventFilter(self, watched, event):
        if watched == getattr(self, "_logo_label", None):
            if (
                event.type() == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self._request_return_to_startup()
                return True
        if watched in (
            getattr(self, "_title_bar", None),
            getattr(self, "_title_drag_label", None),
        ):
            if (
                event.type() == QEvent.Type.MouseButtonDblClick
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self._toggle_maximized()
                return True

            if (
                event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                try:
                    handle = self.windowHandle()
                    if handle is not None:
                        handle.startSystemMove()
                        return True
                except Exception:
                    pass

        return super().eventFilter(watched, event)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._sync_maximize_button()

    @staticmethod
    def _report_project_loading(
        progress_callback,
        value: float,
        message: str,
    ) -> None:
        if progress_callback is None:
            return

        try:
            progress_callback(value, message)
        except Exception:
            pass

    def load_project_data(
        self,
        data: dict,
        project_path: str,
        mode: str = "edit",
        progress_callback=None,
    ) -> None:
        self._report_project_loading(
            progress_callback,
            0.56,
            "Restoring project settings",
        )

        apply_project_dict_to_state(self.state, data, project_path)
        self.state.app_mode = str(mode)

        self._apply_patient_id_to_ui()

        self._report_project_loading(
            progress_callback,
            0.62,
            "Loading imaging files",
        )

        self._load_project_files_into_memory(
            progress_callback=progress_callback,
        )

        self._report_project_loading(
            progress_callback,
            0.84,
            "Restoring electrodes",
        )

        self._restore_electrodes()
        self.apply_app_mode(self.state.app_mode)

        try:
            if getattr(self, "files_page", None) is not None and hasattr(
                self.files_page, "restore_from_state"
            ):
                self.files_page.restore_from_state()
        except Exception:
            pass

        self._report_project_loading(
            progress_callback,
            0.90,
            "Preparing reconstruction data",
        )

        try:
            if (
                getattr(self.state, "ct_coreg_path", None)
                and bool(getattr(self.state, "ct_validated", False))
                and not bool(getattr(self.state, "ct_ready_for_reconstruction", False))
                and hasattr(self.reco_page, "_show_coreg_warning")
            ):
                self.reco_page._show_coreg_warning()
            else:
                self.reco_page.init_from_volume()
        except Exception:
            pass
        # Important:
        # Do not render 3D View or Oblique Slice here.
        # Those pages now render themselves behind their own loading overlay
        # when the user actually opens them.

        try:
            self.electrodes_controller.refresh_all()
        except Exception:
            pass

        self._report_project_loading(
            progress_callback,
            0.96,
            "Finalizing workspace",
        )

    def _apply_patient_id_to_ui(self) -> None:
        try:
            le = getattr(self.ui, "le_menu_patientID", None)

            if le is None:
                le = self.ui.findChild(QObject, "le_menu_patientID")

            if le is None:
                return

            le.setText(str(self.state.patient_id or ""))
            le.setReadOnly(True)
            le.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            le.setCursor(Qt.CursorShape.ArrowCursor)
            le.setAlignment(Qt.AlignmentFlag.AlignCenter)
            le.setToolTip("Patient ID")

        except Exception:
            pass

    def _load_project_files_into_memory(
        self,
        progress_callback=None,
    ) -> None:
        self._report_project_loading(
            progress_callback,
            0.64,
            "Loading anatomical MRI",
        )

        # -------------------------
        # T1
        # -------------------------
        try:
            if self.state.t1_path:
                self.state.t1_sitk = sitk.ReadImage(self.state.t1_path)
                try:
                    self.view3d_page.set_t1(self.state.t1_sitk, t1_path=self.state.t1_path)
                except Exception:
                    pass
        except Exception:
            pass
        self._report_project_loading(
            progress_callback,
            0.68,
            "Loading CT and registered modalities",
        )
        # -------------------------
        # CT
        # -------------------------
        try:
            # Reconstruction must start from the raw/current CT and remain
            # blocked until CT is checked again during the current session.
            self.state.ct_ready_for_reconstruction = False

            if self.state.ct_path:
                raw_ct = sitk.ReadImage(self.state.ct_path)
                raw_arr = sitk.GetArrayFromImage(raw_ct).astype(np.float32)
                raw_arr = np.transpose(raw_arr, (2, 1, 0))  # z,y,x -> x,y,z
                self.state.volume = Volume(
                    data=raw_arr,
                    path=self.state.ct_path,
                )
            else:
                self.state.volume = None

            # A validated CT saved in the project remains available for
            # visualization pages, but not yet for Reconstruction.
            if bool(getattr(self.state, "ct_validated", False)) and self.state.ct_coreg_path:
                ct_validated_img = sitk.ReadImage(self.state.ct_coreg_path)
                self.state.ct_coreg_in_t1 = ct_validated_img
                self.state.ct_in_t1 = ct_validated_img

                try:
                    if hasattr(self.view3d_page, "set_ct"):
                        self.view3d_page.set_ct(
                            ct_validated_img,
                            ct_path=self.state.ct_coreg_path,
                        )
                except Exception:
                    pass

            else:
                self.state.ct_coreg_in_t1 = None
                self.state.ct_in_t1 = None

                try:
                    if hasattr(self.view3d_page, "set_ct"):
                        self.view3d_page.set_ct(None)
                except Exception:
                    pass

            try:
                if hasattr(self.oblique_page, "refresh_available_modalities"):
                    self.oblique_page.refresh_available_modalities(refresh=False)
            except Exception:
                pass

        except Exception as e:
            print("[Project Load] Failed to restore CT:", e)

        # -------------------------
        # T2
        # -------------------------
        try:
            if self.state.t2_coreg_path:
                img = sitk.ReadImage(self.state.t2_coreg_path)
                self.state.t2_coreg_in_t1 = img
                self.state.t2_in_t1 = img

                # Keep validation true if the project JSON says it was validated
                # or if a saved coregistered T2 exists.
                self.state.t2_validated = bool(
                    getattr(self.state, "t2_validated", False) or self.state.t2_coreg_path
                )

                # Push T2 to 3D View so checkBox_3dView_T2 becomes clickable.
                try:
                    if getattr(self, "view3d_page", None) is not None and hasattr(
                        self.view3d_page, "set_t2"
                    ):
                        self.view3d_page.set_t2(img, t2_path=self.state.t2_coreg_path)
                except Exception:
                    pass

                # Push T2 to Oblique Slice controls so checkBox_ObliqueSlice_T2 becomes clickable.
                try:
                    if getattr(self, "oblique_page", None) is not None:
                        if hasattr(
                            self.oblique_page,
                            "refresh_mri_source_controls",
                        ):
                            self.oblique_page.refresh_mri_source_controls()
                except Exception:
                    pass

            else:
                self.state.t2_coreg_in_t1 = None
                self.state.t2_in_t1 = None
                self.state.t2_validated = False

                try:
                    if getattr(self, "view3d_page", None) is not None and hasattr(
                        self.view3d_page, "set_t2"
                    ):
                        self.view3d_page.set_t2(None)
                except Exception:
                    pass

                try:
                    if getattr(self, "oblique_page", None) is not None and hasattr(
                        self.oblique_page, "refresh_mri_source_controls"
                    ):
                        self.oblique_page.refresh_mri_source_controls()
                except Exception:
                    pass

        except Exception:
            pass

        self._report_project_loading(
            progress_callback,
            0.72,
            "Loading functional imaging",
        )
        # -------------------------
        # PET
        # -------------------------
        try:
            pet_path = getattr(self.state, "pet_coreg_path", None) or getattr(
                self.state, "pet_path", None
            )

            if bool(getattr(self.state, "pet_validated", False)) and pet_path:
                img = sitk.ReadImage(pet_path)

                self.state.pet_coreg_in_t1 = img
                self.state.pet_in_t1 = img

                try:
                    self.view3d_page.set_pet(
                        img,
                        pet_path=pet_path,
                        activate=False,
                    )
                except TypeError:
                    self.view3d_page.set_pet(img, pet_path=pet_path)

                    try:
                        cb = getattr(self.view3d_page, "chk_pet", None)
                        if cb is not None:
                            cb.blockSignals(True)
                            cb.setEnabled(True)
                            cb.setChecked(True)
                            cb.blockSignals(False)

                        if hasattr(self.view3d_page, "_refresh_pet_only"):
                            self.view3d_page._refresh_pet_only()
                    except Exception:
                        pass
                except Exception:
                    pass

            else:
                self.state.pet_coreg_in_t1 = None
                self.state.pet_in_t1 = None

                try:
                    self.view3d_page.set_pet(None)
                except Exception:
                    pass

            try:
                if hasattr(self.oblique_page, "refresh_available_modalities"):
                    self.oblique_page.refresh_available_modalities(
                        refresh=False,
                        activate_validated=False,
                    )
            except TypeError:
                self.oblique_page.refresh_available_modalities(refresh=False)
            except Exception:
                pass

        except Exception as e:
            print("[Project Load] Failed to restore PET:", e)

        # -------------------------
        # ictal SPECT
        # -------------------------
        try:
            ictal_path = self.state.ictal_spect_coreg_path or self.state.ictal_spect_path
            if ictal_path:
                img = sitk.ReadImage(ictal_path)
                if self.state.ictal_spect_coreg_path:
                    self.state.ictal_spect_coreg_in_t1 = img
                    self.state.ictal_spect_in_t1 = img
        except Exception:
            pass

        # -------------------------
        # interictal SPECT
        # -------------------------
        try:
            interictal_path = (
                self.state.interictal_spect_coreg_path or self.state.interictal_spect_path
            )
            if interictal_path:
                img = sitk.ReadImage(interictal_path)
                if self.state.interictal_spect_coreg_path:
                    self.state.interictal_spect_coreg_in_t1 = img
                    self.state.interictal_spect_in_t1 = img
        except Exception:
            pass

        # -------------------------
        # SISCOM
        # -------------------------
        try:
            siscom_path = self.state.siscom_coreg_path or self.state.siscom_path

            if siscom_path:
                img = sitk.ReadImage(siscom_path)

                # Keep the image in memory so Files/Coregistration can still
                # open Check SISCOM after a project reload.
                self.state.siscom_z_in_t1 = img
                self.state.siscom_thr_in_t1 = None

                if bool(getattr(self.state, "siscom_validated", False)):
                    self.state.siscom_coreg_in_t1 = img

                    try:
                        self.view3d_page.set_siscom(
                            img,
                            siscom_path=siscom_path,
                        )
                    except Exception:
                        pass
                else:
                    self.state.siscom_coreg_in_t1 = None

                    try:
                        self.view3d_page.set_siscom(None)
                    except Exception:
                        pass

            else:
                self.state.siscom_coreg_in_t1 = None
                self.state.siscom_z_in_t1 = None
                self.state.siscom_thr_in_t1 = None

                try:
                    self.view3d_page.set_siscom(None)
                except Exception:
                    pass

            try:
                if hasattr(self.oblique_page, "refresh_available_modalities"):
                    self.oblique_page.refresh_available_modalities(refresh=False)
            except Exception:
                pass

        except Exception as e:
            print("[Project Load] Failed to restore SISCOM:", e)
        # -------------------------
        # Brainmask
        # -------------------------
        try:
            bm_path = getattr(self.state, "brainmask_path", None)

            # Only restore a brainmask from disk if it was explicitly saved/loaded.
            # If it was only generated but not saved, it will still be reported as unsaved,
            # but it cannot be restored after reopening unless the file was saved.
            if bm_path:
                self.state.brainmask_sitk = sitk.ReadImage(bm_path)

                try:
                    self.view3d_page.set_brainmask(
                        self.state.brainmask_sitk,
                        brainmask_path=bm_path,
                    )
                except Exception:
                    pass

                try:
                    if getattr(self, "oblique_page", None) is not None:
                        if hasattr(self.oblique_page, "_last_brain_key"):
                            self.oblique_page._last_brain_key = None

                        if hasattr(self.oblique_page, "_last_brain_kind"):
                            self.oblique_page._last_brain_kind = None
                except Exception:
                    pass

        except Exception:
            pass
        # -------------------------
        # Pial surfaces
        # -------------------------
        try:
            lh = getattr(self.state, "lh_pial_path", None)
            rh = getattr(self.state, "rh_pial_path", None)
            if lh and rh:
                try:
                    self.view3d_page.set_pial_surfaces(
                        lh_path=lh,
                        rh_path=rh,
                        assume_lps=True,
                    )
                except Exception:
                    pass
        except Exception:
            pass
        self._report_project_loading(
            progress_callback,
            0.80,
            "Loading parcellations",
        )
        # -------------------------
        # Parcellation 1
        # -------------------------
        try:
            if getattr(self.state, "parcel1_path", None):
                img = sitk.ReadImage(self.state.parcel1_path)
                self.state.parcel1_img = img

                try:
                    self.oblique_page.set_parcellation1(img, self.state.parcel1_path)
                except Exception:
                    pass

                try:
                    self.view3d_page.set_parcellation1(img, self.state.parcel1_path)
                except Exception:
                    pass
        except Exception:
            pass

        # -------------------------
        # Parcellation 2
        # -------------------------
        try:
            if getattr(self.state, "parcel2_path", None):
                img = sitk.ReadImage(self.state.parcel2_path)
                self.state.parcel2_img = img

                # Rebuild LUT exactly like load_parcellation2()
                self.state.parcellation2_lut = {}

                filename = Path(self.state.parcel2_path).name.lower()
                lut_path = Path(__file__).resolve().parent / "utils" / "FreeSurferColorLUT.txt"

                if "aparc+aseg" in filename and lut_path.exists():
                    self.state.parcellation2_lut = self.files_page._load_freesurfer_lut_dict(
                        lut_path
                    )

                if not getattr(self.state, "parcellation2_lut", {}):
                    try:
                        arr = sitk.GetArrayViewFromImage(img)
                        uniq = np.unique(arr)

                        if np.any((uniq >= 1000) & (uniq < 5000)) and lut_path.exists():
                            self.state.parcellation2_lut = (
                                self.files_page._load_freesurfer_lut_dict(lut_path)
                            )
                    except Exception:
                        pass

                # Push Parcellation 2 to 3D View
                try:
                    if getattr(self, "view3d_page", None) is not None and hasattr(
                        self.view3d_page, "set_parcellation2"
                    ):
                        self.view3d_page.set_parcellation2(img, self.state.parcel2_path)
                except Exception:
                    pass

                # Push Parcellation 2 to Oblique Slice
                try:
                    if getattr(self, "oblique_page", None) is not None and hasattr(
                        self.oblique_page, "set_parcellation2"
                    ):
                        self.oblique_page.set_parcellation2(img, self.state.parcel2_path)
                except Exception:
                    pass

        except Exception as e:
            print("[Project Load] Failed to restore Parcellation 2:", e)

        # -------------------------
        # Oblique Slice / 3D / Reconstruction full refresh
        # -------------------------
        try:
            if getattr(self, "oblique_page", None) is not None:
                self.oblique_page._slice_cache_1 = None
                self.oblique_page._slice_cache_2 = None
                self.oblique_page._render_cache_1 = None
                self.oblique_page._render_cache_2 = None
                self.oblique_page._base_cache_1 = None
                self.oblique_page._base_cache_2 = None
                self.oblique_page._pet_cache_1 = None
                self.oblique_page._pet_cache_2 = None
        except Exception:
            pass

        try:
            self.state.sync_aliases_from_new_names()
        except Exception:
            pass

    def _restore_electrodes(self) -> None:
        self.reco_page._electrodes = self.state.electrodes

    def apply_app_mode(self, mode: str) -> None:
        mode = str(mode or "edit").lower().strip()
        self.state.app_mode = mode

        files_enabled = mode == "edit"
        reco_enabled = mode == "edit"

        # menu buttons
        for name in ("btn_menu_fileCoreg", "btn_menu_reconstruction"):
            try:
                btn = getattr(self.ui, name, None)
                if btn is not None:
                    btn.setEnabled(files_enabled if name == "btn_menu_fileCoreg" else reco_enabled)
            except Exception:
                pass

        # if visualization mode and current page is forbidden, move to 3D view
        if mode == "visualization":
            sw = self.ui.findChild(QObject, "stackedWidget")
            if sw is not None:
                try:
                    current = sw.currentWidget()
                    if current is not None and current.objectName() in (
                        "pageFiles",
                        "pageReconstruction",
                    ):
                        self.set_current_page_by_name("page3DView")
                except Exception:
                    pass

    def set_current_page(self, idx: int) -> None:
        sw = self.ui.findChild(QStackedWidget, "stackedWidget")

        if sw is None:
            return

        try:
            sw.setCurrentIndex(idx)

            w = sw.widget(idx)
            if w is not None:
                self._sync_left_menu_selection(w.objectName())

        except Exception:
            pass

    def _init_left_menu_selection_state(self) -> None:
        """
        Make left-menu buttons stateful so they can visually follow
        programmatic page changes, including Ctrl+D / Ctrl+F shortcuts.
        """
        for btn_name in getattr(self, "_menu_page_by_button", {}):
            btn = self.ui.findChild(QAbstractButton, btn_name)

            if btn is None:
                continue

            try:
                btn.setCheckable(True)
                btn.setAutoExclusive(False)
            except Exception:
                pass

    def _sync_left_menu_selection(self, page_object_name: str) -> None:
        """
        Synchronize the left menu with the current QStackedWidget page.

        This is called both after menu clicks and after programmatic page changes
        such as Ctrl+D and Ctrl+F.
        """
        mapping = getattr(self, "_menu_page_by_button", {})

        for btn_name, target_page in mapping.items():
            btn = self.ui.findChild(QAbstractButton, btn_name)

            if btn is None:
                continue

            active = str(target_page) == str(page_object_name)

            try:
                btn.blockSignals(True)
                btn.setChecked(active)

                # Useful if your .ui stylesheet uses dynamic properties.
                btn.setProperty("activePage", active)
                btn.setProperty("selected", active)

                btn.style().unpolish(btn)
                btn.style().polish(btn)
                btn.update()

            except Exception:
                pass

            finally:
                try:
                    btn.blockSignals(False)
                except Exception:
                    pass

    def set_current_page_by_name(self, page_object_name: str) -> None:
        sw = self.ui.findChild(QStackedWidget, "stackedWidget")

        if sw is None:
            return

        try:
            for i in range(sw.count()):
                w = sw.widget(i)

                if w is not None and w.objectName() == page_object_name:
                    sw.setCurrentIndex(i)

                    # Important when the page is already current:
                    # currentChanged may not fire, so force menu sync.
                    self._sync_left_menu_selection(page_object_name)
                    return

        except Exception:
            pass

    def _on_ct_volume_updated(self, switch_to_reco: bool = False) -> None:
        """
        Refresh Reconstruction after CT changes.

        If a CT coregistration exists but has not been revalidated for the
        current Reconstruction session, show the validation warning instead
        of displaying the CT.
        """
        try:
            if (
                getattr(self.state, "ct_coreg_in_t1", None) is not None
                and bool(getattr(self.state, "ct_validated", False))
                and not bool(getattr(self.state, "ct_ready_for_reconstruction", False))
                and hasattr(self.reco_page, "_show_coreg_warning")
            ):
                self.reco_page._show_coreg_warning()
            else:
                self.reco_page.init_from_volume()
        except Exception:
            pass

        if switch_to_reco:
            self.set_current_page_by_name("pageReconstruction")
        else:
            try:
                if bool(getattr(self.state, "ct_ready_for_reconstruction", False)):
                    self.reco_page.render_all()
            except Exception:
                pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.ui is not None:
            self.ui.setGeometry(self.rect())
        try:
            self.reco_page.render_all()
        except Exception:
            pass

    def _on_page_changed(self, index: int):
        """
        Change de page sans afficher une vue partiellement rendue.

        Le menu conserve son fonctionnement actuel : il change simplement
        l'index du QStackedWidget. Le signal currentChanged arrive ici avant
        que Qt ne peigne la nouvelle page. On bloque donc temporairement
        l'affichage pendant l'activation et le rendu de la page cible.
        """
        if getattr(self, "_is_closing", False):
            return

        sw = self.ui.findChild(QObject, "stackedWidget")
        if sw is None:
            return

        current_widget = sw.widget(index)
        if current_widget is None:
            return

        try:
            # Désactiver toutes les pages
            for page in [
                getattr(self, "oblique_page", None),
                getattr(self, "view3d_page", None),
                getattr(self, "reco_page", None),
                getattr(self, "files_page", None),
            ]:
                if hasattr(page, "set_active_page"):
                    try:
                        page.set_active_page(False)
                    except Exception:
                        pass

            # Activer uniquement la page courante.
            # Les méthodes de rendu existantes restent dans chaque page.
            page_name = current_widget.objectName()
            self._sync_left_menu_selection(page_name)

            if page_name == "pageObliqueSlices" and hasattr(self.oblique_page, "set_active_page"):
                try:
                    self.oblique_page.set_active_page(True)
                except Exception:
                    pass

            elif page_name == "page3DView" and hasattr(self.view3d_page, "set_active_page"):
                try:
                    self.view3d_page.set_active_page(True)
                except Exception:
                    pass

            elif page_name == "pageReconstruction" and hasattr(self.reco_page, "set_active_page"):
                try:
                    self.reco_page.set_active_page(True)
                except Exception:
                    pass

            elif page_name == "pageFiles" and hasattr(self.files_page, "set_active_page"):
                try:
                    self.files_page.set_active_page(True)
                except Exception:
                    pass

            # Prépare la géométrie avant de réafficher la nouvelle page.
            try:
                sw.updateGeometry()
                self.ui.updateGeometry()
            except Exception:
                pass

        finally:
            pass

    def _confirm_and_save_before_leaving_project(self) -> bool:
        """
        Return True if the user accepts leaving the current project.

        This reuses the same safety logic as closeEvent:
        validated coregistered files that are not saved on disk are reported
        before leaving the project.
        """
        try:
            unsaved = get_unsaved_validated_modalities(self.state)

            if unsaved:
                txt = (
                    "You have validated coregistered files that have not "
                    "been saved on disk:\n\n"
                    "• "
                    + "\n• ".join(unsaved)
                    + "\n\nIf you leave this project now, these files cannot "
                    "be restored from the project file."
                )

                choice = NeuXelecMessageDialog.choice(
                    self,
                    "Unsaved coregistered files",
                    txt,
                    choices=[
                        (
                            "leave_anyway",
                            "Leave anyway",
                            False,
                        ),
                        (
                            "save_and_leave",
                            "Save project and leave",
                            True,
                        ),
                    ],
                    cancel_text="Cancel",
                )

                if choice is None:
                    return False

                if choice == "save_and_leave" and self.state.project_path:
                    try:
                        save_project_json(
                            self.state,
                            self.state.project_path,
                        )
                    except Exception as e:
                        NeuXelecMessageDialog.critical(
                            self,
                            "Project save failed",
                            (
                                "The project could not be saved before "
                                "leaving.\n\n"
                                f"Details:\n{e}"
                            ),
                        )
                        return False

            else:
                if self.state.project_path:
                    try:
                        save_project_json(
                            self.state,
                            self.state.project_path,
                        )
                    except Exception:
                        pass

            return True

        except Exception:
            return True

    def _request_return_to_startup(self) -> None:
        """
        Close the current project and ask app.py to show StartupDialog again.
        """
        if getattr(self, "_is_closing", False):
            return

        if not self._confirm_and_save_before_leaving_project():
            return

        self._returning_to_startup = True
        self._skip_close_prompt = True

        try:
            self.restart_requested.emit()
        except Exception:
            pass

        self.close()

    def closeEvent(self, event):
        # Prevent page switches / renders while closing
        self._is_closing = True

        # 1) Save project / ask user
        if not bool(getattr(self, "_skip_close_prompt", False)):
            if not self._confirm_and_save_before_leaving_project():
                self._is_closing = False
                event.ignore()
                return

        # 2) Properly close the VTK QtInteractors FIRST, while their widgets
        #    are still shown and their OpenGL contexts are still current.
        #
        #    QtInteractor.close() (pyvistaqt) does the full, safe teardown:
        #      - stops the internal render timer,
        #      - finalizes the OpenGL render window (releases the GL context
        #        while the window handle is still valid), and
        #      - sets the interactor's `_closed` flag so any later paintGL is a
        #        no-op.
        #
        #    Both parts matter on return-to-menu:
        #      * Finalizing here avoids "wglMakeCurrent failed ... invalid
        #        handle" (Qt destroying the HWND before VTK releases the GL
        #        context).
        #      * The `_closed` flag avoids the follow-up crash where a pending
        #        repaint re-initializes the render window on an already
        #        torn-down widget ("failed to get valid pixel format").
        #
        #    This must run BEFORE the plotters are hidden in step 5.
        for page, attr in (
            (getattr(self, "view3d_page", None), "interactor"),
            (getattr(self, "oblique_page", None), "_brain_plotter"),
        ):
            try:
                interactor = getattr(page, attr, None) if page is not None else None
                if interactor is None:
                    continue
                if hasattr(interactor, "close"):
                    interactor.close()
                else:
                    render_window = getattr(interactor, "render_window", None)
                    if render_window is None and hasattr(interactor, "GetRenderWindow"):
                        render_window = interactor.GetRenderWindow()
                    if render_window is not None:
                        render_window.Finalize()
            except Exception:
                pass

        # 3) Disconnect stackedWidget page-change signal to avoid reactivation during destruction
        try:
            sw = self.ui.findChild(QObject, "stackedWidget")
            if sw is not None:
                try:
                    sw.currentChanged.disconnect(self._on_page_changed)
                except Exception:
                    pass
        except Exception:
            pass

        # 4) Explicitly mark all pages inactive before closing
        for page in (
            getattr(self, "oblique_page", None),
            getattr(self, "view3d_page", None),
            getattr(self, "reco_page", None),
            getattr(self, "files_page", None),
        ):
            try:
                if page is not None:
                    if hasattr(page, "_is_active_page"):
                        page._is_active_page = False
                    if hasattr(page, "set_active_page"):
                        page.set_active_page(False)
            except Exception:
                pass

        # 5) Stop oblique timers/renders (this also hides the brain plotter).
        try:
            if getattr(self, "oblique_page", None) is not None and hasattr(
                self.oblique_page, "cleanup"
            ):
                self.oblique_page.cleanup()
        except Exception:
            pass

        event.accept()
