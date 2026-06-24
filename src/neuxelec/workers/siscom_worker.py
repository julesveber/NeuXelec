from __future__ import annotations

import SimpleITK as sitk
from PySide6.QtCore import QThread, Signal

from ..siscom import compute_siscom


class SISCOMWorker(QThread):
    """Compute SISCOM (ictal - interictal) in T1 space in a background thread."""

    progress = Signal(int)
    finished_ok = Signal(dict)  # IMPORTANT: dict for PySide6 stability
    failed = Signal(str)

    def __init__(
        self,
        t1_img: sitk.Image,
        ictal_img: sitk.Image,
        interictal_img: sitk.Image,
        brainmask_img: sitk.Image | None = None,
        z_threshold: float = 2.0,
        smooth_fwhm_mm: float = 0.0,
    ):
        super().__init__()
        self.t1_img = t1_img
        self.ictal_img = ictal_img
        self.interictal_img = interictal_img
        self.brainmask_img = brainmask_img
        self.z_threshold = float(z_threshold)
        self.smooth_fwhm_mm = float(smooth_fwhm_mm)

    def run(self):
        try:

            def cb(v: int):
                try:
                    self.progress.emit(int(v))
                except Exception:
                    pass

            out = compute_siscom(
                t1_img=self.t1_img,
                ictal_in_t1=self.ictal_img,
                interictal_in_t1=self.interictal_img,
                brainmask_in_t1=self.brainmask_img,
                z_threshold=self.z_threshold,
                smooth_fwhm_mm=self.smooth_fwhm_mm,
                progress_cb=cb,
            )
            self.finished_ok.emit(out)
        except Exception as e:
            self.failed.emit(str(e))
