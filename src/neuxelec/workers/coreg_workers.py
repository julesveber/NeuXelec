from __future__ import annotations

from typing import Literal

import SimpleITK as sitk
from PySide6.QtCore import QThread, Signal

from ..coregistration import ants_generate_brainmask_t1, rigid_coreg_to_fixed

Modality = Literal["T2", "CT", "PET", "ictalSPECT", "interictalSPECT"]


class CoregWorker(QThread):
    """Run ONE rigid registration in a background thread."""

    progress = Signal(int)
    finished_ok = Signal(str, object)  # modality, CoregResult
    failed = Signal(str, str)  # modality, message

    def __init__(
        self,
        modality: Modality,
        fixed_path: str,
        moving_path: str,
        initial_transform: sitk.Transform | None = None,
        transforms_dir: str | None = None,
    ):
        super().__init__()
        self.modality = modality
        self.fixed_path = fixed_path
        self.moving_path = moving_path
        self.initial_transform = initial_transform
        self.transforms_dir = transforms_dir

    def run(self):
        try:
            res = rigid_coreg_to_fixed(
                fixed_path=self.fixed_path,
                moving_path=self.moving_path,
                progress_cb=self.progress.emit,
                initial_transform=self.initial_transform,
                moving_modality=(
                    self.modality
                    if self.modality not in ("ictalSPECT", "interictalSPECT")
                    else "SPECT"
                ),
                transforms_dir=self.transforms_dir,
            )
            self.finished_ok.emit(self.modality, res)
        except Exception as e:
            self.failed.emit(self.modality, str(e))


class BrainMaskWorker(QThread):
    """Generate a brain mask for the loaded T1 in a background thread."""

    progress = Signal(int)
    finished_ok = Signal(str)  # path
    failed = Signal(str)

    def __init__(self, t1_path: str, transforms_dir: str | None = None):
        super().__init__()
        self.t1_path = t1_path
        self.transforms_dir = transforms_dir

    def run(self):
        try:
            out_path = ants_generate_brainmask_t1(
                t1_path=self.t1_path,
                out_dir=self.transforms_dir,
                progress_cb=self.progress.emit,
            )
            self.finished_ok.emit(out_path)
        except Exception as e:
            self.failed.emit(str(e))
