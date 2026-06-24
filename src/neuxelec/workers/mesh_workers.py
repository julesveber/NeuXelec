from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from PySide6.QtCore import QThread, Signal


class IsoSurfaceWorker(QThread):
    """Generate iso-surface mesh (marching cubes) in a background thread."""

    progress = Signal(int)
    finished_ok = Signal(object)  # dict (points, faces)
    failed = Signal(str)

    def __init__(self, t1_img: sitk.Image, iso_percentile: int = 60):
        super().__init__()
        self.t1_img = t1_img
        self.iso_percentile = int(iso_percentile)

    def run(self):
        try:
            # local import: keep dependencies optional unless used
            from skimage.measure import marching_cubes

            self.progress.emit(0)

            arr = sitk.GetArrayFromImage(self.t1_img).astype(np.float32, copy=False)  # z,y,x
            self.progress.emit(10)

            finite = np.isfinite(arr)
            nz = arr[finite & (arr > 0)]
            if nz.size < 1000:
                nz = arr[finite]
            if nz.size == 0:
                raise RuntimeError("T1 appears empty/invalid for iso-surface.")

            pct = int(np.clip(self.iso_percentile, 1, 99))
            iso = float(np.percentile(nz, pct))
            self.progress.emit(25)

            verts_zyx, faces, _, _ = marching_cubes(arr, level=iso)
            self.progress.emit(60)

            if verts_zyx.size == 0 or faces.size == 0:
                raise RuntimeError("marching_cubes produced an empty surface.")

            # Convert to physical mm using SimpleITK geometry
            verts_xyz = verts_zyx[:, ::-1].astype(np.float64)  # x,y,z voxel
            origin = np.array(self.t1_img.GetOrigin(), dtype=np.float64)
            spacing = np.array(self.t1_img.GetSpacing(), dtype=np.float64)
            direction = np.array(self.t1_img.GetDirection(), dtype=np.float64).reshape(3, 3)

            phys = origin[None, :] + ((verts_xyz * spacing[None, :]) @ direction.T)
            self.progress.emit(85)

            out = {
                "points": phys.astype(np.float32, copy=False),  # (N,3) in mm
                "faces": faces.astype(np.int32, copy=False),  # (M,3)
            }

            self.progress.emit(100)
            self.finished_ok.emit(out)

        except Exception as e:
            self.failed.emit(str(e))
