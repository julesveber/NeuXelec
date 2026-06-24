from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def _resample_to_ref(moving: sitk.Image, ref: sitk.Image, interp=sitk.sitkLinear) -> sitk.Image:
    """Resample moving image onto ref grid."""
    res = sitk.ResampleImageFilter()
    res.SetReferenceImage(ref)
    res.SetInterpolator(interp)
    res.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    res.SetDefaultPixelValue(0.0)
    return res.Execute(moving)


def _gaussian_sigma_from_fwhm_mm(fwhm_mm: float) -> float:
    """
    Convert FWHM in mm to sigma in mm.
    sigma = FWHM / (2*sqrt(2*ln(2)))
    """
    if fwhm_mm <= 0:
        return 0.0
    return float(fwhm_mm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def compute_siscom(
    t1_img: sitk.Image,
    ictal_in_t1: sitk.Image,
    interictal_in_t1: sitk.Image,
    brainmask_in_t1: sitk.Image | None = None,
    z_threshold: float = 2.0,
    smooth_fwhm_mm: float = 6.0,
    progress_cb: Callable[[int], None] | None = None,
) -> dict[str, sitk.Image]:
    """
    Compute SISCOM maps in T1 space.

    Clinical-like order:
      1) resample ictal/interictal to T1
      2) require / apply brain mask
      3) global mean scaling within mask
      4) optional smoothing
      5) subtraction
      6) z-score within mask
      7) positive thresholding

    Returns dict:
      - "diff": ictal_norm - interictal_norm (float32)
      - "z": z-score map of diff within mask (float32)
      - "thr": thresholded z map (z > z_threshold) as float32 (others 0)
    """

    def prog(v: int):
        if progress_cb is not None:
            try:
                progress_cb(int(v))
            except Exception:
                logger.debug("SISCOM progress callback failed at %s%%", v, exc_info=True)

    prog(0)

    # 1) Put both SPECT on the T1 grid
    ict = _resample_to_ref(ictal_in_t1, t1_img, interp=sitk.sitkLinear)
    prog(10)
    inter = _resample_to_ref(interictal_in_t1, t1_img, interp=sitk.sitkLinear)
    prog(20)

    # 2) Brain mask is required for a more clinical workflow
    if brainmask_in_t1 is None:
        raise ValueError("Brain mask is required to compute SISCOM.")

    m = _resample_to_ref(brainmask_in_t1, t1_img, interp=sitk.sitkNearestNeighbor)
    m_arr = sitk.GetArrayFromImage(m) > 0

    if not np.any(m_arr):
        raise ValueError("Brain mask is empty after resampling to T1 space.")

    prog(30)

    ict_arr = sitk.GetArrayFromImage(ict).astype(np.float32, copy=False)
    inter_arr = sitk.GetArrayFromImage(inter).astype(np.float32, copy=False)

    valid = m_arr & np.isfinite(ict_arr) & np.isfinite(inter_arr)
    if not np.any(valid):
        raise ValueError("No valid ictal/interictal voxels inside the brain mask.")

    # 3) Global mean normalization within mask
    mean_ict = float(np.mean(ict_arr[valid]))
    mean_inter = float(np.mean(inter_arr[valid]))

    if abs(mean_ict) < 1e-8:
        mean_ict = 1.0
    if abs(mean_inter) < 1e-8:
        mean_inter = 1.0

    ict_n = np.zeros_like(ict_arr, dtype=np.float32)
    inter_n = np.zeros_like(inter_arr, dtype=np.float32)
    ict_n[valid] = ict_arr[valid] / mean_ict
    inter_n[valid] = inter_arr[valid] / mean_inter

    prog(45)

    # 4) Optional smoothing AFTER scaling, restricted back to mask afterwards
    if smooth_fwhm_mm and smooth_fwhm_mm > 0:
        sigma_mm = _gaussian_sigma_from_fwhm_mm(float(smooth_fwhm_mm))

        ict_n_img = sitk.GetImageFromArray(ict_n)
        inter_n_img = sitk.GetImageFromArray(inter_n)
        ict_n_img.CopyInformation(t1_img)
        inter_n_img.CopyInformation(t1_img)

        ict_n_img = sitk.SmoothingRecursiveGaussian(ict_n_img, sigma_mm)
        inter_n_img = sitk.SmoothingRecursiveGaussian(inter_n_img, sigma_mm)

        ict_n = sitk.GetArrayFromImage(ict_n_img).astype(np.float32, copy=False)
        inter_n = sitk.GetArrayFromImage(inter_n_img).astype(np.float32, copy=False)

        # keep only brain voxels
        ict_n[~valid] = 0.0
        inter_n[~valid] = 0.0

    prog(60)

    # 5) Subtraction
    diff = np.zeros_like(ict_n, dtype=np.float32)
    diff[valid] = ict_n[valid] - inter_n[valid]

    # 6) Z-score inside mask only
    mu = float(np.mean(diff[valid]))
    sigma = float(np.std(diff[valid]))
    if sigma < 1e-8:
        sigma = 1.0

    z = np.zeros_like(diff, dtype=np.float32)
    z[valid] = (diff[valid] - mu) / sigma

    prog(80)

    # 7) Positive thresholding inside mask only
    thr = np.zeros_like(z, dtype=np.float32)
    keep = valid & (z > float(z_threshold))
    thr[keep] = z[keep]

    diff_img = sitk.GetImageFromArray(diff.astype(np.float32, copy=False))
    z_img = sitk.GetImageFromArray(z.astype(np.float32, copy=False))
    thr_img = sitk.GetImageFromArray(thr.astype(np.float32, copy=False))
    diff_img.CopyInformation(t1_img)
    z_img.CopyInformation(t1_img)
    thr_img.CopyInformation(t1_img)

    prog(100)
    return {"diff": diff_img, "z": z_img, "thr": thr_img}
