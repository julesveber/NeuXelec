from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PySide6.QtWidgets import QFileDialog, QWidget

from ..ui.neuxelec_message_dialog import NeuXelecMessageDialog

DEFAULT_TARGET_SPACING = (1.0, 1.0, 1.0)
DEFAULT_SPACING_TOLERANCE = 1e-3


def get_image_spacing(img: sitk.Image) -> tuple[float, float, float]:
    """
    Return image spacing as (sx, sy, sz).
    """
    sp = img.GetSpacing()
    return float(sp[0]), float(sp[1]), float(sp[2])


def is_spacing_1mm_isotropic(
    img: sitk.Image,
    target_spacing: tuple[float, float, float] = DEFAULT_TARGET_SPACING,
    tolerance: float = DEFAULT_SPACING_TOLERANCE,
) -> bool:
    """
    Check whether an image is already 1x1x1 mm isotropic.
    """
    spacing = get_image_spacing(img)

    return all(
        abs(float(spacing[i]) - float(target_spacing[i])) <= float(tolerance) for i in range(3)
    )


def spacing_to_text(spacing: tuple[float, float, float]) -> str:
    """
    Format spacing for UI messages.
    """
    return f"{spacing[0]:.4g} × {spacing[1]:.4g} × {spacing[2]:.4g} mm"


def make_conformed_t1_output_path(input_path: str | Path) -> str:
    """
    Suggest an output path for the 1 mm isotropic T1.
    """
    p = Path(input_path)

    name = p.name

    if name.endswith(".nii.gz"):
        stem = name[:-7]
        return str(p.with_name(f"{stem}_1mm_iso.nii.gz"))

    if p.suffix.lower() == ".nii":
        return str(p.with_name(f"{p.stem}_1mm_iso.nii.gz"))

    if p.suffix.lower() in (".mgz", ".mgh"):
        return str(p.with_name(f"{p.stem}_1mm_iso.nii.gz"))

    return str(p.with_name(f"{p.stem}_1mm_iso.nii.gz"))


def _compute_resampled_size(
    img: sitk.Image,
    target_spacing: tuple[float, float, float],
) -> tuple[int, int, int]:
    """
    Compute output size preserving the physical field of view.

    SimpleITK size is in x, y, z.
    """
    old_size = np.asarray(img.GetSize(), dtype=np.float64)
    old_spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
    new_spacing = np.asarray(target_spacing, dtype=np.float64)

    new_size = np.round(old_size * old_spacing / new_spacing).astype(int)
    new_size = np.maximum(new_size, 1)

    return int(new_size[0]), int(new_size[1]), int(new_size[2])


def resample_image_to_spacing(
    img: sitk.Image,
    target_spacing: tuple[float, float, float] = DEFAULT_TARGET_SPACING,
    interpolator=sitk.sitkLinear,
    default_value: float = 0.0,
    output_pixel_type=None,
) -> sitk.Image:
    """
    Resample a SimpleITK image to target spacing while preserving:
      - origin
      - direction
      - physical field of view as much as possible

    For T1 images, use linear interpolation.
    For masks/parcellations, do NOT use this function with linear interpolation;
    use nearest-neighbor instead.
    """
    if output_pixel_type is None:
        output_pixel_type = img.GetPixelID()

    target_spacing = tuple(float(v) for v in target_spacing)
    new_size = _compute_resampled_size(img, target_spacing)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetOutputPixelType(output_pixel_type)

    return resampler.Execute(img)


def conform_t1_to_1mm(
    input_path: str | Path,
    output_path: str | Path,
    target_spacing: tuple[float, float, float] = DEFAULT_TARGET_SPACING,
) -> sitk.Image:
    """
    Read a T1 image, resample it to 1 mm isotropic, save it, and return the image.
    """
    input_path = str(input_path)
    output_path = str(output_path)

    img = sitk.ReadImage(input_path)

    out = resample_image_to_spacing(
        img,
        target_spacing=target_spacing,
        interpolator=sitk.sitkLinear,
        default_value=0.0,
        output_pixel_type=img.GetPixelID(),
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, output_path)

    return out


def ask_user_to_conform_t1_if_needed(
    input_path: str | Path,
    parent: QWidget | None = None,
    target_spacing: tuple[float, float, float] = DEFAULT_TARGET_SPACING,
    tolerance: float = DEFAULT_SPACING_TOLERANCE,
    force_ask_output_path: bool = False,
) -> tuple[str, sitk.Image, dict]:
    """
    Read a T1 image and verify whether it is 1 mm isotropic.

    Returns:
        final_path, final_img, info

    info contains:
        {
            "was_conformed": bool,
            "original_path": str,
            "final_path": str,
            "original_spacing": [sx, sy, sz],
            "final_spacing": [sx, sy, sz],
        }

    Behavior:
        - If T1 is already 1x1x1 mm, returns original path/image.
        - If not, asks the user whether to resample.
        - If user accepts, asks where to save the conformed file unless
          force_ask_output_path=False, in which case a default path is proposed first.
        - If user refuses, returns the original image/path.
    """
    input_path = str(input_path)
    img = sitk.ReadImage(input_path)

    original_spacing = get_image_spacing(img)

    info = {
        "was_conformed": False,
        "original_path": input_path,
        "final_path": input_path,
        "original_spacing": [float(v) for v in original_spacing],
        "final_spacing": [float(v) for v in original_spacing],
    }

    if is_spacing_1mm_isotropic(
        img,
        target_spacing=target_spacing,
        tolerance=tolerance,
    ):
        return input_path, img, info

    target_txt = spacing_to_text(target_spacing)
    current_txt = spacing_to_text(original_spacing)

    create_conformed_copy = NeuXelecMessageDialog.question(
        parent,
        "T1 resolution check",
        (
            "The loaded T1 image is not 1 mm isotropic.\n\n"
            f"Current spacing: {current_txt}\n"
            f"Recommended spacing: {target_txt}\n\n"
            "NeuXelec electrode reconstruction currently assumes a 1 × 1 × 1 mm "
            "T1 / CT-in-T1 grid. Continuing with a non-isotropic T1 may produce "
            "spatially shifted or scaled electrode coordinates.\n\n"
            "Do you want to create a 1 mm isotropic copy of this T1 and use it "
            "for the current project?"
        ),
        accept_text="Create 1 mm copy",
        reject_text="Keep original",
    )

    if not create_conformed_copy:
        return input_path, img, info

    suggested = make_conformed_t1_output_path(input_path)

    if force_ask_output_path:
        start_path = suggested
    else:
        start_path = suggested

    output_path, _ = QFileDialog.getSaveFileName(
        parent,
        "Save 1 mm isotropic T1",
        start_path,
        "NIfTI image (*.nii.gz *.nii);;All files (*)",
    )

    if not output_path:
        return input_path, img, info

    if not output_path.lower().endswith((".nii", ".nii.gz")):
        output_path += ".nii.gz"

    try:
        out_img = conform_t1_to_1mm(
            input_path=input_path,
            output_path=output_path,
            target_spacing=target_spacing,
        )

    except Exception as e:
        NeuXelecMessageDialog.critical(
            parent,
            "T1 conform failed",
            f"Could not resample the T1 image to 1 mm isotropic:\n\n{e}",
        )
        return input_path, img, info

    final_spacing = get_image_spacing(out_img)

    info.update(
        {
            "was_conformed": True,
            "final_path": str(output_path),
            "final_spacing": [float(v) for v in final_spacing],
        }
    )

    NeuXelecMessageDialog.information(
        parent,
        "T1 conformed",
        (
            "A 1 mm isotropic T1 copy was created and will be used "
            "for the current project.\n\n"
            f"Saved as:\n{output_path}"
        ),
    )

    return str(output_path), out_img, info
