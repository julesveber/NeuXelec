from __future__ import annotations

from pathlib import Path

import nibabel as nib
import SimpleITK as sitk


def _reorient_sitk(img: sitk.Image, orient: str = "RAS") -> sitk.Image:
    """
    Reorient image to a canonical orientation (RAS recommended for neuro tools).
    This avoids left/right or anterior/posterior flips when mixing DICOM and NIfTI.
    """
    try:
        return sitk.DICOMOrient(img, orient)
    except Exception:
        # If orientation fails for any reason, keep original
        return img


def convert_dicom_folder_to_nifti(dicom_folder: str, out_nifti_path: str) -> str:
    """
    Convert a DICOM series folder to NIfTI (.nii/.nii.gz) using SimpleITK.
    Returns the output path.
    """
    dicom_folder = str(dicom_folder)
    out_nifti_path = str(out_nifti_path)

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_folder)
    if not series_ids:
        raise RuntimeError("No DICOM series found in the selected folder.")

    # Take first series by default (can be improved later with series selection UI)
    series_id = series_ids[0]
    file_names = reader.GetGDCMSeriesFileNames(dicom_folder, series_id)
    reader.SetFileNames(file_names)

    img = reader.Execute()
    img = _reorient_sitk(img, "RAS")  # ✅ important
    sitk.WriteImage(img, out_nifti_path)
    return out_nifti_path


def convert_mgz_to_nifti(mgz_path: str, out_nifti_path: str) -> str:
    """
    Convert FreeSurfer .mgz/.mgh to NIfTI (.nii/.nii.gz) using nibabel.

    IMPORTANT:
      FreeSurfer volumes can come in a different orientation than typical neuro NIfTI.
      We reorient to a canonical orientation to avoid flips in downstream viewers.
    """
    mgz_path = str(mgz_path)
    out_nifti_path = str(out_nifti_path)

    img = nib.load(mgz_path)  # supports .mgz and .mgh

    # ✅ Make orientation canonical (RAS-like in nibabel terminology)
    img = nib.as_closest_canonical(img)

    # ✅ Ensure qform/sform are set (some tools behave better with both)
    try:
        affine = img.affine
        img.set_qform(affine, code=1)
        img.set_sform(affine, code=1)
    except Exception:
        pass

    nib.save(img, out_nifti_path)
    return out_nifti_path


def convert_to_nifti_if_needed(
    input_path: str,
    parent=None,
    output_path: str | None = None,
) -> str:
    """Convert DICOM folder or .mgz/.mgh to NIfTI and return the NIfTI path.

    Rules:
      - If input_path is already NIfTI (.nii / .nii.gz), it is returned as-is.
      - If it is a directory, it is treated as a DICOM folder.
      - If it is .mgz/.mgh, it is treated as a FreeSurfer volume.
      - If it is a single DICOM file (.dcm/.ima), we do a best-effort conversion.

    If output_path is provided, the conversion is saved there without opening
    a save dialog. This is useful for bulk DICOM import.
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    # Already NIfTI
    if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
        return str(p)

    def _normalize_nifti_output_path(path_like: str) -> str:
        out = str(path_like)

        # Keep .nii.gz unchanged
        if out.endswith(".nii.gz"):
            return out

        # Keep .nii unchanged
        if out.endswith(".nii"):
            return out

        # Otherwise default to .nii.gz
        return out + ".nii.gz"

    # If no output path was provided, keep the old behavior:
    # ask the user where to save this converted file.
    if output_path is None:
        from PySide6.QtWidgets import QFileDialog

        source_dir = p.parent if p.is_file() else p
        default_name = (p.stem + ".nii.gz") if p.is_file() else (p.name + ".nii.gz")
        default_path = str(source_dir / default_name)

        dialog_options = QFileDialog.Options()

        output_path, _ = QFileDialog.getSaveFileName(
            parent,
            "Save converted NIfTI",
            default_path,
            "NIfTI (*.nii.gz *.nii);;All files (*.*)",
            options=dialog_options,
        )

        if not output_path:
            raise RuntimeError("Conversion cancelled by user.")

    out_path = _normalize_nifti_output_path(str(output_path))

    if p.is_dir():
        return convert_dicom_folder_to_nifti(str(p), out_path)

    if p.suffix.lower() in {".mgz", ".mgh"}:
        return convert_mgz_to_nifti(str(p), out_path)

    if p.suffix.lower() in {".dcm", ".ima"}:
        img = sitk.ReadImage(str(p))
        img = _reorient_sitk(img, "RAS")
        sitk.WriteImage(img, out_path)
        return out_path

    return str(p)
