from __future__ import annotations

"""
Coregistration utilities (CT/PET/SPECT/T2 -> T1).

Public API kept stable:
  - rigid_coreg_to_fixed(...)
  - CoregResult
  - save_nifti(...)

Backend:
  - ANTs ONLY (external binaries shipped with the app)

Design goals (requested):
  - Do NOT require the rest of the codebase to change.
  - Keep LPS physical space (SimpleITK convention).
  - Write outputs (Warped NIfTI + transforms) to a user-chosen GLOBAL directory
    when provided; otherwise use a temporary directory.

Notes:
  * Display flips/rotations in the viewer do not change physical coordinates.
  * For SEEG/electrodes, CT->T1 is kept RIGID-only by default (no affine scaling).
"""

import logging
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import SimpleITK as sitk

logger = logging.getLogger(__name__)


# -----------------------------
# Public data structure
# -----------------------------
@dataclass
class CoregResult:
    transform: sitk.Transform
    moving_in_fixed: sitk.Image
    fixed: sitk.Image
    moving: sitk.Image
    affine_mat_path: str | None = None
    ants_work_dir: str | None = None


def _report_progress(progress_cb: Callable[[int], None] | None, value: int) -> None:
    """Invoke a progress callback, ignoring (but logging) any UI-side failure.

    A failing progress bar must never abort a coregistration, but the failure
    is recorded at debug level rather than silently swallowed.
    """
    if progress_cb is None:
        return
    try:
        progress_cb(value)
    except Exception:
        logger.debug("Progress callback failed at %d%%", value, exc_info=True)


def save_nifti(img: sitk.Image, out_path: str) -> None:
    sitk.WriteImage(img, out_path)


# =============================================================================
# Path helpers (robust in dev + PyInstaller)
# =============================================================================


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False) is True


def _candidate_roots() -> list[Path]:
    """
    Build a list of candidate roots where we might find tools/ants and tools/templates.
    We search upward from this file, plus PyInstaller's _MEIPASS when frozen.
    """
    roots: list[Path] = []

    if _is_frozen() and hasattr(sys, "_MEIPASS"):
        roots.append(Path(sys._MEIPASS))

    p = Path(__file__).resolve()
    # Search a few levels up to be safe (works even if package layout changes)
    # coregistration.py is usually: <root>/src/neuxelec/coregistration.py
    for i in range(0, 7):
        try:
            roots.append(p.parents[i])
        except IndexError:
            # Reached the filesystem root: no more parent levels.
            break

    # De-duplicate preserving order
    seen = set()
    uniq: list[Path] = []
    for r in roots:
        rp = r.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


def _find_tools_dir() -> Path:
    """
    Find the 'tools' directory.
    Expected structure:
      <root>/tools/ants/antsRegistration.exe
      <root>/tools/templates/<template files>
    """
    for root in _candidate_roots():
        tools = root / "tools"
        if (tools / "ants" / "antsRegistration.exe").exists() and (
            tools / "ants" / "antsApplyTransforms.exe"
        ).exists():
            return tools
    # fallback: try plain 'tools' relative to CWD (last resort)
    tools = Path.cwd() / "tools"
    if (tools / "ants" / "antsRegistration.exe").exists() and (
        tools / "ants" / "antsApplyTransforms.exe"
    ).exists():
        return tools

    # build a readable error with what we tried
    tried = "\n".join(str((r / "tools" / "ants").resolve()) for r in _candidate_roots())
    raise FileNotFoundError(
        "ANTs binaries not found.\n"
        "Expected to find:\n"
        "  tools/ants/antsRegistration.exe\n"
        "  tools/ants/antsApplyTransforms.exe\n\n"
        "Searched (tools/ants) under:\n"
        f"{tried}\n\n"
        "Fix: create <project_root>/tools/ants/ and copy ALL ANTs executables there."
    )


def _ants_dir() -> Path:
    return _find_tools_dir() / "ants"


def _templates_dir() -> Path:
    return _find_tools_dir() / "templates"


def _ants_exe(name: str) -> Path:
    p = _ants_dir() / name
    if not p.exists():
        raise FileNotFoundError(f"Missing ANTs executable: {p}")
    return p


def _run_cmd(cmd: list[str], cwd: str | None = None) -> None:
    """
    Run a command and raise a readable error if it fails.
    Ensures ANTs folder is on PATH so dependencies can be found.
    """
    env = os.environ.copy()
    ants_path = str(_ants_dir())
    env["PATH"] = ants_path + os.pathsep + env.get("PATH", "")

    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    try:
        subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            creationflags=creation_flags,
        )
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip()
        if not msg:
            msg = f"ANTs command failed with exit code {e.returncode}"
        raise RuntimeError(msg)


def _safe_slug(s: str) -> str:
    out = []
    for ch in s or "":
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "modality"


def _ants_out_dir(transforms_dir: str | None) -> Path:
    if transforms_dir:
        d = Path(transforms_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d
    return Path(tempfile.mkdtemp(prefix="neuxelec_ants_"))


# =============================================================================
# Template selection (accept .nii or .nii.gz)
# =============================================================================


def _pick_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _default_brainmask_template_paths() -> tuple[Path, Path]:
    """
    Accept common names + your suggested names:
      - tools/templates/template_T1.nii(.gz)
      - tools/templates/template_brain_mask.nii(.gz)
    Also accept older MNI-style names if you prefer them.
    """
    tdir = _templates_dir()

    template_candidates = [
        tdir / "template_T1.nii.gz",
        tdir / "template_T1.nii",
        # optional alt names
        tdir / "MNI152_T1_1mm.nii.gz",
        tdir / "MNI152_T1_1mm.nii",
    ]

    mask_candidates = [
        tdir / "template_brain_mask.nii.gz",
        tdir / "template_brain_mask.nii",
        tdir / "template_brainmask.nii.gz",
        tdir / "template_brainmask.nii",
        # optional alt names
        tdir / "MNI152_T1_1mm_brainmask.nii.gz",
        tdir / "MNI152_T1_1mm_brainmask.nii",
        tdir / "MNI152_T1_1mm_brain_mask.nii.gz",
        tdir / "MNI152_T1_1mm_brain_mask.nii",
    ]

    tpl = _pick_existing(template_candidates)
    msk = _pick_existing(mask_candidates)

    # If you kept original ICBM names, accept them too:
    if tpl is None:
        tpl = _pick_existing(list(tdir.glob("*t1*VI*.nii*")) + list(tdir.glob("*t1*.nii*")))
    if msk is None:
        msk = _pick_existing(list(tdir.glob("*_mask.nii*")))

    if tpl is None or msk is None:
        raise FileNotFoundError(
            "Brainmask templates not found.\n"
            "Place template files under tools/templates/ with one of these names:\n"
            "  - template_T1.nii   (or .nii.gz)\n"
            "  - template_brain_mask.nii   (or .nii.gz)\n\n"
            f"Templates directory searched: {tdir}"
        )

    return tpl, msk


# =============================================================================
# ANTs brain mask (on-demand)
# =============================================================================


def ants_generate_brainmask_t1(
    t1_path: str,
    out_dir: str | None = None,
    progress_cb: Callable[[int], None] | None = None,
    template_t1_path: str | None = None,
    template_mask_path: str | None = None,
) -> str:
    """
    Generate a brain mask for a subject T1 using ANTs (Windows-friendly, no bash scripts).

    Method: template-based
      1) Register subject T1 (moving) to template T1 (fixed) (Affine + light SyN)
      2) Bring the TEMPLATE brain mask back into SUBJECT space using inverse transforms
      3) Write SUBJECT brainmask as NIfTI (uint8 0/1)

    Returns: path to T1_brainmask.nii.gz (subject space)
    """
    # triggers clear error early if not found:
    _ants_exe("antsRegistration.exe")
    _ants_exe("antsApplyTransforms.exe")

    _report_progress(progress_cb, 0)

    if template_t1_path is None or template_mask_path is None:
        tpl, msk = _default_brainmask_template_paths()
        template_t1_path = template_t1_path or str(tpl)
        template_mask_path = template_mask_path or str(msk)

    if not Path(template_t1_path).exists() or not Path(template_mask_path).exists():
        raise FileNotFoundError(
            "Brainmask template files not found:\n"
            f"  template T1:  {template_t1_path}\n"
            f"  template mask:{template_mask_path}\n"
        )

    outd = _ants_out_dir(out_dir)
    prefix = str(outd / "brainmask_")
    ants_reg = str(_ants_exe("antsRegistration.exe"))
    ants_apply = str(_ants_exe("antsApplyTransforms.exe"))

    # Register subject -> template
    reg_cmd = [
        ants_reg,
        "--dimensionality",
        "3",
        "--float",
        "1",
        "--output",
        f"[{prefix},{prefix}Warped.nii.gz,{prefix}InverseWarped.nii.gz]",
        "--interpolation",
        "Linear",
        "--winsorize-image-intensities",
        "[0.005,0.995]",
        "--use-histogram-matching",
        "0",
        "--initial-moving-transform",
        f"[{template_t1_path},{t1_path},1]",
        # Affine
        "--transform",
        "Affine[0.1]",
        "--metric",
        f"MI[{template_t1_path},{t1_path},1,32,Regular,0.25]",
        "--convergence",
        "[500x250x100,1e-6,10]",
        "--shrink-factors",
        "8x4x2",
        "--smoothing-sigmas",
        "3x2x1vox",
        # Light SyN
        "--transform",
        "SyN[0.1,3,0]",
        "--metric",
        f"CC[{template_t1_path},{t1_path},1,4]",
        "--convergence",
        "[80x40x20,1e-6,10]",
        "--shrink-factors",
        "8x4x2",
        "--smoothing-sigmas",
        "3x2x1vox",
    ]
    _run_cmd(reg_cmd, cwd=str(outd))

    _report_progress(progress_cb, 70)

    affine = f"{prefix}0GenericAffine.mat"
    inv_warp = f"{prefix}1InverseWarp.nii.gz"

    out_mask = str(outd / "T1_brainmask.nii.gz")

    apply_cmd = [
        ants_apply,
        "-d",
        "3",
        "-i",
        str(template_mask_path),
        "-r",
        str(t1_path),
        "-o",
        out_mask,
        "-n",
        "NearestNeighbor",
        "-t",
        inv_warp,
        "-t",
        f"[{affine},1]",
    ]
    _run_cmd(apply_cmd, cwd=str(outd))

    # Binarize + cast for safety
    try:
        m = sitk.ReadImage(out_mask)
        m = sitk.Cast(m > 0.5, sitk.sitkUInt8)
        sitk.WriteImage(m, out_mask)
    except Exception:
        logger.warning(
            "Failed to binarize/cast brain mask %s; leaving ANTs output as-is",
            out_mask,
            exc_info=True,
        )

    _report_progress(progress_cb, 100)

    return out_mask


# =============================================================================
# ANTs coregistration
# =============================================================================


def _ants_registration_args(
    fixed_path: str,
    moving_path: str,
    out_prefix: str,
    modality: str,
) -> list[str]:
    """
    Build antsRegistration arguments.

    CT->T1: Rigid-only (SEEG safe)
    Others: Rigid + Affine (often improves overlap)
    """
    ants_reg = str(_ants_exe("antsRegistration.exe"))
    mod = (modality or "AUTO").upper()

    shrink = "8x4x2x1"
    smooth = "3x2x1x0"

    args = [
        ants_reg,
        "--dimensionality",
        "3",
        "--float",
        "1",
        "--output",
        f"[{out_prefix},{out_prefix}Warped.nii.gz]",
        "--interpolation",
        "Linear",
        "--use-histogram-matching",
        "0",
        "--winsorize-image-intensities",
        "[0.005,0.995]",
        "--initial-moving-transform",
        f"[{fixed_path},{moving_path},1]",
        "--transform",
        "Rigid[0.1]",
        "--metric",
        f"MI[{fixed_path},{moving_path},1,64,Regular,0.25]",
        "--convergence",
        "[1000x500x250x100,1e-6,10]",
        "--shrink-factors",
        shrink,
        "--smoothing-sigmas",
        smooth,
    ]

    if mod != "CT":
        args += [
            "--transform",
            "Affine[0.1]",
            "--metric",
            f"MI[{fixed_path},{moving_path},1,64,Regular,0.25]",
            "--convergence",
            "[1000x500x250x100,1e-6,10]",
            "--shrink-factors",
            shrink,
            "--smoothing-sigmas",
            smooth,
        ]

    return args


def ants_coreg_to_fixed(
    fixed_path: str,
    moving_path: str,
    transforms_dir: str | None,
    progress_cb: Callable[[int], None] | None = None,
    moving_modality: str = "AUTO",
) -> tuple[sitk.Transform, sitk.Image, str | None, str | None]:
    """
    Run ANTs registration moving->fixed and return:
      - a sitk.Transform (identity placeholder; use .mat files if needed later)
      - moving resampled into fixed space as sitk.Image

    Files produced in transforms_dir (or temp):
      - <prefix>0GenericAffine.mat (always)
      - <prefix>Warped.nii.gz (moving in fixed)
    """
    # fail early if binaries missing:
    _ants_exe("antsRegistration.exe")

    out_dir = _ants_out_dir(transforms_dir)
    mod_slug = _safe_slug(moving_modality.upper() if moving_modality else "moving")
    prefix = out_dir / f"{mod_slug}_to_T1_"
    out_prefix = str(prefix)

    _report_progress(progress_cb, 0)

    reg_cmd = _ants_registration_args(
        fixed_path=fixed_path,
        moving_path=moving_path,
        out_prefix=out_prefix,
        modality=moving_modality,
    )
    _run_cmd(reg_cmd, cwd=str(out_dir))

    warped_path = str(prefix) + "Warped.nii.gz"
    if not Path(warped_path).exists():
        raise RuntimeError(f"ANTs did not produce the expected output image: {warped_path}")

    moving_in_fixed = sitk.ReadImage(warped_path)

    # App uses the resampled image; if later you need transforms for exporting points,
    # we will parse <prefix>0GenericAffine.mat and compose LPS transform properly.
    transform = sitk.Transform(3, sitk.sitkIdentity)

    _report_progress(progress_cb, 100)

    affine_mat_path = str(prefix) + "0GenericAffine.mat"
    if not Path(affine_mat_path).exists():
        affine_mat_path = None

    return transform, moving_in_fixed, affine_mat_path, str(out_dir)


# =============================================================================
# Public API (used by the app)
# =============================================================================


def rigid_coreg_to_fixed(
    fixed_path: str,
    moving_path: str,
    progress_cb: Callable[[int], None] | None = None,
    initial_transform: sitk.Transform | None = None,  # kept for compatibility, unused in ANTs-only
    n_iter: int = 200,  # kept for compatibility, unused in ANTs-only
    use_elastix: bool = True,  # kept for compatibility, unused in ANTs-only
    moving_modality: str = "AUTO",
    use_ants: bool = True,  # kept for compatibility; must be True in ANTs-only
    transforms_dir: str | None = None,
) -> CoregResult:
    """
    Coregister moving -> fixed (ANTs only).
    """
    fixed = sitk.ReadImage(fixed_path)
    moving = sitk.ReadImage(moving_path)

    # Preferred/only: ANTs
    if not use_ants:
        raise RuntimeError("ANTs-only build: use_ants must be True.")

    t, moving_in_fixed, affine_mat_path, ants_work_dir = ants_coreg_to_fixed(
        fixed_path=fixed_path,
        moving_path=moving_path,
        transforms_dir=transforms_dir,
        progress_cb=progress_cb,
        moving_modality=moving_modality,
    )
    return CoregResult(
        transform=t,
        moving_in_fixed=moving_in_fixed,
        fixed=fixed,
        moving=moving,
        affine_mat_path=affine_mat_path,
        ants_work_dir=ants_work_dir,
    )


def transform_params(
    transform: sitk.Transform,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return (rotation_deg, translation_mm) if possible (mostly for debug)."""
    rot_deg = (float("nan"), float("nan"), float("nan"))
    trans_mm = (float("nan"), float("nan"), float("nan"))
    try:
        if (
            isinstance(transform, sitk.Euler3DTransform)
            or transform.GetName() == "Euler3DTransform"
        ):
            e = sitk.Euler3DTransform(transform)
            rx, ry, rz = e.GetAngleX(), e.GetAngleY(), e.GetAngleZ()
            tx, ty, tz = e.GetTranslation()
            rot_deg = (math.degrees(rx), math.degrees(ry), math.degrees(rz))
            trans_mm = (float(tx), float(ty), float(tz))
        elif (
            isinstance(transform, sitk.AffineTransform) or transform.GetName() == "AffineTransform"
        ):
            tx, ty, tz = transform.GetTranslation()
            trans_mm = (float(tx), float(ty), float(tz))
    except Exception:
        logger.debug("Could not extract transform parameters", exc_info=True)
    return rot_deg, trans_mm
