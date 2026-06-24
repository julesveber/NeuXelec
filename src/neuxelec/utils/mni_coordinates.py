from __future__ import annotations

import csv
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

from neuxelec.coregistration import (
    _ants_dir,
    _ants_exe,
    _default_brainmask_template_paths,
)


def _report_progress(progress_callback, message: str, value: int, maximum: int = 100):
    """
    Safe progress reporter.

    progress_callback signature:
        progress_callback(message: str, value: int, maximum: int)
    """
    if progress_callback is None:
        return

    try:
        progress_callback(str(message), int(value), int(maximum))
    except Exception:
        logger.debug("MNI progress callback failed", exc_info=True)


def _run_cmd_with_progress(
    cmd,
    *,
    cwd=None,
    progress_callback=None,
    start_value: int = 0,
    end_value: int = 100,
    message: str = "Running command...",
):
    """
    Run a command while keeping the Qt UI responsive through progress callbacks.

    This does not give a perfect ANTs percentage, because ANTs does not expose
    a clean percentage for all registration stages. Instead, it gives an
    approximate staged progress.
    """
    _report_progress(progress_callback, message, start_value, 100)

    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    # Ensure ANTs folder is on PATH so dependent DLLs can be found.
    env = os.environ.copy()
    ants_path = str(_ants_dir())
    env["PATH"] = ants_path + os.pathsep + env.get("PATH", "")

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        creationflags=creation_flags,
    )

    output_lines = []
    current_value = int(start_value)
    last_tick = time.time()

    while True:
        line = process.stdout.readline() if process.stdout is not None else ""

        if line:
            output_lines.append(line)

            # Increment slowly while ANTs prints logs.
            current_value = min(int(end_value) - 1, current_value + 1)

            short_line = line.strip()
            if len(short_line) > 120:
                short_line = short_line[:117] + "..."

            _report_progress(
                progress_callback,
                f"{message}\n{short_line}",
                current_value,
                100,
            )

        ret = process.poll()

        # Even if ANTs is quiet, keep the progress dialog alive.
        now = time.time()
        if now - last_tick > 0.5:
            current_value = min(int(end_value) - 1, current_value + 1)
            _report_progress(progress_callback, message, current_value, 100)
            last_tick = now

        if ret is not None:
            break

        time.sleep(0.05)

    if process.stdout is not None:
        remaining = process.stdout.read()
        if remaining:
            output_lines.append(remaining)

    if process.returncode != 0:
        msg = "".join(output_lines).strip()
        raise RuntimeError("Command failed:\n" + " ".join(str(x) for x in cmd) + "\n\n" + msg)

    _report_progress(progress_callback, message, end_value, 100)


def _as_existing_path(path_value) -> str | None:
    if not path_value:
        return None

    try:
        p = Path(str(path_value))
        if p.exists():
            return str(p)
    except Exception:
        logger.debug("Invalid path value: %r", path_value, exc_info=True)

    return None


def _default_mni_output_dir(state) -> Path:
    """
    Choose a persistent folder for T1 -> MNI transforms.

    Priority:
    1. state.transforms_dir if available
    2. project folder / transforms
    3. temp folder as fallback
    """
    transforms_dir = getattr(state, "transforms_dir", None)

    if transforms_dir:
        out_dir = Path(str(transforms_dir)) / "mni"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    project_path = getattr(state, "project_path", None)

    if project_path:
        out_dir = Path(str(project_path)).parent / "transforms" / "mni"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    out_dir = Path(tempfile.mkdtemp(prefix="neuxelec_mni_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_existing_t1_to_mni_transform_paths(
    state,
    expected_template_path: str | None = None,
) -> dict[str, str] | None:
    """
    Return existing transform paths only if they match the current MNI template.
    This prevents reusing old transforms after changing the MNI template/brainmask.
    """
    affine = _as_existing_path(getattr(state, "t1_to_mni_affine_path", None))
    warp = _as_existing_path(getattr(state, "t1_to_mni_warp_path", None))
    inverse_warp = _as_existing_path(getattr(state, "t1_to_mni_inverse_warp_path", None))
    stored_template = _as_existing_path(getattr(state, "mni_template_path", None))

    if not affine:
        return None

    if not inverse_warp:
        return None

    if expected_template_path:
        try:
            expected = str(Path(expected_template_path).resolve())
            stored = str(Path(stored_template).resolve()) if stored_template else ""

            if stored != expected:
                logger.info(
                    "Existing MNI transforms ignored because template changed "
                    "(stored=%s, expected=%s)",
                    stored,
                    expected,
                )
                return None

        except Exception:
            logger.warning("Could not compare MNI template paths", exc_info=True)
            return None

    return {
        "affine": affine,
        "warp": warp or "",
        "inverse_warp": inverse_warp or "",
        "template": stored_template or "",
        "space_name": str(getattr(state, "mni_space_name", "") or "MNI152NLin2009cAsym"),
    }


def ensure_t1_to_mni_transforms(
    state,
    *,
    force: bool = False,
    template_t1_path: str | None = None,
    progress_callback=None,
) -> dict[str, str]:
    """
    Ensure that a T1-native -> MNI transform exists.

    This runs ANTs registration:
        moving = patient T1
        fixed  = MNI/template T1

    Produced transforms:
        T1_to_MNI_0GenericAffine.mat
        T1_to_MNI_1Warp.nii.gz
        T1_to_MNI_1InverseWarp.nii.gz

    These are stored in state for later BIDS export and project saving.
    """
    _report_progress(progress_callback, "Checking existing T1 → MNI transforms...", 2, 100)

    # Always resolve the current default MNI template first.
    # Do not blindly reuse state.mni_template_path, because it may point to
    # an old template after the user replaced the MNI files.
    if template_t1_path is None:
        template_t1, _template_mask = _default_brainmask_template_paths()
        template_t1_path = str(template_t1)

    if not Path(template_t1_path).exists():
        raise FileNotFoundError(f"MNI template T1 not found:\n{template_t1_path}")

    if not force:
        existing = get_existing_t1_to_mni_transform_paths(
            state,
            expected_template_path=template_t1_path,
        )
        if existing is not None:
            _report_progress(progress_callback, "Existing T1 → MNI transforms found.", 15, 100)
            return existing

    t1_path = _as_existing_path(getattr(state, "t1_path", None)) or _as_existing_path(
        getattr(state, "t1_source_path", None)
    )

    if not t1_path:
        raise FileNotFoundError(
            "No T1 image is available. Load a T1 MRI before exporting MNI coordinates."
        )

    out_dir = _default_mni_output_dir(state)
    prefix = str(out_dir / "T1_to_MNI_")

    affine_path = f"{prefix}0GenericAffine.mat"
    warp_path = f"{prefix}1Warp.nii.gz"
    inverse_warp_path = f"{prefix}1InverseWarp.nii.gz"
    warped_path = f"{prefix}Warped.nii.gz"
    inverse_warped_path = f"{prefix}InverseWarped.nii.gz"

    ants_reg = str(_ants_exe("antsRegistration.exe"))

    reg_cmd = [
        ants_reg,
        "--dimensionality",
        "3",
        "--float",
        "1",
        "--output",
        f"[{prefix},{warped_path},{inverse_warped_path}]",
        "--interpolation",
        "Linear",
        "--winsorize-image-intensities",
        "[0.005,0.995]",
        "--use-histogram-matching",
        "0",
        # fixed = MNI template, moving = patient T1
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

    _run_cmd_with_progress(
        reg_cmd,
        cwd=str(out_dir),
        progress_callback=progress_callback,
        start_value=10,
        end_value=85,
        message="Registering patient T1 to MNI template with ANTs...",
    )

    if not Path(affine_path).exists():
        raise RuntimeError(f"ANTs did not produce affine transform:\n{affine_path}")

    if not Path(warp_path).exists():
        raise RuntimeError(f"ANTs did not produce forward warp:\n{warp_path}")

    # Store in state
    state.mni_template_path = str(template_t1_path)
    state.mni_space_name = str(getattr(state, "mni_space_name", "") or "MNI152NLin2009cAsym")
    state.t1_to_mni_affine_path = str(affine_path)
    state.t1_to_mni_warp_path = str(warp_path)
    state.t1_to_mni_inverse_warp_path = (
        str(inverse_warp_path) if Path(inverse_warp_path).exists() else None
    )
    state.t1_to_mni_warped_path = str(warped_path) if Path(warped_path).exists() else None
    _report_progress(progress_callback, "T1 → MNI transform ready.", 88, 100)
    return {
        "affine": str(affine_path),
        "warp": str(warp_path),
        "inverse_warp": str(inverse_warp_path) if Path(inverse_warp_path).exists() else "",
        "template": str(template_t1_path),
        "space_name": str(state.mni_space_name),
    }


def apply_t1_lps_to_mni_lps(
    points_lps: Iterable[tuple[float, float, float]],
    *,
    affine_path: str,
    inverse_warp_path: str | None = None,
    progress_callback=None,
) -> list[tuple[float, float, float]]:
    """
    Transform native T1 physical LPS points into MNI physical LPS points.

    Important ANTs convention:
    antsApplyTransformsToPoints uses the opposite direction compared with
    image resampling.

    Registration was:
        fixed  = MNI template
        moving = patient T1

    For image resampling T1 -> MNI, one would use:
        -t T1_to_MNI_1Warp.nii.gz
        -t T1_to_MNI_0GenericAffine.mat

    For POINTS T1 -> MNI, we must use:
        -t [T1_to_MNI_0GenericAffine.mat,1]
        -t T1_to_MNI_1InverseWarp.nii.gz
    """
    exe = str(_ants_exe("antsApplyTransformsToPoints.exe"))

    affine_path = str(affine_path)
    inverse_warp_path = str(inverse_warp_path) if inverse_warp_path else ""

    if not Path(affine_path).exists():
        raise FileNotFoundError(f"Affine transform not found:\n{affine_path}")

    if inverse_warp_path and not Path(inverse_warp_path).exists():
        raise FileNotFoundError(f"Inverse warp transform not found:\n{inverse_warp_path}")

    points = [tuple(float(v) for v in p) for p in points_lps]

    if not points:
        return []

    with tempfile.TemporaryDirectory(prefix="neuxelec_mni_points_") as tmp:
        tmp = Path(tmp)

        in_csv = tmp / "points_native_t1_lps.csv"
        out_csv = tmp / "points_mni_lps.csv"

        # ANTs is happier with x,y,z,t columns.
        with open(in_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["x", "y", "z", "t"])
            writer.writeheader()

            for x, y, z in points:
                writer.writerow(
                    {
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "t": 0.0,
                    }
                )

        cmd = [
            exe,
            "-d",
            "3",
            "-i",
            str(in_csv),
            "-o",
            str(out_csv),
            # POINT transform T1 -> MNI:
            # invert affine, then apply inverse warp.
            "-t",
            f"[{affine_path},1]",
        ]

        if inverse_warp_path:
            cmd += ["-t", inverse_warp_path]

        _run_cmd_with_progress(
            cmd,
            cwd=str(tmp),
            progress_callback=progress_callback,
            start_value=88,
            end_value=95,
            message="Transforming SEEG contacts to MNI coordinates...",
        )

        if not out_csv.exists():
            raise RuntimeError("ANTs did not produce the transformed point file.")

        out_points: list[tuple[float, float, float]] = []

        with open(out_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                out_points.append(
                    (
                        float(row["x"]),
                        float(row["y"]),
                        float(row["z"]),
                    )
                )

        if len(out_points) != len(points):
            raise RuntimeError(
                f"ANTs returned {len(out_points)} points for {len(points)} input points."
            )

        return out_points


def transform_contacts_t1_lps_to_mni_lps(
    state,
    contacts_lps: Iterable[tuple[float, float, float]],
    *,
    force_recompute_transform: bool = False,
    progress_callback=None,
) -> list[tuple[float, float, float]]:
    """
    High-level helper used by the export dialog.
    """
    _report_progress(progress_callback, "Preparing MNI export...", 1, 100)

    transforms = ensure_t1_to_mni_transforms(
        state,
        force=bool(force_recompute_transform),
        progress_callback=progress_callback,
    )

    out = apply_t1_lps_to_mni_lps(
        contacts_lps,
        affine_path=transforms["affine"],
        inverse_warp_path=transforms.get("inverse_warp") or None,
        progress_callback=progress_callback,
    )

    _report_progress(progress_callback, "MNI coordinates computed.", 96, 100)

    return out
