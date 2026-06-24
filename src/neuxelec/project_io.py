from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_SCHEMA_VERSION = 1


def _as_str_or_none(v):
    return None if v in (None, "") else str(v)


def _safe_list(v):
    return v if isinstance(v, list) else []


def build_project_dict_from_state(state) -> dict[str, Any]:
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "last_saved": datetime.now().isoformat(timespec="seconds"),
        "patient_id": _as_str_or_none(getattr(state, "patient_id", None)),
        "mri_labels": {
            "mri1_filename_label": _as_str_or_none(getattr(state, "mri1_filename_label", None))
            or "MRI1",
            "mri2_filename_label": _as_str_or_none(getattr(state, "mri2_filename_label", None))
            or "MRI2",
        },
        "files": {
            "t1": {
                # Actual T1 used by NeuXelec.
                # This can be the original NIfTI, or the 1 mm isotropic conformed copy.
                "path": _as_str_or_none(getattr(state, "t1_path", None)),
                # Original source selected by the user:
                # DICOM folder / MGZ / MGH / original NIfTI.
                "source_path": _as_str_or_none(getattr(state, "t1_source_path", None)),
                # T1 conform metadata.
                "was_conformed": bool(getattr(state, "t1_was_conformed", False)),
                "conformed_path": _as_str_or_none(getattr(state, "t1_conformed_path", None)),
                "original_spacing": getattr(state, "t1_original_spacing", None),
                "conformed_spacing": getattr(state, "t1_conformed_spacing", None),
            },
            "t2": {
                "path": _as_str_or_none(getattr(state, "t2_path", None)),
                "source_path": _as_str_or_none(getattr(state, "t2_source_path", None)),
                "coreg_in_t1_path": _as_str_or_none(getattr(state, "t2_coreg_path", None)),
                "validated": bool(getattr(state, "t2_validated", False)),
            },
            "ct": {
                "path": _as_str_or_none(getattr(state, "ct_path", None)),
                "source_path": _as_str_or_none(getattr(state, "ct_source_path", None)),
                "coreg_in_t1_path": _as_str_or_none(getattr(state, "ct_coreg_path", None)),
                "validated": bool(getattr(state, "ct_validated", False)),
            },
            "pet": {
                "path": _as_str_or_none(getattr(state, "pet_path", None)),
                "source_path": _as_str_or_none(getattr(state, "pet_source_path", None)),
                "coreg_in_t1_path": _as_str_or_none(getattr(state, "pet_coreg_path", None)),
                "validated": bool(getattr(state, "pet_validated", False)),
            },
            "ictal_spect": {
                "path": _as_str_or_none(getattr(state, "ictal_spect_path", None)),
                "source_path": _as_str_or_none(getattr(state, "ictal_spect_source_path", None)),
                "coreg_in_t1_path": _as_str_or_none(getattr(state, "ictal_spect_coreg_path", None)),
                "validated": bool(getattr(state, "ictal_spect_validated", False)),
            },
            "interictal_spect": {
                "path": _as_str_or_none(getattr(state, "interictal_spect_path", None)),
                "source_path": _as_str_or_none(
                    getattr(state, "interictal_spect_source_path", None)
                ),
                "coreg_in_t1_path": _as_str_or_none(
                    getattr(state, "interictal_spect_coreg_path", None)
                ),
                "validated": bool(getattr(state, "interictal_spect_validated", False)),
            },
            "brainmask": {
                "path": _as_str_or_none(getattr(state, "brainmask_path", None)),
                "generated": bool(getattr(state, "brainmask_generated", False)),
                "saved": bool(getattr(state, "brainmask_saved", False)),
                "generated_path": _as_str_or_none(getattr(state, "brainmask_generated_path", None)),
            },
            "mni": {
                "space_name": _as_str_or_none(getattr(state, "mni_space_name", None)),
                "template_path": _as_str_or_none(getattr(state, "mni_template_path", None)),
                "t1_to_mni_affine_path": _as_str_or_none(
                    getattr(state, "t1_to_mni_affine_path", None)
                ),
                "t1_to_mni_warp_path": _as_str_or_none(getattr(state, "t1_to_mni_warp_path", None)),
                "t1_to_mni_inverse_warp_path": _as_str_or_none(
                    getattr(state, "t1_to_mni_inverse_warp_path", None)
                ),
                "t1_to_mni_warped_path": _as_str_or_none(
                    getattr(state, "t1_to_mni_warped_path", None)
                ),
            },
            "parcel1": {
                "path": _as_str_or_none(getattr(state, "parcel1_path", None)),
            },
            "parcel2": {
                "path": _as_str_or_none(getattr(state, "parcel2_path", None)),
            },
            "lh_pial": {
                "path": _as_str_or_none(getattr(state, "lh_pial_path", None)),
            },
            "rh_pial": {
                "path": _as_str_or_none(getattr(state, "rh_pial_path", None)),
            },
            "pial_surfaces": {
                "available": bool(getattr(state, "pial_surfaces_available", False)),
                "assume_lps": bool(getattr(state, "pial_surfaces_assume_lps", True)),
            },
            "siscom": {
                "path": _as_str_or_none(getattr(state, "siscom_path", None)),
                "coreg_in_t1_path": _as_str_or_none(getattr(state, "siscom_coreg_path", None)),
                "validated": bool(getattr(state, "siscom_validated", False)),
            },
        },
        "electrodes": list(getattr(state, "electrodes", []) or []),
        "markers": list(getattr(state, "markers", []) or []),
        "view3d": {
            "saved_camera": getattr(state, "view3d_saved_camera", None),
        },
        "oblique_slices": {
            "saved_views": getattr(state, "oblique_slice_saved_views", {}) or {},
        },
    }


def save_project_json(state, project_path: str | Path) -> Path:
    project_path = Path(project_path)
    project_path.parent.mkdir(parents=True, exist_ok=True)

    data = build_project_dict_from_state(state)
    project_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return project_path


def create_empty_project_file(project_path: str | Path, patient_id: str) -> Path:
    project_path = Path(project_path)
    project_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "last_saved": datetime.now().isoformat(timespec="seconds"),
        "patient_id": str(patient_id).strip(),
        "mri_labels": {
            "mri1_filename_label": "MRI1",
            "mri2_filename_label": "MRI2",
        },
        "files": {
            "t1": {
                "path": None,
                "source_path": None,
                "was_conformed": False,
                "conformed_path": None,
                "original_spacing": None,
                "conformed_spacing": None,
            },
            "t2": {
                "path": None,
                "source_path": None,
                "coreg_in_t1_path": None,
                "validated": False,
            },
            "ct": {
                "path": None,
                "source_path": None,
                "coreg_in_t1_path": None,
                "validated": False,
            },
            "pet": {
                "path": None,
                "source_path": None,
                "coreg_in_t1_path": None,
                "validated": False,
            },
            "ictal_spect": {
                "path": None,
                "source_path": None,
                "coreg_in_t1_path": None,
                "validated": False,
            },
            "interictal_spect": {
                "path": None,
                "source_path": None,
                "coreg_in_t1_path": None,
                "validated": False,
            },
            "brainmask": {
                "path": None,
                "generated": False,
                "saved": False,
                "generated_path": None,
            },
            "mni": {
                "space_name": "MNI152NLin2009cAsym",
                "template_path": None,
                "t1_to_mni_affine_path": None,
                "t1_to_mni_warp_path": None,
                "t1_to_mni_inverse_warp_path": None,
                "t1_to_mni_warped_path": None,
            },
            "parcel1": {"path": None},
            "parcel2": {"path": None},
            "lh_pial": {"path": None},
            "rh_pial": {"path": None},
            "pial_surfaces": {
                "available": False,
                "assume_lps": True,
            },
            "siscom": {"path": None, "coreg_in_t1_path": None, "validated": False},
        },
        "electrodes": [],
        "markers": [],
        "view3d": {
            "saved_camera": None,
        },
        "oblique_slices": {
            "saved_views": {},
        },
    }
    project_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return project_path


def load_project_json(project_path: str | Path) -> dict[str, Any]:
    project_path = Path(project_path)
    data = json.loads(project_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid project file")
    return data


def apply_project_dict_to_state(state, data: dict[str, Any], project_path: str | Path) -> None:
    state.project_path = str(project_path)
    state.patient_id = str(data.get("patient_id", "") or "").strip()
    mri_labels = data.get("mri_labels", {}) if isinstance(data.get("mri_labels"), dict) else {}

    state.mri1_filename_label = (
        _as_str_or_none(mri_labels.get("mri1_filename_label"))
        or _as_str_or_none(data.get("mri1_filename_label"))
        or "MRI1"
    )

    state.mri2_filename_label = (
        _as_str_or_none(mri_labels.get("mri2_filename_label"))
        or _as_str_or_none(data.get("mri2_filename_label"))
        or "MRI2"
    )

    files = data.get("files", {}) if isinstance(data.get("files"), dict) else {}

    t1 = files.get("t1", {}) if isinstance(files.get("t1"), dict) else {}
    t2 = files.get("t2", {}) if isinstance(files.get("t2"), dict) else {}
    ct = files.get("ct", {}) if isinstance(files.get("ct"), dict) else {}
    pet = files.get("pet", {}) if isinstance(files.get("pet"), dict) else {}
    ictal = files.get("ictal_spect", {}) if isinstance(files.get("ictal_spect"), dict) else {}
    interictal = (
        files.get("interictal_spect", {}) if isinstance(files.get("interictal_spect"), dict) else {}
    )
    siscom = files.get("siscom", {}) if isinstance(files.get("siscom"), dict) else {}
    lh_pial = files.get("lh_pial", {}) if isinstance(files.get("lh_pial"), dict) else {}
    rh_pial = files.get("rh_pial", {}) if isinstance(files.get("rh_pial"), dict) else {}
    pial_surfaces = (
        files.get("pial_surfaces", {}) if isinstance(files.get("pial_surfaces"), dict) else {}
    )
    brainmask = files.get("brainmask", {}) if isinstance(files.get("brainmask"), dict) else {}
    mni = files.get("mni", {}) if isinstance(files.get("mni"), dict) else {}
    parcel1 = files.get("parcel1", {}) if isinstance(files.get("parcel1"), dict) else {}
    parcel2 = files.get("parcel2", {}) if isinstance(files.get("parcel2"), dict) else {}

    state.t1_path = _as_str_or_none(t1.get("path"))
    state.t1_source_path = _as_str_or_none(t1.get("source_path"))

    state.t1_was_conformed = bool(t1.get("was_conformed", False))
    state.t1_conformed_path = _as_str_or_none(t1.get("conformed_path"))

    orig_spacing = t1.get("original_spacing", None)
    conf_spacing = t1.get("conformed_spacing", None)

    state.t1_original_spacing = orig_spacing if isinstance(orig_spacing, list) else None
    state.t1_conformed_spacing = conf_spacing if isinstance(conf_spacing, list) else None
    state.t2_path = _as_str_or_none(t2.get("path"))
    state.t2_source_path = _as_str_or_none(t2.get("source_path"))

    state.ct_path = _as_str_or_none(ct.get("path"))
    state.ct_source_path = _as_str_or_none(ct.get("source_path"))

    state.pet_path = _as_str_or_none(pet.get("path"))
    state.pet_source_path = _as_str_or_none(pet.get("source_path"))

    state.ictal_spect_path = _as_str_or_none(ictal.get("path"))
    state.ictal_spect_source_path = _as_str_or_none(ictal.get("source_path"))

    state.interictal_spect_path = _as_str_or_none(interictal.get("path"))
    state.interictal_spect_source_path = _as_str_or_none(interictal.get("source_path"))
    state.siscom_path = _as_str_or_none(siscom.get("path"))
    state.brainmask_path = _as_str_or_none(brainmask.get("path"))
    state.brainmask_generated = bool(brainmask.get("generated", False))
    state.brainmask_saved = bool(brainmask.get("saved", bool(state.brainmask_path)))
    state.brainmask_generated_path = _as_str_or_none(brainmask.get("generated_path"))
    state.mni_space_name = str(mni.get("space_name", "") or "MNI152NLin2009cAsym")
    state.mni_template_path = _as_str_or_none(mni.get("template_path"))
    state.t1_to_mni_affine_path = _as_str_or_none(mni.get("t1_to_mni_affine_path"))
    state.t1_to_mni_warp_path = _as_str_or_none(mni.get("t1_to_mni_warp_path"))
    state.t1_to_mni_inverse_warp_path = _as_str_or_none(mni.get("t1_to_mni_inverse_warp_path"))
    state.t1_to_mni_warped_path = _as_str_or_none(mni.get("t1_to_mni_warped_path"))
    state.parcel1_path = _as_str_or_none(parcel1.get("path"))
    state.parcel2_path = _as_str_or_none(parcel2.get("path"))
    state.lh_pial_path = _as_str_or_none(lh_pial.get("path"))
    state.rh_pial_path = _as_str_or_none(rh_pial.get("path"))
    state.pial_surfaces_available = bool(
        pial_surfaces.get(
            "available",
            bool(state.lh_pial_path and state.rh_pial_path),
        )
    )

    state.pial_surfaces_assume_lps = bool(pial_surfaces.get("assume_lps", True))

    state.t2_coreg_path = _as_str_or_none(t2.get("coreg_in_t1_path"))
    state.ct_coreg_path = _as_str_or_none(ct.get("coreg_in_t1_path"))
    state.pet_coreg_path = _as_str_or_none(pet.get("coreg_in_t1_path"))
    state.ictal_spect_coreg_path = _as_str_or_none(ictal.get("coreg_in_t1_path"))
    state.interictal_spect_coreg_path = _as_str_or_none(interictal.get("coreg_in_t1_path"))
    state.siscom_coreg_path = _as_str_or_none(siscom.get("coreg_in_t1_path"))

    state.t2_validated = bool(t2.get("validated", False))

    # Persistent CT validation is restored for visualization pages.
    # Reconstruction uses a separate session-only safety flag.
    state.ct_validated = bool(ct.get("validated", False))
    state.ct_ready_for_reconstruction = False

    state.pet_validated = bool(pet.get("validated", False))
    state.ictal_spect_validated = bool(ictal.get("validated", False))
    state.interictal_spect_validated = bool(interictal.get("validated", False))
    state.siscom_validated = bool(siscom.get("validated", False))

    view3d = data.get("view3d", {}) if isinstance(data.get("view3d"), dict) else {}
    saved_camera = view3d.get("saved_camera", None)
    state.view3d_saved_camera = saved_camera if isinstance(saved_camera, dict) else None

    oblique_slices = (
        data.get("oblique_slices", {}) if isinstance(data.get("oblique_slices"), dict) else {}
    )
    saved_views = oblique_slices.get("saved_views", {})
    state.oblique_slice_saved_views = saved_views if isinstance(saved_views, dict) else {}

    state.electrodes = _safe_list(data.get("electrodes"))
    state.markers = _safe_list(data.get("markers"))


def get_unsaved_validated_modalities(state) -> list[str]:
    missing = []

    checks = [
        ("CT", getattr(state, "ct_validated", False), getattr(state, "ct_coreg_path", None)),
        ("T2", getattr(state, "t2_validated", False), getattr(state, "t2_coreg_path", None)),
        ("PET", getattr(state, "pet_validated", False), getattr(state, "pet_coreg_path", None)),
        (
            "ictal SPECT",
            getattr(state, "ictal_spect_validated", False),
            getattr(state, "ictal_spect_coreg_path", None),
        ),
        (
            "interictal SPECT",
            getattr(state, "interictal_spect_validated", False),
            getattr(state, "interictal_spect_coreg_path", None),
        ),
        (
            "SISCOM",
            getattr(state, "siscom_validated", False),
            getattr(state, "siscom_coreg_path", None),
        ),
    ]

    for label, validated, saved_path in checks:
        if validated and not saved_path:
            missing.append(label)
    if bool(getattr(state, "brainmask_generated", False)) and not bool(
        getattr(state, "brainmask_saved", False)
    ):
        missing.append("Brain Mask")
    try:
        t1_was_conformed = bool(getattr(state, "t1_was_conformed", False))
        t1_conformed_path = getattr(state, "t1_conformed_path", None)

        if t1_was_conformed and t1_conformed_path and not Path(str(t1_conformed_path)).exists():
            missing.append("T1 1 mm isotropic copy")
    except Exception:
        logger.warning("Could not check conformed T1 copy state", exc_info=True)
    return missing
