"""Tests for project persistence (save / load round-trip).

These cover the scientifically important guarantee that a project written to
disk can be read back without losing or corrupting patient metadata and
electrode definitions. They use no GUI and no imaging data, so they run fast
and deterministically in CI.
"""

from __future__ import annotations

import json

import pytest

from neuxelec import project_io
from neuxelec.state import AppState


def test_helpers_as_str_or_none():
    assert project_io._as_str_or_none(None) is None
    assert project_io._as_str_or_none("") is None
    assert project_io._as_str_or_none("abc") == "abc"
    assert project_io._as_str_or_none(42) == "42"


def test_helpers_safe_list():
    assert project_io._safe_list([1, 2]) == [1, 2]
    assert project_io._safe_list(None) == []
    assert project_io._safe_list("not a list") == []


def test_create_empty_project_file_is_valid_json(tmp_path):
    project_file = tmp_path / "patient" / "project.json"
    returned = project_io.create_empty_project_file(project_file, "SUBJ-001")

    assert returned == project_file
    assert project_file.exists()

    data = json.loads(project_file.read_text(encoding="utf-8"))
    assert data["schema_version"] == project_io.PROJECT_SCHEMA_VERSION
    assert data["patient_id"] == "SUBJ-001"
    # Core modality slots are present in a fresh project.
    assert "t1" in data["files"]
    assert "ct" in data["files"]


def test_load_project_json_reads_back(tmp_path):
    project_file = tmp_path / "project.json"
    project_io.create_empty_project_file(project_file, "SUBJ-002")

    data = project_io.load_project_json(project_file)
    assert isinstance(data, dict)
    assert data["patient_id"] == "SUBJ-002"
    assert data["schema_version"] == project_io.PROJECT_SCHEMA_VERSION


def test_apply_to_state_sets_patient_id(tmp_path):
    project_file = tmp_path / "project.json"
    project_io.create_empty_project_file(project_file, "SUBJ-003")
    data = project_io.load_project_json(project_file)

    state = AppState()
    project_io.apply_project_dict_to_state(state, data, project_file)

    assert state.patient_id == "SUBJ-003"
    assert state.project_path == str(project_file)


def test_full_round_trip_preserves_patient_id(tmp_path):
    """create -> load -> apply -> build -> the patient id survives."""
    project_file = tmp_path / "project.json"
    project_io.create_empty_project_file(project_file, "SUBJ-004")

    state = AppState()
    project_io.apply_project_dict_to_state(
        state, project_io.load_project_json(project_file), project_file
    )

    rebuilt = project_io.build_project_dict_from_state(state)
    assert rebuilt["schema_version"] == project_io.PROJECT_SCHEMA_VERSION
    assert rebuilt["patient_id"] == "SUBJ-004"


def test_save_then_reload_after_save_project_json(tmp_path):
    """save_project_json should produce a file that load_project_json accepts."""
    project_file = tmp_path / "project.json"
    project_io.create_empty_project_file(project_file, "SUBJ-005")

    state = AppState()
    project_io.apply_project_dict_to_state(
        state, project_io.load_project_json(project_file), project_file
    )

    saved_path = project_io.save_project_json(state, project_file)
    reloaded = project_io.load_project_json(saved_path)
    assert reloaded["patient_id"] == "SUBJ-005"
    assert reloaded["schema_version"] == project_io.PROJECT_SCHEMA_VERSION
