"""Tests for the update-checker pure logic (no network, no GUI)."""

from __future__ import annotations

import pytest

from neuxelec.updater import UpdateInfo, is_newer, parse_manifest


def test_is_newer_basic():
    assert is_newer("1.1.0", "1.0.0")
    assert is_newer("2.0.0", "1.9.9")
    assert is_newer("1.0.1", "1.0.0")


def test_is_newer_not_when_equal_or_older():
    assert not is_newer("1.0.0", "1.0.0")
    assert not is_newer("1.0.0", "1.1.0")
    assert not is_newer("0.9.0", "1.0.0")


def test_is_newer_tolerates_prerelease_suffix():
    # "1.2.0-beta" -> (1, 2, 0)
    assert not is_newer("1.2.0-beta", "1.2.0")
    assert is_newer("1.2.1-beta", "1.2.0")


def test_is_newer_handles_different_lengths():
    assert is_newer("1.0.1", "1.0")
    assert not is_newer("1.0", "1.0.1")


def test_parse_manifest_full():
    info = parse_manifest(
        {
            "version": "1.1.0",
            "url": "https://neuxelec.com/downloads/NeuXelec_Setup_1.1.0.exe",
            "notes": "Bug fixes.",
            "sha256": "ABCDEF",
            "mandatory": True,
        }
    )
    assert isinstance(info, UpdateInfo)
    assert info.version == "1.1.0"
    assert info.url.endswith("NeuXelec_Setup_1.1.0.exe")
    assert info.notes == "Bug fixes."
    assert info.sha256 == "abcdef"  # normalized to lowercase
    assert info.mandatory is True


def test_parse_manifest_minimal_defaults():
    info = parse_manifest({"version": "1.0.0", "url": "https://x/y.exe"})
    assert info.notes == ""
    assert info.sha256 is None
    assert info.mandatory is False


def test_parse_manifest_rejects_missing_fields():
    with pytest.raises(KeyError):
        parse_manifest({"version": "1.0.0"})  # no url
    with pytest.raises(ValueError):
        parse_manifest({"version": "", "url": ""})
