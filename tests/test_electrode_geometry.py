"""Tests for the pure SEEG electrode reconstruction geometry.

These lock the scientific core of the two-point reconstruction: contacts must
lie on the axis through the two clicked points, evenly spaced by the
inter-contact distance. A regression here would silently shift electrode
coordinates, so these tests double as a numerical safety net across versions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from neuxelec.utils.electrode_geometry import (
    axis_decomposition,
    next_contact_voxel,
    predict_contact_voxels,
)


def test_axis_decomposition_along_x():
    delta, sign, length = axis_decomposition((0, 0, 0), (10, 0, 0))
    assert length == pytest.approx(10.0)
    assert delta[0] == pytest.approx(1.0)
    assert delta[1] == pytest.approx(0.0)
    assert delta[2] == pytest.approx(0.0)
    assert tuple(sign) == (1.0, 1.0, 1.0)


def test_axis_decomposition_negative_direction():
    _, sign, _ = axis_decomposition((10, 10, 10), (0, 10, 10))
    assert sign[0] == -1.0


def test_contacts_evenly_spaced_along_axis():
    p0 = (0.0, 0.0, 0.0)
    p1 = (0.0, 0.0, 10.0)
    spacings = [3.5] * 4  # 5 contacts, 3.5 mm apart
    contacts = predict_contact_voxels(p0, p1, spacings)

    assert len(contacts) == 5
    # All on the z axis.
    for c in contacts:
        assert c[0] == pytest.approx(0.0)
        assert c[1] == pytest.approx(0.0)
    # Even spacing of 3.5 mm.
    zs = [c[2] for c in contacts]
    assert zs == pytest.approx([0.0, 3.5, 7.0, 10.5, 14.0])


def test_contacts_on_diagonal_axis_preserve_spacing():
    p0 = (0.0, 0.0, 0.0)
    p1 = (1.0, 1.0, 1.0)
    spacings = [math.sqrt(3.0)] * 2  # step of sqrt(3) -> +1 on each axis
    contacts = predict_contact_voxels(p0, p1, spacings)

    assert contacts[1][0] == pytest.approx(1.0)
    assert contacts[1][1] == pytest.approx(1.0)
    assert contacts[1][2] == pytest.approx(1.0)
    # Euclidean distance between consecutive contacts equals the spacing.
    d = math.dist(contacts[0], contacts[1])
    assert d == pytest.approx(math.sqrt(3.0))


def test_variable_spacing_profile():
    p0 = (0.0, 0.0, 0.0)
    p1 = (0.0, 0.0, 100.0)
    spacings = [2.0, 5.0, 1.5]
    contacts = predict_contact_voxels(p0, p1, spacings)
    zs = [c[2] for c in contacts]
    assert zs == pytest.approx([0.0, 2.0, 7.0, 8.5])


def test_bounds_clipping():
    p0 = (0.0, 0.0, 0.0)
    p1 = (0.0, 0.0, 10.0)
    spacings = [100.0]  # would overshoot
    contacts = predict_contact_voxels(p0, p1, spacings, bounds=(5, 5, 5))
    assert contacts[1][2] == pytest.approx(4.0)  # clipped to size-1


def test_degenerate_coincident_points():
    contacts = predict_contact_voxels((1, 1, 1), (1, 1, 1), [2.0, 2.0])
    assert contacts == [(1.0, 1.0, 1.0)]


def test_next_contact_matches_legacy_per_axis_formula():
    """next_contact_voxel must reproduce the original per-axis sqrt formula."""
    p0 = np.array([3.0, 7.0, 2.0])
    p1 = np.array([9.0, 1.0, 14.0])
    step_d = 4.0

    delta, sign, length = axis_decomposition(p0, p1)
    got = next_contact_voxel(p0, delta, sign, step_d)

    # Legacy formulation, copied verbatim from the original controller.
    v = p1 - p0
    denom = max(1e-12, float(np.linalg.norm(v)) ** 2)
    dx, dy, dz = float(v[0]), float(v[1]), float(v[2])
    delta_x, delta_y, delta_z = dx**2 / denom, dy**2 / denom, dz**2 / denom
    sgn = lambda a: 1.0 if a >= 0 else -1.0
    expected = (
        p0[0] + sgn(dx) * math.sqrt(max(0.0, delta_x * step_d**2)),
        p0[1] + sgn(dy) * math.sqrt(max(0.0, delta_y * step_d**2)),
        p0[2] + sgn(dz) * math.sqrt(max(0.0, delta_z * step_d**2)),
    )
    assert got == pytest.approx(expected)
