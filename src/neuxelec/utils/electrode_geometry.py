"""Pure geometry for SEEG electrode reconstruction.

This module isolates the *scientific core* of the two-point reconstruction so
it can be unit-tested without any GUI, image or ANTs dependency.

Method ("Voxeloc"-style)
------------------------
The user clicks two contacts of the same electrode on the CT - the deepest
contact ``p0`` and a more superficial reference point ``p1``. Working on a
1 mm isotropic CT-in-T1 grid, every contact lies on the straight line through
``p0`` in the direction ``p1 - p0``, separated by the electrode's known
inter-contact spacing.

A single contact is predicted from the previous one by stepping ``step_d``
voxels along that axis. Keeping the step as a *single-step primitive*
(:func:`next_contact_voxel`) lets the caller optionally refine each predicted
position against the CT intensity (local-max snapping) and feed the refined
position back as the basis for the next step - exactly as the interactive
reconstruction does.

All coordinates are voxel coordinates ``(x, y, z)`` on the CT grid.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Vec3 = tuple[float, float, float]


def axis_decomposition(
    p0: Sequence[float], p1: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Decompose the ``p0 -> p1`` axis used to march along the electrode.

    Returns
    -------
    delta:
        Per-axis squared direction fractions ``(dx**2, dy**2, dz**2) / |v|**2``.
    sign:
        Per-axis sign of the direction vector (``+1`` or ``-1``).
    length:
        Euclidean distance ``|p1 - p0|`` (the axis length, in voxels).
    """
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    v = b - a
    length = float(np.linalg.norm(v))
    denom = max(1e-12, length**2)
    delta = (v**2) / denom
    sign = np.where(v >= 0.0, 1.0, -1.0)
    return delta, sign, length


def next_contact_voxel(
    prev: Sequence[float],
    delta: np.ndarray,
    sign: np.ndarray,
    step_d: float,
    bounds: Sequence[int] | None = None,
) -> Vec3:
    """Predict the next contact, one ``step_d`` step along the axis.

    Parameters
    ----------
    prev:
        Previous contact voxel position ``(x, y, z)``.
    delta, sign:
        Axis decomposition from :func:`axis_decomposition`.
    step_d:
        Inter-contact distance to this contact (voxels / mm on a 1 mm grid).
    bounds:
        Optional ``(size_x, size_y, size_z)``; the result is clipped to
        ``[0, size-1]`` per axis when provided.
    """
    p = np.asarray(prev, dtype=np.float64)
    step = sign * np.sqrt(np.maximum(0.0, delta * (float(step_d) ** 2)))
    pred = p + step
    if bounds is not None:
        sx, sy, sz = bounds
        pred[0] = np.clip(pred[0], 0, sx - 1)
        pred[1] = np.clip(pred[1], 0, sy - 1)
        pred[2] = np.clip(pred[2], 0, sz - 1)
    return (float(pred[0]), float(pred[1]), float(pred[2]))


def predict_contact_voxels(
    p0: Sequence[float],
    p1: Sequence[float],
    spacings: Sequence[float],
    bounds: Sequence[int] | None = None,
) -> list[Vec3]:
    """Predict all contact voxel positions along the ``p0 -> p1`` axis.

    The first contact is ``p0``; each subsequent contact is one entry of
    ``spacings`` further along the axis. ``len(result) == len(spacings) + 1``.

    This is the baseline (no local-max refinement) reconstruction and is the
    deterministic core covered by the unit tests.
    """
    delta, sign, length = axis_decomposition(p0, p1)
    contacts: list[Vec3] = [(float(p0[0]), float(p0[1]), float(p0[2]))]
    if length < 1e-6:
        # Degenerate: the two points coincide; cannot define an axis.
        return contacts
    for step_d in spacings:
        contacts.append(next_contact_voxel(contacts[-1], delta, sign, step_d, bounds))
    return contacts
