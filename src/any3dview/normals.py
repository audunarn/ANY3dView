"""Crease-angle limited smooth vertex normals for faceted display meshes.

Display tessellations of curved surfaces (cylinders, cones, fillets) are drawn
flat, one normal per triangle, which makes the facets visible.  Finite element
meshes must stay flat: each element should be distinguishable.  Smoothing is
therefore an explicit, per-mesh choice with a crease angle, and sharp edges
(box corners, plate junctions) keep their hard shading.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _corner_angles(corners: np.ndarray) -> np.ndarray:
    """Interior angle at each corner of every triangle, shape (T, 3)."""

    angles = np.empty(corners.shape[:2], dtype=np.float64)
    for slot in range(3):
        first = corners[:, (slot + 1) % 3] - corners[:, slot]
        second = corners[:, (slot + 2) % 3] - corners[:, slot]
        lengths = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        cosine = np.einsum("ij,ij->i", first, second) / np.maximum(lengths, 1.0e-30)
        angles[:, slot] = np.arccos(np.clip(cosine, -1.0, 1.0))
    return angles


def smooth_corner_normals(
    positions: np.ndarray,
    triangles: np.ndarray,
    crease_degrees: float,
    *,
    weld_tolerance: Optional[float] = None,
) -> np.ndarray:
    """Return one unit normal per triangle corner, shape ``(T, 3, 3)``.

    Corners at the same location (within ``weld_tolerance``; default a
    millionth of the bounding diagonal) are averaged, weighted by the interior
    angle, when their faces differ by at most ``crease_degrees``.  Windings may
    be inconsistent between neighbours: a neighbour whose normal points the
    other way is flipped before averaging, so the result follows each face's
    own orientation.  Degenerate triangles get a zero normal.
    """

    positions = np.asarray(positions, dtype=np.float64).reshape((-1, 3))
    triangles = np.asarray(triangles, dtype=np.int64).reshape((-1, 3))
    count = len(triangles)
    if count == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)
    corners = positions[triangles]
    raw = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    norm = np.linalg.norm(raw, axis=1)
    valid = norm > 1.0e-30
    face = np.zeros_like(raw)
    face[valid] = raw[valid] / norm[valid, None]

    extent = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    tolerance = weld_tolerance if weld_tolerance is not None else 1.0e-6 * max(extent, 1.0e-12)
    quantised = np.round(positions / tolerance).astype(np.int64)
    _, weld = np.unique(quantised, axis=0, return_inverse=True)
    weld = np.asarray(weld).reshape(-1)

    corner_weld = weld[triangles].reshape(-1)            # (3T,)
    corner_face = np.repeat(np.arange(count), 3)         # (3T,)
    weight = (_corner_angles(corners) * valid[:, None]).reshape(-1)

    order = np.argsort(corner_weld, kind="stable")
    sorted_weld = corner_weld[order]
    starts = np.flatnonzero(np.r_[True, sorted_weld[1:] != sorted_weld[:-1]])
    sizes = np.diff(np.r_[starts, len(order)])
    group_of = np.repeat(np.arange(len(starts)), sizes)  # per sorted position
    per_corner = sizes[group_of]                         # group size per sorted position
    total = int(per_corner.sum())
    first = np.repeat(np.arange(len(order)), per_corner)           # sorted position i
    offset = np.arange(total) - np.repeat(np.cumsum(per_corner) - per_corner, per_corner)
    second = np.repeat(starts[group_of], per_corner) + offset      # sorted position j

    i_corner, j_corner = order[first], order[second]
    face_i, face_j = corner_face[i_corner], corner_face[j_corner]
    alignment = np.einsum("ij,ij->i", face[face_i], face[face_j])
    keep = np.abs(alignment) >= np.cos(np.radians(max(0.0, float(crease_degrees))))
    keep &= valid[face_i] & valid[face_j]
    signed = np.where(alignment < 0.0, -1.0, 1.0) * weight[j_corner] * keep
    contribution = face[face_j] * signed[:, None]

    accumulated = np.stack(
        [np.bincount(i_corner, weights=contribution[:, axis], minlength=3 * count)
         for axis in range(3)],
        axis=1,
    )
    length = np.linalg.norm(accumulated, axis=1)
    own = face[corner_face]
    smooth = np.where(length[:, None] > 1.0e-12, accumulated / np.maximum(length, 1.0e-30)[:, None], own)
    # Keep every corner on its own side of the surface.
    flipped = np.einsum("ij,ij->i", smooth, own) < 0.0
    smooth[flipped] *= -1.0
    return smooth.reshape(count, 3, 3).astype(np.float32)


def expand_with_smooth_normals(
    positions: np.ndarray,
    triangles: np.ndarray,
    crease_degrees: float,
    lines: Optional[np.ndarray] = None,
    *,
    weld_tolerance: Optional[float] = None,
):
    """Return ``(positions, triangles, normals, lines)`` with one vertex per corner.

    Corner vertices let neighbouring triangles carry different normals where
    the crease angle separates them.  ``lines`` indices are remapped to the
    first corner that uses each original vertex; vertices used only by lines
    are appended unchanged.
    """

    positions = np.asarray(positions, dtype=np.float64).reshape((-1, 3))
    triangles = np.asarray(triangles, dtype=np.int64).reshape((-1, 3))
    normals = smooth_corner_normals(
        positions, triangles, crease_degrees, weld_tolerance=weld_tolerance
    ).reshape(-1, 3)
    flat = triangles.reshape(-1)
    expanded = positions[flat]
    new_triangles = np.arange(len(flat), dtype=np.uint32).reshape(-1, 3)
    remapped = None
    if lines is not None and len(lines):
        first_corner = np.full(len(positions), -1, dtype=np.int64)
        first_corner[flat[::-1]] = np.arange(len(flat))[::-1]
        unused = np.flatnonzero(first_corner < 0)
        first_corner[unused] = len(flat) + np.arange(len(unused))
        expanded = np.vstack((expanded, positions[unused]))
        normals = np.vstack((normals, np.zeros((len(unused), 3), dtype=np.float32)))
        remapped = first_corner[np.asarray(lines, dtype=np.int64)].astype(np.uint32)
    return expanded, new_triangles, normals.astype(np.float32), remapped
