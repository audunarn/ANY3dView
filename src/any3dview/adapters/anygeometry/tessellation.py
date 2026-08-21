"""Display-only tessellation using public ANYgeometry evaluation methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .policy import TessellationPolicy


class UnsupportedDisplayGeometry(ValueError):
    """One entity cannot be represented without silently changing its shape."""


def sampled_edge(model, edge_id: int, policy: TessellationPolicy, lod: int) -> np.ndarray:
    edge = model.edges[int(edge_id)]
    curve_name = type(edge.curve).__name__.casefold()
    if "straight" in curve_name:
        segments = 1
    else:
        target = (8, 24, 64)[min(2, int(lod))]
        segments = min(policy.max_curve_segments, target)
    parameters = np.linspace(0.0, 1.0, segments + 1, dtype=np.float64)
    return np.asarray(model.evaluate_edge_many(edge.id, parameters), dtype=np.float64)


def oriented_loop(model, oriented_edges: Iterable[tuple[int, bool]], policy, lod) -> np.ndarray:
    blocks = []
    for edge_id, forward in oriented_edges:
        points = sampled_edge(model, edge_id, policy, lod)
        if not forward:
            points = points[::-1]
        blocks.append(points[:-1])
    if not blocks:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(blocks)


def _project(points: np.ndarray) -> tuple[np.ndarray, int]:
    shifted = np.roll(points, -1, axis=0)
    normal = np.sum(np.cross(points, shifted), axis=0)
    if np.linalg.norm(normal) <= 1.0e-14:
        raise UnsupportedDisplayGeometry("face boundary is degenerate")
    dropped = int(np.argmax(np.abs(normal)))
    return np.delete(points, dropped, axis=1), dropped


@dataclass(frozen=True, slots=True)
class TessellatedFace:
    positions: np.ndarray
    triangles: np.ndarray


def tessellate_face(model, face_id: int, policy: TessellationPolicy, lod: int) -> TessellatedFace:
    face = model.faces[int(face_id)]
    support = face.support_surface
    if support is not None and type(support).__name__.casefold() != "plane":
        samples = min(policy.max_curve_segments + 1, (9, 25, 65)[min(2, int(lod))])
        try:
            uv_loops = model.face_trim_loops_uv(face_id, curve_samples=samples)
            flat_uv = np.ascontiguousarray(np.concatenate(uv_loops), dtype=np.float64)
            ring_ends = np.cumsum([len(loop) for loop in uv_loops], dtype=np.uint32)
            import mapbox_earcut

            indices = np.asarray(
                mapbox_earcut.triangulate_float64(flat_uv, ring_ends),
                dtype=np.uint32,
            ).reshape((-1, 3))
            positions = np.asarray(model.evaluate_face_many(face_id, flat_uv), dtype=np.float64)
            return TessellatedFace(positions, indices)
        except Exception as error:
            raise UnsupportedDisplayGeometry(
                f"face {face_id} curved display tessellation failed: {error}"
            ) from error
    loops = [face.loop, *face.holes]
    points_3d: list[np.ndarray] = []
    points_2d: list[np.ndarray] = []
    ring_ends: list[int] = []
    count = 0
    dropped_axis = None
    for loop in loops:
        points = oriented_loop(
            model,
            ((item.edge, bool(item.forward)) for item in loop),
            policy,
            lod,
        )
        if len(points) < 3:
            raise UnsupportedDisplayGeometry(f"face {face_id} has an incomplete loop")
        projected, dropped = _project(points)
        if dropped_axis is None:
            dropped_axis = dropped
        elif dropped != dropped_axis:
            projected = np.delete(points, dropped_axis, axis=1)
        points_3d.append(points)
        points_2d.append(projected)
        count += len(points)
        ring_ends.append(count)
    positions = np.ascontiguousarray(np.concatenate(points_3d), dtype=np.float64)
    flat = np.ascontiguousarray(np.concatenate(points_2d), dtype=np.float64)
    try:
        import mapbox_earcut
    except ImportError as error:
        if len(loops) > 1:
            raise UnsupportedDisplayGeometry(
                "faces with holes require the ANY3dView geometry extra"
            ) from error
        # Deterministic fan for simple convex faces; concave production use is
        # supplied by mapbox-earcut in the geometry extra.
        indices = np.asarray(
            [(0, index, index + 1) for index in range(1, len(positions) - 1)],
            dtype=np.uint32,
        ).reshape((-1, 3))
    else:
        indices = np.asarray(
            mapbox_earcut.triangulate_float64(
                flat, np.asarray(ring_ends, dtype=np.uint32)
            ),
            dtype=np.uint32,
        ).reshape((-1, 3))
    return TessellatedFace(positions, indices)
