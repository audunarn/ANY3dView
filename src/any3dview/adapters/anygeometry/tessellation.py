"""Display tessellation using public kernel evaluation, with sampled error bounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .policy import TessellationPolicy


class UnsupportedDisplayGeometry(ValueError):
    """An entity cannot be displayed within the requested policy."""


def _triangulator():
    try:
        import mapbox_earcut
    except ImportError as error:
        raise UnsupportedDisplayGeometry(
            'Geometry display requires mapbox-earcut; install "ANY3dView[geometry]".'
        ) from error
    return mapbox_earcut


def _limits(points, policy, lod):
    if not 0 <= lod < policy.lod_levels:
        raise ValueError("lod is outside configured tessellation levels")
    factor = 2.0 ** (policy.lod_levels - 1 - lod)
    scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    return (max(policy.chord_tolerance, policy.relative_chord_tolerance * scale) * factor,
            policy.angular_tolerance * factor)


def _edge_parameters(model, edge_id, policy, lod):
    # Quarter probes catch inflections that a midpoint alone can miss.
    parameters = np.linspace(0., 1., min(4, policy.max_curve_segments) + 1)
    initial = np.asarray(model.evaluate_edge_many(edge_id, parameters), dtype=float)
    tolerance, angle = _limits(initial, policy, lod)
    while True:
        probes = parameters[:-1, None] + np.diff(parameters)[:, None] * np.linspace(0., 1., 5)
        points = np.asarray(model.evaluate_edge_many(edge_id, probes.ravel()), dtype=float).reshape(-1, 5, 3)
        if not np.isfinite(points).all():
            raise UnsupportedDisplayGeometry(f"edge {edge_id} returned non-finite coordinates")
        linear = points[:, :1] + (points[:, -1:] - points[:, :1]) * np.linspace(0., 1., 5)[None, :, None]
        deviation = np.linalg.norm(points - linear, axis=2).max(axis=1)
        tangents = np.diff(points, axis=1)
        lengths = np.linalg.norm(tangents, axis=2)
        unit = tangents / np.maximum(lengths[..., None], np.finfo(float).tiny)
        turns = np.arccos(np.clip(np.sum(unit[:, :-1] * unit[:, 1:], axis=2), -1., 1.))
        turns[(lengths[:, :-1] == 0) | (lengths[:, 1:] == 0)] = 0
        refine = (deviation > tolerance) | (turns.sum(axis=1) > angle)
        if not refine.any():
            return parameters
        if len(parameters) - 1 + int(refine.sum()) > policy.max_curve_segments:
            raise UnsupportedDisplayGeometry(
                f"edge {edge_id} exceeds max_curve_segments before meeting display tolerances"
            )
        parameters = np.sort(np.concatenate((parameters, ((parameters[:-1] + parameters[1:]) / 2)[refine])))


def sampled_edge(model, edge_id: int, policy: TessellationPolicy, lod: int) -> np.ndarray:
    edge = model.edges[int(edge_id)]
    parameters = (np.array([0., 1.]) if type(edge.curve).__name__.casefold() == "straight"
                  else _edge_parameters(model, edge_id, policy, lod))
    return np.asarray(model.evaluate_edge_many(edge.id, parameters), dtype=np.float64)


def oriented_loop(model, oriented_edges: Iterable[tuple[int, bool]], policy, lod) -> np.ndarray:
    blocks = []
    for edge_id, forward in oriented_edges:
        points = sampled_edge(model, edge_id, policy, lod)
        blocks.append((points if forward else points[::-1])[:-1])
    return np.concatenate(blocks) if blocks else np.empty((0, 3), np.float64)


@dataclass(frozen=True, slots=True)
class TessellatedFace:
    positions: np.ndarray
    triangles: np.ndarray


def _triangulate(loops):
    earcut = _triangulator()
    if not loops or any(len(loop) < 3 for loop in loops):
        raise UnsupportedDisplayGeometry("face boundary is incomplete")
    uv = np.ascontiguousarray(np.concatenate(loops), dtype=np.float64)
    if not np.isfinite(uv).all():
        raise UnsupportedDisplayGeometry("face boundary contains non-finite coordinates")
    ends = np.cumsum([len(loop) for loop in loops], dtype=np.uint32)
    local = np.ascontiguousarray(uv - uv[0])
    triangles = np.asarray(earcut.triangulate_float64(local, ends), dtype=np.uint32).reshape(-1, 3)
    def area(p):
        return abs(float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)))) / 2
    shifted = [np.asarray(loop) - uv[0] for loop in loops]
    expected = area(shifted[0]) - sum(area(loop) for loop in shifted[1:])
    p = local[triangles]
    a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    actual = float(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()) / 2
    if expected <= 0 or not np.isclose(actual, expected, rtol=1e-9, atol=np.finfo(float).eps * expected * 32):
        raise UnsupportedDisplayGeometry("triangulation does not preserve the trimmed face area")
    return uv, triangles


def _refine_surface(model, face_id, uv, triangles, policy, lod):
    positions = np.asarray(model.evaluate_face_many(face_id, uv), dtype=float)
    tolerance, angle = _limits(positions, policy, lod)
    while True:
        if len(triangles) > policy.max_surface_triangles:
            raise UnsupportedDisplayGeometry(f"face {face_id} exceeds max_surface_triangles")
        weights = np.array([[.5, .5, 0], [0, .5, .5], [.5, 0, .5], [1/3, 1/3, 1/3]])
        probes = np.einsum('ij,kjl->kil', weights, uv[triangles])
        actual = np.asarray(model.evaluate_face_many(face_id, probes.reshape(-1, 2))).reshape(-1, 4, 3)
        linear = np.einsum('ij,kjl->kil', weights, positions[triangles])
        if not np.isfinite(actual).all():
            raise UnsupportedDisplayGeometry(f"face {face_id} returned non-finite coordinates")
        deviation = np.linalg.norm(actual - linear, axis=2).max()
        normals = np.asarray(model.face_normal_many(face_id, np.concatenate((uv, probes.reshape(-1, 2)))))
        lengths = np.linalg.norm(normals, axis=1)
        if not np.isfinite(normals).all() or np.any(lengths == 0):
            raise UnsupportedDisplayGeometry(f"face {face_id} has undefined normals")
        normals = normals / lengths[:, None]
        cosine = np.einsum('kij,klj->kil', normals[:len(uv)][triangles], normals[len(uv):].reshape(-1, 4, 3))
        turn = np.arccos(np.clip(cosine, -1., 1.)).max()
        if deviation <= tolerance and turn <= angle:
            return TessellatedFace(np.ascontiguousarray(positions), triangles)
        if len(triangles) * 4 > policy.max_surface_triangles:
            raise UnsupportedDisplayGeometry(
                f"face {face_id} exceeds max_surface_triangles before meeting display tolerances"
            )
        # Shared edge midpoints keep refinement conforming, including trim edges.
        edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
        unique, inverse = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
        midpoint = len(uv) + inverse.reshape(3, -1).T
        uv = np.concatenate((uv, uv[unique].mean(axis=1)))
        positions = np.asarray(model.evaluate_face_many(face_id, uv), dtype=float)
        a, b, c = triangles.T
        ab, bc, ca = midpoint.T
        triangles = np.asarray(np.concatenate((
            np.column_stack((a, ab, ca)), np.column_stack((ab, b, bc)),
            np.column_stack((ca, bc, c)), np.column_stack((ab, bc, ca)),
        )), dtype=np.uint32)


def tessellate_face(model, face_id: int, policy: TessellationPolicy, lod: int) -> TessellatedFace:
    _triangulator()
    face = model.faces[int(face_id)]
    try:
        if face.support_surface is not None:
            # Kernel UV trims use uniform samples; match the smallest adaptive step.
            segments = 2
            for loop in (face.loop, *face.holes):
                for item in loop:
                    if type(model.edges[item.edge].curve).__name__.casefold() != "straight":
                        parameters = _edge_parameters(model, item.edge, policy, lod)
                        segments = max(segments, int(np.ceil(1 / np.diff(parameters).min())))
            if segments > policy.max_curve_segments:
                raise UnsupportedDisplayGeometry("trim sampling exceeds max_curve_segments")
            loops = model.face_trim_loops_uv(face_id, curve_samples=segments + 1)
            uv, triangles = _triangulate(loops)
            return _refine_surface(model, face_id, uv, triangles, policy, lod)
        loops = [oriented_loop(model, ((e.edge, bool(e.forward)) for e in loop), policy, lod)
                 for loop in (face.loop, *face.holes)]
        outer = loops[0]
        local = outer - outer[0]
        normal = np.sum(np.cross(local, np.roll(local, -1, axis=0)), axis=0)
        if np.linalg.norm(normal) == 0:
            raise UnsupportedDisplayGeometry("face boundary is degenerate")
        axis = int(np.argmax(np.abs(normal)))
        _, triangles = _triangulate([np.delete(loop, axis, axis=1) for loop in loops])
        return TessellatedFace(np.ascontiguousarray(np.concatenate(loops)), triangles)
    except UnsupportedDisplayGeometry:
        raise
    except Exception as error:
        raise UnsupportedDisplayGeometry(f"face {face_id} display tessellation failed: {error}") from error
