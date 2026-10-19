"""Bounded display detail while the camera moves; never changes model data."""
from dataclasses import replace
import numpy as np

from ..ui.scene import Scene


def _indices(length, limit):
    if length <= limit:
        return range(length)
    return np.linspace(0, length - 1, limit, dtype=int).tolist()


def camera_scene(scene, *, polygon_budget=160, line_budget=350, point_budget=64):
    """Sample draw primitives, retaining their original owners and coordinates.

    Beam sections disappear during movement, leaving their centre lines. Dense
    shell tessellation is sampled rather than inventing replacement topology.
    Selection always uses the restored complete scene, not these samples.
    """
    faces = [face for face in scene.faces if getattr(face.ref, "kind", None) != "edge"]
    faces = [faces[i] for i in _indices(len(faces), polygon_budget)]
    result = Scene(legend=scene.legend)
    remaining = polygon_budget
    for offset, face in enumerate(faces):
        allowance = max(1, remaining // (len(faces) - offset))
        indices = list(_indices(len(face.polygons), allowance))
        remaining -= len(indices)
        colors = face.colors
        result.faces.append(replace(face,
            polygons=[face.polygons[i] for i in indices],
            colors=[colors[i] if len(colors) > 1 else colors[0] for i in indices] if colors else [],
            polygon_owners=None if face.polygon_owners is None else [face.polygon_owners[i] for i in indices],
            polygon_visibility_owners=None if face.polygon_visibility_owners is None else [face.polygon_visibility_owners[i] for i in indices]))
    result.lines = [scene.lines[i] for i in _indices(len(scene.lines), line_budget)]
    result.points = [scene.points[i] for i in _indices(len(scene.points), point_budget)]
    result.arrows = list(scene.arrows)
    result.spheres = list(scene.spheres)
    return result
