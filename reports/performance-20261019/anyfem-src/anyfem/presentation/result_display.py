"""Canonical frontend-neutral import surface for result display units."""

from ..ui.result_display import *


def deformation_scale(text, project, shape=None):
    """Existing visual scale policy; stored SI displacements remain unchanged."""
    import math
    value=str(text).strip().lower()
    if value in {"","auto"}:
        if shape is None:return 1.0
        try:
            _node,magnitude=shape.max_translation()
        except KeyError:return 0.0
        if magnitude<=0:return 1.0
        from .scene import build_mesh_scene
        submitted=getattr(shape.built,"project",None)
        span=build_mesh_scene(project if submitted is None else submitted,shape.built.mesh).characteristic_size()
        return 0.08*span/magnitude
    scale=float(value)
    if not math.isfinite(scale):raise ValueError("deformation scale must be finite")
    return scale
