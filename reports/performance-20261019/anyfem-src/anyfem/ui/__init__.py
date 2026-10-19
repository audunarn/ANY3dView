"""The ANYfem application: 3D viewport, model tree, stage panels.

This is the only part of ANYfem that needs Tk and ANYtk3D.  Importing the
scene builder alone stays display-free, so what gets drawn can be tested
without a window.
"""

from .scene import (
    FacePatch,
    PointMarker,
    Polyline,
    Scene,
    build_geometry_scene,
    build_mesh_scene,
    build_result_scene,
    face_display_polygons,
)

__all__ = [
    "FacePatch",
    "PointMarker",
    "Polyline",
    "Scene",
    "ScriptingPanel",
    "build_geometry_scene",
    "build_mesh_scene",
    "build_result_scene",
    "face_display_polygons",
    "main",
]


def __getattr__(name: str):
    """Load the Tk-dependent parts only when they are actually asked for."""

    if name in ("AnyFemApp", "main", "default_project"):
        from . import tk

        return getattr(tk, name)
    if name == "Viewport":
        from .viewport import Viewport

        return Viewport
    if name in ("ModelTree",):
        from .tree import ModelTree

        return ModelTree
    if name == "ScriptingPanel":
        from .scripting import ScriptingPanel

        return ScriptingPanel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
