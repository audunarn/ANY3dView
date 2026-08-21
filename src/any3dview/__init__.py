"""Backend-neutral geometry, camera, shading, clipping and selection core."""

from . import clipping, core, selection, shading, shapes
from .clipping import SectionPlane
from .core import (
    Camera3D,
    DEFAULT_COLOR_STOPS,
    Point3D,
    get_color_stops,
    reset_color_stops,
    set_color_stops,
)
from .selection import (
    PickBinding,
    PickOwner,
    ProjectedPrimitive,
    ProjectedSelectionIndex,
    SelectionConfig,
    SelectionDepth,
    SelectionEvent,
    SelectionFilter,
    SelectionGesture,
    SelectionHit,
    SelectionOperation,
    SelectionTool,
)
from .shading import Light
from .shapes import Mesh

__version__ = "0.1.0"

__all__ = [
    "Camera3D",
    "DEFAULT_COLOR_STOPS",
    "Light",
    "Mesh",
    "PickBinding",
    "PickOwner",
    "Point3D",
    "ProjectedPrimitive",
    "ProjectedSelectionIndex",
    "SectionPlane",
    "SelectionConfig",
    "SelectionDepth",
    "SelectionEvent",
    "SelectionFilter",
    "SelectionGesture",
    "SelectionHit",
    "SelectionOperation",
    "SelectionTool",
    "clipping",
    "core",
    "get_color_stops",
    "reset_color_stops",
    "selection",
    "set_color_stops",
    "shading",
    "shapes",
]
