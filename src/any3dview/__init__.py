"""Backend-neutral geometry, camera, shading, clipping and selection core."""

from . import arrays, benchmarks, capabilities, clipping, core, ownership, retained, selection, shading, shapes
from .arrays import MeshArrays
from .capabilities import CORE_CAPABILITIES, ViewerCapabilities
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
from .ownership import ApplicationOwner, ModelOwner, PackedOwnerTable
from .retained import DirtyGenerations, MeshHandle, RetainedViewer
from .scheduler import ViewerScheduler
from .errors import GPUUnavailableError
from .factory import create_viewer
from .shading import Light
from .shapes import Mesh

__version__ = "0.4.0"

__all__ = [
    "Camera3D",
    "DEFAULT_COLOR_STOPS",
    "Light",
    "Mesh",
    "MeshArrays",
    "MeshHandle",
    "DirtyGenerations",
    "ViewerCapabilities",
    "ViewerScheduler",
    "CORE_CAPABILITIES",
    "ApplicationOwner",
    "ModelOwner",
    "PackedOwnerTable",
    "RetainedViewer",
    "GPUUnavailableError",
    "create_viewer",
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
    "arrays",
    "benchmarks",
    "capabilities",
    "core",
    "get_color_stops",
    "reset_color_stops",
    "selection",
    "ownership",
    "retained",
    "set_color_stops",
    "shading",
    "shapes",
]
