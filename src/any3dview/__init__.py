"""Backend-neutral geometry, camera, shading, clipping and selection core."""

from . import arrays, benchmarks, capabilities, clipping, contracts, core, ownership, retained, selection, shading, shapes
from .arrays import MeshArrays
from .capabilities import CORE_CAPABILITIES, ViewerCapabilities
from .contracts import Pick, ViewerBackend, ViewerState
from .commands import (
    VIEWER_COMMANDS,
    ViewerCommand,
    ViewerCommandController,
    ViewerCommandDescriptor,
    ViewerCommandPriority,
    ViewerCommandResult,
    ViewerObservation,
    viewer_command_manifest,
)
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
from .semantic import SemanticRef, VisibilityState
from .scheduler import ViewerScheduler
from .errors import GPUUnavailableError
from .factory import create_viewer
from .shading import Light
from .shapes import Mesh

__version__ = "0.5.2"

__all__ = [
    "Camera3D",
    "DEFAULT_COLOR_STOPS",
    "Light",
    "Mesh",
    "MeshArrays",
    "MeshHandle",
    "DirtyGenerations",
    "ViewerCapabilities",
    "ViewerBackend",
    "ViewerState",
    "ViewerScheduler",
    "ViewerCommand",
    "ViewerCommandController",
    "ViewerCommandDescriptor",
    "ViewerCommandPriority",
    "ViewerCommandResult",
    "ViewerObservation",
    "VIEWER_COMMANDS",
    "viewer_command_manifest",
    "SemanticRef",
    "VisibilityState",
    "CORE_CAPABILITIES",
    "ApplicationOwner",
    "ModelOwner",
    "PackedOwnerTable",
    "RetainedViewer",
    "GPUUnavailableError",
    "create_viewer",
    "PickBinding",
    "Pick",
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
    "contracts",
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
