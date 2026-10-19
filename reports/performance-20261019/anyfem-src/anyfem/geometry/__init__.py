"""Compatibility namespace for shared geometry and mapped decomposition."""

from .curves import (
    Arc,
    ArcFrame,
    CurveShape,
    DegenerateArcError,
    Spline,
    Straight,
    arc_frame,
)
from .entities import Edge, EntityKind, EntityRef, Face, OrientedEdge, Vertex
from .model import GeometryError, GeometryModel
from .operations import (
    MappabilityReport,
    check_mappable,
    punch_circular_hole,
    split_face_at,
    split_face_between,
    strip_face,
    surface_point,
    triangle_to_quads,
)
from .surfaces import CoonsSurface, Cone, Cylinder, Plane, RuledSurface
from .construction import (
    ConstructionMode,
    ConstructionResult,
    ConstructionTask,
    CoordinateConstruction,
)
from .snapping import (
    GeometrySnapData,
    SnapCandidate,
    SnapEngine,
    SnapKind,
    SnapPoint,
    SnapResult,
    SnapSegment,
    geometry_snap_data,
)
from .sketching import FaceSketchTask

__all__ = [
    "Arc",
    "ArcFrame",
    "CoonsSurface",
    "ConstructionMode",
    "ConstructionResult",
    "ConstructionTask",
    "CoordinateConstruction",
    "Cone",
    "CurveShape",
    "Cylinder",
    "DegenerateArcError",
    "Edge",
    "EntityKind",
    "EntityRef",
    "Face",
    "FaceSketchTask",
    "GeometryError",
    "GeometryModel",
    "GeometrySnapData",
    "MappabilityReport",
    "OrientedEdge",
    "Plane",
    "RuledSurface",
    "Spline",
    "SnapCandidate",
    "SnapEngine",
    "SnapKind",
    "SnapPoint",
    "SnapResult",
    "SnapSegment",
    "Straight",
    "Vertex",
    "arc_frame",
    "check_mappable",
    "geometry_snap_data",
    "punch_circular_hole",
    "split_face_at",
    "split_face_between",
    "strip_face",
    "surface_point",
    "triangle_to_quads",
]
