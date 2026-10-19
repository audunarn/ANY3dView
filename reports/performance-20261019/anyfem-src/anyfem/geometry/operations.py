"""Compatibility facade over general geometry and mapped decomposition."""

from anygeometry.operations import surface_point
from anymesher.decomposition import (
    MappabilityReport,
    check_mappable,
    punch_circular_hole,
    split_face_at,
    split_face_between,
    strip_face,
    triangle_to_quads,
)

__all__ = [
    "MappabilityReport",
    "check_mappable",
    "punch_circular_hole",
    "split_face_at",
    "split_face_between",
    "strip_face",
    "surface_point",
    "triangle_to_quads",
]
