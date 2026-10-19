"""ANYfem-owned extensions to ANYgeometry's public feature registry.

Neutral modelling features remain owned by ANYgeometry.  Consumer-specific
mesh preparation intent, such as the mapped butterfly around a circular
opening, is registered here so it can participate in the same detached,
atomic regeneration contract without adding an ANYfem dependency to
ANYgeometry.
"""

from __future__ import annotations

from anygeometry.errors import GeometryError
from anygeometry.features import (
    FeatureRegistry,
    FeatureTopologyRole,
    builtin_feature_registry,
)
from anymesher.decomposition import punch_circular_hole


BUTTERFLY_HOLE_KIND = "anyfem.mesh.butterfly_hole"


def _one_face(inputs):
    values = tuple(inputs.get("face", ()))
    if len(values) != 1 or values[0].kind != "face":
        raise GeometryError("a butterfly-hole feature needs one plate face")
    return values[0]


def _butterfly_hole(geometry, feature, inputs):
    face = _one_face(inputs)
    faces, arcs = punch_circular_hole(
        geometry,
        face.id,
        feature.parameters["centre"],
        float(feature.parameters["radius"]),
    )
    return {
        **{
            f"face/{index}": geometry.entity_ref("face", identifier)
            for index, identifier in enumerate(faces)
        },
        **{
            f"boundary/{index}": geometry.entity_ref("edge", identifier)
            for index, identifier in enumerate(arcs)
        },
    }


def anyfem_feature_registry() -> FeatureRegistry:
    """Return neutral executors plus ANYfem's versioned consumer features."""

    registry = builtin_feature_registry()
    registry.register(
        BUTTERFLY_HOLE_KIND,
        _butterfly_hole,
        replay_mode="mutating",
        topology_role=FeatureTopologyRole.MODIFIER,
    )
    return registry


__all__ = ["BUTTERFLY_HOLE_KIND", "anyfem_feature_registry"]
