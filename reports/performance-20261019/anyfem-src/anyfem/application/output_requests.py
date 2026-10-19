"""Freeze output-region membership at the immutable submission boundary."""

from __future__ import annotations

from typing import Iterable

from anygeometry.entities import EntityRef

from ..model.regions import ElementFaceRef, MeshEntityRef, RegionDomain
from ..solve.build import _elements_on_target, _nodes_on_target


def freeze_output_requests(project, identifiers: Iterable[str], mesh, *, mesh_id: str):
    """Resolve canonical scopes once; never re-resolve against the live document.

    Geometry lineage and feature output resolution remain with ANYgeometry.
    Geometry-to-mesh association uses the same targets as loads and supports.
    """

    frozen = []
    for identifier in identifiers:
        request = project.output_requests[identifier]
        region = project.regions[request.region.id]
        geometry = None if project.mesh_only else project.geometry
        if region.domain is RegionDomain.GEOMETRY:
            if geometry is None:
                raise ValueError(f"output request {request.label!r} needs unavailable geometry")
            candidates = tuple(
                EntityRef(kind, int(entity_id))
                for kind, collection in (
                    ("vertex", geometry.vertices), ("edge", geometry.edges),
                    ("face", geometry.faces),
                )
                for entity_id in collection
                if kind == region.entity_kind
            )
            properties = None
        else:
            candidates = tuple(
                MeshEntityRef(mesh_id, kind, int(entity_id))
                for kind, collection in (("node", mesh.nodes),
                                         ("element", {**mesh.shells, **mesh.beams}))
                for entity_id in collection
                if kind == region.entity_kind
            )

            def properties(target):
                values = {"kind": target.kind, "id": target.id}
                if target.kind == "node":
                    values.update(zip(("x", "y", "z"), map(float, mesh.nodes[target.id])))
                return values

        targets = project.regions.resolve(
            region.id, geometry=geometry, mesh_id=mesh_id, candidates=candidates,
            properties=properties,
            feature_resolver=(None if geometry is None else
                              lambda anchor: geometry.features.resolve(anchor, geometry)),
        )
        if not targets:
            raise ValueError(f"output request {request.label!r} has an empty region")
        nodes = sorted({node for target in targets for node in _nodes_on_target(mesh, target)})
        elements = sorted({element for target in targets
                           for element in _elements_on_target(mesh, target)})
        if request.location == "node" and not nodes:
            raise ValueError(f"output request {request.label!r} has no associated mesh nodes")
        if request.location in {"element", "element_face", "integration_point"} and not elements:
            raise ValueError(f"output request {request.label!r} has no associated mesh elements")
        frozen.append({
            "request": request.to_dict(), "mesh_id": mesh_id,
            "node_ids": nodes, "element_ids": elements,
            "element_faces": [[target.element_id, target.local_face]
                              for target in targets if isinstance(target, ElementFaceRef)],
        })
    return tuple(frozen)
