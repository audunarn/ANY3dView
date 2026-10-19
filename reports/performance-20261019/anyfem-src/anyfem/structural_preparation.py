"""Deterministic structural connection preparation on a detached geometry.

ANYgeometry owns intersection intent and topology mutation; ANYmesher consumes
only declared Sheets, Members, Attachments and Junctions.  This module is the
workflow bridge.  It operates exclusively on an immutable mesh-job closure,
uses the public query -> plan -> apply contract, and returns provenance that
ANYfem persists with the mesh record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from anygeometry import (
    ConnectionIntent,
    IntersectionDimension,
    IntersectionKind,
)
from anygeometry.closure import ModelClosure
from anygeometry.entities import EntityRef
from anygeometry.errors import GeometryError


__all__ = [
    "StructuralPreparationError",
    "StructuralPreparationReport",
    "prepare_structural_connectivity",
    "remap_mesh_to_source",
    "source_work_mapping",
]


class StructuralPreparationError(ValueError):
    """A candidate connection could not be classified or prepared safely."""


@dataclass(frozen=True)
class PreparedConnection:
    first: str
    second: str
    intersection: str
    operation: str
    reused: bool
    relations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "first": self.first,
            "second": self.second,
            "intersection": self.intersection,
            "operation": self.operation,
            "reused": bool(self.reused),
            "relations": list(self.relations),
        }


@dataclass
class StructuralPreparationReport:
    source_model_id: str
    source_revision: int
    working_model_id: str
    working_revision: int = 0
    connections: list[PreparedConnection] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    source_to_working: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def created_count(self) -> int:
        return sum(not item.reused for item in self.connections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "anyfem.structural-preparation",
            "version": 1,
            "source_model_id": self.source_model_id,
            "source_revision": int(self.source_revision),
            "working_model_id": self.working_model_id,
            "working_revision": int(self.working_revision),
            "created_count": self.created_count,
            "connections": [item.to_dict() for item in self.connections],
            "diagnostics": list(self.diagnostics),
            "source_to_working": {
                key: list(values)
                for key, values in sorted(self.source_to_working.items())
            },
        }


def _label(handle) -> str:
    return f"{handle.kind}:{handle.id}"


def _check(
    cancellation_check: Callable[[str], None] | None, phase: str
) -> None:
    if cancellation_check is not None:
        cancellation_check(phase)


def _face_sheet_map(geometry) -> dict[int, int]:
    return {
        geometry.face_uses[use_id].face_id: sheet.id
        for sheet in geometry.sheets.values()
        for use_id in sheet.face_use_ids
    }


def _candidate_pairs(
    geometry,
    *,
    member_ids: Iterable[int] | None = None,
) -> tuple[tuple[Any, Any], ...]:
    """Spatially bounded structural parent candidates in stable order."""

    face_sheets = _face_sheet_map(geometry)
    face_edges = {
        face_id: {
            item.edge
            for loop in (face.loop,) + face.holes
            for item in loop
        }
        for face_id, face in geometry.faces.items()
    }
    sheet_edges: dict[int, set[int]] = {}
    for face_id, sheet_id in face_sheets.items():
        sheet_edges.setdefault(sheet_id, set()).update(face_edges[face_id])
    face_pairs: set[tuple[int, int]] = set()
    for face_id in sorted(geometry.faces):
        bounds = geometry.entity_bounds_many((('face', face_id),))[0]
        if bounds is None:
            continue
        for kind, other in geometry.spatial_candidates(
            bounds[:3], bounds[3:], kinds=("face",)
        ):
            if other <= face_id:
                continue
            first_sheet = face_sheets.get(face_id)
            second_sheet = face_sheets.get(other)
            if first_sheet is not None and first_sheet == second_sheet:
                continue
            # Reusing the exact edge definition is already explicit,
            # conformal topology.  It must not be sent back through the
            # geometric intersection classifier (which would at best be
            # redundant and, for a generic Coons parameterisation, may be
            # unavailable even though the topology is exact).
            if face_edges[face_id] & face_edges[other]:
                continue
            # Faces intentionally grouped into one Sheet are already one
            # structural shell.  This also prevents conservative curved-face
            # bounds (for example opposite cylinder panels) from producing
            # redundant self-connections.  Independently authored Sheets may
            # still meet again elsewhere in the connected topology: adjacent
            # edge extrusions are the common case, and their coincident free
            # edges must be imprinted rather than hidden by a broad connected-
            # component shortcut.
            face_pairs.add((face_id, other))

    active_members = (
        None if member_ids is None else {int(identifier) for identifier in member_ids}
    )
    edge_members = {
        use.edge_id: use.member_id
        for use in geometry.member_edge_uses.values()
        if active_members is None or use.member_id in active_members
    }
    member_vertices: dict[int, set[int]] = {}
    for edge_id, member_id in edge_members.items():
        edge = geometry.edges[edge_id]
        member_vertices.setdefault(member_id, set()).update(
            (edge.start, edge.end)
        )
    member_pairs: set[tuple[int, int]] = set()
    member_sheet_pairs: set[tuple[int, int]] = set()
    for edge_id, member_id in sorted(edge_members.items()):
        bounds = geometry.entity_bounds_many((('edge', edge_id),))[0]
        if bounds is None:
            continue
        for kind, other in geometry.spatial_candidates(
            bounds[:3], bounds[3:], kinds=("edge", "face")
        ):
            if kind == "edge":
                other_member = edge_members.get(other)
                if other_member is not None and other_member != member_id:
                    # Members that already use the same persistent topology
                    # vertex are conformal.  Their beam meshes reuse that
                    # vertex directly; sending the pair through the imprint
                    # planner would create a redundant junction and, for a
                    # segmented member axis, can produce an ambiguous
                    # multi-component intersection plan.
                    if member_vertices.get(member_id, set()) & member_vertices.get(
                        other_member, set()
                    ):
                        continue
                    member_pairs.add(tuple(sorted((member_id, other_member))))
            else:
                sheet_id = face_sheets.get(other)
                if sheet_id is not None:
                    # A member defined on a face boundary edge already shares
                    # the shell edge registry and therefore the shell nodes.
                    # ANYgeometry's strict audit treats this exact shared edge
                    # as the connectivity witness; no attachment or MPC is
                    # needed merely to prove coincidence.
                    if edge_id in sheet_edges.get(sheet_id, ()):
                        continue
                    member_sheet_pairs.add((member_id, sheet_id))

    pairs: list[tuple[Any, Any]] = []
    pairs.extend(
        (
            geometry.handle("face", first),
            geometry.handle("face", second),
        )
        for first, second in sorted(face_pairs)
    )
    pairs.extend(
        (
            geometry.handle("member", first),
            geometry.handle("member", second),
        )
        for first, second in sorted(member_pairs)
    )
    pairs.extend(
        (
            geometry.handle("member", member),
            geometry.handle("sheet", sheet),
        )
        for member, sheet in sorted(member_sheet_pairs)
    )
    return tuple(pairs)


def prepare_structural_connectivity(
    geometry,
    *,
    source_model_id: str | None = None,
    source_revision: int | None = None,
    member_ids: Iterable[int] | None = None,
    cancellation_check: Callable[[str], None] | None = None,
) -> StructuralPreparationReport:
    """Declare every qualified plate/beam connection on ``geometry``.

    Coplanar area overlap is intentionally not assigned an implicit owner.
    Engineers must run ANYfem's previewable Fragment Overlaps geometry command
    first.  Any unclassified local candidate likewise blocks instead of being
    silently welded or ignored.  ``member_ids`` limits beam connectivity to
    active FE members; generated but unassigned construction members remain
    available in the editable model without entering this mesh job.
    """

    # Keep headless project imports available with published owner versions.
    # The new batch capability is required only when preparing a mesh job;
    # it must fail explicitly rather than substitute different weld semantics.
    try:
        from anygeometry import (
            IntersectionBatchPolicy, plan_intersections, apply_intersections,
        )
    except ImportError as error:
        raise StructuralPreparationError(
            "ANYgeometry lacks the batch-intersection API required for structural "
            "preparation; install the coordinated ANYgeometry owner candidate"
        ) from error

    report = StructuralPreparationReport(
        source_model_id=str(source_model_id or geometry.model_id),
        source_revision=int(
            geometry.revision if source_revision is None else source_revision
        ),
        working_model_id=str(geometry.model_id),
    )
    _check(cancellation_check, "structural intersection batch planning")
    active_members = tuple(sorted(geometry.members if member_ids is None else set(member_ids)))
    if not active_members and geometry.faces:
        from anygeometry import query_trimmed_surface_charts
        from anygeometry.surfaces import CoonsSurface, Cone, RuledSurface
        if all(isinstance(face.surface, (CoonsSurface, Cone, RuledSurface))
               for face in geometry.faces.values()):
            unavailable = set()
            for face in geometry.faces:
                try:
                    query_trimmed_surface_charts(geometry, (geometry.handle("face", face),))
                except GeometryError as error:
                    if str(error) not in (
                            f"face {face} has curved Coons boundaries",
                            f"face {face} is not planar",
                            f"face {face} has an unsupported intersection surface"):
                        raise StructuralPreparationError(str(error)) from error
                    unavailable.add(face)
            if unavailable == set(geometry.faces):
                # Preserve established curved authoring/discretization. The
                # mesher performs its qualified ownership audit on a detached
                # candidate; general interior Coons joints are not inferred.
                from anymesher.preparation import prepare_structural_closure
                from anymesher.errors import MeshError
                try:
                    candidate, _ = prepare_structural_closure(
                        geometry, face_ids=tuple(geometry.faces),
                        options={"declare_missing_owners": False},
                        cancellation_check=cancellation_check,
                        reuse_working_copy=True)
                except (MeshError, GeometryError) as error:
                    raise StructuralPreparationError(str(error)) from error
                geometry.restore_topology(candidate.topology_snapshot())
                report.working_revision = int(geometry.revision)
                return report
    owned_faces = {geometry.face_uses[use].face_id for sheet in geometry.sheets.values()
                   for use in sheet.face_use_ids}
    operands = (*[geometry.handle("sheet", sheet) for sheet in sorted(geometry.sheets)],
                *[geometry.handle("face", face) for face in sorted(set(geometry.faces) - owned_faces)],
                *[geometry.handle("member", member) for member in active_members])
    def cancelled():
        _check(cancellation_check, "structural intersection batch")
        return False
    policy = IntersectionBatchPolicy(ConnectionIntent.CONNECT, cancellation_check=cancelled)
    # Report physical design-owner connections rather than one event per
    # cylindrical panel. Topology still retains every exact panel relation.
    sheets = _face_sheet_map(geometry)
    representatives = {}
    for face in sorted(geometry.faces):
        key = ("sheet", sheets[face]) if face in sheets else ("face", face)
        representatives.setdefault(key, face)
    def face_label(face):
        key = ("sheet", sheets[face]) if face in sheets else ("face", face)
        return f"face:{representatives[key]}"
    try:
        plan = plan_intersections(geometry, operands, policy=policy)
        pairs = set()
        for arrangement in plan.arrangements:
            for path in arrangement.paths:
                for index, first in enumerate(path.owners):
                    for second in path.owners[index+1:]:
                        labels = (face_label(first), face_label(second))
                        if labels[0] != labels[1]:
                            pairs.add(tuple(sorted(labels)))
                for member in path.member_ids:
                    pairs.add((face_label(arrangement.face_id), f"member:{member}"))
        for contact in plan.contacts:
            members = sorted({member for member, _parameter in contact.member_parameters})
            if contact.face_id is not None:
                pairs.update((face_label(contact.face_id), f"member:{member}") for member in members)
            for index, first in enumerate(members):
                pairs.update((f"member:{first}", f"member:{second}") for second in members[index+1:])
        application = apply_intersections(geometry, plan, policy=policy)
    except GeometryError as error:
        if "positive-area" in str(error):
            raise StructuralPreparationError(
                "coplanar plate overlap must be resolved with Fragment Overlaps before meshing: "
                + str(error)) from error
        raise StructuralPreparationError(f"cannot prepare structural intersection batch: {error}") from error
    relations = tuple(_label(handle) for handle in application.joint_edges)
    report.connections.extend(PreparedConnection(first, second, "intersection_curve",
        "batch_arrangement", application.reused, relations) for first, second in sorted(pairs))
    report.working_revision = int(geometry.revision)
    return report


def source_work_mapping(closure: ModelClosure) -> dict[str, tuple[str, ...]]:
    """Current exact lineage from source handles to prepared work handles."""

    geometry = closure.working_model
    mapping: dict[str, tuple[str, ...]] = {}
    for source, initial in sorted(
        closure.source_to_work.items(), key=lambda item: (item[0].kind, item[0].id)
    ):
        if source.kind in ("vertex", "edge", "face"):
            current = geometry.resolve_ref(EntityRef(initial.kind, initial.id))
            if not current and initial.id in {
                "vertex": geometry.vertices,
                "edge": geometry.edges,
                "face": geometry.faces,
            }[initial.kind]:
                current = (EntityRef(initial.kind, initial.id),)
            values = tuple(f"{item.kind}:{item.id}" for item in current)
        else:
            store = getattr(geometry, f"{source.kind}s", {})
            values = (
                (f"{initial.kind}:{initial.id}",)
                if initial.id in store
                else ()
            )
        mapping[_label(source)] = values
    return mapping


def _descendant_to_sources(
    closure: ModelClosure, kind: str
) -> dict[int, tuple[int, ...]]:
    geometry = closure.working_model
    made: dict[int, list[int]] = {}
    for source, initial in closure.source_to_work.items():
        if source.kind != kind:
            continue
        resolved = geometry.resolve_ref(EntityRef(kind, initial.id))
        if not resolved and initial.id in getattr(geometry, f"{kind}s"):
            resolved = (EntityRef(kind, initial.id),)
        for item in resolved:
            owners = made.setdefault(item.id, [])
            if source.id not in owners:
                owners.append(source.id)
    return {
        working_id: tuple(sorted(source_ids))
        for working_id, source_ids in made.items()
    }


def _aggregate(
    mapping: Mapping[int, Iterable[int]],
    owners: Mapping[int, tuple[int, ...]],
):
    result: dict[int, list[int]] = {}
    for working_id, values in mapping.items():
        source_ids = owners.get(int(working_id), ())
        if not source_ids:
            continue
        materialized = [int(item) for item in values]
        for source_id in source_ids:
            result.setdefault(source_id, []).extend(materialized)
    return {
        # Preserve boundary-node order.  Sorting node IDs puts both endpoint
        # nodes before the later-allocated interior nodes and turns a straight
        # 1 m path into a backtracking polyline.  Dict insertion order retains
        # each prepared child edge's parametric sequence while removing only
        # shared junction duplicates.
        key: list(dict.fromkeys(values)) for key, values in sorted(result.items())
    }


def remap_mesh_to_source(mesh, closure: ModelClosure) -> None:
    """Publish prepared mesh associations against immutable source handles."""

    qualified_authority: dict[str, Any] | None = None
    preparation = getattr(mesh, "structural_preparation", None)
    if isinstance(preparation, Mapping) and "qualified_s3" in preparation:
        qualified_s3 = preparation["qualified_s3"]
        if not isinstance(qualified_s3, dict):
            raise StructuralPreparationError(
                "qualified-S3 preparation record must be a mutable mapping"
            )
        authority = qualified_s3.get("authority_model")
        if not isinstance(authority, dict):
            raise StructuralPreparationError(
                "qualified-S3 preparation record lacks model authority"
            )
        qualified_authority = authority

    face_owner = _descendant_to_sources(closure, "face")
    edge_owner = _descendant_to_sources(closure, "edge")
    vertex_owner = _descendant_to_sources(closure, "vertex")
    mesh.elements_of_face = _aggregate(mesh.elements_of_face, face_owner)
    mesh.elements_of_edge = _aggregate(mesh.elements_of_edge, edge_owner)
    mesh.nodes_of_edge = _aggregate(mesh.nodes_of_edge, edge_owner)
    mesh.offset_nodes_of_edge = _aggregate(
        mesh.offset_nodes_of_edge, edge_owner
    )
    seeding = getattr(mesh, "seeding", None)
    if seeding is not None:
        source_divisions: dict[int, int] = {}
        source_classes: dict[int, int] = {}
        for working_id, divisions in seeding.divisions.items():
            for source_id in edge_owner.get(int(working_id), ()):
                source_divisions[source_id] = (
                    source_divisions.get(source_id, 0) + int(divisions)
                )
                if working_id in seeding.classes:
                    source_classes.setdefault(
                        source_id, int(seeding.classes[working_id])
                    )
        seeding.divisions = source_divisions
        seeding.classes = source_classes
    mesh.node_of_vertex = {
        source_id: node_id
        for working_id, node_id in mesh.node_of_vertex.items()
        for source_id in vertex_owner.get(working_id, ())
    }
    # A prepared source face may be represented by several work-face grids;
    # the neutral Mesh fallback derives its nodes from aggregated elements.
    mesh.grid_of_face = {}
    mesh.thickness_of_face = {
        source_id: value
        for working_id, value in mesh.thickness_of_face.items()
        for source_id in face_owner.get(working_id, ())
    }
    mesh.geometry_model_id = closure.source_model_id
    mesh.geometry_revision = closure.source_revision
    if qualified_authority is not None:
        qualified_authority["source_model_id"] = str(closure.source_model_id)
        qualified_authority["source_revision"] = int(closure.source_revision)
