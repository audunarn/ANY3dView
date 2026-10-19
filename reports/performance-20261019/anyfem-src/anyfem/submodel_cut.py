"""Create a submodel definition by cutting a box out of a parent project.

Implements workflow contract section 4 (attribute carry-over) and section 5
(identity map) on top of ANYgeometry's volume cut (slices S2a and S2b).  Given a
committed snapshot of the parent, a box and the source reference, :func:`cut_submodel`

* cuts the parent's geometry on a detached clone (planes and cylinders; beams on
  plate edges, such as stiffeners, are partitioned at the box with their plates;
  free-standing beams near the box refuse),
* builds a detached child :class:`Project` on the extracted geometry and carries
  materials, plate and beam sections, supports, loads, masses and refinements
  through the cut's lineage; anything that cannot be carried refuses with a typed
  code and nothing is dropped silently.  Every carried beam keeps its parent's
  section frame and neutral-axis offset (checked, slice S4b),
* returns a ready :class:`SubmodelDefinition`: the child, its lineage (child face
  to parent faces) and the artificial cut faces (one group per box side).

The parent is never modified: the work is done on ``snapshot.thaw()``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from uuid import uuid4

import numpy as np

from .model.project import Project
from .model.submodels import SourceReference, SubmodelDefinition, SubmodelRecordError
from .submodel import SubmodelError

__all__ = ["CutReport", "CutSubmodel", "cut_submodel"]


class _Skip(Exception):
    """A load case cannot be carried (reason in the message)."""


@dataclass(frozen=True)
class CutReport:
    """What the cut did; evidence, not persisted."""

    carried: Mapping[str, int]  # attribute kind -> records created in the child
    dropped_outside: Mapping[str, int]  # attribute kind -> parent records with no child entity
    skipped_cases: Mapping[str, str]  # parent load case -> why it was not carried
    cut_edges: Mapping[str, int]  # box side -> number of artificial cut edges
    original_boundary_edges: int
    notes: Tuple[str, ...]
    cut: Any  # anygeometry.volume_cut.VolumeCutResult (in memory only)


@dataclass(frozen=True)
class CutSubmodel:
    definition: SubmodelDefinition
    report: CutReport


# ----------------------------------------------------------------------
def _refuse(code: str, message: str) -> None:
    raise SubmodelError(code, message)


def _targets(parent: Project, region: Any, ref: Any, what: str) -> Tuple[Any, ...]:
    """Parent geometry entities a record applies to (region first, direct ref as fallback)."""

    from anygeometry.entities import EntityRef

    from .model.project import ProjectError
    from .solve.build import _attribute_targets

    try:
        found = _attribute_targets(parent, region, ref)
    except ProjectError as error:
        raise SubmodelError("CUT_REGION_UNRESOLVED", f"{what}: {error}") from error
    for target in found:
        if not isinstance(target, EntityRef) or target.kind not in ("face", "edge", "vertex"):
            raise SubmodelError("CUT_REGION_UNSUPPORTED", f"{what} targets {target!r}, which is not a face, edge or vertex")
    return found


class _Lineage:
    """Source-to-child entity maps from the volume cut."""

    def __init__(self, cut: Any, volume: Any, tolerance: float) -> None:
        self.cut = cut
        self.volume = volume
        self.tolerance = tolerance
        inverse: Dict[str, Dict[int, List[int]]] = {"face": {}, "edge": {}, "vertex": {}}
        for (kind, child_id), sources in cut.lineage.items():
            if kind in inverse:
                for source_id in sources:
                    inverse[kind].setdefault(int(source_id), []).append(int(child_id))
        self.inverse = {kind: {k: sorted(v) for k, v in table.items()} for kind, table in inverse.items()}
        # Working-clone entities that have a child counterpart (the cut's inside).
        self.kept = {source.key for source in cut.closure.work_to_source.values()}
        self.cut_edge_ids = {e for edges in cut.cut_edges.values() for e in edges}
        # Vertices of the artificial cut: the end points and control vertices of its edges.
        geometry = cut.closure.working_model
        vertices: Set[int] = set()
        for edge_id in self.cut_edge_ids:
            edge = geometry.edges[edge_id]
            vertices.update((edge.start, edge.end))
            via = getattr(edge.curve, "via_vertex", None)
            if via is not None:
                vertices.add(int(via))
            vertices.update(int(v) for v in getattr(edge.curve, "control_vertices", ()))
        self.cut_vertex_ids = vertices

    def children(self, targets: Sequence[Any]) -> List[Tuple[str, int]]:
        found: List[Tuple[str, int]] = []
        for target in targets:
            for child_id in self.inverse[target.kind].get(int(target.id), ()):
                found.append((target.kind, child_id))
        return sorted(set(found))

    def whole_inside(self, targets: Sequence[Any]) -> bool:
        """Every descendant of every target survived into the child."""

        from anygeometry.entities import EntityRef

        for target in targets:
            descendants = self.cut.working_model.resolve_ref(EntityRef(target.kind, int(target.id)))
            if not descendants or any((d.kind, d.id) not in self.kept for d in descendants):
                return False
        return True

    def on_cut(self, entity: Tuple[str, int], child: Any) -> bool:
        kind, identifier = entity
        if kind == "edge":
            return identifier in self.cut_edge_ids
        if kind == "vertex":
            return identifier in self.cut_vertex_ids
        return False

    def region_for(self, child: Any, entities: Sequence[Tuple[str, int]], name: str) -> Tuple[Any, Any]:
        """``(ref, region)`` standing for ``entities``: the first entity and, when there are
        several, a hidden manual region holding all of them (one record keeps its total)."""

        from .model.regions import ManualRegion, Region, RegionDomain
        from .model.regions import RegionRef

        kinds = {kind for kind, _identifier in entities}
        if len(kinds) != 1:
            raise SubmodelError("CUT_REGION_UNSUPPORTED", f"{name} spans {sorted(kinds)}; one kind is supported")
        refs = [child.geometry.entity_ref(kind, identifier) for kind, identifier in entities]
        if len(refs) == 1:
            return refs[0], None
        region = Region(f"{name} selection", RegionDomain.GEOMETRY, next(iter(kinds)), ManualRegion(refs), hidden=True)
        child.regions.add(region)
        return refs[0], RegionRef(region.id)


# ----------------------------------------------------------------------
def cut_submodel(
    snapshot: Any,
    volume: Any,
    *,
    source: SourceReference,
    policy: str,
    target_size: float,
    name: str = "submodel",
    child_case: Optional[str] = None,
    tolerances: Optional[Mapping[str, Optional[float]]] = None,
) -> CutSubmodel:
    """Cut ``volume`` out of the parent in ``snapshot`` and build the submodel definition.

    ``snapshot`` is a committed ``ProjectSnapshot`` (``session.snapshot()``) or a
    ``SubmodelSnapshot``.  ``policy`` has no default (workflow contract section 1).
    The result is not registered: pass ``definition`` to :func:`add_submodel`.
    """

    try:
        from anygeometry.errors import GeometryError
        from anygeometry.volume_cut import BoxVolume, VolumeCutError, apply_volume_cut, plan_volume_cut
        from anygeometry.volume_cut import _edge_extent
    except ImportError as error:  # an ANYgeometry without the volume cut
        raise SubmodelError("CUT_UNAVAILABLE", f"this ANYgeometry has no volume cut: {error}") from error
    if not isinstance(volume, BoxVolume):
        raise SubmodelError("CUT_VOLUME_INVALID", "the volume must be an anygeometry BoxVolume")
    snapshot = getattr(snapshot, "snapshot", snapshot)
    parent = snapshot.thaw()
    parent.resolve_section_assignments(strict=False)

    # ---- refusals that need no geometry
    if parent.mesh_only or parent.imported_format:
        _refuse("CUT_PARENT_UNSUPPORTED", "imported or mesh-only parents cannot be cut")
    if parent.imperfections:
        _refuse("CUT_IMPERFECTIONS_UNSUPPORTED", "parents with imperfections cannot be cut in v1")
    if source.job_id not in parent.jobs:
        _refuse("CUT_SOURCE_JOB_MISSING", f"the source job {source.job_id!r} is not in the parent project")
    child_case = source.case if child_case is None else child_case
    tolerance = parent.geometry.tolerance.effective_length(volume.diagonal)
    for edge_id in sorted(parent.edge_sections):  # beams on plate edges are partitioned with their plates
        if (edge_id in parent.geometry.edges and not parent.geometry.faces_using_edge(edge_id)
                and volume.reaches(_edge_extent(parent.geometry, edge_id, volume), tolerance)):
            _refuse("CUT_BEAMS_UNSUPPORTED",
                    f"free-standing beam edge {edge_id} reaches the box (only beams on plate edges are carried)")

    # ---- the cut.  Members none of whose edges carries a beam section are not meshed
    # as beams and carry no structure: they may be left out of the child (reported).
    geometry = parent.geometry
    unassigned = [
        member_id for member_id, member in geometry.members.items()
        if not any(geometry.member_edge_uses[use].edge_id in parent.edge_sections for use in member.edge_use_ids)
    ]
    try:
        plan = plan_volume_cut(geometry, volume, droppable_members=unassigned)
        cut = apply_volume_cut(geometry, plan)
    except VolumeCutError as error:
        raise SubmodelError(error.code, str(error)) from error
    except GeometryError as error:
        raise SubmodelError("CUT_GEOMETRY", str(error)) from error
    if not cut.cut_edges:
        _refuse("CUT_NO_CUT_EDGES", "the box contains whole parent faces and creates no artificial cut")
    if cut.unresolved:
        _refuse("CUT_LINEAGE_UNRESOLVED",
                f"the cut left extracted entities without a source or a cut origin: {list(cut.unresolved)}")
    lineage = _Lineage(cut, volume, tolerance)

    # ---- the detached child
    child = Project(name=f"{parent.name} submodel", geometry=cut.closure.working_model)
    child.units = parent.units
    child.coordinate_systems = dict(parent.coordinate_systems)
    child.element_order = parent.element_order
    child.shell_formulation_policy = parent.shell_formulation_policy
    child.set_native_triangulation_backend(parent.native_triangulation_backend)
    child.native_mesh_settings = parent.native_mesh_settings
    child.target_size = float(target_size)
    carried: Dict[str, int] = {}
    dropped: Dict[str, int] = {}
    notes: List[str] = []
    if cut.dropped_members:
        count_dropped = len(cut.dropped_members)
        dropped["unassigned_members"] = count_dropped
        notes.append(f"{count_dropped} member(s) with no beam section reach the box and are not carried: "
                     f"{list(cut.dropped_members)}")

    def count(table: Dict[str, int], kind: str, n: int = 1) -> None:
        table[kind] = table.get(kind, 0) + n

    for spec in parent.materials.values():
        child.add_material(spec)
    for section in parent.plate_sections.values():
        child.add_plate_section(section.name, section.thickness, section.material, id=section.id)

    # sections: every child face must resolve to exactly one parent section
    face_sources = cut.source_faces
    for child_face in sorted(child.geometry.faces):
        names = {parent.face_sections[s] for s in face_sources.get(child_face, ()) if s in parent.face_sections}
        if not names:
            _refuse("CUT_SECTION_UNRESOLVED", f"child face {child_face} has no plate section in the parent")
        if len(names) > 1:
            _refuse("CUT_SECTION_AMBIGUOUS", f"child face {child_face} descends from faces with sections {sorted(names)}")
        child.assign_plate(child_face, next(iter(names)))
        count(carried, "plate_assignments")

    # beams: every child edge descending from a beam edge takes that edge's section
    for section in parent.beam_sections.values():
        child.add_beam_section(section)
    edge_sources: Dict[int, List[int]] = {}
    for parent_edge in sorted(parent.edge_sections):
        children = lineage.inverse["edge"].get(int(parent_edge), [])
        if not children:
            count(dropped, "beam_assignments")
        for child_edge in children:
            edge_sources.setdefault(child_edge, []).append(int(parent_edge))
    for child_edge, sources in sorted(edge_sources.items()):
        names = {parent.edge_sections[s] for s in sources}
        if len(names) > 1:
            _refuse("CUT_SECTION_AMBIGUOUS", f"child edge {child_edge} descends from beams with sections {sorted(names)}")
        child.assign_beam(child_edge, next(iter(names)))
        count(carried, "beam_assignments")
    child.resolve_section_assignments(strict=True)
    scalar_offsets = {e for e, offset in child.beam_offsets.items() if not isinstance(offset, tuple)}
    for child_edge, sources in sorted(edge_sources.items()):
        _require_same_beam_frame(parent, child, sources[0], child_edge)
        if child_edge in scalar_offsets:
            _require_same_offset_normals(parent, cut, child_edge)

    def child_ref(entity: Tuple[str, int]):
        return child.geometry.entity_ref(entity[0], entity[1])

    # supports
    for support in parent.supports:
        targets = _targets(parent, support.region, support.ref, f"support {support.name!r}")
        found = lineage.children(targets)
        if not found:
            count(dropped, "supports")
            continue
        for entity in found:
            child.add_support(replace(support, ref=child_ref(entity), region=None, id=str(uuid4())))
            count(carried, "supports")

    # masses
    for mass in parent.masses:
        targets = _targets(parent, mass.region, mass.ref, f"mass {mass.name!r}")
        found = lineage.children(targets)
        if not found:
            count(dropped, "masses")
            continue
        if mass.distribution_policy == "total_distributed" and not lineage.whole_inside(targets):
            _refuse("CUT_MASS_CLIPPED", f"mass {mass.name!r} is distributed over geometry the box only partly contains")
        if any(lineage.on_cut(entity, child) for entity in found):
            _refuse("CUT_MASS_ON_CUT", f"mass {mass.name!r} sits on the artificial cut")
        ref, region = lineage.region_for(child, found, f"mass {mass.name!r}")
        child.add_mass(replace(mass, ref=ref, region=region, id=str(uuid4())))
        count(carried, "masses")

    # wet surfaces and connections: refused when they touch the child
    for surface in parent.wet_surfaces:
        if lineage.children(_targets(parent, surface.region, surface.ref, f"wet surface {surface.name!r}")):
            _refuse("CUT_WET_UNSUPPORTED", f"wet surface {surface.name!r} touches the box (not supported in v1)")
        count(dropped, "wet_surfaces")
    for connection in parent.connections:
        if lineage.children(_targets(parent, connection.region, connection.ref, f"connection {connection.name!r}")):
            _refuse("CUT_CONNECTIONS_UNSUPPORTED", f"connection {connection.name!r} touches the box (not supported in v1)")
        count(dropped, "connections")

    # load cases
    skipped: Dict[str, str] = {}
    for parent_case in parent.load_cases.values():
        case_carried: Dict[str, int] = {}
        case_dropped: Dict[str, int] = {}
        regions_before = {region.id for region in child.regions}
        try:
            _carry_case(parent, child, parent_case, lineage, case_carried, case_dropped)
            for kind, n in case_carried.items():
                count(carried, kind, n)
            for kind, n in case_dropped.items():
                count(dropped, kind, n)
        except _Skip as reason:
            child.load_cases.pop(parent_case.name, None)
            for region in [r for r in child.regions if r.id not in regions_before]:
                child.regions.remove(region.id)  # hidden scopes of the discarded case
            if parent_case.name == child_case:
                raise SubmodelError("CUT_LOAD_UNSUPPORTED", f"load case {parent_case.name!r}: {reason}") from None
            skipped[parent_case.name] = str(reason)
    if child_case not in child.load_cases:
        _refuse("CUT_CASE_MISSING", f"the parent has no load case {child_case!r} to carry")
    for combination in parent.combinations.values():
        if all(case in child.load_cases for case in combination.factors):
            child.add_combination(combination.name, dict(combination.factors))
        else:
            notes.append(f"combination {combination.name!r} not carried: it uses a skipped load case")

    # refinements and seeding
    for refinement in parent.refinements:
        if refinement.ref is not None:
            found = lineage.children(_targets(parent, None, refinement.ref, f"refinement {refinement.name!r}"))
            for entity in found:
                child.add_refinement(replace(refinement, ref=child_ref(entity)))
                count(carried, "refinements")
            if not found:
                count(dropped, "refinements")
        else:
            lower, upper = np.asarray(volume.lower), np.asarray(volume.upper)
            centre = volume.local(np.asarray(refinement.center, dtype=float))[0]
            distance = float(np.linalg.norm(np.maximum(np.maximum(lower - centre, centre - upper), 0.0)))
            if distance <= float(refinement.radius):
                child.add_refinement(refinement)
                count(carried, "refinements")
            else:
                count(dropped, "refinements")
    for edge_id, divisions in parent.seeding_overrides.items():
        found = lineage.children([parent.geometry.entity_ref("edge", int(edge_id))])
        if not found:
            count(dropped, "seeding_overrides")
        for kind, child_id in found:
            child.seeding_overrides[child_id] = int(divisions)
            count(carried, "seeding_overrides")

    try:
        definition = SubmodelDefinition(
            name=name, volume=volume.to_dict(), target_size=float(target_size), policy=policy, source=source,
            child_case=child_case, child=child,
            source_faces={face: tuple(sources) for face, sources in face_sources.items()},
            cut_faces={side: tuple(edges) for side, edges in cut.cut_edges.items()},
            **({"tolerances": dict(tolerances)} if tolerances is not None else {}),
        )
        definition.validate()
    except SubmodelRecordError as error:
        raise SubmodelError("CUT_DEFINITION", str(error)) from error
    report = CutReport(
        carried=carried, dropped_outside=dropped, skipped_cases=skipped,
        cut_edges={side: len(edges) for side, edges in cut.cut_edges.items()},
        original_boundary_edges=len(cut.original_boundary_edges), notes=tuple(notes), cut=cut,
    )
    return CutSubmodel(definition, report)


def _require_same_beam_frame(parent: Project, child: Project, parent_edge: int, child_edge: int) -> None:
    """A carried beam keeps its section frame and neutral-axis offset.

    The frame follows the first plate using the line unless the section prescribes
    a web direction; the child's plates are new entities, so the frame is checked
    rather than assumed.
    """

    from .model.project import ProjectError

    try:
        expected = [np.asarray(v, dtype=float) for v in parent.beam_section_frame(parent_edge)]
        found = [np.asarray(v, dtype=float) for v in child.beam_section_frame(child_edge)]
        offsets = (np.asarray(parent.beam_offset_vector(parent_edge), dtype=float),
                   np.asarray(child.beam_offset_vector(child_edge), dtype=float))
    except ProjectError as error:
        raise SubmodelError("CUT_BEAM_FRAME", f"beam edge {child_edge}: {error}") from error
    scale = max(1.0, float(np.max(np.abs(offsets[0]))))
    if (any(float(np.max(np.abs(a - b))) > 1e-9 for a, b in zip(expected, found))
            or float(np.max(np.abs(offsets[0] - offsets[1]))) > 1e-9 * scale):
        _refuse("CUT_BEAM_FRAME",
                f"child beam edge {child_edge} would not keep the section frame or offset of parent edge {parent_edge}")


def _require_same_offset_normals(parent: Project, cut: Any, child_edge: int) -> None:
    """A scalar beam offset is carried only where it cannot turn.

    The mesher stands a scalar-offset beam node off along the nodal normal: the
    average of the shell element normals at that node, each aligned with its
    owner face's normal (ANYmesh ``_owner_normals_around``), or the attached faces'
    normal where no shell is.  The average depends on the mesh, so a remeshed
    child can change it, at the cut and inside the box alike: where plates at an
    angle meet along the beam (element counts per plate change), on curved
    plating (facets change), and where the cut removes plating.

    It cannot change when every parent plate the beam line touches is flat and all
    of them share one signed normal: then every element normal at every station,
    mid-side nodes included, is that normal, in the parent mesh and in the child's
    (whose faces keep their source normals: the volume cut refuses a turned face).
    The plates are the parent's faces using the beam's source edges or meeting its
    source end vertices, and the faces the mesher would fall back to through
    attachments of the members on those edges.  Anything else refuses; an explicit
    offset vector (automatic, back-side or rotated attachment) is carried exactly
    and checked by :func:`_require_same_beam_frame` instead.
    """

    from anygeometry.structural import AttachmentTargetKind
    from anygeometry.surfaces import Plane
    from anygeometry.volume_cut import _planar_coons

    geometry = parent.geometry
    child_geometry = cut.closure.working_model
    edge = child_geometry.edges[child_edge]
    source_edges = set(cut.lineage.get(("edge", int(child_edge)), ()))
    faces = {f for e in source_edges for f in geometry.faces_using_edge(e)}
    for vertex in (edge.start, edge.end):
        for source_vertex in cut.lineage.get(("vertex", int(vertex)), ()):
            faces.update(f for e in geometry.topological_edges_using_vertex(source_vertex)
                         for f in geometry.faces_using_edge(e))
    members = {m for e in source_edges for m in geometry.members_using_edge(e)}
    for attachment in geometry.attachments.values():
        if attachment.member_id is None or int(attachment.member_id) not in members:
            continue
        if attachment.target_kind is AttachmentTargetKind.FACE:
            faces.add(int(attachment.target_id))
        elif attachment.target_kind is AttachmentTargetKind.EDGE:
            faces.update(geometry.faces_using_edge(attachment.target_id))
        elif attachment.target_kind is AttachmentTargetKind.SHEET:
            faces.update(geometry.face_uses[u].face_id for u in geometry.sheets[int(attachment.target_id)].face_use_ids)
    normals = []
    for face_id in sorted(faces):
        surface = getattr(geometry.faces[face_id], "surface", None)
        if not (isinstance(surface, Plane) or _planar_coons(geometry, face_id, surface)):
            _refuse("CUT_BEAM_OFFSET",
                    f"beam edge {child_edge} has a scalar offset along the plate normal and touches the curved plate "
                    f"{face_id}, where the remeshed child could turn it; give the section an explicit offset vector "
                    "(automatic, back side or rotated attachment)")
        normals.append(np.asarray(geometry.face_normal_many(face_id, [(0.5, 0.5)])[0], dtype=float))
    if not normals or any(float(np.max(np.abs(n - normals[0]))) > 1e-9 for n in normals[1:]):
        _refuse("CUT_BEAM_OFFSET",
                f"beam edge {child_edge} has a scalar offset along the plate normal where plates with different normals "
                f"meet ({sorted(faces)}), so the remeshed child could turn it; give the section an explicit offset "
                "vector (automatic, back side or rotated attachment)")


# ----------------------------------------------------------------------
def _carry_case(parent: Project, child: Project, case: Any, lineage: _Lineage,
                carried: Dict[str, int], dropped: Dict[str, int]) -> None:
    """Carry one static load case; raises :class:`_Skip` with the reason when it cannot be."""

    if case.parent is not None or case.times is not None or case.series is not None:
        raise _Skip("it belongs to a load series (time-dependent loading is not carried)")
    if case.wet_pressures:
        raise _Skip("it has wet pressures (not carried in v1)")
    if case.connection_loads:
        raise _Skip("it has connection loads (not carried in v1)")
    for load in (*case.pressures, *case.line_loads, *case.surface_tractions, *case.point_loads):
        if getattr(load, "envelope", None) is not None:
            raise _Skip("a load has a time envelope (not carried)")
    target = child.load_case(case.name)

    def add(table: Dict[str, int], kind: str) -> None:
        table[kind] = table.get(kind, 0) + 1

    for load in case.pressures:
        found = lineage.children(_targets(parent, load.region, load.ref, f"pressure in {case.name!r}"))
        if not found:
            add(dropped, "pressures")
        for entity in found:
            target.add_pressure(child.geometry.entity_ref(*entity), float(load.value))
            add(carried, "pressures")
    for load in case.surface_tractions:
        found = lineage.children(_targets(parent, load.region, load.ref, f"surface traction in {case.name!r}"))
        if not found:
            add(dropped, "surface_tractions")
        for entity in found:
            target.add_surface_traction(child.geometry.entity_ref(*entity), load.traction,
                                        coordinate_system_id=load.coordinate_system_id)
            add(carried, "surface_tractions")
    for load in case.line_loads:
        found = lineage.children(_targets(parent, load.region, load.ref, f"line load in {case.name!r}"))
        if not found:
            add(dropped, "line_loads")
        for entity in found:
            target.add_line_load(child.geometry.entity_ref(*entity), load.force_per_length,
                                 coordinate_system_id=load.coordinate_system_id)
            add(carried, "line_loads")
    for load in case.point_loads:
        targets = _targets(parent, load.region, load.ref, f"point load in {case.name!r}")
        found = lineage.children(targets)
        if not found:
            add(dropped, "point_loads")
            continue
        if load.distribution_policy == "total_distributed" and not lineage.whole_inside(targets):
            raise _Skip("a point load is distributed over geometry the box only partly contains")
        if any(lineage.on_cut(entity, child) for entity in found):
            raise _Skip("a point load sits on the artificial cut")
        try:
            ref, region = lineage.region_for(child, found, f"point load in {case.name!r}")
        except SubmodelError as error:
            raise _Skip(str(error)) from error
        target.add_point_load(ref, load.force, load.moment, region=region,
                              coordinate_system_id=load.coordinate_system_id,
                              distribution_policy=load.distribution_policy)
        add(carried, "point_loads")
    if case.gravity is not None:
        target.gravity = np.array(case.gravity, dtype=float)
        target.gravity_coordinate_system_id = case.gravity_coordinate_system_id
        add(carried, "gravity")
    target.follower_pressure = bool(case.follower_pressure)
