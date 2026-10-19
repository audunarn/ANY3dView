"""Staged feature regeneration and engineering-attachment projections."""
from __future__ import annotations
from copy import copy, deepcopy
from dataclasses import replace
from typing import Any, Dict, TYPE_CHECKING
from anygeometry import EntityRef, GeometryError
if TYPE_CHECKING:
    from .project import Project


def _stage_feature_project(project: Project) -> Project:
    working_project = copy(project)
    working_project.geometry = project.geometry.clone(include_features=True)
    # Projection rebuilding and canonical legacy adoption must remain on
    # independent containers until the complete edit is known valid.
    for name in ("face_sections", "edge_sections", "face_assignment_ids", "edge_assignment_ids",
                 "section_assignments", "sheet_join_intents", "geometry_attachment_regions"):
        setattr(working_project, name, dict(getattr(project, name)))
    working_project.regions = deepcopy(project.regions)
    working_project._singleton_region_cache = {}
    working_project._singleton_region_cache_size = -1
    for name in ("supports", "masses", "wet_surfaces", "connections", "imperfections", "refinements"):
        setattr(working_project, name, list(getattr(project, name)))
    working_project.load_cases = {name: copy(case) for name, case in project.load_cases.items()}
    for case in working_project.load_cases.values():
        for name in ("point_loads", "pressures", "line_loads", "surface_tractions",
                     "wet_pressures", "connection_loads"):
            setattr(case, name, list(getattr(case, name)))
        case.parameters = dict(case.parameters)
        case._region_factory = working_project.singleton_region
    working_project.adopt_geometry_attachment_regions()
    return working_project


def _attribute_snapshot(project: Project) -> Dict[str, Any]:
    """Cheap snapshot of everything attached to geometry."""

    return {
        "face_sections": dict(project.face_sections),
        "edge_sections": dict(project.edge_sections),
        "face_assignment_ids": dict(project.face_assignment_ids),
        "edge_assignment_ids": dict(project.edge_assignment_ids),
        "section_assignments": dict(project.section_assignments),
        "regions": deepcopy(project.regions),
        "geometry_attachment_regions": dict(project.geometry_attachment_regions),
        "sheet_join_intents": dict(project.sheet_join_intents),
        "supports": list(project.supports),
        "masses": list(project.masses),
        "wet_surfaces": list(project.wet_surfaces),
        "connections": list(project.connections),
        "imperfections": list(project.imperfections),
        "refinements": list(project.refinements),
        "element_order": project.element_order,
        "loads": {
            name: (
                list(case.point_loads),
                list(case.pressures),
                list(case.line_loads),
                list(case.surface_tractions),
                list(case.wet_pressures),
                list(case.connection_loads),
            )
            for name, case in project.load_cases.items()
        },
    }


def _restore_attributes(project: Project, snapshot: Dict[str, Any]) -> None:
    if not snapshot:
        return
    project.face_sections.clear()
    project.face_sections.update(snapshot["face_sections"])
    project.edge_sections.clear()
    project.edge_sections.update(snapshot["edge_sections"])
    for name in ("face_assignment_ids", "edge_assignment_ids", "section_assignments"):
        if name in snapshot:
            getattr(project, name).clear()
            getattr(project, name).update(snapshot[name])
    if "regions" in snapshot:
        project.regions = deepcopy(snapshot["regions"])
        project._singleton_region_cache = {}
        project._singleton_region_cache_size = -1
    project.geometry_attachment_regions = dict(snapshot.get("geometry_attachment_regions", {}))
    project.sheet_join_intents.clear()
    project.sheet_join_intents.update(snapshot.get("sheet_join_intents", {}))
    project.supports[:] = list(snapshot["supports"])
    project.masses[:] = list(snapshot.get("masses", ()))
    project.wet_surfaces[:] = list(snapshot.get("wet_surfaces", ()))
    project.connections[:] = list(snapshot.get("connections", ()))
    project.imperfections[:] = list(snapshot.get("imperfections", ()))
    project.refinements[:] = list(snapshot.get("refinements", ()))
    project.element_order = snapshot.get("element_order", project.element_order)
    for name, loads in snapshot["loads"].items():
        points, pressures, lines, *optional = loads
        case = project.load_case(name)
        case.point_loads[:] = list(points)
        case.pressures[:] = list(pressures)
        case.line_loads[:] = list(lines)
        case.surface_tractions[:] = list(optional[0] if optional else ())
        case.wet_pressures[:] = list(optional[1] if len(optional) > 1 else ())
        case.connection_loads[:] = list(optional[2] if len(optional) > 2 else ())


def _inactive_feature_ids(project):
    """Explicit suppression and owner-blocked descendants only."""
    records = project.geometry.features.records
    inactive = {record.feature_id for record in records if record.suppressed}
    # Only explicit suppression and its owner-reported blocked descendants
    # allow latent intent. An unrelated failed/unknown output remains an error.
    while True:
        descendants = {record.feature_id for record in records
                       if record.state == "blocked" and (
                           any(dependency in inactive for dependency in record.dependencies)
                           or any(getattr(anchor, "feature_id", None) in inactive
                                  for anchors in record.inputs.values() for anchor in anchors))}
        if descendants.issubset(inactive):
            break
        inactive.update(descendants)
    return inactive


def _resolve_attachment_scope(project, region_id, *, inactive=None):
    """Validate all active anchors; retain inactive anchors as design markers."""
    inactive = _inactive_feature_ids(project) if inactive is None else inactive
    inactive_targets = set()
    def resolve_feature(anchor):
        if getattr(anchor, "feature_id", None) in inactive:
            # Exact persisted intent, never a fabricated geometry entity.
            inactive_targets.add(anchor)
            return (anchor,)
        return project.geometry.features.resolve(anchor, project.geometry)
    targets = project.regions.resolve(region_id, geometry=project.geometry,
                                     feature_resolver=resolve_feature)
    return targets, bool(inactive_targets)


def _rebind_feature_attachments(project: Project, log, *, previous_geometry=None) -> None:
    """Preserve authored compatibility refs and validate canonical live scope."""
    replacements = dict(log)
    inactive = _inactive_feature_ids(project)
    inactive_refs = set()
    if previous_geometry is not None:
        for record in previous_geometry.features.records:
            if record.feature_id in inactive:
                for reference in record.outputs.values():
                    inactive_refs.add(reference)
                    inactive_refs.update(previous_geometry.resolve_ref(reference))

    def rebind(item):
        reference = getattr(item, "ref", None)
        if not isinstance(reference, EntityRef):
            return item
        raw_region = project.raw_geometry_attachment_region(item) if getattr(item, "region", None) is None else None
        region = getattr(item, "region", None) or raw_region
        if region is not None:
            try:
                targets, has_inactive = _resolve_attachment_scope(project, region.id, inactive=inactive)
            except (KeyError, ValueError) as error:
                raise GeometryError(f"feature edit cannot uniquely rebind {reference} for {type(item).__name__}: {error}") from error
            if has_inactive:
                return item
        else:
            targets = replacements.get(reference, project.geometry.resolve_ref(reference))
            if not targets and reference in inactive_refs:
                raise GeometryError(f"suppression would expire {type(item).__name__} attachment {reference}; a persisted output anchor is required")
        targets = tuple(dict.fromkeys(targets))
        if raw_region is not None and len(targets) == 1 and targets[0].kind == reference.kind:
            return item
        lineage = tuple(project.geometry.resolve_ref(reference))
        if len(lineage) == 1 and lineage[0].kind == reference.kind and lineage[0] in targets:
            # The authored scalar cache may name a predecessor. Canonical
            # regions own scope; consumers materialize their live targets.
            return item
        if reference in targets:
            target = reference
        elif len(targets) == 1 and targets[0].kind == reference.kind:
            target = targets[0]
        else:
            raise GeometryError(f"feature edit cannot uniquely rebind {reference} for {type(item).__name__}")
        if target not in project.geometry.resolve_ref(target):
            raise GeometryError(f"feature edit resolved an unavailable attachment {target}")
        return item if target == reference else replace(item, ref=target)

    for name in ("supports", "masses", "wet_surfaces", "connections", "imperfections", "refinements"):
        getattr(project, name)[:] = [rebind(item) for item in getattr(project, name)]
    for case in project.load_cases.values():
        for name in ("point_loads", "pressures", "line_loads", "surface_tractions"):
            getattr(case, name)[:] = [rebind(item) for item in getattr(case, name)]
    # Preserve inactive/unresolved section intent (e.g. feature suppression).
    # Its diagnostic continues to block mesh/solve through the project gate.
    project.resolve_section_assignments()


