"""Lazy public static loads over one fixed structural model."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, fields, is_dataclass
import sys
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from ..model.attributes import LoadCase
from ..model.project import Project, ProjectError
from . import build as builder

__all__ = ["StaticLoadRequest", "iter_linear_static_batches"]


def _checked_cleanup(action, stage: str) -> None:
    """Attach secondary cleanup failures without replacing an active failure."""
    primary = sys.exception()
    # Normal generator termination must still expose a failed unchanged check.
    if isinstance(primary, (GeneratorExit, StopIteration)):
        primary = None
    try:
        action()
    except BaseException as secondary:
        if primary is None:
            raise
        primary.add_note(f"Additional static batch failure during {stage}: "
                         f"{type(secondary).__name__}: {secondary}")
        attached = getattr(primary, "static_batch_secondary_errors", ())
        primary.static_batch_secondary_errors = (*attached, secondary)


class _CaseIterator:
    """Keep the caller's failure visible when closing the records generator."""

    def __init__(self, source):
        self.source = source

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.source)

    def close(self):
        _checked_cleanup(self.source.close, "case records cleanup")


@dataclass(frozen=True)
class StaticLoadRequest:
    """A case or an explicit combination, with independent result identity."""

    case_id: str
    load_case: str | LoadCase | None = None
    combination: str | None = None
    time: float | None = None

    def __post_init__(self) -> None:
        if (self.load_case is None) == (self.combination is None):
            raise ValueError("StaticLoadRequest needs exactly one load_case or combination")
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("case_id must be a nonempty string")
        if self.time is not None and not np.isfinite(float(self.time)):
            raise ValueError("time must be finite")


def _freeze(value: Any, active: set[int] | None = None) -> Any:
    """Capture structural data, including direct edits to public arrays/records."""
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, np.generic):
        return value.item()
    active = set() if active is None else active
    if id(value) in active:
        return ("cycle", id(value))
    active.add(id(value))
    try:
        if isinstance(value, dict):
            return tuple((_freeze(k, active), _freeze(v, active)) for k, v in value.items())
        if isinstance(value, (tuple, list)):
            return tuple(_freeze(v, active) for v in value)
        if isinstance(value, (set, frozenset)):
            return frozenset(_freeze(v, active) for v in value)
        if is_dataclass(value):
            return (type(value), tuple((f.name, _freeze(getattr(value, f.name), active))
                                      for f in fields(value)))
        if hasattr(value, "__dict__"):
            return (type(value), _freeze(vars(value), active))
        return (type(value), repr(value))
    finally:
        active.remove(id(value))


class _LoadBuildContext:
    """Run-local mappings; evaluated loads and case records are never cached."""

    def __init__(self, built: builder.BuiltModel) -> None:
        self.built = built
        self.connections = built.connections
        self._targets: dict[Any, Any] = {}
        self._nodes: dict[Any, Any] = {}
        self._elements: dict[Any, Any] = {}
        self._wet: dict[Any, Any] = {}
        self._centroids: dict[int, Any] = {}
        self._initial = self._structure()

    def _structure(self) -> Any:
        from anygeometry.serialization import to_dict as geometry_to_dict

        project = self.built.project
        names = (
            "materials", "material_ids", "plate_sections", "beam_sections",
            "face_sections", "edge_sections", "section_assignments",
            "supports", "masses", "wet_surfaces", "connections", "imperfections",
            "coordinate_systems", "mesh_only", "shell_formulation_policy",
            "geometry_attachment_regions", "sheet_join_intents",
        )
        return (
            id(project), id(self.built.mesh), id(self.built.fe_model),
            _freeze(geometry_to_dict(project.geometry)),
            tuple((name, _freeze(getattr(project, name, None))) for name in names),
            _freeze(project.regions.to_list()),
            _freeze(self.built.mesh),
            _freeze(self.built.connections),
            _freeze({k: v for k, v in self.built.fe_model.revision_signature().items()
                     if k != "load"}),
        )

    def require_unchanged(self, stage: str) -> None:
        if self._structure() != self._initial:
            raise ProjectError(f"static batch structural model changed during {stage}")

    def attribute_targets(self, project, region, ref, **kwargs):
        key = (region, ref, tuple(kwargs.get("geometry_kinds") or ()),
               tuple(kwargs.get("mesh_kinds") or ()))
        if key not in self._targets:
            self._targets[key] = builder._attribute_targets(project, region, ref, **kwargs)
        return self._targets[key]

    def nodes_on_target(self, mesh, target, *, constrained=False):
        key = (target, constrained)
        if key not in self._nodes:
            self._nodes[key] = tuple(builder._nodes_on_target(mesh, target, constrained=constrained))
        return self._nodes[key]

    def elements_on_target(self, mesh, target):
        if target not in self._elements:
            self._elements[target] = tuple(builder._elements_on_target(mesh, target))
        return self._elements[target]

    def wet_elements(self, selected):
        key = tuple(surface.id for surface in selected)
        if key not in self._wet:
            self._wet[key] = builder._wet_elements(self.built.project, self.built.mesh, selected)
        return self._wet[key]

    def centroid_normal(self, element_id):
        if element_id not in self._centroids:
            self._centroids[element_id] = builder._element_centroid_normal(self.built.mesh, element_id)
        centroid, normal = self._centroids[element_id]
        return centroid.copy(), normal.copy()


def _resolve_record(project, built, context, request):
    from anysolver import StaticCase

    if isinstance(request, StaticLoadRequest):
        identifier = request.case_id
        time_override = None if request.time is None else float(request.time)
        if request.combination is not None:
            solver_case = builder._build_combination(
                project, built.mesh, built.fe_model, request.combination,
                _load_context=context, _register=False, _time_override=time_override,
            )
            return StaticCase(identifier, solver_case, time_override)
        requested_case = request.load_case
    else:
        requested_case = request
        identifier = None
        time_override = None
    case = builder._resolve_load_case(project, requested_case)
    if case is None:
        raise ProjectError("a streaming static case cannot be empty")
    builder._refuse_definition(project, case)
    case = copy(project.resolve_case(case))
    if time_override is not None:
        case.time = float(time_override)
    solver_case = builder._build_load_case(project, built.mesh, case, _load_context=context)
    return StaticCase(case.id if identifier is None else identifier, solver_case, case.time)


def _output_selection(built, output_nodes, output_refs, output_connections,
                      output_dofs, recovery_config):
    nodes = set(int(node) for node in output_nodes)
    for ref in output_refs:
        nodes.update(int(node) for node in built.mesh.nodes_on(ref))
    selected_connections = {}
    for name in output_connections:
        item = built.connections.get(name)
        if item is None:
            item = next((value for value in built.connections.values()
                         if value.connection.name == name), None)
        if item is None:
            raise KeyError(f"no connection {name!r} in the structural model")
        selected_connections[item.connection.id] = item
        nodes.add(item.reference_node)
        if item.section_node is not None:
            nodes.add(item.section_node)
    configured_nodes = None if recovery_config is None else recovery_config.node_ids
    requested_nodes = nodes.copy()
    if configured_nodes is not None:
        configured = set(int(node) for node in configured_nodes)
        if nodes and configured != nodes:
            raise ValueError("output node selection conflicts with RecoveryConfig.node_ids")
        nodes = configured
    manager = built.fe_model.mesh.dof_manager
    dofs = set() if output_dofs is None else set(int(dof) for dof in output_dofs)
    for node in (nodes if output_dofs is None else requested_nodes):
        node_dofs = manager.get_node_dofs(node)
        if not node_dofs:
            raise KeyError(f"node {node} is not in the structural model")
        node_dofs = set(int(dof) for dof in node_dofs)
        if output_dofs is not None and not node_dofs.issubset(dofs):
            raise ValueError("output node selection conflicts with explicit output_dofs")
        if output_dofs is None:
            dofs.update(node_dofs)
    selected = bool(nodes or output_connections or output_refs or output_dofs is not None
                    or configured_nodes is not None)
    if selected:
        return tuple(sorted(dofs)), selected_connections
    return None, built.connections


def iter_linear_static_batches(
    project: Project | None = None, *, built: builder.BuiltModel | None = None,
    mesh=None, target_size: float | None = None, overrides=None,
    load_cases: Iterable[str | LoadCase | StaticLoadRequest] | None = None,
    batch_size: int = 32, output_nodes: Sequence[int] = (), output_refs: Sequence[Any] = (),
    output_connections: Sequence[str] = (), output_dofs: Sequence[int] | None = None,
    recovery_config=None, progress=None, cancellation_token=None, **solver_options,
) -> Iterator[Any]:
    """Yield bounded selected results for exact independent static cases.

    Time labels evaluate load expressions and envelopes; they introduce no
    dynamics or proportional-load assumption. An explicitly empty iterable
    yields no results. Close the iterator when stopping early.
    """
    from anysolver import iter_linear_batches
    from anysolver.control import cancellation_safe_point
    from ..post.results import LinearStaticBatchResult
    from .run import _resolve_built, _step_reporter

    cancellation_safe_point(cancellation_token, "static_batch.adapter_start")
    if built is None:
        built = _resolve_built(
            project, None, mesh=mesh, target_size=target_size, overrides=overrides,
            progress=progress, load_case=None, require_loads=False,
            require_supports=False, allow_quad_first=True,
        )
    elif project is not None and project is not built.project:
        raise ValueError("built model belongs to a different project")
    if mesh is not None and mesh is not built.mesh:
        raise ValueError("mesh conflicts with the built structural model")
    project = built.project
    context = _LoadBuildContext(built)
    output_dofs, connections = _output_selection(
        built, output_nodes, output_refs, output_connections, output_dofs, recovery_config,
    )
    manager = built.fe_model.mesh.dof_manager
    node_dofs = {
        int(node): tuple(int(dof) for dof in manager.get_node_dofs(node))
        for node in built.fe_model.mesh.nodes
    }
    requested = (
        (case for case in project.load_cases.values() if not case.is_template)
        if load_cases is None else load_cases
    )
    labels: dict[int, str] = {}

    def records():
        source = None
        try:
            try:
                source = iter(requested)
            finally:
                _checked_cleanup(lambda: context.require_unchanged("case input iterator"),
                                 "case input iterator")
            index = 0
            while True:
                context.require_unchanged("case input")
                try:
                    try:
                        request = next(source)
                    finally:
                        _checked_cleanup(lambda: context.require_unchanged("case input provider"),
                                         "case input provider")
                except StopIteration:
                    return
                cancellation_safe_point(cancellation_token, "static_batch.load_construction")
                try:
                    record = _resolve_record(project, built, context, request)
                finally:
                    _checked_cleanup(lambda: context.require_unchanged("load construction"),
                                     "load construction")
                labels[index] = record.load_case.name
                index += 1
                yield record
        finally:
            def close_source():
                try:
                    close = getattr(source, "close", None)
                    if close is not None:
                        close()
                finally:
                    _checked_cleanup(lambda: context.require_unchanged("case input cleanup"),
                                     "case input cleanup")
            _checked_cleanup(close_source, "case input cleanup")

    structured = solver_options.pop("progress_callback", None)
    reporter = _step_reporter(progress, "batch static solve", structured)
    def checked_progress(payload):
        try:
            if reporter is not None:
                reporter(payload)
        finally:
            _checked_cleanup(lambda: context.require_unchanged("progress callback"),
                             "progress callback")

    iterator = iter_linear_batches(
        built.fe_model, _CaseIterator(records()), batch_size=batch_size, output_dofs=output_dofs,
        recovery_config=recovery_config, cancellation_token=cancellation_token,
        progress_callback=None if reporter is None else checked_progress, **solver_options,
    )
    try:
        for block in iterator:
            context.require_unchanged("solver result")
            block_labels = tuple(labels.pop(index) for index in block.input_indices)
            yield LinearStaticBatchResult(built, block, connections, block_labels, node_dofs)
            context.require_unchanged("result consumer")
    finally:
        _checked_cleanup(iterator.close, "solver iterator cleanup")
