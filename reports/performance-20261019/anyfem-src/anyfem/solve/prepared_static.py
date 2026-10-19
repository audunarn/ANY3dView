"""Prepared static evaluation over one owned fixed-model snapshot.

The live :func:`anyfem.solve.iter_linear_static_batches` stream re-observes the
application model for every case: it freezes the whole structural model,
re-resolves geometry targets and refuses to continue when the authored
project moved.  The prepared route pays that observation once.  Preparation
captures an owned snapshot of everything load construction reads -- the
project's structural geometry, coordinate frames, wet surfaces, load-case
definitions and combinations, the ANYmesh mesh data and the resolved
connections -- and hands the fixed solver model to
``anysolver.prepare_static_evaluator``.  After preparation no case build
touches the mutable source model, so later source edits or deletions cannot
change results, and no per-case structural freeze is needed.

Case loads are still built by the canonical ``_resolve_record`` /
``_build_load_case`` / ``_build_combination`` mechanics, so every ANYfem load
family, its coordinate frames and its separate case-versus-per-load envelope
factors behave exactly as in the live stream.  Wet expression, mapping, table
and Python providers are user code: they evaluate once per required sample at
ingress, outside the solver's numerical spans, and the owned geometry's
revision token is checked after each evaluation so a provider that mutates the
observed geometry fails closed instead of yielding bogus loads.

This route requires the solver's prepared evaluator.  When the installed
``anysolver`` cannot provide it the request fails with
:class:`PreparedStaticUnavailableError`; there is never a silent fallback to
the live stream.
"""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import fields, is_dataclass, replace
import sys
import types
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ..model.attributes import LoadCase
from ..model.project import Project, ProjectError
from . import build as builder
from .static_batch import (
    StaticLoadRequest,
    _CaseIterator,
    _LoadBuildContext,
    _checked_cleanup,
    _freeze,
    _output_selection,
    _resolve_record,
)

__all__ = [
    "PreparedStaticEvaluator",
    "PreparedStaticUnavailableError",
    "prepare_linear_static_evaluator",
]


class PreparedStaticUnavailableError(ProjectError):
    """The solver cannot prepare a static evaluator for this request."""


# Geometry stores named by the canonical prepared reference scope; the owned
# clone must carry exactly the source's entity population.
_GEOMETRY_STORES = (
    "vertices", "edges", "faces", "face_uses", "sheets", "parts", "members",
    "coedges", "member_edge_uses", "junctions", "attachments",
)

# Load-record lists captured once when an external LoadCase is ingested.
_CAPTURED_LOAD_LISTS = (
    "point_loads", "pressures", "line_loads", "surface_tractions",
    "wet_pressures", "connection_loads",
)


def _input_storage_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Preparation-only Python/container/array storage estimate, alias-deduplicated.

    This is retained object storage, not RSS or a native allocator measurement.
    Methods/classes/modules do not own application input storage.
    """
    seen = set() if seen is None else seen
    if id(value) in seen or isinstance(value, (type, types.ModuleType,
                                              types.FunctionType, types.MethodType)):
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        if value.base is not None:
            size += _input_storage_bytes(value.base, seen)
        if value.dtype.hasobject:
            size += sum(_input_storage_bytes(item, seen) for item in value.flat)
    elif isinstance(value, Mapping):
        size += sum(_input_storage_bytes(key, seen) + _input_storage_bytes(item, seen)
                    for key, item in value.items())
    elif isinstance(value, (tuple, list, set, frozenset)):
        size += sum(_input_storage_bytes(item, seen) for item in value)
    elif hasattr(value, "__dict__"):
        size += _input_storage_bytes(vars(value), seen)
    elif is_dataclass(value):
        size += sum(_input_storage_bytes(getattr(value, item.name), seen)
                    for item in fields(value))
    return size


def _owned_project(project: Project) -> Project:
    """A detached copy owning every structural input load building reads.

    This is the canonical prepared-mesh preassembly copy: the geometry owner
    makes the detached clone (preserving entity identity so mesh associations
    still resolve), and the project is deep-copied around it.
    """

    geometry = project.geometry.clone(preserve_identity=True)
    owned = deepcopy(project, {id(project.geometry): geometry})
    if owned is project or owned.geometry is project.geometry:
        raise ProjectError(
            "prepared static capture did not detach the project snapshot"
        )
    for store in _GEOMETRY_STORES:
        source_count = len(getattr(project.geometry, store, ()) or ())
        owned_count = len(getattr(owned.geometry, store, ()) or ())
        if source_count != owned_count:
            raise ProjectError(
                f"prepared static capture changed geometry {store!r}: "
                f"{source_count} source entities became {owned_count}"
            )
    return owned


def _owned_mesh(mesh: Any) -> Any:
    """A detached copy of the ANYmesh data load building observes.

    The copy is rebuilt from the canonical declared Mesh schema.  Private
    undeclared instance attributes (for example the mesher's boundary
    registry, which cannot be detached and is not read by load
    construction) are deliberately not carried over.
    """

    from dataclasses import fields as dataclass_fields

    from ..mesh.mapped import Mesh

    captured = {}
    for item in dataclass_fields(mesh):
        try:
            captured[item.name] = deepcopy(getattr(mesh, item.name))
        except Exception as error:
            raise ProjectError(
                f"prepared static capture cannot own mesh field {item.name!r}: "
                f"{type(error).__name__}: {error}"
            ) from error
    owned = Mesh(**captured)
    if owned is mesh:
        raise ProjectError("prepared static capture did not detach the mesh")
    for node_id, value in owned.nodes.items():
        source = mesh.nodes.get(node_id)
        if source is None or np.shares_memory(np.asarray(value), np.asarray(source)):
            raise ProjectError(
                "prepared static capture mesh nodes alias source storage"
            )
    return owned


def _owned_connections(connections: Mapping[str, Any]) -> dict[str, Any]:
    """A detached copy of the resolved connection snapshot."""

    owned = deepcopy(dict(connections))
    if any(owned.get(name) is item for name, item in connections.items()):
        raise ProjectError(
            "prepared static capture did not detach the resolved connections"
        )
    return owned


def _capture_external_case(case: LoadCase) -> LoadCase:
    """Capture an externally supplied load record once, at ingress.

    The record's load lists, parameters and gravity field are copied so the
    captured definition no longer aliases caller storage. The interactive
    region factory is removed because load construction never uses it.
    """

    captured = copy(case)
    for name in _CAPTURED_LOAD_LISTS:
        setattr(captured, name, [deepcopy(load) for load in getattr(case, name)])
    captured.parameters = deepcopy(case.parameters)
    captured._region_factory = None
    captured.gravity = (
        None if case.gravity is None
        else np.array(case.gravity, dtype=float, copy=True)
    )
    # ``times``/``_window_times`` are mutable lists read during case resolution;
    # ``parent`` is a series name (immutable) resolved against the owned
    # project. The interactive region factory is deliberately absent.
    if case.times is not None:
        captured.times = list(case.times)
    window = getattr(case, "_window_times", None)
    if window is not None:
        setattr(captured, "_window_times", list(window))
    return captured


def _capture_request(request: Any) -> Any:
    """Apply one-time ingress capture to a load request."""

    if isinstance(request, StaticLoadRequest) and isinstance(request.load_case, LoadCase):
        return replace(request, load_case=_capture_external_case(request.load_case))
    if isinstance(request, LoadCase):
        return _capture_external_case(request)
    return request


class _PreparedLoadContext(_LoadBuildContext):
    """Load-build caches bound to the owned snapshot.

    The canonical context freezes and re-freezes the live structural model
    around every case.  The prepared context owns its inputs instead, so the
    target, node, element, wet-surface and centroid caches are the only
    shared state, and the remaining integrity check is that the owned
    geometry itself was not mutated: a wet-pressure provider is user code,
    and the geometry owner's cheap ``(identity, model, revision)`` token
    detects API-level mutations without re-serializing the model per case.
    Providers never observe internal geometry storage: the geometry owner
    keeps its position arrays read-only, and the centroid/normal samples
    handed to providers are fresh copies, so a provider cannot corrupt the
    owned snapshot through the values it receives.
    """

    def __init__(self, built: builder.BuiltModel) -> None:
        self.built = built
        self.connections = built.connections
        self._targets: dict[Any, Any] = {}
        self._nodes: dict[Any, Any] = {}
        self._elements: dict[Any, Any] = {}
        self._wet: dict[Any, Any] = {}
        # M31: per wet set, its sorted element ids, centroids, per-element geometry
        # and lazily filled fluid sides (fixed owned geometry; kept like _wet). M31d:
        # the sides are valid for the surface state recorded with the entry; any
        # other state is dropped at the next lookup or not reused (build.py).
        self._wet_geometry: dict[Any, Any] = {}
        # M31b: the wet-set keys are (surface id, object identity); this holds the
        # surfaces (so identities are not reused) and their content fingerprints.
        self._wet_kept: dict[Any, Any] = {}
        self._centroids: dict[int, Any] = {}
        # M20: fixed owned geometry, captured once; never cleared per case.
        self._geometry_normals: dict[int, Any] = {}
        geometry = built.project.geometry
        self._geometry_token = (id(geometry), geometry.model_id, geometry.revision)
        self.cache_limit_bytes = 0
        self._cache_bytes = 0
        # M30: bytes of the entries kept for the context's lifetime (wet elements).
        self._persistent_bytes = 0

    def require_unchanged(self, stage: str) -> None:
        geometry = self.built.project.geometry
        token = (id(geometry), geometry.model_id, geometry.revision)
        if token != self._geometry_token:
            raise ProjectError(
                f"prepared static owned geometry changed during {stage}"
            )


    def clear_case_caches(self) -> None:
        # M30: the wet-element sets depend only on the owned project and mesh and on
        # the selected wet surfaces (owned geometry is checked by its token; M31b:
        # the surfaces by identity and content), so they are kept for the context's
        # lifetime with their bytes still reserved; the other caches remain per
        # case. A request-resolution failure clears everything.
        for cache in (self._targets, self._nodes, self._elements, self._centroids):
            cache.clear()
        if self._persistent_bytes > self.cache_limit_bytes:
            # M31b: a reduced reserve no longer holds the kept entries.
            self._drop_persistent()
        elif self._wet_kept and not self._kept_entries_current():
            # M31f: a kept surface changed since it was charged (it may be larger or
            # no longer selected): nothing kept outlives the request boundary.
            self._drop_persistent()
        self._cache_bytes = self._persistent_bytes

    def _kept_entries_current(self) -> bool:
        for selected, state in self._wet_kept.values():
            if tuple(builder._wet_surface_state(surface) for surface in selected) != state:
                return False
        return True

    def _drop_persistent(self) -> None:
        self._wet.clear()
        self._wet_geometry.clear()
        self._wet_kept.clear()
        self._cache_bytes -= self._persistent_bytes
        self._persistent_bytes = 0

    def clear_all_caches(self) -> None:
        self._drop_persistent()
        self.clear_case_caches()

    @staticmethod
    def _kept_bytes(selected, state):
        """M31c: the kept selection tuple and its content fingerprint; charged with
        each entry of the wet set (twice when both caches hold it: conservative).
        M31i: also the selected surface records themselves, which the kept
        selection holds alive whichever entry was admitted."""
        return 128 + 64 + 8 * len(selected) + _input_storage_bytes(state) + _input_storage_bytes(tuple(selected))

    def _wet_key(self, selected):
        """M31b: the key of a wet set and a check that its surfaces still hold the
        content the kept entries were built from (anything else starts over).

        M31h (owner decision): wet-surface records may change between requests
        (including from the request iterator, progress callbacks or other caller
        code that runs between resolutions); changing them while a request is
        resolved (from a pressure source or a custom pressure record) is outside the
        supported contract. The pre-M30 prepared path memoised the wet set per
        request only when it fitted the cache reserve, so its result under such a
        change depended on the reserve. The checks made after a pressure source are
        defence in depth."""
        state = tuple(builder._wet_surface_state(surface) for surface in selected)
        if any(item is None for item in state):
            return None, None              # M31d: unusual records: nothing is cached
        key = tuple((surface.id, id(surface)) for surface in selected)
        kept = self._wet_kept.get(key)
        if kept is not None and kept[1] != state:
            self._drop_persistent()
        return key, state

    def _keep(self, key, selected, state):
        if key in self._wet or key in self._wet_geometry:
            self._wet_kept[key] = (tuple(selected), state)

    def _cached(self, cache, key, create, persistent=False, extra_bytes=0):
        if key in cache:
            return cache[key]
        value = create()
        # Account conservatively for dictionary capacity as well as keys/values.
        entry_bytes = 128 + _input_storage_bytes((key, value)) + extra_bytes
        if self._cache_bytes + entry_bytes <= self.cache_limit_bytes:
            cache[key] = value
            self._cache_bytes += entry_bytes
            if persistent:
                self._persistent_bytes += entry_bytes
        return value

    def attribute_targets(self, project, region, ref, **kwargs):
        key = (region, ref, tuple(kwargs.get("geometry_kinds") or ()),
               tuple(kwargs.get("mesh_kinds") or ()))
        return self._cached(self._targets, key,
            lambda: builder._attribute_targets(project, region, ref, **kwargs))

    def nodes_on_target(self, mesh, target, *, constrained=False):
        return self._cached(self._nodes, (target, constrained),
            lambda: tuple(builder._nodes_on_target(mesh, target, constrained=constrained)))

    def elements_on_target(self, mesh, target):
        return self._cached(self._elements, target,
            lambda: tuple(builder._elements_on_target(mesh, target)))

    def wet_elements(self, selected):
        key, state = self._wet_key(selected)
        if key is None:
            return builder._wet_elements(self.built.project, self.built.mesh, selected)
        value = self._cached(self._wet, key,
            lambda: builder._wet_elements(self.built.project, self.built.mesh, selected), True,
            self._kept_bytes(selected, state))
        self._keep(key, selected, state)
        return value

    def wet_geometry(self, selected, owner):
        """M31: the record of a wet set (element_ids, centroids, geometry, signs, the
        surface state it was built for, bounded), built once from the same calls the
        uncached path makes per case; None for unusual records or when the context's
        cache reserve cannot hold it (the caller then takes that path). The arrays
        are read-only; the signs are filled on first use (build._signs_reusable)."""
        key, state = self._wet_key(selected)
        if key is None:
            return None
        found = self._wet_geometry.get(key)
        if found is not None:
            return found
        element_ids = np.array(sorted(owner), dtype=int)
        geometry = {}
        for element_id in element_ids:
            centroid, normal = self.centroid_normal(int(element_id))
            centroid.setflags(write=False)
            normal.setflags(write=False)
            geometry[int(element_id)] = (centroid, normal)
        centroids = np.array([geometry[int(e)][0] for e in element_ids]).reshape(-1, 3)
        centroids.setflags(write=False)
        normals = np.array([geometry[int(e)][1] for e in element_ids]).reshape(-1, 3)
        # M31b: whether every centroid and normal is finite and below 1e150 in size (then a
        # fluid side computed from them cannot overflow; see build._signs_reusable).
        bounded = builder._bounded(centroids) and builder._bounded(normals)
        made = {"element_ids": element_ids, "centroids": centroids, "geometry": geometry,
                "signs": {}, "state": state, "bounded": bounded}
        # The sign dictionary is reserved at its full size (one float per element);
        # M31c: also the record itself (with the state it was built for) and the kept
        # selection with its fingerprint.
        entry_bytes = (128 + _input_storage_bytes((key, element_ids, centroids, geometry))
                       + 160 * len(element_ids) + _input_storage_bytes(made) + _input_storage_bytes(state)
                       + self._kept_bytes(selected, state))
        if self._cache_bytes + entry_bytes > self.cache_limit_bytes:
            return None
        self._wet_geometry[key] = made
        self._cache_bytes += entry_bytes
        self._persistent_bytes += entry_bytes
        self._keep(key, selected, state)
        return made

    def _uncached_centroid_normal(self, element_id):
        try:
            return builder._element_centroid_normal(self.built.mesh, element_id)
        finally:
            # The canonical helper has its own mesh-attached cache. Keep all
            # retained cache entries under this context's accounted ceiling.
            self.built.mesh.__dict__.pop("_anyfem_centroid_normal", None)

    def capture_geometry_normals(self) -> None:
        """Corner-mean centroid and unit normal of every owned shell element, once."""
        mesh = self.built.mesh
        ids = [int(eid) for store in ('quads', 'tris') for eid in (getattr(mesh, store, None) or ())]
        for element_id in ids:
            centroid, normal = self._uncached_centroid_normal(element_id)
            centroid = np.array(centroid, dtype=float)
            normal = np.array(normal, dtype=float)
            centroid.setflags(write=False)
            normal.setflags(write=False)
            self._geometry_normals[element_id] = (centroid, normal)

    def centroid_normal(self, element_id):
        found = self._geometry_normals.get(int(element_id))
        if found is not None:
            return found[0].copy(), found[1].copy()
        centroid, normal = self._cached(self._centroids, element_id,
            lambda: self._uncached_centroid_normal(element_id))
        return centroid.copy(), normal.copy()


class PreparedStaticEvaluator:
    """Context owner for prepared static evaluation of one fixed model.

    Use as a context manager, or call :meth:`close` deterministically.  The
    owned snapshot is private: results are consumed through
    :meth:`iter_batches` facades, whose metadata is copied, so a caller can
    never mutate the internal snapshot through them.  No live source object
    survives preparation here: the source project, mesh and FE model are
    released once the solver has taken its own snapshot and the output
    mapping is captured, so the caller may delete or edit them freely.
    """

    def __init__(self, owned: builder.BuiltModel, context: _PreparedLoadContext,
                 output_dofs: Any, connections: Mapping[str, Any],
                 node_dofs: Mapping[int, tuple[int, ...]], owner: Any,
                 provenance: Mapping[str, Any]) -> None:
        self._owned = owned
        self._context = context
        self._output_dofs = output_dofs
        self._connections = connections
        self._node_dofs = node_dofs
        self._owner = owner
        self._provenance = dict(provenance)
        self._prepared_cases = 0
        self._closed = False
        self._active: list[tuple[Any, Any]] = []

    def __enter__(self) -> "PreparedStaticEvaluator":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        _checked_cleanup(self.close, "prepared evaluator cleanup")
        return False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Close the solver owner exactly once; later use fails explicitly."""

        if self._closed:
            return
        self._closed = True
        failures = []
        actions = [action for iterator, cases in tuple(self._active)
                   for action in (iterator.close, cases.close)]
        close_owner = getattr(self._owner, "close", None)
        if close_owner is not None:
            actions.append(close_owner)
        try:
            for action in actions:
                try:
                    action()
                except BaseException as error:
                    failures.append(error)
        finally:
            self._active.clear()
            self._owned = None
            self._context = None
        if failures:
            first = failures[0]
            for error in failures[1:]:
                first.add_note(f"Additional prepared cleanup failure: {type(error).__name__}: {error}")
            raise first

    def diagnostics(self) -> dict[str, Any]:
        """Adapter and solver diagnostics; safe to call after close."""

        solver: Any = {}
        diagnostics = getattr(self._owner, "diagnostics", None)
        if diagnostics is not None:
            solver = diagnostics()
        return {
            "status": "closed" if self._closed else "prepared",
            "route": "anyfem.prepared_static",
            "prepared_cases": self._prepared_cases,
            "output_selection": {
                "selected": self._output_dofs is not None or bool(self._connections),
                "output_dofs": None if self._output_dofs is None else len(self._output_dofs),
                "connections": tuple(sorted(self._connections)),
            },
            "snapshot": deepcopy(self._provenance),
            "solver": solver,
        }

    def envelope_result(self):
        """Return the detached solver envelope and application provenance.

        The solver reduces whole-model recovery while selected histories stream.
        A result is complete only after normal exhaustion of its input stream.
        This accessor remains available after closing the evaluator.
        """
        from ..post.prepared_static import PreparedStaticEnvelope

        result = self._owner.envelope_result()
        return None if result is None else PreparedStaticEnvelope(result, self._provenance)

    def iter_batches(
        self, load_cases: Iterable[str | LoadCase | StaticLoadRequest] | None = None, *,
        batch_size: int = 32, progress: Any = None, cancellation_token: Any = None,
        **solver_options: Any,
    ) -> Iterator[Any]:
        """Stream selected static blocks; refuse a closed owner eagerly."""
        if self._closed:
            raise ValueError("the prepared static evaluator is closed")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return self._iter_batches_open(load_cases, batch_size=batch_size,
            progress=progress, cancellation_token=cancellation_token, **solver_options)

    def _resolve_prepared_request(self, request):
        return _resolve_record(self._owned.project, self._owned, self._context,
                               _capture_request(request))

    def _iter_batches_open(
        self, load_cases: Iterable[str | LoadCase | StaticLoadRequest] | None = None, *,
        batch_size: int = 32, progress: Any = None, cancellation_token: Any = None,
        **solver_options: Any,
    ) -> Iterator[Any]:
        """Yield bounded selected results for exact independent static cases.

        Requests are named cases (resolved from the owned captured
        definitions), externally supplied :class:`LoadCase` records (captured
        once at ingress) or :class:`StaticLoadRequest` cases and combinations
        with optional time labels.  ``None`` streams every non-template case
        of the captured project.  Close the iterator when stopping early.
        """

        from anysolver.control import cancellation_safe_point
        from ..post.prepared_static import PreparedStaticBlock
        from .run import _step_reporter

        if self._closed:
            raise ValueError("the prepared static evaluator is closed")
        cancellation_safe_point(cancellation_token, "prepared_static.adapter_start")
        project = self._owned.project
        context = self._context
        requested = (
            (case for case in project.load_cases.values() if not case.is_template)
            if load_cases is None else load_cases
        )
        labels: dict[int, str] = {}
        evaluator = self

        def records():
            source = None
            index = 0
            try:
                source = iter(requested)
                while True:
                    cancellation_safe_point(
                        cancellation_token, "prepared_static.load_construction"
                    )
                    try:
                        request = next(source)
                    except StopIteration:
                        return
                    context.clear_case_caches()
                    try:
                        record = evaluator._resolve_prepared_request(request)
                    except BaseException:
                        context.clear_all_caches()
                        raise
                    finally:
                        context.clear_case_caches()
                    native = getattr(record, "load_case", None)
                    labels[index] = record.case_id if native is None else native.name
                    evaluator._prepared_cases += 1
                    index += 1
                    yield record
            finally:
                try:
                    close = getattr(source, "close", None)
                    if close is not None:
                        _checked_cleanup(close, "request source cleanup")
                finally:
                    # M31g: a request boundary also when the request iterator ends,
                    # fails or is closed: kept entries whose surfaces changed go.
                    context.clear_case_caches()

        structured = solver_options.pop("progress_callback", None)
        reporter = _step_reporter(progress, "prepared static solve", structured)
        cases = _CaseIterator(records())
        iterator = None
        try:
            iterator = self._owner.iter_batches(
                cases, batch_size=batch_size,
                progress_callback=None if reporter is None else reporter,
                cancellation_token=cancellation_token, **solver_options,
            )
            active = (iterator, cases)
            self._active.append(active)
            for block in iterator:
                block_labels = tuple(labels.pop(index) for index in block.input_indices)
                yield PreparedStaticBlock(
                    block, self._connections, block_labels, self._node_dofs,
                    self._provenance,
                )
        finally:
            if iterator is not None:
                self._active[:] = [item for item in self._active if item[0] is not iterator]
            try:
                if iterator is not None:
                    _checked_cleanup(iterator.close, "prepared solver iterator cleanup")
            finally:
                _checked_cleanup(cases.close, "prepared cases cleanup")


def _prepare_linear_static_evaluator(
    project: Project | None = None, *, built: builder.BuiltModel | None = None,
    mesh: Any = None, target_size: float | None = None, overrides: Any = None,
    output_nodes: Sequence[int] = (), output_refs: Sequence[Any] = (),
    output_connections: Sequence[str] = (), output_dofs: Sequence[int] | None = None,
    recovery_config: Any = None, envelope_recovery_config: Any = None,
    resource_config: Any = None,
    constraint_mode: str = "auto", allow_unbalanced_free_free: bool = False,
    session: Any = None, load_families: Any = None,
    progress: Any = None, cancellation_token: Any = None,
    _factory_name: str = "prepare_static_evaluator", _prepare_application: Any = None,
    _adapter_type: Any = PreparedStaticEvaluator, **solver_options: Any,
) -> PreparedStaticEvaluator:
    """Prepare one fixed structural model for repeated static evaluation.

    The application snapshot (structural geometry, ANYmesh data, targets,
    coordinate frames, resolved connections and load-case definitions) is
    owned from here on; the solver owns its own model snapshot.  After
    preparation, source-model edits or deletions cannot change results and no
    per-case structural freeze runs.  Output selections follow the live
    batch stream; the solver's prepared evaluator certifies the actual
    assembled loads of every case.
    """

    try:
        import anysolver
        prepare_static_evaluator = getattr(anysolver, _factory_name)
    except (ImportError, AttributeError) as error:
        raise PreparedStaticUnavailableError(
            "this anysolver build cannot prepare a static evaluator; the "
            f"prepared static route requires anysolver.{_factory_name}"
        ) from error
    from anysolver.control import cancellation_safe_point
    from .run import _resolve_built, _step_reporter

    cancellation_safe_point(cancellation_token, "prepared_static.prepare_start")
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

    from anysolver.prepared_static_snapshot import capture_source_witness
    # Boundary application is the solver's canonical normalization, preceding
    # the composite witness. Both captures below must then observe one state.
    built.fe_model.apply_boundary_conditions()
    source_context = _LoadBuildContext(built)
    def source_witness():
        return (source_context._structure(),
                _freeze((built.project.load_cases, built.project.combinations)),
                capture_source_witness(built.fe_model))
    before_capture = source_witness()
    owned_project = _owned_project(built.project)
    owned_mesh = _owned_mesh(built.mesh)
    owned_connections = _owned_connections(built.connections)
    owned = builder.BuiltModel(
        fe_model=built.fe_model, load_case=None, mesh=owned_mesh,
        project=owned_project, connections=owned_connections,
    )
    context = _PreparedLoadContext(owned)
    context.capture_geometry_normals()
    output_dofs, connections = _output_selection(
        owned, output_nodes, output_refs, output_connections, output_dofs,
        recovery_config,
    )
    manager = built.fe_model.mesh.dof_manager
    node_dofs = {
        int(node): tuple(int(dof) for dof in manager.get_node_dofs(node))
        for node in built.fe_model.mesh.nodes
    }
    extra_options, application_state = ({}, None) if _prepare_application is None else (
        _prepare_application(owned, context, cancellation_token))
    cancellation_safe_point(cancellation_token, "prepared_static.application_patterns")
    application_input_bytes = _input_storage_bytes(
        (owned_project, owned_mesh, owned_connections, node_dofs, output_dofs,
         context._targets, context._nodes, context._elements, context._wet, context._centroids,
         context._geometry_normals,
         application_state, extra_options))
    cache_reserve = application_input_bytes
    solver_resources = resource_config
    if resource_config is not None and resource_config.memory_limit_bytes is not None:
        remaining = int(resource_config.memory_limit_bytes) - application_input_bytes
        if remaining <= 0:
            raise ProjectError("prepared application snapshot exceeds memory_limit_bytes")
        cache_reserve = min(cache_reserve, remaining // 4)
        solver_resources = replace(resource_config, memory_limit_bytes=remaining-cache_reserve)
    context.cache_limit_bytes = cache_reserve
    # Load construction passes fe_model only to the non-registering combination
    # path. The adapter therefore retains no numerical model reference.
    owned.fe_model = None
    geometry = owned_project.geometry
    provenance = {
        "source_project": built.project.name,
        "owned_snapshot": True,
        "application_retained_input_bytes": application_input_bytes,
        "application_case_cache_limit_bytes": cache_reserve,
        "input_storage_accounting": "deduplicated Python object/array estimate; excludes native allocator overhead",
        "geometry_entities": {
            store: len(getattr(geometry, store, ()) or ()) for store in _GEOMETRY_STORES
        },
        "mesh_nodes": len(owned_mesh.nodes),
        "load_cases": tuple(owned_project.load_cases),
        "retained_input_case_count": len(owned_project.load_cases),
        "retained_input_history_sample_count": sum(
            len(getattr(case, "times", None) or ())
            + len(getattr(case, "_window_times", None) or ())
            for case in owned_project.load_cases.values()),
        "output_dofs": None if output_dofs is None else len(output_dofs),
    }
    structured = solver_options.pop("progress_callback", None)
    reporter = _step_reporter(progress, "prepared static preparation", structured)
    owner = prepare_static_evaluator(
        built.fe_model, output_dofs=output_dofs, recovery_config=recovery_config,
        envelope_recovery_config=envelope_recovery_config,
        cancellation_token=cancellation_token, progress_callback=reporter,
        resource_config=solver_resources, constraint_mode=constraint_mode,
        allow_unbalanced_free_free=allow_unbalanced_free_free, session=session,
        load_families=load_families, **extra_options, **solver_options,
    )
    try:
        if source_witness() != before_capture:
            raise ProjectError("source model changed during prepared static capture")
        adapter = _adapter_type(
            owned, context, output_dofs, connections, node_dofs, owner,
            provenance,
        )
        if application_state is not None:
            adapter._configure_prepared(application_state)
        return adapter
    except BaseException:
        _checked_cleanup(owner.close, "incoherent prepared capture cleanup")
        raise


def prepare_linear_static_evaluator(
    project: Project | None = None, *, built: builder.BuiltModel | None = None,
    mesh: Any = None, target_size: float | None = None, overrides: Any = None,
    output_nodes: Sequence[int] = (), output_refs: Sequence[Any] = (),
    output_connections: Sequence[str] = (), output_dofs: Sequence[int] | None = None,
    recovery_config: Any = None, envelope_recovery_config: Any = None,
    resource_config: Any = None,
    constraint_mode: str = "auto", allow_unbalanced_free_free: bool = False,
    session: Any = None, load_families: Any = None,
    progress: Any = None, cancellation_token: Any = None, **solver_options: Any,
) -> PreparedStaticEvaluator:
    """Prepare one owned fixed structural snapshot for direct static cases."""
    options = locals().copy()
    options.pop("solver_options")
    return _prepare_linear_static_evaluator(**options, **solver_options)
