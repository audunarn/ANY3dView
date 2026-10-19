"""Background meshing over immutable document snapshots.

The mapped mesher is intentionally a synchronous, headless API.  This module
keeps that useful contract while providing the desktop application with a
single-worker adapter that never calls Tk from its worker thread.  Cancellation
is cooperative: checks surround the mesher and quality pass, so an
uninterruptible topology operation remains visibly ``cancelling`` until it
returns to a safe point.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import replace
import queue
import threading
import traceback
from typing import Any, Mapping

from .document import ProjectSnapshot, canonical_hash
from .job_process import WorkerProcessError, run_in_worker_process
from .jobs import _resolve_executor
from .mesh_controls import MeshControls, StructuredMeshControls

__all__ = [
    "clone_mesh_for_job",
    "MeshJobEvent",
    "MeshJobResult",
    "MeshProgress",
    "MeshSettings",
    "MeshTaskManager",
    "mesh_job_entry",
    "mesh_semantic_hash",
    "run_mesh_job",
]


def clone_mesh_for_job(mesh: Any) -> Any:
    """Return a detached, solver-ready mesh through its public wire format.

    A current ANYmesher mesh may retain immutable ANYgeometry provenance such
    as ``mappingproxy`` objects.  Those values are intentionally not pickle or
    ``deepcopy`` compatible.  The versioned mesh codec is the supported
    boundary for background jobs and also guarantees that the worker receives
    no mutable objects shared with the live document.
    """

    from anymesher.serialize import mesh_from_dict, mesh_to_dict

    return mesh_from_dict(mesh_to_dict(mesh))


def _exception_diagnostic(error: BaseException) -> dict[str, Any]:
    """Preserve public structured evidence exposed by a meshing error."""

    payload: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }
    provider = getattr(error, "to_diagnostic", None)
    if callable(provider):
        try:
            details = provider()
        except BaseException as diagnostic_error:  # noqa: BLE001
            payload["diagnostic_error"] = (
                f"{type(diagnostic_error).__name__}: {diagnostic_error}"
            )
        else:
            if isinstance(details, Mapping):
                payload["details"] = dict(details)
    return payload


@dataclass(frozen=True)
class MeshSettings:
    """All explicit inputs needed to reproduce one mesh submission."""

    target_size: float
    element_order: str = "linear"
    overrides: tuple[tuple[int, int], ...] = ()
    # None preserves the established headless behaviour: inherit the strategy
    # from the snapshotted project, whose legacy default is Automatic.
    strategy: str | None = None
    structure_preference: str = "balanced"
    quality_policy: tuple[tuple[str, float], ...] = ()
    controls: MeshControls | None = None
    structured_controls: StructuredMeshControls | None = None
    quad_options: object | None = None
    layout_policy: str = "existing"
    automation: object | None = None
    # Execution-only: how much parallelism the *submission* asks for.  It is
    # deliberately excluded from ``input_hash`` (a serial and a parallel run of
    # the same mesh inputs must share the reproducibility hash) and never
    # enters the project or the mesh artifact.
    execution_policy: object | None = None

    def __post_init__(self) -> None:
        if self.execution_policy is not None:
            from .mesh_execution import MeshExecutionPolicy

            if not isinstance(self.execution_policy, MeshExecutionPolicy):
                raise TypeError("execution_policy must be a MeshExecutionPolicy")
        if self.automation is not None:
            try:
                from anymesher import MeshAutomationOptions
            except ImportError as error:
                raise RuntimeError(
                    "automatic mesh recovery requires a compatible ANYmesher installation"
                ) from error
            if not isinstance(self.automation, MeshAutomationOptions):
                raise TypeError("automation must be MeshAutomationOptions")
        if self.controls is not None and not isinstance(self.controls, MeshControls):
            raise TypeError("mesh controls must be MeshControls")
        if self.structured_controls is not None and not isinstance(
            self.structured_controls, StructuredMeshControls
        ):
            raise TypeError(
                "structured mesh controls must be StructuredMeshControls"
            )
        quad_selected = str(self.strategy).strip().lower() == "quad_first"
        if quad_selected:
            from .quad_first import effective_quad_options

            object.__setattr__(self, "quad_options", effective_quad_options(self.quad_options))
        elif self.quad_options is not None:
            raise ValueError("quad_options requires strategy='quad_first'")
        if self.layout_policy not in ("existing", "adaptive"):
            raise ValueError("layout_policy must be 'existing' or 'adaptive'")
        if self.layout_policy == "adaptive" and not quad_selected:
            raise ValueError("adaptive layout requires strategy='quad_first'")
        from anymesher.hybrid import MeshingStrategy
        from anymesher.structured import StructurePreference
        from anymesher.structured import MeshQualityPolicy

        if self.strategy is not None:
            try:
                strategy = (
                    "quad_first" if str(self.strategy).strip().lower() == "quad_first"
                    else MeshingStrategy(str(self.strategy).strip().lower()).value
                )
            except ValueError as error:
                choices = ", ".join(item.value for item in MeshingStrategy)
                raise ValueError(
                    f"unknown meshing strategy {self.strategy!r}; expected one of {choices}"
                ) from error
            object.__setattr__(self, "strategy", strategy)
        try:
            preference = StructurePreference(
                str(self.structure_preference).strip().lower()
            ).value
        except ValueError as error:
            choices = ", ".join(item.value for item in StructurePreference)
            raise ValueError(
                "unknown structure preference "
                f"{self.structure_preference!r}; expected one of {choices}"
            ) from error
        object.__setattr__(self, "structure_preference", preference)
        quality = MeshQualityPolicy.create(dict(self.quality_policy))
        object.__setattr__(
            self,
            "quality_policy",
            tuple(sorted((key, float(value)) for key, value in quality.to_dict().items())),
        )

    @classmethod
    def create(
        cls,
        target_size: float,
        *,
        element_order: str,
        overrides: Mapping[int, int] | None = None,
        strategy: str | None = None,
        structure_preference: str = "balanced",
        quality_policy: Mapping[str, float] | None = None,
        controls: MeshControls | None = None,
        structured_controls: StructuredMeshControls | None = None,
        quad_options: object | None = None,
        layout_policy: str = "existing",
        automation: object | None = None,
        execution_policy: object | None = None,
    ) -> "MeshSettings":
        size = float(target_size)
        if size <= 0.0:
            raise ValueError("element size must be positive")
        return cls(
            target_size=size,
            element_order=str(element_order),
            overrides=tuple(
                sorted((int(key), int(value)) for key, value in (overrides or {}).items())
            ),
            strategy=strategy,
            controls=controls,
            structured_controls=structured_controls,
            quad_options=quad_options,
            layout_policy=layout_policy,
            automation=automation,
            structure_preference=structure_preference,
            execution_policy=execution_policy,
            quality_policy=tuple(
                sorted((str(key), float(value)) for key, value in (quality_policy or {}).items())
            ),
        )

    @property
    def input_hash(self) -> str:
        payload = {
                "target_size": self.target_size,
                "element_order": self.element_order,
                "overrides": dict(self.overrides),
                "strategy": self.strategy,
                "quad_options": (
                    None if self.quad_options is None else self.quad_options.to_dict()
                ),
                "layout_policy": self.layout_policy if self.strategy == "quad_first" else None,
                "controls": (None if self.controls is None else
                             self.controls.effective_dict(self.strategy)),
                "structure_preference": (
                    self.structure_preference
                    if self.strategy in (None, "auto")
                    else None
                ),
                "quality_policy": (
                    dict(self.quality_policy)
                    if self.strategy in (None, "auto")
                    else None
                ),
                "structured_controls": (
                    None
                    if self.strategy not in (None, "auto")
                    or self.structured_controls is None
                    else self.structured_controls.effective_dict()
                ),
            }
        if self.automation is not None:
            payload["automation"] = self.automation.to_dict()
        return canonical_hash(payload)


@dataclass(frozen=True)
class MeshProgress:
    """Structured progress suitable for both a status bar and future logs."""

    stage: str
    message: str
    fraction: float


@dataclass(frozen=True)
class MeshJobResult:
    mesh: Any
    mesh_hash: str
    quality: Mapping[str, Any]
    structural_preparation: Mapping[str, Any]
    # Execution-only route record (route/workers/reason, parallel summary);
    # never part of any hash or the persisted project.
    execution: Mapping[str, Any] = dataclass_field(default_factory=dict)


# Execution-route diagnostics the owner parallel block records beside the
# mesh (see anyfem.mesh_execution: the policy is execution-only and never
# enters the semantic mesh hash).  Unknown parallel keys stay hashed.
_PARALLEL_ROUTE_KEYS = frozenset(
    {
        "used",
        "route",
        "reason",
        "components",
        "separation",
        "merge_reasons",
        "runtime",
    }
)


def mesh_semantic_hash(
    mesh: Any,
    *,
    model_hash: str,
    mesh_input_hash: str,
    structural_preparation: Mapping[str, Any] | None = None,
) -> str:
    """Hash solver-affecting mesh semantics, excluding owner bookkeeping."""

    from anymesher.serialize import mesh_to_dict

    mesh_payload = dict(mesh_to_dict(mesh))
    mesh_payload.pop("geometry_model_id", None)
    mesh_payload.pop("geometry_revision", None)
    # ANYmesher 0.4 records its full preparation provenance in the neutral
    # mesh.  That block contains the detached working model UUID and hashes
    # derived from it, so it changes on every otherwise identical mesh job.
    # The canonicalized preparation report below already carries the
    # solver-affecting preparation semantics; keep the mesh portion limited to
    # materialized nodes, elements, associations, and coupling equations.
    mesh_payload.pop("structural_preparation", None)
    hybrid_diagnostics = mesh_payload.get("hybrid_diagnostics")
    if isinstance(hybrid_diagnostics, Mapping):
        hybrid_diagnostics = dict(hybrid_diagnostics)
        hybrid_diagnostics.pop("geometry_model_id", None)
        hybrid_diagnostics.pop("geometry_revision", None)
        hybrid_diagnostics.pop("structural_preparation_hash", None)
        # Candidate counts and phase measurements describe execution, not the
        # materialized mesh or its geometry/quality authority.
        hybrid_diagnostics.pop("runtime", None)
        parallel = hybrid_diagnostics.get("parallel")
        if isinstance(parallel, Mapping):
            # The owner's whole ``parallel`` block is the execution-route
            # record (used, route, reason, components, separation,
            # merge_reasons, runtime).  The execution policy is
            # execution-only and never enters the semantic mesh hash, so a
            # serial and a parallel run of the same mesh must hash
            # identically whether the block is present or absent.  Unknown
            # keys stay hashed; when nothing semantic remains the block is
            # dropped so its presence cannot change the hash either.
            semantic_parallel = {
                key: value
                for key, value in parallel.items()
                if key not in _PARALLEL_ROUTE_KEYS
            }
            if semantic_parallel:
                hybrid_diagnostics["parallel"] = semantic_parallel
            else:
                hybrid_diagnostics.pop("parallel", None)

        def without_runtime_timings(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    key: without_runtime_timings(item)
                    for key, item in value.items()
                    if not str(key).endswith("_seconds")
                }
            if isinstance(value, list):
                return [without_runtime_timings(item) for item in value]
            if isinstance(value, tuple):
                return tuple(without_runtime_timings(item) for item in value)
            return value

        mesh_payload["hybrid_diagnostics"] = without_runtime_timings(
            hybrid_diagnostics
        )
    preparation = dict(structural_preparation or {})
    preparation.pop("working_model_id", None)
    preparation.pop("working_revision", None)
    preparation.pop("source_revision", None)
    preparation.pop("source_model_id", None)
    qualified_s3 = preparation.get("qualified_s3")
    if isinstance(qualified_s3, Mapping):
        qualified_s3 = dict(qualified_s3)
        authority_model = qualified_s3.get("authority_model")
        if isinstance(authority_model, Mapping):
            authority_model = dict(authority_model)
            # The authority remains the same physical model after exact undo;
            # its transaction revision is provenance rather than FE semantics.
            authority_model.pop("source_revision", None)
            qualified_s3["authority_model"] = authority_model
        preparation["qualified_s3"] = qualified_s3
    structural_closure = preparation.get("structural_closure")
    if isinstance(structural_closure, Mapping):
        structural_closure = dict(structural_closure)
        # Both values are derived from the detached job clone and therefore
        # change on an otherwise identical remesh.
        structural_closure.pop("model_id", None)
        structural_closure.pop("preparation_hash", None)
        preparation["structural_closure"] = structural_closure
    structured_layout = preparation.get("structured_layout")
    if isinstance(structured_layout, Mapping):
        structured_layout = dict(structured_layout)
        plan = structured_layout.get("plan")
        if isinstance(plan, Mapping):
            plan = dict(plan)
            # The detached clone receives a new UUID on every mesh job.  It is
            # provenance, not solver-affecting mesh semantics.  The plan hash
            # already captures its canonical topology/options without either
            # transaction identifier.
            plan.pop("model_id", None)
            plan.pop("revision", None)
            structured_layout["plan"] = plan
        structured_layout.pop("source_to_working_faces", None)
        structured_layout.pop("source_to_working_edges", None)
        preparation["structured_layout"] = structured_layout
    return canonical_hash(
        {
            "model_hash": str(model_hash),
            "mesh_input_hash": str(mesh_input_hash),
            "structural_preparation": preparation,
            "mesh": mesh_payload,
        }
    )


@dataclass(frozen=True)
class MeshJobEvent:
    job_id: str
    kind: str
    message: str = ""
    payload: Any = None


@dataclass
class _ActiveMeshJob:
    job_id: str
    cancellation: Any
    future: Future


class MeshTaskManager:
    """Run at most one local mesh operation without blocking the UI thread."""

    def __init__(self, *, executor: str | None = None) -> None:
        # "thread" (the library default) or "process"; see anyfem.job_process.
        self.executor = _resolve_executor(executor)
        self._process: Any = None  # the worker process while a "process" mesh runs
        self._tree: Any = None  # the owned worker tree while a contained mesh runs
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="anyfem-mesh"
        )
        self._events: "queue.Queue[MeshJobEvent]" = queue.Queue()
        self._lock = threading.RLock()
        self._active: _ActiveMeshJob | None = None
        self._closing = False

    @property
    def active_job_id(self) -> str | None:
        with self._lock:
            return None if self._active is None else self._active.job_id

    @property
    def running(self) -> bool:
        with self._lock:
            return self._active is not None and not self._active.future.done()

    @property
    def busy(self) -> bool:
        """Whether a submission still owns the manager, including final polling."""

        with self._lock:
            return self._active is not None

    def submit(
        self,
        job_id: str,
        snapshot: ProjectSnapshot,
        settings: MeshSettings,
    ) -> Future:
        with self._lock:
            if self._closing:
                raise RuntimeError("mesh task manager is closed")
            if self._active is not None:
                raise ValueError("a mesh is already being generated")
            if self._process is not None and self._process.is_alive():
                # A worker that outlived its bounded cleanup stays registered
                # so shutdown can still stop it.  A new submission must not
                # replace that live registration and lose shutdown reachability.
                raise RuntimeError(
                    "a previous mesh worker is still running; shutdown must "
                    "stop it before a new mesh may start"
                )
            # Freeze the effective default/environment choice before deciding
            # whether the submitted job needs owned process containment.
            from .mesh_execution import resolve_policy

            settings = replace(settings, execution_policy=resolve_policy(settings.execution_policy))
            cancellation = _cancellation_token()
            future = self._executor.submit(
                self._run, str(job_id), snapshot, settings, cancellation
            )
            self._active = _ActiveMeshJob(str(job_id), cancellation, future)
            return future

    def cancel(self, job_id: str | None = None) -> bool:
        with self._lock:
            active = self._active
            if active is None or (job_id is not None and job_id != active.job_id):
                return False
            if active.future.done():
                return False
            changed = active.cancellation.cancel("mesh cancellation requested")
            if changed:
                self._events.put(
                    MeshJobEvent(
                        active.job_id,
                        "cancelling",
                        "cancelling mesh; waiting for the current safe phase",
                    )
                )
            return bool(changed)

    def poll(self, limit: int | None = None) -> list[MeshJobEvent]:
        events: list[MeshJobEvent] = []
        while limit is None or len(events) < limit:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            events.append(event)
            if event.kind in ("completed", "cancelled", "failed"):
                with self._lock:
                    if self._active is not None and self._active.job_id == event.job_id:
                        self._active = None
        return events

    def shutdown(self) -> None:
        with self._lock:
            self._closing = True
            active = self._active
            if active is not None:
                active.cancellation.cancel("application is closing")
            process = self._process
            tree = self._tree
        # A worker process does not wait for a cooperative safe point on exit.
        # The registered owned tree is retired FIRST: it covers the outer
        # worker AND any pool children, so touching the outer process
        # separately is out of order.  The standalone process is terminated
        # only when no tree is registered.
        if tree is not None:
            # Bounded and idempotent, and never an unowned PID or group.
            tree.terminate()
        elif process is not None and process.is_alive():
            process.terminate()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(
        self,
        job_id: str,
        snapshot: ProjectSnapshot,
        settings: MeshSettings,
        cancellation: Any,
    ) -> None:
        def progress(stage: str, message: str, fraction: float) -> None:
            self._events.put(
                MeshJobEvent(
                    job_id,
                    "progress",
                    message,
                    MeshProgress(stage, message, float(fraction)),
                )
            )

        self._events.put(MeshJobEvent(job_id, "started", "preparing mesh snapshot"))
        try:
            cancellation.raise_if_cancelled("mesh snapshot")
            if self.executor == "process":
                result = self._run_in_worker(snapshot, settings, cancellation, progress)
            else:
                serial_settings = self._thread_executor_settings(settings, progress)
                result = run_mesh_job(
                    snapshot.thaw(),
                    snapshot.revision.model_hash,
                    serial_settings,
                    progress,
                    cancellation,
                )
                if serial_settings is not settings:
                    execution = dict(result.execution)
                    execution["route"] = "serial"
                    execution["workers"] = None
                    execution["reason"] = (
                        "the thread executor cannot contain a parallel mesh worker tree"
                    )
                    result = replace(result, execution=execution)
        except BaseException as error:  # noqa: BLE001 - retained as diagnostics
            if _is_cancelled(cancellation):
                self._events.put(
                    MeshJobEvent(job_id, "cancelled", "mesh generation cancelled")
                )
            else:
                diagnostic = (
                    error.remote_diagnostic
                    if isinstance(error, WorkerProcessError) and error.remote_diagnostic
                    else _exception_diagnostic(error)
                )
                self._events.put(
                    MeshJobEvent(job_id, "failed", str(error), diagnostic)
                )
            return
        self._events.put(
            MeshJobEvent(job_id, "completed", "mesh generation complete", result)
        )

    def _thread_executor_settings(
        self,
        settings: MeshSettings,
        progress: Any,
    ) -> MeshSettings:
        """The in-process thread executor cannot own a worker tree: force serial."""

        policy = settings.execution_policy
        if policy is None or not policy.requests_parallel:
            return settings
        progress(
            "execution",
            "serial meshing: the thread executor cannot contain a parallel "
            "mesh worker tree",
            0.15,
        )
        from .mesh_execution import MeshExecutionPolicy

        return replace(settings, execution_policy=MeshExecutionPolicy.off())

    def _run_in_worker(
        self,
        snapshot: ProjectSnapshot,
        settings: MeshSettings,
        cancellation: Any,
        progress: Any,
    ) -> "MeshJobResult":
        """Mesh in a separate process, so meshing cannot take the GUI's interpreter lock.

        See :mod:`anyfem.job_process`.  The mesh comes back through the public codec.
        A submission whose execution policy asks for parallel meshing runs in a
        contained, non-daemonic worker tree (the ANYmesh pool creates children);
        when containment cannot be established the job falls back to an
        ordinary serial worker *before anything is dispatched*, with the reason
        recorded -- unless the manager is closing, in which case nothing is
        dispatched at all, or the refusal carries cleanup failures, in which
        case the job fails rather than spawn any further worker.  A failure
        after execution begins is never retried serially.  Registration is
        published and retracted per spawn attempt, so shutdown always
        observes one consistent worker/tree pair.
        """

        def forward(value: Any) -> None:
            if isinstance(value, MeshProgress):
                progress(value.stage, value.message, value.fraction)

        # The spawn attempt that currently owns the registration slots.
        # Ownership is published and retracted per attempt, so shutdown
        # never observes a stale tree from a refused contained attempt
        # beside a live fallback worker, and never observes empty slots
        # while a worker that outlived its bounded cleanup still runs.
        attempt: list[Any] = []

        def register(process: Any) -> None:
            with self._lock:
                if self._closing:
                    # shutdown() already ran: refuse ownership so no worker
                    # may run (or keep running) after shutdown.  The late
                    # worker is also stopped here, under the same lock: the
                    # caller's teardown is bounded and joins first, so a
                    # bare refusal could let it run a whole mesh first.
                    # The cancelled token reports the job as cancelled.
                    if process.is_alive():
                        process.terminate()
                    raise RuntimeError(
                        "mesh task manager is closing; no mesh worker may start"
                    )
                attempt[:] = [process, None]
                self._process = process

        def register_tree(process: Any, tree: Any) -> None:
            with self._lock:
                if self._closing:
                    # shutdown() ran before the tree was registered here:
                    # refuse the registration so nothing is dispatched after
                    # shutdown.  The caller terminates the tree.
                    raise RuntimeError(
                        "mesh task manager is closing; no parallel mesh tree may "
                        "be registered"
                    )
                if attempt and attempt[0] is process:
                    attempt[1] = tree
                self._tree = tree


        def retract() -> None:
            """Retract the finished attempt's registration, atomically.

            A worker confirmed dead is unregistered, exactly as before.  A
            worker that outlived its bounded cleanup stays registered, with
            its closed tree dropped so shutdown's standalone branch can
            stop the survivor: nothing may keep running after shutdown,
            and no later attempt may inherit a stale tree.
            """

            with self._lock:
                if attempt:
                    process, tree = attempt
                else:
                    process = tree = None
                attempt[:] = []
                if tree is not None and self._tree is tree:
                    self._tree = None
                if (
                    process is not None
                    and self._process is process
                    and not process.is_alive()
                ):
                    self._process = None


        def spawn(worker_settings: Any, contained_worker: bool) -> Any:
            try:
                return self._spawn_worker(
                    snapshot,
                    worker_settings,
                    cancellation,
                    forward,
                    contained_worker,
                    register,
                    register_tree,
                )
            finally:
                retract()

        policy = settings.execution_policy
        contained = False
        if policy is not None and policy.requests_parallel:
            from .mesh_execution import parallel_mesh_api

            if parallel_mesh_api() is not None:
                # Only a route that can actually go parallel needs the
                # contained tree; a missing API resolves serial inside the
                # ordinary daemon worker.
                contained = True
            else:
                progress(
                    "execution",
                    "serial meshing: this ANYmesher installation does not "
                    "provide the parallel meshing API",
                    0.15,
                )
        try:
            transport = spawn(settings, contained)
        except WorkerProcessError as error:
            if not contained or error.remote_type != "ContainmentUnavailable":
                raise
            with self._lock:
                closing = self._closing
            if closing:
                # Containment failed while the application is closing: never
                # dispatch a fallback worker after shutdown.
                raise
            if getattr(error, "cleanup_failures", None):
                # Containment failed AND its worker tree could not be
                # cleaned up: the environment cannot be trusted to run any
                # worker right now, so the serial fallback is refused and
                # the job fails with the original refusal (cleanup details
                # attached).  A pure predispatch refusal carries no cleanup
                # failures and still falls back.
                raise
            # Containment failed at startup, before any pool dispatch: one
            # serial fallback is allowed and the reason is surfaced.
            reason = f"contained worker unavailable: {error}"
            progress("execution", f"serial meshing fallback: {reason}", 0.15)
            from .mesh_execution import MeshExecutionPolicy

            serial_settings = replace(
                settings, execution_policy=MeshExecutionPolicy.off()
            )
            transport = spawn(serial_settings, False)
            execution = dict(transport.execution)
            execution["route"] = "serial"
            execution["workers"] = None
            execution["reason"] = reason
            transport = replace(transport, execution=execution)
        from anymesher.serialize import mesh_from_dict

        return MeshJobResult(
            mesh=mesh_from_dict(transport.mesh),
            mesh_hash=transport.mesh_hash,
            quality=transport.quality,
            structural_preparation=transport.structural_preparation,
            execution=transport.execution,
        )

    def _spawn_worker(
        self,
        snapshot: ProjectSnapshot,
        settings: MeshSettings,
        cancellation: Any,
        progress: Any,
        contained: bool,
        register: Any,
        register_tree: Any,
    ) -> Any:
        return run_in_worker_process(
            function=mesh_job_entry,
            kwargs={"settings": settings, "model_hash": snapshot.revision.model_hash},
            snapshot=snapshot,
            project_override=None,
            progress=progress,
            token=cancellation,
            on_process=register,
            containment=contained,
            on_tree=register_tree if contained else None,
        )


def run_mesh_job(
    project: Any,
    model_hash: str,
    settings: "MeshSettings",
    progress: Any,
    cancellation: Any,
) -> "MeshJobResult":
    """Generate, check and hash one mesh.

    Shared by the in-process path and the worker process, so both run exactly the
    same meshing.  ``progress(stage, message, fraction)`` reports stages; the
    cancellation token is checked between them.  Raises on failure or cancellation.
    """

    from anymesher.hybrid import MeshingStrategy

    requested_strategy = settings.strategy
    if requested_strategy is None:
        project_settings = project.native_mesh_settings
        backend = (
            "automatic"
            if project_settings is None
            else str(
                getattr(
                    project_settings.backend,
                    "value",
                    project_settings.backend,
                )
            )
        )
        requested_strategy = {
            "automatic": "auto",
            "auto": "auto",
            "mapped": "mapped",
            "native": "native",
        }.get(backend, backend)
    resolved_strategy = (
        "quad_first" if requested_strategy == "quad_first"
        else MeshingStrategy(requested_strategy).value
    )
    resolved_controls = settings.controls or MeshControls.from_settings(project.native_mesh_settings)
    resolved_structured_controls = (
        settings.structured_controls
        or StructuredMeshControls.from_settings(
            project.native_mesh_settings
        )
    )
    progress("snapshot", "prepared immutable model snapshot", 0.1)
    cancellation.raise_if_cancelled("mesh generation")
    progress("generation", "generating mesh elements", 0.2)
    last_parallel_stage = None

    def check_and_report(stage):
        nonlocal last_parallel_stage
        cancellation.raise_if_cancelled(stage)
        # The owner polls cancellation repeatedly while waiting. Forward only
        # changed public component stages so the run monitor remains useful
        # without flooding the GUI thread with identical updates.
        text = str(stage)
        if text.startswith("parallel ") and text != last_parallel_stage:
            fraction = 0.2
            words = text.split()
            if len(words) == 5 and words[:2] == ["parallel", "component"] and words[3] == "of":
                try:
                    done, total = int(words[2]), int(words[4])
                except ValueError:
                    pass  # unfamiliar owner stage stays a useful text update
                else:
                    if total > 0:
                        fraction += 0.6 * min(1.0, max(0.0, done / total))
            progress("generation", text, fraction)
            last_parallel_stage = text

    mesh = project.generate_mesh(
        settings.target_size,
        overrides=dict(settings.overrides),
        order=settings.element_order,
        strategy=settings.strategy,
        structure_preference=settings.structure_preference,
        quality_policy=dict(settings.quality_policy),
        mesh_controls=resolved_controls,
        structured_controls=resolved_structured_controls,
        quad_options=settings.quad_options,
        layout_policy=settings.layout_policy,
        automation=settings.automation,
        cancellation_check=check_and_report,
        execution_policy=settings.execution_policy,
    )
    execution = dict(getattr(project, "_last_mesh_execution", {}) or {})
    parallel_diagnostics = getattr(mesh, "hybrid_diagnostics", {})
    if isinstance(parallel_diagnostics, Mapping):
        runtime = parallel_diagnostics.get("runtime")
        if isinstance(runtime, Mapping):
            execution["meshing"] = deepcopy(dict(runtime))
        parallel_summary = parallel_diagnostics.get("parallel")
        if isinstance(parallel_summary, Mapping):
            # Deterministic route/component metadata only; the runtime block
            # (counts/timings/estimates) stays execution-only.
            execution["parallel"] = {
                key: value
                for key, value in parallel_summary.items()
                if key != "runtime"
            }
    route = execution.get("route", "serial")
    workers = execution.get("workers")
    reason = execution.get("reason", "")
    progress(
        "execution",
        (
            f"mesh route: {route}"
            + (f" with {workers} worker(s)" if workers else "")
            + (f" ({reason})" if reason else "")
        ),
        0.8,
    )
    cancellation.raise_if_cancelled("mesh quality")
    progress("quality", "checking element quality", 0.85)
    from anymesher import verify_mesh_quality

    quality = verify_mesh_quality(mesh).as_dict()
    cancellation.raise_if_cancelled("mesh hashing")
    progress("hash", "hashing immutable mesh result", 0.94)
    structural_preparation = dict(project._last_mesh_preparation)
    resolved_settings_hash = settings.input_hash
    if (
        settings.strategy is None
        or settings.controls is None
        or settings.structured_controls is None
    ):
        # Older/headless callers inherit the snapshotted project
        # strategy. Canonicalize that resolved value so their hash is
        # identical to an explicit UI submission of the same method.
        resolved_settings_hash = MeshSettings.create(
            settings.target_size,
            element_order=settings.element_order,
            overrides=dict(settings.overrides),
            strategy=resolved_strategy,
            structure_preference=settings.structure_preference,
            quality_policy=dict(settings.quality_policy),
            controls=resolved_controls,
            structured_controls=resolved_structured_controls,
            quad_options=settings.quad_options,
            layout_policy=settings.layout_policy,
            automation=settings.automation,
        ).input_hash
    semantic_input_hash = canonical_hash(
        {
            "mesh_settings": resolved_settings_hash,
            # The constrained triangulator is not consulted by an
            # explicitly mapped job and therefore must not affect its
            # reproducibility hash.
            "native_backend": (
                None
                if resolved_strategy == "mapped"
                else project.native_triangulation_backend
            ),
        }
    )
    mesh_hash = mesh_semantic_hash(
        mesh,
        model_hash=model_hash,
        mesh_input_hash=semantic_input_hash,
        structural_preparation=structural_preparation,
    )
    cancellation.raise_if_cancelled("mesh completion")
    progress("complete", "mesh and quality checks complete", 1.0)
    return MeshJobResult(
        mesh=mesh,
        mesh_hash=mesh_hash,
        quality=quality,
        structural_preparation=structural_preparation,
        execution=execution,
    )


@dataclass(frozen=True)
class _MeshTransport:
    """A finished mesh as it leaves the worker process.

    A live mesh holds objects that cannot be pickled (immutable ANYgeometry views,
    local closures).  ANYmesher's public versioned codec is the supported boundary,
    the same one a saved project uses, so the mesh crosses as its codec dictionary.
    """

    mesh: Mapping[str, Any]
    mesh_hash: str
    quality: Mapping[str, Any]
    structural_preparation: Mapping[str, Any]
    execution: Mapping[str, Any] = dataclass_field(default_factory=dict)


def mesh_job_entry(*, project, progress, cancellation_token, settings, model_hash):
    """The worker-process entry point: mesh, then hand the result over as codec data."""

    from anymesher.serialize import mesh_to_dict

    def staged(stage: str, message: str, fraction: float) -> None:
        progress(MeshProgress(stage, message, float(fraction)))

    result = run_mesh_job(project, model_hash, settings, staged, cancellation_token)
    return _MeshTransport(
        mesh=mesh_to_dict(result.mesh),
        mesh_hash=result.mesh_hash,
        quality=dict(result.quality),
        structural_preparation=dict(result.structural_preparation),
        execution=dict(result.execution),
    )


# Structured evidence a mesher error exposes is computed where the error is raised.
mesh_job_entry.error_diagnostic = _exception_diagnostic

def _cancellation_token():
    try:
        from anysolver import CancellationToken

        return CancellationToken()
    except ImportError:  # pragma: no cover - ANYsolver is a normal dependency
        class Token:
            def __init__(self) -> None:
                self._cancelled = threading.Event()

            @property
            def is_cancelled(self) -> bool:
                return self._cancelled.is_set()

            def cancel(self, _reason: str = "") -> bool:
                changed = not self._cancelled.is_set()
                self._cancelled.set()
                return changed

            def raise_if_cancelled(self, _stage: str = "") -> None:
                if self.is_cancelled:
                    raise RuntimeError("mesh cancelled")

        return Token()


def _is_cancelled(token: Any) -> bool:
    value = getattr(token, "is_cancelled", None)
    if value is None:
        value = getattr(token, "cancelled", False)
    return bool(value() if callable(value) else value)
