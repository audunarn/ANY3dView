"""Toolkit-neutral execution policy for whole-project meshing.

Meshing can run component-parallel through ANYmesh's public parallel API.  This
module owns ONLY the execution decision: whether parallel work may be
submitted, with how many workers, and why a route was or was not taken.  It is
deliberately free of Qt, Tk and geometry knowledge:

* component discovery, element estimation and the worker pool itself are owned
  by ANYmesh (``generate_hybrid_mesh_parallel`` /
  ``generate_hybrid_mesh_result_parallel``, ``ParallelOptions``,
  ``create_parallel_pool``); ANYfem never duplicates that planner;
* the policy is execution-only.  It never enters project serialization, the
  mesh input hash or the semantic mesh hash, and it stays a plain picklable
  dataclass so it can cross the worker-process boundary unchanged.

The default is ``auto``: eligible whole-project jobs propose bounded
parallelism. An explicit ``off`` policy keeps meshing serial.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
import sys
from typing import Any, Mapping

__all__ = [
    "AUTO_MAX_WORKERS",
    "AUTO_MIN_COMPONENTS",
    "AUTO_RESERVED_CORES",
    "MIN_ESTIMATED_ELEMENTS_COLD",
    "MIN_ESTIMATED_ELEMENTS_WARM",
    "WINDOWS_MAX_WORKERS",
    "MeshExecutionPolicy",
    "MeshExecutionRoute",
    "ParallelMeshUnsupported",
    "auto_worker_count",
    "environment_policy",
    "parallel_mesh_api",
    "resolve_policy",
    "resolve_route",
    "run_parallel_hybrid_mesh",
]

MODES = ("off", "auto", "explicit")

#: Provisional automatic gates (measured acceptance stays with the root).
AUTO_MIN_COMPONENTS = 4
MIN_ESTIMATED_ELEMENTS_COLD = 3000
MIN_ESTIMATED_ELEMENTS_WARM = 500

#: Sizing bounds for automatic mode.
AUTO_RESERVED_CORES = 2
AUTO_MAX_WORKERS = 16
WINDOWS_MAX_WORKERS = 61

ENVIRONMENT_VARIABLE = "ANYFEM_MESH_WORKERS"


@dataclass(frozen=True)
class MeshExecutionPolicy:
    """How much parallel meshing a caller asks for.  Plain and picklable.

    ``off`` keeps every mesh serial.  ``auto`` (the default) lets the sizing
    rules below propose a worker count; ANYmesh still owns the eligibility
    gates (independent components, element estimates).  ``explicit`` names a
    positive worker count and bypasses sizing, never safety.
    """

    mode: str = "auto"
    workers: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(
                f"unknown mesh execution mode {self.mode!r}; use one of {', '.join(MODES)}"
            )
        if self.mode == "explicit":
            if not isinstance(self.workers, int) or isinstance(self.workers, bool):
                raise ValueError("explicit mesh execution requires an integer worker count")
            if self.workers <= 0:
                raise ValueError("explicit mesh execution requires a positive worker count")
        elif self.workers is not None:
            raise ValueError("workers may only be set for explicit mesh execution")

    @classmethod
    def off(cls) -> "MeshExecutionPolicy":
        return cls(mode="off")

    @classmethod
    def auto(cls) -> "MeshExecutionPolicy":
        return cls(mode="auto")

    @classmethod
    def explicit(cls, workers: int) -> "MeshExecutionPolicy":
        # ``workers`` is passed through unchanged: the constructor's
        # integer-only validation rejects bools, floats and strings here
        # instead of silently coercing them.
        return cls(mode="explicit", workers=workers)

    @property
    def requests_parallel(self) -> bool:
        return self.mode != "off"

    def describe(self) -> str:
        if self.mode == "off":
            return "off"
        if self.mode == "auto":
            return "automatic"
        return f"explicit ({self.workers} workers)"


def environment_policy(environ: Mapping[str, str] | None = None) -> MeshExecutionPolicy | None:
    """Read ``ANYFEM_MESH_WORKERS``; ``None`` when the variable is unset.

    ``off``/``0`` disables parallel meshing, ``auto`` selects automatic sizing
    and a positive integer selects an explicit worker count.  Anything else
    refuses clearly instead of guessing.
    """

    raw = (os.environ if environ is None else environ).get(ENVIRONMENT_VARIABLE)
    if raw is None or not str(raw).strip():
        return None
    text = str(raw).strip().lower()
    if text in ("off", "0", "none", "false"):
        return MeshExecutionPolicy.off()
    if text == "auto":
        return MeshExecutionPolicy.auto()
    try:
        workers = int(text)
    except ValueError as error:
        raise ValueError(
            f"{ENVIRONMENT_VARIABLE}={raw!r} is not a valid mesh worker setting; "
            "use 'off', 'auto' or a positive integer"
        ) from error
    if workers <= 0:
        raise ValueError(
            f"{ENVIRONMENT_VARIABLE}={raw!r} is not a valid mesh worker setting; "
            "use 'off', 'auto' or a positive integer"
        )
    return MeshExecutionPolicy.explicit(workers)


def resolve_policy(
    requested: MeshExecutionPolicy | None = None,
    environ: Mapping[str, str] | None = None,
) -> MeshExecutionPolicy:
    """Effective policy: an explicit call (or Qt) wins over the environment."""

    if requested is not None:
        if not isinstance(requested, MeshExecutionPolicy):
            raise ValueError("execution_policy must be a MeshExecutionPolicy")
        return requested
    return environment_policy(environ) or MeshExecutionPolicy.auto()


def auto_worker_count(
    *,
    cpu_count: int | None = None,
    platform: str | None = None,
) -> int:
    """Automatic sizing: reserve cores for the application, cap the pool.

    The count is only a proposal; ANYmesh still applies its own eligibility
    gates before any pool exists.
    """

    cores = int(cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    workers = max(1, cores - AUTO_RESERVED_CORES)
    workers = min(workers, AUTO_MAX_WORKERS)
    if (platform or sys.platform).startswith("win"):
        workers = min(workers, WINDOWS_MAX_WORKERS)
    return workers


@dataclass(frozen=True)
class MeshExecutionRoute:
    """The resolved route for one mesh submission.  Plain and picklable.

    ``route`` is ``"parallel"`` or ``"serial"``; ``reason`` always names why
    (also for a parallel route, e.g. which policy selected it).  Execution-only:
    never serialized into the project and never hashed.
    """

    route: str
    workers: int | None = None
    reason: str = ""
    min_estimated_elements: int | None = None
    min_components: int | None = None

    @property
    def parallel(self) -> bool:
        return self.route == "parallel"

    def describe(self) -> str:
        if self.parallel:
            return f"parallel meshing with {self.workers} worker(s): {self.reason}"
        return f"serial meshing: {self.reason}"


def parallel_mesh_api() -> tuple[Any, Any, str] | None:
    """Probe ANYmesh's public parallel meshing API lazily.

    The public route lives on the ``anymesher`` package itself:
    ``anymesher.generate_hybrid_mesh_parallel(model, *, parallel=ParallelOptions(...), ...)``
    and the result variant ``anymesher.generate_hybrid_mesh_result_parallel``
    (whose return value carries the ``Mesh`` as ``.mesh``).  Returns
    ``(generate_function, ParallelOptions, result_kind)`` where ``result_kind``
    is ``"mesh"`` or ``"result"``, or ``None`` when the installed ANYmesh does
    not provide the agreed public API (older environments).  Nothing is
    imported unless a caller actually asks for parallel execution, so the
    explicit ``off`` policy never loads parallel extras.
    """

    try:
        import anymesher
    except ImportError:
        return None
    options_type = getattr(anymesher, "ParallelOptions", None)
    generate = getattr(anymesher, "generate_hybrid_mesh_parallel", None)
    result_kind = "mesh" if callable(generate) else None
    if generate is None:
        generate = getattr(anymesher, "generate_hybrid_mesh_result_parallel", None)
        result_kind = "result" if callable(generate) else None
    if generate is None or result_kind is None or not isinstance(options_type, type):
        return None
    return generate, options_type, result_kind


def resolve_route(
    policy: MeshExecutionPolicy | None,
    *,
    strategy: str | None,
    change_set: object | None = None,
    quad_options: object | None = None,
    automation: object | None = None,
    strict: bool = False,
    warm_pool: bool = False,
    cpu_count: int | None = None,
    platform: str | None = None,
    api: tuple[Any, Any, str] | None = None,
    api_probe: Any = None,
) -> MeshExecutionRoute:
    """Decide serial vs parallel for one whole-project hybrid mesh.

    Every unsupported path fails closed to serial with an explicit reason.
    ``api``/``api_probe`` exist for tests; production always probes the real
    ANYmesh API here, before any pool is submitted.
    """

    resolved = resolve_policy(policy)
    if not resolved.requests_parallel:
        return MeshExecutionRoute("serial", reason="parallel meshing is off")
    if automation is not None:
        return MeshExecutionRoute(
            "serial", reason="automatic mesh recovery runs serially"
        )
    if change_set is not None:
        return MeshExecutionRoute(
            "serial", reason="change-set meshing is not a whole-project mesh"
        )
    if quad_options is not None:
        return MeshExecutionRoute(
            "serial", reason="quad-first meshing is unsupported for parallel execution"
        )
    if strict:
        return MeshExecutionRoute(
            "serial", reason="strict certification requests serial meshing"
        )
    if strategy == "mapped":
        return MeshExecutionRoute(
            "serial", reason="mapped meshing has no parallel route"
        )
    probed = api if api is not None else (api_probe or parallel_mesh_api)()
    if probed is None:
        return MeshExecutionRoute(
            "serial",
            reason="this ANYmesher installation does not provide the parallel meshing API",
        )
    if resolved.mode == "explicit":
        # An explicit count bypasses the component-size thresholds only;
        # ANYmesh's certified independence/containment gates still apply.
        return MeshExecutionRoute(
            "parallel",
            workers=resolved.workers,
            reason="explicit worker count requested",
            min_estimated_elements=None,
        )
    # Automatic sizing is provisional: ANYmesh owns the real eligibility gates
    # (independent components, element estimates) and refuses unknown
    # estimates; ANYfem only proposes a bounded count and thresholds.  The
    # component and element gates travel with the route: an installation that
    # cannot carry them refuses the submission instead of silently dropping
    # them (see run_parallel_hybrid_mesh).
    return MeshExecutionRoute(
        "parallel",
        workers=auto_worker_count(cpu_count=cpu_count, platform=platform),
        reason="automatic sizing proposed; ANYmesh applies component and estimate gates",
        min_estimated_elements=(
            MIN_ESTIMATED_ELEMENTS_WARM if warm_pool else MIN_ESTIMATED_ELEMENTS_COLD
        ),
        min_components=AUTO_MIN_COMPONENTS,
    )


class ParallelMeshUnsupported(RuntimeError):
    """The probed parallel API cannot serve this call (fail closed to serial).

    Raised before anything is submitted: the caller falls back to the ordinary
    serial hybrid mesher and records the reason.  A failure *after* submission
    is the mesher's own typed error and is never retried serially here.
    """


def _accepted_keywords(function: Any) -> tuple[set[str], bool]:
    parameters = inspect.signature(function).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    return set(parameters), accepts_kwargs


def run_parallel_hybrid_mesh(
    model: Any,
    parameters: Mapping[str, Any],
    route: MeshExecutionRoute,
    probed: tuple[Any, Any, str],
    *,
    pool_lease: Any = None,
) -> Any:
    """Call ANYmesh's public parallel hybrid API for one whole-project mesh.

    ``probed`` is the capability tuple from :func:`parallel_mesh_api`.  The
    options are built from the resolved route only with keywords the installed
    ``ParallelOptions`` actually accepts, and the public generate function is
    called with its agreed ``parallel=`` keyword; a signature that cannot carry
    the execution settings fails closed as :class:`ParallelMeshUnsupported`
    before any pool is submitted.  Returns the mesher's ``Mesh``.
    """

    generate, options_type, result_kind = probed
    accepted, _ = _accepted_keywords(options_type)
    option_kwargs: dict[str, Any] = {"workers": route.workers}
    if route.min_estimated_elements is not None:
        option_kwargs["min_estimated_elements"] = route.min_estimated_elements
    if route.min_components is not None:
        option_kwargs["min_components"] = route.min_components
    if pool_lease is not None:
        # A warm lease stays in this process; it is never pickled into Qt.
        option_kwargs["pool_lease"] = pool_lease
    missing = sorted(key for key in option_kwargs if key not in accepted)
    if missing:
        # A requested control the installed ParallelOptions does not declare
        # is refused BEFORE submission, even when it accepts ``**kwargs``:
        # silently dropping it could dispatch a cold pool or remove an
        # automatic eligibility gate.  The settled public signature declares
        # every control explicitly, so a compatible installation is unaffected.
        raise ParallelMeshUnsupported(
            "the installed ParallelOptions cannot carry the requested execution "
            f"controls ({', '.join(missing)}); refusing rather than dropping them"
        )
    try:
        options = options_type(**option_kwargs)
    except TypeError as error:
        raise ParallelMeshUnsupported(
            f"the installed ParallelOptions rejected the execution settings: {error}"
        ) from error
    generate_accepted, generate_kwargs = _accepted_keywords(generate)
    if "parallel" not in generate_accepted:
        # Fail closed even when the function accepts ``**kwargs``: a generate
        # that does not declare the agreed keyword cannot prove it carries the
        # execution settings instead of silently dropping them.
        raise ParallelMeshUnsupported(
            "the installed parallel meshing API does not accept the parallel= keyword"
        )
    result = generate(model, parallel=options, **dict(parameters))
    if result_kind == "result":
        result = getattr(result, "mesh")
    return result
