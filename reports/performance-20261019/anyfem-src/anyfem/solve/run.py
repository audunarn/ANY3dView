"""Analysis dispatch.

Every analysis follows the same shape: validate, build, hand the FEModel to
``anysolver``, and wrap the raw DOF vectors in something addressed by geometry.
What differs is only what the solver needs and what comes back.

Long-running analyses take a ``progress`` callable.  It is called from whatever
thread the solve runs on, so it must not touch widgets -- the GUI passes one
that puts the message on a queue.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..mesh.mapped import Mesh
from ..model.attributes import LoadCase
from ..model.project import Project, ProjectError
from .build import BuiltModel, build_fe_model

if TYPE_CHECKING:
    from anysolver import CancellationToken

__all__ = [
    "ContactConfigurationError",
    "eigenmode_imperfection",
    "preflight",
    "solve_arc_length",
    "solve_buckling",
    "solve_impact",
    "solve_linear_static",
    "solve_linear_static_many",
    "solve_modal",
    "solve_nonlinear_static",
    "solve_transient",
]

Progress = Optional[Callable[[Any], None]]


class _StructuredProgressText(str):
    """A readable log line that retains its structured solver payload.

    Existing headless callers treat progress values as strings.  The job
    queue, however, also needs the numerical metadata for live plots.  A
    string subclass preserves both contracts without emitting duplicate log
    entries for one solver event.
    """

    def __new__(cls, message: str, payload: Mapping[str, Any]):
        instance = super().__new__(cls, message)
        instance.payload = payload
        return instance

    @property
    def message(self) -> str:
        return str(self)

    def get(self, key: str, default: Any = None) -> Any:
        getter = getattr(self.payload, "get", None)
        return getter(key, default) if callable(getter) else default


def _report(progress: Progress, message: str) -> None:
    if progress is not None:
        progress(message)


def _resolve_mesh(
    project: Project,
    mesh: Optional[Mesh],
    target_size: Optional[float],
    overrides: Optional[Mapping[int, int]],
) -> Mesh:
    if mesh is not None:
        return mesh
    if target_size is None:
        raise ValueError("pass either an existing mesh or a target element size")
    return project.generate_mesh(target_size, overrides=overrides)


def _resolve_built(
    project: Optional[Project],
    built: Optional[BuiltModel],
    *,
    mesh: Optional[Mesh],
    target_size: Optional[float],
    overrides: Optional[Mapping[int, int]],
    progress: Progress,
    allow_quad_first: bool = False,
    **build_options: Any,
) -> BuiltModel:
    """Use a model that is already built, or mesh and build one.

    An imported model arrives already built and has no geometry to mesh, so
    every analysis takes this path rather than assuming a Project behind it.
    """

    if not allow_quad_first:
        selected_mesh = built.mesh if built is not None else mesh
        selected = bool(
            selected_mesh is not None
            and getattr(selected_mesh, "hybrid_diagnostics", {}).get("quad_first_api")
        )
        if project is not None and project.native_mesh_settings is not None:
            selected = selected or project.native_mesh_settings.backend.value == "quad_first"
        if selected:
            raise ProjectError(
                "quad-first application meshes are accepted for linear-static "
                "and nonlinear-static consumption only; this analysis requires "
                "separate qualification"
            )
    if built is not None:
        return built
    if project is None:
        raise ValueError("pass either a project or an already-built model")
    resolved = _resolve_mesh(project, mesh, target_size, overrides)
    _report(progress, "building the model")
    return build_fe_model(project, resolved, **build_options)


# ----------------------------------------------------------------------
# linear static
# ----------------------------------------------------------------------
def solve_linear_static(
    project: Optional[Project] = None,
    *,
    built: Optional[BuiltModel] = None,
    mesh: Optional[Mesh] = None,
    target_size: Optional[float] = None,
    load_case: str | LoadCase = "default",
    combination: Optional[str] = None,
    overrides: Optional[Mapping[int, int]] = None,
    progress: Progress = None,
    cancellation_token: Optional["CancellationToken"] = None,
    **solver_options: Any,
):
    """Mesh if needed, build, and run a linear static solve.

    Pass ``combination`` to solve a factored sum of cases instead of a single
    one.
    """

    from anysolver import solve_linear

    from ..post.results import LinearSolution

    built = _resolve_built(
        project, built, mesh=mesh, target_size=target_size, overrides=overrides,
        progress=progress, load_case=load_case, combination=combination,
        allow_quad_first=True,
    )

    _report(progress, "solving")
    structured_progress = solver_options.pop("progress_callback", None)
    displacements, info = solve_linear(
        built.fe_model,
        built.load_case,
        cancellation_token=cancellation_token,
        progress_callback=_step_reporter(
            progress, "static solve", structured_progress
        ),
        **solver_options,
    )
    return LinearSolution(
        displacements=displacements, built=built, info=info, label="static"
    )


def solve_linear_static_many(
    project: Optional[Project] = None,
    *,
    built: Optional[BuiltModel] = None,
    mesh: Optional[Mesh] = None,
    target_size: Optional[float] = None,
    load_cases: Optional[Sequence[str | LoadCase]] = None,
    overrides: Optional[Mapping[int, int]] = None,
    progress: Progress = None,
    cancellation_token: Optional["CancellationToken"] = None,
    load_tolerance: Optional[float] = 1.0e-6,
    **solver_options: Any,
):
    """Solve several unchanged-stiffness cases with one factorization.

    The returned shapes retain their own solver load case, so reactions and
    stress recovery remain case-specific. Geometry, sections, supports,
    masses, imperfections, and affine constraints are assembled only once.

    The response is linear in the loads, and the stiffness is the same for
    every case, so a long time history needs far fewer solves than it has
    cases: the cases' loads (a pressure per element, a force per node) are
    compressed by singular value decomposition, only the dominant load shapes
    are solved, and every case is recombined from them.  Two real cases are
    also solved in full as a check, and further patterns are added until the
    recombined result agrees with them to ``max(10 * load_tolerance, 1e-8)`` of
    the largest displacement.  Few cases, gravity, follower pressure,
    prescribed motion, or compression that would not pay solve every case.
    ``load_tolerance`` (first-tier cut-off on the nodal-force singular values)
    ``None`` or 0 always solves every case.
    """

    from anysolver import solve_linear_many

    from ..post.results import LinearBatchSolution, LinearSolution
    from .build import _build_load_case, _resolve_load_case

    if built is None:
        built = _resolve_built(
            project,
            None,
            mesh=mesh,
            target_size=target_size,
            overrides=overrides,
            progress=progress,
            load_case=None,
            require_loads=True,
            allow_quad_first=True,
        )
    resolved_project = built.project
    # Without an explicit list, solve every case that can be solved: a template
    # that only generates a time series has no time to evaluate its equations.
    requested = tuple(
        load_cases
        or (
            name
            for name, case in resolved_project.load_cases.items()
            if not case.is_template
        )
    )
    if not requested:
        raise ProjectError("a batch linear solve needs at least one load case")

    solver_cases = []
    case_names = []
    for requested_case in requested:
        case = _resolve_load_case(resolved_project, requested_case)
        if case is None:  # pragma: no cover - rejected by the public type
            raise ProjectError("a batch linear solve cannot include an empty case")
        solver_case = _build_load_case(resolved_project, built.mesh, case)
        built.fe_model.add_load_case(solver_case)
        solver_cases.append(solver_case)
        case_names.append(case.name)

    structured_progress = solver_options.pop("progress_callback", None)
    reduced = None
    if load_tolerance:
        reduced = _solve_compressed(
            built.fe_model, built.mesh, solver_cases, float(load_tolerance), progress,
            cancellation_token, structured_progress, solver_options,
        )
    if reduced is not None:
        matrix, info = reduced
    else:
        _report(progress, f"solving {len(solver_cases)} cases with one factorization")
        matrix, info = solve_linear_many(
            built.fe_model,
            solver_cases,
            cancellation_token=cancellation_token,
            progress_callback=_step_reporter(
                progress, "batch static solve", structured_progress
            ),
            **solver_options,
        )
    shapes = []
    for index, (name, solver_case) in enumerate(zip(case_names, solver_cases)):
        case_built = replace(built, load_case=solver_case)
        shapes.append(
            LinearSolution(
                displacements=np.asarray(matrix[:, index], dtype=float),
                built=case_built,
                info={**info, "batch_case_index": index, "batch_case": name},
                label=f"static: {name}",
                value=float(index),
            )
        )
    return LinearBatchSolution(
        built=built,
        shapes=shapes,
        status=str(info.get("status", "unknown")),
        info=info,
        case_names=tuple(case_names),
    )


#: Fewer cases than this are simply all solved.
_COMPRESSION_MIN_CASES = 8


def _compressible(model: Any, cases: Sequence[Any]) -> bool:
    """Whether the cases differ only in nodal forces and dead element pressures."""

    if len(cases) < _COMPRESSION_MIN_CASES:
        return False
    for case in cases:
        if (
            case.gravity is not None
            or case.follower_pressure
            or case.element_loads
            or case.added_node_masses
        ):
            return False
    # A prescribed non-zero motion enters every solution once, not once per mode.
    if any(
        value != 0.0
        for bc in model.boundary_conditions
        for value in getattr(bc, "dof_constraints", {}).values()
    ):
        return False
    return not any(float(equation.rhs) != 0.0 for equation in model.constraint_equations)


def _load_blocks(cases: Sequence[Any]):
    """The cases' loads as two matrices: pressure per element, force per node DOF."""

    pressure_keys = sorted({key for case in cases for key in case.pressure_loads})
    nodal_keys = sorted(
        {
            (node, component)
            for case in cases
            for node, load in case.nodal_loads.items()
            for component, value in enumerate(load)
            if value != 0.0
        }
    )
    pressure_row = {key: row for row, key in enumerate(pressure_keys)}
    nodal_row = {key: row for row, key in enumerate(nodal_keys)}
    pressure = np.zeros((len(pressure_keys), len(cases)))
    nodal = np.zeros((len(nodal_keys), len(cases)))
    for column, case in enumerate(cases):
        for key, value in case.pressure_loads.items():
            pressure[pressure_row[key], column] = value
        for node, load in case.nodal_loads.items():
            for component, value in enumerate(load):
                if value != 0.0:
                    nodal[nodal_row[(node, component)], column] = value
    return pressure_keys, pressure, nodal_keys, nodal


def _modes(matrix: np.ndarray, tolerance: float):
    """Dominant left/right singular vectors of a load block, and the rank kept."""

    if matrix.size == 0 or not np.any(matrix):
        return np.zeros((matrix.shape[0], 0)), np.zeros(0), np.zeros((0, matrix.shape[1])), 0.0
    left, values, right = np.linalg.svd(matrix, full_matrices=False)
    keep = int(np.count_nonzero(values > tolerance * values[0]))
    dropped = float(np.sqrt(np.sum(values[keep:] ** 2)))
    return left[:, :keep], values[:keep], right[:keep], dropped


def _nodal_force_proxy(mesh: Any, pressure_keys: Sequence[int], pressure: np.ndarray,
                       nodal_keys: Sequence[Any], nodal: np.ndarray) -> np.ndarray:
    """The cases' loads as nodal forces: ``G[(node, component), case]``.

    A pressure on an element is the consistent nodal force of a uniform
    traction along the element normal, which ANYfem already lumps for surface
    tractions.  Pressure shapes that cancel between neighbouring elements
    (a checkerboard) leave no nodal force, so ranking *forces* needs far fewer
    shapes than ranking element pressures.  It is only used to choose which
    time patterns matter; the solver still receives real pressures.
    """

    from .build import _element_centroid_normal, _traction_elements_to_nodes

    rows: dict[Any, int] = {}

    def row(key: Any) -> int:
        return rows.setdefault(key, len(rows))

    triplets: list[tuple[int, int, float]] = []
    for column, element_id in enumerate(pressure_keys):
        _, normal = _element_centroid_normal(mesh, int(element_id))
        for node, force in _traction_elements_to_nodes(mesh, (int(element_id),), normal):
            for component in range(3):
                if force[component] != 0.0:
                    triplets.append((row((node, component)), column, float(force[component])))
    nodal_rows = [row(key) for key in nodal_keys]
    proxy = np.zeros((len(rows), pressure.shape[1]))
    if triplets:
        index_rows = np.array([t[0] for t in triplets])
        index_columns = np.array([t[1] for t in triplets])
        values = np.array([t[2] for t in triplets])
        weights = np.zeros((len(rows), len(pressure_keys)))
        np.add.at(weights, (index_rows, index_columns), values)
        proxy = weights @ pressure
    for position, target in enumerate(nodal_rows):
        proxy[target] += nodal[position]
    return proxy


def _synthetic_cases(pressure_keys, pressure, nodal_keys, nodal, patterns: np.ndarray):
    """One solver load case per pattern: the real loads of that combination of steps."""

    from anysolver import LoadCase as SolverCase

    cases = []
    for index, pattern in enumerate(patterns):
        case = SolverCase(name=f"load shape {index}")
        for key, value in zip(pressure_keys, pressure @ pattern):
            if value != 0.0:
                case.pressure_loads[key] = float(value)
        for (node, component), value in zip(nodal_keys, nodal @ pattern):
            if value != 0.0:
                case.nodal_loads.setdefault(node, np.zeros(6))[component] = float(value)
        cases.append(case)
    return cases


def _solve_compressed(
    model: Any,
    mesh: Any,
    cases: Sequence[Any],
    tolerance: float,
    progress: Progress,
    cancellation_token: Any,
    structured_progress: Any,
    solver_options: Mapping[str, Any],
):
    """Solve many unchanged-stiffness cases through a few dominant load shapes.

    The step-to-step patterns that matter are the dominant right singular
    vectors of the cases' nodal-force matrix.  Each is turned into a real load
    case (the same combination of the steps' pressures and nodal loads) and
    solved once, and every step is recombined as the matching combination of
    those solutions.

    Accuracy is judged on displacements, not forces: a dropped force shape can
    excite a flexible local mode far more than its size suggests.  So two real
    cases (the largest load and the middle one) are also solved in full, and
    when the recombined result disagrees with them the next tier of shapes is
    added - reusing everything already solved and the same factorization -
    until they agree or compressing no longer pays.  Returns
    ``(displacement matrix, info)``, or ``None`` when the cases cannot be
    compressed, in which case the caller solves every case.
    """

    from anysolver import FactorizationCache, solve_linear_many

    if not _compressible(model, cases):
        return None
    pressure_keys, pressure, nodal_keys, nodal = _load_blocks(cases)
    try:
        proxy = _nodal_force_proxy(mesh, pressure_keys, pressure, nodal_keys, nodal)
    except Exception:
        return None
    if not np.any(proxy):
        return None
    _, values, right, _ = _modes(proxy, 1.0e-14)
    norm_total = max(float(np.linalg.norm(proxy)), 1.0e-300)

    norms = np.sqrt((pressure**2).sum(axis=0) + (nodal**2).sum(axis=0))
    checks = sorted({int(np.argmax(norms)), len(cases) // 2})
    limit = max(10.0 * tolerance, 1.0e-8)
    options = dict(solver_options)
    cache = options.pop("factorization_cache", None) or FactorizationCache(
        name="load_shapes", max_entries=2
    )

    solved = 0
    shape_solutions = np.zeros((0, 0))
    check_solutions = None
    info: dict[str, Any] = {}
    calls = 0
    level = tolerance
    last = -1
    error = float("inf")
    while level >= 1.0e-14:
        rank = int(np.count_nonzero(values > level * values[0]))
        level *= 1.0e-2
        if rank == last:
            continue
        last = rank
        # Two verification solves are extra, so compress only when it clearly pays.
        if rank + 2 >= len(cases) // 2:
            break
        new = _synthetic_cases(
            pressure_keys, pressure, nodal_keys, nodal, right[solved:rank]
        )
        batch = [*new, *(cases[index] for index in checks)] if check_solutions is None else new
        _report(
            progress,
            f"solving {len(new)} more load shapes ({rank} of {len(cases)} cases) with one "
            "factorization" + ("" if check_solutions is not None else " + 2 checks"),
        )
        matrix, info = solve_linear_many(
            model,
            batch,
            cancellation_token=cancellation_token,
            progress_callback=_step_reporter(progress, "batch static solve", structured_progress),
            factorization_cache=cache,
            **options,
        )
        calls += 1
        if str(info.get("status", "converged")) not in ("ok", "converged", "success"):
            return None
        if check_solutions is None:
            check_solutions = matrix[:, len(new):]
            matrix = matrix[:, : len(new)]
        shape_solutions = (
            matrix if shape_solutions.size == 0 else np.hstack([shape_solutions, matrix])
        )
        solved = rank
        combined = shape_solutions @ right[:rank]
        scale = float(np.max(np.linalg.norm(combined, axis=0)))
        error = max(
            float(np.linalg.norm(combined[:, index] - check_solutions[:, offset]))
            / max(scale, 1.0e-300)
            for offset, index in enumerate(checks)
        )
        if np.isfinite(error) and error <= limit:
            info = dict(info)
            info["load_compression"] = {
                "cases": len(cases),
                "solved_load_shapes": rank,
                "solve_calls": calls,
                "check_cases": [int(index) for index in checks],
                "tolerance": tolerance,
                "dropped_load_norm_relative": float(
                    np.sqrt(np.sum(values[rank:] ** 2)) / norm_total
                ),
                "check_error_relative_to_largest_case": error,
            }
            return combined, info
    _report(
        progress,
        "load compression does not pay for these cases"
        + ("" if not np.isfinite(error) else f" (check error {error:.2e})")
        + "; solving every case",
    )
    return None

# ----------------------------------------------------------------------
# modal
# ----------------------------------------------------------------------
def solve_modal(
    project: Optional[Project] = None,
    *,
    built: Optional[BuiltModel] = None,
    mesh: Optional[Mesh] = None,
    target_size: Optional[float] = None,
    num_modes: int = 6,
    shift: Optional[float] = 0.0,
    overrides: Optional[Mapping[int, int]] = None,
    progress: Progress = None,
    cancellation_token: Optional["CancellationToken"] = None,
    **solver_options: Any,
):
    """Natural frequencies and mode shapes.

    Needs neither loads nor supports: a free-free modal analysis of a floating
    structure is a real case, and the solver handles the rigid-body modes.

    ``shift`` defaults to zero, which puts the sparse eigensolver into
    shift-invert mode.  Without it ARPACK routinely fails to converge on
    ordinary shell models, because the drilling degrees of freedom carry
    almost no mass; leaving that as the out-of-the-box behaviour would be
    indefensible.  Pass ``shift=None`` for the solver's unshifted path.
    """

    from anysolver import solve_free_vibration

    from ..post.results import ModalSolution, ShapeView

    built = _resolve_built(
        project, built, mesh=mesh, target_size=target_size, overrides=overrides,
            progress=progress, load_case=None, require_loads=False,
        require_supports=False,
    )

    _report(progress, f"solving for {num_modes} modes")
    structured_progress = solver_options.pop("progress_callback", None)
    result = solve_free_vibration(
        built.fe_model,
        num_modes=num_modes,
        shift=shift,
        cancellation_token=cancellation_token,
        progress_callback=_step_reporter(
            progress, "modal solve", structured_progress
        ),
        **solver_options,
    )

    shapes = [
        ShapeView(
            displacements=mode.mode_shape,
            built=built,
            label=f"mode {mode.mode_number}",
            value=float(mode.frequency_hz),
        )
        for mode in result.modes
    ]
    return ModalSolution(
        built=built,
        shapes=shapes,
        status=result.solver_status,
        info={"diagnostics": result.diagnostics, "raw": result},
        rigid_body_modes=sum(1 for mode in result.modes if mode.is_rigid_body),
    )


# ----------------------------------------------------------------------
# buckling
# ----------------------------------------------------------------------
def solve_buckling(
    project: Optional[Project] = None,
    *,
    built: Optional[BuiltModel] = None,
    mesh: Optional[Mesh] = None,
    target_size: Optional[float] = None,
    load_case: str = "default",
    combination: Optional[str] = None,
    num_modes: int = 3,
    overrides: Optional[Mapping[int, int]] = None,
    progress: Progress = None,
    cancellation_token: Optional["CancellationToken"] = None,
    **solver_options: Any,
):
    """Elastic buckling factors for a reference action.

    The factor multiplies the reference load case or the prescribed
    displacement path.  Prestress therefore always comes from solving the
    actual affine static problem first.  That solve, and recovery of its
    stresses into element states, are the solver's own -- this only sequences
    them.  A revision-aware analysis session is shared between the static and
    buckling stages so unchanged assembly work can be reused.  Callers may
    supply ``session=...`` and retain ownership of that session.
    """

    from anysolver import (
        AnalysisSession,
        recover_prestress_from_static_result,
        solve_eigenvalue_buckling,
        solve_linear,
    )

    from ..post.results import BucklingSolution, ShapeView

    built = _resolve_built(
        project, built, mesh=mesh, target_size=target_size, overrides=overrides,
        progress=progress, load_case=load_case, combination=combination,
    )

    _report(progress, "solving the reference load case")
    # Buckling is one analysis workflow even though it contains a prerequisite
    # static solve.  Apply an explicitly requested resource policy to both
    # stages; when none was supplied, omit the keyword so each solver retains
    # its established backend defaults.
    reference_options = {}
    if "resource_config" in solver_options:
        reference_options["resource_config"] = solver_options["resource_config"]
    supplied_session = solver_options.pop("session", None)
    analysis_session = (
        AnalysisSession(built.fe_model)
        if supplied_session is None
        else supplied_session
    )
    reference_options["session"] = analysis_session
    structured_progress = solver_options.pop("progress_callback", None)
    try:
        displacements, _info = solve_linear(
            built.fe_model,
            built.load_case,
            cancellation_token=cancellation_token,
            progress_callback=_step_reporter(progress, "buckling reference solve"),
            **reference_options,
        )

        _report(progress, "recovering prestress")
        states, provenance = recover_prestress_from_static_result(
            built.fe_model, displacements
        )

        _report(progress, f"solving for {num_modes} buckling modes")
        result = solve_eigenvalue_buckling(
            built.fe_model,
            element_states=states,
            num_modes=num_modes,
            cancellation_token=cancellation_token,
            progress_callback=_step_reporter(
                progress, "buckling solve", structured_progress
            ),
            session=analysis_session,
            **solver_options,
        )
    finally:
        if supplied_session is None:
            analysis_session.close()

    shapes = [
        ShapeView(
            displacements=mode.mode_shape,
            built=built,
            label=f"mode {mode.mode_number}",
            value=float(mode.load_factor),
        )
        for mode in result.modes
    ]
    reference_name = (
        combination
        or (built.load_case.name if built.load_case is not None else None)
        or "prescribed displacement"
    )
    return BucklingSolution(
        built=built,
        shapes=shapes,
        status=result.solver_status,
        info={
            "prestress": provenance,
            "diagnostics": result.diagnostics,
            "raw": result,
        },
        reference_case=reference_name,
    )


def eigenmode_imperfection(
    buckling_solution,
    mode_number: int = 1,
    amplitude: float = 0.0,
    dof_filter: str = "translations",
):
    """A stress-free imperfection shaped like a buckling mode.

    Pass the result to :func:`solve_nonlinear_static` or
    :func:`solve_arc_length`.  This is deliberately a runtime object rather
    than a project attribute: it only means anything alongside the buckling
    run that produced it.
    """

    from anysolver import imperfection_from_buckling_mode

    raw = buckling_solution.info.get("raw")
    if raw is None:
        raise ProjectError(
            "this buckling result did not retain its solver result, so no "
            "eigenmode imperfection can be built from it"
        )
    return imperfection_from_buckling_mode(
        buckling_solution.built.fe_model,
        raw,
        mode_number=mode_number,
        amplitude=amplitude,
        dof_filter=dof_filter,
    )


def solve_capacity(
    project: Optional[Project] = None,
    *,
    built: Optional[BuiltModel] = None,
    mesh: Optional[Mesh] = None,
    target_size: Optional[float] = None,
    load_case: str | LoadCase = "default",
    combination: Optional[str] = None,
    num_buckling_modes: int = 3,
    buckling_mode_number: int = 1,
    imperfection_amplitude: float = 0.0,
    num_steps: int = 10,
    max_load_factor: float = 1.0,
    config: Any = None,
    resources: Any = None,
    overrides: Optional[Mapping[int, int]] = None,
    progress: Progress = None,
    cancellation_token: Optional["CancellationToken"] = None,
    record_increment_snapshots: bool = False,
    **config_options: Any,
):
    """Static, prestress, buckling, imperfection, then nonlinear collapse.

    Every step of this exists separately in this module and could be chained by
    hand, which is exactly why it is *not* chained by hand here: the solver
    packages the sequence, including the prestress recovery between the static
    and the buckling solve and the mesh-adequacy check on the mode it picks.
    Reimplementing the chain would mean maintaining a second opinion about the
    order of operations.

    ``imperfection_amplitude`` of zero runs the nonlinear stage on the perfect
    shape, which is a different question -- an imperfection is what makes a
    capacity assessment mean anything for a buckling-governed structure, so it
    is worth setting deliberately rather than defaulting to a guess.

    The result is a :class:`CapacitySolution`, which is a nonlinear result
    carrying the buckling stage alongside it, so anything that displays a
    nonlinear solve displays this unchanged.

    Set ``record_increment_snapshots`` to retain the converged displacement
    and committed element state at each nonlinear increment for true playback.
    """

    from anysolver import CapacityWorkflowConfig, run_nonlinear_capacity_workflow

    from ..post.results import BucklingSolution, CapacitySolution, ShapeView

    built = _resolve_built(
        project, built, mesh=mesh, target_size=target_size, overrides=overrides,
        progress=progress, load_case=load_case, combination=combination,
    )
    if config is None:
        config = CapacityWorkflowConfig(
            num_buckling_modes=int(num_buckling_modes),
            buckling_mode_number=int(buckling_mode_number),
            eigenmode_imperfection_amplitude=float(imperfection_amplitude),
            nonlinear_num_steps=int(num_steps),
            nonlinear_max_load_factor=float(max_load_factor),
            nonlinear_resource_config=resources,
            **config_options,
        )

    _report(progress, "running the capacity workflow")
    # The owner config applies resources to nonlinear analysis; scope the
    # native thread limit across its prerequisite static and buckling work too.
    from anysolver.threading_policy import solver_thread_scope
    workflow_resources = getattr(config, "nonlinear_resource_config", resources)
    with solver_thread_scope(workflow_resources) as thread_policy:
        if progress is not None:
            requested = thread_policy.get("requested_threads")
            effective = [pool.get("num_threads") for pool in thread_policy.get("pools_active", ())]
            _report(progress, f"Capacity native solver threads: requested {requested}, effective pools {effective}")
        result = run_nonlinear_capacity_workflow(
            built.fe_model,
            built.load_case,
            config=config,
            status_callback=None if progress is None else progress,
            progress_callback=_capacity_reporter(progress),
            cancellation_token=cancellation_token,
            record_increment_snapshots=bool(record_increment_snapshots),
        )

    reference_name = (
        combination
        or (built.load_case.name if built.load_case is not None else None)
        or "prescribed displacement"
    )
    buckling = BucklingSolution(
        built=built,
        shapes=[
            ShapeView(
                displacements=mode.mode_shape,
                built=built,
                label=f"mode {mode.mode_number}",
                value=float(mode.load_factor),
            )
            for mode in result.buckling_result.modes
        ],
        status=result.buckling_result.solver_status,
        info={"raw": result.buckling_result},
        reference_case=reference_name,
    )
    adequacy = result.mesh_adequacy
    return CapacitySolution(
        displacements=result.nonlinear_result.displacements,
        built=built,
        value=float(result.capacity_factor),
        steps=list(result.nonlinear_result.steps),
        status=result.status,
        info={
            "prestress": result.prestress_summary,
            "diagnostics": result.diagnostics,
            "nonlinear_status": result.nonlinear_result.status,
            "thread_policy": thread_policy,
            "raw": result,
        },
        peak_load_factor=float(result.capacity_factor),
        deleted_elements=_deleted_elements(result.nonlinear_result),
        raw_result=result.nonlinear_result,
        critical_factor=(
            None
            if result.critical_load_factor is None
            else float(result.critical_load_factor)
        ),
        buckling=buckling,
        mesh_adequacy=(
            adequacy.to_dict() if hasattr(adequacy, "to_dict") else {}
        ),
    )


# ----------------------------------------------------------------------
# nonlinear
# ----------------------------------------------------------------------
def solve_nonlinear_static(
    project: Optional[Project] = None,
    *,
    built: Optional[BuiltModel] = None,
    mesh: Optional[Mesh] = None,
    target_size: Optional[float] = None,
    load_case: str | LoadCase = "default",
    combination: Optional[str] = None,
    max_load_factor: float = 1.0,
    num_steps: int = 10,
    imperfection: Any = None,
    fracture: Any = None,
    resources: Any = None,
    overrides: Optional[Mapping[int, int]] = None,
    progress: Progress = None,
    cancellation_token: Optional["CancellationToken"] = None,
    record_increment_snapshots: bool = False,
    **solver_options: Any,
):
    """Incremental geometric and material nonlinear statics.

    Material nonlinearity needs a material that actually yields: build it with
    ``steel(..., nonlinear=True)``.  Without a hardening curve the solve is
    geometrically nonlinear and elastically linear, which is a different
    analysis.

    ``fracture`` takes the solver's ``FractureConfig`` to erode elements that
    exceed a strain measure.  Erosion is residual stiffness scaling after a
    converged increment, not crack mechanics.

    ``resources`` takes a :func:`~anyfem.solve.policy.resource_policy` for
    thread counts, determinism and a memory ceiling.  This is the one analysis
    the solver accepts it on.

    Set ``record_increment_snapshots`` to retain converged displacement and
    committed element-state snapshots for postprocessing and animation.
    """

    from anysolver import solve_static_nonlinear

    from ..post.results import NonlinearSolution

    built = _resolve_built(
        project, built, mesh=mesh, target_size=target_size, overrides=overrides,
        progress=progress, allow_quad_first=True,
        load_case=load_case, combination=combination,
    )

    _report(
        progress,
        "starting adaptive incremental solve: "
        f"{int(num_steps)} initial increment(s), target load factor "
        f"{float(max_load_factor):.5g}; cutbacks/growth may change the final count",
    )
    structured_progress = solver_options.pop("progress_callback", None)
    result = solve_static_nonlinear(
        built.fe_model,
        built.load_case,
        max_load_factor=max_load_factor,
        num_steps=num_steps,
        imperfection=imperfection,
        fracture_config=fracture,
        resource_config=resources,
        status_callback=None if progress is None else progress,
        progress_callback=_step_reporter(
            progress, "converged increment", structured_progress
        ),
        cancellation_token=cancellation_token,
        record_increment_snapshots=bool(record_increment_snapshots),
        **solver_options,
    )
    return NonlinearSolution(
        displacements=result.displacements,
        built=built,
        label="nonlinear",
        value=float(result.load_factor),
        steps=list(result.steps),
        status=result.status,
        info={**dict(result.info), "raw": result},
        deleted_elements=_deleted_elements(result),
        raw_result=result,
    )


def solve_arc_length(
    project: Optional[Project] = None,
    *,
    built: Optional[BuiltModel] = None,
    mesh: Optional[Mesh] = None,
    target_size: Optional[float] = None,
    load_case: str | LoadCase = "default",
    combination: Optional[str] = None,
    control: Any = None,
    imperfection: Any = None,
    overrides: Optional[Mapping[int, int]] = None,
    progress: Progress = None,
    cancellation_token: Optional["CancellationToken"] = None,
    record_increment_snapshots: bool = False,
    **solver_options: Any,
):
    """Arc-length continuation through a limit point.

    Where the incremental solve stops at the peak, this traces past it, so the
    result carries the peak load factor as well as the final state.

    Set ``record_increment_snapshots`` to retain each converged point on the
    continuation path rather than only the final committed state.
    """

    from anysolver import solve_static_arc_length

    from ..post.results import NonlinearSolution

    built = _resolve_built(
        project, built, mesh=mesh, target_size=target_size, overrides=overrides,
        progress=progress, load_case=load_case, combination=combination,
    )
    _report(progress, "tracing the equilibrium path")
    structured_progress = solver_options.pop("progress_callback", None)
    result = solve_static_arc_length(
        built.fe_model,
        built.load_case,
        control=control,
        imperfection=imperfection,
        progress_callback=_step_reporter(
            progress, "arc step", structured_progress
        ),
        cancellation_token=cancellation_token,
        record_increment_snapshots=bool(record_increment_snapshots),
        **solver_options,
    )
    return NonlinearSolution(
        displacements=result.displacements,
        built=built,
        label="arc length",
        value=float(result.load_factor),
        steps=list(result.steps),
        status=result.status,
        info={**dict(result.info), "raw": result},
        peak_load_factor=float(result.peak_load_factor),
        deleted_elements=_deleted_elements(result),
        raw_result=result,
    )


def _step_reporter(
    progress: Progress,
    noun: str,
    structured_progress: Optional[Callable[[Mapping[str, Any]], None]] = None,
):
    """Turn structured solver progress events into short status lines.

    ``ProgressEvent`` deliberately implements ``Mapping`` for compatibility
    with the solver's former dictionary payloads, so this adapter accepts both
    contracts while ANYfem's existing UI-facing callback remains a string
    callback.
    """

    if progress is None:
        return structured_progress

    def report(record: Mapping[str, Any]) -> None:
        # 0.3 keeps numerical workload, path control, increment and Newton
        # iteration as separate concepts.  Prefer those canonical fields over
        # a legacy preformatted message, which could misleadingly render an
        # adaptive increment as e.g. "1460/10 load steps".
        increment = record.get("increment")
        increment_total = record.get("increment_total")
        iteration = record.get("iteration")
        iteration_total = record.get("iteration_total")
        residual = record.get("residual")
        if residual is None:
            residual = record.get("residual_norm")
        control = str(record.get("control") or "").replace("_", " ")
        control_value = record.get("control_value")
        control_target = record.get("control_target")
        legacy_load_factor = False
        if control_value is None and record.get("load_factor") is not None:
            control = control or "load factor"
            control_value = record.get("load_factor")
            legacy_load_factor = True
            if control_target is None:
                control_target = record.get("total")
        work_done = record.get("workload_completed")
        work_total = record.get("workload_total")
        time_s = record.get("time_s")
        status = record.get("status")
        maximum = record.get("max_translation")
        plastic = record.get("max_equivalent_plastic_strain")
        load_increment = record.get("load_increment")
        canonical = any(
            value is not None
            for value in (
                increment,
                iteration,
                residual,
                control_value,
                control_target,
                work_done,
                work_total,
                time_s,
                status,
            )
        )
        if canonical:
            pieces = []
            heading = noun
            if increment is not None:
                # An adaptive algorithm can legitimately exceed the nominal
                # starting increment count.  Only show a denominator when it
                # is a true bound that still contains the current increment.
                bounded_total = (
                    int(increment_total)
                    if increment_total is not None
                    and int(increment_total) >= int(increment)
                    else None
                )
                increment_text = f"{int(increment)}" + (
                    "" if bounded_total is None else f"/{bounded_total}"
                )
                if noun.casefold().endswith("increment"):
                    heading = f"{noun} {increment_text}"
                else:
                    pieces.append(f"increment {increment_text}")
            if control_value is not None:
                control_label = control or "control"
                value_text = f"{control_label} {float(control_value):.5g}"
                if control_target is not None:
                    target_value = float(control_target)
                    value_text += f" / target {target_value:.5g}"
                    if legacy_load_factor and target_value != 0.0:
                        value_text += (
                            f" ({100.0 * float(control_value) / target_value:.1f}%)"
                        )
                pieces.append(value_text)
            if time_s is not None:
                pieces.append(f"t = {float(time_s):.5g} s")
            if iteration is not None:
                iteration_text = f"Newton iteration {int(iteration)}"
                if iteration_total is not None:
                    iteration_text += f"/{int(iteration_total)}"
                pieces.append(iteration_text)
            if residual is not None:
                pieces.append(f"residual {float(residual):.3g}")
            if maximum is not None:
                pieces.append(f"max |u| {float(maximum):.4g} m")
            if plastic is not None:
                pieces.append(f"max PEEQ {float(plastic):.4g}")
            if load_increment is not None:
                pieces.append(f"load increment {float(load_increment):.3g}")
            if status is not None:
                pieces.append(str(status))
            if work_done is not None and work_total not in (None, 0, 0.0):
                pieces.append(
                    f"work {float(work_done):g}/{float(work_total):g}"
                )
            text = f"{heading}: " + "; ".join(pieces)
            progress(_StructuredProgressText(text, record))
            if structured_progress is not None:
                structured_progress(record)
            return

        message = record.get("message")
        if message:
            progress(_StructuredProgressText(str(message), record))
            if structured_progress is not None:
                structured_progress(record)
            return

        index = record.get("step_index", record.get("step"))
        factor = record.get("load_factor")
        time_s = record.get("time_s")
        details = []
        iteration = record.get("iteration", record.get("iterations"))
        residual = record.get("residual_norm")
        maximum = record.get("max_translation")
        plastic = record.get("max_equivalent_plastic_strain")
        load_increment = record.get("load_increment")
        if iteration is not None:
            details.append(f"{int(iteration)} iteration(s)")
        if residual is not None:
            details.append(f"residual {float(residual):.3g}")
        if maximum is not None:
            details.append(f"max |u| {float(maximum):.4g} m")
        if plastic is not None:
            details.append(f"max PEEQ {float(plastic):.4g}")
        if load_increment is not None:
            details.append(f"load increment {float(load_increment):.3g}")
        suffix = f"; {', '.join(details)}" if details else ""
        if index is not None and factor is not None:
            target = record.get("total")
            if target not in (None, 0, 0.0):
                target = float(target)
                factor_text = (
                    f"load factor {float(factor):.4g} / target {target:.4g} "
                    f"({100.0 * float(factor) / target:.1f}%)"
                )
            else:
                factor_text = f"load factor {float(factor):.4g}"
            text = f"{noun} {index}: {factor_text}{suffix}"
        elif index is not None and time_s is not None:
            text = f"{noun} {index}: t = {float(time_s):.4g} s{suffix}"
        elif index is not None:
            text = f"{noun} {index}{suffix}"
        elif record.get("status") is not None:
            text = f"{noun}: {record['status']}"
        elif record.get("fraction") is not None:
            text = f"{noun}: {100.0 * float(record['fraction']):.0f}%"
        else:
            stage = str(record.get("stage", noun)).replace("_", " ")
            text = stage

        progress(_StructuredProgressText(text, record))

        if structured_progress is not None:
            structured_progress(record)

    return report


# ----------------------------------------------------------------------
# transient
# ----------------------------------------------------------------------
def solve_transient(
    project: Optional[Project] = None,
    *,
    dt: float,
    t_end: float,
    built: Optional[BuiltModel] = None,
    mesh: Optional[Mesh] = None,
    target_size: Optional[float] = None,
    load_case: str | LoadCase = "default",
    combination: Optional[str] = None,
    save_every: int = 1,
    rayleigh_alpha: float = 0.0,
    rayleigh_beta: float = 0.0,
    overrides: Optional[Mapping[int, int]] = None,
    progress: Progress = None,
    cancellation_token: Optional["CancellationToken"] = None,
    **config_options: Any,
):
    """Implicit Newmark transient response to a constant load case."""

    from anysolver import TransientConfig, solve_transient_newmark

    from ..post.results import ShapeView, TransientSolution

    built = _resolve_built(
        project, built, mesh=mesh, target_size=target_size, overrides=overrides,
        progress=progress, load_case=load_case, combination=combination,
    )

    config = TransientConfig(
        dt=float(dt),
        t_end=float(t_end),
        save_every=int(save_every),
        rayleigh_alpha=float(rayleigh_alpha),
        rayleigh_beta=float(rayleigh_beta),
        **config_options,
    )

    _report(progress, f"integrating to t = {t_end:g} s")
    result = solve_transient_newmark(
        built.fe_model,
        config,
        base_load_case=built.load_case,
        cancellation_token=cancellation_token,
        progress_callback=_step_reporter(progress, "transient step"),
    )

    times = np.asarray(result.times, dtype=float)
    shapes = [
        ShapeView(
            displacements=result.displacements[index],
            built=built,
            label=f"t = {times[index]:.4g} s",
            value=float(times[index]),
        )
        for index in range(len(times))
    ]
    return TransientSolution(
        built=built,
        shapes=shapes,
        status=result.status,
        info={"diagnostics": result.diagnostics, "raw": result},
        times=times,
        peak_displacement=float(result.peak_displacement),
        peak_node=result.peak_displacement_node,
    )



# ----------------------------------------------------------------------
# impact
# ----------------------------------------------------------------------
class ContactConfigurationError(ProjectError):
    """Raised when a contact setup would not produce a meaningful answer."""


def solve_impact(
    project: Optional[Project] = None,
    *,
    collision,
    built: Optional[BuiltModel] = None,
    mesh: Optional[Mesh] = None,
    target_size: Optional[float] = None,
    load_case: Optional[str] = None,
    dt: Optional[float] = None,
    t_end: Optional[float] = None,
    save_every: int = 1,
    steps_per_contact: float = 20.0,
    steps_per_radius: float = 20.0,
    post_contact_periods: float = 20.0,
    skip_approach: bool = True,
    nonlinear: bool = False,
    damage: Any = None,
    plastic_damage: Any = None,
    fracture: Any = None,
    contact: Any = None,
    rayleigh_alpha: float = 0.0,
    rayleigh_beta: float = 0.0,
    overrides: Optional[Mapping[int, int]] = None,
    strict: bool = True,
    progress: Progress = None,
    cancellation_token: Optional["CancellationToken"] = None,
    **config_options: Any,
):
    """A rigid sphere striking the structure.

    Two settings decide whether the answer means anything, and both have a
    defensible automatic value, so both are computed rather than left at zero:

    * the **contact penalty stiffness** comes from the solver's own
      ``recommend_sphere_contact_penalty`` unless one is supplied;
    * the **time step and duration** resolve one sphere radius into
      ``steps_per_radius`` increments and run ``post_contact_radii`` radii past
      first contact, unless ``dt`` and ``t_end`` are given.

    A third thing decides it too, and this function deliberately does *not*
    take it on: the mesh under the sphere.  A contact patch spread over one
    element cannot produce a peak force that means anything, but refining a
    mapped mesh locally means decomposing the plate, and that is a change to
    the model rather than to the run.  Use the ``RefineForImpact`` command
    beforehand; it is undoable and shows up in the history, where a solve that
    quietly reshaped the geometry would not.

    The solver's contact validation runs before anything is integrated.  With
    ``strict`` its errors stop the run: a badly conditioned contact does not
    fail loudly, it produces a plausible-looking answer, which is worse.
    """

    from anysolver import (
        SphereContactConfig,
        TransientConfig,
        recommend_sphere_contact_penalty,
        solve_transient_sphere_impact,
        validate_contact_configuration,
    )

    from ..model.collision import auto_timing
    from ..post.results import ImpactSolution, ShapeView

    built = _resolve_built(
        project, built, mesh=mesh, target_size=target_size, overrides=overrides,
        progress=progress, load_case=load_case,
        require_loads=load_case is not None, require_supports=True,
    )

    # The penalty comes first, because the time step depends on it: the
    # contact period is 2 pi sqrt(m/k), and a step near that period makes the
    # contact iteration fail rather than merely lose accuracy.
    sphere = collision.to_solver()
    contact_config = contact or SphereContactConfig()
    if contact_config.penalty_stiffness is None:
        _report(progress, "estimating the contact penalty stiffness")
        contact_config = replace(
            contact_config,
            penalty_stiffness=float(
                recommend_sphere_contact_penalty(
                    built.fe_model,
                    sphere,
                    target_penetration_fraction=(
                        contact_config.target_penetration_fraction
                    ),
                    safety_factor=contact_config.auto_penalty_safety_factor,
                )
            ),
        )

    timing = None
    if dt is None or t_end is None:
        _report(progress, "choosing a time step")
        resolution = _contact_resolution(built.mesh, collision)
        timing = auto_timing(
            built.mesh,
            collision,
            penalty_stiffness=contact_config.penalty_stiffness,
            steps_per_contact=steps_per_contact,
            steps_per_radius=steps_per_radius,
            post_contact_periods=post_contact_periods,
            skip_approach=skip_approach,
            wave_speed=_wave_speed(built),
            min_element_size=resolution["smallest_element"],
        )
        dt = timing.dt if dt is None else dt
        t_end = timing.t_end if t_end is None else t_end
        if timing.start is not None:
            # Free flight is exact, so the sphere starts just clear of the
            # structure instead of being integrated across the approach.
            collision = replace(collision, start=timing.start)
            sphere = collision.to_solver()
        _report(progress, timing.summary())

    transient_config = TransientConfig(
        dt=float(dt),
        t_end=float(t_end),
        save_every=int(save_every),
        rayleigh_alpha=float(rayleigh_alpha),
        rayleigh_beta=float(rayleigh_beta),
        **config_options,
    )

    _report(progress, "checking the contact configuration")
    report = validate_contact_configuration(
        built.fe_model, sphere, contact_config, transient_config
    )
    blocking = _contact_errors(report)
    if blocking and strict:
        raise ContactConfigurationError(
            "the contact configuration was rejected before running:\n  - "
            + "\n  - ".join(blocking)
            + "\nPass strict=False to run it anyway."
        )
    for message in blocking:
        _report(progress, f"warning: {message}")

    nonlinear_config = None
    if nonlinear:
        from anysolver import NonlinearTransientConfig

        nonlinear_config = NonlinearTransientConfig(enabled=True)

    _report(progress, f"integrating the impact to t = {t_end:g} s")
    result = solve_transient_sphere_impact(
        built.fe_model,
        transient_config,
        sphere,
        contact_config=contact_config,
        base_load_case=built.load_case,
        damage_config=damage,
        plastic_damage_config=plastic_damage,
        fracture_config=fracture,
        nonlinear_config=nonlinear_config,
        progress_callback=_step_reporter(progress, "impact step"),
        status_callback=None if progress is None else progress,
        cancellation_token=cancellation_token,
    )

    if strict and result.status not in _IMPACT_SUCCESS:
        raise ContactConfigurationError(
            f"the impact run ended as {result.status!r} rather than "
            "completing. Its contact force, absorbed energy and damage are "
            "whatever the integration reached before it gave up, not the "
            f"structure's response.\n{_impact_advice(result.status)}\n"
            "Pass strict=False to inspect the partial result anyway."
        )

    times = np.asarray(result.times, dtype=float)
    shapes = [
        ShapeView(
            displacements=result.displacements[index],
            built=built,
            label=f"t = {times[index]:.4g} s",
            value=float(times[index]),
        )
        for index in range(len(times))
    ]

    return ImpactSolution(
        built=built,
        shapes=shapes,
        status=result.status,
        info={
            "diagnostics": result.diagnostics,
            "contact": contact_config,
            "contact_resolution": _contact_resolution(built.mesh, collision),
            "raw": result,
        },
        times=times,
        sphere_positions=np.asarray(result.sphere_positions, dtype=float),
        sphere_velocities=np.asarray(result.sphere_velocities, dtype=float),
        contact_force_history=np.asarray(
            result.contact_force_history, dtype=float
        ),
        peak_contact_force=float(result.peak_contact_force),
        contact_duration=float(result.contact_duration),
        max_penetration=float(result.max_penetration),
        max_penetration_ratio=float(result.max_penetration_ratio),
        momentum_balance_error=float(result.sphere_momentum_balance_error),
        peak_displacement=float(result.peak_displacement),
        peak_node=result.peak_displacement_node,
        deleted_elements=_deleted_elements(result),
        collision=collision,
        timing=timing,
    )


def _contact_errors(report: Any) -> List[str]:
    """The blocking messages from the solver's contact validation."""

    messages: List[str] = []
    for issue in getattr(report, "issues", ()) or ():
        severity = str(getattr(issue, "severity", "error")).lower()
        if severity in ("error", "critical"):
            code = getattr(issue, "code", "")
            text = getattr(issue, "message", str(issue))
            messages.append(f"[{code}] {text}" if code else text)
    return messages


# Statuses that mean the integration actually ran to the end.  "no_contact" is
# a completed run in which the sphere never touched, which the miss check
# should already have caught; the rest are the solver giving up part-way.
_IMPACT_SUCCESS = frozenset(
    {"completed", "ok", "no_contact", "max_deleted_fraction_reached"}
)


def _impact_advice(status: str) -> str:
    """What to try, for the failures that have a known cause."""

    if status == "contact_iteration_failed":
        return (
            "The contact iteration diverged, which almost always means the "
            "time step is too coarse for the contact. Raise "
            "steps_per_contact, or pass a smaller dt. Lowering the penalty "
            "stiffness will also make it converge, but that changes the "
            "contact rather than resolving it, and the absorbed energy comes "
            "out wrong."
        )
    if status.startswith("nonlinear_"):
        return (
            "The nonlinear iteration failed. Try more increments, or check "
            "that the material and the impact energy are consistent."
        )
    return "Check the solver diagnostics on the result for the cause."


def _contact_resolution(mesh: Mesh, collision: Any) -> Dict[str, float]:
    """How many elements lie across the sphere radius at the contact point.

    Reported whether or not the mesh was refined, because it is the number that
    decides whether a peak contact force means anything: a patch spread over
    one element is a single spring, and its force is a property of the
    discretisation rather than of the structure.
    """

    from ..model.collision import impact_point

    point = impact_point(mesh, collision)
    lengths = []
    for element_id in mesh.shells:
        corners = [mesh.nodes[node] for node in mesh.corners_of(element_id)]
        centre = np.mean(corners, axis=0)
        # Only the contact patch itself: elements a radius away are not what
        # the sphere presses on, and averaging them in would flatter a mesh
        # that is coarse exactly where it matters.
        if np.linalg.norm(centre - point) > collision.radius:
            continue
        lengths.append(
            float(
                np.mean(
                    [
                        np.linalg.norm(second - first)
                        for first, second in zip(corners, corners[1:] + corners[:1])
                    ]
                )
            )
        )
    if not lengths:
        return {
            "element_size": float("nan"),
            "smallest_element": float("nan"),
            "elements_per_radius": 0.0,
        }
    size = float(np.mean(lengths))
    return {
        "element_size": size,
        # The smallest one, which is what sets the stable time step.
        "smallest_element": float(np.min(lengths)),
        "elements_per_radius": float(collision.radius) / size,
    }


def _wave_speed(built: BuiltModel) -> float:
    """The fastest dilatational wave speed in the model, ``sqrt(E / rho)``.

    The fastest one, because the time step has to suit the quickest thing
    happening. Materials without a density contribute nothing rather than an
    infinite speed.

    An orthotropic material has no single ``elastic_modulus``, so its stiffest
    direction is used: reading only the isotropic attribute would quietly
    return zero, and a zero wave speed does not raise anything -- it just drops
    the transit-time bound on the impact step and lets a refined contact
    diverge again.
    """

    speeds = [0.0]
    for material in built.fe_model.materials.values():
        density = float(getattr(material, "density", 0.0) or 0.0)
        if density <= 0.0:
            continue
        modulus = _stiffest_modulus(material)
        if modulus > 0.0:
            speeds.append(float(np.sqrt(modulus / density)))
    return max(speeds)


def _stiffest_modulus(material: Any) -> float:
    """The largest direct elastic modulus a material declares."""

    moduli = [float(getattr(material, "elastic_modulus", 0.0) or 0.0)]
    for axis in (1, 2, 3):
        moduli.append(
            float(getattr(material, f"elastic_modulus_{axis}", 0.0) or 0.0)
        )
    return max(moduli)


def _deleted_elements(result: Any) -> tuple:
    """Element IDs the run removed, whatever shape the record takes."""

    sources = [
        getattr(result, "diagnostics", {}) or {},
        getattr(result, "info", {}) or {},
    ]
    diagnostics = {}
    for source in sources:
        if isinstance(source, Mapping):
            diagnostics.update(source)
    # Where the solver actually puts it: a fracture run reports its erosion
    # under a summary block, not at the top level.
    summary = diagnostics.get("fracture_summary")
    if isinstance(summary, Mapping):
        for key in ("deleted_element_ids", "records"):
            if summary.get(key):
                diagnostics.setdefault("deleted_elements", summary[key])
                break
    for key in ("deleted_elements", "deleted_element_records", "erosion"):
        records = diagnostics.get(key)
        if not records:
            continue
        found = []
        for record in records:
            element_id = getattr(record, "element_id", None)
            if element_id is None and isinstance(record, Mapping):
                element_id = record.get("element_id")
            if element_id is None and isinstance(record, (int, np.integer)):
                element_id = int(record)
            if element_id is not None:
                found.append(int(element_id))
        if found:
            return tuple(sorted(set(found)))
    return ()


# ----------------------------------------------------------------------
def preflight(
    built: BuiltModel,
    *,
    analysis_type: str | None = None,
    load_cases: Optional[Sequence[Any]] = None,
    kinematics: str = "von_karman",
    corotational_tangent: str = "auto",
    allow_free_mechanisms: bool | None = None,
    **validation_options: Any,
):
    """Run the solver's production validation against a built model.

    Returns the solver's own report rather than a paraphrase of it, so the
    scope statements stay the solver's to make.
    """

    from anysolver import validate_production_model

    normalized = (
        None
        if analysis_type is None
        else str(analysis_type).strip().lower().replace(" ", "_")
    )
    if load_cases is None:
        load_cases = () if built.load_case is None else (built.load_case,)
    if allow_free_mechanisms is None:
        allow_free_mechanisms = normalized in ("modal", "free_vibration")
    return validate_production_model(
        built.fe_model,
        load_cases=tuple(load_cases),
        analysis_type=normalized,
        kinematics=kinematics,
        corotational_tangent=corotational_tangent,
        allow_free_mechanisms=bool(allow_free_mechanisms),
        **validation_options,
    )


def _capacity_reporter(progress):
    """Distinguish prerequisite completion from completion of the whole run."""
    if progress is None:return None
    def report(record):
        stage = str(record.get("stage", ""))
        if stage.startswith("linear_static"):
            noun = "Capacity - initial linear prestress solve"
        elif stage.startswith("buckling"):
            noun = "Capacity - buckling modes"
        elif stage.startswith("capacity"):
            noun = "Capacity workflow"
        else:
            noun = "Capacity - nonlinear analysis"
        _step_reporter(progress, noun)(record)
        if stage == "linear_static.complete" and record.get("status") == "converged":
            progress(_StructuredProgressText(
                "Initial linear solve converged. Recovering prestress and solving buckling modes; capacity run is still active.",
                {"stage": "Prestress recovery and buckling modes"}))
        elif stage == "buckling.complete":
            progress(_StructuredProgressText(
                "Buckling stage finished. Checking mode adequacy and preparing nonlinear capacity analysis.",
                {"stage": "Mode checks and nonlinear capacity preparation"}))
    return report
