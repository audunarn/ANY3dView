"""Headless engineering summaries for the Results workspace.

The GUI consumes these immutable values, but keeping the interpretation out
of Tk makes nonlinear termination semantics directly testable.  In
particular, ``stopped_at_limit`` is not presented as a completed target run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional


_STOP_REASONS = {
    "target_load_factor_reached": "Requested load target reached",
    "target_displacement_reached": "Requested displacement target reached",
    "minimum_increment_reached": "Minimum load increment reached after repeated cutbacks",
    "minimum_load_increment_reached": "Minimum load increment reached after repeated cutbacks",
    "minimum_step_size_reached": "Minimum load increment reached after repeated cutbacks",
    "max_iterations_reached": "Maximum equilibrium iterations reached",
    "maximum_iterations_reached": "Maximum equilibrium iterations reached",
    "nonfinite_residual": "Non-finite residual encountered",
    "singular_tangent": "Tangent stiffness became singular",
    "load_factor_limit_reached": "Configured load-factor limit reached",
    "post_peak_drop_reached": "Configured post-peak drop reached",
    "empty_reduced_system": "No unconstrained degrees of freedom remain",
}

_FAILURE_DETAILS = {
    "line_search_failed": (
        "the Newton line search could not find a correction that reduced the residual"
    ),
    "maximum_iterations_reached": "the Newton iteration limit was reached",
    "singular_tangent_factorization": "the tangent stiffness could not be factorized",
    "nonfinite_residual": "the equilibrium residual became non-finite",
}


def _humanize(value: Any) -> str:
    text = str(value or "").strip()
    return _STOP_REASONS.get(text, text.replace("_", " ").strip().capitalize())


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


@dataclass(frozen=True)
class NonlinearPathSummary:
    """What began, what converged, and why the nonlinear path ended."""

    status: str
    severity: str
    start_load_factor: float
    first_converged_load_factor: Optional[float]
    peak_load_factor: Optional[float]
    last_converged_load_factor: Optional[float]
    target_load_factor: Optional[float]
    first_failed_load_factor: Optional[float]
    stop_reason: str
    failed_iteration_reason: str
    converged_steps: int
    total_iterations: int
    max_peeq: Optional[float]
    saved_increments: int

    @property
    def progress_fraction(self) -> Optional[float]:
        if (
            self.last_converged_load_factor is None
            or self.target_load_factor is None
            or self.target_load_factor == 0.0
        ):
            return None
        return max(0.0, min(1.0, self.last_converged_load_factor / self.target_load_factor))


def nonlinear_path_summary(
    solution: Any, *, target_load_factor: Optional[float] = None
) -> Optional[NonlinearPathSummary]:
    """Describe an incremental solution, returning ``None`` for other results."""

    steps = tuple(getattr(solution, "steps", ()) or ())
    raw = getattr(solution, "raw_result", None)
    if not steps and raw is None:
        return None
    info = _mapping(getattr(raw, "info", None)) or _mapping(
        getattr(solution, "info", None)
    )
    # ANYfem's wrapper also retains the solver object under info['raw'].
    nested_raw = info.get("raw")
    if nested_raw is not None and nested_raw is not raw:
        raw = nested_raw
        info = _mapping(getattr(raw, "info", None)) or info

    status = str(getattr(solution, "status", getattr(raw, "status", "available")))
    category = str(info.get("status_category", status)).casefold()
    stop_key = info.get("stop_reason") or info.get("failure_reason") or status
    completed = status.casefold() in {"completed", "converged", "ok"}
    severity = "success" if completed else (
        "warning" if category in {"stopped", "limit", "partial", "stopped_at_limit"}
        or status.casefold() == "stopped_at_limit" else "error"
    )

    result_case = _mapping(info.get("result_case"))
    settings = _mapping(result_case.get("settings"))
    target = _number(settings.get("max_load_factor"))
    if target is None:
        target = _number(target_load_factor)
    first = _number(getattr(steps[0], "load_factor", None)) if steps else None
    last = _number(
        info.get(
            "last_converged_load_factor",
            getattr(solution, "load_factor", getattr(solution, "value", None)),
        )
    )
    peak = _number(
        info.get("peak_load_factor", getattr(solution, "peak_load_factor", None))
    )
    if peak is None and steps:
        peak = max(float(getattr(step, "load_factor", 0.0)) for step in steps)
    peeq_values = [
        _number(getattr(step, "max_equivalent_plastic_strain", None))
        for step in steps
    ]
    peeq = max((value for value in peeq_values if value is not None), default=None)
    iterations = int(
        info.get(
            "total_newton_iterations",
            sum(int(getattr(step, "iterations", 0) or 0) for step in steps),
        )
        or 0
    )
    snapshots = tuple(getattr(raw, "snapshots", ()) or ())
    return NonlinearPathSummary(
        status=status,
        severity=severity,
        start_load_factor=0.0,
        first_converged_load_factor=first,
        peak_load_factor=peak,
        last_converged_load_factor=last,
        target_load_factor=target,
        first_failed_load_factor=_number(info.get("first_failed_load_factor")),
        stop_reason=_humanize(stop_key) or "Termination reason unavailable",
        failed_iteration_reason=_FAILURE_DETAILS.get(
            str(info.get("first_failed_iteration_reason") or "").strip(),
            _humanize(info.get("first_failed_iteration_reason")),
        ),
        converged_steps=len(steps),
        total_iterations=iterations,
        max_peeq=peeq,
        saved_increments=len(snapshots),
    )


def submitted_input_data(value: Any) -> Mapping[str, Any]:
    """Return one submitted-input report as a mapping, failing harmlessly."""

    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def submitted_target_load_factor(value: Any) -> Optional[float]:
    """Read the requested nonlinear target from immutable submitted inputs."""

    data = submitted_input_data(value)
    analysis = _mapping(data.get("analysis"))
    settings = _mapping(analysis.get("settings"))
    submitted = _mapping(data.get("submitted_options"))
    return _number(
        settings.get("max_load_factor", submitted.get("max_load_factor"))
    )


def prescribed_path_progress(
    value: Any,
    *,
    last_load_factor: Optional[float],
    target_load_factor: Optional[float],
) -> tuple[str, ...]:
    """Describe non-zero prescribed DOFs at the last and requested factors.

    Constraint values in the submitted report are the values at load factor
    one.  Showing their scaled engineering values makes a load factor tangible
    for displacement-driven analyses.
    """

    if last_load_factor is None or target_load_factor is None:
        return ()
    data = submitted_input_data(value)
    rows: list[str] = []
    for support in data.get("supports", ()) or ():
        if not isinstance(support, Mapping):
            continue
        name = str(support.get("name") or "prescribed support")
        engineering = _mapping(support.get("constraints_engineering"))
        for dof, specification in engineering.items():
            item = _mapping(specification)
            base = _number(item.get("value"))
            if base is None or abs(base) <= 1.0e-15:
                continue
            unit = str(item.get("unit") or "SI")
            rows.append(
                f"{name} {dof}: {base * last_load_factor:.5g} / "
                f"{base * target_load_factor:.5g} {unit} (last / requested)"
            )
    return tuple(rows)
