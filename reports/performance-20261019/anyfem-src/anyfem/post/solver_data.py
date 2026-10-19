"""ANYsolver's authoritative quantity resolver at the ANYfem boundary."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np

__all__ = [
    "available_solution_quantities",
    "resolve_solution_quantity",
    "solver_result_candidates",
]


def _unique(values: Iterable[Any]) -> tuple[Any, ...]:
    result = []
    seen: set[int] = set()
    for value in values:
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        result.append(value)
    return tuple(result)


def _safe_getattr(value: Any, name: str, default: Any = None) -> Any:
    """Read an adapter attribute without invoking an empty-view failure.

    ``MultiShapeSolution.displacements`` intentionally delegates to its first
    shape and raises ``IndexError`` when no display shapes were materialized.
    Raw transient/impact arrays are still valid in that situation, so carrier
    discovery must treat the wrapper property as absent rather than aborting
    canonical quantity discovery.
    """

    try:
        return getattr(value, name)
    except (AttributeError, IndexError):
        return default


def _wrapper_carrier(solution: Any, *, quantity_id=None) -> Any:
    """Expose wrapper-only arrays through the solver's attribute contract."""

    info = _safe_getattr(solution, "info")
    info = info if isinstance(info, Mapping) else {}
    shapes = tuple(_safe_getattr(solution, "shapes", ()) or ())
    if quantity_id is not None and quantity_id not in {"displacement", "mode_shape"}:
        displacements = None
    elif shapes:
        vectors = [_safe_getattr(shape, "displacements") for shape in shapes]
        if vectors and all(value is not None for value in vectors):
            displacements = np.stack(
                [np.asarray(value, dtype=float) for value in vectors]
            )
        else:
            displacements = _safe_getattr(solution, "displacements")
    else:
        displacements = _safe_getattr(solution, "displacements")
    return SimpleNamespace(
        displacements=displacements,
        reactions=info.get("reactions"),
        reaction_history=info.get(
            "reaction_history", info.get("reactions_by_case")
        ),
        times=_safe_getattr(solution, "times"),
    )


def solver_result_candidates(
    solution: Any,
    *,
    recovered: Any = None,
) -> tuple[Any, ...]:
    """Ordered raw datasets that may own a canonical solver quantity."""

    info = _safe_getattr(solution, "info")
    raw = info.get("raw") if isinstance(info, Mapping) else None
    direct = _safe_getattr(solution, "raw_result")
    return _unique(
        (
            recovered,
            raw,
            _safe_getattr(raw, "nonlinear_result"),
            direct,
            _safe_getattr(direct, "nonlinear_result"),
            _wrapper_carrier(solution),
        )
    )


def resolve_solution_quantity(
    solution: Any,
    quantity_id: str,
    *,
    recovered: Any = None,
):
    """Resolve one canonical ID through ANYsolver, or fail as unavailable."""

    from anysolver import QuantityUnavailableError, resolve_result_quantity

    key = str(quantity_id)
    info=_safe_getattr(solution,"info")
    raw=info.get("raw") if isinstance(info,Mapping) else None
    direct=_safe_getattr(solution,"raw_result")
    for candidate in _unique((recovered,raw,_safe_getattr(raw,"nonlinear_result"),direct,_safe_getattr(direct,"nonlinear_result"))):
        try:return resolve_result_quantity(candidate,key)
        except QuantityUnavailableError:continue
    try:return resolve_result_quantity(_wrapper_carrier(solution,quantity_id=key),key)
    except QuantityUnavailableError:pass
    raise QuantityUnavailableError(
        f"canonical quantity {key!r} is unavailable on "
        f"{type(solution).__name__}"
    )


def available_solution_quantities(
    solution: Any,
    *,
    recovered: Any = None,
) -> tuple[Any, ...]:
    """Resolve, rather than merely describe, every available canonical ID."""

    from anysolver import (
        QuantityUnavailableError,
        registered_result_quantity_ids,
    )

    available = []
    for key in registered_result_quantity_ids():
        try:
            available.append(
                resolve_solution_quantity(solution, key, recovered=recovered)
            )
        except QuantityUnavailableError:
            continue
    return tuple(available)


def solution_quantity_descriptors(solution):
    """Source-owned descriptor discovery without resolving every quantity.

    No stress recovery is launched and no multishape displacement stack is
    assembled merely to populate the browser.
    """
    from anysolver import describe_result_quantities
    info=_safe_getattr(solution,"info")
    raw=info.get("raw") if isinstance(info,Mapping) else None
    direct=_safe_getattr(solution,"raw_result")
    shapes=tuple(_safe_getattr(solution,"shapes",()) or ())
    candidates=_unique((raw,direct,_safe_getattr(solution,"_stress"),shapes[0] if shapes else solution))
    found={}
    for candidate in candidates:
        for descriptor in describe_result_quantities(candidate):
            found.setdefault(descriptor.quantity_id,descriptor)
    return tuple(found.values())
