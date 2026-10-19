"""Result histories as plottable series.

A transient has a displacement against time, an impact has a contact force
against time as well, and an incremental solve has a load factor against
displacement.  They are different analyses but the same *shape* of thing: a
pair of arrays with names and units.  Reducing them to one type here means the
plot widget does not have to know which analysis it is looking at, in the same
way :class:`~anyfem.post.fields.Field` means the contour does not.

Kept headless, so the series can be extracted, tested and written to CSV
without Tk.  The widget that draws it lives in ``anyfem.ui``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from .fields import field_unit

__all__ = ["Series", "history_series", "has_history"]


def _prescribed_control(solution: Any):
    """Return the single non-zero prescribed DOF that drives a path.

    A Euclidean norm over every nodal DOF grows with mesh density and is not
    the engineering control value.  When the submitted model has exactly one
    non-zero prescribed component, use that component for the path axis.
    """

    built = getattr(solution, "built", None)
    project = getattr(built, "project", None)
    candidates = []
    for support in getattr(project, "supports", ()) or ():
        for dof, value in getattr(support, "constraints", {}).items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if abs(numeric) > 1.0e-15:
                candidates.append((str(getattr(support, "name", "support")), str(dof), numeric))
    if len(candidates) != 1:
        return None
    name, dof, value = candidates[0]
    unit = "m" if dof.startswith("u") else "rad"
    return name, dof, value, unit


@dataclass(frozen=True)
class Series:
    """One curve: two equal-length arrays, each with a label and a unit."""

    name: str
    x: np.ndarray
    y: np.ndarray
    x_label: str = ""
    y_label: str = ""
    x_unit: str = ""
    y_unit: str = ""

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError(
                f"series {self.name!r}: {len(self.x)} x values against "
                f"{len(self.y)} y values"
            )

    def __len__(self) -> int:
        return len(self.x)

    @property
    def is_empty(self) -> bool:
        return len(self.x) == 0

    def peak(self) -> tuple:
        """The point of largest magnitude in y, as ``(x, y)``.

        Largest *magnitude*, because the interesting extreme of a deflection or
        a load factor is usually negative.
        """

        if self.is_empty:
            raise ValueError(f"series {self.name!r} is empty")
        index = int(np.argmax(np.abs(self.y)))
        return float(self.x[index]), float(self.y[index])

    def to_rows(self) -> List[tuple]:
        """Rows for a CSV, header included."""

        header = (
            f"{self.x_label} [{self.x_unit}]" if self.x_unit else self.x_label,
            f"{self.y_label} [{self.y_unit}]" if self.y_unit else self.y_label,
        )
        return [header] + [
            (float(x), float(y)) for x, y in zip(self.x, self.y)
        ]


def has_history(solution: Any) -> bool:
    """Whether a result has anything worth plotting against an axis.

    Asks for the same series a caller would get, peak-node trace included.
    Suppressing the node trace here would answer "no history" for a transient,
    whose node trace is the whole point of it.
    """

    return bool(history_series(solution))


def _source_history_series(
    solution: Any,
    *,
    probe: Optional[int] = -1,
    component: str = "uz",
) -> List[Series]:
    """Every curve a result can offer.

    ``probe`` names the node to trace for a time history.  The default of -1
    means "the peak node", which is the one a reader almost always wants and
    the only one the result can choose for itself; pass an explicit node ID for
    another, or None to skip the node trace entirely.

    An analysis with nothing to plot returns an empty list rather than raising.
    Asking whether a result has a history is a reasonable question, and the
    answer "it does not" is not an error.
    """

    series: List[Series] = []
    times = getattr(solution, "times", None)

    if times is not None and len(times) and probe is not None:
        node = getattr(solution, "peak_node", None) if probe == -1 else probe
        if node is not None:
            try:
                values = solution.node_history(int(node), component)
            except (KeyError, ValueError, AttributeError):
                values = None
            if values is not None and len(values) == len(times):
                series.append(
                    Series(
                        name=f"{component} at node {int(node)}",
                        x=np.asarray(times, dtype=float),
                        y=np.asarray(values, dtype=float),
                        x_label="time",
                        y_label=component,
                        x_unit="s",
                        y_unit=field_unit(component),
                    )
                )

    forces = getattr(solution, "contact_force_history", None)
    if times is not None and forces is not None and len(forces) == len(times):
        force_values = np.asarray(forces, dtype=float)
        force_label = "contact force"
        if force_values.ndim > 1:
            force_values = np.linalg.norm(force_values, axis=-1)
            force_label = "contact force magnitude"
        series.append(
            Series(
                name="contact force",
                x=np.asarray(times, dtype=float),
                y=force_values,
                x_label="time",
                y_label=force_label,
                x_unit="s",
                y_unit="N",
            )
        )

    if hasattr(solution, "history") and getattr(solution, "steps", None):
        path = solution.history()
        norms = np.asarray(path.get("displacement_norm", ()), dtype=float)
        path_factors = np.asarray(path.get("load_factor", ()), dtype=float)
        factors = path_factors
        if len(norms) and len(norms) == len(factors):
            control = _prescribed_control(solution)
            x_label = "displacement norm"
            x_unit = "m"
            if control is not None:
                support_name, dof, control_value, x_unit = control
                norms = factors * control_value
                x_label = f"prescribed {support_name} {dof}"
            # The equilibrium path starts from the unloaded reference state.
            # Solver step records intentionally contain converged increments
            # only, so make that physical start point visible in plots without
            # pretending it is a saved contour frame.
            if not (
                np.isclose(norms[0], 0.0, atol=1.0e-14)
                and np.isclose(factors[0], 0.0, atol=1.0e-14)
            ):
                norms = np.concatenate(([0.0], norms))
                factors = np.concatenate(([0.0], factors))
            # Displacement on x and load on y: this is a load-displacement
            # path, and it is read as "how far did it go before it stopped
            # taking more", which puts load on the vertical axis.
            series.append(
                Series(
                    name="load-displacement path",
                    x=norms,
                    y=factors,
                    x_label=x_label,
                    y_label="load factor",
                    x_unit=x_unit,
                    y_unit="",
                )
            )

        if len(path_factors):
            control = _prescribed_control(solution)
            reaction_x = path_factors
            reaction_x_label = "load factor"
            reaction_x_unit = ""
            if control is not None:
                support_name, dof, control_value, reaction_x_unit = control
                reaction_x = path_factors * control_value
                reaction_x_label = f"prescribed {support_name} {dof}"
            for key, item in path.items():
                if not str(key).startswith("support_reaction::"):
                    continue
                values = np.asarray(item, dtype=float).reshape(-1)
                if values.size != path_factors.size:
                    continue
                _prefix, support_name, component = str(key).split("::", 2)
                x_values = reaction_x
                if not (
                    np.isclose(path_factors[0], 0.0, atol=1.0e-14)
                    and np.isclose(values[0], 0.0, atol=1.0e-10)
                ):
                    x_values = np.concatenate(([0.0], reaction_x))
                    values = np.concatenate(([0.0], values))
                label = (
                    "force magnitude"
                    if component == "force_magnitude"
                    else component
                )
                series.append(
                    Series(
                        name=f"Reaction {support_name} {label}",
                        x=x_values,
                        y=values,
                        x_label=reaction_x_label,
                        y_label=f"reaction {support_name} {label}",
                        x_unit=reaction_x_unit,
                        y_unit="N",
                    )
                )

    return series


def history_series(solution, *, probe=-1, component="uz"):
    """Read source history through the default ANYpost provider boundary."""
    from anypost import open_memory_results
    from .access import WORKING_BYTES
    # The source owns history extraction. Its prepared arrays are measured,
    # rather than assigning an arbitrary fraction of the transport budget.
    prepared=_source_history_series(solution,probe=probe,component=component)
    actual_bytes=sum(item.x.nbytes+item.y.nbytes for item in prepared)
    class HistoryProvider:
        provider_work_bytes = actual_bytes
        def metadata(self):
            return {"model_identity": f"anyfem-history:{id(solution)}", "cases": (), "descriptors": ()}
        def read_history(self):
            return [dict(item.__dict__) for item in prepared]
    with open_memory_results(HistoryProvider(), cache_bytes=0, working_bytes=WORKING_BYTES) as reader:
        return [Series(**item) for item in reader.read_history()]
