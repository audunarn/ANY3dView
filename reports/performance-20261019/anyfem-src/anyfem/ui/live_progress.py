"""Bounded live-solve chart data, independent of Tk.

Newton status callbacks remain human-readable log lines while converged
increments may arrive as structured progress mappings.  This adapter accepts
both contracts so the monitor can graph progress without coupling widgets to
the numerical solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping

import numpy as np

from ..post.history import Series

__all__ = ["GRAPH_CHOICES", "LiveProgressData"]


GRAPH_CHOICES = (
    "Newton residual (current trial)",
    "Load factor path",
    "Maximum displacement path",
    "Maximum PEEQ path",
    "Adaptive load increment",
    "Largest support reaction force",
)

_ITERATION = re.compile(
    r"Increment trial\s+(?P<trial>\d+)\s*\|\s*"
    r"load factor\s+(?P<load>[-+0-9.eE]+)\s*/\s*(?P<target>[-+0-9.eE]+)\s*\|\s*"
    r"increment\s+(?P<increment>[-+0-9.eE]+)\s*\|\s*"
    r"Newton iteration\s+(?P<iteration>\d+)\s*\|\s*"
    r"residual\s+(?P<residual>[-+0-9.eE]+)",
    re.IGNORECASE,
)
_CONVERGED = re.compile(
    r"converged increment\s+(?P<increment_index>\d+)\s*:\s*"
    r"load factor\s+(?P<load>[-+0-9.eE]+)",
    re.IGNORECASE,
)


def _mapping_value(value: Any, name: str, default=None):
    attribute = getattr(value, name, None)
    if attribute is not None:
        return attribute
    getter = getattr(value, "get", None)
    return getter(name, default) if callable(getter) else default


def _token(text: str, pattern: str):
    match = re.search(pattern, text, re.IGNORECASE)
    return None if match is None else float(match.group(1))


@dataclass
class LiveProgressData:
    """Small in-memory histories used by the live monitor.

    Only the current Newton trial retains per-iteration residuals.  Converged
    paths are capped so a pathological adaptive run cannot make Tk consume
    unbounded memory; evenly removing old interior points preserves the start
    and newest engineering state.
    """

    max_path_points: int = 10_000
    trial_key: tuple[int, float] | None = None
    trial_iterations: list[int] = field(default_factory=list)
    trial_residuals: list[float] = field(default_factory=list)
    increments: list[int] = field(default_factory=list)
    load_factors: list[float] = field(default_factory=list)
    max_displacements: list[float] = field(default_factory=list)
    max_peeq: list[float] = field(default_factory=list)
    load_increments: list[float] = field(default_factory=list)
    reaction_force_max: list[float] = field(default_factory=list)
    support_reactions: dict[str, list[np.ndarray]] = field(default_factory=dict)
    target_load_factor: float | None = None

    def clear(self) -> None:
        self.trial_key = None
        self.trial_iterations.clear()
        self.trial_residuals.clear()
        self.increments.clear()
        self.load_factors.clear()
        self.max_displacements.clear()
        self.max_peeq.clear()
        self.load_increments.clear()
        self.reaction_force_max.clear()
        self.support_reactions.clear()
        self.target_load_factor = None

    def ingest(self, text: str, payload: Any = None) -> bool:
        """Consume one progress notification; return whether graph data changed."""

        line = str(text or "").strip()
        changed = self._ingest_mapping(payload)
        iteration = _ITERATION.search(line)
        if iteration is not None:
            trial = int(iteration.group("trial"))
            load = float(iteration.group("load"))
            key = (trial, load)
            if key != self.trial_key:
                self.trial_key = key
                self.trial_iterations.clear()
                self.trial_residuals.clear()
            self.target_load_factor = float(iteration.group("target"))
            number = int(iteration.group("iteration"))
            residual = float(iteration.group("residual"))
            if self.trial_iterations and self.trial_iterations[-1] == number:
                self.trial_residuals[-1] = residual
            else:
                self.trial_iterations.append(number)
                self.trial_residuals.append(residual)
            changed = True

        converged = _CONVERGED.search(line)
        if converged is not None:
            index = int(converged.group("increment_index"))
            load = float(converged.group("load"))
            changed = self._append_converged(
                index,
                load,
                displacement=_token(line, r"max\s*\|u\|\s*([-+0-9.eE]+)"),
                peeq=_token(line, r"max\s+PEEQ\s+([-+0-9.eE]+)"),
                load_increment=_token(line, r"load increment\s+([-+0-9.eE]+)"),
            ) or changed
        return changed

    def _ingest_mapping(self, payload: Any) -> bool:
        if payload is None:
            return False
        if isinstance(payload, str) and not callable(getattr(payload, "get", None)):
            return False
        event_type = str(_mapping_value(payload, "event_type", _mapping_value(payload, "type", "")))
        index = _mapping_value(payload, "step_index")
        load = _mapping_value(payload, "load_factor")
        if index is None or load is None or "step" not in event_type:
            return False
        total = _mapping_value(payload, "total")
        if total not in (None, 0, 0.0):
            self.target_load_factor = float(total)
        return self._append_converged(
            int(index),
            float(load),
            displacement=_mapping_value(payload, "max_translation"),
            peeq=_mapping_value(payload, "max_equivalent_plastic_strain"),
            load_increment=_mapping_value(payload, "load_increment"),
            support_reactions=_mapping_value(payload, "support_reactions"),
        )

    def _append_converged(
        self,
        index: int,
        load: float,
        *,
        displacement=None,
        peeq=None,
        load_increment=None,
        support_reactions=None,
    ) -> bool:
        reaction_rows = support_reactions if isinstance(support_reactions, Mapping) else {}
        reaction_max = max(
            (
                float(np.linalg.norm(np.asarray(values, dtype=float).reshape(-1)[:3]))
                for values in reaction_rows.values()
            ),
            default=np.nan,
        )
        values = (
            float(displacement) if displacement is not None else np.nan,
            float(peeq) if peeq is not None else np.nan,
            float(load_increment) if load_increment is not None else np.nan,
            reaction_max,
        )
        if self.increments and self.increments[-1] == int(index):
            self.load_factors[-1] = float(load)
            for target, value in zip(
                (
                    self.max_displacements,
                    self.max_peeq,
                    self.load_increments,
                    self.reaction_force_max,
                ),
                values,
            ):
                if np.isfinite(value):
                    target[-1] = value
            self._update_support_reactions(reaction_rows, replace_last=True)
            return True
        self.increments.append(int(index))
        self.load_factors.append(float(load))
        self.max_displacements.append(values[0])
        self.max_peeq.append(values[1])
        self.load_increments.append(values[2])
        self.reaction_force_max.append(values[3])
        self._update_support_reactions(reaction_rows, replace_last=False)
        self._bound_paths()
        return True

    def _update_support_reactions(
        self, rows: Mapping[str, Any], *, replace_last: bool
    ) -> None:
        target_count = len(self.increments)
        for history in self.support_reactions.values():
            if not replace_last:
                history.append(np.full(6, np.nan, dtype=float))
        for raw_name, raw_values in rows.items():
            name = str(raw_name)
            history = self.support_reactions.get(name)
            if history is None:
                history = [np.full(6, np.nan, dtype=float) for _ in range(target_count)]
                self.support_reactions[name] = history
            elif len(history) < target_count:
                history.extend(
                    np.full(6, np.nan, dtype=float)
                    for _ in range(target_count - len(history))
                )
            values = np.asarray(raw_values, dtype=float).reshape(-1)
            if values.size >= 6 and history:
                history[-1] = values[:6].copy()

    def _bound_paths(self) -> None:
        if len(self.increments) <= self.max_path_points:
            return
        # Preserve the starting point and newest half while thinning older
        # interior values deterministically.
        keep = [0] + list(range(2, len(self.increments) // 2, 2)) + list(
            range(len(self.increments) // 2, len(self.increments))
        )
        keep = keep[-self.max_path_points :]
        for values in (
            self.increments,
            self.load_factors,
            self.max_displacements,
            self.max_peeq,
            self.load_increments,
            self.reaction_force_max,
        ):
            values[:] = [values[index] for index in keep]
        for values in self.support_reactions.values():
            values[:] = [values[index] for index in keep]

    @property
    def graph_choices(self) -> tuple[str, ...]:
        dynamic = []
        for name in sorted(self.support_reactions):
            dynamic.extend(
                f"Reaction: {name} | {component}"
                for component in ("force magnitude", "Fx", "Fy", "Fz")
            )
        return tuple(GRAPH_CHOICES) + tuple(dynamic)

    def series(self, choice: str) -> Series | None:
        if choice == GRAPH_CHOICES[0]:
            if not self.trial_iterations:
                return None
            residuals = np.maximum(np.asarray(self.trial_residuals, dtype=float), 1.0e-300)
            return Series(
                choice,
                np.asarray(self.trial_iterations, dtype=float),
                np.log10(residuals),
                "Newton iteration",
                "log10 residual norm",
                "",
                "",
            )
        mapping = {
            GRAPH_CHOICES[1]: (self.load_factors, "load factor", ""),
            GRAPH_CHOICES[2]: (self.max_displacements, "maximum translation", "m"),
            GRAPH_CHOICES[3]: (self.max_peeq, "maximum PEEQ", "1"),
            GRAPH_CHOICES[4]: (self.load_increments, "load increment", ""),
            GRAPH_CHOICES[5]: (
                self.reaction_force_max,
                "largest support reaction force",
                "N",
            ),
        }
        if choice.startswith("Reaction: ") and " | " in choice:
            name, component = choice[len("Reaction: ") :].rsplit(" | ", 1)
            rows = self.support_reactions.get(name, ())
            if not rows:
                return None
            matrix = np.asarray(rows, dtype=float)
            if component == "force magnitude":
                values = np.linalg.norm(matrix[:, :3], axis=1)
            else:
                component_index = {"Fx": 0, "Fy": 1, "Fz": 2}.get(component)
                if component_index is None:
                    return None
                values = matrix[:, component_index]
            mapping[choice] = (values, f"{name} {component}", "N")
        values, label, unit = mapping.get(choice, ((), "", ""))
        if not self.increments:
            return None
        x = np.asarray(self.increments, dtype=float)
        y = np.asarray(values, dtype=float)
        finite = np.isfinite(y)
        if not np.any(finite):
            return None
        return Series(choice, x[finite], y[finite], "converged increment", label, "", unit)

    def caption(self, choice: str) -> str:
        if choice == GRAPH_CHOICES[0] and self.trial_key is not None:
            trial, load = self.trial_key
            target = "?" if self.target_load_factor is None else f"{self.target_load_factor:.5g}"
            return f"Trial increment {trial} • load factor {load:.5g} / {target}"
        if self.increments:
            return (
                f"{len(self.increments)} retained converged points • latest increment "
                f"{self.increments[-1]} • load factor {self.load_factors[-1]:.5g}"
            )
        return "Waiting for solver progress…"
