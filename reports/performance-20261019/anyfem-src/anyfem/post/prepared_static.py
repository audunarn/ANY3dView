"""Lightweight prepared static result facade over the owned snapshot."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

import numpy as np

__all__ = ["PreparedStaticBlock", "PreparedStaticEnvelope"]


@dataclass(frozen=True)
class PreparedStaticEnvelope:
    """Detached bounded solver extrema plus captured application provenance.

    ``extrema`` carries governing case IDs, times, input indices and associated
    component values. No per-case whole-model results are retained here.
    """

    result: Any
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "provenance", deepcopy(self.provenance))

    @property
    def complete(self):
        return self.result.complete

    @property
    def cases_processed(self):
        return self.result.cases_processed

    @property
    def extrema(self):
        return self.result.extrema

    @property
    def snapshot_identity(self):
        return self.result.snapshot_identity

    @property
    def diagnostics(self):
        return self.result.diagnostics


@dataclass
class PreparedStaticBlock:
    """One certified prepared-static block; only selected results are kept.

    The block exposes the solver's StaticBatch-compatible fields
    (``input_indices``, ``case_ids``, ``times``, ``output_dofs``,
    ``displacements``, ``case_results``, ``diagnostics``) plus the
    application's case labels and snapshot provenance.  It deliberately does
    not hand out the evaluator's owned project, mesh or connections: motion,
    stress and reaction accessors return copies, and the provenance mapping is
    copied on access, so caller-visible metadata can never mutate the
    internal snapshot.  No full solution objects are retained.
    """

    block: Any
    connections: Mapping[str, Any]
    labels: tuple[str, ...] = ()
    node_dofs: Mapping[int, tuple[int, ...]] = field(default_factory=dict, repr=False)
    provenance: Mapping[str, Any] = field(default_factory=dict, repr=False)
    _rows: Mapping[int, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.connections = deepcopy(self.connections)
        self.node_dofs = dict(self.node_dofs)
        self.provenance = deepcopy(self.provenance)
        object.__setattr__(self, "_rows", {
            int(dof): index for index, dof in enumerate(self.output_dofs)
        })

    @property
    def input_indices(self):
        return self.block.input_indices

    @property
    def case_ids(self):
        return self.block.case_ids

    @property
    def times(self):
        return self.block.times

    @property
    def output_dofs(self):
        return self.block.output_dofs

    @property
    def displacements(self):
        return self.block.displacements

    @property
    def case_results(self):
        return self.block.case_results

    @property
    def diagnostics(self):
        return self.block.diagnostics

    @property
    def snapshot_provenance(self) -> Dict[str, Any]:
        """A detached copy of the snapshot provenance."""

        return deepcopy(self.provenance)

    def node_displacement(self, node_id: int, case_index: int = 0) -> np.ndarray:
        dofs = self.node_dofs.get(int(node_id), ())
        if not dofs:
            raise KeyError(f"node {node_id} is not in the structural model")
        rows = self._rows
        if any(int(dof) not in rows for dof in dofs):
            raise KeyError(f"node {node_id} displacement was not selected")
        return self.displacements[[rows[int(dof)] for dof in dofs], case_index].copy()

    def connection_motion(self, connection: str, case_index: int = 0) -> Dict[str, np.ndarray]:
        item = self.connections.get(connection)
        if item is None:
            item = next((value for value in self.connections.values()
                         if value.connection.name == connection), None)
        if item is None:
            raise KeyError(f"connection {connection!r} motion was not selected")
        values = self.node_displacement(item.reference_node, case_index)
        motion = {
            "reference_point": np.asarray(item.reference, dtype=float).copy(),
            "displacement": values[:3], "rotation": values[3:],
        }
        if item.section_node is not None:
            section = self.node_displacement(item.section_node, case_index)
            motion.update(section_displacement=section[:3], section_rotation=section[3:])
        return motion

    def stresses(self, case_index: int = 0):
        """Return the solver's selected element/component/station stress maps."""

        return deepcopy(self.case_results[case_index]["element_stresses"])

    def reactions(self, case_index: int = 0):
        return deepcopy(self.case_results[case_index]["reactions"])
