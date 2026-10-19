"""A linear six-degree-of-freedom spring between two nodes.

ANYsolver has no public discrete spring, so a flexible connection brings its
own: a diagonal stiffness in global axes, ``F = K (u_b - u_a)`` per component.
It is purely elastic, carries no mass and recovers no stress, so it adds
nothing to modal mass or stress output and behaves the same in every analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from anysolver.elements import Element

__all__ = ["SpringConnectorElement"]


class SpringConnectorElement(Element):
    """Springs ``stiffness`` (N/m for ux-uz, N*m/rad for rx-rz) between two nodes."""

    def __init__(
        self,
        element_id: int,
        node_a: int,
        node_b: int,
        stiffness: Sequence[float],
        material_name: str = "default",
    ):
        super().__init__(element_id, [int(node_a), int(node_b)], material_name)
        values = np.asarray(stiffness, dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("spring stiffness is six finite, non-negative numbers")
        self.stiffness = values

    @property
    def num_nodes(self) -> int:
        return 2

    @property
    def dofs_per_node(self) -> int:
        return 6

    def get_node_coordinates(self, mesh: Any) -> np.ndarray:
        coordinates = []
        for node_id in self.node_ids:
            node = mesh.get_node(node_id)
            if node is None:
                raise ValueError(
                    f"spring element {self.element_id} references missing node {node_id}"
                )
            coordinates.append(node.coords())
        return np.asarray(coordinates, dtype=float)

    def get_dof_mapping(self, mesh: Any) -> List[int]:
        dofs: List[int] = []
        for node_id in self.node_ids:
            dofs.extend(mesh.get_node(node_id).dofs)
        return dofs

    def compute_stiffness_matrix(self, mesh: Any, material: Any) -> np.ndarray:
        diagonal = np.diag(self.stiffness)
        matrix = np.block([[diagonal, -diagonal], [-diagonal, diagonal]])
        self._stiffness_matrix = matrix
        return matrix

    def compute_stresses(
        self,
        mesh: Any,
        displacements: np.ndarray,
        material: Any,
        return_global: bool = False,
    ) -> Dict[str, Any]:
        return {}
