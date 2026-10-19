"""Interactive working-copy state for one persistent face sketch."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from anygeometry import (
    SketchConstraint,
    SketchDefinition,
    SketchPlane,
    solve_sketch,
)

from .construction import ConstructionMode, ConstructionTask
from .snapping import SnapResult

__all__ = ["FaceSketchTask"]


class FaceSketchTask(ConstructionTask):
    """Polyline clicks plus editable dimensional/coincidence intent."""

    def __init__(
        self,
        plane: SketchPlane,
        snap_tolerance: float,
        definition: SketchDefinition | None = None,
    ) -> None:
        super().__init__(ConstructionMode.POLYLINE, close=True if definition is None else definition.closed)
        self.plane = plane
        self.snap_tolerance = float(snap_tolerance)
        self.point_keys: list[str] = []
        self.constraints: list[SketchConstraint] = []
        if definition is not None:
            solved = solve_sketch(definition, plane)
            for key in definition.path:
                self._points.append(tuple(float(item) for item in plane.world(solved[key])))
                self._snaps.append(None)
                self.point_keys.append(key)
            self.constraints.extend(definition.constraints)

    def _next_key(self) -> str:
        number = 1
        while f"p{number}" in self.point_keys:
            number += 1
        return f"p{number}"

    def add(self, value: Sequence[float] | SnapResult) -> tuple[float, float, float]:
        point = super().add(value)
        key = self._next_key()
        self.point_keys.append(key)
        local = self.plane.local(point)
        vertex_distances = [float(np.linalg.norm(local - item)) for item in self.plane.boundary_vertices]
        if vertex_distances and min(vertex_distances) <= self.snap_tolerance:
            self.constraints.append(
                SketchConstraint("on_vertex", key, boundary_index=int(np.argmin(vertex_distances)))
            )
            return point
        edge_distances: list[float] = []
        for start, end in self.plane.boundary_edges:
            direction = end - start
            length_squared = float(direction @ direction)
            parameter = 0.0 if length_squared <= 0.0 else float(np.clip((local - start) @ direction / length_squared, 0.0, 1.0))
            edge_distances.append(float(np.linalg.norm(local - (start + parameter * direction))))
        if edge_distances and min(edge_distances) <= self.snap_tolerance:
            self.constraints.append(
                SketchConstraint("on_edge", key, boundary_index=int(np.argmin(edge_distances)))
            )
        return point

    def backspace(self) -> bool:
        if not self.point_keys:
            return False
        removed = self.point_keys[-1]
        if not super().backspace():
            return False
        self.point_keys.pop()
        self.constraints[:] = [
            item for item in self.constraints
            if item.first != removed and item.second != removed
        ]
        return True

    def reset(self) -> None:
        super().reset()
        self.point_keys.clear()
        self.constraints.clear()

    def cancel(self) -> None:
        super().cancel()
        self.point_keys.clear()
        self.constraints.clear()

    def add_distance(self, first: int, second: int, value: float) -> SketchConstraint:
        constraint = SketchConstraint(
            "distance", self.point_keys[int(first)], self.point_keys[int(second)], float(value)
        )
        self.constraints.append(constraint)
        return constraint

    def add_coincidence(self, first: int, second: int) -> SketchConstraint:
        constraint = SketchConstraint(
            "coincident", self.point_keys[int(first)], self.point_keys[int(second)]
        )
        self.constraints.append(constraint)
        return constraint

    def remove_last_constraint(self) -> bool:
        if not self.constraints:
            return False
        self.constraints.pop()
        return True

    def solve_preview(self, extrusion: float = 0.0) -> SketchDefinition:
        definition = self.definition(extrusion)
        solved = solve_sketch(definition, self.plane)
        self._points[:] = [
            tuple(float(item) for item in self.plane.world(solved[key]))
            for key in self.point_keys
        ]
        return self.definition(extrusion)

    def definition(self, extrusion: float, *, closed: bool | None = None) -> SketchDefinition:
        if len(self.points) < (3 if (self.close if closed is None else closed) else 2):
            raise ValueError("the sketch does not yet have enough points")
        return SketchDefinition(
            points={
                key: tuple(float(item) for item in self.plane.local(point))
                for key, point in zip(self.point_keys, self.points)
            },
            path=tuple(self.point_keys),
            constraints=tuple(self.constraints),
            closed=self.close if closed is None else bool(closed),
            extrusion=float(extrusion),
        )
