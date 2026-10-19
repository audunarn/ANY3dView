"""Session workplanes resolved from named model coordinate systems.

Workplanes are deliberately not part of the solver model.  A desktop session
may switch grid spacing, snap policy or construction plane without dirtying a
project or invalidating a mesh.  The coordinate-system UUID is kept instead of
copying a basis, so a workplane follows an edited named coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .coordinates import CoordinateSystem

__all__ = ["Workplane", "WorkplaneError", "WorkplaneFrame"]


class WorkplaneError(ValueError):
    """Raised when a construction plane cannot be resolved safely."""


def _finite(value: float, label: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise WorkplaneError(f"{label} must be finite")
    return number


def _point(value: Sequence[float], label: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise WorkplaneError(f"{label} needs three finite components")
    return point.copy()


@dataclass(frozen=True)
class WorkplaneFrame:
    """Resolved orthonormal plane geometry in world coordinates."""

    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    normal: np.ndarray

    def __post_init__(self) -> None:
        origin = _point(self.origin, "workplane origin")
        axes = np.column_stack(
            (
                _point(self.x_axis, "workplane x axis"),
                _point(self.y_axis, "workplane y axis"),
                _point(self.normal, "workplane normal"),
            )
        )
        gram = axes.T @ axes
        if not np.allclose(gram, np.eye(3), atol=1.0e-9, rtol=0.0):
            raise WorkplaneError("workplane axes must be orthonormal")
        if float(np.linalg.det(axes)) <= 1.0e-9:
            raise WorkplaneError("workplane axes must be right-handed")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "x_axis", axes[:, 0].copy())
        object.__setattr__(self, "y_axis", axes[:, 1].copy())
        object.__setattr__(self, "normal", axes[:, 2].copy())

    def plane_coordinates(self, position: Sequence[float]) -> np.ndarray:
        """Return ``(u, v)`` after orthogonally projecting a world point."""

        delta = _point(position, "position") - self.origin
        return np.array(
            (float(delta @ self.x_axis), float(delta @ self.y_axis)),
            dtype=float,
        )

    def normal_distance(self, position: Sequence[float]) -> float:
        return float((_point(position, "position") - self.origin) @ self.normal)

    def world_position(self, coordinates: Sequence[float]) -> np.ndarray:
        local = np.asarray(coordinates, dtype=float)
        if local.shape != (2,) or not np.all(np.isfinite(local)):
            raise WorkplaneError("plane coordinates need two finite components")
        return self.origin + local[0] * self.x_axis + local[1] * self.y_axis

    def project(self, position: Sequence[float]) -> np.ndarray:
        return self.world_position(self.plane_coordinates(position))


@dataclass(frozen=True)
class Workplane:
    """A structural construction plane based on a coordinate-system UUID.

    ``offset`` is measured along the coordinate system's local z axis.  The
    local x/y axes span the plane.  Grid and object-snap switches are session
    preferences; all numerical geometry remains in world SI coordinates.
    """

    coordinate_system_id: str = "global"
    offset: float = 0.0
    grid_spacing: float = 1.0
    snap_tolerance: float = 0.05
    snap_grid: bool = True
    snap_axes: bool = True
    snap_endpoints: bool = True
    snap_midpoints: bool = True
    snap_intersections: bool = True

    def __post_init__(self) -> None:
        identifier = str(self.coordinate_system_id).strip()
        if not identifier:
            raise WorkplaneError("workplane needs a coordinate-system ID")
        offset = _finite(self.offset, "workplane offset")
        spacing = _finite(self.grid_spacing, "grid spacing")
        tolerance = _finite(self.snap_tolerance, "snap tolerance")
        if spacing <= 0.0:
            raise WorkplaneError("grid spacing must be greater than zero")
        if tolerance <= 0.0:
            raise WorkplaneError("snap tolerance must be greater than zero")
        object.__setattr__(self, "coordinate_system_id", identifier)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "grid_spacing", spacing)
        object.__setattr__(self, "snap_tolerance", tolerance)

    def resolve(
        self,
        coordinate_systems: Mapping[str, CoordinateSystem],
    ) -> WorkplaneFrame:
        try:
            system = coordinate_systems[self.coordinate_system_id]
        except KeyError:
            raise WorkplaneError(
                f"coordinate system {self.coordinate_system_id!r} is unavailable"
            ) from None
        basis = np.asarray(system.basis_at(system.origin), dtype=float)
        normal = basis[:, 2]
        origin = np.asarray(system.origin, dtype=float) + self.offset * normal
        return WorkplaneFrame(origin, basis[:, 0], basis[:, 1], normal)

    def with_grid(self, spacing: float) -> "Workplane":
        """Return a validated copy suitable for a compact Details control."""

        from dataclasses import replace

        return replace(self, grid_spacing=spacing)
