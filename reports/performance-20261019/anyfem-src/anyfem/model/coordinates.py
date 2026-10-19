"""Named Cartesian and cylindrical coordinate systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence
from uuid import uuid4

import numpy as np

__all__ = ["CoordinateSystem", "CoordinateSystemError", "GLOBAL_COORDINATES"]


class CoordinateSystemError(ValueError):
    """Raised for a degenerate or inconsistent coordinate system."""


def _vector(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise CoordinateSystemError(f"{name} needs three finite components")
    return vector.copy()


def _unit(value: Sequence[float], name: str) -> np.ndarray:
    vector = _vector(value, name)
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-12:
        raise CoordinateSystemError(f"{name} must be non-zero")
    return vector / length


@dataclass(frozen=True)
class CoordinateSystem:
    """A named right-handed basis.

    The returned basis matrix stores local axes as columns, so
    ``basis_at(point) @ local_vector`` converts a local component vector to
    global components.
    """

    name: str
    kind: Literal["cartesian", "cylindrical"] = "cartesian"
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis: np.ndarray = field(default_factory=lambda: np.array((0.0, 0.0, 1.0)))
    reference: np.ndarray = field(default_factory=lambda: np.array((1.0, 0.0, 0.0)))
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if self.kind not in ("cartesian", "cylindrical"):
            raise CoordinateSystemError(
                f"unknown coordinate-system kind {self.kind!r}"
            )
        if not str(self.name).strip():
            raise CoordinateSystemError("coordinate system needs a name")
        origin = _vector(self.origin, "origin")
        axis = _unit(self.axis, "axis")
        reference = _unit(self.reference, "reference direction")
        projection = reference - float(np.dot(reference, axis)) * axis
        length = float(np.linalg.norm(projection))
        if length <= 1.0e-10:
            raise CoordinateSystemError(
                "reference direction must not be parallel to the axis"
            )
        reference = projection / length
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "id", str(self.id))

    def basis_at(self, position: Sequence[float] | None = None) -> np.ndarray:
        if self.kind == "cartesian":
            x_axis = self.reference
            z_axis = self.axis
        else:
            if position is None:
                x_axis = self.reference
            else:
                radial = _vector(position, "position") - self.origin
                radial = radial - float(np.dot(radial, self.axis)) * self.axis
                length = float(np.linalg.norm(radial))
                # At the cylindrical axis the radial direction is undefined;
                # use the declared reference rather than returning NaNs.
                x_axis = self.reference if length <= 1.0e-12 else radial / length
            z_axis = self.axis
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        return np.column_stack((x_axis, y_axis, z_axis))

    def to_global(self, vector: Sequence[float], position=None) -> np.ndarray:
        return self.basis_at(position) @ _vector(vector, "local vector")

    def to_local(self, vector: Sequence[float], position=None) -> np.ndarray:
        return self.basis_at(position).T @ _vector(vector, "global vector")

    def point_to_global(self, coordinates: Sequence[float]) -> np.ndarray:
        local = _vector(coordinates, "local coordinates")
        if self.kind == "cartesian":
            return self.origin + self.basis_at() @ local
        radius, angle, axial = local
        basis = self.basis_at()
        return (
            self.origin
            + radius * np.cos(angle) * basis[:, 0]
            + radius * np.sin(angle) * basis[:, 1]
            + axial * basis[:, 2]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "origin": self.origin.tolist(),
            "axis": self.axis.tolist(),
            "reference": self.reference.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CoordinateSystem":
        return cls(
            id=str(data.get("id", uuid4())),
            name=str(data.get("name", "Coordinates")),
            kind=str(data.get("kind", "cartesian")),  # type: ignore[arg-type]
            origin=data.get("origin", (0.0, 0.0, 0.0)),  # type: ignore[arg-type]
            axis=data.get("axis", (0.0, 0.0, 1.0)),  # type: ignore[arg-type]
            reference=data.get("reference", (1.0, 0.0, 0.0)),  # type: ignore[arg-type]
        )


GLOBAL_COORDINATES = CoordinateSystem(name="Global", id="global")
