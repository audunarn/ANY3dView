"""Validated, array-first geometry shared by viewer backends.

Compatible NumPy inputs are retained without a copy.  Callers must therefore
treat arrays as immutable for as long as a backend retains the corresponding
``MeshArrays`` value.  :meth:`MeshArrays.owned_copy` provides an isolated copy
when that lifetime contract is inconvenient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


def _float_array(value: Any, name: str, shape: tuple[Optional[int], ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        array = np.asarray(value, dtype=np.float64)
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape)
    ):
        rendered = ", ".join("*" if item is None else str(item) for item in shape)
        raise ValueError(f"{name} must have shape ({rendered})")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _index_array(value: Any, name: str, width: Optional[int]) -> np.ndarray:
    array = np.asarray(value)
    expected_ndim = 1 if width is None else 2
    if array.ndim != expected_ndim or (width is not None and array.shape[1] != width):
        suffix = "[*]" if width is None else f"[*, {width}]"
        raise ValueError(f"{name} must have shape {suffix}")
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        if array.size and not np.equal(array, np.floor(array)).all():
            raise ValueError(f"{name} must contain integer indices")
    if array.size and (np.min(array) < 0 or np.max(array) > np.iinfo(np.uint32).max):
        raise ValueError(f"{name} contains an index outside the uint32 range")
    return np.ascontiguousarray(array, dtype=np.uint32)


def _optional_ids(value: Any, name: str, length: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must have shape [{length}]")
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        if array.size and not np.equal(array, np.floor(array)).all():
            raise ValueError(f"{name} must contain integer identifiers")
    if array.size and (np.min(array) < 0 or np.max(array) > np.iinfo(np.uint64).max):
        raise ValueError(f"{name} contains an identifier outside the uint64 range")
    return np.ascontiguousarray(array, dtype=np.uint64)


def _optional_scalar(value: Any, name: str, length: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must have shape [{length}]")
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        array = np.asarray(value, dtype=np.float32)
    # NaN is meaningful for result fields; infinity is not.
    if np.isinf(array).any():
        raise ValueError(f"{name} cannot contain infinite values")
    return np.ascontiguousarray(array)


@dataclass(frozen=True, slots=True)
class MeshArrays:
    """One indexed mesh and its optional dynamic engineering fields."""

    positions: np.ndarray
    triangles: np.ndarray
    lines: Optional[np.ndarray] = None
    point_indices: Optional[np.ndarray] = None
    triangle_to_element: Optional[np.ndarray] = None
    node_ids: Optional[np.ndarray] = None
    element_ids: Optional[np.ndarray] = None
    displacements: Optional[np.ndarray] = None
    node_scalars: Optional[np.ndarray] = None
    element_scalars: Optional[np.ndarray] = None
    active_elements: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        positions = _float_array(self.positions, "positions", (None, 3))
        triangles = _index_array(self.triangles, "triangles", 3)
        lines = None if self.lines is None else _index_array(self.lines, "lines", 2)
        points = (
            None
            if self.point_indices is None
            else _index_array(self.point_indices, "point_indices", None)
        )
        node_count = len(positions)
        for name, indices in (
            ("triangles", triangles),
            ("lines", lines),
            ("point_indices", points),
        ):
            if indices is not None and indices.size and int(indices.max()) >= node_count:
                raise ValueError(f"{name} references a missing position")

        mapping = (
            None
            if self.triangle_to_element is None
            else _index_array(self.triangle_to_element, "triangle_to_element", None)
        )
        if mapping is not None and len(mapping) != len(triangles):
            raise ValueError("triangle_to_element must have one value per triangle")

        explicit_lengths = [
            len(value)
            for value in (self.element_ids, self.element_scalars, self.active_elements)
            if value is not None
        ]
        if mapping is not None and mapping.size:
            element_count = int(mapping.max()) + 1
            if explicit_lengths and any(length != element_count for length in explicit_lengths):
                raise ValueError("element fields do not match triangle_to_element")
        elif explicit_lengths:
            element_count = explicit_lengths[0]
            if any(length != element_count for length in explicit_lengths):
                raise ValueError("element fields must have the same length")
            if mapping is None and element_count != len(triangles):
                raise ValueError(
                    "triangle_to_element is required when element count differs from triangle count"
                )
        else:
            element_count = len(triangles)

        node_ids = _optional_ids(self.node_ids, "node_ids", node_count)
        element_ids = _optional_ids(self.element_ids, "element_ids", element_count)
        displacements = (
            None
            if self.displacements is None
            else _float_array(self.displacements, "displacements", (node_count, 3))
        )
        node_scalars = _optional_scalar(self.node_scalars, "node_scalars", node_count)
        element_scalars = _optional_scalar(
            self.element_scalars, "element_scalars", element_count
        )
        active = None
        if self.active_elements is not None:
            active = np.asarray(self.active_elements)
            if active.ndim != 1 or len(active) != element_count:
                raise ValueError(f"active_elements must have shape [{element_count}]")
            active = np.ascontiguousarray(active, dtype=np.bool_)

        for name, value in (
            ("positions", positions),
            ("triangles", triangles),
            ("lines", lines),
            ("point_indices", points),
            ("triangle_to_element", mapping),
            ("node_ids", node_ids),
            ("element_ids", element_ids),
            ("displacements", displacements),
            ("node_scalars", node_scalars),
            ("element_scalars", element_scalars),
            ("active_elements", active),
        ):
            object.__setattr__(self, name, value)

    @property
    def node_count(self) -> int:
        return len(self.positions)

    @property
    def triangle_count(self) -> int:
        return len(self.triangles)

    @property
    def element_count(self) -> int:
        if self.triangle_to_element is not None and self.triangle_to_element.size:
            return int(self.triangle_to_element.max()) + 1
        for value in (self.element_ids, self.element_scalars, self.active_elements):
            if value is not None:
                return len(value)
        return self.triangle_count

    def owned_copy(self) -> "MeshArrays":
        """Return an isolated, C-contiguous copy of every supplied array."""

        values = {
            field: None if (value := getattr(self, field)) is None else value.copy(order="C")
            for field in self.__dataclass_fields__
        }
        return MeshArrays(**values)

    @classmethod
    def from_mesh(cls, mesh: Any) -> "MeshArrays":
        """Triangulate an object-oriented ``Mesh`` once at the backend boundary."""

        positions = np.asarray(mesh.vertices, dtype=np.float64)
        triangles: list[tuple[int, int, int]] = []
        for face in mesh.faces:
            if len(face) < 3:
                continue
            first = int(face[0])
            triangles.extend(
                (first, int(face[index]), int(face[index + 1]))
                for index in range(1, len(face) - 1)
            )
        return cls(positions, np.asarray(triangles, dtype=np.uint32).reshape((-1, 3)))
