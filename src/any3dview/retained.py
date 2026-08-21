"""Backend-neutral retained mesh handles and dirty generations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import get_ident
from typing import Callable, Hashable, Optional, Protocol

import numpy as np

from .arrays import MeshArrays
from .ownership import PackedOwnerTable


@dataclass(frozen=True, slots=True)
class DirtyGenerations:
    topology: int = 0
    position: int = 0
    displacement: int = 0
    scalar: int = 0
    active: int = 0
    selection: int = 0
    appearance: int = 0
    transform: int = 0


ChangeCallback = Callable[["MeshHandle", str], None]


class RetainedViewer(Protocol):
    def add_mesh_arrays(self, mesh: MeshArrays, **options: object) -> "MeshHandle": ...


class MeshHandle:
    """Mutable retained state with immutable, validated array snapshots."""

    __slots__ = (
        "_mesh",
        "_chunks",
        "_chunk_ownership",
        "_callback",
        "_owner_thread",
        "_generations",
        "_removed",
        "_visible",
        "_transform",
        "_deformation_scale",
        "_selected_elements",
    )

    def __init__(
        self,
        mesh: MeshArrays,
        *,
        on_change: Optional[ChangeCallback] = None,
        owner_thread: Optional[int] = None,
    ) -> None:
        if not isinstance(mesh, MeshArrays):
            raise TypeError("mesh must be MeshArrays")
        self._mesh = mesh
        self._chunks: dict[Hashable, MeshArrays] = {}
        self._chunk_ownership: dict[
            Hashable, tuple[Optional[PackedOwnerTable], Optional[Callable[..., object]]]
        ] = {}
        self._callback = on_change
        self._owner_thread = get_ident() if owner_thread is None else int(owner_thread)
        self._generations = {name: 0 for name in DirtyGenerations.__dataclass_fields__}
        self._removed = False
        self._visible = True
        self._transform = np.eye(4, dtype=np.float64)
        self._deformation_scale = 0.0
        self._selected_elements = np.empty(0, dtype=np.uint64)

    def _check(self) -> None:
        if get_ident() != self._owner_thread:
            raise RuntimeError("retained mesh updates must run on the owning thread")
        if self._removed:
            raise RuntimeError("retained mesh handle has been removed")

    def _change(self, generation: str) -> None:
        self._generations[generation] += 1
        if self._callback is not None:
            self._callback(self, generation)

    @property
    def mesh(self) -> MeshArrays:
        return self._mesh

    @property
    def chunks(self) -> tuple[tuple[Hashable, MeshArrays], ...]:
        return tuple(self._chunks.items())

    @property
    def chunk_records(
        self,
    ) -> tuple[
        tuple[
            Hashable,
            MeshArrays,
            Optional[PackedOwnerTable],
            Optional[Callable[..., object]],
        ],
        ...,
    ]:
        """Return chunk arrays and their optional chunk-local semantic owners."""

        return tuple(
            (chunk_id, mesh, *self._chunk_ownership[chunk_id])
            for chunk_id, mesh in self._chunks.items()
        )

    def chunk_ownership(
        self, chunk_id: Hashable
    ) -> tuple[Optional[PackedOwnerTable], Optional[Callable[..., object]]]:
        """Return the ownership table/resolver registered for one chunk."""

        if chunk_id not in self._chunks:
            raise KeyError(chunk_id)
        return self._chunk_ownership[chunk_id]

    @property
    def generations(self) -> DirtyGenerations:
        return DirtyGenerations(**self._generations)

    @property
    def removed(self) -> bool:
        return self._removed

    @property
    def visible(self) -> bool:
        return self._visible

    @property
    def transform(self) -> np.ndarray:
        view = self._transform.view()
        view.flags.writeable = False
        return view

    @property
    def deformation_scale(self) -> float:
        return self._deformation_scale

    @property
    def selected_elements(self) -> np.ndarray:
        view = self._selected_elements.view()
        view.flags.writeable = False
        return view

    def _replace(self, field: str, value: object, generation: str) -> None:
        self._check()
        candidate = replace(self._mesh, **{field: value})
        self._mesh = candidate
        self._change(generation)

    def update_positions(self, positions: object) -> None:
        array = np.asarray(positions)
        if array.shape != self._mesh.positions.shape:
            raise ValueError("position updates cannot change node count")
        self._replace("positions", positions, "position")

    def update_displacements(self, displacements: object) -> None:
        self._replace("displacements", displacements, "displacement")

    def update_node_scalars(self, values: object) -> None:
        self._replace("node_scalars", values, "scalar")

    def update_element_scalars(self, values: object) -> None:
        self._replace("element_scalars", values, "scalar")

    def set_deformation_scale(self, scale: float) -> None:
        self._check()
        value = float(scale)
        if not np.isfinite(value):
            raise ValueError("deformation scale must be finite")
        if value != self._deformation_scale:
            self._deformation_scale = value
            self._change("displacement")

    def set_active_elements(self, values: object) -> None:
        self._replace("active_elements", values, "active")

    def set_selected_elements(self, values: object) -> None:
        self._check()
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError("selected elements must be a one-dimensional sequence")
        if array.dtype == np.bool_:
            if len(array) != self._mesh.element_count:
                raise ValueError("selection mask must have one value per element")
            selected = np.flatnonzero(array).astype(np.uint64)
        else:
            selected = np.ascontiguousarray(array, dtype=np.uint64)
        if selected.size and int(selected.max()) >= self._mesh.element_count:
            raise ValueError("selected element is out of range")
        self._selected_elements = np.unique(selected)
        self._change("selection")

    def set_visible(self, visible: bool) -> None:
        self._check()
        value = bool(visible)
        if value != self._visible:
            self._visible = value
            self._change("appearance")

    def set_transform(self, transform: object) -> None:
        self._check()
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("transform must be a finite 4 x 4 matrix")
        self._transform = np.ascontiguousarray(matrix)
        self._change("transform")

    @staticmethod
    def _validate_chunk_ownership(
        mesh: MeshArrays,
        owners: Optional[PackedOwnerTable],
        owner_resolver: Optional[Callable[..., object]],
    ) -> None:
        if owners is None:
            if owner_resolver is not None:
                raise ValueError("owner_resolver requires a chunk owner table")
            return
        if not isinstance(owners, PackedOwnerTable):
            raise TypeError("owners must be a PackedOwnerTable")
        for primitive_kind, total in (
            ("triangle", mesh.triangle_count),
            ("line", 0 if mesh.lines is None else len(mesh.lines)),
            ("point", 0 if mesh.point_indices is None else len(mesh.point_indices)),
        ):
            mapped = len(getattr(owners, f"{primitive_kind}_offsets")) - 1
            if mapped not in (0, total):
                raise ValueError(
                    f"{primitive_kind} owner mappings must match chunk {primitive_kind} count"
                )

    def replace_chunk(
        self,
        chunk_id: Hashable,
        replacement: MeshArrays,
        *,
        owners: Optional[PackedOwnerTable] = None,
        owner_resolver: Optional[Callable[..., object]] = None,
    ) -> None:
        """Replace arrays, preserving ownership unless a new table is supplied."""

        self._check()
        if chunk_id not in self._chunks:
            raise KeyError(chunk_id)
        if not isinstance(replacement, MeshArrays):
            raise TypeError("replacement must be MeshArrays")
        current_owners, current_resolver = self._chunk_ownership[chunk_id]
        effective_owners = current_owners if owners is None else owners
        effective_resolver = (
            current_resolver if owners is None and owner_resolver is None
            else owner_resolver
        )
        self._validate_chunk_ownership(
            replacement, effective_owners, effective_resolver
        )
        self._chunks[chunk_id] = replacement
        self._chunk_ownership[chunk_id] = (effective_owners, effective_resolver)
        self._change("topology")

    def add_chunk(
        self,
        chunk_id: Hashable,
        chunk: MeshArrays,
        *,
        owners: Optional[PackedOwnerTable] = None,
        owner_resolver: Optional[Callable[..., object]] = None,
    ) -> None:
        """Add a chunk with optional chunk-local packed semantic ownership."""

        self._check()
        if chunk_id in self._chunks:
            raise KeyError(f"chunk {chunk_id!r} already exists")
        if not isinstance(chunk, MeshArrays):
            raise TypeError("chunk must be MeshArrays")
        self._validate_chunk_ownership(chunk, owners, owner_resolver)
        self._chunks[chunk_id] = chunk
        self._chunk_ownership[chunk_id] = (owners, owner_resolver)
        self._change("topology")

    def set_chunk_ownership(
        self,
        chunk_id: Hashable,
        owners: Optional[PackedOwnerTable],
        *,
        owner_resolver: Optional[Callable[..., object]] = None,
    ) -> None:
        """Replace or clear one chunk's ownership without replacing its arrays."""

        self._check()
        if chunk_id not in self._chunks:
            raise KeyError(chunk_id)
        self._validate_chunk_ownership(
            self._chunks[chunk_id], owners, owner_resolver
        )
        self._chunk_ownership[chunk_id] = (owners, owner_resolver)
        self._change("selection")

    def remove_chunk(self, chunk_id: Hashable) -> None:
        self._check()
        del self._chunks[chunk_id]
        del self._chunk_ownership[chunk_id]
        self._change("topology")

    def remove(self) -> None:
        if self._removed:
            return
        if get_ident() != self._owner_thread:
            raise RuntimeError("retained mesh updates must run on the owning thread")
        self._removed = True
        callback = self._callback
        if callback is not None:
            callback(self, "remove")
        self._callback = None
