"""Packed semantic ownership without an object allocation per primitive."""

from __future__ import annotations

from dataclasses import dataclass
from sys import intern
from typing import Callable, Iterable, Optional, Sequence
from uuid import UUID

import numpy as np

from .selection import PickBinding, PickOwner

_NONE = np.iinfo(np.uint32).max


@dataclass(frozen=True, slots=True)
class ModelOwner:
    """Toolkit-neutral coordinates of a model-bound entity handle."""

    model_id: UUID | str
    kind: str
    id: int
    priority: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", UUID(str(self.model_id)))
        object.__setattr__(self, "kind", intern(str(self.kind)))
        object.__setattr__(self, "id", int(self.id))
        object.__setattr__(self, "priority", int(self.priority))
        if self.id <= 0 or not self.kind:
            raise ValueError("model owners require a kind and positive integer ID")


@dataclass(frozen=True, slots=True)
class ApplicationOwner:
    """An application or FE owner whose stable key is not a geometry handle."""

    key: str | int
    kind: str = ""
    priority: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.key, str) and not self.key:
            raise ValueError("application owner key cannot be empty")
        if not isinstance(self.key, (str, int)) or isinstance(self.key, bool):
            raise TypeError("application owner key must be a string or integer")
        object.__setattr__(self, "kind", intern(str(self.kind)))
        object.__setattr__(self, "priority", int(self.priority))


Owner = PickOwner | ModelOwner | ApplicationOwner
ModelResolver = Callable[[UUID, str, int], object]


def _csr(rows: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.empty(len(rows) + 1, dtype=np.uint32)
    offsets[0] = 0
    flattened: list[int] = []
    for index, row in enumerate(rows):
        flattened.extend(row)
        offsets[index + 1] = len(flattened)
    return offsets, np.asarray(flattened, dtype=np.uint32)


@dataclass(frozen=True, slots=True)
class PackedOwnerTable:
    """Deduplicated owner rows plus CSR primitive-to-owner mappings."""

    documents: tuple[UUID, ...]
    kinds: tuple[str, ...]
    string_keys: tuple[str, ...]
    domains: np.ndarray
    document_slots: np.ndarray
    kind_slots: np.ndarray
    numeric_ids: np.ndarray
    string_slots: np.ndarray
    priorities: np.ndarray
    triangle_offsets: np.ndarray
    triangle_indices: np.ndarray
    line_offsets: np.ndarray
    line_indices: np.ndarray
    point_offsets: np.ndarray
    point_indices: np.ndarray

    @classmethod
    def from_owners(
        cls,
        *,
        triangles: Sequence[Iterable[Owner] | PickBinding | None] = (),
        lines: Sequence[Iterable[Owner] | PickBinding | None] = (),
        points: Sequence[Iterable[Owner] | PickBinding | None] = (),
    ) -> "PackedOwnerTable":
        documents: list[UUID] = []
        kinds: list[str] = []
        strings: list[str] = []
        owner_rows: list[tuple[int, int, int, int, int, int]] = []
        row_lookup: dict[tuple[object, ...], int] = {}

        def slot(items: list[object], value: object) -> int:
            try:
                return items.index(value)
            except ValueError:
                items.append(value)
                return len(items) - 1

        def normalized(value: Iterable[Owner] | PickBinding | None) -> tuple[Owner, ...]:
            if value is None:
                return ()
            if isinstance(value, PickBinding):
                return value.owners
            return tuple(value)

        def encode(owner: Owner) -> int:
            if isinstance(owner, ModelOwner):
                document = slot(documents, owner.model_id)
                kind = slot(kinds, owner.kind)
                key = (0, document, kind, owner.id, _NONE, owner.priority)
            else:
                if isinstance(owner, PickOwner):
                    owner = ApplicationOwner(owner.key, owner.kind, owner.priority)
                kind = slot(kinds, owner.kind)
                if isinstance(owner.key, str):
                    string = slot(strings, owner.key)
                    key = (1, _NONE, kind, 0, string, owner.priority)
                else:
                    if owner.key < 0:
                        raise ValueError("numeric application owner IDs cannot be negative")
                    key = (1, _NONE, kind, owner.key, _NONE, owner.priority)
            existing = row_lookup.get(key)
            if existing is not None:
                return existing
            row_lookup[key] = len(owner_rows)
            owner_rows.append(key)
            return len(owner_rows) - 1

        def encoded(values: Sequence[Iterable[Owner] | PickBinding | None]) -> list[list[int]]:
            return [[encode(owner) for owner in normalized(value)] for value in values]

        triangle_rows = encoded(triangles)
        line_rows = encoded(lines)
        point_rows = encoded(points)
        triangle_offsets, triangle_indices = _csr(triangle_rows)
        line_offsets, line_indices = _csr(line_rows)
        point_offsets, point_indices = _csr(point_rows)
        columns = list(zip(*owner_rows)) if owner_rows else [()] * 6
        return cls(
            tuple(documents),
            tuple(kinds),
            tuple(strings),
            np.asarray(columns[0], dtype=np.uint8),
            np.asarray(columns[1], dtype=np.uint32),
            np.asarray(columns[2], dtype=np.uint32),
            np.asarray(columns[3], dtype=np.uint64),
            np.asarray(columns[4], dtype=np.uint32),
            np.asarray(columns[5], dtype=np.int32),
            triangle_offsets,
            triangle_indices,
            line_offsets,
            line_indices,
            point_offsets,
            point_indices,
        )

    @property
    def owner_count(self) -> int:
        return len(self.domains)

    def owner(self, index: int, model_resolver: Optional[ModelResolver] = None) -> object:
        row = int(index)
        if row < 0 or row >= self.owner_count:
            raise IndexError("owner row is out of range")
        kind = self.kinds[int(self.kind_slots[row])]
        priority = int(self.priorities[row])
        numeric = int(self.numeric_ids[row])
        if int(self.domains[row]) == 0:
            document = self.documents[int(self.document_slots[row])]
            if model_resolver is not None:
                identity = model_resolver(document, kind, numeric)
                return PickOwner(str(identity), f"geometry.{kind}", priority, identity)
            return ModelOwner(document, kind, numeric, priority)
        string_slot = int(self.string_slots[row])
        key: str | int = self.string_keys[string_slot] if string_slot != _NONE else numeric
        return PickOwner(str(key), kind, priority)

    def owners_for(
        self,
        primitive_kind: str,
        primitive: int,
        model_resolver: Optional[ModelResolver] = None,
    ) -> tuple[object, ...]:
        try:
            offsets = getattr(self, f"{primitive_kind}_offsets")
            indices = getattr(self, f"{primitive_kind}_indices")
        except AttributeError as error:
            raise ValueError("primitive kind must be triangle, line, or point") from error
        item = int(primitive)
        if item < 0 or item + 1 >= len(offsets):
            raise IndexError("primitive is out of range")
        start, stop = int(offsets[item]), int(offsets[item + 1])
        return tuple(self.owner(int(row), model_resolver) for row in indices[start:stop])
