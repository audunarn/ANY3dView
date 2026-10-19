"""Headless commercial selection state and viewport-owner adaptation.

Geometry keeps using ANYgeometry's persistent :class:`EntityRef`.  Mesh
selection uses :class:`MeshEntityRef`, whose shape deliberately mirrors it, so
callers can work with ``kind`` and ``id`` without losing the important domain
boundary.  No GUI package is imported here: ANYtk3D ``PickOwner`` instances are
accepted by duck typing through :func:`owner_to_ref`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
import re
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence

from anygeometry.entities import EntityRef

__all__ = [
    "GEOMETRY_KINDS",
    "MESH_KINDS",
    "SELECTION_KINDS",
    "SELECTION_MODES",
    "MeshEntityRef",
    "Selection",
    "SelectionChange",
    "SelectionDomain",
    "SelectionFilter",
    "SelectionOperation",
    "SelectionRejection",
    "adapt_pick_owner",
    "entity_tag",
    "mode_label",
    "owner_to_ref",
    "parse_entity_tag",
    "selection_key",
]


class SelectionDomain(str, Enum):
    GEOMETRY = "geometry"
    MESH = "mesh"


class SelectionOperation(str, Enum):
    REPLACE = "replace"
    ADD = "add"
    TOGGLE = "toggle"
    REMOVE = "remove"


GEOMETRY_KINDS: tuple[str, ...] = ("vertex", "edge", "face")
MESH_KINDS: tuple[str, ...] = ("node", "element", "element_face")
SELECTION_KINDS: tuple[str, ...] = GEOMETRY_KINDS + MESH_KINDS

# Compatibility name: callers have historically used this to build the three
# point/line/plate mode buttons.  Mesh filters are exposed separately rather
# than silently adding three more buttons to an old UI.
SELECTION_MODES: tuple[str, ...] = GEOMETRY_KINDS

_KINDS_BY_DOMAIN = {
    SelectionDomain.GEOMETRY: GEOMETRY_KINDS,
    SelectionDomain.MESH: MESH_KINDS,
}
_DOMAIN_BY_KIND = {
    kind: domain for domain, kinds in _KINDS_BY_DOMAIN.items() for kind in kinds
}
_MODE_LABELS = {
    "vertex": "Point",
    "edge": "Line",
    "face": "Plate",
    "node": "Node",
    "element": "Element",
    "element_face": "Element face",
}

# Prefix so a legacy canvas pick can tell our tags from unrelated canvas tags.
TAG_PREFIX = "ent_"


@dataclass(frozen=True)
class MeshEntityRef:
    """Stable reference to one entity in a particular mesh realization.

    A node or element has an integer ID.  An element face uses
    ``(element_id, local_face_index)``; keeping the pair explicit avoids a
    collision-prone integer packing convention.
    """

    kind: str
    id: int | tuple[int, int]

    def __post_init__(self) -> None:
        kind = str(self.kind)
        if kind not in MESH_KINDS:
            raise ValueError(
                f"unknown mesh entity kind {kind!r}; expected one of "
                f"{', '.join(MESH_KINDS)}"
            )
        if kind == "element_face":
            if not isinstance(self.id, (tuple, list)) or len(self.id) != 2:
                raise ValueError(
                    "an element-face reference needs (element_id, local_face_index)"
                )
            identifier = (_identifier(self.id[0]), _identifier(self.id[1]))
        else:
            identifier = _identifier(self.id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "id", identifier)

    @property
    def domain(self) -> SelectionDomain:
        return SelectionDomain.MESH

    @property
    def element_id(self) -> int:
        if self.kind == "element_face":
            return int(self.id[0])  # type: ignore[index]
        if self.kind == "element":
            return int(self.id)
        raise AttributeError("a node reference has no element ID")

    @property
    def local_face(self) -> int:
        if self.kind != "element_face":
            raise AttributeError(f"a mesh {self.kind} has no local face index")
        return int(self.id[1])  # type: ignore[index]


SelectableRef = EntityRef | MeshEntityRef


def _identifier(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("selection entity IDs must be integers")
    identifier = int(value)
    if identifier < 0:
        raise ValueError("selection entity IDs cannot be negative")
    return identifier


def _ref_kind(ref: object) -> str | None:
    kind = getattr(ref, "kind", None)
    return str(kind) if kind is not None else None


def _ref_domain(ref: object) -> SelectionDomain | None:
    if isinstance(ref, MeshEntityRef):
        return SelectionDomain.MESH
    kind = _ref_kind(ref)
    return _DOMAIN_BY_KIND.get(kind or "")


@dataclass(frozen=True)
class SelectionFilter:
    """One domain and one or more entity kinds accepted by selection tools."""

    domain: SelectionDomain | str = SelectionDomain.GEOMETRY
    kinds: frozenset[str] = frozenset({"face"})

    def __post_init__(self) -> None:
        domain = SelectionDomain(self.domain)
        valid = _KINDS_BY_DOMAIN[domain]
        kinds = frozenset(str(kind) for kind in self.kinds) or frozenset(valid)
        unknown = sorted(kinds - set(valid))
        if unknown:
            raise ValueError(
                f"{domain.value} selection does not support kind(s) {unknown}; "
                f"expected {', '.join(valid)}"
            )
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "kinds", kinds)

    @classmethod
    def one(
        cls, domain: SelectionDomain | str, kind: str
    ) -> "SelectionFilter":
        return cls(domain, frozenset({kind}))

    def accepts(self, ref: object) -> bool:
        return _ref_domain(ref) == self.domain and _ref_kind(ref) in self.kinds

    def explain_rejection(self, ref: object) -> str | None:
        domain = _ref_domain(ref)
        kind = _ref_kind(ref)
        if domain is None or kind is None:
            return "the picked owner is not an ANYfem geometry or mesh entity"
        if domain != self.domain:
            return (
                f"picked {domain.value} {mode_label(kind).lower()}, but the "
                f"active selection domain is {self.domain.value}"
            )
        if kind not in self.kinds:
            accepted = ", ".join(mode_label(item).lower() for item in _ordered_kinds(self))
            return (
                f"picked {mode_label(kind).lower()}, but the active "
                f"{self.domain.value} filter accepts {accepted}"
            )
        return None

    @property
    def qualified_kinds(self) -> frozenset[str]:
        return frozenset(f"{self.domain.value}.{kind}" for kind in self.kinds)


@dataclass(frozen=True)
class SelectionRejection:
    ref: object
    reason: str


@dataclass(frozen=True)
class SelectionChange:
    operation: SelectionOperation
    before: tuple[SelectableRef, ...]
    after: tuple[SelectableRef, ...]
    accepted: tuple[SelectableRef, ...] = ()
    rejected: tuple[SelectionRejection, ...] = ()

    @property
    def changed(self) -> bool:
        return self.before != self.after


def mode_label(mode: str) -> str:
    return _MODE_LABELS.get(mode, mode)


def selection_key(ref: SelectableRef) -> str:
    """Canonical semantic key suitable for an ANYtk3D ``PickOwner``."""

    domain = _ref_domain(ref)
    kind = _ref_kind(ref)
    if domain is None or kind is None:
        raise TypeError("selection keys need an EntityRef or MeshEntityRef")
    if isinstance(ref, MeshEntityRef) and ref.kind == "element_face":
        element, local_face = ref.id
        return f"mesh.element_face:{element}:{local_face}"
    return f"{domain.value}.{kind}:{int(ref.id)}"  # type: ignore[arg-type]


def entity_tag(ref: SelectableRef) -> str:
    """Legacy canvas tag for a selectable entity."""

    if isinstance(ref, MeshEntityRef) and ref.kind == "element_face":
        return f"{TAG_PREFIX}element_face{ref.element_id}_{ref.local_face}"
    return f"{TAG_PREFIX}{ref.kind}{int(ref.id)}"  # type: ignore[arg-type]


def parse_entity_tag(tag: str) -> Optional[SelectableRef]:
    """Turn a legacy canvas tag into a reference, or return ``None``."""

    if not tag or not tag.startswith(TAG_PREFIX):
        return None
    body = tag[len(TAG_PREFIX) :]
    match = re.fullmatch(r"element_face(\d+)[_:](\d+)", body)
    if match:
        return MeshEntityRef("element_face", (int(match[1]), int(match[2])))
    for kind in GEOMETRY_KINDS:
        if body.startswith(kind):
            digits = body[len(kind) :]
            if digits.isdigit():
                return EntityRef(kind, int(digits))  # type: ignore[arg-type]
    for kind in ("element", "node"):
        if body.startswith(kind):
            digits = body[len(kind) :]
            if digits.isdigit():
                return MeshEntityRef(kind, int(digits))
    return None


def owner_to_ref(owner: object, *, strict: bool = False) -> Optional[SelectableRef]:
    """Adapt an ANYtk3D-style owner without importing ANYtk3D.

    Supported objects expose ``key`` and optionally ``kind``.  A
    ``SelectionHit``-like object exposing ``owner`` is unwrapped as a
    convenience.  Qualified kinds such as ``geometry.face`` and
    ``mesh.element`` are authoritative; canonical keys come from
    :func:`selection_key`, while short keys such as ``face7`` remain accepted.
    """

    if isinstance(owner, (EntityRef, MeshEntityRef)):
        return owner
    if not hasattr(owner, "key") and hasattr(owner, "owner"):
        owner = getattr(owner, "owner")
    key = getattr(owner, "key", None)
    kind_hint = str(getattr(owner, "kind", "") or "")
    try:
        if key is None:
            raise ValueError("pick owner has no key")
        return _owner_parts_to_ref(str(key), kind_hint)
    except (TypeError, ValueError) as error:
        if strict:
            raise ValueError(f"cannot adapt pick owner: {error}") from None
        return None


adapt_pick_owner = owner_to_ref


def _owner_parts_to_ref(key: str, kind_hint: str) -> SelectableRef:
    if not key:
        raise ValueError("pick owner key is empty")
    if kind_hint:
        if "." in kind_hint:
            domain_text, kind = kind_hint.split(".", 1)
            domain = SelectionDomain(domain_text)
        else:
            kind = kind_hint
            domain = _DOMAIN_BY_KIND.get(kind)
            if domain is None:
                raise ValueError(f"unknown owner kind {kind_hint!r}")
        if kind not in _KINDS_BY_DOMAIN[domain]:
            raise ValueError(f"kind {kind!r} does not belong to {domain.value}")
        identifiers = [int(value) for value in re.findall(r"\d+", key)]
        return _make_ref(domain, kind, identifiers)

    canonical = re.fullmatch(
        r"(geometry|mesh)\.(vertex|edge|face|node|element|element_face)"
        r"[:/]([0-9]+)(?:[:/_]([0-9]+))?",
        key,
    )
    if canonical:
        domain = SelectionDomain(canonical[1])
        kind = canonical[2]
        identifiers = [int(canonical[3])]
        if canonical[4] is not None:
            identifiers.append(int(canonical[4]))
        return _make_ref(domain, kind, identifiers)
    tagged = parse_entity_tag(key)
    if tagged is not None:
        return tagged
    short_face = re.fullmatch(r"element_face(\d+)[_:./-](\d+)", key)
    if short_face:
        return MeshEntityRef("element_face", (int(short_face[1]), int(short_face[2])))
    short = re.fullmatch(r"(vertex|edge|face|node|element)(\d+)", key)
    if short:
        kind = short[1]
        return _make_ref(_DOMAIN_BY_KIND[kind], kind, [int(short[2])])
    raise ValueError(f"owner key {key!r} carries no supported entity ID")


def _make_ref(
    domain: SelectionDomain, kind: str, identifiers: Sequence[int]
) -> SelectableRef:
    if kind not in _KINDS_BY_DOMAIN[domain]:
        raise ValueError(f"kind {kind!r} does not belong to {domain.value}")
    if kind == "element_face":
        if len(identifiers) < 2:
            raise ValueError("an element-face owner needs element and local-face IDs")
        return MeshEntityRef(kind, (identifiers[-2], identifiers[-1]))
    if not identifiers:
        raise ValueError(f"owner key carries no {kind} ID")
    identifier = identifiers[-1]
    if domain == SelectionDomain.GEOMETRY:
        return EntityRef(kind, identifier)  # type: ignore[arg-type]
    return MeshEntityRef(kind, identifier)


class Selection:
    """Ordered selection set with an explicit domain and kind filter."""

    def __init__(
        self,
        mode: str = "face",
        *,
        domain: SelectionDomain | str | None = None,
        kinds: Iterable[str] | None = None,
    ) -> None:
        if domain is None:
            resolved_domain = _DOMAIN_BY_KIND.get(mode)
            if resolved_domain is None:
                self._check_mode(mode)
                raise AssertionError("unreachable")
        else:
            resolved_domain = SelectionDomain(domain)
        if kinds is None:
            resolved_mode = (
                mode
                if mode in _KINDS_BY_DOMAIN[resolved_domain]
                else _default_kind(resolved_domain)
            )
            resolved_kinds = frozenset({resolved_mode})
        else:
            resolved_kinds = frozenset(str(kind) for kind in kinds)
            normalized_filter = SelectionFilter(resolved_domain, resolved_kinds)
            resolved_kinds = normalized_filter.kinds
            candidate = mode if mode in normalized_filter.kinds else ""
            resolved_mode = candidate or next(
                kind
                for kind in _KINDS_BY_DOMAIN[resolved_domain]
                if kind in normalized_filter.kinds
            )
        self._filter = SelectionFilter(resolved_domain, resolved_kinds)
        self._mode = resolved_mode
        self._items: List[SelectableRef] = []
        self._listeners: List[Callable[[], None]] = []
        self._last_rejection: str | None = None

    # ------------------------------------------------------------------
    @staticmethod
    def _check_mode(mode: str) -> str:
        if mode not in SELECTION_KINDS:
            raise ValueError(
                f"unknown selection mode {mode!r}; expected one of "
                f"{', '.join(SELECTION_KINDS)}"
            )
        return mode

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def domain(self) -> SelectionDomain:
        return self._filter.domain

    @property
    def filter(self) -> SelectionFilter:
        return self._filter

    @property
    def allowed_kinds(self) -> frozenset[str]:
        return self._filter.kinds

    @property
    def last_rejection(self) -> str | None:
        return self._last_rejection

    def set_mode(self, mode: str) -> None:
        """Switch to one geometry or mesh kind, dropping wrong-scope items."""

        mode = self._check_mode(mode)
        domain = _DOMAIN_BY_KIND[mode]
        requested = SelectionFilter.one(domain, mode)
        if requested == self._filter and mode == self._mode:
            return
        self._mode = mode
        self._filter = requested
        self._items = [ref for ref in self._items if requested.accepts(ref)]
        self._last_rejection = None
        self._notify()

    def set_domain(self, domain: SelectionDomain | str) -> None:
        resolved = SelectionDomain(domain)
        mode = (
            self._mode
            if self._mode in _KINDS_BY_DOMAIN[resolved]
            else _default_kind(resolved)
        )
        self.set_filter(SelectionFilter.one(resolved, mode))

    def set_filter(self, selection_filter: SelectionFilter) -> None:
        if not isinstance(selection_filter, SelectionFilter):
            raise TypeError("set_filter expects a SelectionFilter")
        if selection_filter == self._filter:
            return
        self._filter = selection_filter
        ordered = _ordered_kinds(selection_filter)
        self._mode = self._mode if self._mode in selection_filter.kinds else ordered[0]
        self._items = [ref for ref in self._items if selection_filter.accepts(ref)]
        self._last_rejection = None
        self._notify()

    # ------------------------------------------------------------------
    @property
    def items(self) -> List[SelectableRef]:
        return list(self._items)

    @property
    def ordered_items(self) -> tuple[SelectableRef, ...]:
        """Pick order, suitable for start/via/end and similar workflows."""

        return tuple(self._items)

    def pick_index(self, ref: SelectableRef) -> int | None:
        try:
            return self._items.index(ref)
        except ValueError:
            return None

    def require_ordered(
        self, count: int, *, kind: str | None = None
    ) -> tuple[SelectableRef, ...]:
        selected = tuple(
            ref for ref in self._items if kind is None or _ref_kind(ref) == kind
        )
        if len(selected) != int(count):
            noun = "entities" if kind is None else mode_label(kind).lower() + "s"
            raise ValueError(
                f"this operation needs {int(count)} ordered {noun}; "
                f"{len(selected)} selected"
            )
        return selected

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        # A Selection is a service object, not a container.  An empty shared
        # selection must not be replaced by ``selection or Selection()``.
        return True

    def __contains__(self, ref: object) -> bool:
        return ref in self._items

    def __iter__(self) -> Iterator[SelectableRef]:
        return iter(list(self._items))

    @property
    def first(self) -> Optional[SelectableRef]:
        return self._items[0] if self._items else None

    def is_empty(self) -> bool:
        return not self._items

    # ------------------------------------------------------------------
    def explain_rejection(self, ref: object) -> str | None:
        return self._filter.explain_rejection(ref)

    def apply(
        self,
        refs: Iterable[SelectableRef],
        operation: SelectionOperation | str = SelectionOperation.REPLACE,
    ) -> SelectionChange:
        """Apply a commercial replace/add/toggle/remove set operation.

        A non-empty hit list containing only out-of-scope entities leaves the
        prior selection intact and explains the rejection.  An actually empty
        replacement is an empty-space pick and clears it.
        """

        resolved_operation = SelectionOperation(operation)
        supplied = list(refs)
        accepted: list[SelectableRef] = []
        seen: set[SelectableRef] = set()
        rejected: list[SelectionRejection] = []
        for ref in supplied:
            reason = self._filter.explain_rejection(ref)
            if reason is not None:
                rejected.append(SelectionRejection(ref, reason))
            elif ref not in seen:
                seen.add(ref)
                accepted.append(ref)
        self._last_rejection = rejected[-1].reason if rejected else None
        before = tuple(self._items)
        if supplied and not accepted and resolved_operation == SelectionOperation.REPLACE:
            return SelectionChange(
                resolved_operation,
                before,
                before,
                (),
                tuple(rejected),
            )
        updated = _apply_operation(self._items, accepted, resolved_operation)
        if updated != self._items:
            self._items = updated
            self._notify()
        return SelectionChange(
            resolved_operation,
            before,
            tuple(self._items),
            tuple(accepted),
            tuple(rejected),
        )

    def apply_owners(
        self,
        owners: Iterable[object],
        operation: SelectionOperation | str = SelectionOperation.REPLACE,
    ) -> SelectionChange:
        supplied_owners = list(owners)
        resolved_operation = SelectionOperation(operation)
        references: list[SelectableRef] = []
        invalid: list[object] = []
        for owner in supplied_owners:
            ref = owner_to_ref(owner)
            if ref is None:
                invalid.append(owner)
            else:
                references.append(ref)
        # A non-empty owner list that cannot be adapted is a rejected pick,
        # not an empty-space click.  In particular, REPLACE must not erase a
        # valid prior selection merely because a viewport owner is unknown.
        if (
            supplied_owners
            and not references
            and resolved_operation == SelectionOperation.REPLACE
        ):
            before = tuple(self._items)
            change = SelectionChange(resolved_operation, before, before, (), ())
        else:
            change = self.apply(references, resolved_operation)
        if not invalid:
            return change
        rejected = list(change.rejected)
        rejected.extend(
            SelectionRejection(
                owner, "the picked owner has no supported ANYfem entity reference"
            )
            for owner in invalid
        )
        self._last_rejection = rejected[-1].reason
        return SelectionChange(
            change.operation,
            change.before,
            change.after,
            change.accepted,
            tuple(rejected),
        )

    def select_all(
        self,
        universe: Iterable[SelectableRef],
        operation: SelectionOperation | str = SelectionOperation.REPLACE,
    ) -> SelectionChange:
        supplied = list(universe)
        accepted = [ref for ref in supplied if self._filter.accepts(ref)]
        # Calling apply with the accepted subset gives an all-wrong universe
        # the standard Select All meaning: no matching entities are selected.
        return self.apply(accepted, operation)

    def invert(self, universe: Iterable[SelectableRef]) -> SelectionChange:
        available: list[SelectableRef] = []
        for ref in universe:
            if self._filter.accepts(ref) and ref not in available:
                available.append(ref)
        inverted = [ref for ref in available if ref not in self._items]
        return self.apply(inverted, SelectionOperation.REPLACE)

    # Legacy API -------------------------------------------------------
    def select(self, ref: EntityRef | MeshEntityRef, extend: bool = False) -> None:
        """Select one entity; legacy ``extend`` toggles it."""

        if not self._filter.accepts(ref):
            if (
                self.domain == SelectionDomain.GEOMETRY
                and len(self.allowed_kinds) == 1
                and _ref_kind(ref) in GEOMETRY_KINDS
            ):
                raise ValueError(
                    f"cannot select a {_ref_kind(ref)} while in {self._mode} mode"
                )
            reason = self.explain_rejection(ref) or "entity is outside the selection scope"
            raise ValueError(reason)
        operation = SelectionOperation.TOGGLE if extend else SelectionOperation.REPLACE
        self.apply([ref], operation)

    def select_many(
        self, refs: Iterable[EntityRef | MeshEntityRef], extend: bool = False
    ) -> None:
        # Preserve the historical contract: mixed input is filtered silently,
        # and an all-wrong replacement clears the current selection.
        wanted = [ref for ref in refs if self._filter.accepts(ref)]
        operation = SelectionOperation.ADD if extend else SelectionOperation.REPLACE
        self.apply(wanted, operation)

    def restore(self, refs: Iterable[SelectableRef]) -> None:
        """Restore an exact ordered snapshot with one notification."""

        restored = list(dict.fromkeys(refs))
        if restored == self._items:
            return
        self._items = restored
        self._notify()

    def apply_replacements(
        self,
        replacements: Iterable[tuple[EntityRef, tuple[EntityRef, ...]]],
    ) -> None:
        """Follow geometry selections through an ordered replacement log."""

        selected = list(self._items)
        for old, new in replacements:
            updated: List[SelectableRef] = []
            for item in selected:
                updated.extend(new if item == old else (item,))
            selected = updated
        self.restore(selected)

    def clear(self) -> None:
        self.apply((), SelectionOperation.REPLACE)

    def handle_tag(self, tag: str, extend: bool = False) -> Optional[SelectableRef]:
        """Legacy tag pick; an unrecognized/wrong-scope tag clears as before."""

        ref = parse_entity_tag(tag)
        if ref is None or not self._filter.accepts(ref):
            if not extend:
                self.clear()
            return None
        self.select(ref, extend=extend)
        return ref

    # ------------------------------------------------------------------
    def tags(self) -> List[str]:
        return [entity_tag(ref) for ref in self._items]

    def describe(self) -> str:
        ordered = _ordered_kinds(self._filter)
        if not self._items:
            if len(ordered) == 1:
                return f"No {mode_label(ordered[0]).lower()} selected"
            return f"No {self.domain.value} entities selected"
        if len(self._items) == 1:
            ref = self._items[0]
            identifier = (
                f"{ref.element_id}:{ref.local_face}"
                if isinstance(ref, MeshEntityRef) and ref.kind == "element_face"
                else str(ref.id)
            )
            return f"{mode_label(ref.kind)} {identifier}"
        kinds = {_ref_kind(ref) for ref in self._items}
        if len(kinds) == 1:
            kind = next(iter(kinds)) or "entity"
            return f"{len(self._items)} {_plural_label(kind)} selected"
        return f"{len(self._items)} {self.domain.value} entities selected"

    def add_listener(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[], None]) -> None:
        while callback in self._listeners:
            self._listeners.remove(callback)

    def _notify(self) -> None:
        for callback in list(self._listeners):
            callback()

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Selection(domain={self.domain.value!r}, mode={self._mode!r}, "
            f"kinds={sorted(self.allowed_kinds)!r}, items={self._items!r})"
        )


def _default_kind(domain: SelectionDomain) -> str:
    return "face" if domain == SelectionDomain.GEOMETRY else "element"


def _ordered_kinds(selection_filter: SelectionFilter) -> list[str]:
    return [
        kind
        for kind in _KINDS_BY_DOMAIN[selection_filter.domain]
        if kind in selection_filter.kinds
    ]


def _apply_operation(
    current: Sequence[SelectableRef],
    supplied: Sequence[SelectableRef],
    operation: SelectionOperation,
) -> list[SelectableRef]:
    if operation == SelectionOperation.REPLACE:
        return list(supplied)
    # Dict insertion order preserves the existing ordered-set contract while
    # avoiding repeated linear scans for large box/lasso selections.
    updated = dict.fromkeys(current)
    if operation == SelectionOperation.ADD:
        for ref in supplied:
            updated.setdefault(ref, None)
    elif operation == SelectionOperation.TOGGLE:
        for ref in supplied:
            if ref in updated:
                del updated[ref]
            else:
                updated[ref] = None
    elif operation == SelectionOperation.REMOVE:
        remove = set(supplied)
        return [ref for ref in updated if ref not in remove]
    return list(updated)


def _plural_label(kind: str) -> str:
    label = mode_label(kind).lower()
    return label[:-1] + "ies" if label.endswith("y") else label + "s"
