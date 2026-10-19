"""Reusable geometry and mesh scopes.

Regions are the indirection between design intent and physical attributes.
Geometry anchors can follow feature outputs and geometry groups; mesh anchors
are deliberately tied to one mesh revision so a remesh cannot silently move a
load or an output request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Literal, Mapping, Sequence
from uuid import uuid4

import numpy as np
from anygeometry.entities import EntityRef

__all__ = [
    "BooleanRegion",
    "ElementFaceRef",
    "GeometryGroupRef",
    "ManualRegion",
    "MeshEntityRef",
    "QueryClause",
    "QueryGroup",
    "QueryRegion",
    "Region",
    "RegionDomain",
    "RegionError",
    "RegionRegistry",
    "RegionRef",
    "RegionStatus",
    "region_from_dict",
]


class RegionError(ValueError):
    """Raised when a region is malformed or cannot be resolved safely."""


class RegionDomain(str, Enum):
    GEOMETRY = "geometry"
    MESH = "mesh"


class RegionStatus(str, Enum):
    VALID = "valid"
    EMPTY = "empty"
    UNRESOLVED = "unresolved"
    STALE = "stale"
    INVALID = "invalid"


@dataclass(frozen=True)
class RegionRef:
    """Stable reference used canonically by assignments and attributes."""

    id: str

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise RegionError("region reference needs a UUID")
        object.__setattr__(self, "id", str(self.id))


@dataclass(frozen=True)
class GeometryGroupRef:
    name: str
    kind: str | None = None


@dataclass(frozen=True)
class MeshEntityRef:
    mesh_id: str
    kind: Literal["node", "element"]
    id: int

    def __post_init__(self) -> None:
        if self.kind not in ("node", "element"):
            raise RegionError(f"unknown mesh entity kind {self.kind!r}")
        if int(self.id) <= 0:
            raise RegionError("mesh entity IDs must be positive")


@dataclass(frozen=True)
class ElementFaceRef:
    mesh_id: str
    element_id: int
    local_face: int

    def __post_init__(self) -> None:
        if int(self.element_id) <= 0 or int(self.local_face) < 0:
            raise RegionError("element-face IDs must be non-negative/positive")


GeometryAnchor = EntityRef | GeometryGroupRef | Any
MeshAnchor = MeshEntityRef | ElementFaceRef


@dataclass(frozen=True)
class ManualRegion:
    anchors: tuple[GeometryAnchor | MeshAnchor, ...]

    def __init__(self, anchors: Iterable[GeometryAnchor | MeshAnchor]):
        object.__setattr__(self, "anchors", tuple(dict.fromkeys(anchors)))


_QUERY_OPERATORS = frozenset(
    {"eq", "ne", "lt", "le", "gt", "ge", "between", "in", "contains"}
)


@dataclass(frozen=True)
class QueryClause:
    """One validated, non-executable region predicate."""

    property: str
    operator: str
    value: Any

    def __post_init__(self) -> None:
        if not self.property or self.property.startswith("_"):
            raise RegionError("query property must be a public property name")
        if self.operator not in _QUERY_OPERATORS:
            raise RegionError(
                f"unknown query operator {self.operator!r}; expected one of "
                f"{', '.join(sorted(_QUERY_OPERATORS))}"
            )

    def matches(self, properties: Mapping[str, Any]) -> bool:
        if self.property not in properties:
            return False
        actual = properties[self.property]
        expected = self.value
        if self.operator == "eq":
            return actual == expected
        if self.operator == "ne":
            return actual != expected
        if self.operator == "lt":
            return actual < expected
        if self.operator == "le":
            return actual <= expected
        if self.operator == "gt":
            return actual > expected
        if self.operator == "ge":
            return actual >= expected
        if self.operator == "between":
            try:
                low, high = expected
            except (TypeError, ValueError):
                raise RegionError("between expects [low, high]") from None
            return low <= actual <= high
        if self.operator == "in":
            return actual in expected
        if self.operator == "contains":
            return expected in actual
        return False  # pragma: no cover - validated above

    def to_dict(self) -> dict[str, Any]:
        return {"property": self.property, "operator": self.operator, "value": self.value}


@dataclass(frozen=True)
class QueryGroup:
    operation: Literal["all", "any", "not"] = "all"
    clauses: tuple[QueryClause | "QueryGroup", ...] = ()

    def __post_init__(self) -> None:
        if self.operation not in ("all", "any", "not"):
            raise RegionError(f"unknown query group operation {self.operation!r}")
        if self.operation == "not" and len(self.clauses) != 1:
            raise RegionError("a not query group needs exactly one child")

    def matches(self, properties: Mapping[str, Any]) -> bool:
        values = [clause.matches(properties) for clause in self.clauses]
        if self.operation == "all":
            return all(values)
        if self.operation == "any":
            return any(values)
        return not values[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "clauses": [item.to_dict() for item in self.clauses],
        }


@dataclass(frozen=True)
class QueryRegion:
    expression: QueryGroup


@dataclass(frozen=True)
class BooleanRegion:
    operation: Literal["union", "intersection", "subtract"]
    region_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.operation not in ("union", "intersection", "subtract"):
            raise RegionError(f"unknown Boolean operation {self.operation!r}")
        minimum = 1 if self.operation == "union" else 2
        if len(self.region_ids) < minimum:
            raise RegionError(
                f"{self.operation} region needs at least {minimum} input(s)"
            )


RegionDefinition = ManualRegion | QueryRegion | BooleanRegion


@dataclass
class Region:
    name: str
    domain: RegionDomain | str
    entity_kind: str
    definition: RegionDefinition
    id: str = field(default_factory=lambda: str(uuid4()))
    hidden: bool = False
    mesh_id: str | None = None

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.domain = RegionDomain(self.domain)
        if not self.name.strip():
            raise RegionError("region needs a name")
        if not self.entity_kind:
            raise RegionError("region needs an entity kind")
        if self.domain is RegionDomain.MESH and not self.mesh_id:
            # Manual mesh anchors carry the ID themselves, but an explicit
            # region mesh ID makes staleness cheap and unambiguous.
            anchors = getattr(self.definition, "anchors", ())
            mesh_ids = {getattr(item, "mesh_id", None) for item in anchors}
            mesh_ids.discard(None)
            if len(mesh_ids) == 1:
                self.mesh_id = str(next(iter(mesh_ids)))
            else:
                raise RegionError("mesh region needs one mesh_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain.value,
            "entity_kind": self.entity_kind,
            "hidden": bool(self.hidden),
            "mesh_id": self.mesh_id,
            "definition": _definition_to_dict(self.definition),
        }


class RegionRegistry:
    """Owns regions and resolves dependencies with cycle detection."""

    def __init__(self, regions: Iterable[Region] = ()) -> None:
        self._regions: dict[str, Region] = {}
        for region in regions:
            self.add(region)

    def __len__(self) -> int:
        return len(self._regions)

    def __iter__(self):
        return iter(self._regions.values())

    def __contains__(self, region_id: object) -> bool:
        return region_id in self._regions

    def __getitem__(self, region_id: str) -> Region:
        try:
            return self._regions[region_id]
        except KeyError:
            raise RegionError(f"unknown region {region_id!r}") from None

    def add(self, region: Region) -> Region:
        if region.id in self._regions:
            raise RegionError(f"duplicate region ID {region.id}")
        self._regions[region.id] = region
        self._validate_cycles()
        return region

    def remove(self, region_id: str, *, cascade: bool = False) -> tuple[str, ...]:
        dependents = self.dependents(region_id, transitive=True)
        if dependents and not cascade:
            raise RegionError(
                f"region {region_id!r} is used by {', '.join(dependents)}"
            )
        removed = tuple(dependents) + (region_id,)
        for key in removed:
            self._regions.pop(key, None)
        return removed

    def dependents(self, region_id: str, *, transitive: bool = False) -> tuple[str, ...]:
        found: set[str] = set()
        frontier = {region_id}
        while frontier:
            next_frontier: set[str] = set()
            for key, region in self._regions.items():
                definition = region.definition
                if isinstance(definition, BooleanRegion) and set(definition.region_ids) & frontier:
                    if key not in found:
                        found.add(key)
                        next_frontier.add(key)
            if not transitive:
                break
            frontier = next_frontier
        return tuple(sorted(found))

    def resolve(
        self,
        region_id: str,
        *,
        geometry=None,
        mesh_id: str | None = None,
        candidates: Iterable[Any] | None = None,
        properties=None,
        feature_resolver=None,
    ) -> tuple[Any, ...]:
        return self._resolve(
            region_id,
            stack=(),
            geometry=geometry,
            mesh_id=mesh_id,
            candidates=candidates,
            properties=properties,
            feature_resolver=feature_resolver,
        )

    def status(self, region_id: str, **kwargs: Any) -> RegionStatus:
        region = self[region_id]
        if region.domain is RegionDomain.MESH and kwargs.get("mesh_id") not in (None, region.mesh_id):
            return RegionStatus.STALE
        try:
            resolved = self.resolve(region_id, **kwargs)
        except (RegionError, KeyError, ValueError):
            return RegionStatus.UNRESOLVED
        return RegionStatus.VALID if resolved else RegionStatus.EMPTY

    def to_list(self) -> list[dict[str, Any]]:
        return [region.to_dict() for region in sorted(self, key=lambda item: item.id)]

    def _resolve(self, region_id: str, *, stack: tuple[str, ...], **context: Any) -> tuple[Any, ...]:
        if region_id in stack:
            raise RegionError("cyclic region dependency: " + " -> ".join(stack + (region_id,)))
        region = self[region_id]
        active_mesh = context.get("mesh_id")
        if region.domain is RegionDomain.MESH and active_mesh not in (None, region.mesh_id):
            raise RegionError(
                f"region {region.name!r} belongs to mesh {region.mesh_id}, not {active_mesh}"
            )
        definition = region.definition
        if isinstance(definition, ManualRegion):
            return _resolve_manual(region, definition, **context)
        if isinstance(definition, QueryRegion):
            candidates = tuple(context.get("candidates") or ())
            getter = context.get("properties") or (lambda value: _default_properties(value, context.get("geometry")))
            return tuple(item for item in candidates if definition.expression.matches(getter(item)))
        values = [
            set(self._resolve(child, stack=stack + (region_id,), **context))
            for child in definition.region_ids
        ]
        if definition.operation == "union":
            result = set().union(*values)
        elif definition.operation == "intersection":
            result = set.intersection(*values)
        else:
            result = values[0].difference(*values[1:])
        return tuple(sorted(result, key=repr))

    def _validate_cycles(self) -> None:
        for key in self._regions:
            self._walk(key, ())

    def _walk(self, key: str, stack: tuple[str, ...]) -> None:
        if key in stack:
            raise RegionError("cyclic region dependency: " + " -> ".join(stack + (key,)))
        definition = self[key].definition
        if not isinstance(definition, BooleanRegion):
            return
        for child in definition.region_ids:
            if child not in self._regions:
                raise RegionError(f"region {key!r} references missing region {child!r}")
            self._walk(child, stack + (key,))


def _resolve_manual(region: Region, definition: ManualRegion, **context: Any) -> tuple[Any, ...]:
    geometry = context.get("geometry")
    feature_resolver = context.get("feature_resolver")
    resolved: list[Any] = []
    for anchor in definition.anchors:
        if isinstance(anchor, GeometryGroupRef):
            if geometry is None:
                raise RegionError("geometry group resolution needs a geometry model")
            if anchor.name not in geometry.groups:
                raise RegionError(
                    f"geometry group {anchor.name!r} is not available"
                )
            members = geometry.group(anchor.name)
            resolved.extend(
                item for item in members if anchor.kind is None or item.kind == anchor.kind
            )
        elif isinstance(anchor, EntityRef):
            if geometry is not None:
                # Follow only ANYgeometry's explicit replacement lineage.  A
                # split/fragment operation can legitimately replace one owner
                # with several; nearest-geometry retargeting is never used.
                try:
                    current = geometry.entity_ref(anchor.kind, anchor.id)
                except (KeyError, ValueError):
                    descendants = tuple(geometry.resolve_ref(anchor))
                    if not descendants:
                        # Preserve the authoritative missing-entity diagnostic.
                        geometry.entity_ref(anchor.kind, anchor.id)
                    resolved.extend(descendants)
                else:
                    resolved.append(current)
            else:
                resolved.append(anchor)
        elif isinstance(anchor, (MeshEntityRef, ElementFaceRef)):
            if anchor.mesh_id != region.mesh_id:
                raise RegionError("mesh anchor belongs to another mesh")
            resolved.append(anchor)
        elif feature_resolver is not None:
            values = feature_resolver(anchor)
            values = (
                tuple(values)
                if isinstance(values, (list, tuple, set))
                else (() if values is None else (values,))
            )
            if not values:
                detail = (
                    f"feature {anchor.feature_id} output "
                    f"{anchor.output_key!r} is unresolved"
                    if all(
                        hasattr(anchor, item)
                        for item in ("feature_id", "output_key")
                    )
                    else f"region anchor {anchor!r} is unresolved"
                )
                raise RegionError(detail)
            resolved.extend(values)
        else:
            raise RegionError(f"cannot resolve region anchor {anchor!r}")
    return tuple(dict.fromkeys(resolved))


def _default_properties(value: Any, geometry) -> dict[str, Any]:
    properties = {
        "kind": getattr(value, "kind", type(value).__name__.lower()),
        "id": getattr(value, "id", None),
    }
    if isinstance(value, EntityRef) and geometry is not None:
        properties["tags"] = geometry.tags_for(value)
        if value.kind == "vertex":
            position = np.asarray(geometry.vertex_position(value.id), dtype=float)
            properties.update(x=float(position[0]), y=float(position[1]), z=float(position[2]))
        elif value.kind == "edge":
            properties["length"] = float(geometry.edge_length(value.id))
    return properties


def _anchor_to_dict(anchor: Any) -> dict[str, Any]:
    if isinstance(anchor, EntityRef):
        return {"type": "entity", "kind": anchor.kind, "id": int(anchor.id)}
    if isinstance(anchor, GeometryGroupRef):
        return {"type": "geometry_group", "name": anchor.name, "kind": anchor.kind}
    if isinstance(anchor, MeshEntityRef):
        return {"type": "mesh_entity", "mesh_id": anchor.mesh_id, "kind": anchor.kind, "id": int(anchor.id)}
    if isinstance(anchor, ElementFaceRef):
        return {"type": "element_face", "mesh_id": anchor.mesh_id, "element_id": int(anchor.element_id), "local_face": int(anchor.local_face)}
    # FeatureOutputRef is intentionally duck-typed so ANYfem can still import
    # against ANYgeometry v1 while the coordinated v2 release is installed.
    if all(hasattr(anchor, name) for name in ("feature_id", "output_key", "kind")):
        return {"type": "feature_output", "feature_id": int(anchor.feature_id), "output_key": str(anchor.output_key), "kind": str(anchor.kind)}
    raise RegionError(f"unsupported region anchor {anchor!r}")


def _definition_to_dict(definition: RegionDefinition) -> dict[str, Any]:
    if isinstance(definition, ManualRegion):
        return {"type": "manual", "anchors": [_anchor_to_dict(item) for item in definition.anchors]}
    if isinstance(definition, QueryRegion):
        return {"type": "query", "expression": definition.expression.to_dict()}
    return {"type": "boolean", "operation": definition.operation, "region_ids": list(definition.region_ids)}


def _query_from_dict(data: Mapping[str, Any]) -> QueryClause | QueryGroup:
    if "property" in data:
        return QueryClause(
            property=str(data["property"]),
            operator=str(data.get("operator", "eq")),
            value=data.get("value"),
        )
    raw = data.get("clauses", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RegionError("query clauses must be a list")
    return QueryGroup(
        operation=str(data.get("operation", "all")),  # type: ignore[arg-type]
        clauses=tuple(_query_from_dict(item) for item in raw),
    )


def _anchor_from_dict(data: Mapping[str, Any]):
    kind = str(data.get("type", "entity"))
    if kind == "entity":
        return EntityRef(str(data["kind"]), int(data["id"]))  # type: ignore[arg-type]
    if kind == "geometry_group":
        return GeometryGroupRef(
            name=str(data["name"]),
            kind=None if data.get("kind") is None else str(data["kind"]),
        )
    if kind == "mesh_entity":
        return MeshEntityRef(
            mesh_id=str(data["mesh_id"]),
            kind=str(data["kind"]),  # type: ignore[arg-type]
            id=int(data["id"]),
        )
    if kind == "element_face":
        return ElementFaceRef(
            mesh_id=str(data["mesh_id"]),
            element_id=int(data["element_id"]),
            local_face=int(data["local_face"]),
        )
    if kind == "feature_output":
        try:
            from anygeometry.features import FeatureOutputRef
        except ImportError:
            # Preserve a structurally valid future anchor even if the
            # coordinated ANYgeometry feature package is unavailable.
            @dataclass(frozen=True)
            class FeatureOutputRef:  # type: ignore[no-redef]
                feature_id: int
                output_key: str
                kind: str
        return FeatureOutputRef(
            feature_id=int(data["feature_id"]),
            output_key=str(data["output_key"]),
            kind=str(data["kind"]),
        )
    raise RegionError(f"unknown region anchor type {kind!r}")


def region_from_dict(data: Mapping[str, Any]) -> Region:
    raw_definition = data.get("definition", {})
    if not isinstance(raw_definition, Mapping):
        raise RegionError("region definition must be an object")
    kind = str(raw_definition.get("type", "manual"))
    if kind == "manual":
        anchors = raw_definition.get("anchors", ())
        if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
            raise RegionError("manual region anchors must be a list")
        definition: RegionDefinition = ManualRegion(
            _anchor_from_dict(item) for item in anchors
        )
    elif kind == "query":
        expression = raw_definition.get("expression", {})
        if not isinstance(expression, Mapping):
            raise RegionError("query expression must be an object")
        parsed = _query_from_dict(expression)
        if isinstance(parsed, QueryClause):
            parsed = QueryGroup("all", (parsed,))
        definition = QueryRegion(parsed)
    elif kind == "boolean":
        definition = BooleanRegion(
            operation=str(raw_definition.get("operation", "union")),  # type: ignore[arg-type]
            region_ids=tuple(str(item) for item in raw_definition.get("region_ids", ())),
        )
    else:
        raise RegionError(f"unknown region definition type {kind!r}")
    return Region(
        id=str(data.get("id", uuid4())),
        name=str(data.get("name", "Region")),
        domain=str(data.get("domain", "geometry")),
        entity_kind=str(data.get("entity_kind", "face")),
        definition=definition,
        hidden=bool(data.get("hidden", False)),
        mesh_id=None if data.get("mesh_id") is None else str(data["mesh_id"]),
    )
