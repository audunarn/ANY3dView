"""JSON-safe semantic view state shared by renderers and automation clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping
from uuid import UUID

from .ownership import ApplicationOwner, ModelOwner
from .selection import PickOwner


@dataclass(frozen=True, slots=True)
class SemanticRef:
    """Stable, JSON-safe reference to one rendered semantic owner."""

    source: str
    kind: str
    key: str | int
    model_id: str | None = None

    def __post_init__(self) -> None:
        source = str(self.source).strip().lower()
        kind = str(self.kind).strip()
        if source not in {"model", "application"}:
            raise ValueError("semantic source must be 'model' or 'application'")
        if not kind:
            raise ValueError("semantic kind cannot be empty")
        if isinstance(self.key, bool) or not isinstance(self.key, (str, int)):
            raise TypeError("semantic key must be a string or integer")
        if isinstance(self.key, str) and not self.key:
            raise ValueError("semantic key cannot be empty")
        model_id = self.model_id
        if source == "model":
            if not isinstance(self.key, int) or self.key <= 0:
                raise ValueError("model semantic keys must be positive integers")
            if model_id is None:
                raise ValueError("model semantic references require model_id")
            model_id = str(UUID(str(model_id)))
        elif model_id is not None:
            raise ValueError("application semantic references cannot have model_id")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "model_id", model_id)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticRef":
        allowed = {"source", "kind", "key", "model_id"}
        extras = set(value).difference(allowed)
        if extras:
            raise ValueError(f"unknown semantic reference fields: {sorted(extras)}")
        return cls(
            source=value.get("source", ""),
            kind=value.get("kind", ""),
            key=value.get("key", ""),
            model_id=value.get("model_id"),
        )

    @classmethod
    def from_owner(cls, owner: object) -> "SemanticRef":
        if isinstance(owner, ModelOwner):
            return cls("model", owner.kind, owner.id, str(owner.model_id))
        if isinstance(owner, ApplicationOwner):
            return cls("application", owner.kind or "application", owner.key)
        if isinstance(owner, PickOwner):
            identity = owner.identity
            if identity is not None and identity is not owner:
                try:
                    return cls.from_owner(identity)
                except (TypeError, ValueError):
                    pass
            return cls("application", owner.kind or "application", owner.key)
        if isinstance(owner, SemanticRef):
            return owner
        raise TypeError(f"owner is not representable as SemanticRef: {type(owner).__name__}")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "source": self.source,
            "kind": self.kind,
            "key": self.key,
        }
        if self.model_id is not None:
            result["model_id"] = self.model_id
        return result


def semantic_refs(values: Iterable[object]) -> tuple[SemanticRef, ...]:
    """Normalize and stably de-duplicate semantic references."""

    result: list[SemanticRef] = []
    seen: set[SemanticRef] = set()
    for value in values:
        made = (
            SemanticRef.from_dict(value)
            if isinstance(value, Mapping)
            else SemanticRef.from_owner(value)
        )
        if made not in seen:
            seen.add(made)
            result.append(made)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class VisibilityState:
    """Semantic exclusions and optional isolation policy for one viewport."""

    hidden: tuple[SemanticRef, ...] = field(default_factory=tuple)
    hidden_kinds: tuple[str, ...] = field(default_factory=tuple)
    isolated: tuple[SemanticRef, ...] = field(default_factory=tuple)
    isolated_kinds: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden", semantic_refs(self.hidden))
        object.__setattr__(self, "isolated", semantic_refs(self.isolated))
        object.__setattr__(
            self,
            "hidden_kinds",
            tuple(dict.fromkeys(str(value).strip() for value in self.hidden_kinds if str(value).strip())),
        )
        object.__setattr__(
            self,
            "isolated_kinds",
            tuple(dict.fromkeys(str(value).strip() for value in self.isolated_kinds if str(value).strip())),
        )

    @property
    def is_default(self) -> bool:
        return not (self.hidden or self.hidden_kinds or self.isolated or self.isolated_kinds)

    def accepts(self, owners: Iterable[object]) -> bool:
        refs: tuple[SemanticRef, ...] = tuple(
            ref
            for owner in owners
            for ref in _try_ref(owner)
        )
        if any(ref in self.hidden or ref.kind in self.hidden_kinds for ref in refs):
            return False
        if self.isolated or self.isolated_kinds:
            return any(ref in self.isolated or ref.kind in self.isolated_kinds for ref in refs)
        return True

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden": [value.to_dict() for value in self.hidden],
            "hidden_kinds": list(self.hidden_kinds),
            "isolated": [value.to_dict() for value in self.isolated],
            "isolated_kinds": list(self.isolated_kinds),
        }


def _try_ref(owner: object) -> tuple[SemanticRef, ...]:
    try:
        return (SemanticRef.from_owner(owner),)
    except (TypeError, ValueError):
        return ()


__all__ = ["SemanticRef", "VisibilityState", "semantic_refs"]
