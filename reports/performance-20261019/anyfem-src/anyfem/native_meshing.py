"""Incremental, handle-based orchestration for native component meshing.

The geometry callback in this module deliberately does almost nothing: it
places an ANYgeometry ``ChangeSet`` in a queue.  Resolution, snapshot capture,
prediction, generation, certification, and publication happen only when the
owner explicitly flushes that queue.  This keeps geometry transactions fast
and makes the module usable by a desktop UI without importing a viewport.

Generators receive immutable requests and a cooperative cancellation token.
Publication is guarded by component and control generation numbers, so a
generator that cannot stop promptly can never overwrite a newer mesh.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
import math
import queue
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Protocol, runtime_checkable

from anygeometry import EntityHandle


SCHEMA_VERSION = 3

__all__ = [
    "SCHEMA_VERSION",
    "AutomaticMeshPredictor",
    "CertificationMode",
    "CoalescedChangeSet",
    "ComponentMeshUpdateEvent",
    "ComponentUpdateEvent",
    "ComponentUpdateKind",
    "ControlScope",
    "DirtyComponentResolution",
    "GenerationToken",
    "IncrementalMeshingOrchestrator",
    "MeshBackend",
    "MeshCertification",
    "MeshControl",
    "MeshGenerationRequest",
    "MeshGenerationResult",
    "MeshPrediction",
    "MeshPredictionRequest",
    "MeshPredictor",
    "MeshingRoute",
    "NativeMeshCancelled",
    "NativeMeshControl",
    "NativeMeshControlScope",
    "NativeMeshOrchestrator",
    "NativeMeshResult",
    "NativeMeshSettings",
    "NativeMeshingCancellation",
    "NativeMeshingRuntime",
    "PublishedComponentMesh",
    "coalesce_change_sets",
    "handle_from_dict",
    "handle_to_dict",
]


class CertificationMode(StrEnum):
    """Policy used when deciding whether a completed mesh may be published."""

    INTERACTIVE = "interactive"
    STRICT = "strict"


class MeshBackend(StrEnum):
    """Requested or predicted component meshing implementation."""

    AUTOMATIC = "automatic"
    AUTO = "automatic"
    MAPPED = "mapped"
    NATIVE = "native"
    QUAD_FIRST = "quad_first"


class ComponentUpdateKind(StrEnum):
    """Viewport-neutral lifecycle states emitted per component."""

    QUEUED = "queued"
    STARTED = "started"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    PUBLISHED = "published"
    STALE = "stale"
    REJECTED = "rejected"
    FAILED = "failed"
    REMOVED = "removed"


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite positive number") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _backend(value: MeshBackend | str) -> MeshBackend:
    if value == "auto":
        value = MeshBackend.AUTOMATIC
    try:
        return MeshBackend(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown mesh backend {value!r}") from error


def _mode(value: CertificationMode | str) -> CertificationMode:
    try:
        return CertificationMode(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown certification mode {value!r}") from error


def _schema(payload: Mapping[str, Any], label: str) -> None:
    raw = payload.get("schema", payload.get("version"))
    if isinstance(raw, bool) or raw != SCHEMA_VERSION:
        raise ValueError(f"{label} must use schema {SCHEMA_VERSION}")


@dataclass(frozen=True, slots=True)
class _FrozenObject:
    items: tuple[tuple[str, Any], ...]


def _freeze_json(value: Any, label: str) -> Any:
    if isinstance(value, _FrozenObject):
        return value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must contain finite JSON values")
        return value
    if isinstance(value, Mapping):
        items: list[tuple[str, Any]] = []
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} keys must be non-empty strings")
            items.append((key, _freeze_json(item, label)))
        names = [key for key, _item in items]
        if len(names) != len(set(names)):
            raise ValueError(f"{label} contains duplicate keys")
        return _FrozenObject(tuple(sorted(items)))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, label) for item in value)
    raise TypeError(f"{label} must contain only JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, _FrozenObject):
        return {key: _thaw_json(item) for key, item in value.items}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _parameters(
    value: Mapping[str, Any] | Iterable[tuple[str, Any]], label: str
) -> tuple[tuple[str, Any], ...]:
    if isinstance(value, Mapping):
        source = value.items()
    else:
        try:
            source = tuple(value)
        except TypeError as error:
            raise TypeError(f"{label} must be a mapping or key/value pairs") from error
    made: list[tuple[str, Any]] = []
    for item in source:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError(f"{label} must contain key/value pairs")
        key, raw = item
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        made.append((key, _freeze_json(raw, label)))
    names = [key for key, _raw in made]
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate keys")
    return tuple(sorted(made))


def handle_to_dict(handle: EntityHandle) -> dict[str, Any]:
    """Serialize one model-bound ANYgeometry handle using schema 3."""

    if not isinstance(handle, EntityHandle):
        raise TypeError("handle_to_dict needs an EntityHandle")
    return {
        "schema": SCHEMA_VERSION,
        "model_id": str(handle.model_id),
        "kind": handle.kind,
        "id": handle.id,
    }


def handle_from_dict(payload: Mapping[str, Any]) -> EntityHandle:
    """Deserialize one schema-3 model-bound handle."""

    if not isinstance(payload, Mapping):
        raise TypeError("serialized handle must be an object")
    _schema(payload, "serialized handle")
    allowed = {"schema", "version", "model_id", "kind", "id"}
    unexpected = set(payload) - allowed
    if unexpected:
        raise ValueError(
            "serialized handle has unexpected fields: "
            + ", ".join(sorted(str(item) for item in unexpected))
        )
    try:
        return EntityHandle(payload["model_id"], payload["kind"], payload["id"])
    except KeyError as error:
        raise ValueError(f"serialized handle is missing {error.args[0]!r}") from error


def _normalise_handles(handles: Iterable[EntityHandle]) -> tuple[EntityHandle, ...]:
    try:
        made = tuple(handles)
    except TypeError as error:
        raise TypeError("component scopes must be iterables of EntityHandle values") from error
    if any(not isinstance(handle, EntityHandle) for handle in made):
        raise TypeError("component scopes must contain only EntityHandle values")
    return tuple(sorted(set(made)))


@dataclass(frozen=True, slots=True)
class ControlScope:
    """An immutable direct-entity scope for one meshing control.

    An empty handle tuple denotes a global control.  Descendant expansion is
    intentionally deferred to the snapshot/generator layer; this runtime
    never resolves topology from a geometry transaction callback.
    """

    handles: tuple[EntityHandle, ...] = ()
    include_descendants: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "handles", _normalise_handles(self.handles))
        if not isinstance(self.include_descendants, bool):
            raise TypeError("include_descendants must be a boolean")

    @classmethod
    def create(
        cls,
        handles: Iterable[EntityHandle] = (),
        *,
        include_descendants: bool = False,
    ) -> "ControlScope":
        return cls(tuple(handles), include_descendants)

    def applies_directly_to(self, handle: EntityHandle) -> bool:
        if not isinstance(handle, EntityHandle):
            raise TypeError("scope matching needs an EntityHandle")
        return not self.handles or handle in self.handles

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "handles": [handle_to_dict(handle) for handle in self.handles],
            "include_descendants": self.include_descendants,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlScope":
        if not isinstance(payload, Mapping):
            raise TypeError("serialized control scope must be an object")
        _schema(payload, "serialized control scope")
        try:
            raw_handles = payload["handles"]
        except KeyError as error:
            raise ValueError("serialized control scope is missing 'handles'") from error
        if isinstance(raw_handles, (str, bytes)) or not isinstance(raw_handles, Iterable):
            raise TypeError("serialized control scope handles must be an array")
        return cls(
            tuple(handle_from_dict(item) for item in raw_handles),
            payload.get("include_descendants", False),
        )


@dataclass(frozen=True, slots=True)
class NativeMeshControl:
    """One immutable control with a stable ID and handle-based scope."""

    control_id: str
    scope: ControlScope = ControlScope()
    target_size: float | None = None
    backend: MeshBackend | str | None = None
    parameters: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.control_id, str) or not self.control_id.strip():
            raise ValueError("control_id must be a non-empty string")
        object.__setattr__(self, "control_id", self.control_id.strip())
        if not isinstance(self.scope, ControlScope):
            raise TypeError("control scope must be a ControlScope")
        if self.target_size is not None:
            object.__setattr__(
                self, "target_size", _positive_float(self.target_size, "control target size")
            )
        if self.backend is not None:
            object.__setattr__(self, "backend", _backend(self.backend))
        object.__setattr__(
            self,
            "parameters",
            _parameters(self.parameters, "control parameters"),
        )

    @property
    def id(self) -> str:
        return self.control_id

    @classmethod
    def create(
        cls,
        control_id: str,
        *,
        scope: ControlScope | None = None,
        target_size: float | None = None,
        backend: MeshBackend | str | None = None,
        parameters: Mapping[str, Any] | Iterable[tuple[str, Any]] = (),
    ) -> "NativeMeshControl":
        return cls(
            control_id,
            scope or ControlScope(),
            target_size,
            backend,
            _parameters(parameters, "control parameters"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "id": self.control_id,
            "scope": self.scope.to_dict(),
            "target_size": self.target_size,
            "backend": None if self.backend is None else self.backend.value,
            "parameters": {
                key: _thaw_json(value) for key, value in self.parameters
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NativeMeshControl":
        if not isinstance(payload, Mapping):
            raise TypeError("serialized mesh control must be an object")
        _schema(payload, "serialized mesh control")
        try:
            control_id = payload["id"]
            scope = ControlScope.from_dict(payload["scope"])
        except KeyError as error:
            raise ValueError(
                f"serialized mesh control is missing {error.args[0]!r}"
            ) from error
        return cls.create(
            control_id,
            scope=scope,
            target_size=payload.get("target_size"),
            backend=payload.get("backend"),
            parameters=payload.get("parameters", {}),
        )


@dataclass(frozen=True, slots=True)
class NativeMeshSettings:
    """Complete immutable settings captured by every component request."""

    target_size: float
    element_order: str = "linear"
    backend: MeshBackend | str = MeshBackend.AUTOMATIC
    certification_mode: CertificationMode | str = CertificationMode.INTERACTIVE
    controls: tuple[NativeMeshControl, ...] = ()
    parameters: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_size", _positive_float(self.target_size, "target size")
        )
        if not isinstance(self.element_order, str) or not self.element_order.strip():
            raise ValueError("element_order must be a non-empty string")
        object.__setattr__(self, "element_order", self.element_order.strip())
        object.__setattr__(self, "backend", _backend(self.backend))
        object.__setattr__(
            self, "certification_mode", _mode(self.certification_mode)
        )
        controls = tuple(self.controls)
        if any(not isinstance(control, NativeMeshControl) for control in controls):
            raise TypeError("settings controls must all be NativeMeshControl values")
        identifiers = [control.control_id for control in controls]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("settings contain duplicate control IDs")
        object.__setattr__(self, "controls", controls)
        object.__setattr__(
            self,
            "parameters",
            _parameters(self.parameters, "mesh parameters"),
        )

    @classmethod
    def create(
        cls,
        target_size: float,
        *,
        element_order: str = "linear",
        backend: MeshBackend | str = MeshBackend.AUTOMATIC,
        certification_mode: CertificationMode | str = CertificationMode.INTERACTIVE,
        controls: Iterable[NativeMeshControl] = (),
        parameters: Mapping[str, Any] | Iterable[tuple[str, Any]] = (),
    ) -> "NativeMeshSettings":
        return cls(
            target_size,
            element_order,
            backend,
            certification_mode,
            tuple(controls),
            _parameters(parameters, "mesh parameters"),
        )

    @property
    def mode(self) -> CertificationMode:
        return self.certification_mode

    @property
    def handles(self) -> tuple[EntityHandle, ...]:
        return _normalise_handles(
            handle
            for control in self.controls
            for handle in control.scope.handles
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "target_size": self.target_size,
            "element_order": self.element_order,
            "backend": self.backend.value,
            "certification_mode": self.certification_mode.value,
            "controls": [control.to_dict() for control in self.controls],
            "parameters": {
                key: _thaw_json(value) for key, value in self.parameters
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NativeMeshSettings":
        if not isinstance(payload, Mapping):
            raise TypeError("serialized native mesh settings must be an object")
        _schema(payload, "serialized native mesh settings")
        try:
            target_size = payload["target_size"]
        except KeyError as error:
            raise ValueError(
                "serialized native mesh settings are missing 'target_size'"
            ) from error
        raw_controls = payload.get("controls", ())
        if isinstance(raw_controls, (str, bytes)) or not isinstance(
            raw_controls, Iterable
        ):
            raise TypeError("serialized mesh controls must be an array")
        return cls.create(
            target_size,
            element_order=payload.get("element_order", "linear"),
            backend=payload.get("backend", MeshBackend.AUTOMATIC),
            certification_mode=payload.get(
                "certification_mode", CertificationMode.INTERACTIVE
            ),
            controls=(NativeMeshControl.from_dict(item) for item in raw_controls),
            parameters=payload.get("parameters", {}),
        )


def _sorted_unique(values: Iterable[Any]) -> tuple[Any, ...]:
    made = set(values)
    try:
        return tuple(sorted(made))
    except TypeError:
        return tuple(sorted(made, key=repr))


def _stable_unique(values: Iterable[Any]) -> tuple[Any, ...]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CoalescedChangeSet:
    """The deterministic union of one or more queued geometry changes."""

    revision_before: int
    revision_after: int
    added: tuple[Any, ...] = ()
    removed: tuple[Any, ...] = ()
    modified: tuple[Any, ...] = ()
    replacements: tuple[Any, ...] = ()
    ownership_changes: tuple[Any, ...] = ()
    member_changes: tuple[Any, ...] = ()
    attachment_changes: tuple[Any, ...] = ()
    group_changes: tuple[str, ...] = ()
    tag_changes: tuple[Any, ...] = ()
    affected_aabbs: tuple[Any, ...] = ()
    invalidated_caches: tuple[Any, ...] = ()
    spatial_updates: tuple[Any, ...] = ()
    feature_history_changed: bool = False
    document_settings_changed: bool = False
    source_count: int = 1

    @property
    def changed(self) -> tuple[Any, ...]:
        return _sorted_unique((*self.added, *self.removed, *self.modified))


def coalesce_change_sets(change_sets: Iterable[Any]) -> CoalescedChangeSet:
    """Coalesce queued ANYgeometry ``ChangeSet`` values without model access."""

    changes = tuple(change_sets)
    if not changes:
        raise ValueError("at least one ChangeSet is required")
    for change in changes:
        if not hasattr(change, "revision_before") or not hasattr(
            change, "revision_after"
        ):
            raise TypeError("queued geometry changes must be ChangeSet-like values")

    def values(name: str) -> Iterable[Any]:
        for change in changes:
            yield from tuple(getattr(change, name, ()))

    return CoalescedChangeSet(
        revision_before=int(changes[0].revision_before),
        revision_after=max(int(change.revision_after) for change in changes),
        added=_sorted_unique(values("added")),
        removed=_sorted_unique(values("removed")),
        modified=_sorted_unique(values("modified")),
        replacements=_stable_unique(values("replacements")),
        ownership_changes=_sorted_unique(values("ownership_changes")),
        member_changes=_sorted_unique(values("member_changes")),
        attachment_changes=_sorted_unique(values("attachment_changes")),
        group_changes=_sorted_unique(values("group_changes")),
        tag_changes=_sorted_unique(values("tag_changes")),
        affected_aabbs=_stable_unique(values("affected_aabbs")),
        invalidated_caches=_sorted_unique(values("invalidated_caches")),
        spatial_updates=_sorted_unique(values("spatial_updates")),
        feature_history_changed=any(
            bool(getattr(change, "feature_history_changed", False))
            for change in changes
        ),
        document_settings_changed=any(
            bool(getattr(change, "document_settings_changed", False))
            for change in changes
        ),
        source_count=len(changes),
    )


@dataclass(frozen=True, slots=True)
class DirtyComponentResolution:
    """Active components to regenerate and components removed from the model."""

    dirty: tuple[EntityHandle, ...] = ()
    removed: tuple[EntityHandle, ...] = ()

    def __post_init__(self) -> None:
        removed = _normalise_handles(self.removed)
        dirty = tuple(
            handle for handle in _normalise_handles(self.dirty) if handle not in removed
        )
        object.__setattr__(self, "dirty", dirty)
        object.__setattr__(self, "removed", removed)

    @property
    def components(self) -> tuple[EntityHandle, ...]:
        return self.dirty


@dataclass(frozen=True, slots=True)
class GenerationToken:
    """Optimistic publication token captured by one background request."""

    component_generation: int
    control_generation: int

    def __post_init__(self) -> None:
        if self.component_generation < 0 or self.control_generation < 0:
            raise ValueError("generation numbers cannot be negative")

    @property
    def component(self) -> int:
        return self.component_generation

    @property
    def controls(self) -> int:
        return self.control_generation


@dataclass(frozen=True, slots=True)
class MeshPredictionRequest:
    component: EntityHandle
    token: GenerationToken
    snapshot: Any
    settings: NativeMeshSettings
    controls: tuple[NativeMeshControl, ...]
    changes: CoalescedChangeSet | None = None


@dataclass(frozen=True, slots=True)
class MeshPrediction:
    backend: MeshBackend | str
    reason: str = ""
    confidence: float = 1.0

    def __post_init__(self) -> None:
        backend = _backend(self.backend)
        if backend is MeshBackend.AUTOMATIC:
            raise ValueError("a predictor must choose mapped or native meshing")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("prediction confidence must be between zero and one")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "reason", str(self.reason))


@runtime_checkable
class MeshPredictor(Protocol):
    """Automatic predictor contract used by :class:`NativeMeshingRuntime`."""

    def predict(self, request: MeshPredictionRequest) -> MeshPrediction:
        ...


def _prediction_from_signal(signal: Any, reason: str) -> MeshPrediction | None:
    if isinstance(signal, MeshPrediction):
        return signal
    if isinstance(signal, tuple) and len(signal) == 2:
        signal, supplied_reason = signal
        reason = str(supplied_reason)
    if isinstance(signal, bool):
        return MeshPrediction(
            MeshBackend.MAPPED if signal else MeshBackend.NATIVE,
            reason,
            0.75,
        )
    if isinstance(signal, (MeshBackend, str)):
        backend = _backend(signal)
        if backend is MeshBackend.AUTOMATIC:
            return None
        return MeshPrediction(backend, reason)
    return None


class AutomaticMeshPredictor:
    """Small mapped/native contract with a conservative native default.

    A snapshot can expose ``predict_mesh_backend(component)`` or one of
    ``mapped_mesh_eligible``/``mapped_eligible`` as a mapping key or attribute.
    Applications with richer topology knowledge can instead inject any object
    implementing ``predict(request)`` or a callable with the same contract.
    """

    def predict(self, request: MeshPredictionRequest) -> MeshPrediction:
        snapshot = request.snapshot
        method = getattr(snapshot, "predict_mesh_backend", None)
        if callable(method):
            prediction = _prediction_from_signal(
                method(request.component), "snapshot backend prediction"
            )
            if prediction is not None:
                return prediction

        if isinstance(snapshot, Mapping):
            for name in ("mesh_backend", "mapped_mesh_eligible", "mapped_eligible"):
                if name in snapshot:
                    prediction = _prediction_from_signal(
                        snapshot[name], f"snapshot {name} signal"
                    )
                    if prediction is not None:
                        return prediction
        else:
            for name in ("mesh_backend", "mapped_mesh_eligible", "mapped_eligible"):
                if not hasattr(snapshot, name):
                    continue
                signal = getattr(snapshot, name)
                if callable(signal):
                    signal = signal()
                prediction = _prediction_from_signal(
                    signal, f"snapshot {name} signal"
                )
                if prediction is not None:
                    return prediction

        return MeshPrediction(
            MeshBackend.NATIVE,
            "no mapped eligibility signal; native is the safe default",
            1.0,
        )

    def __call__(self, request: MeshPredictionRequest) -> MeshPrediction:
        return self.predict(request)


class NativeMeshCancelled(RuntimeError):
    """Raised at a cooperative cancellation point inside a generator."""


class NativeMeshingCancellation:
    """Thread-safe cooperative cancellation token supplied to generators."""

    __slots__ = ("_event", "_lock", "_reason")

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = ""

    def cancel(self, reason: str = "meshing cancellation requested") -> bool:
        with self._lock:
            changed = not self._event.is_set()
            if changed:
                self._reason = str(reason)
                self._event.set()
            return changed

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def cancelled(self) -> bool:
        return self.is_cancelled

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def raise_if_cancelled(self, stage: str = "mesh generation") -> None:
        if self.is_cancelled:
            detail = f": {self.reason}" if self.reason else ""
            raise NativeMeshCancelled(f"{stage} cancelled{detail}")


@dataclass(frozen=True, slots=True)
class MeshGenerationRequest:
    """All immutable inputs for one component generation attempt."""

    component: EntityHandle
    token: GenerationToken
    snapshot: Any
    settings: NativeMeshSettings
    controls: tuple[NativeMeshControl, ...]
    backend: MeshBackend
    prediction: MeshPrediction
    cancellation: NativeMeshingCancellation
    changes: CoalescedChangeSet | None = None

    @property
    def certification_mode(self) -> CertificationMode:
        return self.settings.certification_mode


@dataclass(frozen=True, slots=True)
class NativeMeshResult:
    """Generator result before publication policy is applied."""

    mesh: Any
    valid: bool = True
    certified: bool = False
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "certified", bool(self.certified))
        object.__setattr__(
            self, "diagnostics", tuple(str(item) for item in self.diagnostics)
        )

    @classmethod
    def certified_result(
        cls, mesh: Any, diagnostics: Iterable[str] = ()
    ) -> "NativeMeshResult":
        return cls(mesh, valid=True, certified=True, diagnostics=tuple(diagnostics))


@dataclass(frozen=True, slots=True)
class MeshCertification:
    """Optional independent certification response."""

    certified: bool
    valid: bool = True
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "certified", bool(self.certified))
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(
            self, "diagnostics", tuple(str(item) for item in self.diagnostics)
        )


@dataclass(frozen=True, slots=True)
class PublishedComponentMesh:
    """One atomically published last-valid component mesh."""

    component: EntityHandle
    mesh: Any
    token: GenerationToken
    backend: MeshBackend
    certified: bool
    publication_sequence: int


@dataclass(frozen=True, slots=True)
class ComponentMeshUpdateEvent:
    """A render-toolkit-independent component lifecycle event."""

    sequence: int
    component: EntityHandle
    kind: ComponentUpdateKind
    token: GenerationToken
    backend: MeshBackend | None = None
    message: str = ""
    retained_previous: bool = False

    @property
    def component_generation(self) -> int:
        return self.token.component_generation

    @property
    def control_generation(self) -> int:
        return self.token.control_generation


@dataclass(slots=True)
class _ComponentJob:
    component: EntityHandle
    token: GenerationToken
    snapshot: Any
    settings: NativeMeshSettings
    controls: tuple[NativeMeshControl, ...]
    changes: CoalescedChangeSet | None
    cancellation: NativeMeshingCancellation


@dataclass(frozen=True, slots=True)
class _WorkerOutcome:
    backend: MeshBackend | None = None
    result: NativeMeshResult | None = None
    error: BaseException | None = None
    cancelled: bool = False


def _normalise_resolution(value: Any) -> DirtyComponentResolution:
    if value is None:
        return DirtyComponentResolution()
    if isinstance(value, DirtyComponentResolution):
        return value
    if isinstance(value, Mapping):
        return DirtyComponentResolution(
            tuple(value.get("dirty", value.get("components", ()))),
            tuple(value.get("removed", ())),
        )
    if isinstance(value, EntityHandle):
        return DirtyComponentResolution((value,))
    return DirtyComponentResolution(tuple(value))


def _normalise_generation_result(value: Any) -> NativeMeshResult:
    if isinstance(value, NativeMeshResult):
        return value
    if isinstance(value, Mapping) and "mesh" in value and (
        "certified" in value or "valid" in value or "diagnostics" in value
    ):
        return NativeMeshResult(
            value["mesh"],
            value.get("valid", True),
            value.get("certified", False),
            tuple(value.get("diagnostics", ())),
        )
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], bool):
        return NativeMeshResult(value[0], certified=value[1])
    return NativeMeshResult(value, valid=value is not None)


class NativeMeshingRuntime:
    """Coordinate incremental component meshing without UI dependencies.

    ``resolve_dirty_components`` runs only from :meth:`flush_changes` and must
    return handles (or :class:`DirtyComponentResolution`).  ``capture_component``
    should return an owned immutable snapshot.  ``generate_component`` runs on
    a bounded worker pool and receives :class:`MeshGenerationRequest`.
    """

    def __init__(
        self,
        settings: NativeMeshSettings,
        *,
        resolve_dirty_components: Callable[[CoalescedChangeSet], Any],
        generate_component: Callable[[MeshGenerationRequest], Any],
        capture_component: Callable[[EntityHandle], Any] | None = None,
        predictor: MeshPredictor | Callable[[MeshPredictionRequest], Any] | None = None,
        certifier: Callable[[MeshGenerationRequest, NativeMeshResult], Any] | None = None,
        max_background_jobs: int = 2,
    ) -> None:
        if not isinstance(settings, NativeMeshSettings):
            raise TypeError("settings must be NativeMeshSettings")
        if not callable(resolve_dirty_components) and not callable(
            getattr(resolve_dirty_components, "resolve", None)
        ):
            raise TypeError("resolve_dirty_components must be callable")
        if not callable(generate_component) and not callable(
            getattr(generate_component, "generate", None)
        ):
            raise TypeError("generate_component must be callable")
        if capture_component is not None and not callable(capture_component):
            raise TypeError("capture_component must be callable")
        workers = _positive_integer(max_background_jobs, "max_background_jobs")

        self._settings = settings
        self._resolver = resolve_dirty_components
        self._generator = generate_component
        self._capture = capture_component
        self._predictor = predictor or AutomaticMeshPredictor()
        self._certifier = certifier
        self._max_background_jobs = workers
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="anyfem-native-mesh"
        )

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._changes: deque[Any] = deque()
        self._pending: "OrderedDict[EntityHandle, _ComponentJob]" = OrderedDict()
        self._active: dict[Future[_WorkerOutcome], _ComponentJob] = {}
        self._component_generations: dict[EntityHandle, int] = {}
        self._control_generation = 1
        self._known_components: set[EntityHandle] = set()
        self._publications: dict[EntityHandle, PublishedComponentMesh] = {}
        self._events: "queue.Queue[ComponentMeshUpdateEvent]" = queue.Queue()
        self._event_sequence = 0
        self._closed = False

    @property
    def settings(self) -> NativeMeshSettings:
        with self._lock:
            return self._settings

    @property
    def control_generation(self) -> int:
        with self._lock:
            return self._control_generation

    @property
    def max_background_jobs(self) -> int:
        return self._max_background_jobs

    @property
    def queued_change_count(self) -> int:
        with self._lock:
            return len(self._changes)

    @property
    def active_job_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def pending_job_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def busy(self) -> bool:
        with self._lock:
            return bool(self._active or self._pending)

    @property
    def idle(self) -> bool:
        with self._lock:
            return not self._active and not self._pending and not self._changes

    def generation(self, component: EntityHandle) -> GenerationToken:
        if not isinstance(component, EntityHandle):
            raise TypeError("generation lookup needs an EntityHandle")
        with self._lock:
            return self._token_locked(component)

    def on_geometry_change(self, change_set: Any) -> int:
        """Queue a ChangeSet and return the queue length; perform no resolution."""

        with self._lock:
            self._require_open_locked()
            self._changes.append(change_set)
            self._condition.notify_all()
            return len(self._changes)

    queue_change_set = on_geometry_change
    queue_changes = on_geometry_change

    def flush_changes(self) -> DirtyComponentResolution:
        """Coalesce queued changes, resolve dirty handles, and schedule snapshots."""

        with self._lock:
            self._require_open_locked()
            if not self._changes:
                return DirtyComponentResolution()
            source = tuple(self._changes)
            self._changes.clear()

        try:
            coalesced = coalesce_change_sets(source)
            resolver = getattr(self._resolver, "resolve", self._resolver)
            resolution = _normalise_resolution(resolver(coalesced))
        except BaseException:
            with self._lock:
                for change in reversed(source):
                    self._changes.appendleft(change)
                self._condition.notify_all()
            raise

        if resolution.removed:
            self._remove_components(resolution.removed)
        if resolution.dirty:
            self._schedule_components(
                resolution.dirty,
                bump_component=True,
                reason="superseded by a newer geometry generation",
                changes=coalesced,
            )
        return resolution

    process_changes = flush_changes
    drain_changes = flush_changes

    def request_remesh(
        self, components: EntityHandle | Iterable[EntityHandle]
    ) -> tuple[EntityHandle, ...]:
        """Explicitly invalidate and schedule one or more component handles."""

        if isinstance(components, EntityHandle):
            made = (components,)
        else:
            made = _normalise_handles(components)
        self._schedule_components(
            made,
            bump_component=True,
            reason="superseded by an explicit remesh",
            changes=None,
        )
        return made

    schedule = request_remesh
    invalidate = request_remesh

    def set_settings(self, settings: NativeMeshSettings) -> int:
        """Replace controls atomically and remesh every known component."""

        if not isinstance(settings, NativeMeshSettings):
            raise TypeError("settings must be NativeMeshSettings")
        with self._lock:
            self._require_open_locked()
            if settings == self._settings:
                return self._control_generation
            self._settings = settings
            self._control_generation += 1
            components = tuple(sorted(self._known_components))

        if components:
            self._schedule_components(
                components,
                bump_component=False,
                reason="superseded by newer mesh controls",
                changes=None,
            )
        return self.control_generation

    update_settings = set_settings

    def cancel_component(self, component: EntityHandle) -> bool:
        """Cancel pending/running work while retaining its last valid mesh."""

        if not isinstance(component, EntityHandle):
            raise TypeError("component cancellation needs an EntityHandle")
        with self._lock:
            pending = self._pending.pop(component, None)
            active = self._active_job_locked(component)
            if pending is None and active is None:
                return False
            self._component_generations[component] = (
                self._component_generations.get(component, 0) + 1
            )
            token = self._token_locked(component)
            retained = component in self._publications
            if pending is not None:
                pending.cancellation.cancel("cancelled by caller before start")
                self._emit_locked(
                    component,
                    ComponentUpdateKind.CANCELLED,
                    token,
                    message="component meshing cancelled before start",
                    retained_previous=retained,
                )
            if active is not None and active.cancellation.cancel(
                "cancelled by caller"
            ):
                self._emit_locked(
                    component,
                    ComponentUpdateKind.CANCELLING,
                    token,
                    message="component meshing cancellation requested",
                    retained_previous=retained,
                )
            self._pump_locked()
            self._condition.notify_all()
            return True

    cancel = cancel_component

    def poll_events(self, limit: int | None = None) -> list[ComponentMeshUpdateEvent]:
        events: list[ComponentMeshUpdateEvent] = []
        while limit is None or len(events) < limit:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                break
        return events

    poll = poll_events

    def publication(self, component: EntityHandle) -> PublishedComponentMesh | None:
        if not isinstance(component, EntityHandle):
            raise TypeError("publication lookup needs an EntityHandle")
        with self._lock:
            return self._publications.get(component)

    def published_mesh(self, component: EntityHandle) -> Any | None:
        publication = self.publication(component)
        return None if publication is None else publication.mesh

    def publications(self) -> Mapping[EntityHandle, PublishedComponentMesh]:
        """Return an immutable atomic snapshot of all component publications."""

        with self._lock:
            return MappingProxyType(dict(self._publications))

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        """Wait for scheduled jobs (not unflushed ChangeSets) to finish."""

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while self._active or self._pending:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            return True

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = tuple(self._pending.values())
            self._pending.clear()
            active = tuple(self._active.values())
            for job in pending:
                job.cancellation.cancel("native meshing runtime is closing")
                self._emit_locked(
                    job.component,
                    ComponentUpdateKind.CANCELLED,
                    job.token,
                    message="component meshing cancelled during shutdown",
                    retained_previous=job.component in self._publications,
                )
            for job in active:
                job.cancellation.cancel("native meshing runtime is closing")
            self._condition.notify_all()
        self._executor.shutdown(wait=wait, cancel_futures=True)

    close = shutdown

    def __enter__(self) -> "NativeMeshingRuntime":
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.shutdown()

    def _require_open_locked(self) -> None:
        if self._closed:
            raise RuntimeError("native meshing runtime is closed")

    def _token_locked(self, component: EntityHandle) -> GenerationToken:
        return GenerationToken(
            self._component_generations.get(component, 0),
            self._control_generation,
        )

    def _active_job_locked(self, component: EntityHandle) -> _ComponentJob | None:
        for job in self._active.values():
            if job.component == component:
                return job
        return None

    def _emit_locked(
        self,
        component: EntityHandle,
        kind: ComponentUpdateKind,
        token: GenerationToken,
        *,
        backend: MeshBackend | None = None,
        message: str = "",
        retained_previous: bool = False,
    ) -> int:
        self._event_sequence += 1
        self._events.put(
            ComponentMeshUpdateEvent(
                self._event_sequence,
                component,
                kind,
                token,
                backend,
                str(message),
                retained_previous,
            )
        )
        return self._event_sequence

    def _remove_components(self, components: Iterable[EntityHandle]) -> None:
        with self._lock:
            for component in _normalise_handles(components):
                self._component_generations[component] = (
                    self._component_generations.get(component, 0) + 1
                )
                pending = self._pending.pop(component, None)
                if pending is not None:
                    pending.cancellation.cancel("component was removed")
                active = self._active_job_locked(component)
                if active is not None:
                    active.cancellation.cancel("component was removed")
                self._known_components.discard(component)
                self._publications.pop(component, None)
                self._emit_locked(
                    component,
                    ComponentUpdateKind.REMOVED,
                    self._token_locked(component),
                    message="component removed; publication withdrawn",
                )
            self._pump_locked()
            self._condition.notify_all()

    def _schedule_components(
        self,
        components: Iterable[EntityHandle],
        *,
        bump_component: bool,
        reason: str,
        changes: CoalescedChangeSet | None,
    ) -> None:
        handles = _normalise_handles(components)
        captures: list[tuple[EntityHandle, GenerationToken, NativeMeshSettings]] = []
        with self._lock:
            self._require_open_locked()
            for component in handles:
                self._known_components.add(component)
                if bump_component:
                    self._component_generations[component] = (
                        self._component_generations.get(component, 0) + 1
                    )
                token = self._token_locked(component)
                old_pending = self._pending.pop(component, None)
                if old_pending is not None:
                    old_pending.cancellation.cancel(reason)
                    self._emit_locked(
                        component,
                        ComponentUpdateKind.STALE,
                        old_pending.token,
                        message="pending component request was coalesced",
                        retained_previous=component in self._publications,
                    )
                active = self._active_job_locked(component)
                if active is not None:
                    active.cancellation.cancel(reason)
                captures.append((component, token, self._settings))

        for component, token, settings in captures:
            try:
                snapshot = component if self._capture is None else self._capture(component)
            except BaseException as error:
                with self._lock:
                    if self._token_locked(component) == token:
                        self._emit_locked(
                            component,
                            ComponentUpdateKind.FAILED,
                            token,
                            message=f"component snapshot failed: {type(error).__name__}: {error}",
                            retained_previous=component in self._publications,
                        )
                        self._condition.notify_all()
                continue

            job = _ComponentJob(
                component,
                token,
                snapshot,
                settings,
                settings.controls,
                changes,
                NativeMeshingCancellation(),
            )
            with self._lock:
                if self._closed:
                    job.cancellation.cancel("native meshing runtime is closing")
                    continue
                if self._token_locked(component) != token:
                    job.cancellation.cancel("snapshot became stale before queueing")
                    self._emit_locked(
                        component,
                        ComponentUpdateKind.STALE,
                        token,
                        message="component snapshot became stale before queueing",
                        retained_previous=component in self._publications,
                    )
                    continue
                self._pending[component] = job
                self._emit_locked(
                    component,
                    ComponentUpdateKind.QUEUED,
                    token,
                    message="component meshing queued",
                    retained_previous=component in self._publications,
                )
                self._pump_locked()
                self._condition.notify_all()

    def _pump_locked(self) -> None:
        if self._closed:
            return
        while self._pending and len(self._active) < self._max_background_jobs:
            active_components = {job.component for job in self._active.values()}
            selected: EntityHandle | None = None
            for component in self._pending:
                if component not in active_components:
                    selected = component
                    break
            if selected is None:
                return
            job = self._pending.pop(selected)
            if self._token_locked(job.component) != job.token:
                job.cancellation.cancel("request token is stale")
                self._emit_locked(
                    job.component,
                    ComponentUpdateKind.STALE,
                    job.token,
                    message="stale component request rejected before start",
                    retained_previous=job.component in self._publications,
                )
                continue
            future = self._executor.submit(self._run_job, job)
            self._active[future] = job
            future.add_done_callback(self._complete_job)

    def _predict(self, request: MeshPredictionRequest) -> MeshPrediction:
        if request.settings.backend is not MeshBackend.AUTOMATIC:
            return MeshPrediction(
                request.settings.backend,
                "backend selected explicitly by mesh settings",
            )
        predictor = self._predictor
        target = getattr(predictor, "predict", predictor)
        prediction = target(request)
        normalised = _prediction_from_signal(
            prediction, "application mapped/native prediction"
        )
        if normalised is None:
            raise ValueError("automatic predictor did not choose mapped or native")
        return normalised

    def _run_job(self, job: _ComponentJob) -> _WorkerOutcome:
        backend: MeshBackend | None = None
        try:
            job.cancellation.raise_if_cancelled("mesh prediction")
            prediction_request = MeshPredictionRequest(
                job.component,
                job.token,
                job.snapshot,
                job.settings,
                job.controls,
                job.changes,
            )
            prediction = self._predict(prediction_request)
            backend = prediction.backend
            request = MeshGenerationRequest(
                job.component,
                job.token,
                job.snapshot,
                job.settings,
                job.controls,
                backend,
                prediction,
                job.cancellation,
                job.changes,
            )
            with self._lock:
                self._emit_locked(
                    job.component,
                    ComponentUpdateKind.STARTED,
                    job.token,
                    backend=backend,
                    message=f"{backend.value} component meshing started",
                    retained_previous=job.component in self._publications,
                )
            job.cancellation.raise_if_cancelled("mesh generation")
            generator = getattr(self._generator, "generate", self._generator)
            result = _normalise_generation_result(generator(request))
            job.cancellation.raise_if_cancelled("mesh certification")
            if self._certifier is not None:
                certification = self._certifier(request, result)
                if isinstance(certification, MeshCertification):
                    result = NativeMeshResult(
                        result.mesh,
                        valid=result.valid and certification.valid,
                        certified=certification.certified,
                        diagnostics=(*result.diagnostics, *certification.diagnostics),
                    )
                elif isinstance(certification, bool):
                    result = NativeMeshResult(
                        result.mesh,
                        valid=result.valid,
                        certified=certification,
                        diagnostics=result.diagnostics,
                    )
                elif certification is not None:
                    raise TypeError(
                        "certifier must return bool, MeshCertification, or None"
                    )
            job.cancellation.raise_if_cancelled("mesh completion")
            return _WorkerOutcome(backend=backend, result=result)
        except NativeMeshCancelled:
            return _WorkerOutcome(backend=backend, cancelled=True)
        except BaseException as error:
            return _WorkerOutcome(backend=backend, error=error)

    def _complete_job(self, future: Future[_WorkerOutcome]) -> None:
        try:
            outcome = future.result()
        except BaseException as error:
            outcome = _WorkerOutcome(error=error)

        with self._lock:
            job = self._active.pop(future, None)
            if job is None:
                self._condition.notify_all()
                return
            current = self._token_locked(job.component)
            retained = job.component in self._publications
            reason = job.cancellation.reason

            if current != job.token:
                explicitly_cancelled = reason.startswith("cancelled by caller")
                kind = (
                    ComponentUpdateKind.CANCELLED
                    if explicitly_cancelled
                    else ComponentUpdateKind.STALE
                )
                message = (
                    "cancelled component result rejected"
                    if explicitly_cancelled
                    else "stale component result rejected"
                )
                self._emit_locked(
                    job.component,
                    kind,
                    job.token,
                    backend=outcome.backend,
                    message=message,
                    retained_previous=retained,
                )
            elif job.cancellation.is_cancelled or outcome.cancelled:
                self._emit_locked(
                    job.component,
                    ComponentUpdateKind.CANCELLED,
                    job.token,
                    backend=outcome.backend,
                    message="component meshing cancelled",
                    retained_previous=retained,
                )
            elif outcome.error is not None:
                self._emit_locked(
                    job.component,
                    ComponentUpdateKind.FAILED,
                    job.token,
                    backend=outcome.backend,
                    message=(
                        f"component meshing failed: {type(outcome.error).__name__}: "
                        f"{outcome.error}"
                    ),
                    retained_previous=retained,
                )
            elif outcome.result is None or outcome.result.mesh is None or not outcome.result.valid:
                diagnostics = "\n".join(
                    () if outcome.result is None else outcome.result.diagnostics
                )
                self._emit_locked(
                    job.component,
                    ComponentUpdateKind.REJECTED,
                    job.token,
                    backend=outcome.backend,
                    message=diagnostics or "invalid component mesh rejected",
                    retained_previous=retained,
                )
            elif (
                job.settings.certification_mode is CertificationMode.STRICT
                and not outcome.result.certified
            ):
                self._emit_locked(
                    job.component,
                    ComponentUpdateKind.REJECTED,
                    job.token,
                    backend=outcome.backend,
                    message="strict mode rejected an uncertified component mesh",
                    retained_previous=retained,
                )
            else:
                assert outcome.backend is not None
                sequence = self._event_sequence + 1
                self._publications[job.component] = PublishedComponentMesh(
                    job.component,
                    outcome.result.mesh,
                    job.token,
                    outcome.backend,
                    outcome.result.certified,
                    sequence,
                )
                self._emit_locked(
                    job.component,
                    ComponentUpdateKind.PUBLISHED,
                    job.token,
                    backend=outcome.backend,
                    message="component mesh published atomically",
                    retained_previous=retained,
                )

            self._pump_locked()
            self._condition.notify_all()


# Stable, descriptive aliases for integrations that use orchestration wording.
NativeMeshControlScope = ControlScope
MeshControl = NativeMeshControl
MeshGenerationResult = NativeMeshResult
MeshingRoute = MeshBackend
ComponentUpdateEvent = ComponentMeshUpdateEvent
NativeMeshOrchestrator = NativeMeshingRuntime
IncrementalMeshingOrchestrator = NativeMeshingRuntime
