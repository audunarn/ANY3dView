"""Vendor-neutral, strictly validated commands for interactive 3D viewers."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from enum import IntEnum
import heapq
import math
from threading import Lock
from typing import Any, Callable, Mapping, Sequence
from uuid import UUID, uuid4

from .clipping import SectionPlane
from .contracts import ViewerState
from .core import Point3D, as_point
from .semantic import SemanticRef, VisibilityState, semantic_refs


COMMAND_SCHEMA = "any3dview.viewer_commands"
COMMAND_SCHEMA_VERSION = 1


class ViewerCommandPriority(IntEnum):
    """Trusted queue priorities; model-supplied payloads cannot set these."""

    UNDO = 0
    USER = 10
    AI = 20
    BACKGROUND = 30


@dataclass(frozen=True, slots=True)
class ViewerCommand:
    operation: str
    params: Mapping[str, Any] = field(default_factory=dict)
    command_id: str = field(default_factory=lambda: str(uuid4()))
    source: str = "application"

    def __post_init__(self) -> None:
        operation = str(self.operation).strip()
        if operation not in _OPERATIONS:
            raise ValueError(f"unknown viewer operation: {operation!r}")
        try:
            command_id = str(UUID(str(self.command_id)))
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError("command_id must be a UUID") from error
        if not isinstance(self.params, Mapping):
            raise TypeError("command params must be a JSON object")
        _json_value(dict(self.params), "params")
        source = str(self.source).strip() or "application"
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "params", _validate_params(operation, dict(self.params)))
        object.__setattr__(self, "command_id", command_id)
        object.__setattr__(self, "source", source)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ViewerCommand":
        allowed = {"operation", "params", "command_id", "source"}
        extras = set(value).difference(allowed)
        if extras:
            raise ValueError(f"unknown viewer command fields: {sorted(extras)}")
        return cls(
            operation=value.get("operation", ""),
            params=value.get("params", {}),
            command_id=value.get("command_id", str(uuid4())),
            source=value.get("source", "application"),
        )


@dataclass(frozen=True, slots=True)
class ViewerCommandDescriptor:
    operation: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    permission_category: str = "reversible_edit"
    confirmation_requirement: str = "none"
    reversibility: str = "undoable"
    ai_exposable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "version": self.version,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "units": {},
            "error_codes": [
                "invalid_params", "unknown_target", "target_hidden",
                "queue_full", "history_empty", "execution_failed",
            ],
            "warning_codes": ["observation_truncated", "stale_reference_removed"],
            "permission_category": self.permission_category,
            "preview_support": False,
            "confirmation_requirement": self.confirmation_requirement,
            "reversibility": self.reversibility,
            "estimated_cost_class": "trivial",
            "produced_resource_types": [],
            "required_package": "ANY3dView",
            "minimum_package_version": "0.5.2",
            "ai_exposable": self.ai_exposable,
            "cloud_transfer_possible": False,
            "available": True,
            "unavailability_reason": None,
        }


@dataclass(frozen=True, slots=True)
class ViewerObservation:
    backend: str
    viewport_size: tuple[int, int]
    state: Mapping[str, Any]
    selection: tuple[SemanticRef, ...]
    visibility: VisibilityState
    undo_available: bool
    redo_available: bool
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "viewport_size": list(self.viewport_size),
            "state": dict(self.state),
            "selection": [value.to_dict() for value in self.selection],
            "visibility": self.visibility.to_dict(),
            "undo_available": self.undo_available,
            "redo_available": self.redo_available,
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class ViewerCommandResult:
    command_id: str
    operation: str
    status: str
    changed: bool = False
    result: Mapping[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "operation": self.operation,
            "status": self.status,
            "changed": self.changed,
            "result": dict(self.result),
            "error": None if self.error_code is None else {
                "code": self.error_code,
                "message": self.error_message or "",
            },
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class _Snapshot:
    view: ViewerState
    selection: tuple[SemanticRef, ...]
    visibility: VisibilityState


class ViewerCommandController:
    """Prioritized, transactional command executor bound to one viewer."""

    def __init__(
        self,
        viewer: object,
        *,
        history_limit: int = 50,
        queue_limit: int = 128,
        entity_exists: Callable[[SemanticRef], bool] | None = None,
        selection_sink: Callable[[tuple[SemanticRef, ...]], None] | None = None,
    ) -> None:
        if not hasattr(viewer, "export_view_state") or not hasattr(viewer, "apply_view_state"):
            raise TypeError("viewer must implement export_view_state/apply_view_state")
        self.viewer = viewer
        self.history_limit = max(1, int(history_limit))
        self.queue_limit = max(1, int(queue_limit))
        self.entity_exists = entity_exists
        self.selection_sink = selection_sink
        self._undo: list[_Snapshot] = []
        self._redo: list[_Snapshot] = []
        self._selection: tuple[SemanticRef, ...] = tuple(
            getattr(viewer, "semantic_selection", ())
        )
        self._visibility = getattr(viewer, "visibility_state", VisibilityState())
        self._queue: list[tuple[int, int, ViewerCommand, Future[ViewerCommandResult]]] = []
        self._sequence = 0
        self._lock = Lock()
        self._scheduled = False
        self._interacting = False
        self._model_identity: object = None

    @property
    def undo_available(self) -> bool:
        return bool(self._undo)

    @property
    def redo_available(self) -> bool:
        return bool(self._redo)

    @property
    def pending_commands(self) -> int:
        with self._lock:
            return len(self._queue)

    def set_interacting(self, active: bool) -> None:
        self._interacting = bool(active)
        if not self._interacting:
            self._schedule_drain()

    def clear_history(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def set_model_identity(self, identity: object) -> None:
        """Clear local history only when the application model changes."""

        if identity != self._model_identity:
            self.clear_history()
            self._model_identity = identity

    def revalidate(self) -> tuple[SemanticRef, ...]:
        """Remove stale same-model selection and visibility references."""

        if self.entity_exists is None:
            return ()
        stale = tuple(
            value for value in (
                *self._get_selection(),
                *self._get_visibility().hidden,
                *self._get_visibility().isolated,
            )
            if not self.entity_exists(value)
        )
        if not stale:
            return ()
        stale_set = set(stale)
        visibility = self._get_visibility()
        self._set_visibility(VisibilityState(
            tuple(value for value in visibility.hidden if value not in stale_set),
            visibility.hidden_kinds,
            tuple(value for value in visibility.isolated if value not in stale_set),
            visibility.isolated_kinds,
        ))
        self._set_selection(tuple(
            value for value in self._get_selection() if value not in stale_set
        ))
        return stale

    def submit(
        self,
        command: ViewerCommand | Mapping[str, Any],
        *,
        priority: ViewerCommandPriority = ViewerCommandPriority.AI,
    ) -> Future[ViewerCommandResult]:
        future: Future[ViewerCommandResult] = Future()
        try:
            made = command if isinstance(command, ViewerCommand) else ViewerCommand.from_dict(command)
        except Exception as error:  # strict boundary, returned rather than raised asynchronously
            future.set_result(ViewerCommandResult(
                str(uuid4()), str(getattr(command, "operation", "invalid")), "error",
                error_code="invalid_params", error_message=str(error),
            ))
            return future
        with self._lock:
            if len(self._queue) >= self.queue_limit:
                future.set_result(ViewerCommandResult(
                    made.command_id, made.operation, "error",
                    error_code="queue_full", error_message="viewer command queue is full",
                ))
                return future
            self._sequence += 1
            heapq.heappush(self._queue, (int(priority), self._sequence, made, future))
        self._schedule_drain()
        return future

    def _schedule_drain(self) -> None:
        with self._lock:
            if self._scheduled or not self._queue:
                return
            if (
                self._interacting
                and self._queue[0][0] >= int(ViewerCommandPriority.AI)
            ):
                return
            self._scheduled = True
        submit = getattr(self.viewer, "submit_update", None)
        if callable(submit):
            submit(self.drain)
        else:
            self.drain()

    def drain(self, limit: int = 32) -> int:
        completed = 0
        while completed < max(1, int(limit)):
            with self._lock:
                if not self._queue:
                    self._scheduled = False
                    break
                priority, sequence, command, future = heapq.heappop(self._queue)
                if self._interacting and priority >= int(ViewerCommandPriority.AI):
                    heapq.heappush(self._queue, (priority, sequence, command, future))
                    self._scheduled = False
                    break
            if not future.cancelled():
                future.set_result(self.execute(command))
            completed += 1
        with self._lock:
            more = bool(self._queue) and not self._interacting
            if more:
                self._scheduled = False
        if more:
            self._schedule_drain()
        return completed

    def execute(self, command: ViewerCommand | Mapping[str, Any]) -> ViewerCommandResult:
        try:
            made = command if isinstance(command, ViewerCommand) else ViewerCommand.from_dict(command)
        except Exception as error:
            return ViewerCommandResult(
                str(uuid4()), "invalid", "error",
                error_code="invalid_params", error_message=str(error),
            )
        try:
            if made.operation == "viewer.observe":
                return ViewerCommandResult(
                    made.command_id, made.operation, "ok",
                    result=self.observe().to_dict(),
                )
            if made.operation == "viewer.undo":
                return self._undo_command(made, redo=False)
            if made.operation == "viewer.redo":
                return self._undo_command(made, redo=True)
            before = self._snapshot()
            self._apply(made.operation, dict(made.params))
            after = self._snapshot()
            changed = after != before
            if changed:
                self._undo.append(before)
                del self._undo[:-self.history_limit]
                self._redo.clear()
            return ViewerCommandResult(
                made.command_id, made.operation, "ok", changed=changed,
                result=self.observe().to_dict(),
            )
        except Exception as error:
            try:
                if "before" in locals():
                    self._restore(before)
            except Exception:
                pass
            code = "unknown_target" if isinstance(error, LookupError) else "execution_failed"
            if "hidden" in str(error).lower():
                code = "target_hidden"
            return ViewerCommandResult(
                made.command_id, made.operation, "error",
                error_code=code, error_message=str(error),
            )

    def observe(self, limit: int = 1000) -> ViewerObservation:
        state = self.viewer.export_view_state()
        selection = self._get_selection()
        visibility = self._get_visibility()
        maximum = max(1, int(limit))
        truncated = len(selection) > maximum or len(visibility.hidden) > maximum or len(visibility.isolated) > maximum
        bounded_visibility = VisibilityState(
            visibility.hidden[:maximum], visibility.hidden_kinds,
            visibility.isolated[:maximum], visibility.isolated_kinds,
        )
        return ViewerObservation(
            str(getattr(self.viewer, "backend_name", "unknown")),
            tuple(int(value) for value in getattr(self.viewer, "viewport_size", (0, 0))),
            _state_dict(state), selection[:maximum], bounded_visibility,
            self.undo_available, self.redo_available, truncated,
        )

    def _snapshot(self) -> _Snapshot:
        return _Snapshot(self.viewer.export_view_state(), self._get_selection(), self._get_visibility())

    def _restore(self, snapshot: _Snapshot) -> None:
        self.viewer.apply_view_state(snapshot.view, redraw=False)
        self._set_visibility(snapshot.visibility)
        self._set_selection(snapshot.selection)
        _redraw(self.viewer)

    def _undo_command(self, command: ViewerCommand, *, redo: bool) -> ViewerCommandResult:
        source, destination = (self._redo, self._undo) if redo else (self._undo, self._redo)
        if not source:
            return ViewerCommandResult(
                command.command_id, command.operation, "error",
                error_code="history_empty", error_message="viewer history is empty",
            )
        current = self._snapshot()
        target = source.pop()
        try:
            self._restore(target)
        except Exception:
            source.append(target)
            raise
        destination.append(current)
        del destination[:-self.history_limit]
        return ViewerCommandResult(
            command.command_id, command.operation, "ok", changed=True,
            result=self.observe().to_dict(),
        )

    def _get_selection(self) -> tuple[SemanticRef, ...]:
        return semantic_refs(getattr(self.viewer, "semantic_selection", self._selection))

    def _set_selection(self, values: Sequence[SemanticRef]) -> None:
        made = semantic_refs(values)
        hidden = self._get_visibility()
        if any(not hidden.accepts((value,)) for value in made):
            raise ValueError("cannot select a hidden semantic target")
        self._validate_entities(made)
        setter = getattr(self.viewer, "set_semantic_selection", None)
        if callable(setter):
            setter(made)
        else:
            self._selection = made
        self._selection = made
        if self.selection_sink is not None:
            self.selection_sink(made)

    def _get_visibility(self) -> VisibilityState:
        value = getattr(self.viewer, "visibility_state", self._visibility)
        return value if isinstance(value, VisibilityState) else VisibilityState()

    def _set_visibility(self, value: VisibilityState) -> None:
        self._validate_entities((*value.hidden, *value.isolated))
        setter = getattr(self.viewer, "set_visibility_state", None)
        if callable(setter):
            setter(value)
        else:
            self._visibility = value
        self._visibility = value
        visible_selection = tuple(ref for ref in self._get_selection() if value.accepts((ref,)))
        if visible_selection != self._get_selection():
            self._set_selection(visible_selection)

    def _validate_entities(self, values: Sequence[SemanticRef]) -> None:
        if self.entity_exists is None:
            return
        missing = [value for value in values if not self.entity_exists(value)]
        if missing:
            raise LookupError(f"unknown semantic target: {missing[0].to_dict()}")

    def _apply(self, operation: str, params: dict[str, Any]) -> None:
        viewer = self.viewer
        if operation == "viewer.camera.set":
            state = viewer.export_view_state()
            changes: dict[str, Any] = {}
            if "position" in params: changes["camera_position"] = as_point(params["position"])
            if "target" in params: changes["camera_target"] = as_point(params["target"])
            if "world_up" in params: changes["camera_world_up"] = as_point(params["world_up"])
            if "fov_degrees" in params: changes["fov"] = math.radians(params["fov_degrees"])
            if "near" in params: changes["near"] = params["near"]
            if "far" in params: changes["far"] = params["far"]
            viewer.apply_view_state(replace(state, **changes))
        elif operation == "viewer.camera.orbit":
            viewer.camera.orbit(math.radians(params.get("azimuth_degrees", 0.0)), math.radians(params.get("elevation_degrees", 0.0)))
            _redraw(viewer)
        elif operation == "viewer.camera.pan":
            right, up, _forward = viewer.camera.basis()
            viewer.camera.set_target(viewer.camera.target + right * params.get("right", 0.0) + up * params.get("up", 0.0))
            _redraw(viewer)
        elif operation == "viewer.camera.zoom":
            viewer.camera.zoom(params["factor"])
            _redraw(viewer)
        elif operation == "viewer.camera.preset":
            preset = params["preset"]
            methods = {"top": "set_top_view", "front": "set_front_view", "right": "set_side_view", "isometric": "set_iso_view"}
            if preset in methods and callable(getattr(viewer, methods[preset], None)):
                getattr(viewer, methods[preset])()
            else:
                angles = {
                    "bottom": (-45.0, -90.0), "back": (180.0, 0.0),
                    "left": (90.0, 0.0), "right": (-90.0, 0.0),
                    "front": (0.0, 0.0), "top": (-45.0, 89.5),
                    "isometric": (-45.0, 35.264),
                }[preset]
                viewer.camera.set_orbit(math.radians(angles[0]), math.radians(angles[1]))
                _redraw(viewer)
        elif operation == "viewer.camera.fit":
            viewer.fit_to_scene(params.get("padding", 1.25))
        elif operation == "viewer.camera.reset":
            viewer.reset_camera()
        elif operation == "viewer.display.set":
            state = viewer.export_view_state()
            mapping = {
                "background": "background", "shading": "shading_enabled",
                "occlude_lines": "occlude_lines", "mesh_lines": "mesh_lines",
                "axis_indicator": "axis_indicator", "axis_ruler": "axis_ruler",
            }
            viewer.apply_view_state(replace(state, **{mapping[key]: value for key, value in params.items()}))
        elif operation == "viewer.section.set":
            viewer.set_section_plane(params.get("normal", (1.0, 0.0, 0.0)), params.get("offset", 0.0), enabled=params.get("enabled", True))
        elif operation == "viewer.section.clear":
            viewer.clear_section_plane()
        elif operation == "viewer.selection.set":
            incoming = semantic_refs(params["entities"])
            current = list(self._get_selection())
            mode = params.get("operation", "replace")
            if mode == "replace": result = list(incoming)
            elif mode == "add": result = current + [value for value in incoming if value not in current]
            elif mode == "remove": result = [value for value in current if value not in incoming]
            else:
                result = list(current)
                for value in incoming:
                    result.remove(value) if value in result else result.append(value)
            self._set_selection(result)
        elif operation == "viewer.selection.clear":
            self._set_selection(())
        elif operation.startswith("viewer.visibility."):
            self._apply_visibility(operation, params)
        else:
            raise ValueError(f"unsupported viewer operation: {operation}")

    def _apply_visibility(self, operation: str, params: dict[str, Any]) -> None:
        current = self._get_visibility()
        if operation == "viewer.visibility.show_all":
            self._set_visibility(VisibilityState())
            return
        refs = list(semantic_refs(params.get("entities", ())))
        if params.get("use_selection"):
            refs.extend(value for value in self._get_selection() if value not in refs)
        kinds = tuple(params.get("kinds", ()))
        if operation == "viewer.visibility.hide":
            value = VisibilityState(
                (*current.hidden, *refs), (*current.hidden_kinds, *kinds),
                current.isolated, current.isolated_kinds,
            )
        elif operation == "viewer.visibility.show":
            value = VisibilityState(
                tuple(item for item in current.hidden if item not in refs),
                tuple(item for item in current.hidden_kinds if item not in kinds),
                (*current.isolated, *(item for item in refs if item not in current.isolated)) if (current.isolated or current.isolated_kinds) else current.isolated,
                (*current.isolated_kinds, *(item for item in kinds if item not in current.isolated_kinds)) if (current.isolated or current.isolated_kinds) else current.isolated_kinds,
            )
        elif operation == "viewer.visibility.isolate":
            value = VisibilityState(isolated=tuple(refs), isolated_kinds=kinds)
        else:
            raise ValueError(f"unsupported visibility operation: {operation}")
        self._set_visibility(value)
        _redraw(self.viewer)


def viewer_command_manifest() -> dict[str, Any]:
    """Return the canonical provider-neutral command discovery document."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema": COMMAND_SCHEMA,
        "schema_version": COMMAND_SCHEMA_VERSION,
        "commands": [descriptor.to_dict() for descriptor in VIEWER_COMMANDS],
    }


def _redraw(viewer: object) -> None:
    callback = getattr(viewer, "redraw", None)
    if callable(callback):
        callback()


def _state_dict(state: ViewerState) -> dict[str, Any]:
    plane = state.section_plane
    return {
        "camera": {
            "position": list(state.camera_position.to_tuple()),
            "target": list(state.camera_target.to_tuple()),
            "world_up": list(state.camera_world_up.to_tuple()),
            "fov_degrees": math.degrees(state.fov),
            "near": state.near,
            "far": state.far,
        },
        "section_plane": None if plane is None else {
            "normal": list(plane.normal.to_tuple()), "offset": plane.offset, "enabled": plane.enabled,
        },
        "display": {
            "background": state.background, "shading": state.shading_enabled,
            "occlude_lines": state.occlude_lines, "mesh_lines": state.mesh_lines,
            "axis_indicator": state.axis_indicator, "axis_ruler": state.axis_ruler,
        },
        "interaction_profile": state.interaction_profile,
    }


def _json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain finite numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value): _json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, item in value.items(): _json_value(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} is not JSON-safe")


def _only(params: dict[str, Any], allowed: set[str], *, required: set[str] = frozenset(), nonempty: bool = False) -> None:
    extras = set(params).difference(allowed)
    missing = required.difference(params)
    if extras: raise ValueError(f"unknown command params: {sorted(extras)}")
    if missing: raise ValueError(f"missing command params: {sorted(missing)}")
    if nonempty and not params: raise ValueError("command requires at least one parameter")


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool): raise TypeError(f"{name} must be numeric")
    made = float(value)
    if not math.isfinite(made): raise ValueError(f"{name} must be finite")
    if positive and made <= 0.0: raise ValueError(f"{name} must be positive")
    return made


def _vector(value: Any, name: str, *, nonzero: bool = False) -> list[float]:
    if not isinstance(value, list) or len(value) != 3: raise ValueError(f"{name} must be a three-number array")
    made = [_number(item, name) for item in value]
    if nonzero and math.sqrt(sum(item * item for item in made)) <= 1.0e-12: raise ValueError(f"{name} must be non-zero")
    return made


def _targets(params: dict[str, Any]) -> None:
    _only(params, {"entities", "kinds", "use_selection"})
    if not params.get("entities") and not params.get("kinds") and not params.get("use_selection"):
        raise ValueError("visibility command requires entities, kinds, or use_selection")
    if "entities" in params:
        if not isinstance(params["entities"], list): raise TypeError("entities must be an array")
        params["entities"] = [SemanticRef.from_dict(value).to_dict() for value in params["entities"]]
    if "kinds" in params:
        if not isinstance(params["kinds"], list) or not all(isinstance(value, str) and value for value in params["kinds"]): raise TypeError("kinds must be non-empty strings")
    if "use_selection" in params and not isinstance(params["use_selection"], bool): raise TypeError("use_selection must be boolean")


def _validate_params(operation: str, params: dict[str, Any]) -> dict[str, Any]:
    if operation in {"viewer.observe", "viewer.camera.reset", "viewer.section.clear", "viewer.selection.clear", "viewer.visibility.show_all", "viewer.undo", "viewer.redo"}:
        _only(params, set())
    elif operation == "viewer.camera.set":
        _only(params, {"position", "target", "world_up", "fov_degrees", "near", "far"}, nonempty=True)
        for key in ("position", "target"): 
            if key in params: params[key] = _vector(params[key], key)
        if "world_up" in params: params["world_up"] = _vector(params["world_up"], "world_up", nonzero=True)
        for key in ("fov_degrees", "near", "far"):
            if key in params: params[key] = _number(params[key], key, positive=True)
        if "fov_degrees" in params and not 0.1 <= params["fov_degrees"] < 179.0: raise ValueError("fov_degrees must be in [0.1, 179)")
    elif operation == "viewer.camera.orbit":
        _only(params, {"azimuth_degrees", "elevation_degrees"}, nonempty=True)
        for key in tuple(params): params[key] = _number(params[key], key)
    elif operation == "viewer.camera.pan":
        _only(params, {"right", "up"}, nonempty=True)
        for key in tuple(params): params[key] = _number(params[key], key)
    elif operation == "viewer.camera.zoom":
        _only(params, {"factor"}, required={"factor"}); params["factor"] = _number(params["factor"], "factor", positive=True)
    elif operation == "viewer.camera.preset":
        _only(params, {"preset"}, required={"preset"})
        if params["preset"] not in {"top", "bottom", "front", "back", "left", "right", "isometric"}: raise ValueError("unsupported camera preset")
    elif operation == "viewer.camera.fit":
        _only(params, {"padding"})
        if "padding" in params: params["padding"] = _number(params["padding"], "padding", positive=True)
    elif operation == "viewer.display.set":
        _only(params, {"background", "shading", "occlude_lines", "mesh_lines", "axis_indicator", "axis_ruler"}, nonempty=True)
        if "background" in params and (not isinstance(params["background"], str) or not params["background"]): raise TypeError("background must be a non-empty color string")
        for key, value in params.items():
            if key != "background" and not isinstance(value, bool): raise TypeError(f"{key} must be boolean")
    elif operation == "viewer.section.set":
        _only(params, {"normal", "offset", "enabled"})
        if "normal" in params: params["normal"] = _vector(params["normal"], "normal", nonzero=True)
        if "offset" in params: params["offset"] = _number(params["offset"], "offset")
        if "enabled" in params and not isinstance(params["enabled"], bool): raise TypeError("enabled must be boolean")
    elif operation == "viewer.selection.set":
        _only(params, {"entities", "operation"}, required={"entities"})
        if not isinstance(params["entities"], list): raise TypeError("entities must be an array")
        params["entities"] = [SemanticRef.from_dict(value).to_dict() for value in params["entities"]]
        if params.get("operation", "replace") not in {"replace", "add", "remove", "toggle"}: raise ValueError("unsupported selection operation")
    elif operation in {"viewer.visibility.hide", "viewer.visibility.show", "viewer.visibility.isolate"}:
        _targets(params)
    return params


_OBJECT = {"type": "object", "additionalProperties": False}
_ENTITY = {
    "type": "object", "additionalProperties": False,
    "required": ["source", "kind", "key"],
    "properties": {
        "source": {"enum": ["model", "application"]}, "kind": {"type": "string", "minLength": 1},
        "key": {"type": ["string", "integer"]}, "model_id": {"type": ["string", "null"], "format": "uuid"},
    },
}


def _schema(properties: Mapping[str, Any] | None = None, required: Sequence[str] = ()) -> dict[str, Any]:
    result = dict(_OBJECT)
    result["properties"] = dict(properties or {})
    if required: result["required"] = list(required)
    return result


_DESCRIPTIONS = {
    "viewer.observe": "Read bounded camera, display, selection, visibility and history state.",
    "viewer.camera.set": "Set absolute camera parameters.", "viewer.camera.orbit": "Orbit the camera by degrees.",
    "viewer.camera.pan": "Pan in camera-right and camera-up world units.", "viewer.camera.zoom": "Multiply camera distance.",
    "viewer.camera.preset": "Apply a named engineering view.", "viewer.camera.fit": "Fit the retained scene.",
    "viewer.camera.reset": "Reset the camera.", "viewer.display.set": "Update renderer-neutral display policy.",
    "viewer.section.set": "Set the world-space section plane.", "viewer.section.clear": "Clear section clipping.",
    "viewer.selection.set": "Replace or modify semantic selection.", "viewer.selection.clear": "Clear semantic selection.",
    "viewer.visibility.hide": "Hide semantic entities or kinds.", "viewer.visibility.show": "Show semantic entities or kinds.",
    "viewer.visibility.isolate": "Show only semantic entities or kinds.", "viewer.visibility.show_all": "Clear semantic visibility filters.",
    "viewer.undo": "Undo the most recent successful viewer mutation.", "viewer.redo": "Redo the most recently undone viewer mutation.",
}

_EMPTY_SCHEMA = _schema()
_SCHEMAS = {
    "viewer.camera.set": _schema({key: {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3} for key in ("position", "target", "world_up")} | {key: {"type": "number"} for key in ("fov_degrees", "near", "far")}),
    "viewer.camera.orbit": _schema({"azimuth_degrees": {"type": "number"}, "elevation_degrees": {"type": "number"}}),
    "viewer.camera.pan": _schema({"right": {"type": "number"}, "up": {"type": "number"}}),
    "viewer.camera.zoom": _schema({"factor": {"type": "number", "exclusiveMinimum": 0}}, ("factor",)),
    "viewer.camera.preset": _schema({"preset": {"enum": ["top", "bottom", "front", "back", "left", "right", "isometric"]}}, ("preset",)),
    "viewer.camera.fit": _schema({"padding": {"type": "number", "exclusiveMinimum": 0}}),
    "viewer.display.set": _schema({"background": {"type": "string"}, **{key: {"type": "boolean"} for key in ("shading", "occlude_lines", "mesh_lines", "axis_indicator", "axis_ruler")}}),
    "viewer.section.set": _schema({"normal": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}, "offset": {"type": "number"}, "enabled": {"type": "boolean"}}),
    "viewer.selection.set": _schema({"entities": {"type": "array", "items": _ENTITY, "maxItems": 10000}, "operation": {"enum": ["replace", "add", "remove", "toggle"]}}, ("entities",)),
}
_TARGET_SCHEMA = _schema({"entities": {"type": "array", "items": _ENTITY, "maxItems": 10000}, "kinds": {"type": "array", "items": {"type": "string"}}, "use_selection": {"type": "boolean"}})
for _name in ("viewer.visibility.hide", "viewer.visibility.show", "viewer.visibility.isolate"):
    _SCHEMAS[_name] = _TARGET_SCHEMA

_OPERATIONS = tuple(_DESCRIPTIONS)
VIEWER_COMMANDS = tuple(
    ViewerCommandDescriptor(
        name, description, _SCHEMAS.get(name, _EMPTY_SCHEMA),
        permission_category="read_only" if name == "viewer.observe" else "reversible_edit",
        reversibility="none" if name == "viewer.observe" else "undoable",
    )
    for name, description in _DESCRIPTIONS.items()
)


__all__ = [
    "COMMAND_SCHEMA", "COMMAND_SCHEMA_VERSION", "SemanticRef", "VisibilityState",
    "ViewerCommand", "ViewerCommandController", "ViewerCommandDescriptor",
    "ViewerCommandPriority", "ViewerCommandResult", "ViewerObservation",
    "VIEWER_COMMANDS", "viewer_command_manifest",
]
