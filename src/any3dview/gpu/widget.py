"""Tk/ttk widget backed by :class:`ModernGLRenderer`."""

from __future__ import annotations

import math
import sys
import time
import tkinter as tk
from tkinter import ttk
from dataclasses import replace
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

try:
    import moderngl
except ImportError as error:  # pragma: no cover - isolated-wheel coverage
    raise ImportError(
        "ANY3dView GPU support requires the 'gpu' extra: pip install ANY3dView[gpu]"
    ) from error

from ..arrays import MeshArrays
from ..capabilities import ViewerCapabilities
from ..clipping import SectionPlane
from ..contracts import Pick, ViewerState
from ..core import (
    Camera3D,
    Point3D,
    _flatten_numeric_values,
    _interpolate_thickness_color,
    as_point,
    parse_color,
)
from ..errors import GPUUnavailableError
from ..ownership import ModelOwner, PackedOwnerTable
from ..retained import MeshHandle
from ..semantic import SemanticRef, VisibilityState, semantic_refs
from ..scheduler import ViewerScheduler
from ..shading import Light
from .. import shapes as shapes_module
from ..shapes import Mesh
from ..selection import (
    PickBinding,
    PickOwner,
    ProjectedPrimitive,
    ProjectedSelectionIndex,
    SelectionConfig,
    SelectionDepth,
    SelectionEvent,
    SelectionGesture,
    SelectionHit,
    SelectionOperation,
    SelectionFilter,
    SelectionTool,
)
from .host import TkinterGLHost
from .hud import GPUHudRenderer
from .renderer import ModernGLRenderer


GPU_CAPABILITIES = ViewerCapabilities(
    gpu=True,
    dynamic_arrays=True,
    node_scalar_field=True,
    element_scalar_field=True,
    shader_deformation=True,
    active_element_mask=True,
    incremental_chunks=True,
    integer_picking=True,
    through_selection=True,
    clipping_planes=True,
    transparency=True,
    # ChangeSet coalescing belongs to the renderer-neutral GeometryLayer;
    # neither concrete backend consumes ChangeSet objects directly.
    geometry_changeset=False,
    software_fallback=True,
    legacy_primitives=True,
    text_hud=True,
    legends=True,
    camera_controls=True,
    work_plane_projection=True,
    hover_selection=True,
    region_selection=True,
    lasso_selection=True,
    animation=True,
    image_capture=True,
    line_occlusion=True,
    stippled_transparency=True,
    semantic_selection=True,
    semantic_visibility=True,
    viewer_commands=True,
    command_history=True,
)


_CPU_POINT_STACK_LIMIT = 50_000


class Any3DView(ttk.Frame):
    """Demand-driven OpenGL 3.3 viewer embedded in a Tk application."""

    def __init__(
        self,
        master: tk.Misc,
        width: int = 800,
        height: int = 600,
        bg: str = "white",
        interactive_fps: int = 40,
        shading: bool = True,
        interaction_profile: str = "legacy",
        **canvas_kwargs: Any,
    ) -> None:
        super().__init__(master, **canvas_kwargs)
        self.camera = Camera3D()
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.bg = str(bg)
        self._light = Light()
        self._interactive_fps = max(1, int(interactive_fps))
        self._shading_enabled = bool(shading)
        self._occlude_lines = True
        self.show_mesh_lines = True
        self._show_axis_indicator = True
        self.show_axis_ruler = False
        self._interaction_profile = "legacy"
        self.set_interaction_profile(interaction_profile)
        self._thickness_legend: Optional[dict[str, Any]] = None
        self._world_text: list[dict[str, Any]] = []
        self._section_plane: Optional[SectionPlane] = None
        self._entries: dict[int, dict[str, Any]] = {}
        self._selection_callback: Optional[Callable[[SelectionEvent], None]] = None
        self._selection_hover_callback: Optional[Callable[[Optional[SelectionHit]], None]] = None
        self._pick_callback: Optional[Callable[[Pick], None]] = None
        self._hover_callback: Optional[Callable[[Optional[Pick]], None]] = None
        self._pick_prefix = ""
        self._pick_radius = 4
        self._hover_key: Optional[str] = None
        self._highlighted_tags: frozenset[str] = frozenset()
        self._highlight_fill = "#ff8c00"
        self._highlight_outline = "#b45309"
        self._preselected_key: Optional[str] = None
        self._preselection_from_hit = False
        self._selection_config = SelectionConfig()
        self._semantic_selection: tuple[SemanticRef, ...] = ()
        self._visibility_state = VisibilityState()
        self._selection_index: Optional[ProjectedSelectionIndex] = None
        self._selection_index_key: object = None
        self._mouse = (0, 0)
        self._drag = ""
        self._selection_press: Optional[tuple[int, int]] = None
        self._selection_current: Optional[tuple[int, int]] = None
        self._selection_points: list[tuple[int, int]] = []
        self._selection_dragging = False
        self._selection_press_hit_keys: frozenset[str] = frozenset()
        self._selection_operation = SelectionOperation.REPLACE
        self._selection_modifiers = (False, False, False)
        self._tracked_modifiers = {"shift": False, "ctrl": False, "alt": False}
        self._cycle_candidates: tuple[SelectionHit, ...] = ()
        self._cycle_anchor: Optional[tuple[int, int]] = None
        self._cycle_time = 0.0
        self._cycle_index = -1
        self._legacy_item_counter = 0
        self._animation_cache: list[dict[str, Any]] = []
        self._animation_frame_index = 0
        self._animation_after_id: Optional[str] = None
        self._is_playing_animation = False
        self._animation_handles: list[MeshHandle] = []
        self._animation_entries: dict[int, dict[str, Any]] = {}
        self._animation_frame_active = False
        self._animation_live_hud: Optional[dict[str, Any]] = None
        self._opaque_occluders: list[MeshHandle] = []
        self._closed = False
        self._suspend_redraw = False
        self._backend_diagnostics: tuple[str, ...] = ()
        self._update_scheduler = ViewerScheduler()
        self._update_poll_id: Optional[str] = None
        try:
            self._host = TkinterGLHost(
                self,
                self._draw_now,
                width=max(1, int(width)),
                height=max(1, int(height)),
            )
            self._host.surface.pack(fill=tk.BOTH, expand=True)
            self.canvas = self._host.surface
            self._host.make_current()
            context = moderngl.create_context(require=330)
            self._renderer = ModernGLRenderer(context)
            self._hud = GPUHudRenderer(context)
        except Exception as error:
            host = getattr(self, "_host", None)
            if host is not None:
                host.close()
            self.destroy()
            raise GPUUnavailableError(
                f"GPU backend initialization failed: {error}",
                diagnostics=(type(error).__name__, str(error)),
            ) from error
        self._bind_interaction()
        self._poll_updates()
        self.redraw()

    def submit_update(
        self,
        callback: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Queue an immutable update payload for the owning Tk thread."""

        self._update_scheduler.submit(callback, *args, **kwargs)

    def _poll_updates(self) -> None:
        if self._closed:
            return
        completed = self._update_scheduler.drain()
        if completed:
            self.redraw()
        self._update_poll_id = self.after(16, self._poll_updates)

    @property
    def backend_name(self) -> str:
        return "gpu"

    @property
    def event_widget(self) -> object:
        return self._host.surface

    @property
    def viewport_size(self) -> tuple[int, int]:
        width, height = self._host.framebuffer_size()
        if width <= 1 or height <= 1:
            return self.width, self.height
        return width, height

    def export_view_state(self) -> ViewerState:
        camera = self.camera
        point = lambda value: Point3D(value.x, value.y, value.z)
        plane = self._section_plane
        plane_copy = None if plane is None else SectionPlane(
            point(plane.normal), plane.offset, plane.enabled
        )
        return ViewerState(
            camera_position=point(camera.position),
            camera_target=point(camera.target),
            camera_world_up=point(camera.world_up),
            fov=float(camera.fov),
            near=float(camera.near),
            far=float(camera.far),
            section_plane=plane_copy,
            background=self.bg,
            shading_enabled=self._shading_enabled,
            occlude_lines=self._occlude_lines,
            mesh_lines=self.show_mesh_lines,
            axis_indicator=self._show_axis_indicator,
            axis_ruler=self.show_axis_ruler,
            interaction_profile=self._interaction_profile,
            semantic_selection=self._semantic_selection,
            visibility=self._visibility_state,
        )

    def apply_view_state(self, state: ViewerState, *, redraw: bool = True) -> None:
        if not isinstance(state, ViewerState):
            raise TypeError("state must be ViewerState")
        if not (0.0 < state.near < state.far):
            raise ValueError("viewer state requires 0 < near < far")
        if not (0.0 < state.fov < math.pi):
            raise ValueError("viewer state fov must be between 0 and pi")
        world_up = as_point(state.camera_world_up).normalized()
        if world_up.length() <= 1.0e-12:
            raise ValueError("viewer state camera_world_up must be non-zero")
        profile = str(state.interaction_profile).strip().lower()
        if profile not in {"legacy", "commercial"}:
            raise ValueError("interaction profile must be 'legacy' or 'commercial'")
        self.camera.world_up = world_up
        self.camera.near = float(state.near)
        self.camera.far = float(state.far)
        self.camera.fov = float(state.fov)
        self.camera.set_target(as_point(state.camera_target))
        self.camera.set_position(as_point(state.camera_position))
        self._section_plane = state.section_plane
        self.bg = str(state.background)
        self._shading_enabled = bool(state.shading_enabled)
        self._occlude_lines = bool(state.occlude_lines)
        self.show_mesh_lines = bool(state.mesh_lines)
        self._show_axis_indicator = bool(state.axis_indicator)
        self.show_axis_ruler = bool(state.axis_ruler)
        self.set_interaction_profile(profile)
        self._semantic_selection = semantic_refs(state.semantic_selection)
        self._visibility_state = state.visibility
        self._renderer.pick_dirty = True
        self._selection_index = None
        if redraw:
            self.redraw()

    def project_point(self, point: object) -> Optional[tuple[float, float, float]]:
        value = as_point(point)
        width, height = self.viewport_size
        projected = self.camera.project_point(value, width, height)
        if projected is None:
            return None
        _right, _up, forward = self.camera.basis()
        depth = (value - self.camera.position).dot(forward)
        return float(projected[0]), float(projected[1]), float(depth)

    def project_points(
        self, points: Iterable[object]
    ) -> tuple[Optional[tuple[float, float, float]], ...]:
        return tuple(self.project_point(point) for point in points)

    def screen_ray(self, x: float, y: float) -> tuple[Point3D, Point3D]:
        width, height = self.viewport_size
        return self.camera.screen_ray(x, y, width, height)

    def unproject_to_plane(
        self,
        x: float,
        y: float,
        plane_point: object,
        plane_normal: object,
    ) -> Optional[Point3D]:
        width, height = self.viewport_size
        return self.camera.unproject_to_plane(
            x, y, width, height, as_point(plane_point), as_point(plane_normal)
        )

    @property
    def capabilities(self) -> ViewerCapabilities:
        return GPU_CAPABILITIES

    @property
    def backend_diagnostics(self) -> tuple[str, ...]:
        return self._backend_diagnostics

    @property
    def section_plane(self) -> Optional[SectionPlane]:
        return self._section_plane

    @property
    def light(self) -> Light:
        return self._light

    @property
    def interaction_profile(self) -> str:
        return self._interaction_profile

    @property
    def selection_config(self) -> SelectionConfig:
        return self._selection_config

    @property
    def semantic_selection(self) -> tuple[SemanticRef, ...]:
        return self._semantic_selection

    @property
    def visibility_state(self) -> VisibilityState:
        return self._visibility_state

    def set_semantic_selection(self, values: Sequence[SemanticRef]) -> None:
        self._semantic_selection = semantic_refs(values)
        self._renderer.pick_dirty = True
        self._selection_index = None
        self._apply_highlight_masks()

    def set_visibility_state(self, state: VisibilityState) -> None:
        if not isinstance(state, VisibilityState):
            raise TypeError("state must be VisibilityState")
        self._visibility_state = state
        self._renderer.pick_dirty = True
        self._selection_index = None
        self._selection_index_key = None
        self._apply_highlight_masks()

    def _semantic_binding_visible(self, binding: Optional[PickBinding]) -> bool:
        state = getattr(self, "_visibility_state", VisibilityState())
        if state.is_default:
            return True
        return state.accepts(() if binding is None else binding.owners)

    def set_light(
        self,
        direction: Optional[Point3D] = None,
        ambient: Optional[float] = None,
        diffuse: Optional[float] = None,
        specular: Optional[float] = None,
        shininess: Optional[float] = None,
        follow_camera: Optional[bool] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        if direction is not None:
            value = as_point(direction).normalized()
            if value.length() > 0.0:
                self._light.direction = value
        if ambient is not None:
            self._light.ambient = float(ambient)
        if diffuse is not None:
            self._light.diffuse = float(diffuse)
        if specular is not None:
            self._light.specular = float(specular)
        if shininess is not None:
            self._light.shininess = max(1.0, float(shininess))
        if follow_camera is not None:
            self._light.follow_camera = bool(follow_camera)
        if enabled is not None:
            self._light.enabled = bool(enabled)
        self.redraw()

    def set_shading(self, enabled: bool = True) -> None:
        self._shading_enabled = bool(enabled)
        self.redraw()

    def set_occlude_lines(self, enabled: bool = True) -> None:
        self._occlude_lines = bool(enabled)
        self._renderer.pick_dirty = True
        self._selection_index = None
        self.redraw()

    def set_background(self, color: str) -> None:
        self.bg = str(color)
        self.redraw()

    def set_mesh_lines(self, visible: bool = True) -> None:
        self.show_mesh_lines = bool(visible)
        self._renderer.pick_dirty = True
        self._selection_index = None
        self.redraw()

    def set_axis_indicator(self, visible: bool = True) -> None:
        self._show_axis_indicator = bool(visible)
        self.redraw()

    def set_axis_ruler(self, visible: bool = True) -> None:
        self.show_axis_ruler = bool(visible)
        self.redraw()

    def set_interaction_profile(self, profile: str) -> None:
        value = str(profile).strip().lower()
        if value not in {"legacy", "commercial"}:
            raise ValueError("interaction profile must be 'legacy' or 'commercial'")
        self._interaction_profile = value

    @property
    def renderer_diagnostics(self) -> dict[str, int]:
        return {
            "draw_calls": self._renderer.draw_calls,
            "frame_count": self._renderer.frame_count,
            "geometry_uploads": self._renderer.geometry_uploads,
            "retained_meshes": len(self._entries),
            "hud_uploads": self._hud.uploads,
        }

    def _display_entries(self) -> dict[int, dict[str, Any]]:
        """Entries backing the currently visible live or animation frame."""

        if getattr(self, "_animation_frame_active", False):
            return self._animation_entries
        return self._entries

    def _display_primitive_count(self, limit: Optional[int] = None) -> int:
        total = 0
        for entry in self._display_entries().values():
            handle = entry["handle"]
            for _chunk_id, mesh in [(None, handle.mesh), *handle.chunks]:
                total += (
                    mesh.triangle_count
                    + (0 if mesh.lines is None else len(mesh.lines))
                    + (0 if mesh.point_indices is None else len(mesh.point_indices))
                )
                if limit is not None and total > limit:
                    return total
        return total

    def _bind_interaction(self) -> None:
        surface = self._host.surface
        surface.bind("<ButtonPress-1>", self._press_select, add="+")
        surface.bind("<B1-Motion>", self._drag_select, add="+")
        surface.bind("<ButtonRelease-1>", self._release_select, add="+")
        surface.bind("<Motion>", self._hover_select, add="+")
        surface.bind("<ButtonPress-2>", lambda event: self._press(event, "pan"), add="+")
        surface.bind("<ButtonPress-3>", lambda event: self._press(event, "orbit"), add="+")
        surface.bind("<B2-Motion>", self._motion, add="+")
        surface.bind("<B3-Motion>", self._motion, add="+")
        surface.bind("<ButtonRelease-2>", self._release, add="+")
        surface.bind("<ButtonRelease-3>", self._release, add="+")
        surface.bind("<MouseWheel>", self._wheel, add="+")
        surface.bind("<Button-4>", lambda _event: self._zoom(0.9), add="+")
        surface.bind("<Button-5>", lambda _event: self._zoom(1.1), add="+")
        surface.bind("<Configure>", lambda _event: self.redraw(), add="+")
        surface.bind("<Escape>", self._cancel_interaction, add="+")
        surface.bind("<FocusOut>", self._focus_out, add="+")
        surface.bind("<KeyPress>", self._on_modifier_key, add="+")
        surface.bind("<KeyRelease>", self._on_modifier_key, add="+")
        self._interaction_toplevel = surface.winfo_toplevel()
        self._toplevel_release_binding = self._interaction_toplevel.bind(
            "<ButtonRelease-1>", self._toplevel_release_select, add="+"
        )
        self._toplevel_escape_binding = self._interaction_toplevel.bind(
            "<Escape>", self._cancel_interaction, add="+"
        )

    def _cancel_interaction(self, _event: Optional[tk.Event] = None) -> None:
        self._drag = ""
        self._selection_press = None
        self._selection_current = None
        self._selection_points = []
        self._selection_dragging = False
        self._selection_press_hit_keys = frozenset()
        self._selection_operation = SelectionOperation.REPLACE
        self._selection_modifiers = (False, False, False)
        self.redraw()

    def _focus_out(self, event: tk.Event) -> None:
        for key in self._tracked_modifiers:
            self._tracked_modifiers[key] = False
        self._cancel_interaction(event)

    def _toplevel_release_select(self, event: tk.Event) -> None:
        if self._selection_press is None:
            return
        try:
            event.x = int(event.x_root) - int(self.canvas.winfo_rootx())
            event.y = int(event.y_root) - int(self.canvas.winfo_rooty())
        except (AttributeError, TypeError, ValueError, tk.TclError):
            event.x, event.y = int(event.x), int(event.y)
        self._release_select(event)

    def _press(self, event: tk.Event, mode: str) -> None:
        self._mouse = (int(event.x), int(event.y))
        self._drag = mode

    def _motion(self, event: tk.Event) -> None:
        x, y = int(event.x), int(event.y)
        dx, dy = x - self._mouse[0], y - self._mouse[1]
        self._mouse = (x, y)
        if self._drag == "orbit":
            self.camera.orbit(-0.008 * dx, 0.008 * dy)
        elif self._drag == "pan":
            width, height = self._host.framebuffer_size()
            self.camera.pan_view_pixels(dx, dy, width, height)
        self._renderer.pick_dirty = True
        self._selection_index = None
        self.redraw()

    def _release(self, _event: tk.Event) -> None:
        self._drag = ""

    def _wheel(self, event: tk.Event) -> None:
        self._zoom(0.9 if int(getattr(event, "delta", 0)) > 0 else 1.1)

    def _zoom(self, factor: float) -> None:
        self.camera.zoom(factor)
        self._renderer.pick_dirty = True
        self._selection_index = None
        self.redraw()

    def _on_modifier_key(self, event: tk.Event) -> None:
        keysym = str(getattr(event, "keysym", "")).lower()
        pressed = str(getattr(event, "type", "")) in {"2", "KeyPress"}
        if keysym.startswith("shift"):
            self._tracked_modifiers["shift"] = pressed
        elif keysym.startswith("control"):
            self._tracked_modifiers["ctrl"] = pressed
        elif keysym.startswith(("alt", "option", "meta")):
            self._tracked_modifiers["alt"] = pressed

    @staticmethod
    def _windows_alt_is_down() -> bool:
        try:
            import ctypes

            return bool(ctypes.windll.user32.GetKeyState(0x12) & 0x8000)
        except (AttributeError, OSError):  # pragma: no cover - unusual runtime
            return True

    def _event_modifiers(self, event: tk.Event) -> tuple[bool, bool, bool]:
        state = int(getattr(event, "state", 0) or 0)
        shift = bool(state & 0x0001)
        ctrl = bool(state & 0x0004)
        if sys.platform == "win32":
            # Tk's low Mod1 bit is present on ordinary Windows mouse events;
            # only the high bit, confirmed against the physical key, is Alt.
            alt = bool(state & 0x20000) and self._windows_alt_is_down()
        else:
            alt = bool(state & (0x0008 | 0x20000))
        if sys.platform == "darwin":
            shift = shift or self._tracked_modifiers["shift"]
            ctrl = ctrl or self._tracked_modifiers["ctrl"]
            alt = alt or self._tracked_modifiers["alt"]
        return shift, ctrl, alt

    @staticmethod
    def _operation_from_modifiers(
        shift: bool, ctrl: bool, alt: bool
    ) -> SelectionOperation:
        if alt:
            return SelectionOperation.REMOVE
        if ctrl:
            return SelectionOperation.TOGGLE
        if shift:
            return SelectionOperation.ADD
        return SelectionOperation.REPLACE

    def _press_select(self, event: tk.Event) -> None:
        point = (int(event.x), int(event.y))
        try:
            self.canvas.focus_set()
        except tk.TclError:
            pass
        self._selection_modifiers = self._event_modifiers(event)
        self._selection_operation = self._operation_from_modifiers(
            *self._selection_modifiers
        )
        if self._interaction_profile == "legacy":
            self._selection_press = point
            self._selection_current = point
            self._selection_dragging = False
            self._press(event, "pan")
            return
        self._selection_press = point
        self._selection_current = point
        self._selection_points = [point]
        self._selection_dragging = False
        if self._selection_config.click_on_press:
            pressed = self._emit_click(point, self._selection_operation)
            self._selection_press_hit_keys = frozenset(hit.key for hit in pressed)

    def _drag_select(self, event: tk.Event) -> None:
        if self._interaction_profile == "legacy":
            point = (int(event.x), int(event.y))
            self._selection_current = point
            if self._selection_press is not None:
                self._selection_dragging = math.hypot(
                    point[0] - self._selection_press[0],
                    point[1] - self._selection_press[1],
                ) >= self._selection_config.drag_threshold_px
            self._motion(event)
            return
        if self._selection_press is None or self._selection_config.tool is SelectionTool.SINGLE:
            return
        point = (int(event.x), int(event.y))
        self._selection_current = point
        if not self._selection_dragging:
            self._selection_dragging = math.hypot(
                point[0] - self._selection_press[0], point[1] - self._selection_press[1]
            ) >= self._selection_config.drag_threshold_px
        if self._selection_dragging and self._selection_config.tool is SelectionTool.LASSO:
            previous = self._selection_points[-1]
            if math.hypot(point[0] - previous[0], point[1] - previous[1]) >= 3.0:
                self._selection_points.append(point)
        if self._selection_dragging:
            self.redraw()

    def _release_select(self, event: tk.Event) -> None:
        if self._interaction_profile == "legacy":
            start = self._selection_press
            dragged = self._selection_dragging
            self._release(event)
            self._selection_press = None
            self._selection_current = None
            self._selection_dragging = False
            if start is not None and not dragged:
                self._emit_click(
                    (int(event.x), int(event.y)), self._selection_operation
                )
            self._selection_operation = SelectionOperation.REPLACE
            self._selection_modifiers = (False, False, False)
            return
        start = self._selection_press
        if start is None:
            return
        end = (int(event.x), int(event.y))
        operation = self._selection_operation
        if self._selection_dragging:
            if self._selection_config.tool is SelectionTool.LASSO:
                points = tuple((*self._selection_points, end))
                hits = self.query_lasso(points)
                if self._selection_press_hit_keys:
                    hits = tuple(
                        hit for hit in hits
                        if hit.key not in self._selection_press_hit_keys
                    )
                value = SelectionEvent(
                    SelectionGesture.LASSO, operation, hits=hits,
                    start=start, end=end, points=points,
                )
            else:
                crossing = bool(self._selection_config.directional and end[0] < start[0])
                hits = self.query_rectangle(start, end, crossing=crossing)
                if self._selection_press_hit_keys:
                    hits = tuple(
                        hit for hit in hits
                        if hit.key not in self._selection_press_hit_keys
                    )
                value = SelectionEvent(
                    SelectionGesture.CROSSING if crossing else SelectionGesture.WINDOW,
                    operation, hits=hits, start=start, end=end,
                )
            if self._selection_callback is not None:
                self._selection_callback(value)
        elif not self._selection_config.click_on_press:
            self._emit_click(end, operation)
        self._selection_press = None
        self._selection_current = None
        self._selection_points = []
        self._selection_dragging = False
        self._selection_press_hit_keys = frozenset()
        self._selection_operation = SelectionOperation.REPLACE
        self._selection_modifiers = (False, False, False)
        self.redraw()

    def _emit_click(
        self, point: tuple[int, int], operation: SelectionOperation
    ) -> tuple[SelectionHit, ...]:
        now = time.monotonic()
        repeat_anchor = (
            self._cycle_anchor is not None
            and math.hypot(point[0] - self._cycle_anchor[0], point[1] - self._cycle_anchor[1])
            <= self._selection_config.cycle_radius_px
            and (now - self._cycle_time) * 1000.0
            <= self._selection_config.cycle_timeout_ms
        )
        query_config = (
            replace(self._selection_config, depth=SelectionDepth.THROUGH)
            if repeat_anchor and self._display_primitive_count(_CPU_POINT_STACK_LIMIT)
            > _CPU_POINT_STACK_LIMIT
            else self._selection_config
        )
        previous_candidates = self._cycle_candidates
        candidates = self.query_point(*point, config=query_config)
        same = (
            candidates == self._cycle_candidates
            and repeat_anchor
        )
        expanded_stack = bool(
            repeat_anchor
            and len(previous_candidates) == 1
            and len(candidates) > 1
            and previous_candidates[0].key == candidates[0].key
        )
        if expanded_stack:
            self._cycle_index = 1
        elif same and candidates:
            self._cycle_index = (self._cycle_index + 1) % len(candidates)
        else:
            self._cycle_index = 0 if candidates else -1
        self._cycle_candidates = candidates
        self._cycle_anchor = point
        self._cycle_time = now
        hits = () if self._cycle_index < 0 else (candidates[self._cycle_index],)
        if self._selection_callback is not None:
            self._selection_callback(SelectionEvent(
                SelectionGesture.CLICK, operation, hits=hits, candidates=candidates,
                start=point, end=point, cycle_index=max(0, self._cycle_index),
                cycle_total=len(candidates),
            ))
        if self._pick_callback is not None:
            if self._pick_prefix:
                hit = next(
                    (value for value in candidates if value.key.startswith(self._pick_prefix)),
                    None,
                )
            else:
                hit = hits[0] if hits else None
            modifiers = getattr(self, "_selection_modifiers", None)
            if modifiers is None:
                modifiers = (
                    operation is SelectionOperation.ADD,
                    operation is SelectionOperation.TOGGLE,
                    operation is SelectionOperation.REMOVE,
                )
            self._pick_callback(Pick(
                "" if hit is None else hit.key,
                -1 if hit is None else hit.item,
                point[0], point[1],
                *modifiers,
            ))
        return hits

    def _hover_select(self, event: tk.Event) -> None:
        if self._selection_dragging or self._drag:
            return
        hits = self.query_point(int(event.x), int(event.y))
        hit = hits[0] if hits else None
        key = None if hit is None else hit.key
        if key == self._hover_key:
            return
        self._hover_key = key
        self._set_hit_preselection(key)
        if self._selection_hover_callback is not None:
            self._selection_hover_callback(hit)
        if self._hover_callback is not None:
            legacy_hit = (
                next(
                    (value for value in hits if value.key.startswith(self._pick_prefix)),
                    None,
                )
                if self._pick_prefix else hit
            )
            self._hover_callback(None if legacy_hit is None else Pick(
                legacy_hit.key, legacy_hit.item, int(event.x), int(event.y)
            ))

    def set_selection_callback(
        self, callback: Optional[Callable[[SelectionEvent], None]]
    ) -> None:
        self._selection_callback = callback

    def configure_selection(
        self,
        callback: Optional[Callable[[SelectionEvent], None]],
        *,
        hover_callback: Optional[Callable[[Optional[SelectionHit]], None]] = None,
        config: Optional[SelectionConfig] = None,
        **_options: object,
    ) -> None:
        """Configure semantic selection while retaining the software API shape."""

        self._selection_callback = callback
        self._selection_hover_callback = hover_callback
        if config is not None:
            if not isinstance(config, SelectionConfig):
                raise TypeError("config must be SelectionConfig")
            self._selection_config = config

    def update_selection_config(self, **changes: Any) -> SelectionConfig:
        self._selection_config = replace(self._selection_config, **changes)
        self._cycle_candidates = ()
        self._cycle_anchor = None
        return self._selection_config

    def set_pick_callback(
        self,
        callback: Optional[Callable[[Pick], None]],
        *,
        prefix: str = "",
        radius: Optional[int] = None,
    ) -> None:
        self._pick_callback = callback
        self._pick_prefix = str(prefix)
        if radius is not None:
            self._pick_radius = max(0, int(radius))

    def set_hover_callback(
        self, callback: Optional[Callable[[Optional[Pick]], None]]
    ) -> None:
        self._hover_callback = callback
        self._hover_key = None

    def pick_at(self, x: int, y: int) -> Optional[str]:
        selection_filter = (
            SelectionFilter(key_prefixes=(self._pick_prefix,))
            if self._pick_prefix else None
        )
        hits = self.query_point(x, y, selection_filter=selection_filter,
                                radius=self._pick_radius)
        return hits[0].key if hits else None

    def _apply_highlight_masks(self) -> None:
        selected_keys = set(self._highlighted_tags)
        selected_keys.update(
            self._semantic_key(value)
            for value in getattr(self, "_semantic_selection", ())
        )
        preselected_key = (
            self._preselected_key
            if self._preselected_key not in selected_keys
            else None
        )
        for entry in self._display_entries().values():
            handle: MeshHandle = entry["handle"]
            if handle.removed:
                continue
            entry_tags = set(entry.get("tags", ()))
            masks = self._semantic_masks_for_mesh(
                handle.mesh,
                entry.get("owners"),
                entry.get("owner_resolver"),
                entry,
                entry_tags,
                selected_keys,
                preselected_key,
            )
            if hasattr(self._renderer, "set_semantic_masks"):
                self._renderer.set_semantic_masks(handle, **masks)
            else:  # compatibility with 0.4 renderer doubles
                self._renderer.set_highlighted_elements(
                    handle,
                    np.union1d(masks["selected_elements"], masks["preselected_elements"]),
                )
            chunk_caches = entry.setdefault("chunk_semantics", {})
            current_chunks = set()
            for chunk_id, mesh, owners, resolver in handle.chunk_records:
                current_chunks.add(chunk_id)
                cache = chunk_caches.setdefault(chunk_id, {})
                chunk_masks = self._semantic_masks_for_mesh(
                    mesh,
                    owners,
                    resolver,
                    cache,
                    entry_tags,
                    selected_keys,
                    preselected_key,
                )
                chunk_resolvable = owners is not None or bool(entry_tags)
                if hasattr(self._renderer, "set_chunk_pickable"):
                    self._renderer.set_chunk_pickable(
                        handle, chunk_id, chunk_resolvable
                    )
                if (
                    (chunk_resolvable or not self._visibility_state.is_default)
                    and hasattr(self._renderer, "set_chunk_semantic_masks")
                ):
                    self._renderer.set_chunk_semantic_masks(
                        handle, chunk_id, **chunk_masks
                    )
                elif hasattr(self._renderer, "clear_chunk_semantic_masks"):
                    self._renderer.clear_chunk_semantic_masks(handle, chunk_id)
            for stale_id in set(chunk_caches) - current_chunks:
                chunk_caches.pop(stale_id, None)
        self.redraw()

    def _semantic_masks_for_mesh(
        self,
        mesh: MeshArrays,
        table: object,
        resolver: Optional[Callable[..., object]],
        cache: dict[str, Any],
        entry_tags: set[str],
        selected_keys: set[str],
        preselected_key: Optional[str],
    ) -> dict[str, np.ndarray]:
        """Build compact masks for one primary mesh or incremental chunk."""

        selected: dict[str, list[np.ndarray]] = {
            "elements": [], "lines": [], "points": [],
        }
        preselected: dict[str, list[np.ndarray]] = {
            "elements": [], "lines": [], "points": [],
        }
        hidden: dict[str, list[np.ndarray]] = {
            "elements": [], "lines": [], "points": [],
        }
        if isinstance(table, PackedOwnerTable):
            for primitive_kind, suffix in (
                ("triangle", "elements"),
                ("line", "lines"),
                ("point", "points"),
            ):
                semantic = cache.setdefault(f"semantic_{suffix}", {})
                hit_values = cache.setdefault(f"hit_{suffix}", {})
                for key in selected_keys:
                    values = semantic.get(key)
                    if values is None:
                        values = self._semantic_primitives_for_key(
                            mesh, table, resolver, key, primitive_kind
                        )
                        semantic[key] = values
                    if len(values):
                        selected[suffix].append(values)
                if preselected_key is not None:
                    values = hit_values.get(preselected_key)
                    if (
                        values is None
                        and getattr(self, "_preselection_from_hit", False)
                    ):
                        values = np.empty(0, dtype=np.uint32)
                    if values is None:
                        values = semantic.get(preselected_key)
                    if values is None:
                        values = self._semantic_primitives_for_key(
                            mesh, table, resolver, preselected_key, primitive_kind
                        )
                        semantic[preselected_key] = values
                    if len(values):
                        preselected[suffix].append(values)
                hidden_values = self._hidden_primitives_for_visibility(
                    mesh, table, resolver, primitive_kind
                )
                if len(hidden_values):
                    hidden[suffix].append(hidden_values)

        counts = {
            "elements": mesh.element_count,
            "lines": 0 if mesh.lines is None else len(mesh.lines),
            "points": 0 if mesh.point_indices is None else len(mesh.point_indices),
        }
        if entry_tags & selected_keys:
            for suffix, count in counts.items():
                if count:
                    selected[suffix].append(np.arange(count, dtype=np.uint32))
        if preselected_key is not None and preselected_key in entry_tags:
            for suffix, count in counts.items():
                if count:
                    preselected[suffix].append(np.arange(count, dtype=np.uint32))

        visibility = getattr(self, "_visibility_state", VisibilityState())
        if not isinstance(table, PackedOwnerTable) and not visibility.is_default:
            hidden_tag_keys = {
                self._semantic_key(value) for value in visibility.hidden
            }
            isolated_tag_keys = {
                self._semantic_key(value) for value in visibility.isolated
            }
            hide_entry = bool(entry_tags & hidden_tag_keys)
            if visibility.isolated or visibility.isolated_kinds:
                hide_entry = hide_entry or not bool(entry_tags & isolated_tag_keys)
            if hide_entry:
                for suffix, count in counts.items():
                    if count:
                        hidden[suffix].append(np.arange(count, dtype=np.uint32))

        def combined(parts: list[np.ndarray]) -> np.ndarray:
            return (
                np.unique(np.concatenate(parts)).astype(np.uint32, copy=False)
                if parts else np.empty(0, dtype=np.uint32)
            )

        masks = {
            f"selected_{suffix}": combined(selected[suffix])
            for suffix in ("elements", "lines", "points")
        }
        masks.update({
            f"preselected_{suffix}": combined(preselected[suffix])
            for suffix in ("elements", "lines", "points")
        })
        masks.update({
            f"hidden_{suffix}": combined(hidden[suffix])
            for suffix in ("elements", "lines", "points")
        })
        return masks

    @staticmethod
    def _semantic_key(value: SemanticRef) -> str:
        return (
            f"{value.model_id}:{value.kind}:{value.key}"
            if value.source == "model"
            else str(value.key)
        )

    def _hidden_primitives_for_visibility(
        self,
        mesh: MeshArrays,
        table: PackedOwnerTable,
        resolver: Optional[Callable[..., object]],
        primitive_kind: str,
    ) -> np.ndarray:
        state = getattr(self, "_visibility_state", VisibilityState())
        if state.is_default:
            return np.empty(0, dtype=np.uint32)
        explicitly_hidden = np.zeros(table.owner_count, dtype=np.uint8)
        isolated = np.zeros(table.owner_count, dtype=np.uint8)
        for row in range(table.owner_count):
            try:
                ref = SemanticRef.from_owner(table.owner(row, resolver))
            except (TypeError, ValueError):
                continue
            explicitly_hidden[row] = (
                ref in state.hidden or ref.kind in state.hidden_kinds
            )
            isolated[row] = (
                ref in state.isolated or ref.kind in state.isolated_kinds
            )
        indices = getattr(table, f"{primitive_kind}_indices")
        offsets = getattr(table, f"{primitive_kind}_offsets")
        primitive_count = max(0, len(offsets) - 1)
        if primitive_count == 0:
            return np.empty(0, dtype=np.uint32)
        counts = np.diff(offsets.astype(np.int64, copy=False))
        primitives = np.repeat(np.arange(primitive_count, dtype=np.int64), counts)
        explicit_counts = (
            np.bincount(
                primitives,
                weights=explicitly_hidden[indices],
                minlength=primitive_count,
            )
            if len(indices) else np.zeros(primitive_count)
        )
        hidden_values = explicit_counts > 0
        if state.isolated or state.isolated_kinds:
            isolated_counts = (
                np.bincount(
                    primitives,
                    weights=isolated[indices],
                    minlength=primitive_count,
                )
                if len(indices) else np.zeros(primitive_count)
            )
            hidden_values |= isolated_counts == 0
        values = np.flatnonzero(hidden_values).astype(np.uint32, copy=False)
        if primitive_kind == "triangle" and mesh.triangle_to_element is not None:
            values = mesh.triangle_to_element[values]
        return np.unique(values).astype(np.uint32, copy=False)

    @staticmethod
    def _semantic_primitives_for_key(
        mesh: MeshArrays,
        table: PackedOwnerTable,
        resolver: Optional[Callable[..., object]],
        key: str,
        primitive_kind: str,
    ) -> np.ndarray:
        """Resolve one semantic key with a single vectorized ownership scan.

        The previous eager inverse compared every unique owner against the
        complete primitive map, which became quadratic for one-owner-per-FE-
        element scenes.  Hits populate this cache in O(1); an application-set
        key that has not been hit pays one O(n) packed-array scan.
        """

        indices = getattr(table, f"{primitive_kind}_indices")
        offsets = getattr(table, f"{primitive_kind}_offsets")
        if not len(indices):
            return np.empty(0, dtype=np.uint32)
        rows: list[int] = []
        try:
            string_slot = table.string_keys.index(str(key))
        except ValueError:
            string_slot = -1
        if string_slot >= 0:
            mask = (table.domains == 1) & (table.string_slots == string_slot)
            rows.extend(np.flatnonzero(mask).tolist())
        try:
            numeric_key = int(key)
        except (TypeError, ValueError):
            numeric_key = -1
        if numeric_key >= 0:
            mask = (
                (table.domains == 1)
                & (table.string_slots == np.iinfo(np.uint32).max)
                & (table.numeric_ids == numeric_key)
            )
            rows.extend(np.flatnonzero(mask).tolist())
        # Model/resolver identities need their public string representation;
        # resolve rows lazily only when the packed application columns did not
        # identify the requested key.
        if not rows:
            for row in range(table.owner_count):
                owner = table.owner(row, resolver)
                owner_key = (
                    owner.key if isinstance(owner, PickOwner)
                    else f"{owner.model_id}:{owner.kind}:{owner.id}"
                )
                if owner_key == key:
                    rows.append(row)
        if not rows:
            return np.empty(0, dtype=np.uint32)
        counts = np.diff(offsets.astype(np.int64, copy=False))
        primitives = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        matches = np.isin(indices, np.asarray(rows, np.uint32))
        owned_primitives = primitives[matches]
        if primitive_kind == "triangle":
            mapping = (
                np.arange(mesh.triangle_count, dtype=np.uint32)
                if mesh.triangle_to_element is None else mesh.triangle_to_element
            )
            owned_primitives = owned_primitives[owned_primitives < len(mapping)]
            owned_primitives = mapping[owned_primitives]
        return np.unique(owned_primitives).astype(np.uint32, copy=False)

    @staticmethod
    def _cache_hit_elements(
        handle: MeshHandle,
        entry: dict[str, Any],
        primitive_kind: str,
        primitive: int,
        owners: Sequence[PickOwner],
        *,
        chunk_id: object = None,
    ) -> None:
        if not owners:
            return
        if chunk_id is None:
            mesh = handle.mesh
            cache = entry
        else:
            chunks = dict(handle.chunks)
            mesh = chunks.get(chunk_id)
            if mesh is None:
                return
            cache = entry.setdefault("chunk_semantics", {}).setdefault(
                chunk_id, {}
            )
        limits = {
            "triangle": mesh.triangle_count,
            "line": 0 if mesh.lines is None else len(mesh.lines),
            "point": 0 if mesh.point_indices is None else len(mesh.point_indices),
        }
        if primitive_kind not in limits or primitive < 0 or primitive >= limits[primitive_kind]:
            return
        value_index = (
            int(mesh.triangle_to_element[int(primitive)])
            if primitive_kind == "triangle" and mesh.triangle_to_element is not None
            else int(primitive)
        )
        suffix = {"triangle": "elements", "line": "lines", "point": "points"}[
            primitive_kind
        ]
        semantic = cache.setdefault(f"hit_{suffix}", {})
        value = np.asarray((value_index,), dtype=np.uint32)
        for owner in owners:
            previous = semantic.get(owner.key)
            semantic[owner.key] = (
                value
                if previous is None
                else np.union1d(previous, value).astype(np.uint32, copy=False)
            )

    def set_highlight(
        self,
        tags: Iterable[str],
        fill: Optional[str] = None,
        outline: Optional[str] = None,
    ) -> None:
        self._highlighted_tags = frozenset(str(tag) for tag in tags)
        if fill is not None:
            self._highlight_fill = str(fill)
        if outline is not None:
            self._highlight_outline = str(outline)
        self._apply_highlight_masks()

    def clear_highlight(self) -> None:
        self.set_highlight(())

    def highlighted_tags(self) -> frozenset[str]:
        return self._highlighted_tags

    def set_preselection(self, key: Optional[str]) -> None:
        self._preselected_key = None if key is None else str(key)
        self._preselection_from_hit = False
        self._apply_highlight_masks()

    def _set_hit_preselection(self, key: Optional[str]) -> None:
        """Apply the one picked primitive without scanning packed owner tables."""

        self._preselected_key = None if key is None else str(key)
        self._preselection_from_hit = key is not None
        self._apply_highlight_masks()

    @property
    def preselected_key(self) -> Optional[str]:
        return self._preselected_key

    @staticmethod
    def _near_clip(polygon: np.ndarray, near: float) -> np.ndarray:
        if len(polygon) == 0:
            return polygon
        result: list[np.ndarray] = []
        for index, current in enumerate(polygon):
            following = polygon[(index + 1) % len(polygon)]
            current_inside = current[2] >= near
            following_inside = following[2] >= near
            if current_inside:
                result.append(current)
            if current_inside != following_inside:
                amount = (near - current[2]) / (following[2] - current[2])
                result.append(current + amount * (following - current))
        return np.asarray(result, dtype=np.float64).reshape((-1, 3))

    @staticmethod
    def _far_clip(polygon: np.ndarray, far: float) -> np.ndarray:
        if len(polygon) == 0:
            return polygon
        result: list[np.ndarray] = []
        for index, current in enumerate(polygon):
            following = polygon[(index + 1) % len(polygon)]
            current_inside = current[2] <= far
            following_inside = following[2] <= far
            if current_inside:
                result.append(current)
            if current_inside != following_inside:
                amount = (far - current[2]) / (following[2] - current[2])
                result.append(current + amount * (following - current))
        return np.asarray(result, dtype=np.float64).reshape((-1, 3))

    def _selection_binding(
        self,
        handle: MeshHandle,
        entry: dict[str, Any],
        primitive_kind: str,
        primitive: int,
        *,
        chunk_id: object = None,
    ) -> Optional[PickBinding]:
        if chunk_id is None:
            table = entry.get("owners")
            resolver = entry.get("owner_resolver")
        else:
            try:
                table, resolver = handle.chunk_ownership(chunk_id)
            except KeyError:
                return None
        if isinstance(table, PackedOwnerTable):
            try:
                owners = table.owners_for(primitive_kind, primitive, resolver)
            except IndexError:
                owners = ()
            converted = tuple(
                owner
                if isinstance(owner, PickOwner)
                else PickOwner(
                    f"{owner.model_id}:{owner.kind}:{owner.id}",
                    f"geometry.{owner.kind}",
                    owner.priority,
                    owner,
                )
                for owner in owners
            )
            if converted:
                return PickBinding(converted)
        tags = tuple(entry.get("tags", ()))
        if tags:
            return PickBinding(
                tuple(PickOwner(tag, "tag") for tag in sorted(tags))
            )
        if chunk_id is not None:
            # Chunk-local identity must be supplied explicitly; a local index
            # alone is not stable across replacement and must not be exposed.
            return None
        return PickBinding.one(
            f"mesh:{id(handle)}:{primitive_kind}:{primitive}",
            f"mesh.{primitive_kind}",
        )

    def _selection_key(self, viewport: tuple[int, int]) -> tuple[object, ...]:
        camera = self.camera
        return (
            viewport,
            camera.position.to_tuple(),
            camera.target.to_tuple(),
            camera.fov,
            camera.near,
            camera.far,
            getattr(self, "show_mesh_lines", True),
            getattr(self, "_occlude_lines", True),
            None if self._section_plane is None else self._section_plane.key,
            getattr(self, "_visibility_state", VisibilityState()),
            tuple(
                (
                    identifier,
                    entry["handle"].generations,
                    entry["handle"].visible,
                    entry["handle"].removed,
                    entry["handle"].deformation_scale,
                )
                for identifier, entry in self._display_entries().items()
            ),
        )

    def _projected_selection_index(self) -> ProjectedSelectionIndex:
        viewport = self._host.framebuffer_size()
        key = self._selection_key(viewport)
        if self._selection_index is not None and key == self._selection_index_key:
            return self._selection_index
        width, height = viewport
        right, up, forward = self.camera.basis()
        basis = np.asarray(
            (right.to_tuple(), up.to_tuple(), forward.to_tuple()), dtype=np.float64
        )
        origin = np.asarray(self.camera.position.to_tuple(), dtype=np.float64)
        scale = 0.5 * height / math.tan(self.camera.fov * 0.5)
        near, far = float(self.camera.near), float(self.camera.far)
        plane = self._section_plane
        projected: list[ProjectedPrimitive] = []
        primitive_id = 0

        def camera_points(world: np.ndarray) -> np.ndarray:
            return (world - origin) @ basis.T

        def screen_points(camera_values: np.ndarray) -> tuple[tuple[float, float], ...]:
            return tuple(
                (
                    0.5 * width + float(point[0] * scale / point[2]),
                    0.5 * height - float(point[1] * scale / point[2]),
                )
                for point in camera_values
            )

        for entry in self._display_entries().values():
            handle: MeshHandle = entry["handle"]
            if handle.removed or not handle.visible:
                continue
            for chunk_id, mesh in [(None, handle.mesh), *handle.chunks]:
                positions = mesh.positions
                if mesh.displacements is not None and handle.deformation_scale:
                    positions = positions + handle.deformation_scale * mesh.displacements
                transform = handle.transform
                world = positions @ transform[:3, :3].T + transform[:3, 3]
                mapping = (
                    mesh.triangle_to_element
                    if mesh.triangle_to_element is not None
                    else np.arange(mesh.triangle_count, dtype=np.uint32)
                )
                for triangle, indices in enumerate(mesh.triangles):
                    if (
                        mesh.active_elements is not None
                        and not bool(mesh.active_elements[int(mapping[triangle])])
                    ):
                        continue
                    polygon = world[indices]
                    if plane is not None and plane.enabled:
                        polygon = np.asarray(
                            [point.to_tuple() for point in plane.clip_polygon(polygon)],
                            dtype=np.float64,
                        ).reshape((-1, 3))
                    camera_polygon = self._far_clip(
                        self._near_clip(camera_points(polygon), near), far
                    )
                    if len(camera_polygon) < 3:
                        continue
                    projected_polygon = screen_points(camera_polygon)
                    if entry.get("appearance", {}).get("cull_backface", True):
                        signed_area = sum(
                            projected_polygon[index][0]
                            * projected_polygon[(index + 1) % len(projected_polygon)][1]
                            - projected_polygon[(index + 1) % len(projected_polygon)][0]
                            * projected_polygon[index][1]
                            for index in range(len(projected_polygon))
                        )
                        # OpenGL's counter-clockwise front face becomes clockwise
                        # after conversion to top-left screen coordinates.
                        if signed_area >= -1.0e-9:
                            continue
                    binding = (
                        self._selection_binding(
                            handle,
                            entry,
                            "triangle",
                            triangle,
                            chunk_id=chunk_id,
                        )
                        if entry.get("appearance", {}).get("pickable", True)
                        else None
                    )
                    if not self._semantic_binding_visible(binding):
                        continue
                    projected.append(
                        ProjectedPrimitive(
                            primitive_id,
                            "polygon",
                            projected_polygon,
                            tuple(float(value) for value in camera_polygon[:, 2]),
                            binding,
                            layer=float(entry.get("layer", 0)),
                            item=int(entry.get("item", -1)),
                        )
                    )
                    primitive_id += 1
                if (
                    mesh.lines is not None
                    and (
                        getattr(self, "show_mesh_lines", True)
                        or not entry.get("appearance", {}).get("mesh_lines", False)
                    )
                ):
                    for line, indices in enumerate(mesh.lines):
                        segment = world[indices]
                        if plane is not None and plane.enabled:
                            clipped = plane.clip_segment(segment[0], segment[1])
                            if clipped is None:
                                continue
                            segment = np.asarray(
                                [point.to_tuple() for point in clipped], dtype=np.float64
                            )
                        camera_segment = camera_points(segment)
                        first_inside = camera_segment[0, 2] >= near
                        second_inside = camera_segment[1, 2] >= near
                        if not first_inside and not second_inside:
                            continue
                        if first_inside != second_inside:
                            amount = (near - camera_segment[0, 2]) / (
                                camera_segment[1, 2] - camera_segment[0, 2]
                            )
                            intersection = camera_segment[0] + amount * (
                                camera_segment[1] - camera_segment[0]
                            )
                            camera_segment[0 if not first_inside else 1] = intersection
                        first_inside = camera_segment[0, 2] <= far
                        second_inside = camera_segment[1, 2] <= far
                        if not first_inside and not second_inside:
                            continue
                        if first_inside != second_inside:
                            amount = (far - camera_segment[0, 2]) / (
                                camera_segment[1, 2] - camera_segment[0, 2]
                            )
                            intersection = camera_segment[0] + amount * (
                                camera_segment[1] - camera_segment[0]
                            )
                            camera_segment[0 if not first_inside else 1] = intersection
                        binding = (
                            self._selection_binding(
                                handle,
                                entry,
                                "line",
                                line,
                                chunk_id=chunk_id,
                            )
                            if entry.get("appearance", {}).get("pickable", True)
                            and not entry.get("appearance", {}).get("mesh_lines", False)
                            else None
                        )
                        if not self._semantic_binding_visible(binding):
                            continue
                        line_depths = tuple(
                            float(value) for value in camera_segment[:, 2]
                        )
                        if (
                            entry.get("appearance", {}).get("line_overlay", False)
                            or not getattr(self, "_occlude_lines", True)
                        ):
                            line_depths = (near, near)
                        projected.append(
                            ProjectedPrimitive(
                                primitive_id,
                                "segment",
                                screen_points(camera_segment),
                                line_depths,
                                binding,
                                radius=2.5,
                                layer=float(entry.get("layer", 0)),
                                item=int(entry.get("item", -1)),
                            )
                        )
                        primitive_id += 1
                if mesh.point_indices is not None:
                    for point, node in enumerate(mesh.point_indices):
                        position = world[int(node)]
                        if plane is not None and plane.enabled and not plane.contains(position):
                            continue
                        camera_point = camera_points(position[None, :])[0]
                        if not near <= camera_point[2] < far:
                            continue
                        binding = (
                            self._selection_binding(
                                handle,
                                entry,
                                "point",
                                point,
                                chunk_id=chunk_id,
                            )
                            if entry.get("appearance", {}).get("pickable", True)
                            else None
                        )
                        if not self._semantic_binding_visible(binding):
                            continue
                        projected.append(
                            ProjectedPrimitive(
                                primitive_id,
                                "point",
                                screen_points(camera_point[None, :]),
                                (
                                    near
                                    if entry.get("appearance", {}).get("point_overlay", False)
                                    else float(camera_point[2]),
                                ),
                                binding,
                                radius=3.5,
                                layer=float(entry.get("layer", 0)),
                                item=int(entry.get("item", -1)),
                            )
                        )
                        primitive_id += 1
        self._selection_index = ProjectedSelectionIndex(projected, width, height)
        self._selection_index_key = key
        return self._selection_index

    def _gpu_point_hits(
        self,
        x: int,
        y: int,
        selection_filter: SelectionFilter,
        radius: int = 0,
    ) -> Optional[tuple[SelectionHit, ...]]:
        """Resolve the front pixel from the cached integer ID/depth target."""

        renderer = getattr(self, "_renderer", None)
        host = getattr(self, "_host", None)
        if renderer is None or host is None or not hasattr(renderer, "pick"):
            return None
        host.make_current()
        viewport = host.framebuffer_size()
        remaining_kinds = {"triangle", "line", "point"}
        owners: tuple[PickOwner, ...] = ()
        filtered_front = False
        chunk_id: object = None
        while remaining_kinds:
            picker = getattr(renderer, "pick_detail", renderer.pick)
            picked = picker(
                int(x), int(y), self.camera, viewport, self._section_plane,
                show_mesh_lines=self.show_mesh_lines,
                occlude_lines=self._occlude_lines,
                radius=radius,
                primitive_kinds=tuple(sorted(remaining_kinds)),
            )
            if picked is None:
                # A rejected owner of a class may still have an eligible
                # same-class primitive behind it. The ID target cannot peel
                # that depth stack, so signal the caller's CPU through-index
                # fallback only for this explicitly filtered edge case.
                return None if filtered_front else ()
            if len(picked) == 4:
                handle, primitive_kind, primitive, chunk_id = picked
            else:  # compatibility with 0.4 renderer doubles
                handle, primitive_kind, primitive = picked
                chunk_id = None
            entry = self._display_entries().get(id(handle))
            if entry is None or not entry.get("appearance", {}).get("pickable", True):
                return ()
            binding = self._selection_binding(
                handle,
                entry,
                primitive_kind,
                int(primitive),
                chunk_id=chunk_id,
            )
            owners = tuple(
                owner
                for owner in (() if binding is None else binding.owners)
                if selection_filter.accepts(owner)
            )
            if owners:
                break
            # A filtered overlay annotation is selection-transparent. Rebuild
            # the cached ID target without that primitive class and expose the
            # eligible geometry behind it; the default unfiltered path remains
            # one cached read and never projects primitives in Python.
            filtered_front = True
            remaining_kinds.discard(primitive_kind)
        if not owners:
            return None if filtered_front else ()
        self._cache_hit_elements(
            handle,
            entry,
            primitive_kind,
            int(primitive),
            owners,
            chunk_id=chunk_id,
        )

        depth = 0.0
        mesh = (
            handle.mesh
            if chunk_id is None
            else dict(handle.chunks).get(chunk_id)
        )
        if mesh is None:
            return ()
        indices: Optional[np.ndarray] = None
        if primitive_kind == "triangle" and primitive < mesh.triangle_count:
            indices = mesh.triangles[int(primitive)]
        elif (
            primitive_kind == "line"
            and mesh.lines is not None
            and primitive < len(mesh.lines)
        ):
            indices = mesh.lines[int(primitive)]
        elif (
            primitive_kind == "point"
            and mesh.point_indices is not None
            and primitive < len(mesh.point_indices)
        ):
            indices = mesh.point_indices[int(primitive):int(primitive) + 1]
        if indices is not None and len(indices):
            positions = mesh.positions
            if mesh.displacements is not None and handle.deformation_scale:
                positions = positions + handle.deformation_scale * mesh.displacements
            transform = handle.transform
            world = positions[np.asarray(indices, dtype=np.intp)] @ transform[:3, :3].T
            world += transform[:3, 3]
            centre = world.mean(axis=0)
            forward = np.asarray(self.camera.basis()[2].to_tuple(), np.float64)
            depth = float(np.dot(
                centre - np.asarray(self.camera.position.to_tuple(), np.float64),
                forward,
            ))
        return tuple(
            SelectionHit(
                owner=owner,
                primitive=int(primitive),
                depth=depth,
                visible=True,
                item=int(entry.get("item", -1)),
            )
            for owner in sorted(owners, key=lambda value: (-value.priority, value.key))
        )

    def query_point(
        self,
        x: int,
        y: int,
        *,
        selection_filter: Optional[SelectionFilter] = None,
        radius: Optional[int] = None,
        config: Optional[SelectionConfig] = None,
    ) -> tuple[SelectionHit, ...]:
        policy = config or self._selection_config
        active_filter = selection_filter or policy.filter
        effective_radius = (
            policy.click_radius_px if radius is None else max(0, int(radius))
        )
        gpu_hits = self._gpu_point_hits(
            int(x), int(y), active_filter, effective_radius
        )
        if (
            policy.depth is SelectionDepth.VISIBLE
            and self._display_primitive_count(_CPU_POINT_STACK_LIMIT)
            > _CPU_POINT_STACK_LIMIT
        ):
            if gpu_hits is not None:
                return gpu_hits
        hits = self._projected_selection_index().point_hits(
            x, y, active_filter,
            radius=effective_radius,
        )
        if not gpu_hits:
            return hits
        front_keys = {hit.key for hit in gpu_hits}
        return (*gpu_hits, *(hit for hit in hits if hit.key not in front_keys))

    def query_rectangle(
        self,
        start: tuple[int, int] | tuple[int, int, int, int],
        end: Optional[tuple[int, int]] = None,
        *,
        crossing: Optional[bool] = None,
        selection_filter: Optional[SelectionFilter] = None,
        depth: Optional[SelectionDepth] = None,
        config: Optional[SelectionConfig] = None,
    ) -> tuple[SelectionHit, ...]:
        policy = config or self._selection_config
        if end is None:
            if len(start) != 4:
                raise ValueError("rectangle requires start/end points or four coordinates")
            rect = tuple(float(value) for value in start)
            start_point, end_point = rect[:2], rect[2:]
        else:
            start_point, end_point = start, end
            rect = (*start_point, *end_point)
        crossing_value = (
            bool(policy.directional and end_point[0] < start_point[0])
            if crossing is None else bool(crossing)
        )
        return self._projected_selection_index().rectangle_hits(
            rect, selection_filter or policy.filter, crossing=crossing_value,
            depth=policy.depth if depth is None else SelectionDepth(depth)
        )

    def query_lasso(
        self,
        points: Sequence[tuple[int, int]],
        *,
        selection_filter: Optional[SelectionFilter] = None,
        depth: Optional[SelectionDepth] = None,
        config: Optional[SelectionConfig] = None,
    ) -> tuple[SelectionHit, ...]:
        policy = config or self._selection_config
        return self._projected_selection_index().polygon_hits(
            tuple(points), selection_filter or policy.filter,
            depth=policy.depth if depth is None else SelectionDepth(depth)
        )

    def _mesh_changed(self, handle: MeshHandle, change: str) -> None:
        if change == "remove":
            self._renderer.remove_mesh(handle)
            self._entries.pop(id(handle), None)
        elif change in {"topology", "selection"}:
            entry = self._entries.get(id(handle))
            if entry is not None:
                # A chunk replacement may preserve its ownership, while an
                # ownership-only change uses the selection generation.  Drop
                # both local inverses so neither can retain stale indices.
                entry["chunk_semantics"] = {}
                if not self._animation_frame_active:
                    self._apply_highlight_masks()
        self._renderer.pick_dirty = True
        self._selection_index = None
        self.redraw()

    def add_mesh_arrays(
        self,
        mesh: MeshArrays,
        *,
        color: str = "#9aa7b4",
        outline: str = "",
        width: int = 1,
        layer: int = 5,
        cull_backface: bool = True,
        opacity: float = 1.0,
        stipple: str = "",
        back_color: str = "",
        tags: str = "",
        lit: bool = True,
        two_sided_shell: bool = False,
        face_colors: Optional[Sequence[str]] = None,
        scalar_range: Optional[tuple[float, float]] = None,
        invalid_color: str = "#808080",
        owners: Optional[PackedOwnerTable] = None,
        owner_resolver: Optional[Callable[..., object]] = None,
        line_color: Optional[str] = None,
        line_width: Optional[int] = None,
        point_color: str = "#2563eb",
        point_outline: str = "",
        point_size: int = 6,
        draw_overlay: bool = False,
        pickable: bool = True,
        depth_only: bool = False,
        **_appearance: object,
    ) -> MeshHandle:
        """Register retained arrays using the shared backend signature."""

        return self._add_mesh_arrays(
            mesh,
            color=color,
            outline=outline,
            width=width,
            layer=layer,
            cull_backface=cull_backface,
            opacity=opacity,
            stipple=stipple,
            back_color=back_color,
            tags=tags,
            lit=lit,
            two_sided_shell=two_sided_shell,
            face_colors=face_colors,
            scalar_range=scalar_range,
            invalid_color=invalid_color,
            owners=owners,
            owner_resolver=owner_resolver,
            line_color=line_color,
            line_width=line_width,
            point_color=point_color,
            point_outline=point_outline,
            point_size=point_size,
            draw_overlay=draw_overlay,
            pickable=pickable,
            depth_only=depth_only,
            **_appearance,
        )

    def _add_mesh_arrays(
        self,
        mesh: MeshArrays,
        *,
        color: str = "#9aa7b4",
        outline: str = "",
        width: int = 1,
        layer: int = 5,
        cull_backface: bool = True,
        opacity: float = 1.0,
        stipple: str = "",
        back_color: str = "",
        tags: str = "",
        lit: bool = True,
        two_sided_shell: bool = False,
        face_colors: Optional[Sequence[str]] = None,
        scalar_range: Optional[tuple[float, float]] = None,
        invalid_color: str = "#808080",
        point_color: str = "#2563eb",
        point_outline: str = "",
        point_size: int = 6,
        owners: Optional[PackedOwnerTable] = None,
        owner_resolver: Optional[Callable[..., object]] = None,
        line_color: Optional[str] = None,
        line_width: Optional[int] = None,
        draw_overlay: bool = False,
        pickable: bool = True,
        depth_only: bool = False,
        **_appearance: object,
    ) -> MeshHandle:
        if not isinstance(mesh, MeshArrays):
            raise TypeError("mesh must be MeshArrays")
        if owners is not None and not isinstance(owners, PackedOwnerTable):
            raise TypeError("owners must be a PackedOwnerTable")
        if owners is not None:
            mapped = len(owners.triangle_offsets) - 1
            if mapped not in (0, mesh.triangle_count):
                raise ValueError("triangle owner mappings must match triangle count")
        if face_colors is not None and len(face_colors) not in (
            0, mesh.triangle_count, mesh.element_count
        ):
            raise ValueError("face_colors must match triangle or element count")
        handle = MeshHandle(mesh, on_change=self._mesh_changed)
        effective_line_color = str(line_color or outline or "#334155")
        effective_line_width = float(width if line_width is None else line_width)
        appearance = {
            "color": str(color),
            "line_color": effective_line_color,
            "point_color": str(point_color),
            "line_width": effective_line_width,
            "point_size": float(point_size),
            "opacity": float(opacity),
            "cull_backface": bool(cull_backface and not two_sided_shell and not back_color),
            "line_overlay": bool(draw_overlay),
            "point_overlay": bool(_appearance.get("_point_overlay", False)),
            "lit": bool(lit),
            "stipple": str(stipple),
            "stipple_phase": int(_appearance.get("stipple_phase", 0)),
            "pickable": bool(pickable),
            "depth_only": bool(depth_only),
            "layer": int(layer),
            "mesh_lines": bool(_appearance.get("_mesh_outline", False)),
            "face_colors": None if face_colors is None else tuple(str(v) for v in face_colors),
            "scalar_range": scalar_range,
            "back_color": str(back_color),
            "invalid_color": str(invalid_color),
            "point_outline": str(point_outline),
        }
        self._legacy_item_counter += 1
        self._entries[id(handle)] = {
            "handle": handle,
            "owners": owners,
            "owner_resolver": owner_resolver,
            "appearance": appearance,
            "layer": int(layer),
            "tags": frozenset(str(tags).split()),
            "item": self._legacy_item_counter,
            "face_colors": None if face_colors is None else tuple(face_colors),
            # Filled lazily from integer-pick hits or one vectorized lookup per
            # application-requested key; never construct an O(n^2) inverse.
            "semantic_elements": {},
            "hit_elements": {},
            "semantic_lines": {},
            "hit_lines": {},
            "semantic_points": {},
            "hit_points": {},
            "chunk_semantics": {},
        }
        self._host.make_current()
        self._renderer.add_mesh(handle, **{
            key: value for key, value in appearance.items()
            if key in {
                "color", "line_color", "point_color", "point_outline", "line_width", "point_size",
                "opacity", "cull_backface", "line_overlay", "point_overlay", "lit", "stipple",
                "stipple_phase",
                "pickable", "scalar_range",
                "depth_only",
                "layer",
                "mesh_lines",
                "face_colors", "back_color", "invalid_color",
            }
        })
        self.redraw()
        self._selection_index = None
        return handle

    def add_layer(self, layer: object) -> object:
        attach = getattr(layer, "attach", None)
        if not callable(attach):
            raise TypeError("layer must provide attach(viewer)")
        return attach(self)

    @staticmethod
    def _coerce_binding(value: object, tags: str = "") -> Optional[PickBinding]:
        if value is None:
            key = next((tag for tag in str(tags).split() if tag), "")
            return None if not key else PickBinding.one(key)
        if isinstance(value, PickBinding):
            return value
        if isinstance(value, PickOwner):
            return PickBinding((value,))
        if isinstance(value, str):
            return PickBinding.one(value)
        try:
            return PickBinding(tuple(
                owner if isinstance(owner, PickOwner) else PickOwner(*owner)
                for owner in value  # type: ignore[union-attr]
            ))
        except (TypeError, ValueError) as error:
            raise TypeError("binding must be PickBinding, PickOwner, or owner iterable") from error

    def _legacy_mesh_batch(
        self,
        vertices: Iterable[object],
        faces: Iterable[Sequence[int]],
        *,
        color: str,
        outline: str,
        width: int,
        layer: int,
        cull_backface: bool,
        opacity: float,
        stipple: str,
        tags: str,
        lit: bool,
        two_sided_shell: bool,
        bindings: Optional[Sequence[Optional[PickBinding]]],
        back_color: str = "",
    ) -> None:
        points = np.asarray([as_point(value).to_tuple() for value in vertices], np.float64)
        face_values = [tuple(int(index) for index in face) for face in faces]
        triangles: list[tuple[int, int, int]] = []
        triangle_bindings: list[Optional[PickBinding]] = []
        edges: set[tuple[int, int]] = set()
        for face_index, face in enumerate(face_values):
            if len(face) < 3:
                continue
            binding = None if bindings is None else bindings[face_index]
            for index in range(1, len(face) - 1):
                triangles.append((face[0], face[index], face[index + 1]))
                triangle_bindings.append(binding)
            if outline:
                edges.update(
                    tuple(sorted((face[index - 1], face[index])))
                    for index in range(len(face))
                )
        if not triangles and not edges:
            return
        lines = None if not edges else np.asarray(sorted(edges), np.uint32).reshape((-1, 2))
        arrays = MeshArrays(
            points,
            np.asarray(triangles, np.uint32).reshape((-1, 3)),
            lines=lines,
        )
        owners = None
        if any(binding is not None for binding in triangle_bindings):
            owners = PackedOwnerTable.from_owners(
                triangles=triangle_bindings,
                lines=[None] * (0 if lines is None else len(lines)),
            )
        effective_two_sided = bool(
            two_sided_shell or stipple or float(opacity) < 0.97
        )
        self.add_mesh_arrays(
            arrays,
            color=color,
            outline=outline,
            width=width,
            layer=layer,
            cull_backface=bool(cull_backface and not effective_two_sided),
            opacity=opacity,
            stipple=stipple,
            tags=tags,
            lit=lit,
            two_sided_shell=effective_two_sided,
            owners=owners,
            line_color=outline or color,
            line_width=width,
            back_color=back_color,
            _mesh_outline=bool(outline),
        )

    def add_line(
        self,
        start: Point3D,
        end: Point3D,
        color: str = "black",
        width: int = 1,
        layer: int = 30,
        draw_overlay: bool = False,
        tags: str = "",
        binding: object = None,
    ) -> None:
        owner = self._coerce_binding(binding, tags)
        arrays = MeshArrays(
            np.asarray((as_point(start).to_tuple(), as_point(end).to_tuple()), np.float64),
            np.empty((0, 3), np.uint32),
            lines=np.asarray(((0, 1),), np.uint32),
        )
        owners = None if owner is None else PackedOwnerTable.from_owners(lines=(owner,))
        self.add_mesh_arrays(
            arrays, color=color, line_color=color, line_width=width, layer=layer,
            tags=tags, owners=owners, draw_overlay=draw_overlay,
        )

    def add_text(
        self,
        point: Point3D,
        text: str,
        color: str = "black",
        font: tuple[str, int, str] = ("Segoe UI", 9, "bold"),
        anchor: str = "center",
        layer: int = 35,
        draw_overlay: bool = True,
    ) -> None:
        self._world_text.append({
            "point": as_point(point), "text": str(text), "color": str(color),
            "font": font, "anchor": str(anchor), "layer": int(layer),
            "draw_overlay": bool(draw_overlay),
        })
        self.redraw()

    def add_polygon(
        self,
        vertices: Iterable[Point3D],
        color: str = "gray",
        outline: str = "black",
        width: int = 1,
        cull_backface: bool = False,
        layer: int = 5,
        stipple: str = "",
        tags: str = "",
        back_color: str = "",
        two_sided_shell: bool = False,
        opacity: Optional[float] = None,
        lit: bool = True,
        binding: object = None,
    ) -> None:
        points = list(vertices)
        self._legacy_mesh_batch(
            points, [tuple(range(len(points)))], color=color, outline=outline,
            width=width, layer=layer, cull_backface=cull_backface,
            opacity=1.0 if opacity is None else float(opacity), stipple=stipple,
            tags=tags, lit=lit, two_sided_shell=two_sided_shell,
            bindings=[self._coerce_binding(binding, tags)], back_color=back_color,
        )

    def add_faces(
        self,
        polygons: Iterable[object],
        colors: object = "#9aa7b4",
        outline: str = "",
        width: int = 1,
        layer: int = 5,
        cull_backface: bool = False,
        opacity: float = 1.0,
        stipple: str = "",
        back_colors: Optional[Sequence[str]] = None,
        tags: str = "",
        lit: bool = True,
        two_sided_shell: bool = False,
        bindings: object = None,
    ) -> None:
        polygon_values = [list(polygon) for polygon in polygons]  # type: ignore[arg-type]
        total = len(polygon_values)
        if total == 0:
            return
        color_values = [str(colors)] * total if isinstance(colors, str) else [str(v) for v in colors]  # type: ignore[union-attr]
        if len(color_values) != total:
            raise ValueError(f"colors has {len(color_values)} entries for {total} faces")
        back_values = [""] * total if back_colors is None else [str(v) for v in back_colors]
        if len(back_values) != total:
            raise ValueError(f"back_colors has {len(back_values)} entries for {total} faces")
        if bindings is None or isinstance(bindings, (PickBinding, PickOwner, str)):
            binding_values = [self._coerce_binding(bindings, tags)] * total
        else:
            raw_bindings = list(bindings)
            if len(raw_bindings) != total:
                raise ValueError(f"bindings has {len(raw_bindings)} entries for {total} faces")
            binding_values = [self._coerce_binding(value, tags) for value in raw_bindings]
        groups: dict[tuple[str, str], list[int]] = {}
        for index, key in enumerate(zip(color_values, back_values)):
            groups.setdefault(key, []).append(index)
        for (front, back), indices in groups.items():
            vertices: list[object] = []
            faces: list[tuple[int, ...]] = []
            group_bindings: list[Optional[PickBinding]] = []
            for index in indices:
                offset = len(vertices)
                polygon = polygon_values[index]
                vertices.extend(polygon)
                faces.append(tuple(range(offset, offset + len(polygon))))
                group_bindings.append(binding_values[index])
            self._legacy_mesh_batch(
                vertices, faces, color=front, outline=outline, width=width,
                layer=layer, cull_backface=cull_backface, opacity=opacity,
                stipple=stipple, tags=tags, lit=lit,
                two_sided_shell=two_sided_shell, bindings=group_bindings,
                back_color=back,
            )

    def add_markers(
        self,
        points: Iterable[object],
        colors: object = "#2563eb",
        size: object = 6,
        outline: object = "",
        layer: int = 32,
        tags: str = "",
        bindings: object = None,
    ) -> None:
        values = [as_point(point) for point in points]
        total = len(values)
        if total == 0:
            return

        def expand(value: object, transform: Callable[[object], object], name: str) -> list[object]:
            if isinstance(value, str):
                return [transform(value)] * total
            try:
                result = [transform(item) for item in value]  # type: ignore[union-attr]
            except TypeError:
                result = [transform(value)] * total
            if len(result) != total:
                raise ValueError(f"{name} has {len(result)} entries for {total} markers")
            return result

        color_values = expand(colors, lambda value: str(value), "colors")
        size_values = expand(size, lambda value: max(1, int(value)), "size")
        outline_values = expand(outline, lambda value: str(value), "outline")
        if bindings is None or isinstance(bindings, (PickBinding, PickOwner, str)):
            binding_values = [self._coerce_binding(bindings, tags)] * total
        else:
            raw = list(bindings)
            if len(raw) != total:
                raise ValueError(f"bindings has {len(raw)} entries for {total} markers")
            binding_values = [self._coerce_binding(value, tags) for value in raw]
        groups: dict[tuple[str, int, str], list[int]] = {}
        for index, key in enumerate(zip(color_values, size_values, outline_values)):
            groups.setdefault(key, []).append(index)  # type: ignore[arg-type]
        for (color, marker_size, marker_outline), indices in groups.items():
            positions = np.asarray([values[index].to_tuple() for index in indices], np.float64)
            arrays = MeshArrays(
                positions, np.empty((0, 3), np.uint32),
                point_indices=np.arange(len(indices), dtype=np.uint32),
            )
            selected_bindings = [binding_values[index] for index in indices]
            owners = None if not any(selected_bindings) else PackedOwnerTable.from_owners(points=selected_bindings)
            self.add_mesh_arrays(
                arrays, color=color, point_color=color, point_outline=marker_outline,
                point_size=marker_size, layer=layer, tags=tags, owners=owners,
                _point_overlay=True,
            )

    def add_mesh(
        self,
        vertices: Iterable[object],
        faces: Iterable[Sequence[int]],
        color: str = "#9aa7b4",
        outline: str = "",
        width: int = 1,
        layer: int = 5,
        cull_backface: bool = True,
        opacity: float = 1.0,
        stipple: str = "",
        back_color: str = "",
        tags: str = "",
        lit: bool = True,
        face_colors: Optional[Sequence[str]] = None,
        two_sided_shell: bool = False,
        bindings: object = None,
    ) -> None:
        vertex_values = list(vertices)
        face_values = [tuple(int(index) for index in face) for face in faces]
        total = len(face_values)
        if bindings is None or isinstance(bindings, (PickBinding, PickOwner, str)):
            binding_values = [self._coerce_binding(bindings, tags)] * total
        else:
            raw = list(bindings)
            if len(raw) != total:
                raise ValueError(f"bindings has {len(raw)} entries for {total} faces")
            binding_values = [self._coerce_binding(value, tags) for value in raw]
        colors = [color] * total if face_colors is None else [str(value) for value in face_colors]
        if len(colors) != total:
            raise ValueError(f"face_colors has {len(colors)} entries for {total} faces")
        groups: dict[str, list[int]] = {}
        for index, face_color in enumerate(colors):
            groups.setdefault(face_color, []).append(index)
        for face_color, indices in groups.items():
            self._legacy_mesh_batch(
                vertex_values, [face_values[index] for index in indices],
                color=face_color, outline=outline, width=width, layer=layer,
                cull_backface=cull_backface, opacity=opacity, stipple=stipple,
                tags=tags, lit=lit, two_sided_shell=two_sided_shell,
                bindings=[binding_values[index] for index in indices],
                back_color=back_color,
            )

    def add_shape(
        self,
        mesh: Mesh,
        position: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        up: Optional[Point3D] = None,
        **material: Any,
    ) -> None:
        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be Mesh")
        placed = mesh.placed(origin=position, axis=axis, up_hint=up)
        self.add_mesh(placed.points(), placed.faces, **material)

    def add_box(self, size_x: float = 1.0, size_y: Optional[float] = None,
                size_z: Optional[float] = None, center: Optional[Point3D] = None,
                axis: Optional[Point3D] = None, **material: Any) -> None:
        self.add_shape(shapes_module.box(size_x, size_y, size_z), center, axis, **material)

    def add_box_from_bounds(self, minimum: Point3D, maximum: Point3D,
                            **material: Any) -> None:
        self.add_shape(shapes_module.box_from_bounds(minimum, maximum), **material)

    def add_sphere(self, radius: float = 1.0, center: Optional[Point3D] = None,
                   segments: int = 24, rings: int = 16, **material: Any) -> None:
        self.add_shape(shapes_module.sphere(radius, segments, rings), center, **material)

    def add_cone(self, radius: float = 1.0, height: float = 1.0,
                 center: Optional[Point3D] = None, axis: Optional[Point3D] = None,
                 segments: int = 32, capped: bool = True, **material: Any) -> None:
        self.add_shape(shapes_module.cone(radius, height, segments, capped), center, axis, **material)

    def add_frustum(self, radius_bottom: float = 1.0, radius_top: float = 0.5,
                    height: float = 1.0, center: Optional[Point3D] = None,
                    axis: Optional[Point3D] = None, segments: int = 32,
                    height_segments: int = 1, capped: bool = True,
                    **material: Any) -> None:
        self.add_shape(shapes_module.frustum(radius_bottom, radius_top, height, segments,
                                             height_segments, capped), center, axis, **material)

    def add_tube(self, outer_radius: float = 1.0, inner_radius: float = 0.7,
                 height: float = 1.0, center: Optional[Point3D] = None,
                 axis: Optional[Point3D] = None, segments: int = 32,
                 height_segments: int = 1, capped: bool = True,
                 **material: Any) -> None:
        self.add_shape(shapes_module.tube(outer_radius, inner_radius, height, segments,
                                          height_segments, capped), center, axis, **material)

    def add_torus(self, major_radius: float = 1.0, minor_radius: float = 0.25,
                  center: Optional[Point3D] = None, axis: Optional[Point3D] = None,
                  segments: int = 36, rings: int = 18, **material: Any) -> None:
        self.add_shape(shapes_module.torus(major_radius, minor_radius, segments, rings),
                       center, axis, **material)

    def add_pyramid(self, base_radius: float = 1.0, height: float = 1.0,
                    center: Optional[Point3D] = None, axis: Optional[Point3D] = None,
                    sides: int = 4, capped: bool = True, **material: Any) -> None:
        self.add_shape(shapes_module.pyramid(base_radius, height, sides, capped),
                       center, axis, **material)

    def add_wedge(self, size_x: float = 1.0, size_y: float = 1.0,
                  size_z: float = 1.0, center: Optional[Point3D] = None,
                  axis: Optional[Point3D] = None, **material: Any) -> None:
        self.add_shape(shapes_module.wedge(size_x, size_y, size_z), center, axis, **material)

    def add_prism(self, profile: Sequence[tuple[float, float]], height: float = 1.0,
                  center: Optional[Point3D] = None, axis: Optional[Point3D] = None,
                  capped: bool = True, **material: Any) -> None:
        self.add_shape(shapes_module.prism(profile, height, capped), center, axis, **material)

    def add_extrusion(self, profile: Sequence[tuple[float, float]],
                      path: Sequence[Point3D], capped: bool = True,
                      up: Optional[Point3D] = None, **material: Any) -> None:
        self.add_shape(shapes_module.extrusion(profile, path, capped, up), **material)

    def add_disk(self, outer_radius: float = 1.0, inner_radius: float = 0.0,
                 center: Optional[Point3D] = None, axis: Optional[Point3D] = None,
                 segments: int = 32, **material: Any) -> None:
        material.setdefault("cull_backface", False)
        self.add_shape(shapes_module.disk(outer_radius, inner_radius, segments), center, axis, **material)

    def add_plane(self, size_x: float = 1.0, size_y: float = 1.0,
                  center: Optional[Point3D] = None, axis: Optional[Point3D] = None,
                  nx: int = 1, ny: int = 1, **material: Any) -> None:
        material.setdefault("cull_backface", False)
        self.add_shape(shapes_module.plane(size_x, size_y, nx, ny), center, axis, **material)

    def add_arrow(self, start: Point3D, end: Point3D,
                  shaft_radius: Optional[float] = None,
                  head_radius: Optional[float] = None,
                  head_length: Optional[float] = None, segments: int = 16,
                  **material: Any) -> None:
        material.setdefault("color", "#b45309")
        self.add_shape(shapes_module.arrow(start, end, shaft_radius, head_radius,
                                           head_length, segments), **material)

    def add_beam(self, start: Point3D, end: Point3D, kind: str = "T",
                 web_height: float = 0.2, web_thickness: float = 0.01,
                 flange_width: float = 0.1, flange_thickness: float = 0.015,
                 up: Optional[Point3D] = None, capped: bool = True,
                 **material: Any) -> None:
        self.add_shape(shapes_module.beam(start, end, kind, web_height, web_thickness,
                                          flange_width, flange_thickness, up, capped), **material)

    def add_grid(self, size_x: float = 10.0, size_y: float = 10.0,
                 step: float = 1.0, z: float = 0.0,
                 center: Optional[Point3D] = None, color: str = "#c9d2dc",
                 width: int = 1, layer: int = 2) -> None:
        segments = shapes_module.grid_lines(size_x, size_y, step, z, center)
        positions: list[tuple[float, float, float]] = []
        lines: list[tuple[int, int]] = []
        for start, end in segments:
            index = len(positions)
            positions.extend((start.to_tuple(), end.to_tuple()))
            lines.append((index, index + 1))
        if positions:
            self.add_mesh_arrays(
                MeshArrays(np.asarray(positions, np.float64), np.empty((0, 3), np.uint32),
                           lines=np.asarray(lines, np.uint32)),
                color=color, line_color=color, line_width=width, layer=layer,
            )

    def set_thickness_legend(
        self,
        values: Sequence[float],
        unit: str = "mm",
        title: str = "Plate thickness",
        width: int = 170,
        value_range: Optional[tuple[float, float]] = None,
        colors: Optional[Sequence[str]] = None,
    ) -> None:
        clean = sorted({float(value) for value in values if math.isfinite(float(value))})
        if not clean and value_range is None:
            self.clear_thickness_legend()
            return
        minimum, maximum = (
            (clean[0], clean[-1]) if value_range is None
            else (float(value_range[0]), float(value_range[1]))
        )
        if maximum < minimum:
            minimum, maximum = maximum, minimum
        color_values: list[str] = []
        if colors is not None and len(colors) == len(values):
            lookup = {float(value): str(color) for value, color in zip(values, colors)}
            color_values = [lookup[value] for value in clean]
        self._thickness_legend = {
            "values": clean, "minimum": minimum, "maximum": maximum,
            "unit": str(unit), "title": str(title), "width": max(130, int(width)),
            "colors": color_values,
        }
        self.redraw()

    def clear_thickness_legend(self) -> None:
        self._thickness_legend = None
        self.redraw()

    def thickness_color(
        self, thickness: float,
        value_range: Optional[tuple[float, float]] = None,
    ) -> str:
        if value_range is not None:
            minimum, maximum = value_range
        elif self._thickness_legend is not None:
            minimum = float(self._thickness_legend["minimum"])
            maximum = float(self._thickness_legend["maximum"])
        else:
            minimum = maximum = float(thickness)
        return _interpolate_thickness_color(float(thickness), float(minimum), float(maximum))

    def add_cylinder(
        self,
        radius: float,
        height: float,
        radius_top: Optional[float] = None,
        center: Optional[Point3D] = None,
        color: str = "lightgray",
        back_color: str = "",
        outline: str = "black",
        segments: int = 32,
        height_segments: int = 24,
        capped: bool = True,
        opacity: float = 1.0,
        show_backfaces: Optional[bool] = None,
        plate_thickness: object = None,
        thickness_range: Optional[tuple[float, float]] = None,
        thickness_unit: str = "mm",
        thickness_legend_title: str = "Plate thickness",
        show_thickness_legend: bool = True,
    ) -> None:
        mesh = shapes_module.frustum(
            radius, radius if radius_top is None else radius_top, height,
            segments, height_segments, capped,
        )
        face_colors = None
        values = _flatten_numeric_values(plate_thickness)
        if values:
            minimum, maximum = thickness_range or (min(values), max(values))
            shell_faces = max(1, int(segments) * int(height_segments))
            expanded = (
                values if len(values) == shell_faces
                else [values[min(len(values) - 1, index * len(values) // shell_faces)]
                      for index in range(shell_faces)]
            )
            face_colors = [
                _interpolate_thickness_color(value, minimum, maximum)
                for value in expanded
            ]
            face_colors.extend([color] * max(0, len(mesh.faces) - len(face_colors)))
        effective_backfaces = (
            bool(back_color) or float(opacity) < 0.90
            if show_backfaces is None else bool(show_backfaces)
        )
        self.add_shape(
            mesh, center, color=color, back_color=back_color, outline=outline,
            opacity=opacity,
            cull_backface=not effective_backfaces,
            two_sided_shell=effective_backfaces, face_colors=face_colors,
        )
        if show_thickness_legend and (values or thickness_range is not None):
            legend_values = values or [thickness_range[0], thickness_range[1]]  # type: ignore[index]
            self.set_thickness_legend(
                legend_values, unit=thickness_unit, title=thickness_legend_title,
                value_range=thickness_range,
            )

    def add_longitudinal_stiffener(
        self, radius: float, height: float, angle: float,
        radius_top: Optional[float] = None, web_height: float = 0.1,
        web_thickness: float = 0.01, flange_width: float = 0.05,
        flange_thickness: float = 0.01, color: str = "silver",
        outline: str = "black", segments: int = 4, height_segments: int = 16,
        inside: bool = False, z_offset: float = 0.0,
    ) -> None:
        lower_radius = float(radius) + (-0.5 if inside else 0.5) * web_height
        upper_base = radius if radius_top is None else radius_top
        upper_radius = float(upper_base) + (-0.5 if inside else 0.5) * web_height
        direction = Point3D(math.cos(angle), math.sin(angle), 0.0)
        start = direction * lower_radius + Point3D(0.0, 0.0, z_offset - 0.5 * height)
        end = direction * upper_radius + Point3D(0.0, 0.0, z_offset + 0.5 * height)
        up = direction * (-1.0 if inside else 1.0)
        self.add_beam(
            start, end, "T", web_height, web_thickness, flange_width,
            flange_thickness, up=up, color=color, outline=outline,
        )

    def add_ring_stiffener(
        self, radius: float, z_position: float, web_height: float = 0.1,
        web_thickness: float = 0.01, flange_width: float = 0.05,
        flange_thickness: float = 0.01, color: str = "dimgray",
        outline: str = "black", segments: int = 32, inside: bool = False,
    ) -> None:
        total = max(3, int(segments))
        web_height = max(0.0, float(web_height))
        web_thickness = max(0.0, float(web_thickness))
        flange_width = max(0.0, float(flange_width))
        flange_thickness = max(0.0, float(flange_thickness))
        radial_direction = -1.0 if inside else 1.0
        attachment_radius = max(1.0e-12, float(radius))
        tip_radius = max(
            1.0e-12, float(radius) + radial_direction * web_height
        )
        z_lower = float(z_position) - 0.5 * web_thickness
        z_upper = float(z_position) + 0.5 * web_thickness
        web_value = web_thickness * 1000.0 if web_thickness < 1.0 else web_thickness
        flange_value = (
            flange_thickness * 1000.0
            if flange_thickness < 1.0 else flange_thickness
        )
        web_color = self.thickness_color(web_value) if self._thickness_legend else color
        flange_color = (
            self.thickness_color(flange_value) if self._thickness_legend else color
        )
        angles = tuple(2.0 * math.pi * index / total for index in range(total))

        def ring(radial: float, axial: float) -> tuple[Point3D, ...]:
            return tuple(
                Point3D(radial * math.cos(angle), radial * math.sin(angle), axial)
                for angle in angles
            )

        attachment_lower = ring(attachment_radius, z_lower)
        attachment_upper = ring(attachment_radius, z_upper)
        tip_lower = ring(tip_radius, z_lower)
        tip_upper = ring(tip_radius, z_upper)
        web_faces: list[tuple[Point3D, ...]] = []
        for index in range(total):
            following = (index + 1) % total
            web_faces.extend((
                (
                    attachment_lower[index], tip_lower[index],
                    tip_lower[following], attachment_lower[following],
                ),
                (
                    attachment_upper[following], tip_upper[following],
                    tip_upper[index], attachment_upper[index],
                ),
                (
                    tip_lower[index], tip_upper[index],
                    tip_upper[following], tip_lower[following],
                ),
            ))
        self.add_faces(
            web_faces, colors=web_color, outline=outline, width=1,
            layer=20, cull_backface=False,
        )

        if flange_width > 0.0:
            flange_radius = max(
                1.0e-12,
                tip_radius + radial_direction * 0.5 * flange_thickness,
            )
            flange_lower = ring(flange_radius, float(z_position) - 0.5 * flange_width)
            flange_upper = ring(flange_radius, float(z_position) + 0.5 * flange_width)
            flange_faces = tuple(
                (
                    flange_lower[index], flange_lower[(index + 1) % total],
                    flange_upper[(index + 1) % total], flange_upper[index],
                )
                for index in range(total)
            )
            self.add_faces(
                flange_faces, colors=flange_color, outline=outline, width=1,
                layer=21, cull_backface=False,
            )

    def add_rectangular_plate(
        self, x_start: float, x_end: float, y_start: float, y_end: float,
        z: float = 0.0, color: str = "gray", outline: str = "black",
        stipple: str = "", layer: int = 5, back_color: str = "",
        nx: int = 24, ny: int = 24, opacity: Optional[float] = None,
    ) -> None:
        self.add_plane(
            abs(x_end - x_start), abs(y_end - y_start),
            center=Point3D(0.5 * (x_start + x_end), 0.5 * (y_start + y_end), z),
            nx=nx, ny=ny, color=color, outline=outline, stipple=stipple,
            layer=layer, back_color=back_color,
            opacity=1.0 if opacity is None else opacity,
        )

    def add_flat_stiffener(
        self, x_start: float, x_end: float, y: float, z_base: float,
        hw: float, b: float, color: str = "gray", outline: str = "black",
        stipple: str = "", layer_web: int = 12, layer_flange: int = 13,
        nx: int = 24, opacity: Optional[float] = None,
    ) -> None:
        alpha = 1.0 if opacity is None else opacity
        self.add_polygon(
            (Point3D(x_start, y, z_base), Point3D(x_end, y, z_base),
             Point3D(x_end, y, z_base + hw), Point3D(x_start, y, z_base + hw)),
            color, outline, layer=layer_web, stipple=stipple, opacity=alpha,
        )
        if b > 0:
            self.add_polygon(
                (Point3D(x_start, y - b / 2, z_base + hw),
                 Point3D(x_end, y - b / 2, z_base + hw),
                 Point3D(x_end, y + b / 2, z_base + hw),
                 Point3D(x_start, y + b / 2, z_base + hw)),
                color, outline, layer=layer_flange, stipple=stipple, opacity=alpha,
            )

    def add_flat_girder(
        self, x: float, y_start: float, y_end: float, z_base: float,
        ghw: float, gb: float, color: str = "gray", outline: str = "black",
        stipple: str = "", layer_web: int = 14, layer_flange: int = 15,
        ny: int = 24, opacity: Optional[float] = None,
    ) -> None:
        alpha = 1.0 if opacity is None else opacity
        self.add_polygon(
            (Point3D(x, y_start, z_base), Point3D(x, y_end, z_base),
             Point3D(x, y_end, z_base + ghw), Point3D(x, y_start, z_base + ghw)),
            color, outline, layer=layer_web, stipple=stipple, opacity=alpha,
        )
        if gb > 0:
            self.add_polygon(
                (Point3D(x - gb / 2, y_start, z_base + ghw),
                 Point3D(x - gb / 2, y_end, z_base + ghw),
                 Point3D(x + gb / 2, y_end, z_base + ghw),
                 Point3D(x + gb / 2, y_start, z_base + ghw)),
                color, outline, layer=layer_flange, stipple=stipple, opacity=alpha,
            )

    def set_section_plane(
        self,
        normal: object = (1.0, 0.0, 0.0),
        offset: float = 0.0,
        *,
        enabled: bool = True,
    ) -> None:
        self._section_plane = SectionPlane(as_point(normal), offset, enabled)
        self._renderer.pick_dirty = True
        self._selection_index = None
        self.redraw()

    def clear_section_plane(self) -> None:
        if self._section_plane is not None:
            self._section_plane = None
            self._renderer.pick_dirty = True
            self._selection_index = None
            self.redraw()

    def set_camera_position(self, position: Point3D) -> None:
        self.camera.set_position(as_point(position))
        self._renderer.pick_dirty = True
        self._selection_index = None
        self.redraw()

    def set_camera_target(self, target: Point3D) -> None:
        self.camera.set_target(as_point(target))
        self._renderer.pick_dirty = True
        self._selection_index = None
        self.redraw()

    def set_view(self, azimuth_degrees: float, elevation_degrees: float) -> None:
        self.camera.set_orbit(
            azimuth=math.radians(float(azimuth_degrees)),
            elevation=math.radians(float(elevation_degrees)),
        )
        self._renderer.pick_dirty = True
        self._selection_index = None
        self.redraw()

    def set_iso_view(self) -> None:
        self.set_view(-45.0, 25.0)

    def set_top_view(self) -> None:
        self.set_view(-90.0, 89.0)

    def set_side_view(self) -> None:
        self.set_view(0.0, 0.0)

    def set_front_view(self) -> None:
        self.set_view(-90.0, 0.0)

    def reset_camera(self) -> None:
        self.camera = Camera3D()
        self.fit_to_scene(redraw=False)
        self._renderer.pick_dirty = True
        self._selection_index = None
        self.redraw()

    def _scene_bounds(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        values: list[np.ndarray] = []
        for entry in self._display_entries().values():
            handle: MeshHandle = entry["handle"]
            if handle.removed or not handle.visible:
                continue
            for _chunk_id, mesh in [(None, handle.mesh), *handle.chunks]:
                positions = mesh.positions
                if mesh.displacements is not None and handle.deformation_scale:
                    positions = positions + handle.deformation_scale * mesh.displacements
                transform = handle.transform
                values.append(positions @ transform[:3, :3].T + transform[:3, 3])
        if not values or not any(len(value) for value in values):
            return None
        positions = np.concatenate([value for value in values if len(value)])
        return positions.min(axis=0), positions.max(axis=0)

    def fit_to_scene(self, padding: float = 1.25, redraw: bool = True) -> None:
        bounds = self._scene_bounds()
        if bounds is None:
            if redraw:
                self.redraw()
            return
        low, high = bounds
        target = 0.5 * (low + high)
        radius = max(1.0e-6, float(np.linalg.norm(high - low)) * 0.5)
        width, height = self.viewport_size
        aspect = max(1.0e-9, float(width) / max(1.0, float(height)))
        limiting_tangent = math.tan(self.camera.fov / 2) * min(1.0, aspect)
        self.camera.set_target(Point3D(*target))
        self.camera.set_orbit(
            distance=max(
                self.camera.near * 2,
                radius * float(padding) / max(1.0e-9, limiting_tangent),
            )
        )
        self._renderer.pick_dirty = True
        self._selection_index = None
        if redraw:
            self.redraw()

    def set_interactive_detail(self, faces: int) -> None:
        """Accept the software LOD control; retained GPU rendering needs no rebuild."""

        if int(faces) < 1:
            raise ValueError("interactive detail must be positive")

    def set_opaque_cylinder_occluder(
        self, radius: float, height: float, center: Optional[Point3D] = None
    ) -> None:
        mesh = MeshArrays.from_mesh(
            shapes_module.cylinder(radius, height, segments=48, height_segments=1)
            .placed(origin=center)
        )
        handle = self.add_mesh_arrays(
            mesh, color=self.bg, outline="", pickable=False, depth_only=True,
        )
        self._opaque_occluders.append(handle)

    @property
    def animation_frames(self) -> int:
        return len(self._animation_cache)

    @property
    def animation_frame_index(self) -> int:
        return self._animation_frame_index

    @property
    def is_playing_animation(self) -> bool:
        return self._is_playing_animation

    def begin_animation_cache(self) -> None:
        self.stop_animation()
        self._animation_cache.clear()
        self._animation_frame_index = 0

    @staticmethod
    def _copy_chunk_semantics(
        caches: dict[object, dict[str, Any]],
    ) -> dict[object, dict[str, dict[str, np.ndarray]]]:
        """Copy chunk-local semantic inverses for an immutable frame."""

        return {
            chunk_id: {
                name: {
                    key: np.asarray(value).copy()
                    for key, value in values.items()
                }
                for name, values in cache.items()
                if isinstance(values, dict)
            }
            for chunk_id, cache in caches.items()
        }

    def capture_animation_frame(self) -> None:
        meshes: list[dict[str, Any]] = []
        for entry in self._entries.values():
            handle: MeshHandle = entry["handle"]
            if handle.removed:
                continue
            meshes.append({
                "mesh": handle.mesh.owned_copy(),
                "chunks": tuple(
                    (key, mesh.owned_copy(), owners, resolver)
                    for key, mesh, owners, resolver in handle.chunk_records
                ),
                "appearance": dict(entry.get("appearance", {})),
                "transform": np.asarray(handle.transform).copy(),
                "deformation_scale": handle.deformation_scale,
                "selected": np.asarray(handle.selected_elements).copy(),
                "visible": handle.visible,
                "owners": entry.get("owners"),
                "owner_resolver": entry.get("owner_resolver"),
                "layer": int(entry.get("layer", 0)),
                "tags": frozenset(entry.get("tags", ())),
                "item": int(entry.get("item", -1)),
                "semantic_elements": {
                    key: np.asarray(value).copy()
                    for key, value in entry.get("semantic_elements", {}).items()
                },
                "hit_elements": {
                    key: np.asarray(value).copy()
                    for key, value in entry.get("hit_elements", {}).items()
                },
                "semantic_lines": {
                    key: np.asarray(value).copy()
                    for key, value in entry.get("semantic_lines", {}).items()
                },
                "hit_lines": {
                    key: np.asarray(value).copy()
                    for key, value in entry.get("hit_lines", {}).items()
                },
                "semantic_points": {
                    key: np.asarray(value).copy()
                    for key, value in entry.get("semantic_points", {}).items()
                },
                "hit_points": {
                    key: np.asarray(value).copy()
                    for key, value in entry.get("hit_points", {}).items()
                },
                "chunk_semantics": self._copy_chunk_semantics(
                    entry.get("chunk_semantics", {})
                ),
            })
        self._animation_cache.append({
            "meshes": tuple(meshes),
            "world_text": tuple(dict(value) for value in self._world_text),
            "legend": (
                None if self._thickness_legend is None
                else dict(self._thickness_legend)
            ),
            "highlighted_tags": self._highlighted_tags,
            "preselected_key": self._preselected_key,
            "preselection_from_hit": self._preselection_from_hit,
            "highlight_fill": self._highlight_fill,
            "highlight_outline": self._highlight_outline,
        })

    def _restore_live_renderer(self) -> None:
        for handle in self._animation_handles:
            self._renderer.remove_mesh(handle)
        self._animation_handles.clear()
        self._animation_entries.clear()
        self._animation_frame_active = False
        for entry in self._entries.values():
            handle = entry["handle"]
            if handle.removed:
                continue
            self._renderer.add_mesh(handle, **{
                key: value for key, value in entry.get("appearance", {}).items()
                if key in {
                    "color", "line_color", "point_color", "point_outline", "line_width", "point_size",
                    "opacity", "cull_backface", "line_overlay", "point_overlay", "lit", "stipple",
                    "stipple_phase",
                    "pickable", "depth_only", "scalar_range",
                    "layer",
                    "mesh_lines",
                    "face_colors", "back_color", "invalid_color",
                }
            })
        self._selection_index = None
        self._selection_index_key = None

    def _show_animation_frame(self, index: int) -> None:
        for entry in self._entries.values():
            self._renderer.remove_mesh(entry["handle"])
        for handle in self._animation_handles:
            self._renderer.remove_mesh(handle)
        self._animation_handles.clear()
        self._animation_entries.clear()
        self._animation_frame_active = True
        frame = self._animation_cache[index]
        for item in frame["meshes"]:
            handle = MeshHandle(item["mesh"])
            for key, mesh, owners, resolver in item["chunks"]:
                handle.add_chunk(
                    key,
                    mesh,
                    owners=owners,
                    owner_resolver=resolver,
                )
            handle.set_transform(item["transform"])
            handle.set_deformation_scale(item["deformation_scale"])
            handle.set_selected_elements(item["selected"])
            handle.set_visible(item["visible"])
            self._renderer.add_mesh(handle, **{
                key: value for key, value in item["appearance"].items()
                if key in {
                    "color", "line_color", "point_color", "point_outline", "line_width", "point_size",
                    "opacity", "cull_backface", "line_overlay", "point_overlay", "lit", "stipple",
                    "stipple_phase",
                    "pickable", "depth_only", "scalar_range",
                    "layer",
                    "mesh_lines",
                    "face_colors", "back_color", "invalid_color",
                }
            })
            self._animation_handles.append(handle)
            self._animation_entries[id(handle)] = {
                "handle": handle,
                "owners": item["owners"],
                "owner_resolver": item["owner_resolver"],
                "appearance": item["appearance"],
                "layer": item["layer"],
                "tags": item["tags"],
                "item": item["item"],
                "face_colors": item["appearance"].get("face_colors"),
                "semantic_elements": {
                    key: np.asarray(value).copy()
                    for key, value in item["semantic_elements"].items()
                },
                "hit_elements": {
                    key: np.asarray(value).copy()
                    for key, value in item["hit_elements"].items()
                },
                "semantic_lines": {
                    key: np.asarray(value).copy()
                    for key, value in item["semantic_lines"].items()
                },
                "hit_lines": {
                    key: np.asarray(value).copy()
                    for key, value in item["hit_lines"].items()
                },
                "semantic_points": {
                    key: np.asarray(value).copy()
                    for key, value in item["semantic_points"].items()
                },
                "hit_points": {
                    key: np.asarray(value).copy()
                    for key, value in item["hit_points"].items()
                },
                "chunk_semantics": self._copy_chunk_semantics(
                    item.get("chunk_semantics", {})
                ),
            }
        self._world_text = [dict(value) for value in frame["world_text"]]
        self._thickness_legend = (
            None if frame["legend"] is None else dict(frame["legend"])
        )
        self._highlighted_tags = frame["highlighted_tags"]
        self._preselected_key = frame["preselected_key"]
        self._preselection_from_hit = frame["preselection_from_hit"]
        self._highlight_fill = frame["highlight_fill"]
        self._highlight_outline = frame["highlight_outline"]
        self._selection_index = None
        self._selection_index_key = None
        self._apply_highlight_masks()

    def play_animation(self, fps: int = 30, fast: Optional[bool] = None) -> None:
        del fast  # API-compatible; GPU renders the full retained frame.
        if not self._animation_cache:
            return
        self.stop_animation()
        self._animation_live_hud = {
            "world_text": [dict(value) for value in self._world_text],
            "legend": (
                None if self._thickness_legend is None
                else dict(self._thickness_legend)
            ),
            "highlighted_tags": self._highlighted_tags,
            "preselected_key": self._preselected_key,
            "preselection_from_hit": self._preselection_from_hit,
            "highlight_fill": self._highlight_fill,
            "highlight_outline": self._highlight_outline,
        }
        self._is_playing_animation = True
        delay = max(1, int(1000 / max(1, int(fps))))

        def tick() -> None:
            if not self._is_playing_animation:
                return
            self._show_animation_frame(self._animation_frame_index)
            self._animation_frame_index = (
                self._animation_frame_index + 1
            ) % len(self._animation_cache)
            self._animation_after_id = self.after(delay, tick)

        tick()

    def stop_animation(self) -> None:
        was_playing = self._is_playing_animation or bool(self._animation_handles)
        self._is_playing_animation = False
        if self._animation_after_id is not None:
            try:
                self.after_cancel(self._animation_after_id)
            except tk.TclError:
                pass
            self._animation_after_id = None
        if was_playing:
            self._restore_live_renderer()
            if self._animation_live_hud is not None:
                live = self._animation_live_hud
                self._world_text = [dict(value) for value in live["world_text"]]
                self._thickness_legend = (
                    None if live["legend"] is None else dict(live["legend"])
                )
                self._highlighted_tags = live["highlighted_tags"]
                self._preselected_key = live["preselected_key"]
                self._preselection_from_hit = live["preselection_from_hit"]
                self._highlight_fill = live["highlight_fill"]
                self._highlight_outline = live["highlight_outline"]
                self._animation_live_hud = None
            self._apply_highlight_masks()
            self.redraw()

    def _render_hud(self, viewport: tuple[int, int]) -> None:
        """Render labels, legends, selection outlines and gestures in OpenGL."""

        width, height = viewport
        hud = self._hud
        hud.begin(viewport)
        for value in sorted(self._world_text, key=lambda item: item["layer"]):
            plane = self._section_plane
            if plane is not None and plane.enabled and not plane.contains(value["point"]):
                continue
            projected = self.project_point(value["point"])
            if projected is not None:
                text_depth = None
                if not value["draw_overlay"]:
                    near, far = float(self.camera.near), float(self.camera.far)
                    camera_depth = max(near, float(projected[2]))
                    text_depth = (
                        far / (far - near)
                        - far * near / ((far - near) * camera_depth)
                        - 1.0e-5
                    )
                hud.text(
                    projected[:2], value["text"], value["color"],
                    font=value["font"], anchor=value["anchor"], depth=text_depth,
                )

        if self._show_axis_indicator and width >= 95 and height >= 95:
            right, up, _forward = self.camera.basis()
            origin = (58.0, height - 58.0)
            for text_value, axis, color in (
                ("X", Point3D(1, 0, 0), "#9b111e"),
                ("Y", Point3D(0, 1, 0), "#159447"),
                ("Z", Point3D(0, 0, 1), "#0d47a1"),
            ):
                end = (
                    origin[0] + axis.dot(right) * 42,
                    origin[1] - axis.dot(up) * 42,
                )
                hud.line(origin, end, color, width=2.0)
                dx, dy = end[0] - origin[0], end[1] - origin[1]
                length = max(1.0, math.hypot(dx, dy))
                normal = (-dy / length, dx / length)
                back = (end[0] - 8 * dx / length, end[1] - 8 * dy / length)
                hud.line(end, (back[0] + 4 * normal[0], back[1] + 4 * normal[1]), color, width=2)
                hud.line(end, (back[0] - 4 * normal[0], back[1] - 4 * normal[1]), color, width=2)
                hud.text((end[0] + 9 * dx / length, end[1] + 9 * dy / length),
                         text_value, color, font=("Segoe UI", 10, "bold"))

        if self.show_axis_ruler:
            bounds = self._scene_bounds()
            if bounds is not None:
                low, high = bounds
                origin = Point3D(*low)
                axes = (
                    (Point3D(high[0], low[1], low[2]), "X", low[0], high[0]),
                    (Point3D(low[0], high[1], low[2]), "Y", low[1], high[1]),
                    (Point3D(low[0], low[1], high[2]), "Z", low[2], high[2]),
                )
                projected_origin = self.project_point(origin)
                if projected_origin is not None:
                    hud.text(projected_origin[:2],
                             f"({low[0]:g}, {low[1]:g}, {low[2]:g})",
                             "#1f2937", font=("Segoe UI", 8, ""), anchor="s")
                    for endpoint, axis_name, minimum, maximum in axes:
                        projected = self.project_point(endpoint)
                        if projected is None:
                            continue
                        hud.line(projected_origin[:2], projected[:2], "#1f2937", width=1.5)
                        hud.text(projected[:2], f"{axis_name} {maximum:g}", "#1f2937",
                                 font=("Segoe UI", 8, "bold"), anchor="n")
                        dx = projected[0] - projected_origin[0]
                        dy = projected[1] - projected_origin[1]
                        length = max(1.0, math.hypot(dx, dy))
                        normal = (-dy / length, dx / length)
                        for tick in range(1, 5):
                            fraction = tick / 5.0
                            world_tick = origin + (endpoint - origin) * fraction
                            screen_tick = self.project_point(world_tick)
                            if screen_tick is None:
                                continue
                            point = screen_tick[:2]
                            hud.line(
                                (point[0] - 3 * normal[0], point[1] - 3 * normal[1]),
                                (point[0] + 3 * normal[0], point[1] + 3 * normal[1]),
                                "#1f2937",
                            )
                            hud.text(
                                (point[0] + 5 * normal[0], point[1] + 5 * normal[1]),
                                f"{minimum + fraction * (maximum - minimum):g}",
                                "#374151", font=("Segoe UI", 7, ""), anchor="w",
                            )

        legend = self._thickness_legend
        if legend is not None:
            legend_width = int(legend["width"])
            x0, y0 = width - legend_width - 8, 8
            values = list(legend["values"])
            continuous = len(values) > 10 or not values
            box_height = 132 if continuous else 32 + 20 * len(values)
            hud.quad((x0, y0, width - 8, y0 + box_height), "#f8fafc", alpha=0.96)
            hud.rectangle((x0, y0, width - 8, y0 + box_height), "#64748b")
            hud.text((x0 + 7, y0 + 6), legend["title"], "#0f172a",
                     font=("Segoe UI", 9, "bold"), anchor="nw")
            if continuous:
                minimum, maximum = legend["minimum"], legend["maximum"]
                for index in range(48):
                    fraction = index / 47.0
                    color = _interpolate_thickness_color(
                        maximum - fraction * (maximum - minimum), minimum, maximum
                    )
                    top = y0 + 26 + 1.7 * index
                    hud.quad((x0 + 8, top, x0 + 28, top + 2.2), color)
                hud.text((x0 + 35, y0 + 26), f"{maximum:g} {legend['unit']}",
                         "#0f172a", anchor="nw")
                hud.text((x0 + 35, y0 + 108), f"{minimum:g} {legend['unit']}",
                         "#0f172a", anchor="sw")
            else:
                colors = list(legend["colors"])
                for row, value in enumerate(reversed(values)):
                    source_index = len(values) - row - 1
                    color = colors[source_index] if colors else _interpolate_thickness_color(
                        value, legend["minimum"], legend["maximum"]
                    )
                    top = y0 + 26 + 20 * row
                    hud.quad((x0 + 8, top, x0 + 25, top + 13), color)
                    hud.text((x0 + 31, top + 6), f"{value:g} {legend['unit']}",
                             "#0f172a", anchor="w")

        active = set(self._highlighted_tags)
        if self._preselected_key:
            active.add(self._preselected_key)
        # Retained FE scenes use compact GPU masks for camera-time selection.
        # Screen-space custom outlines are bounded to small legacy scenes so
        # orbiting a million-triangle mesh never projects elements in Python.
        if (
            active
            and self._display_primitive_count(_CPU_POINT_STACK_LIMIT)
            <= _CPU_POINT_STACK_LIMIT
        ):
            item_tags = {
                int(entry.get("item", -1)): set(entry.get("tags", ()))
                for entry in self._display_entries().values()
            }
            index = self._projected_selection_index()
            for primitive in index.primitives:
                keys = set(item_tags.get(primitive.item, ()))
                if primitive.binding is not None:
                    keys.update(owner.key for owner in primitive.binding.owners)
                matched = keys & active
                if not matched:
                    continue
                preselected = self._preselected_key in matched and not (
                    matched & self._highlighted_tags
                )
                color = "#b77900" if preselected else self._highlight_outline
                if primitive.shape == "polygon":
                    hud.polyline(primitive.points, color, width=2.5, closed=True)
                elif primitive.shape == "segment":
                    hud.line(primitive.points[0], primitive.points[1], color, width=3.0)
                else:
                    hud.circle(primitive.points[0], max(5.0, primitive.radius + 2), color, width=2.0)

        if self._selection_dragging and self._selection_press and self._selection_current:
            start, current = self._selection_press, self._selection_current
            if self._selection_config.tool is SelectionTool.LASSO:
                hud.polyline(tuple((*self._selection_points, current)), "#2563eb", width=2.0)
            else:
                crossing = bool(self._selection_config.directional and current[0] < start[0])
                color = "#16a34a" if crossing else "#2563eb"
                fill = "#dcfce7" if crossing else "#dbeafe"
                hud.rectangle((*start, *current), color, width=2.0,
                              fill=fill, fill_alpha=0.22)

        hud.render()

    def capture_image(self):
        """Return the current framebuffer as a top-left-oriented Pillow image."""

        try:
            from PIL import Image
        except ImportError as error:  # pragma: no cover - malformed optional install
            raise RuntimeError("capture_image requires Pillow from ANY3dView[gpu]") from error
        self._host.make_current()
        width, height = self.viewport_size
        self._renderer.render(
            self.camera, (width, height), section_plane=self._section_plane,
            clear_color=tuple((parse_color(self.bg) or (255, 255, 255))[index] / 255.0
                              for index in range(3)) + (1.0,),
            light=self._light, shading_enabled=self._shading_enabled,
            occlude_lines=self._occlude_lines, show_mesh_lines=self.show_mesh_lines,
            selection_color=self._highlight_fill,
            preselection_color="#ffd166",
        )
        self._render_hud((width, height))
        payload = self._renderer.ctx.screen.read(components=4, alignment=1)
        image = Image.frombytes("RGBA", (width, height), payload).transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        )
        return image

    def redraw(self) -> None:
        if not self._closed and not self._suspend_redraw:
            self._host.request_redraw()

    def _draw_now(self) -> None:
        if self._closed:
            return
        self._host.make_current()
        size = self._host.framebuffer_size()
        color = parse_color(self.bg) or (255, 255, 255)
        clear = tuple(channel / 255.0 for channel in color) + (1.0,)
        self._renderer.render(
            self.camera,
            size,
            section_plane=self._section_plane,
            clear_color=clear,
            light=self._light,
            shading_enabled=self._shading_enabled,
            occlude_lines=self._occlude_lines,
            show_mesh_lines=self.show_mesh_lines,
            selection_color=self._highlight_fill,
            preselection_color="#ffd166",
        )
        self._render_hud(size)
        self._host.swap_buffers()
        self.width, self.height = size

    def clear(self, keep_canvas: bool = False) -> None:
        previous_suspend = self._suspend_redraw
        self._suspend_redraw = previous_suspend or bool(keep_canvas)
        try:
            self.stop_animation()
            for entry in list(self._entries.values()):
                entry["handle"].remove()
            self._entries.clear()
            self._world_text.clear()
            self._thickness_legend = None
            self._opaque_occluders.clear()
            self._selection_index = None
            self._preselected_key = None
            self._preselection_from_hit = False
            self._hover_key = None
            self._cycle_candidates = ()
            self._cycle_anchor = None
            self._cycle_index = -1
            self._selection_press = None
            self._selection_current = None
            self._selection_points = []
            self._selection_dragging = False
            self._selection_press_hit_keys = frozenset()
            self._selection_operation = SelectionOperation.REPLACE
            self._selection_modifiers = (False, False, False)
        finally:
            self._suspend_redraw = previous_suspend
        if keep_canvas and hasattr(self._host, "cancel_redraw"):
            self._host.cancel_redraw()
        if not keep_canvas:
            self.redraw()

    def destroy(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        self.stop_animation()
        if self._update_poll_id is not None:
            try:
                self.after_cancel(self._update_poll_id)
            except tk.TclError:
                pass
            self._update_poll_id = None
        self._update_scheduler.close()
        toplevel = getattr(self, "_interaction_toplevel", None)
        if toplevel is not None:
            for sequence, identifier in (
                ("<ButtonRelease-1>", getattr(self, "_toplevel_release_binding", None)),
                ("<Escape>", getattr(self, "_toplevel_escape_binding", None)),
            ):
                if identifier:
                    try:
                        toplevel.unbind(sequence, identifier)
                    except tk.TclError:
                        pass
        for entry in list(self._entries.values()):
            entry["handle"].remove()
        self._entries.clear()
        renderer = getattr(self, "_renderer", None)
        if renderer is not None:
            try:
                self._host.make_current()
                hud = getattr(self, "_hud", None)
                if hud is not None:
                    hud.release()
                renderer.release()
                # The ModernGL wrapper must be released while TkGL's native
                # WGL context is still current.  Deferring this to Python
                # finalization after the Tk surface has gone away can enter
                # the display driver with a stale context and terminate the
                # process instead of raising a Python exception.
                renderer.ctx.release()
            except (RuntimeError, tk.TclError, moderngl.Error):
                # Parent-driven Tcl/GL teardown may have already invalidated
                # the drawable. Python ownership is still closed below.
                pass
            finally:
                self._hud = None
                self._renderer = None
        host = getattr(self, "_host", None)
        if host is not None:
            try:
                host.close()
            except (RuntimeError, tk.TclError):
                pass
        try:
            super().destroy()
        except tk.TclError:
            pass
