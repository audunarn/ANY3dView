"""Tk/ttk widget backed by :class:`ModernGLRenderer`."""

from __future__ import annotations

import math
from copy import deepcopy
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
from ..retained_viewer import RetainedViewer
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


class Any3DView(RetainedViewer, ttk.Frame):
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
        self._wheel_finish_after_id: Optional[str] = None
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


    def _toplevel_release_select(self, event: tk.Event) -> None:
        if self._selection_press is None:
            return
        try:
            event.x = int(event.x_root) - int(self.canvas.winfo_rootx())
            event.y = int(event.y_root) - int(self.canvas.winfo_rooty())
        except (AttributeError, TypeError, ValueError, tk.TclError):
            event.x, event.y = int(event.x), int(event.y)
        self._release_select(event)


    def _wheel(self, event: tk.Event) -> None:
        self._zoom(0.9 if int(getattr(event, "delta", 0)) > 0 else 1.1)


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
        # The default framebuffer may retain its size from context creation.
        payload = self._renderer.ctx.screen.read(
            viewport=(0, 0, width, height), components=4, alignment=1
        )
        image = Image.frombytes("RGBA", (width, height), payload).transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        )
        return image


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


    def destroy(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        self.stop_animation()
        if self._wheel_finish_after_id is not None:
            try:
                self.after_cancel(self._wheel_finish_after_id)
            except tk.TclError:
                pass
            self._wheel_finish_after_id = None
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
