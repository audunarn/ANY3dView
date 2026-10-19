"""The 3D viewport: draws a Scene and turns clicks back into entities.

This is the only module that loads a concrete 3D viewer.  It does so lazily so
the rest of ANYfem stays usable without the GUI extra installed.
"""

from __future__ import annotations

import io
from pathlib import Path
import tkinter as tk
from typing import Any, Callable, List, Mapping, Optional

import numpy as np

from ..geometry.construction import ConstructionTask
from ..geometry.snapping import GeometrySnapData, SnapEngine, SnapResult
from ..model.coordinates import CoordinateSystem
from ..model.workplanes import Workplane, WorkplaneFrame
from ..selection import (
    TAG_PREFIX,
    Selection,
    entity_tag,
    owner_to_ref,
    parse_entity_tag,
)
from ..presentation.viewport import SceneViewport
from .scene import PointMarker, Scene
from .visualization import VisualizationStyle

__all__ = ["Viewport", "require_canvas"]


_BACKEND_ALIASES = {
    "auto": "auto",
    "automatic": "auto",
    "gpu": "gpu",
    "moderngl": "gpu",
    "software": "software",
    "tk": "software",
    "tkinter": "software",
}


def _backend_name(value: str) -> str:
    try:
        return _BACKEND_ALIASES[str(value).strip().casefold()]
    except KeyError as error:
        raise ValueError(
            "viewer backend must be 'auto', 'gpu', or 'software'"
        ) from error


def require_canvas():
    """Import the shared point type and lazy backend factory."""

    try:
        from any3dview import Point3D, create_viewer
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "the ANYfem viewport needs ANY3dView and ANYtk3D. Install them with:\n"
            "    python -m pip install ANYfem[gui]"
        ) from error
    return Point3D, create_viewer


def _commercial_selection_api() -> Optional[dict[str, Any]]:
    """Return shared semantic-selection types without importing a renderer."""

    names = (
        "PickBinding",
        "PickOwner",
        "SelectionConfig",
        "SelectionDepth",
        "SelectionFilter",
        "SelectionGesture",
        "SelectionOperation",
        "SelectionTool",
    )
    try:
        import any3dview

        return {name: getattr(any3dview, name) for name in names}
    except (ImportError, AttributeError):  # pragma: no cover - install dependent
        # This fallback keeps source compatibility with the pre-0.5 software
        # backend while the coordinated release graph is being upgraded.
        try:
            import anytk3d

            return {name: getattr(anytk3d, name) for name in names}
        except (ImportError, AttributeError):
            return None


class Viewport(SceneViewport):
    """A 3D canvas that knows how to draw a Scene and report picks."""

    def __init__(
        self,
        master,
        selection: Optional[Selection] = None,
        width: int = 900,
        height: int = 640,
        background: str = "#fbfbfd",
        commercial_interaction: bool = True,
        backend: str = "auto",
        viewer_host=None,
    ) -> None:
        self._master = master
        self._width = int(width)
        self._height = int(height)
        self._background = str(background)
        self._commercial_interaction_requested = bool(commercial_interaction)
        self._requested_backend = _backend_name(backend)
        self._viewer_host = viewer_host
        self._point3d, self._viewer_factory = require_canvas()
        # Not ``selection or Selection()``: Selection defines __len__, so an
        # empty one is falsy and that would quietly make a second, unshared
        # selection object.
        self.selection = Selection() if selection is None else selection
        self._scene: Optional[Scene] = None
        self._interaction_scene: Optional[Scene] = None
        self._interaction_scene_active = False
        self._interaction_restore_after_id: object | None = None
        self._on_pick: Optional[Callable[[Optional[object]], None]] = None
        self._on_hover: Optional[Callable[[Optional[object]], None]] = None
        self._on_frame_selection: Optional[Callable[[List[object]], Any]] = None
        self._hovered: Optional[object] = None
        self._hover_key: Optional[str] = None
        self._marker_size = 0.0
        self._visualization = VisualizationStyle(background=background)
        self._selection_api = _commercial_selection_api()
        self._selection_tool = "box"
        self._selection_depth = "visible"
        self._selection_operation = "replace"
        self._canvas_filter_kinds: frozenset[str] = frozenset()
        self._construction_task: ConstructionTask | None = None
        self._construction_workplane: Workplane | None = None
        self._construction_coordinate_systems: Mapping[
            str, CoordinateSystem
        ] | None = None
        self._construction_snap_engine: SnapEngine | None = None
        self._construction_snap_data: (
            GeometrySnapData | Callable[[], GeometrySnapData] | None
        ) = None
        self._on_construction_update: Optional[
            Callable[[ConstructionTask, SnapResult | None], None]
        ] = None
        self._on_construction_apply: Optional[Callable[[], Any]] = None
        self._construction_grid_extent: tuple[float, float, float, float] | None = None
        self._construction_length_formatter: Callable[[float], str] | None = None
        self._layout_manager: str | None = None
        self._layout_options: dict[str, Any] = {}
        self._event_bindings: list[tuple[str, Callable[..., Any], str]] = []
        self.canvas = self._new_canvas(self._requested_backend)
        self._commercial_selection = False
        self._configure_canvas(self.canvas)
        self.bind_event("<Escape>", self._handle_construction_escape, add="+")
        self.bind_event("<Return>", self._handle_construction_enter, add="+")
        self.bind_event("<KP_Enter>", self._handle_construction_enter, add="+")
        # Geometry scenes may contain a high-quality retained preview and a
        # lower-detail sibling for camera motion.  Only pan/orbit use this
        # path; LMB selection remains on the complete semantic scene.
        self.bind_event("<ButtonPress-2>", self._begin_interaction_scene, add="+")
        self.bind_event("<ButtonPress-3>", self._begin_interaction_scene, add="+")
        self.bind_event("<ButtonRelease-2>", self._schedule_full_scene, add="+")
        self.bind_event("<ButtonRelease-3>", self._schedule_full_scene, add="+")
        self.selection.add_listener(self._apply_highlight)


    def bind_event(
        self,
        sequence: str,
        callback: Callable[..., Any],
        *,
        add: str = "+",
    ) -> None:
        """Bind an input callback and carry it across renderer switches."""

        binding = (str(sequence), callback, str(add))
        self._event_bindings.append(binding)
        widget = self.event_widget
        bind = getattr(widget, "bind", None)
        if callable(bind):
            bind(binding[0], binding[1], add=binding[2])

    def _bind_registered_events(self) -> None:
        widget = self.event_widget
        bind = getattr(widget, "bind", None)
        if callable(bind):
            for sequence, callback, add in self._event_bindings:
                bind(sequence, callback, add=add)

    # ------------------------------------------------------------------
    def pack(self, **kwargs):
        self._layout_manager = "pack"
        self._layout_options = dict(kwargs)
        return self.canvas.pack(**kwargs)

    def grid(self, **kwargs):
        self._layout_manager = "grid"
        self._layout_options = dict(kwargs)
        return self.canvas.grid(**kwargs)


    def switch_backend(self, backend: str) -> str:
        """Transactionally rebuild the current scene on another renderer.

        The live canvas is never cleared or destroyed until its replacement has
        accepted the complete scene, callbacks and view state.
        """

        requested = _backend_name(backend)
        if requested == self._requested_backend:
            return self.active_backend
        self._cancel_interaction_restore()
        self._interaction_scene_active = False
        old = self.canvas
        exporter = getattr(old, "export_view_state", None)
        state = exporter() if callable(exporter) else None
        section = getattr(old, "section_plane", None)
        old_commercial = self._commercial_selection
        old_filter_kinds = self._canvas_filter_kinds
        candidate = None
        try:
            candidate = self._new_canvas(requested)
            self.canvas = candidate
            self._configure_canvas(candidate)
            self._bind_registered_events()
            background = getattr(candidate, "set_background", None)
            if callable(background):
                background(self._visualization.background)
            candidate.clear(keep_canvas=True)
            if self._scene is not None:
                self._draw(self._scene)
                self._draw_construction_overlay()
            importer = getattr(candidate, "apply_view_state", None)
            if state is not None and callable(importer):
                try:
                    importer(state, redraw=False)
                except TypeError:
                    # Compatibility with pre-0.5 viewers and lightweight
                    # integration probes that do not expose redraw control.
                    importer(state)
            elif section is not None and hasattr(candidate, "set_section_plane"):
                normal = getattr(section.normal, "to_tuple", None)
                candidate.set_section_plane(
                    normal=(
                        tuple(normal())
                        if callable(normal)
                        else tuple(section.normal)
                    ),
                    offset=float(section.offset),
                    enabled=bool(section.enabled),
                )
            self._apply_highlight()
            if self._hover_key is not None and hasattr(candidate, "set_preselection"):
                candidate.set_preselection(self._hover_key)
            candidate.redraw()
            self._map_candidate(candidate, old)
        except Exception:
            self.canvas = old
            self._commercial_selection = old_commercial
            self._canvas_filter_kinds = old_filter_kinds
            if candidate is not None:
                try:
                    candidate.destroy()
                except Exception:
                    pass
            raise
        self._requested_backend = requested
        try:
            old.destroy()
        except (AttributeError, RuntimeError, tk.TclError):
            pass
        return self.active_backend

    def _map_candidate(self, candidate, old) -> None:
        if self._layout_manager == "grid":
            candidate.grid(**self._layout_options)
        elif self._layout_manager == "pack":
            options = dict(self._layout_options)
            try:
                candidate.pack(before=old, **options)
            except (TypeError, tk.TclError):
                candidate.pack(**options)
        updater = getattr(candidate, "update_idletasks", None)
        if callable(updater):
            updater()

    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # workplane projection and click construction
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # capture and optional section-plane capabilities
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------


    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
