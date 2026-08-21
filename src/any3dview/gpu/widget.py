"""Tk/ttk widget backed by :class:`ModernGLRenderer`."""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional

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
from ..core import Camera3D, Point3D, as_point, parse_color
from ..errors import GPUUnavailableError
from ..ownership import ModelOwner, PackedOwnerTable
from ..retained import MeshHandle
from ..scheduler import ViewerScheduler
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
)
from .host import TkinterGLHost
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
    geometry_changeset=True,
    software_fallback=True,
)


class Any3DView(ttk.Frame):
    """Demand-driven OpenGL 3.3 viewer embedded in a Tk application."""

    def __init__(
        self,
        master: tk.Misc,
        width: int = 800,
        height: int = 600,
        bg: str = "white",
        **frame_options: Any,
    ) -> None:
        super().__init__(master, **frame_options)
        self.camera = Camera3D()
        self.bg = str(bg)
        self._section_plane: Optional[SectionPlane] = None
        self._entries: dict[int, dict[str, Any]] = {}
        self._selection_callback: Optional[Callable[[SelectionEvent], None]] = None
        self._selection_config = SelectionConfig()
        self._selection_index: Optional[ProjectedSelectionIndex] = None
        self._selection_index_key: object = None
        self._mouse = (0, 0)
        self._drag = ""
        self._closed = False
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
            self._host.make_current()
            context = moderngl.create_context(require=330)
            self._renderer = ModernGLRenderer(context)
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
    def capabilities(self) -> ViewerCapabilities:
        return GPU_CAPABILITIES

    @property
    def backend_diagnostics(self) -> tuple[str, ...]:
        return self._backend_diagnostics

    @property
    def section_plane(self) -> Optional[SectionPlane]:
        return self._section_plane

    @property
    def renderer_diagnostics(self) -> dict[str, int]:
        return {
            "draw_calls": self._renderer.draw_calls,
            "frame_count": self._renderer.frame_count,
            "geometry_uploads": self._renderer.geometry_uploads,
            "retained_meshes": len(self._entries),
        }

    def _bind_interaction(self) -> None:
        surface = self._host.surface
        surface.bind("<ButtonPress-1>", self._press_select, add="+")
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

    def _press_select(self, event: tk.Event) -> None:
        if self._selection_callback is None:
            return
        viewport = self._host.framebuffer_size()
        if self._selection_config.depth is SelectionDepth.THROUGH:
            hits = self.query_point(
                int(event.x), int(event.y), config=self._selection_config
            )
            self._selection_callback(
                SelectionEvent(
                    SelectionGesture.CLICK,
                    SelectionOperation.REPLACE,
                    hits=hits,
                    candidates=hits,
                    start=(int(event.x), int(event.y)),
                    end=(int(event.x), int(event.y)),
                )
            )
            return
        picked = self._renderer.pick(
            int(event.x), int(event.y), self.camera, viewport, self._section_plane
        )
        hits: tuple[SelectionHit, ...] = ()
        if picked is not None:
            handle, primitive_kind, primitive = picked
            entry = self._entries.get(id(handle), {})
            table = entry.get("owners")
            resolved: tuple[object, ...] = ()
            if isinstance(table, PackedOwnerTable):
                resolved = table.owners_for(
                    primitive_kind, primitive, entry.get("owner_resolver")
                )
            owner: Optional[PickOwner] = None
            identity: object = handle
            if resolved:
                candidate = resolved[0]
                if isinstance(candidate, PickOwner):
                    owner = candidate
                    identity = candidate.identity or candidate
                elif isinstance(candidate, ModelOwner):
                    identity = candidate
                    owner = PickOwner(
                        f"{candidate.model_id}:{candidate.kind}:{candidate.id}",
                        f"geometry.{candidate.kind}",
                        candidate.priority,
                        candidate,
                    )
            if owner is None:
                owner = PickOwner(
                    f"mesh:{id(handle)}:{primitive_kind}:{primitive}",
                    f"mesh.{primitive_kind}",
                )
            if self._selection_config.filter.accepts(owner):
                hits = (SelectionHit(owner, primitive, 0.0, identity=identity),)
        self._selection_callback(
            SelectionEvent(
                SelectionGesture.CLICK,
                SelectionOperation.REPLACE,
                hits=hits,
                candidates=hits,
                start=(int(event.x), int(event.y)),
                end=(int(event.x), int(event.y)),
            )
        )

    def set_selection_callback(
        self, callback: Optional[Callable[[SelectionEvent], None]]
    ) -> None:
        self._selection_callback = callback

    def configure_selection(
        self,
        callback: Optional[Callable[[SelectionEvent], None]],
        *,
        config: Optional[SelectionConfig] = None,
        **_options: object,
    ) -> None:
        """Configure semantic selection while retaining the software API shape."""

        self._selection_callback = callback
        if config is not None:
            if not isinstance(config, SelectionConfig):
                raise TypeError("config must be SelectionConfig")
            self._selection_config = config

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

    def _selection_binding(
        self,
        handle: MeshHandle,
        entry: dict[str, Any],
        primitive_kind: str,
        primitive: int,
    ) -> PickBinding:
        table = entry.get("owners")
        if isinstance(table, PackedOwnerTable):
            try:
                owners = table.owners_for(
                    primitive_kind, primitive, entry.get("owner_resolver")
                )
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
            None if self._section_plane is None else self._section_plane.key,
            tuple(
                (
                    identifier,
                    entry["handle"].generations,
                    entry["handle"].visible,
                    entry["handle"].removed,
                    entry["handle"].deformation_scale,
                )
                for identifier, entry in self._entries.items()
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

        for entry in self._entries.values():
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
                    camera_polygon = self._near_clip(camera_points(polygon), near)
                    if len(camera_polygon) < 3 or np.all(camera_polygon[:, 2] >= far):
                        continue
                    binding = self._selection_binding(
                        handle, entry, "triangle", triangle
                    ) if chunk_id is None else None
                    projected.append(
                        ProjectedPrimitive(
                            primitive_id,
                            "polygon",
                            screen_points(camera_polygon),
                            tuple(float(value) for value in camera_polygon[:, 2]),
                            binding,
                        )
                    )
                    primitive_id += 1
                if mesh.lines is not None:
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
                        binding = self._selection_binding(
                            handle, entry, "line", line
                        ) if chunk_id is None else None
                        projected.append(
                            ProjectedPrimitive(
                                primitive_id,
                                "segment",
                                screen_points(camera_segment),
                                tuple(float(value) for value in camera_segment[:, 2]),
                                binding,
                                radius=2.5,
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
                        binding = self._selection_binding(
                            handle, entry, "point", point
                        ) if chunk_id is None else None
                        projected.append(
                            ProjectedPrimitive(
                                primitive_id,
                                "point",
                                screen_points(camera_point[None, :]),
                                (float(camera_point[2]),),
                                binding,
                                radius=3.5,
                            )
                        )
                        primitive_id += 1
        self._selection_index = ProjectedSelectionIndex(projected, width, height)
        self._selection_index_key = key
        return self._selection_index

    def query_point(
        self,
        x: float,
        y: float,
        *,
        config: Optional[SelectionConfig] = None,
    ) -> tuple[SelectionHit, ...]:
        policy = config or self._selection_config
        hits = self._projected_selection_index().point_hits(
            x, y, policy.filter, radius=policy.click_radius_px
        )
        return hits if policy.depth is SelectionDepth.THROUGH else tuple(
            hit for hit in hits if hit.visible
        )

    def query_rectangle(
        self,
        rect: tuple[float, float, float, float],
        *,
        crossing: bool,
        config: Optional[SelectionConfig] = None,
    ) -> tuple[SelectionHit, ...]:
        policy = config or self._selection_config
        return self._projected_selection_index().rectangle_hits(
            rect, policy.filter, crossing=crossing, depth=policy.depth
        )

    def query_lasso(
        self,
        points: tuple[tuple[float, float], ...],
        *,
        config: Optional[SelectionConfig] = None,
    ) -> tuple[SelectionHit, ...]:
        policy = config or self._selection_config
        return self._projected_selection_index().polygon_hits(
            points, policy.filter, depth=policy.depth
        )

    def _mesh_changed(self, handle: MeshHandle, change: str) -> None:
        if change == "remove":
            self._renderer.remove_mesh(handle)
            self._entries.pop(id(handle), None)
        self._renderer.pick_dirty = True
        self._selection_index = None
        self.redraw()

    def add_mesh_arrays(
        self,
        mesh: MeshArrays,
        *,
        color: str = "#9aa7b4",
        line_color: str = "#334155",
        point_color: str = "#334155",
        line_width: float = 1.5,
        point_size: float = 7.0,
        opacity: float = 1.0,
        owners: Optional[PackedOwnerTable] = None,
        owner_resolver: Optional[Callable[..., object]] = None,
        **_appearance: object,
    ) -> MeshHandle:
        if not isinstance(mesh, MeshArrays):
            raise TypeError("mesh must be MeshArrays")
        handle = MeshHandle(mesh, on_change=self._mesh_changed)
        self._entries[id(handle)] = {
            "handle": handle,
            "owners": owners,
            "owner_resolver": owner_resolver,
        }
        self._host.make_current()
        self._renderer.add_mesh(
            handle,
            str(color),
            line_color=str(line_color),
            point_color=str(point_color),
            line_width=line_width,
            point_size=point_size,
            opacity=opacity,
        )
        self.redraw()
        self._selection_index = None
        return handle

    def add_layer(self, layer: object) -> object:
        attach = getattr(layer, "attach", None)
        if not callable(attach):
            raise TypeError("layer must provide attach(viewer)")
        return attach(self)

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

    def fit_to_scene(self, padding: float = 1.25, redraw: bool = True) -> None:
        values: list[np.ndarray] = []
        for entry in self._entries.values():
            handle = entry["handle"]
            if handle.removed or not handle.visible:
                continue
            for _chunk_id, mesh in [(None, handle.mesh), *handle.chunks]:
                positions = mesh.positions
                if mesh.displacements is not None and handle.deformation_scale:
                    positions = positions + handle.deformation_scale * mesh.displacements
                transform = handle.transform
                values.append(positions @ transform[:3, :3].T + transform[:3, 3])
        if not values:
            return
        positions = np.concatenate(values)
        low, high = positions.min(axis=0), positions.max(axis=0)
        target = 0.5 * (low + high)
        radius = max(1.0e-6, float(np.linalg.norm(high - low)) * 0.5)
        self.camera.set_target(Point3D(*target))
        self.camera.set_orbit(distance=max(self.camera.near * 2, radius * float(padding) / math.tan(self.camera.fov / 2)))
        self._renderer.pick_dirty = True
        self._selection_index = None
        if redraw:
            self.redraw()

    def redraw(self) -> None:
        if not self._closed:
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
        )
        self._host.swap_buffers()

    def clear(self) -> None:
        for entry in list(self._entries.values()):
            entry["handle"].remove()
        self._entries.clear()
        self.redraw()

    def destroy(self) -> None:
        if getattr(self, "_closed", True):
            super().destroy()
            return
        self._closed = True
        if self._update_poll_id is not None:
            self.after_cancel(self._update_poll_id)
            self._update_poll_id = None
        self._update_scheduler.close()
        for entry in list(self._entries.values()):
            entry["handle"].remove()
        self._entries.clear()
        renderer = getattr(self, "_renderer", None)
        if renderer is not None:
            self._host.make_current()
            renderer.release()
        host = getattr(self, "_host", None)
        if host is not None:
            host.close()
        super().destroy()
