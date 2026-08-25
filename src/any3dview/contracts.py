"""Public contracts shared by software and GPU viewer backends.

This module is deliberately toolkit independent.  In particular, importing it
must not import Tk, ModernGL, Pillow, or a concrete viewer implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence, runtime_checkable

from .arrays import MeshArrays
from .capabilities import ViewerCapabilities
from .clipping import SectionPlane
from .core import Camera3D, Point3D
from .ownership import PackedOwnerTable
from .retained import MeshHandle
from .semantic import SemanticRef, VisibilityState
from .selection import (
    SelectionConfig,
    SelectionDepth,
    SelectionEvent,
    SelectionFilter,
    SelectionHit,
)
from .shading import Light
from .shapes import Mesh


@dataclass(frozen=True, slots=True)
class Pick:
    """Legacy tag-based hit returned by click and hover callbacks."""

    tag: str
    item: int
    x: int
    y: int
    shift: bool = False
    ctrl: bool = False
    alt: bool = False


@dataclass(frozen=True, slots=True)
class ViewerState:
    """Portable view policy used when replacing one backend with another.

    Scene geometry is intentionally excluded.  Applications remain its owner
    and repopulate a candidate viewer before atomically switching widgets.
    """

    camera_position: Point3D
    camera_target: Point3D
    camera_world_up: Point3D
    fov: float
    near: float
    far: float
    section_plane: Optional[SectionPlane]
    background: str
    shading_enabled: bool = True
    occlude_lines: bool = True
    mesh_lines: bool = True
    axis_indicator: bool = True
    axis_ruler: bool = False
    interaction_profile: str = "legacy"
    semantic_selection: tuple[SemanticRef, ...] = ()
    visibility: VisibilityState = VisibilityState()


Projection = Optional[tuple[float, float, float]]


@runtime_checkable
class ViewerBackend(Protocol):
    """Complete renderer-neutral integration surface for bundled backends.

    Signatures follow the established ANYtk3D API.  A backend may accept
    additional keyword-only options for backwards compatibility, but every
    declaration below has matching behavior on the software and GPU viewers.
    Backend-specific pixels and raw Tk Canvas item identifiers are excluded.
    """

    camera: Camera3D
    canvas: object

    @property
    def backend_name(self) -> str: ...

    @property
    def capabilities(self) -> ViewerCapabilities: ...

    @property
    def backend_diagnostics(self) -> tuple[str, ...]: ...

    @property
    def event_widget(self) -> object: ...

    @property
    def viewport_size(self) -> tuple[int, int]: ...

    @property
    def light(self) -> Light: ...

    @property
    def section_plane(self) -> Optional[SectionPlane]: ...

    @property
    def interaction_profile(self) -> str: ...

    @property
    def selection_config(self) -> SelectionConfig: ...

    @property
    def semantic_selection(self) -> tuple[SemanticRef, ...]: ...

    @property
    def visibility_state(self) -> VisibilityState: ...

    @property
    def preselected_key(self) -> Optional[str]: ...

    @property
    def animation_frames(self) -> int: ...

    @property
    def animation_frame_index(self) -> int: ...

    @property
    def is_playing_animation(self) -> bool: ...

    def submit_update(
        self, callback: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> None: ...

    def export_view_state(self) -> ViewerState: ...

    def apply_view_state(self, state: ViewerState, *, redraw: bool = True) -> None: ...

    def project_point(self, point: object) -> Projection: ...

    def project_points(self, points: Iterable[object]) -> tuple[Projection, ...]: ...

    def screen_ray(self, x: float, y: float) -> tuple[Point3D, Point3D]: ...

    def unproject_to_plane(
        self, x: float, y: float, plane_point: object, plane_normal: object
    ) -> Optional[Point3D]: ...

    def configure_selection(
        self,
        callback: Optional[Callable[[SelectionEvent], None]],
        *,
        hover_callback: Optional[Callable[[Optional[SelectionHit]], None]] = None,
        config: Optional[SelectionConfig] = None,
    ) -> None: ...

    def set_selection_callback(
        self, callback: Optional[Callable[[SelectionEvent], None]]
    ) -> None: ...

    def update_selection_config(self, **changes: Any) -> SelectionConfig: ...

    def query_point(
        self,
        x: int,
        y: int,
        *,
        selection_filter: Optional[SelectionFilter] = None,
        radius: Optional[int] = None,
    ) -> tuple[SelectionHit, ...]: ...

    def query_rectangle(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        *,
        crossing: Optional[bool] = None,
        selection_filter: Optional[SelectionFilter] = None,
        depth: Optional[SelectionDepth] = None,
    ) -> tuple[SelectionHit, ...]: ...

    def query_lasso(
        self,
        points: Sequence[tuple[int, int]],
        *,
        selection_filter: Optional[SelectionFilter] = None,
        depth: Optional[SelectionDepth] = None,
    ) -> tuple[SelectionHit, ...]: ...

    def set_pick_callback(
        self,
        callback: Optional[Callable[[Pick], None]],
        *,
        prefix: str = "",
        radius: Optional[int] = None,
    ) -> None: ...

    def set_hover_callback(
        self, callback: Optional[Callable[[Optional[Pick]], None]]
    ) -> None: ...

    def pick_at(self, x: int, y: int) -> Optional[str]: ...

    def set_highlight(
        self,
        tags: Sequence[str] | Iterable[str],
        fill: Optional[str] = None,
        outline: Optional[str] = None,
    ) -> None: ...

    def clear_highlight(self) -> None: ...

    def highlighted_tags(self) -> frozenset[str]: ...

    def set_preselection(self, key: Optional[str]) -> None: ...

    def set_semantic_selection(self, values: Sequence[SemanticRef]) -> None: ...

    def set_visibility_state(self, state: VisibilityState) -> None: ...

    def set_light(
        self,
        direction: Optional[Point3D] = None,
        ambient: Optional[float] = None,
        diffuse: Optional[float] = None,
        specular: Optional[float] = None,
        shininess: Optional[float] = None,
        follow_camera: Optional[bool] = None,
        enabled: Optional[bool] = None,
    ) -> None: ...

    def set_shading(self, enabled: bool = True) -> None: ...

    def set_occlude_lines(self, enabled: bool = True) -> None: ...

    def set_background(self, color: str) -> None: ...

    def set_mesh_lines(self, visible: bool = True) -> None: ...

    def set_axis_indicator(self, visible: bool = True) -> None: ...

    def set_axis_ruler(self, visible: bool = True) -> None: ...

    def set_interaction_profile(self, profile: str) -> None: ...

    def set_thickness_legend(
        self,
        values: Sequence[float],
        unit: str = "mm",
        title: str = "Plate thickness",
        width: int = 170,
        value_range: Optional[tuple[float, float]] = None,
        colors: Optional[Sequence[str]] = None,
    ) -> None: ...

    def clear_thickness_legend(self) -> None: ...

    def thickness_color(
        self,
        thickness: float,
        value_range: Optional[tuple[float, float]] = None,
    ) -> str: ...

    def set_section_plane(
        self,
        normal: object = (1.0, 0.0, 0.0),
        offset: float = 0.0,
        *,
        enabled: bool = True,
    ) -> None: ...

    def clear_section_plane(self) -> None: ...

    def set_camera_position(self, position: Point3D) -> None: ...

    def set_camera_target(self, target: Point3D) -> None: ...

    def set_view(self, azimuth_degrees: float, elevation_degrees: float) -> None: ...

    def set_iso_view(self) -> None: ...

    def set_top_view(self) -> None: ...

    def set_side_view(self) -> None: ...

    def set_front_view(self) -> None: ...

    def reset_camera(self) -> None: ...

    def fit_to_scene(self, padding: float = 1.25, redraw: bool = True) -> None: ...

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
    ) -> MeshHandle: ...

    def add_layer(self, layer: object) -> object: ...

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
    ) -> None: ...

    def add_text(
        self,
        point: Point3D,
        text: str,
        color: str = "black",
        font: tuple[str, int, str] = ("Segoe UI", 9, "bold"),
        anchor: str = "center",
        layer: int = 35,
        draw_overlay: bool = True,
    ) -> None: ...

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
    ) -> None: ...

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
    ) -> None: ...

    def add_markers(
        self,
        points: Iterable[object],
        colors: object = "#2563eb",
        size: object = 6,
        outline: object = "",
        layer: int = 32,
        tags: str = "",
        bindings: object = None,
    ) -> None: ...

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
    ) -> None: ...

    def add_shape(
        self,
        mesh: Mesh,
        position: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        up: Optional[Point3D] = None,
        **material: Any,
    ) -> None: ...

    def add_box(
        self,
        size_x: float = 1.0,
        size_y: Optional[float] = None,
        size_z: Optional[float] = None,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        **material: Any,
    ) -> None: ...

    def add_box_from_bounds(
        self, minimum: Point3D, maximum: Point3D, **material: Any
    ) -> None: ...

    def add_sphere(
        self,
        radius: float = 1.0,
        center: Optional[Point3D] = None,
        segments: int = 24,
        rings: int = 16,
        **material: Any,
    ) -> None: ...

    def add_cone(
        self,
        radius: float = 1.0,
        height: float = 1.0,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        segments: int = 32,
        capped: bool = True,
        **material: Any,
    ) -> None: ...

    def add_frustum(
        self,
        radius_bottom: float = 1.0,
        radius_top: float = 0.5,
        height: float = 1.0,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        segments: int = 32,
        height_segments: int = 1,
        capped: bool = True,
        **material: Any,
    ) -> None: ...

    def add_tube(
        self,
        outer_radius: float = 1.0,
        inner_radius: float = 0.7,
        height: float = 1.0,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        segments: int = 32,
        height_segments: int = 1,
        capped: bool = True,
        **material: Any,
    ) -> None: ...

    def add_torus(
        self,
        major_radius: float = 1.0,
        minor_radius: float = 0.25,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        segments: int = 36,
        rings: int = 18,
        **material: Any,
    ) -> None: ...

    def add_pyramid(
        self,
        base_radius: float = 1.0,
        height: float = 1.0,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        sides: int = 4,
        capped: bool = True,
        **material: Any,
    ) -> None: ...

    def add_wedge(
        self,
        size_x: float = 1.0,
        size_y: float = 1.0,
        size_z: float = 1.0,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        **material: Any,
    ) -> None: ...

    def add_prism(
        self,
        profile: Sequence[tuple[float, float]],
        height: float = 1.0,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        capped: bool = True,
        **material: Any,
    ) -> None: ...

    def add_extrusion(
        self,
        profile: Sequence[tuple[float, float]],
        path: Sequence[Point3D],
        capped: bool = True,
        up: Optional[Point3D] = None,
        **material: Any,
    ) -> None: ...

    def add_disk(
        self,
        outer_radius: float = 1.0,
        inner_radius: float = 0.0,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        segments: int = 32,
        **material: Any,
    ) -> None: ...

    def add_plane(
        self,
        size_x: float = 1.0,
        size_y: float = 1.0,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        nx: int = 1,
        ny: int = 1,
        **material: Any,
    ) -> None: ...

    def add_arrow(
        self,
        start: Point3D,
        end: Point3D,
        shaft_radius: Optional[float] = None,
        head_radius: Optional[float] = None,
        head_length: Optional[float] = None,
        segments: int = 16,
        **material: Any,
    ) -> None: ...

    def add_beam(
        self,
        start: Point3D,
        end: Point3D,
        kind: str = "T",
        web_height: float = 0.2,
        web_thickness: float = 0.01,
        flange_width: float = 0.1,
        flange_thickness: float = 0.015,
        up: Optional[Point3D] = None,
        capped: bool = True,
        **material: Any,
    ) -> None: ...

    def add_grid(
        self,
        size_x: float = 10.0,
        size_y: float = 10.0,
        step: float = 1.0,
        z: float = 0.0,
        center: Optional[Point3D] = None,
        color: str = "#c9d2dc",
        width: int = 1,
        layer: int = 2,
    ) -> None: ...

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
    ) -> None: ...

    def add_longitudinal_stiffener(
        self,
        radius: float,
        height: float,
        angle: float,
        radius_top: Optional[float] = None,
        web_height: float = 0.1,
        web_thickness: float = 0.01,
        flange_width: float = 0.05,
        flange_thickness: float = 0.01,
        color: str = "silver",
        outline: str = "black",
        segments: int = 4,
        height_segments: int = 16,
        inside: bool = False,
        z_offset: float = 0.0,
    ) -> None: ...

    def add_ring_stiffener(
        self,
        radius: float,
        z_position: float,
        web_height: float = 0.1,
        web_thickness: float = 0.01,
        flange_width: float = 0.05,
        flange_thickness: float = 0.01,
        color: str = "dimgray",
        outline: str = "black",
        segments: int = 32,
        inside: bool = False,
    ) -> None: ...

    def add_rectangular_plate(
        self,
        x_start: float,
        x_end: float,
        y_start: float,
        y_end: float,
        z: float = 0.0,
        color: str = "gray",
        outline: str = "black",
        stipple: str = "",
        layer: int = 5,
        back_color: str = "",
        nx: int = 24,
        ny: int = 24,
        opacity: Optional[float] = None,
    ) -> None: ...

    def add_flat_stiffener(
        self,
        x_start: float,
        x_end: float,
        y: float,
        z_base: float,
        hw: float,
        b: float,
        color: str = "gray",
        outline: str = "black",
        stipple: str = "",
        layer_web: int = 12,
        layer_flange: int = 13,
        nx: int = 24,
        opacity: Optional[float] = None,
    ) -> None: ...

    def add_flat_girder(
        self,
        x: float,
        y_start: float,
        y_end: float,
        z_base: float,
        ghw: float,
        gb: float,
        color: str = "gray",
        outline: str = "black",
        stipple: str = "",
        layer_web: int = 14,
        layer_flange: int = 15,
        ny: int = 24,
        opacity: Optional[float] = None,
    ) -> None: ...

    def set_opaque_cylinder_occluder(
        self, radius: float, height: float, center: Optional[Point3D] = None
    ) -> None: ...

    def set_interactive_detail(self, faces: int) -> None: ...

    def begin_animation_cache(self) -> None: ...

    def capture_animation_frame(self) -> None: ...

    def play_animation(self, fps: int = 30, fast: Optional[bool] = None) -> None: ...

    def stop_animation(self) -> None: ...

    def capture_image(self) -> Any: ...

    def clear(self, keep_canvas: bool = False) -> None: ...

    def redraw(self) -> None: ...

    def destroy(self) -> None: ...
