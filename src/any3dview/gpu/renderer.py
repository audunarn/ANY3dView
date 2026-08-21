"""Persistent ModernGL resources for the optional GPU backend."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Any, Optional

import numpy as np

try:
    import moderngl
except ImportError as error:  # pragma: no cover - isolated-wheel coverage
    raise ImportError(
        "ANY3dView GPU support requires the 'gpu' extra: pip install ANY3dView[gpu]"
    ) from error

from ..arrays import MeshArrays
from ..clipping import SectionPlane
from ..core import Camera3D, _interpolate_thickness_color, parse_color
from ..retained import DirtyGenerations, MeshHandle


_VERTEX = """
#version 330
uniform mat4 u_view_rotation;
uniform mat4 u_projection;
uniform mat3 u_linear;
uniform vec3 u_origin_from_camera;
uniform float u_deformation_scale;
in vec3 in_position;
in vec3 in_displacement;
in float in_node_scalar;
out vec3 v_relative;
out float v_node_scalar;
void main() {
    vec3 local = in_position + u_deformation_scale * in_displacement;
    v_relative = u_linear * local + u_origin_from_camera;
    v_node_scalar = in_node_scalar;
    gl_Position = u_projection * u_view_rotation * vec4(v_relative, 1.0);
}
"""

_FRAGMENT = """
#version 330
uniform vec3 u_color;
uniform vec3 u_selection_color;
uniform float u_opacity;
uniform vec3 u_light_direction;
uniform bool u_section_enabled;
uniform vec3 u_section_normal;
uniform float u_section_relative_offset;
uniform bool u_has_scalars;
uniform bool u_node_scalars;
uniform vec2 u_scalar_range;
uniform uint u_element_width;
uniform uint u_triangle_width;
uniform usampler2D u_triangle_to_element;
uniform usampler2D u_active_elements;
uniform usampler2D u_selected_elements;
uniform sampler2D u_element_scalars;
uniform sampler2D u_colormap;
in vec3 v_relative;
in float v_node_scalar;
out vec4 frag_color;
ivec2 address(uint index, uint width) {
    return ivec2(int(index % width), int(index / width));
}
void main() {
    uint element = texelFetch(
        u_triangle_to_element, address(uint(gl_PrimitiveID), u_triangle_width), 0
    ).r;
    if (texelFetch(u_active_elements, address(element, u_element_width), 0).r == 0u) discard;
    if (u_section_enabled && dot(u_section_normal, v_relative) < u_section_relative_offset) discard;
    vec3 base = u_color;
    if (u_has_scalars) {
        float value = u_node_scalars
            ? v_node_scalar
            : texelFetch(u_element_scalars, address(element, u_element_width), 0).r;
        if (isnan(value)) base = vec3(0.5);
        else {
            float span = max(1.0e-20, u_scalar_range.y - u_scalar_range.x);
            base = texture(u_colormap, vec2(clamp((value - u_scalar_range.x) / span, 0.0, 1.0), 0.5)).rgb;
        }
    }
    vec3 dx = dFdx(v_relative);
    vec3 dy = dFdy(v_relative);
    vec3 normal = normalize(cross(dx, dy));
    float diffuse = 0.25 + 0.75 * abs(dot(normal, normalize(u_light_direction)));
    vec3 shaded = base * diffuse;
    if (texelFetch(u_selected_elements, address(element, u_element_width), 0).r != 0u)
        shaded = mix(shaded, u_selection_color, 0.65);
    frag_color = vec4(shaded, u_opacity);
}
"""

_PICK_FRAGMENT = """
#version 330
uniform bool u_section_enabled;
uniform vec3 u_section_normal;
uniform float u_section_relative_offset;
uniform uint u_pick_base;
uniform uint u_element_width;
uniform uint u_triangle_width;
uniform usampler2D u_triangle_to_element;
uniform usampler2D u_active_elements;
in vec3 v_relative;
layout(location = 0) out uint pick_id;
ivec2 address(uint index, uint width) {
    return ivec2(int(index % width), int(index / width));
}
void main() {
    uint element = texelFetch(
        u_triangle_to_element, address(uint(gl_PrimitiveID), u_triangle_width), 0
    ).r;
    if (texelFetch(u_active_elements, address(element, u_element_width), 0).r == 0u) discard;
    if (u_section_enabled && dot(u_section_normal, v_relative) < u_section_relative_offset) discard;
    pick_id = u_pick_base + uint(gl_PrimitiveID) + 1u;
}
"""

_LINE_VERTEX = """
#version 330
uniform mat4 u_view_rotation;
uniform mat4 u_projection;
uniform mat3 u_linear;
uniform vec3 u_origin_from_camera;
uniform float u_deformation_scale;
uniform vec2 u_viewport;
uniform float u_half_width;
in vec3 in_start;
in vec3 in_end;
in vec3 in_start_displacement;
in vec3 in_end_displacement;
out vec3 v_relative;
flat out uint v_primitive;
void main() {
    vec3 start = u_linear * (in_start + u_deformation_scale * in_start_displacement)
        + u_origin_from_camera;
    vec3 end = u_linear * (in_end + u_deformation_scale * in_end_displacement)
        + u_origin_from_camera;
    vec4 clip_start = u_projection * u_view_rotation * vec4(start, 1.0);
    vec4 clip_end = u_projection * u_view_rotation * vec4(end, 1.0);
    vec2 screen_start = clip_start.xy / clip_start.w * u_viewport;
    vec2 screen_end = clip_end.xy / clip_end.w * u_viewport;
    vec2 direction = screen_end - screen_start;
    float length_px = max(length(direction), 1.0e-6);
    vec2 normal = vec2(-direction.y, direction.x) / length_px;
    bool at_end = gl_VertexID >= 2;
    bool positive = gl_VertexID == 1 || gl_VertexID == 3;
    vec4 clip = at_end ? clip_end : clip_start;
    vec2 offset = normal * (positive ? u_half_width : -u_half_width);
    clip.xy += offset * (2.0 / u_viewport) * clip.w;
    gl_Position = clip;
    v_relative = at_end ? end : start;
    v_primitive = uint(gl_InstanceID);
}
"""

_LINE_FRAGMENT = """
#version 330
uniform vec3 u_color;
uniform bool u_section_enabled;
uniform vec3 u_section_normal;
uniform float u_section_relative_offset;
in vec3 v_relative;
flat in uint v_primitive;
out vec4 frag_color;
void main() {
    if (u_section_enabled && dot(u_section_normal, v_relative) < u_section_relative_offset) discard;
    frag_color = vec4(u_color, 1.0);
}
"""

_LINE_PICK_FRAGMENT = """
#version 330
uniform bool u_section_enabled;
uniform vec3 u_section_normal;
uniform float u_section_relative_offset;
uniform uint u_pick_base;
in vec3 v_relative;
flat in uint v_primitive;
layout(location = 0) out uint pick_id;
void main() {
    if (u_section_enabled && dot(u_section_normal, v_relative) < u_section_relative_offset) discard;
    pick_id = u_pick_base + v_primitive + 1u;
}
"""

_POINT_VERTEX = """
#version 330
uniform mat4 u_view_rotation;
uniform mat4 u_projection;
uniform mat3 u_linear;
uniform vec3 u_origin_from_camera;
uniform float u_deformation_scale;
uniform float u_point_size;
in vec3 in_position;
in vec3 in_displacement;
out vec3 v_relative;
void main() {
    vec3 local = in_position + u_deformation_scale * in_displacement;
    v_relative = u_linear * local + u_origin_from_camera;
    gl_Position = u_projection * u_view_rotation * vec4(v_relative, 1.0);
    gl_PointSize = u_point_size;
}
"""

_POINT_FRAGMENT = """
#version 330
uniform vec3 u_color;
uniform bool u_section_enabled;
uniform vec3 u_section_normal;
uniform float u_section_relative_offset;
in vec3 v_relative;
out vec4 frag_color;
void main() {
    vec2 delta = gl_PointCoord * 2.0 - 1.0;
    if (dot(delta, delta) > 1.0) discard;
    if (u_section_enabled && dot(u_section_normal, v_relative) < u_section_relative_offset) discard;
    frag_color = vec4(u_color, 1.0);
}
"""

_POINT_PICK_FRAGMENT = """
#version 330
uniform bool u_section_enabled;
uniform vec3 u_section_normal;
uniform float u_section_relative_offset;
uniform uint u_pick_base;
in vec3 v_relative;
layout(location = 0) out uint pick_id;
void main() {
    vec2 delta = gl_PointCoord * 2.0 - 1.0;
    if (dot(delta, delta) > 1.0) discard;
    if (u_section_enabled && dot(u_section_normal, v_relative) < u_section_relative_offset) discard;
    pick_id = u_pick_base + uint(gl_PrimitiveID) + 1u;
}
"""


def _matrix_bytes(matrix: np.ndarray) -> bytes:
    return np.asarray(matrix, dtype=np.float32).T.tobytes()


def _rgb(color: str) -> tuple[float, float, float]:
    parsed = parse_color(color) or (154, 167, 180)
    return tuple(channel / 255.0 for channel in parsed)


def _packed_texture(ctx: Any, values: np.ndarray, dtype: str) -> tuple[Any, int]:
    flat = np.ascontiguousarray(values).reshape(-1)
    width = min(4096, max(1, len(flat)))
    height = max(1, math.ceil(len(flat) / width))
    padded = np.zeros(width * height, dtype=flat.dtype)
    padded[: len(flat)] = flat
    texture = ctx.texture((width, height), 1, padded.tobytes(), dtype=dtype)
    texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
    return texture, width


@dataclass(slots=True)
class _Chunk:
    ctx: Any
    mesh: MeshArrays
    handle: MeshHandle
    color: str
    origin: np.ndarray
    radius: float
    vertex_buffer: Any
    node_scalar_buffer: Any
    index_buffer: Any
    render_vao: Any
    pick_vao: Any
    line_buffer: Any
    line_render_vao: Any
    line_pick_vao: Any
    point_index_buffer: Any
    point_render_vao: Any
    point_pick_vao: Any
    triangle_map: Any
    triangle_width: int
    active: Any
    element_width: int
    selected: Any
    scalars: Any
    colormap: Any
    has_scalars: bool
    node_scalars: bool
    scalar_range: tuple[float, float]

    @classmethod
    def create(
        cls,
        ctx: Any,
        mesh: MeshArrays,
        handle: MeshHandle,
        render_program: Any,
        pick_program: Any,
        line_program: Any,
        line_pick_program: Any,
        point_program: Any,
        point_pick_program: Any,
        color: str,
    ) -> "_Chunk":
        origin = (
            np.mean(mesh.positions, axis=0, dtype=np.float64)
            if len(mesh.positions)
            else np.zeros(3, dtype=np.float64)
        )
        radius = (
            float(np.max(np.linalg.norm(mesh.positions - origin, axis=1)))
            if len(mesh.positions)
            else 0.0
        )
        relative = np.asarray(mesh.positions - origin, dtype=np.float32)
        displacement = (
            np.zeros_like(relative)
            if mesh.displacements is None
            else np.asarray(mesh.displacements, dtype=np.float32)
        )
        interleaved = np.ascontiguousarray(np.column_stack((relative, displacement)))
        vertex_payload = interleaved.tobytes()
        vertex_buffer = (
            ctx.buffer(vertex_payload) if vertex_payload else ctx.buffer(reserve=24)
        )
        node_scalar_values = (
            np.zeros(mesh.node_count, dtype=np.float32)
            if mesh.node_scalars is None
            else mesh.node_scalars.astype(np.float32, copy=False)
        )
        node_scalar_buffer = ctx.buffer(
            np.ascontiguousarray(node_scalar_values).tobytes()
            if len(node_scalar_values)
            else None,
            reserve=0 if len(node_scalar_values) else 4,
        )
        index_payload = np.ascontiguousarray(mesh.triangles).tobytes()
        index_buffer = (
            ctx.buffer(index_payload) if index_payload else ctx.buffer(reserve=4)
        )
        geometry_content = [(vertex_buffer, "3f 3f", "in_position", "in_displacement")]
        render_content = [
            *geometry_content,
            (node_scalar_buffer, "1f", "in_node_scalar"),
        ]
        render_vao = ctx.vertex_array(render_program, render_content, index_buffer)
        pick_vao = ctx.vertex_array(pick_program, geometry_content, index_buffer)
        line_buffer = None
        line_render_vao = None
        line_pick_vao = None
        if mesh.lines is not None and len(mesh.lines):
            line_data = cls._line_data(mesh, origin)
            line_buffer = ctx.buffer(line_data.tobytes())
            line_content = [
                (
                    line_buffer,
                    "3f 3f 3f 3f /i",
                    "in_start",
                    "in_end",
                    "in_start_displacement",
                    "in_end_displacement",
                )
            ]
            line_render_vao = ctx.vertex_array(line_program, line_content)
            line_pick_vao = ctx.vertex_array(line_pick_program, line_content)
        point_index_buffer = None
        point_render_vao = None
        point_pick_vao = None
        if mesh.point_indices is not None and len(mesh.point_indices):
            point_index_buffer = ctx.buffer(mesh.point_indices.tobytes())
            point_render_vao = ctx.vertex_array(
                point_program, geometry_content, point_index_buffer
            )
            point_pick_vao = ctx.vertex_array(
                point_pick_program, geometry_content, point_index_buffer
            )
        mapping = (
            np.arange(mesh.triangle_count, dtype=np.uint32)
            if mesh.triangle_to_element is None
            else mesh.triangle_to_element.astype(np.uint32, copy=False)
        )
        triangle_map, triangle_width = _packed_texture(ctx, mapping, "u4")
        active_values = (
            np.ones(mesh.element_count, dtype=np.uint32)
            if mesh.active_elements is None
            else mesh.active_elements.astype(np.uint32)
        )
        active, element_width = _packed_texture(ctx, active_values, "u4")
        selected_values = np.zeros(mesh.element_count, dtype=np.uint32)
        selected_indices = handle.selected_elements.astype(np.int64)
        selected_indices = selected_indices[selected_indices < len(selected_values)]
        selected_values[selected_indices] = 1
        selected, _ = _packed_texture(ctx, selected_values, "u4")
        element_scalar_values = (
            np.zeros(mesh.element_count, dtype=np.float32)
            if mesh.element_scalars is None
            else mesh.element_scalars.astype(np.float32, copy=False)
        )
        scalars, _ = _packed_texture(ctx, element_scalar_values, "f4")
        scalar_values = (
            element_scalar_values
            if mesh.element_scalars is not None
            else node_scalar_values
        )
        finite = scalar_values[np.isfinite(scalar_values)]
        scalar_range = (
            (float(finite.min()), float(finite.max()))
            if len(finite)
            else (0.0, 1.0)
        )
        palette = np.empty((256, 3), dtype=np.float32)
        for index in range(256):
            value = index / 255.0
            palette[index] = _rgb(_interpolate_thickness_color(value, 0.0, 1.0))
        colormap = ctx.texture((256, 1), 3, palette.tobytes(), dtype="f4")
        colormap.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return cls(
            ctx,
            mesh,
            handle,
            color,
            origin,
            radius,
            vertex_buffer,
            node_scalar_buffer,
            index_buffer,
            render_vao,
            pick_vao,
            line_buffer,
            line_render_vao,
            line_pick_vao,
            point_index_buffer,
            point_render_vao,
            point_pick_vao,
            triangle_map,
            triangle_width,
            active,
            element_width,
            selected,
            scalars,
            colormap,
            mesh.element_scalars is not None,
            mesh.element_scalars is None and mesh.node_scalars is not None,
            scalar_range,
        )

    @staticmethod
    def _line_data(mesh: MeshArrays, origin: np.ndarray) -> np.ndarray:
        assert mesh.lines is not None
        relative = np.asarray(mesh.positions - origin, dtype=np.float32)
        displacement = (
            np.zeros_like(relative)
            if mesh.displacements is None
            else np.asarray(mesh.displacements, dtype=np.float32)
        )
        return np.ascontiguousarray(
            np.column_stack(
                (
                    relative[mesh.lines[:, 0]],
                    relative[mesh.lines[:, 1]],
                    displacement[mesh.lines[:, 0]],
                    displacement[mesh.lines[:, 1]],
                )
            ),
            dtype=np.float32,
        )

    def release(self) -> None:
        for resource in (
            self.render_vao,
            self.pick_vao,
            self.line_render_vao,
            self.line_pick_vao,
            self.line_buffer,
            self.point_render_vao,
            self.point_pick_vao,
            self.point_index_buffer,
            self.vertex_buffer,
            self.node_scalar_buffer,
            self.index_buffer,
            self.triangle_map,
            self.active,
            self.selected,
            self.scalars,
            self.colormap,
        ):
            if resource is not None:
                resource.release()

    def bind_fields(self, program: Any) -> None:
        self.triangle_map.use(location=0)
        self.active.use(location=1)
        self.selected.use(location=2)
        self.scalars.use(location=3)
        self.colormap.use(location=4)
        for name, value in (
            ("u_triangle_to_element", 0),
            ("u_active_elements", 1),
            ("u_selected_elements", 2),
            ("u_element_scalars", 3),
            ("u_colormap", 4),
            ("u_triangle_width", self.triangle_width),
            ("u_element_width", self.element_width),
        ):
            if name in program:
                program[name].value = value


@dataclass(slots=True)
class _Group:
    handle: MeshHandle
    color: str
    line_color: str
    point_color: str
    line_width: float
    point_size: float
    opacity: float
    generations: DirtyGenerations
    chunks: list[tuple[object, _Chunk]]


class ModernGLRenderer:
    """Demand-driven renderer with one persistent resource group per handle."""

    def __init__(self, context: Any) -> None:
        self.ctx = context
        if int(context.version_code) < 330:
            raise RuntimeError("ANY3dView requires OpenGL 3.3 or newer")
        self.render_program = context.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)
        self.pick_program = context.program(vertex_shader=_VERTEX, fragment_shader=_PICK_FRAGMENT)
        self.line_program = context.program(
            vertex_shader=_LINE_VERTEX, fragment_shader=_LINE_FRAGMENT
        )
        self.line_pick_program = context.program(
            vertex_shader=_LINE_VERTEX, fragment_shader=_LINE_PICK_FRAGMENT
        )
        self.point_program = context.program(
            vertex_shader=_POINT_VERTEX, fragment_shader=_POINT_FRAGMENT
        )
        self.point_pick_program = context.program(
            vertex_shader=_POINT_VERTEX, fragment_shader=_POINT_PICK_FRAGMENT
        )
        self._groups: dict[int, _Group] = {}
        self._pick_fbo = None
        self._pick_texture = None
        self._pick_depth = None
        self._pick_size = (0, 0)
        self._pick_ranges: list[tuple[int, int, MeshHandle, str]] = []
        self.pick_dirty = True
        self.draw_calls = 0
        self.frame_count = 0
        self.geometry_uploads = 0

    def add_mesh(
        self,
        handle: MeshHandle,
        color: str = "#9aa7b4",
        *,
        line_color: str = "#334155",
        point_color: str = "#334155",
        line_width: float = 1.5,
        point_size: float = 7.0,
        opacity: float = 1.0,
    ) -> None:
        self.remove_mesh(handle)
        chunks = [
            (
                chunk_id,
                _Chunk.create(
                    self.ctx,
                    mesh,
                    handle,
                    self.render_program,
                    self.pick_program,
                    self.line_program,
                    self.line_pick_program,
                    self.point_program,
                    self.point_pick_program,
                    color,
                ),
            )
            for chunk_id, mesh in [(None, handle.mesh), *handle.chunks]
        ]
        self._groups[id(handle)] = _Group(
            handle,
            color,
            line_color,
            point_color,
            max(0.5, float(line_width)),
            max(1.0, float(point_size)),
            min(1.0, max(0.0, float(opacity))),
            handle.generations,
            chunks,
        )
        self.geometry_uploads += len(chunks)
        self.pick_dirty = True

    def remove_mesh(self, handle: MeshHandle) -> None:
        group = self._groups.pop(id(handle), None)
        if group is not None:
            for _chunk_id, chunk in group.chunks:
                chunk.release()
            self.pick_dirty = True

    def _sync(self, group: _Group) -> None:
        current = group.handle.generations
        previous = group.generations
        if current.topology != previous.topology or current.position != previous.position:
            self.add_mesh(
                group.handle,
                group.color,
                line_color=group.line_color,
                point_color=group.point_color,
                line_width=group.line_width,
                point_size=group.point_size,
                opacity=group.opacity,
            )
            return
        sources = {chunk_id: mesh for chunk_id, mesh in [(None, group.handle.mesh), *group.handle.chunks]}
        if current.displacement != previous.displacement:
            # A displacement array or scale changed.  Scale is uniform-only;
            # upload only if the underlying array object changed.
            uploads = 0
            for chunk_id, chunk in group.chunks:
                mesh = sources[chunk_id]
                if mesh.displacements is not chunk.mesh.displacements:
                    relative = np.asarray(mesh.positions - chunk.origin, dtype=np.float32)
                    displacement = (
                        np.zeros_like(relative)
                        if mesh.displacements is None
                        else mesh.displacements.astype(np.float32, copy=False)
                    )
                    data = np.ascontiguousarray(
                        np.column_stack((relative, displacement))
                    )
                    chunk.vertex_buffer.write(data.tobytes())
                    if chunk.line_buffer is not None:
                        chunk.line_buffer.write(_Chunk._line_data(mesh, chunk.origin).tobytes())
                    uploads += 1
            self.geometry_uploads += uploads
        if current.active != previous.active:
            for chunk_id, chunk in group.chunks:
                mesh = sources[chunk_id]
                values = (
                    np.ones(mesh.element_count, dtype=np.uint32)
                    if mesh.active_elements is None
                    else mesh.active_elements.astype(np.uint32)
                )
                padded = np.zeros(
                    max(1, chunk.element_width * math.ceil(len(values) / chunk.element_width)),
                    np.uint32,
                )
                padded[: len(values)] = values
                chunk.active.write(padded.tobytes())
        if current.selection != previous.selection:
            for _chunk_id, chunk in group.chunks:
                values = np.zeros(chunk.mesh.element_count, dtype=np.uint32)
                selected = group.handle.selected_elements.astype(np.int64)
                selected = selected[selected < len(values)]
                values[selected] = 1
                padded = np.zeros(
                    max(1, chunk.element_width * math.ceil(len(values) / chunk.element_width)),
                    np.uint32,
                )
                padded[: len(values)] = values
                chunk.selected.write(padded.tobytes())
        if current.scalar != previous.scalar:
            for chunk_id, chunk in group.chunks:
                mesh = sources[chunk_id]
                values = (
                    np.zeros(mesh.element_count, dtype=np.float32)
                    if mesh.element_scalars is None
                    else mesh.element_scalars.astype(np.float32, copy=False)
                )
                padded = np.zeros(
                    max(1, chunk.element_width * math.ceil(len(values) / chunk.element_width)),
                    np.float32,
                )
                padded[: len(values)] = values
                chunk.scalars.write(padded.tobytes())
                node_values = (
                    np.zeros(mesh.node_count, dtype=np.float32)
                    if mesh.node_scalars is None
                    else mesh.node_scalars.astype(np.float32, copy=False)
                )
                chunk.node_scalar_buffer.write(
                    np.ascontiguousarray(node_values).tobytes()
                )
                range_values = values if mesh.element_scalars is not None else node_values
                finite = range_values[np.isfinite(range_values)]
                chunk.scalar_range = (
                    (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
                )
                chunk.has_scalars = (
                    mesh.element_scalars is not None or mesh.node_scalars is not None
                )
                chunk.node_scalars = (
                    mesh.element_scalars is None and mesh.node_scalars is not None
                )
        for chunk_id, chunk in group.chunks:
            chunk.mesh = sources[chunk_id]
        group.generations = current
        self.pick_dirty = True

    @staticmethod
    def _camera_matrices(camera: Camera3D, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        right, up, forward = camera.basis()
        view = np.eye(4, dtype=np.float64)
        view[:3, :3] = np.asarray(
            [right.to_tuple(), up.to_tuple(), (-forward).to_tuple()], dtype=np.float64
        )
        aspect = width / max(1.0, float(height))
        scale = 1.0 / math.tan(camera.fov / 2.0)
        near, far = float(camera.near), float(camera.far)
        projection = np.zeros((4, 4), dtype=np.float64)
        projection[0, 0] = scale / aspect
        projection[1, 1] = scale
        projection[2, 2] = (far + near) / (near - far)
        projection[2, 3] = 2.0 * far * near / (near - far)
        projection[3, 2] = -1.0
        return view, projection

    def _uniforms(
        self,
        program: Any,
        chunk: _Chunk,
        camera: Camera3D,
        viewport: tuple[int, int],
        section_plane: Optional[SectionPlane],
    ) -> None:
        width, height = viewport
        view, projection = self._camera_matrices(camera, width, height)
        transform = chunk.handle.transform
        transformed_origin = transform[:3, :3] @ chunk.origin + transform[:3, 3]
        camera_position = np.asarray(camera.position.to_tuple(), dtype=np.float64)
        program["u_view_rotation"].write(_matrix_bytes(view))
        program["u_projection"].write(_matrix_bytes(projection))
        program["u_linear"].write(_matrix_bytes(transform[:3, :3]))
        program["u_origin_from_camera"].value = tuple(
            np.asarray(transformed_origin - camera_position, dtype=np.float32)
        )
        program["u_deformation_scale"].value = chunk.handle.deformation_scale
        if "u_viewport" in program:
            program["u_viewport"].value = (float(width), float(height))
        enabled = section_plane is not None and section_plane.enabled
        program["u_section_enabled"].value = enabled
        normal = section_plane.normal.to_tuple() if enabled else (1.0, 0.0, 0.0)
        program["u_section_normal"].value = normal
        program["u_section_relative_offset"].value = (
            float(section_plane.offset - np.dot(normal, camera_position)) if enabled else 0.0
        )

    def render(
        self,
        camera: Camera3D,
        viewport: tuple[int, int],
        *,
        section_plane: Optional[SectionPlane] = None,
        target: Any = None,
        clear_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        self.frame_count += 1
        for group in list(self._groups.values()):
            self._sync(group)
        framebuffer = target or self.ctx.screen
        framebuffer.use()
        self.ctx.viewport = (0, 0, int(viewport[0]), int(viewport[1]))
        framebuffer.clear(*clear_color, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.draw_calls = 0
        draw_items = [
            (group, chunk)
            for group in self._groups.values()
            if not group.handle.removed and group.handle.visible
            for _chunk_id, chunk in group.chunks
            if self._chunk_visible(chunk, camera, viewport)
        ]
        camera_position = np.asarray(camera.position.to_tuple(), dtype=np.float64)
        transparent = sorted(
            (item for item in draw_items if item[0].opacity < 0.999),
            key=lambda item: -float(
                np.linalg.norm(
                    item[0].handle.transform[:3, :3] @ item[1].origin
                    + item[0].handle.transform[:3, 3]
                    - camera_position
                )
            ),
        )
        ordered = [item for item in draw_items if item[0].opacity >= 0.999] + transparent
        for group, chunk in ordered:
            is_transparent = group.opacity < 0.999
            if is_transparent:
                self.ctx.enable(moderngl.BLEND)
                self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                self.ctx.depth_mask = False
            else:
                self.ctx.disable(moderngl.BLEND)
                self.ctx.depth_mask = True
            self._uniforms(self.render_program, chunk, camera, viewport, section_plane)
            chunk.bind_fields(self.render_program)
            self.render_program["u_color"].value = _rgb(chunk.color)
            self.render_program["u_selection_color"].value = (1.0, 0.55, 0.0)
            self.render_program["u_light_direction"].value = (0.3, -0.4, 0.85)
            self.render_program["u_has_scalars"].value = chunk.has_scalars
            self.render_program["u_node_scalars"].value = chunk.node_scalars
            self.render_program["u_opacity"].value = group.opacity
            self.render_program["u_scalar_range"].value = chunk.scalar_range
            if chunk.mesh.triangle_count:
                chunk.render_vao.render(mode=moderngl.TRIANGLES)
                self.draw_calls += 1
            if chunk.line_render_vao is not None:
                self.ctx.disable(moderngl.CULL_FACE)
                self._uniforms(self.line_program, chunk, camera, viewport, section_plane)
                self.line_program["u_color"].value = _rgb(group.line_color)
                self.line_program["u_half_width"].value = group.line_width * 0.5
                chunk.line_render_vao.render(
                    mode=moderngl.TRIANGLE_STRIP,
                    vertices=4,
                    instances=len(chunk.mesh.lines),
                )
                self.draw_calls += 1
            if chunk.point_render_vao is not None:
                self.ctx.disable(moderngl.CULL_FACE)
                self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
                self._uniforms(self.point_program, chunk, camera, viewport, section_plane)
                self.point_program["u_color"].value = _rgb(group.point_color)
                self.point_program["u_point_size"].value = group.point_size
                chunk.point_render_vao.render(mode=moderngl.POINTS)
                self.draw_calls += 1
            self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.depth_mask = True
        self.ctx.disable(moderngl.BLEND)

    @staticmethod
    def _chunk_visible(
        chunk: _Chunk,
        camera: Camera3D,
        viewport: tuple[int, int],
    ) -> bool:
        transform = chunk.handle.transform
        center = transform[:3, :3] @ chunk.origin + transform[:3, 3]
        relative = center - np.asarray(camera.position.to_tuple(), dtype=np.float64)
        right, up, forward = camera.basis()
        x = float(np.dot(relative, right.to_tuple()))
        y = float(np.dot(relative, up.to_tuple()))
        depth = float(np.dot(relative, forward.to_tuple()))
        radius = chunk.radius * float(np.linalg.norm(transform[:3, :3], ord=2))
        if depth + radius < camera.near or depth - radius > camera.far:
            return False
        tangent = math.tan(camera.fov * 0.5)
        visible_depth = max(float(camera.near), depth)
        half_height = visible_depth * tangent
        half_width = half_height * viewport[0] / max(1.0, float(viewport[1]))
        return abs(x) <= half_width + radius and abs(y) <= half_height + radius

    def _ensure_pick_target(self, viewport: tuple[int, int]) -> None:
        if viewport == self._pick_size:
            return
        for resource in (self._pick_fbo, self._pick_texture, self._pick_depth):
            if resource is not None:
                resource.release()
        self._pick_texture = self.ctx.texture(viewport, 1, dtype="u4")
        self._pick_depth = self.ctx.depth_renderbuffer(viewport)
        self._pick_fbo = self.ctx.framebuffer(
            color_attachments=[self._pick_texture], depth_attachment=self._pick_depth
        )
        self._pick_size = viewport
        self.pick_dirty = True

    def render_pick(
        self,
        camera: Camera3D,
        viewport: tuple[int, int],
        section_plane: Optional[SectionPlane] = None,
    ) -> None:
        self._ensure_pick_target(viewport)
        self._pick_fbo.use()
        self.ctx.viewport = (0, 0, *viewport)
        self._pick_fbo.clear(0, 0, 0, 0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self._pick_ranges.clear()
        base = 0
        for group in list(self._groups.values()):
            self._sync(group)
        for group in self._groups.values():
            if group.handle.removed or not group.handle.visible:
                continue
            for _chunk_id, chunk in group.chunks:
                self._uniforms(self.pick_program, chunk, camera, viewport, section_plane)
                chunk.bind_fields(self.pick_program)
                self.pick_program["u_pick_base"].value = base
                if chunk.mesh.triangle_count:
                    chunk.pick_vao.render(mode=moderngl.TRIANGLES)
                stop = base + chunk.mesh.triangle_count
                if stop > base:
                    self._pick_ranges.append((base, stop, group.handle, "triangle"))
                base = stop
                if chunk.line_pick_vao is not None:
                    self.ctx.disable(moderngl.CULL_FACE)
                    self._uniforms(
                        self.line_pick_program, chunk, camera, viewport, section_plane
                    )
                    self.line_pick_program["u_half_width"].value = max(
                        2.5, group.line_width * 0.5
                    )
                    self.line_pick_program["u_pick_base"].value = base
                    chunk.line_pick_vao.render(
                        mode=moderngl.TRIANGLE_STRIP,
                        vertices=4,
                        instances=len(chunk.mesh.lines),
                    )
                    stop = base + len(chunk.mesh.lines)
                    self._pick_ranges.append((base, stop, group.handle, "line"))
                    base = stop
                if chunk.point_pick_vao is not None:
                    self.ctx.disable(moderngl.CULL_FACE)
                    self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
                    self._uniforms(
                        self.point_pick_program, chunk, camera, viewport, section_plane
                    )
                    self.point_pick_program["u_point_size"].value = max(
                        7.0, group.point_size
                    )
                    self.point_pick_program["u_pick_base"].value = base
                    chunk.point_pick_vao.render(mode=moderngl.POINTS)
                    stop = base + len(chunk.mesh.point_indices)
                    self._pick_ranges.append((base, stop, group.handle, "point"))
                    base = stop
                self.ctx.enable(moderngl.CULL_FACE)
        self.pick_dirty = False

    def pick(
        self,
        x: int,
        y: int,
        camera: Camera3D,
        viewport: tuple[int, int],
        section_plane: Optional[SectionPlane] = None,
    ) -> Optional[tuple[MeshHandle, str, int]]:
        if self.pick_dirty or viewport != self._pick_size:
            self.render_pick(camera, viewport, section_plane)
        x = max(0, min(viewport[0] - 1, int(x)))
        y = max(0, min(viewport[1] - 1, int(y)))
        payload = self._pick_fbo.read(
            viewport=(x, viewport[1] - y - 1, 1, 1), components=1, dtype="u4"
        )
        value = struct.unpack("I", payload)[0]
        if value == 0:
            return None
        primitive = value - 1
        for start, stop, handle, primitive_kind in self._pick_ranges:
            if start <= primitive < stop:
                return handle, primitive_kind, primitive - start
        return None

    def release(self) -> None:
        for group in list(self._groups.values()):
            self.remove_mesh(group.handle)
        for resource in (self._pick_fbo, self._pick_texture, self._pick_depth):
            if resource is not None:
                resource.release()
        self.render_program.release()
        self.pick_program.release()
        self.line_program.release()
        self.line_pick_program.release()
        self.point_program.release()
        self.point_pick_program.release()
