"""Persistent ModernGL resources for the optional GPU backend."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Any, Optional, Sequence

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
from ..shading import Light


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
uniform vec3 u_back_color;
uniform vec3 u_invalid_color;
uniform bool u_has_back_color;
uniform bool u_has_face_colors;
uniform sampler2D u_face_colors;
uniform uint u_face_color_width;
uniform vec3 u_selection_color;
uniform vec3 u_highlight_color;
uniform vec3 u_preselection_color;
uniform float u_opacity;
uniform bool u_shading_enabled;
uniform int u_stipple_start;
uniform int u_stipple_count;
uniform int u_stipple_back_start;
uniform int u_stipple_back_count;
uniform int u_stipple_rotation;
uniform float u_light_ambient;
uniform float u_light_diffuse;
uniform float u_light_specular;
uniform float u_light_shininess;
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
uniform usampler2D u_semantic_elements;
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
    int stipple_start = gl_FrontFacing ? u_stipple_start : u_stipple_back_start;
    int stipple_count = gl_FrontFacing ? u_stipple_count : u_stipple_back_count;
    if (stipple_count < 64) {
        const int bayer[64] = int[64](
             0, 32,  8, 40,  2, 34, 10, 42,
            48, 16, 56, 24, 50, 18, 58, 26,
            12, 44,  4, 36, 14, 46,  6, 38,
            60, 28, 52, 20, 62, 30, 54, 22,
             3, 35, 11, 43,  1, 33,  9, 41,
            51, 19, 59, 27, 49, 17, 57, 25,
            15, 47,  7, 39, 13, 45,  5, 37,
            63, 31, 55, 23, 61, 29, 53, 21
        );
        ivec2 pixel = ivec2(floor(gl_FragCoord.xy));
        int threshold = bayer[(pixel.y & 7) * 8 + (pixel.x & 7)];
        threshold = (threshold - u_stipple_rotation) & 63;
        if (threshold < stipple_start ||
            threshold >= stipple_start + stipple_count) discard;
    }
    vec3 base = (u_has_back_color && !gl_FrontFacing) ? u_back_color : u_color;
    if (u_has_face_colors && (gl_FrontFacing || !u_has_back_color)) {
        base = texelFetch(
            u_face_colors, address(uint(gl_PrimitiveID), u_face_color_width), 0
        ).rgb;
    } else if (u_has_scalars) {
        float value = u_node_scalars
            ? v_node_scalar
            : texelFetch(u_element_scalars, address(element, u_element_width), 0).r;
        if (isnan(value)) base = u_invalid_color;
        else {
            float span = max(1.0e-20, u_scalar_range.y - u_scalar_range.x);
            base = texture(u_colormap, vec2(clamp((value - u_scalar_range.x) / span, 0.0, 1.0), 0.5)).rgb;
        }
    }
    vec3 dx = dFdx(v_relative);
    vec3 dy = dFdy(v_relative);
    vec3 normal = normalize(cross(dx, dy));
    if (!gl_FrontFacing) normal = -normal;
    vec3 light_direction = normalize(u_light_direction);
    float lambert = max(0.0, dot(normal, light_direction));
    float shade = u_shading_enabled
        ? u_light_ambient + u_light_diffuse * lambert
        : 1.0;
    if (u_shading_enabled && u_light_specular > 0.0 && lambert > 0.0) {
        vec3 view_direction = normalize(-v_relative);
        vec3 half_direction = normalize(light_direction + view_direction);
        shade += u_light_specular * pow(
            max(0.0, dot(normal, half_direction)), u_light_shininess
        );
    }
    vec3 shaded = shade <= 1.0
        ? base * shade
        : mix(base, vec3(1.0), clamp((shade - 1.0) / 0.5, 0.0, 1.0));
    uint semantic = texelFetch(
        u_semantic_elements, address(element, u_element_width), 0
    ).r;
    if ((semantic & 2u) != 0u)
        shaded = mix(shaded, u_preselection_color, 0.78);
    else if ((semantic & 1u) != 0u)
        shaded = mix(shaded, u_highlight_color, 0.72);
    else if (texelFetch(u_selected_elements, address(element, u_element_width), 0).r != 0u)
        shaded = mix(shaded, u_selection_color, 0.65);
    frag_color = vec4(shaded, stipple_count < 64 ? 1.0 : u_opacity);
}
"""

_PICK_FRAGMENT = """
#version 330
uniform bool u_section_enabled;
uniform vec3 u_section_normal;
uniform float u_section_relative_offset;
uniform uint u_pick_base;
uniform bool u_pick_enabled;
uniform int u_stipple_start;
uniform int u_stipple_count;
uniform int u_stipple_back_start;
uniform int u_stipple_back_count;
uniform int u_stipple_rotation;
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
    int stipple_start = gl_FrontFacing ? u_stipple_start : u_stipple_back_start;
    int stipple_count = gl_FrontFacing ? u_stipple_count : u_stipple_back_count;
    if (stipple_count < 64) {
        const int bayer[64] = int[64](
             0, 32,  8, 40,  2, 34, 10, 42,
            48, 16, 56, 24, 50, 18, 58, 26,
            12, 44,  4, 36, 14, 46,  6, 38,
            60, 28, 52, 20, 62, 30, 54, 22,
             3, 35, 11, 43,  1, 33,  9, 41,
            51, 19, 59, 27, 49, 17, 57, 25,
            15, 47,  7, 39, 13, 45,  5, 37,
            63, 31, 55, 23, 61, 29, 53, 21
        );
        ivec2 pixel = ivec2(floor(gl_FragCoord.xy));
        int threshold = bayer[(pixel.y & 7) * 8 + (pixel.x & 7)];
        threshold = (threshold - u_stipple_rotation) & 63;
        if (threshold < stipple_start ||
            threshold >= stipple_start + stipple_count) discard;
    }
    pick_id = u_pick_enabled ? u_pick_base + uint(gl_PrimitiveID) + 1u : 0u;
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
uniform float u_near;
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
    vec4 view_start = u_view_rotation * vec4(start, 1.0);
    vec4 view_end = u_view_rotation * vec4(end, 1.0);
    float start_depth = -view_start.z;
    float end_depth = -view_end.z;
    bool segment_visible = start_depth >= u_near || end_depth >= u_near;
    if (segment_visible && start_depth < u_near) {
        float amount = (u_near - start_depth) / (end_depth - start_depth);
        start = mix(start, end, amount);
        view_start = mix(view_start, view_end, amount);
    } else if (segment_visible && end_depth < u_near) {
        float amount = (u_near - end_depth) / (start_depth - end_depth);
        end = mix(end, start, amount);
        view_end = mix(view_end, view_start, amount);
    }
    vec4 clip_start = u_projection * view_start;
    vec4 clip_end = u_projection * view_end;
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
    gl_Position = segment_visible ? clip : vec4(2.0, 2.0, 2.0, 1.0);
    v_relative = at_end ? end : start;
    v_primitive = uint(gl_InstanceID);
}
"""

_LINE_FRAGMENT = """
#version 330
uniform vec3 u_color;
uniform vec3 u_highlight_color;
uniform vec3 u_preselection_color;
uniform float u_opacity;
uniform usampler2D u_semantic_lines;
uniform uint u_line_width;
uniform bool u_section_enabled;
uniform vec3 u_section_normal;
uniform float u_section_relative_offset;
in vec3 v_relative;
flat in uint v_primitive;
out vec4 frag_color;
ivec2 address(uint index, uint width) {
    return ivec2(int(index % width), int(index / width));
}
void main() {
    if (u_section_enabled && dot(u_section_normal, v_relative) < u_section_relative_offset) discard;
    vec3 color = u_color;
    uint semantic = texelFetch(
        u_semantic_lines, address(v_primitive, u_line_width), 0
    ).r;
    if ((semantic & 2u) != 0u)
        color = mix(color, u_preselection_color, 0.85);
    else if ((semantic & 1u) != 0u)
        color = mix(color, u_highlight_color, 0.80);
    frag_color = vec4(color, u_opacity);
}
"""

_LINE_PICK_FRAGMENT = """
#version 330
uniform bool u_section_enabled;
uniform vec3 u_section_normal;
uniform float u_section_relative_offset;
uniform uint u_pick_base;
uniform bool u_pick_enabled;
in vec3 v_relative;
flat in uint v_primitive;
layout(location = 0) out uint pick_id;
void main() {
    if (u_section_enabled && dot(u_section_normal, v_relative) < u_section_relative_offset) discard;
    pick_id = u_pick_enabled ? u_pick_base + v_primitive + 1u : 0u;
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
uniform vec3 u_outline_color;
uniform vec3 u_highlight_color;
uniform vec3 u_preselection_color;
uniform bool u_has_outline;
uniform float u_point_size;
uniform float u_opacity;
uniform bool u_section_enabled;
uniform vec3 u_section_normal;
uniform float u_section_relative_offset;
uniform usampler2D u_semantic_points;
uniform uint u_point_width;
in vec3 v_relative;
out vec4 frag_color;
ivec2 address(uint index, uint width) {
    return ivec2(int(index % width), int(index / width));
}
void main() {
    vec2 delta = gl_PointCoord * 2.0 - 1.0;
    float radial = length(delta);
    if (radial > 1.0) discard;
    if (u_section_enabled && dot(u_section_normal, v_relative) < u_section_relative_offset) discard;
    float border = 1.0 - min(0.75, 2.0 / max(1.0, u_point_size));
    vec3 color = (u_has_outline && radial >= border) ? u_outline_color : u_color;
    uint semantic = texelFetch(
        u_semantic_points, address(uint(gl_PrimitiveID), u_point_width), 0
    ).r;
    if ((semantic & 2u) != 0u)
        color = mix(color, u_preselection_color, 0.90);
    else if ((semantic & 1u) != 0u)
        color = mix(color, u_highlight_color, 0.85);
    frag_color = vec4(color, u_opacity);
}
"""

_POINT_PICK_FRAGMENT = """
#version 330
uniform bool u_section_enabled;
uniform vec3 u_section_normal;
uniform float u_section_relative_offset;
uniform uint u_pick_base;
uniform bool u_pick_enabled;
in vec3 v_relative;
layout(location = 0) out uint pick_id;
void main() {
    vec2 delta = gl_PointCoord * 2.0 - 1.0;
    if (dot(delta, delta) > 1.0) discard;
    if (u_section_enabled && dot(u_section_normal, v_relative) < u_section_relative_offset) discard;
    pick_id = u_pick_enabled ? u_pick_base + uint(gl_PrimitiveID) + 1u : 0u;
}
"""


def _matrix_bytes(matrix: np.ndarray) -> bytes:
    return np.asarray(matrix, dtype=np.float32).T.tobytes()


@lru_cache(maxsize=256)
def _rgb(color: str) -> tuple[float, float, float]:
    parsed = parse_color(color) or (154, 167, 180)
    return tuple(channel / 255.0 for channel in parsed)


def _stipple_window(value: object, phase: int = 0) -> tuple[int, int, int]:
    """Map Tk stipple names/generated paths to a renderer-neutral Bayer window."""

    specification = str(value or "")
    if not specification:
        return 0, 64, 0
    builtin = {
        "gray12": 8,
        "gray25": 16,
        "gray50": 32,
        "gray75": 48,
    }
    count = builtin.get(specification.casefold())
    start = 0
    rotation = (int(phase) % 4) * 16
    generated = re.search(
        r"anytk3d_(\d{2})_(\d{2})_(\d{2})", specification.casefold()
    )
    if generated is not None:
        start, count, rotation = (int(value) for value in generated.groups())
    elif count is None:
        count = 32
    start = max(0, min(63, int(start)))
    count = max(1, min(64 - start, int(count)))
    return start, count, int(rotation) & 63


def _stipple_windows(
    value: object, opacity: float, phase: int = 0
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return front/back Bayer windows with Tk-compatible shell semantics.

    An explicit stipple is honored verbatim on both sides.  Opacity-generated
    stippling instead allocates the far face a non-overlapping window in the
    uncovered cells, so a near shell cannot mask its own far wall.
    """

    if str(value or ""):
        window = _stipple_window(value, phase)
        return window, window
    coverage = max(0.0, min(1.0, float(opacity)))
    rotation = (int(phase) % 4) * 16
    if coverage >= 0.97:
        window = (0, 64, rotation)
        return window, window
    front_count = max(1, min(63, int(round(coverage * 64))))
    remaining = 64 - front_count
    back_count = max(1, min(remaining, int(round(coverage * remaining))))
    return (0, front_count, rotation), (front_count, back_count, rotation)


_DEFAULT_LIGHT = Light()
_MISSING = object()
_ALL_PICK_KINDS = frozenset(("triangle", "line", "point"))


def _packed_texture(ctx: Any, values: np.ndarray, dtype: str) -> tuple[Any, int]:
    flat = np.ascontiguousarray(values).reshape(-1)
    width = min(4096, max(1, len(flat)))
    height = max(1, math.ceil(len(flat) / width))
    padded = np.zeros(width * height, dtype=flat.dtype)
    padded[: len(flat)] = flat
    texture = ctx.texture((width, height), 1, padded.tobytes(), dtype=dtype)
    texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
    return texture, width


def _normalized_indices(values: Sequence[int]) -> np.ndarray:
    """Return compact, sorted non-negative primitive indices."""

    indices = np.asarray(values, dtype=np.int64).reshape(-1)
    return np.unique(indices[indices >= 0]).astype(np.uint32, copy=False)


def _semantic_mask_payload(
    count: int,
    width: int,
    highlighted: np.ndarray,
    preselected: np.ndarray,
) -> bytes:
    """Pack highlight/preselection into one byte per GPU primitive.

    Bit zero is persistent semantic highlight and bit one is transient
    preselection.  The dense byte texture lets a changed semantic set be one
    bounded buffer upload, independent of camera movement.
    """

    values = np.zeros(max(0, int(count)), dtype=np.uint8)
    if len(values):
        highlighted_valid = highlighted[highlighted < len(values)]
        preselected_valid = preselected[preselected < len(values)]
        values[highlighted_valid.astype(np.intp, copy=False)] |= np.uint8(1)
        values[preselected_valid.astype(np.intp, copy=False)] |= np.uint8(2)
    padded = np.zeros(
        max(1, int(width) * math.ceil(len(values) / max(1, int(width)))),
        dtype=np.uint8,
    )
    padded[: len(values)] = values
    return padded.tobytes()


def _color_texture(ctx: Any, colors: list[str]) -> tuple[Any, int]:
    width = min(4096, max(1, len(colors)))
    height = max(1, math.ceil(len(colors) / width))
    padded = np.zeros((width * height, 3), dtype=np.float32)
    if colors:
        padded[: len(colors)] = np.asarray([_rgb(color) for color in colors], np.float32)
    texture = ctx.texture((width, height), 3, padded.tobytes(), dtype="f4")
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
    semantic_elements: Any
    semantic_lines: Any
    line_semantic_width: int
    semantic_points: Any
    point_semantic_width: int
    scalars: Any
    colormap: Any
    face_colors: Any
    face_color_width: int
    has_face_colors: bool
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
        face_colors: Optional[tuple[str, ...]] = None,
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
        semantic_elements, _ = _packed_texture(
            ctx, np.zeros(mesh.element_count, dtype=np.uint8), "u1"
        )
        semantic_lines, line_semantic_width = _packed_texture(
            ctx,
            np.zeros(0 if mesh.lines is None else len(mesh.lines), dtype=np.uint8),
            "u1",
        )
        semantic_points, point_semantic_width = _packed_texture(
            ctx,
            np.zeros(
                0 if mesh.point_indices is None else len(mesh.point_indices),
                dtype=np.uint8,
            ),
            "u1",
        )
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
        resolved_face_colors: list[str] = []
        if face_colors:
            if len(face_colors) == mesh.triangle_count:
                resolved_face_colors = list(face_colors)
            elif len(face_colors) == mesh.element_count:
                resolved_face_colors = [str(face_colors[int(element)]) for element in mapping]
            else:
                raise ValueError("face_colors must match triangle or element count")
        face_color_texture, face_color_width = _color_texture(ctx, resolved_face_colors)
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
            semantic_elements,
            semantic_lines,
            line_semantic_width,
            semantic_points,
            point_semantic_width,
            scalars,
            colormap,
            face_color_texture,
            face_color_width,
            bool(resolved_face_colors),
            mesh.element_scalars is not None or mesh.node_scalars is not None,
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
            self.semantic_elements,
            self.semantic_lines,
            self.semantic_points,
            self.scalars,
            self.colormap,
            self.face_colors,
        ):
            if resource is not None:
                resource.release()

    def bind_fields(self, program: Any, set_uniform: Any = None) -> None:
        self.triangle_map.use(location=0)
        self.active.use(location=1)
        self.selected.use(location=2)
        self.scalars.use(location=3)
        self.colormap.use(location=4)
        self.semantic_elements.use(location=6)
        if self.has_face_colors and "u_face_colors" in program:
            self.face_colors.use(location=5)
        for name, value in (
            ("u_triangle_to_element", 0),
            ("u_active_elements", 1),
            ("u_selected_elements", 2),
            ("u_element_scalars", 3),
            ("u_colormap", 4),
            ("u_face_colors", 5),
            ("u_semantic_elements", 6),
            ("u_face_color_width", self.face_color_width),
            ("u_triangle_width", self.triangle_width),
            ("u_element_width", self.element_width),
        ):
            if name in program:
                if set_uniform is None:
                    program[name].value = value
                else:
                    set_uniform(program, name, value)

    def bind_line_semantics(self, program: Any, set_uniform: Any = None) -> None:
        self.semantic_lines.use(location=6)
        for name, value in (
            ("u_semantic_lines", 6),
            ("u_line_width", self.line_semantic_width),
        ):
            if name in program:
                if set_uniform is None:
                    program[name].value = value
                else:
                    set_uniform(program, name, value)

    def bind_point_semantics(self, program: Any, set_uniform: Any = None) -> None:
        self.semantic_points.use(location=6)
        for name, value in (
            ("u_semantic_points", 6),
            ("u_point_width", self.point_semantic_width),
        ):
            if name in program:
                if set_uniform is None:
                    program[name].value = value
                else:
                    set_uniform(program, name, value)


@dataclass(slots=True)
class _Group:
    handle: MeshHandle
    color: str
    line_color: str
    point_color: str
    point_outline: str
    line_width: float
    point_size: float
    opacity: float
    cull_backface: bool
    line_overlay: bool
    point_overlay: bool
    lit: bool
    stipple: str
    stipple_phase: int
    pickable: bool
    depth_only: bool
    layer: int
    mesh_lines: bool
    face_colors: Optional[tuple[str, ...]]
    back_color: str
    invalid_color: str
    scalar_range: Optional[tuple[float, float]]
    highlighted_elements: np.ndarray
    preselected_elements: np.ndarray
    highlighted_lines: np.ndarray
    preselected_lines: np.ndarray
    highlighted_points: np.ndarray
    preselected_points: np.ndarray
    chunk_pickable: set[object]
    chunk_semantic_masks: dict[object, dict[str, np.ndarray]]
    generations: DirtyGenerations
    chunks: list[tuple[object, _Chunk]]


class ModernGLRenderer:
    """Demand-driven renderer with one persistent resource group per handle."""

    def __init__(self, context: Any) -> None:
        self.ctx = context
        if int(context.version_code) < 330:
            raise RuntimeError("ANY3dView requires OpenGL 3.3 or newer")
        self.ctx.depth_func = "<="
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
        self._empty_face_color = context.texture(
            (1, 1), 3, np.zeros(3, dtype=np.float32).tobytes(), dtype="f4"
        )
        self._empty_face_color.use(location=5)
        self._groups: dict[int, _Group] = {}
        self._pick_fbo = None
        self._pick_texture = None
        self._pick_depth = None
        self._pick_size = (0, 0)
        self._pick_show_mesh_lines = True
        self._pick_occlude_lines = True
        self._pick_primitive_kinds = _ALL_PICK_KINDS
        self._pick_ranges: list[
            tuple[int, int, MeshHandle, str, object]
        ] = []
        self._uniform_value_cache: dict[tuple[int, str], object] = {}
        self.pick_dirty = True
        self.draw_calls = 0
        self.frame_count = 0
        self.geometry_uploads = 0
        self.semantic_buffer_updates = 0

    def _set_uniform_value(self, program: Any, name: str, value: object) -> None:
        """Avoid repeating unchanged scalar/vector uniform calls each frame."""

        key = (id(program), name)
        if self._uniform_value_cache.get(key, _MISSING) == value:
            return
        program[name].value = value
        self._uniform_value_cache[key] = value

    def add_mesh(
        self,
        handle: MeshHandle,
        color: str = "#9aa7b4",
        *,
        line_color: str = "#334155",
        point_color: str = "#334155",
        point_outline: str = "",
        line_width: float = 1.5,
        point_size: float = 7.0,
        opacity: float = 1.0,
        cull_backface: bool = True,
        line_overlay: bool = False,
        point_overlay: bool = False,
        lit: bool = True,
        stipple: object = "",
        stipple_phase: int = 0,
        pickable: bool = True,
        depth_only: bool = False,
        layer: int = 5,
        mesh_lines: bool = False,
        face_colors: Optional[Sequence[str]] = None,
        back_color: str = "",
        invalid_color: str = "#808080",
        scalar_range: Optional[tuple[float, float]] = None,
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
                    tuple(str(value) for value in face_colors)
                    if face_colors is not None and chunk_id is None else None,
                ),
            )
            for chunk_id, mesh in [(None, handle.mesh), *handle.chunks]
        ]
        self._groups[id(handle)] = _Group(
            handle,
            color,
            line_color,
            point_color,
            str(point_outline),
            max(0.5, float(line_width)),
            max(1.0, float(point_size)),
            min(1.0, max(0.0, float(opacity))),
            bool(cull_backface),
            bool(line_overlay),
            bool(point_overlay),
            bool(lit),
            str(stipple or ""),
            int(stipple_phase),
            bool(pickable),
            bool(depth_only),
            int(layer),
            bool(mesh_lines),
            None if face_colors is None else tuple(str(value) for value in face_colors),
            str(back_color),
            str(invalid_color),
            None if scalar_range is None else (float(scalar_range[0]), float(scalar_range[1])),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
            set(),
            {},
            handle.generations,
            chunks,
        )
        # Primary semantic/application element indices do not name unrelated
        # incremental chunks until a chunk-local ownership contract exists.
        self._write_selection(self._groups[id(handle)])
        self.geometry_uploads += len(chunks)
        self.pick_dirty = True

    def remove_mesh(self, handle: MeshHandle) -> None:
        group = self._groups.pop(id(handle), None)
        if group is not None:
            for _chunk_id, chunk in group.chunks:
                chunk.release()
            self.pick_dirty = True

    @staticmethod
    def _write_selection(group: _Group) -> None:
        for chunk_id, chunk in group.chunks:
            values = np.zeros(chunk.mesh.element_count, dtype=np.uint32)
            application = (
                group.handle.selected_elements.astype(np.int64, copy=False)
                if chunk_id is None
                else np.empty(0, dtype=np.int64)
            )
            application = application[
                (application >= 0) & (application < len(values))
            ]
            if len(application):
                values[application] = 1
            padded = np.zeros(
                max(1, chunk.element_width * math.ceil(len(values) / chunk.element_width)),
                np.uint32,
            )
            padded[: len(values)] = values
            chunk.selected.write(padded.tobytes())

    def set_highlighted_elements(
        self, handle: MeshHandle, elements: Sequence[int]
    ) -> None:
        """Compatibility shorthand for persistent triangle-element highlights."""

        self._set_semantic_masks_partial(handle, highlighted_elements=elements)

    def set_preselected_elements(
        self, handle: MeshHandle, elements: Sequence[int]
    ) -> None:
        self._set_semantic_masks_partial(handle, preselected_elements=elements)

    def set_highlighted_lines(
        self, handle: MeshHandle, lines: Sequence[int]
    ) -> None:
        self._set_semantic_masks_partial(handle, highlighted_lines=lines)

    def set_preselected_lines(
        self, handle: MeshHandle, lines: Sequence[int]
    ) -> None:
        self._set_semantic_masks_partial(handle, preselected_lines=lines)

    def set_highlighted_points(
        self, handle: MeshHandle, points: Sequence[int]
    ) -> None:
        self._set_semantic_masks_partial(handle, highlighted_points=points)

    def set_preselected_points(
        self, handle: MeshHandle, points: Sequence[int]
    ) -> None:
        self._set_semantic_masks_partial(handle, preselected_points=points)

    def clear_semantic_masks(self, handle: MeshHandle) -> None:
        """Clear all persistent and transient semantic masks for *handle*."""

        self.set_semantic_masks(handle)

    def set_chunk_semantic_masks(
        self,
        handle: MeshHandle,
        chunk_id: object,
        *,
        selected_elements: Sequence[int] = (),
        preselected_elements: Sequence[int] = (),
        selected_lines: Sequence[int] = (),
        preselected_lines: Sequence[int] = (),
        selected_points: Sequence[int] = (),
        preselected_points: Sequence[int] = (),
    ) -> None:
        """Replace semantic masks belonging to one retained chunk.

        The handle is authoritative for chunk existence.  Ownership normally
        supplies the local indices; a compatibility layer may instead derive
        them from a stable handle tag.  Masks are stored even when the renderer
        has not synchronized a newly added or replaced chunk yet; the next
        topology sync creates its textures and reapplies the stored state.
        """

        group = self._groups.get(id(handle))
        if group is None:
            return
        records = {
            record_id: (mesh, owners, resolver)
            for record_id, mesh, owners, resolver in handle.chunk_records
        }
        if chunk_id not in records:
            raise KeyError(chunk_id)
        state = group.chunk_semantic_masks.setdefault(
            chunk_id, self._empty_semantic_state()
        )
        specifications = (
            ("highlighted_elements", selected_elements, "element"),
            ("preselected_elements", preselected_elements, "element"),
            ("highlighted_lines", selected_lines, "line"),
            ("preselected_lines", preselected_lines, "line"),
            ("highlighted_points", selected_points, "point"),
            ("preselected_points", preselected_points, "point"),
        )
        changed: set[str] = set()
        for attribute, supplied, kind in specifications:
            values = _normalized_indices(supplied)
            if np.array_equal(values, state[attribute]):
                continue
            state[attribute] = values
            changed.add(kind)
        for kind in changed:
            self._write_chunk_semantic_mask(group, chunk_id, kind)

    def set_chunk_pickable(
        self, handle: MeshHandle, chunk_id: object, pickable: bool = True
    ) -> None:
        """Enable ID picking for an explicitly resolvable retained chunk.

        Packed chunk ownership enables this automatically.  The compatibility
        widget also uses it when a chunk inherits a stable handle-level tag.
        """

        group = self._groups.get(id(handle))
        if group is None:
            return
        if chunk_id not in {record[0] for record in handle.chunk_records}:
            raise KeyError(chunk_id)
        changed = False
        if pickable and chunk_id not in group.chunk_pickable:
            group.chunk_pickable.add(chunk_id)
            changed = True
        elif not pickable and chunk_id in group.chunk_pickable:
            group.chunk_pickable.remove(chunk_id)
            changed = True
        if changed:
            self.pick_dirty = True

    def clear_chunk_semantic_masks(
        self, handle: MeshHandle, chunk_id: object
    ) -> None:
        """Clear and forget one chunk's renderer-owned semantic masks."""

        group = self._groups.get(id(handle))
        if group is None:
            return
        state = group.chunk_semantic_masks.get(chunk_id)
        if state is None:
            return
        group.chunk_semantic_masks[chunk_id] = self._empty_semantic_state()
        for kind in ("element", "line", "point"):
            self._write_chunk_semantic_mask(group, chunk_id, kind)
        del group.chunk_semantic_masks[chunk_id]

    @staticmethod
    def _empty_semantic_state() -> dict[str, np.ndarray]:
        return {
            name: np.empty(0, dtype=np.uint32)
            for name in (
                "highlighted_elements",
                "preselected_elements",
                "highlighted_lines",
                "preselected_lines",
                "highlighted_points",
                "preselected_points",
            )
        }

    def set_semantic_masks(
        self,
        handle: MeshHandle,
        *,
        selected_elements: Sequence[int] = (),
        preselected_elements: Sequence[int] = (),
        selected_lines: Sequence[int] = (),
        preselected_lines: Sequence[int] = (),
        selected_points: Sequence[int] = (),
        preselected_points: Sequence[int] = (),
    ) -> None:
        """Replace every semantic GPU mask for *handle* in three writes.

        ``selected_*`` is persistent semantic highlight state; it remains
        independent of :attr:`MeshHandle.selected_elements`.  ``preselected_*``
        is transient hover state and wins visually when the two overlap.
        Triangle indices name elements, while line and point indices name
        display primitives directly.
        """

        self._set_semantic_masks_partial(
            handle,
            highlighted_elements=selected_elements,
            preselected_elements=preselected_elements,
            highlighted_lines=selected_lines,
            preselected_lines=preselected_lines,
            highlighted_points=selected_points,
            preselected_points=preselected_points,
        )

    def _set_semantic_masks_partial(
        self,
        handle: MeshHandle,
        *,
        highlighted_elements: Any = _MISSING,
        preselected_elements: Any = _MISSING,
        highlighted_lines: Any = _MISSING,
        preselected_lines: Any = _MISSING,
        highlighted_points: Any = _MISSING,
        preselected_points: Any = _MISSING,
    ) -> None:
        """Update supplied semantic GPU masks with at most one write per kind.

        Omitted keyword arguments retain their current values.  Pass an empty
        sequence to clear a mask.  Triangle indices name elements (and thus
        cover all mapped triangles); line and point indices name display
        primitives directly.
        """

        group = self._groups.get(id(handle))
        if group is None:
            return
        specifications = (
            ("highlighted_elements", highlighted_elements, "element"),
            ("preselected_elements", preselected_elements, "element"),
            ("highlighted_lines", highlighted_lines, "line"),
            ("preselected_lines", preselected_lines, "line"),
            ("highlighted_points", highlighted_points, "point"),
            ("preselected_points", preselected_points, "point"),
        )
        changed: set[str] = set()
        for attribute, supplied, kind in specifications:
            if supplied is _MISSING:
                continue
            values = _normalized_indices(supplied)
            if np.array_equal(values, getattr(group, attribute)):
                continue
            setattr(group, attribute, values)
            changed.add(kind)
        for kind in changed:
            self._write_semantic_mask(group, kind)

    def _write_semantic_mask(self, group: _Group, kind: str) -> None:
        if kind == "element":
            highlighted = group.highlighted_elements
            preselected = group.preselected_elements
        elif kind == "line":
            highlighted = group.highlighted_lines
            preselected = group.preselected_lines
        elif kind == "point":
            highlighted = group.highlighted_points
            preselected = group.preselected_points
        else:  # pragma: no cover - internal invariant
            raise ValueError(f"unknown semantic primitive kind: {kind}")

        chunk = next(
            (value for value_id, value in group.chunks if value_id is None),
            None,
        )
        if chunk is not None:
            self._write_chunk_mask_texture(chunk, kind, highlighted, preselected)

    def _write_chunk_semantic_mask(
        self, group: _Group, chunk_id: object, kind: str
    ) -> None:
        chunk = next(
            (value for value_id, value in group.chunks if value_id == chunk_id),
            None,
        )
        if chunk is None:
            return
        state = group.chunk_semantic_masks.get(chunk_id)
        empty = np.empty(0, dtype=np.uint32)
        highlighted = (
            empty if state is None else state[f"highlighted_{kind}s"]
        )
        preselected = (
            empty if state is None else state[f"preselected_{kind}s"]
        )
        self._write_chunk_mask_texture(
            chunk, kind, highlighted, preselected
        )

    def _write_chunk_mask_texture(
        self,
        chunk: _Chunk,
        kind: str,
        highlighted: np.ndarray,
        preselected: np.ndarray,
    ) -> None:
        if kind == "element":
            count = chunk.mesh.element_count
            width = chunk.element_width
            texture = chunk.semantic_elements
        elif kind == "line":
            count = 0 if chunk.mesh.lines is None else len(chunk.mesh.lines)
            width = chunk.line_semantic_width
            texture = chunk.semantic_lines
        elif kind == "point":
            count = (
                0 if chunk.mesh.point_indices is None else len(chunk.mesh.point_indices)
            )
            width = chunk.point_semantic_width
            texture = chunk.semantic_points
        else:  # pragma: no cover - internal invariant
            raise ValueError(f"unknown semantic primitive kind: {kind}")
        texture.write(
            _semantic_mask_payload(count, width, highlighted, preselected)
        )
        self.semantic_buffer_updates += 1

    def _sync(self, group: _Group) -> None:
        current = group.handle.generations
        previous = group.generations
        if current.topology != previous.topology or current.position != previous.position:
            semantic_masks = {
                "selected_elements": group.highlighted_elements.copy(),
                "preselected_elements": group.preselected_elements.copy(),
                "selected_lines": group.highlighted_lines.copy(),
                "preselected_lines": group.preselected_lines.copy(),
                "selected_points": group.highlighted_points.copy(),
                "preselected_points": group.preselected_points.copy(),
            }
            chunk_semantic_masks = {
                chunk_id: {
                    name: values.copy() for name, values in state.items()
                }
                for chunk_id, state in group.chunk_semantic_masks.items()
            }
            chunk_pickable = set(group.chunk_pickable)
            self.add_mesh(
                group.handle,
                group.color,
                line_color=group.line_color,
                point_color=group.point_color,
                point_outline=group.point_outline,
                line_width=group.line_width,
                point_size=group.point_size,
                opacity=group.opacity,
                cull_backface=group.cull_backface,
                line_overlay=group.line_overlay,
                point_overlay=group.point_overlay,
                lit=group.lit,
                stipple=group.stipple,
                stipple_phase=group.stipple_phase,
                pickable=group.pickable,
                depth_only=group.depth_only,
                layer=group.layer,
                mesh_lines=group.mesh_lines,
                face_colors=group.face_colors,
                back_color=group.back_color,
                invalid_color=group.invalid_color,
                scalar_range=group.scalar_range,
            )
            self.set_semantic_masks(group.handle, **semantic_masks)
            current_chunk_ids = {
                chunk_id for chunk_id, *_rest in group.handle.chunk_records
            }
            for chunk_id in chunk_pickable & current_chunk_ids:
                self.set_chunk_pickable(group.handle, chunk_id)
            for chunk_id, state in chunk_semantic_masks.items():
                if chunk_id not in current_chunk_ids:
                    continue
                self.set_chunk_semantic_masks(
                    group.handle,
                    chunk_id,
                    selected_elements=state["highlighted_elements"],
                    preselected_elements=state["preselected_elements"],
                    selected_lines=state["highlighted_lines"],
                    preselected_lines=state["preselected_lines"],
                    selected_points=state["highlighted_points"],
                    preselected_points=state["preselected_points"],
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
            self._write_selection(group)
            owned_chunk_ids = {
                chunk_id
                for chunk_id, _mesh, owners, _resolver in group.handle.chunk_records
                if owners is not None
            }
            for chunk_id in tuple(group.chunk_semantic_masks):
                if chunk_id not in owned_chunk_ids:
                    self.clear_chunk_semantic_masks(group.handle, chunk_id)
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
        if "u_near" in program:
            program["u_near"].value = float(camera.near)
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
        light: Optional[Light] = None,
        shading_enabled: bool = True,
        occlude_lines: bool = True,
        show_mesh_lines: bool = True,
        selection_color: str = "#ff8c00",
        preselection_color: str = "#ffd166",
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
            key=lambda item: (
                item[0].layer,
                -float(
                    np.linalg.norm(
                        item[0].handle.transform[:3, :3] @ item[1].origin
                        + item[0].handle.transform[:3, 3]
                        - camera_position
                    )
                ),
            ),
        )
        ordered = sorted(
            (item for item in draw_items if item[0].opacity >= 0.999),
            key=lambda item: item[0].layer,
        ) + transparent
        for group, chunk in ordered:
            is_transparent = group.opacity < 0.999
            if is_transparent:
                self.ctx.enable(moderngl.BLEND)
                self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                self.ctx.depth_mask = False
            else:
                self.ctx.disable(moderngl.BLEND)
                self.ctx.depth_mask = True
            if not group.cull_backface:
                self.ctx.disable(moderngl.CULL_FACE)
            if group.depth_only:
                self.ctx.color_mask = (False, False, False, False)
            self._uniforms(self.render_program, chunk, camera, viewport, section_plane)
            chunk.bind_fields(self.render_program, self._set_uniform_value)
            self._set_uniform_value(self.render_program, "u_color", _rgb(chunk.color))
            self._set_uniform_value(
                self.render_program,
                "u_back_color",
                _rgb(group.back_color or chunk.color),
            )
            self._set_uniform_value(
                self.render_program, "u_invalid_color", _rgb(group.invalid_color)
            )
            self._set_uniform_value(
                self.render_program, "u_has_back_color", bool(group.back_color)
            )
            self._set_uniform_value(
                self.render_program, "u_has_face_colors", chunk.has_face_colors
            )
            self._set_uniform_value(
                self.render_program, "u_selection_color", _rgb(selection_color)
            )
            self._set_uniform_value(
                self.render_program, "u_highlight_color", _rgb(selection_color)
            )
            self._set_uniform_value(
                self.render_program,
                "u_preselection_color",
                _rgb(preselection_color),
            )
            configured_light = light if light is not None else _DEFAULT_LIGHT
            if configured_light.follow_camera:
                light_direction = tuple(
                    float(value)
                    for value in configured_light.world_direction(camera.basis())
                )
            else:
                direction = configured_light.direction
                light_direction = (direction.x, direction.y, direction.z)
            self._set_uniform_value(
                self.render_program, "u_light_direction", light_direction
            )
            self._set_uniform_value(
                self.render_program, "u_light_ambient", configured_light.ambient
            )
            self._set_uniform_value(
                self.render_program, "u_light_diffuse", configured_light.diffuse
            )
            self._set_uniform_value(
                self.render_program, "u_light_specular", configured_light.specular
            )
            self._set_uniform_value(
                self.render_program, "u_light_shininess", configured_light.shininess
            )
            self._set_uniform_value(
                self.render_program,
                "u_shading_enabled",
                bool(shading_enabled and group.lit and configured_light.enabled),
            )
            front_stipple, back_stipple = _stipple_windows(
                group.stipple, group.opacity, group.stipple_phase
            )
            stipple_start, stipple_count, stipple_rotation = front_stipple
            back_stipple_start, back_stipple_count, _ = back_stipple
            self._set_uniform_value(
                self.render_program, "u_stipple_start", stipple_start
            )
            self._set_uniform_value(
                self.render_program, "u_stipple_count", stipple_count
            )
            self._set_uniform_value(
                self.render_program, "u_stipple_back_start", back_stipple_start
            )
            self._set_uniform_value(
                self.render_program, "u_stipple_back_count", back_stipple_count
            )
            self._set_uniform_value(
                self.render_program, "u_stipple_rotation", stipple_rotation
            )
            self._set_uniform_value(
                self.render_program, "u_has_scalars", chunk.has_scalars
            )
            self._set_uniform_value(
                self.render_program, "u_node_scalars", chunk.node_scalars
            )
            self._set_uniform_value(
                self.render_program, "u_opacity", group.opacity
            )
            self._set_uniform_value(
                self.render_program,
                "u_scalar_range",
                group.scalar_range or chunk.scalar_range,
            )
            if chunk.mesh.triangle_count:
                chunk.render_vao.render(mode=moderngl.TRIANGLES)
                self.draw_calls += 1
            line_without_depth = False
            if (
                not group.depth_only
                and (show_mesh_lines or not group.mesh_lines)
                and chunk.line_render_vao is not None
            ):
                self.ctx.disable(moderngl.CULL_FACE)
                line_without_depth = group.line_overlay or not occlude_lines
                if line_without_depth:
                    self.ctx.disable(moderngl.DEPTH_TEST)
                self._uniforms(self.line_program, chunk, camera, viewport, section_plane)
                chunk.bind_line_semantics(
                    self.line_program, self._set_uniform_value
                )
                self._set_uniform_value(
                    self.line_program, "u_color", _rgb(group.line_color)
                )
                self._set_uniform_value(
                    self.line_program,
                    "u_highlight_color",
                    _rgb(selection_color),
                )
                self._set_uniform_value(
                    self.line_program,
                    "u_preselection_color",
                    _rgb(preselection_color),
                )
                self._set_uniform_value(
                    self.line_program, "u_opacity", group.opacity
                )
                self._set_uniform_value(
                    self.line_program, "u_half_width", group.line_width * 0.5
                )
                chunk.line_render_vao.render(
                    mode=moderngl.TRIANGLE_STRIP,
                    vertices=4,
                    instances=len(chunk.mesh.lines),
                )
                self.draw_calls += 1
                if line_without_depth:
                    self.ctx.enable(moderngl.DEPTH_TEST)
            if not group.depth_only and chunk.point_render_vao is not None:
                self.ctx.disable(moderngl.CULL_FACE)
                self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
                point_without_depth = group.point_overlay
                if point_without_depth:
                    self.ctx.disable(moderngl.DEPTH_TEST)
                self._uniforms(self.point_program, chunk, camera, viewport, section_plane)
                chunk.bind_point_semantics(
                    self.point_program, self._set_uniform_value
                )
                self._set_uniform_value(
                    self.point_program, "u_color", _rgb(group.point_color)
                )
                self._set_uniform_value(
                    self.point_program,
                    "u_highlight_color",
                    _rgb(selection_color),
                )
                self._set_uniform_value(
                    self.point_program,
                    "u_preselection_color",
                    _rgb(preselection_color),
                )
                self._set_uniform_value(
                    self.point_program,
                    "u_outline_color",
                    _rgb(group.point_outline or group.point_color),
                )
                self._set_uniform_value(
                    self.point_program, "u_has_outline", bool(group.point_outline)
                )
                self._set_uniform_value(
                    self.point_program, "u_opacity", group.opacity
                )
                self._set_uniform_value(
                    self.point_program, "u_point_size", group.point_size
                )
                chunk.point_render_vao.render(mode=moderngl.POINTS)
                self.draw_calls += 1
                if point_without_depth:
                    self.ctx.enable(moderngl.DEPTH_TEST)
            if not group.cull_backface or chunk.line_render_vao is not None or chunk.point_render_vao is not None:
                self.ctx.enable(moderngl.CULL_FACE)
            if group.depth_only:
                self.ctx.color_mask = (True, True, True, True)
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
        show_mesh_lines: bool = True,
        occlude_lines: bool = True,
        primitive_kinds: Optional[Sequence[str]] = None,
    ) -> None:
        included_kinds = (
            _ALL_PICK_KINDS
            if primitive_kinds is None
            else frozenset(str(value) for value in primitive_kinds)
        )
        unknown = included_kinds - _ALL_PICK_KINDS
        if unknown:
            raise ValueError(
                "primitive_kinds contains unsupported values: "
                + ", ".join(sorted(unknown))
            )
        self._ensure_pick_target(viewport)
        self._pick_fbo.use()
        self.ctx.viewport = (0, 0, *viewport)
        self._pick_fbo.clear(0, 0, 0, 0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self._pick_ranges.clear()
        base = 0
        for group in list(self._groups.values()):
            self._sync(group)
        for group in sorted(self._groups.values(), key=lambda value: value.layer):
            if group.handle.removed or not group.handle.visible:
                continue
            for chunk_id, chunk in group.chunks:
                chunk_owned = chunk_id is None
                if chunk_id is not None:
                    try:
                        chunk_owners, _chunk_resolver = (
                            group.handle.chunk_ownership(chunk_id)
                        )
                    except KeyError:
                        chunk_owners = None
                    chunk_owned = (
                        chunk_owners is not None
                        or chunk_id in group.chunk_pickable
                    )
                chunk_pickable = bool(group.pickable and chunk_owned)
                triangle_pickable = bool(
                    chunk_pickable and "triangle" in included_kinds
                )
                if group.cull_backface:
                    self.ctx.enable(moderngl.CULL_FACE)
                else:
                    self.ctx.disable(moderngl.CULL_FACE)
                self._uniforms(self.pick_program, chunk, camera, viewport, section_plane)
                chunk.bind_fields(self.pick_program, self._set_uniform_value)
                self._set_uniform_value(self.pick_program, "u_pick_base", base)
                front_stipple, back_stipple = _stipple_windows(
                    group.stipple, group.opacity, group.stipple_phase
                )
                stipple_start, stipple_count, stipple_rotation = front_stipple
                back_stipple_start, back_stipple_count, _ = back_stipple
                self._set_uniform_value(
                    self.pick_program, "u_stipple_start", stipple_start
                )
                self._set_uniform_value(
                    self.pick_program, "u_stipple_count", stipple_count
                )
                self._set_uniform_value(
                    self.pick_program, "u_stipple_back_start", back_stipple_start
                )
                self._set_uniform_value(
                    self.pick_program, "u_stipple_back_count", back_stipple_count
                )
                self._set_uniform_value(
                    self.pick_program, "u_stipple_rotation", stipple_rotation
                )
                self._set_uniform_value(
                    self.pick_program, "u_pick_enabled", triangle_pickable
                )
                if chunk.mesh.triangle_count and (
                    "triangle" in included_kinds or not chunk_pickable
                ):
                    chunk.pick_vao.render(mode=moderngl.TRIANGLES)
                stop = base + (
                    chunk.mesh.triangle_count if triangle_pickable else 0
                )
                if triangle_pickable and stop > base:
                    self._pick_ranges.append(
                        (base, stop, group.handle, "triangle", chunk_id)
                    )
                if triangle_pickable:
                    base = stop
                if (
                    chunk.line_pick_vao is not None
                    and (show_mesh_lines or not group.mesh_lines)
                ):
                    line_pickable = bool(
                        chunk_pickable
                        and not group.mesh_lines
                        and "line" in included_kinds
                    )
                    self.ctx.disable(moderngl.CULL_FACE)
                    line_without_depth = group.line_overlay or not occlude_lines
                    if line_without_depth:
                        self.ctx.disable(moderngl.DEPTH_TEST)
                    self._uniforms(
                        self.line_pick_program, chunk, camera, viewport, section_plane
                    )
                    self._set_uniform_value(
                        self.line_pick_program,
                        "u_half_width",
                        max(2.5, group.line_width * 0.5),
                    )
                    self._set_uniform_value(
                        self.line_pick_program, "u_pick_base", base
                    )
                    self._set_uniform_value(
                        self.line_pick_program, "u_pick_enabled", line_pickable
                    )
                    if line_pickable:
                        chunk.line_pick_vao.render(
                            mode=moderngl.TRIANGLE_STRIP,
                            vertices=4,
                            instances=len(chunk.mesh.lines),
                        )
                    if line_without_depth:
                        self.ctx.enable(moderngl.DEPTH_TEST)
                    stop = base + len(chunk.mesh.lines)
                    if line_pickable:
                        self._pick_ranges.append(
                            (base, stop, group.handle, "line", chunk_id)
                        )
                        base = stop
                point_pickable = bool(
                    chunk_pickable and "point" in included_kinds
                )
                if chunk.point_pick_vao is not None and point_pickable:
                    self.ctx.disable(moderngl.CULL_FACE)
                    self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
                    point_without_depth = group.point_overlay
                    if point_without_depth:
                        self.ctx.disable(moderngl.DEPTH_TEST)
                    self._uniforms(
                        self.point_pick_program, chunk, camera, viewport, section_plane
                    )
                    self._set_uniform_value(
                        self.point_pick_program,
                        "u_point_size",
                        max(7.0, group.point_size),
                    )
                    self._set_uniform_value(
                        self.point_pick_program, "u_pick_base", base
                    )
                    self._set_uniform_value(
                        self.point_pick_program, "u_pick_enabled", chunk_pickable
                    )
                    chunk.point_pick_vao.render(mode=moderngl.POINTS)
                    if point_without_depth:
                        self.ctx.enable(moderngl.DEPTH_TEST)
                    stop = base + len(chunk.mesh.point_indices)
                    if chunk_pickable:
                        self._pick_ranges.append(
                            (base, stop, group.handle, "point", chunk_id)
                        )
                        base = stop
                self.ctx.enable(moderngl.CULL_FACE)
        self.pick_dirty = False
        self._pick_show_mesh_lines = bool(show_mesh_lines)
        self._pick_occlude_lines = bool(occlude_lines)
        self._pick_primitive_kinds = included_kinds

    def pick(
        self,
        x: int,
        y: int,
        camera: Camera3D,
        viewport: tuple[int, int],
        section_plane: Optional[SectionPlane] = None,
        show_mesh_lines: bool = True,
        occlude_lines: bool = True,
        radius: int = 0,
        primitive_kinds: Optional[Sequence[str]] = None,
    ) -> Optional[tuple[MeshHandle, str, int]]:
        """Return the compatibility handle/kind/index pick triple."""

        detail = self.pick_detail(
            x,
            y,
            camera,
            viewport,
            section_plane,
            show_mesh_lines,
            occlude_lines,
            radius,
            primitive_kinds,
        )
        if detail is None:
            return None
        handle, primitive_kind, primitive, _chunk_id = detail
        return handle, primitive_kind, primitive

    def pick_detail(
        self,
        x: int,
        y: int,
        camera: Camera3D,
        viewport: tuple[int, int],
        section_plane: Optional[SectionPlane] = None,
        show_mesh_lines: bool = True,
        occlude_lines: bool = True,
        radius: int = 0,
        primitive_kinds: Optional[Sequence[str]] = None,
    ) -> Optional[tuple[MeshHandle, str, int, object]]:
        """Return a pick including its optional retained ``chunk_id``."""

        included_kinds = (
            _ALL_PICK_KINDS
            if primitive_kinds is None
            else frozenset(str(value) for value in primitive_kinds)
        )
        if (
            self.pick_dirty
            or viewport != self._pick_size
            or bool(show_mesh_lines) != self._pick_show_mesh_lines
            or bool(occlude_lines) != self._pick_occlude_lines
            or included_kinds != self._pick_primitive_kinds
        ):
            self.render_pick(
                camera,
                viewport,
                section_plane,
                show_mesh_lines=show_mesh_lines,
                occlude_lines=occlude_lines,
                primitive_kinds=tuple(included_kinds),
            )
        x = max(0, min(viewport[0] - 1, int(x)))
        y = max(0, min(viewport[1] - 1, int(y)))
        radius = max(0, int(radius))
        left = max(0, x - radius)
        right = min(viewport[0] - 1, x + radius)
        top = max(0, y - radius)
        bottom = min(viewport[1] - 1, y + radius)
        region_width = right - left + 1
        region_height = bottom - top + 1
        payload = self._pick_fbo.read(
            viewport=(
                left,
                viewport[1] - bottom - 1,
                region_width,
                region_height,
            ),
            components=1,
            dtype="u4",
        )
        values = np.frombuffer(payload, dtype=np.uint32).reshape(
            region_height, region_width
        )
        candidates = np.argwhere(values != 0)
        if not len(candidates):
            return None
        # FBO rows are bottom-up. Select the nearest covered pixel so radius
        # matches Tk semantics with one cached sub-rectangle read.
        target_row = bottom - y
        target_column = x - left
        distances = (
            (candidates[:, 0] - target_row) ** 2
            + (candidates[:, 1] - target_column) ** 2
        )
        row, column = candidates[int(np.argmin(distances))]
        value = int(values[int(row), int(column)])
        primitive = value - 1
        for start, stop, handle, primitive_kind, chunk_id in self._pick_ranges:
            if start <= primitive < stop:
                return handle, primitive_kind, primitive - start, chunk_id
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
        self._empty_face_color.release()
