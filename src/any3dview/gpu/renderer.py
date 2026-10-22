"""Persistent ModernGL resources for the optional GPU backend."""

from __future__ import annotations

from dataclasses import dataclass
import os
from functools import lru_cache, wraps
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
from ..appearance import validated_appearance,palette_data
from ..clipping import SectionPlane
from ..core import Camera3D, _interpolate_thickness_color, parse_color
from ..retained import DirtyGenerations, MeshHandle


class _ResourceStage:
    """Own successfully returned GL allocations until a build commits."""
    _factories = frozenset({"buffer","texture","vertex_array","depth_renderbuffer","framebuffer"})

    def __init__(self,ctx):
        self.ctx=ctx
        self.resources=[]

    def __getattr__(self,name):
        value=getattr(self.ctx,name)
        if name not in self._factories:return value
        def allocate(*args,**kwargs):
            resource=value(*args,**kwargs)
            self.resources.append(resource)
            return resource
        return allocate

    def release(self):
        for resource in reversed(self.resources):
            try:resource.release()
            except BaseException:pass  # preserve the primary allocation failure
        self.resources.clear()


def _release_failed_allocations(method):
    @wraps(method)
    def build(cls,ctx,*args,**kwargs):
        stage=_ResourceStage(ctx)
        try:
            result=method(cls,stage,*args,**kwargs)
            result.ctx=ctx
            stage.resources.clear()
            return result
        except BaseException:
            stage.release()
            raise
    return build
from ..shading import Light


_VERTEX = """
#version 330
uniform mat4 u_view_rotation;
uniform mat4 u_projection;
uniform mat3 u_linear;
uniform vec3 u_origin_from_camera;
uniform float u_deformation_scale;
uniform float u_depth_bias;
in vec3 in_position;
in vec3 in_displacement;
in float in_node_scalar;
in vec3 in_normal;
out vec3 v_relative;
out float v_node_scalar;
out float v_depth;
out vec3 v_normal;
void main() {
    vec3 local = in_position + u_deformation_scale * in_displacement;
    v_relative = u_linear * local + u_origin_from_camera;
    v_node_scalar = in_node_scalar;
    v_normal = u_linear * in_normal;
    v_depth = -(u_view_rotation * vec4(v_relative, 1.0)).z;
    gl_Position = u_projection * u_view_rotation * vec4(v_relative, 1.0);
    // Pull translucent surfaces marginally toward the camera so a shell lying
    // exactly on an opaque surface does not z-fight with it.
    gl_Position.z -= u_depth_bias * gl_Position.w;
}
"""

# Shared by the surface, line and point fragment shaders.  In the weighted
# blended OIT pass the colour target accumulates premultiplied colour and
# weight, and the second target accumulates -log(1 - alpha) so that
# exp(-sum) is the exact product of the layers' transmittances without a
# separate blend mode.  ``u_oit_depth_only`` replays opaque geometry for depth
# only (colour masks do not cover both attachments) by writing zeros.
_OIT_EMIT_GLSL = """
uniform bool u_oit;
uniform bool u_oit_depth_only;
uniform vec2 u_oit_depth_range;
layout(location = 0) out vec4 frag_color;
layout(location = 1) out float frag_reveal;
void emit_color(vec3 rgb, float alpha, float view_depth) {
    if (u_oit_depth_only) {
        frag_color = vec4(0.0);
        frag_reveal = 0.0;
    } else if (u_oit) {
        float span = max(1.0e-6, u_oit_depth_range.y - u_oit_depth_range.x);
        float nearness = 1.0 - clamp((view_depth - u_oit_depth_range.x) / span, 0.0, 1.0);
        float weight = 1.0 + 3.0 * nearness;
        float covered = min(alpha, 0.999);
        frag_color = vec4(rgb * covered * weight, covered * weight);
        frag_reveal = -log(1.0 - covered);
    } else {
        frag_color = vec4(rgb, alpha);
        frag_reveal = 0.0;
    }
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
uniform bool u_has_normals;
in vec3 v_relative;
in float v_node_scalar;
in float v_depth;
in vec3 v_normal;
//@OIT_EMIT@
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
    vec3 normal;
    if (u_has_normals && dot(v_normal, v_normal) > 1.0e-12) {
        // Supplied smooth normal, oriented with the triangle's winding.
        normal = normalize(v_normal);
    } else {
        vec3 dx = dFdx(v_relative);
        vec3 dy = dFdy(v_relative);
        normal = normalize(cross(dx, dy));
        if (!gl_FrontFacing) normal = -normal;
    }
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
    if ((semantic & 4u) != 0u) discard;
    if ((semantic & 2u) != 0u)
        shaded = mix(shaded, u_preselection_color, 0.78);
    else if ((semantic & 1u) != 0u)
        shaded = mix(shaded, u_highlight_color, 0.72);
    else if (texelFetch(u_selected_elements, address(element, u_element_width), 0).r != 0u)
        shaded = mix(shaded, u_selection_color, 0.65);
    float alpha = stipple_count < 64 ? 1.0 : u_opacity;
    emit_color(shaded, alpha, v_depth);
}
"""

_OIT_COMPOSITE_VERTEX = """
#version 330
void main() {
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
"""

_MSAA_RESOLVE_FRAGMENT = """
#version 330
uniform sampler2D u_color;
uniform sampler2D u_depth;
out vec4 frag_color;
void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    frag_color = texelFetch(u_color, pixel, 0);
    gl_FragDepth = texelFetch(u_depth, pixel, 0).r;
}
"""

_OIT_LAYER_FRAGMENT = """
#version 330
uniform sampler2D u_layer_accum;
uniform sampler2D u_layer_reveal;
layout(location = 0) out vec4 frag_color;
layout(location = 1) out float frag_reveal;
void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float absorbed = texelFetch(u_layer_reveal, pixel, 0).r;
    if (absorbed <= 0.0) discard;
    frag_color = texelFetch(u_layer_accum, pixel, 0);
    frag_reveal = absorbed;
}
"""

_OIT_COMPOSITE_FRAGMENT = """
#version 330
uniform sampler2D u_accum;
uniform sampler2D u_reveal;
out vec4 frag_color;
void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float absorbed = texelFetch(u_reveal, pixel, 0).r;
    float alpha = 1.0 - exp(-absorbed);
    if (alpha < 1.0e-4) discard;
    vec4 accum = texelFetch(u_accum, pixel, 0);
    frag_color = vec4(accum.rgb / max(accum.a, 1.0e-5), alpha);
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
uniform usampler2D u_semantic_elements;
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
    if ((texelFetch(u_semantic_elements, address(element, u_element_width), 0).r & 4u) != 0u) discard;
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
// 0: butt-ended quad drawn as a 4-vertex triangle strip.  k > 0: a capsule
// with round caps drawn as a triangle fan of 2 * (k + 1) + 2 vertices, which
// fills the wedge at a polyline joint and keeps the rim antialiased by MSAA.
uniform int u_cap_segments;
uniform bool u_node_scalars;
uniform sampler2D u_node_values;
uniform uint u_node_width;
in uvec2 in_endpoint_nodes;
in vec3 in_start;
in vec3 in_end;
in vec3 in_start_displacement;
in vec3 in_end_displacement;
out vec3 v_relative;
out float v_depth;
out float v_node_scalar;
flat out uint v_primitive;
ivec2 node_address(uint index) {return ivec2(int(index % u_node_width),int(index / u_node_width));}
void main() {
    float start_scalar = u_node_scalars ? texelFetch(u_node_values,node_address(in_endpoint_nodes.x),0).r : 0.0;
    float end_scalar = u_node_scalars ? texelFetch(u_node_values,node_address(in_endpoint_nodes.y),0).r : 0.0;
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
        start_scalar = mix(start_scalar,end_scalar,amount);
        start = mix(start, end, amount);
        view_start = mix(view_start, view_end, amount);
    } else if (segment_visible && end_depth < u_near) {
        float amount = (u_near - end_depth) / (start_depth - end_depth);
        end_scalar = mix(end_scalar,start_scalar,amount);
        end = mix(end, start, amount);
        view_end = mix(view_end, view_start, amount);
    }
    vec4 clip_start = u_projection * view_start;
    vec4 clip_end = u_projection * view_end;
    vec2 screen_start = clip_start.xy / clip_start.w * u_viewport;
    vec2 screen_end = clip_end.xy / clip_end.w * u_viewport;
    vec2 direction = screen_end - screen_start;
    float length_px = max(length(direction), 1.0e-6);
    vec2 unit = direction / length_px;
    vec2 normal = vec2(-unit.y, unit.x);
    vec4 clip;
    bool at_end;
    if (u_cap_segments == 0) {
        at_end = gl_VertexID >= 2;
        bool positive = gl_VertexID == 1 || gl_VertexID == 3;
        clip = at_end ? clip_end : clip_start;
        vec2 offset = normal * (positive ? u_half_width : -u_half_width);
        clip.xy += offset * (2.0 / u_viewport) * clip.w;
    } else if (gl_VertexID == 0) {
        // Fan centre: a point on the segment, inside the convex capsule.
        at_end = false;
        clip = 0.5 * (clip_start + clip_end);
    } else {
        int outline = 2 * (u_cap_segments + 1);
        int index = (gl_VertexID - 1) % outline;   // the closing vertex wraps to 0
        at_end = index > u_cap_segments;
        int cap_index = at_end ? index - (u_cap_segments + 1) : index;
        float theta = 3.14159265 * float(cap_index) / float(u_cap_segments);
        // Start cap sweeps +normal -> -direction -> -normal; the end cap
        // continues -normal -> +direction -> +normal.
        vec2 around = at_end
            ? -normal * cos(theta) + unit * sin(theta)
            : normal * cos(theta) - unit * sin(theta);
        clip = at_end ? clip_end : clip_start;
        clip.xy += around * u_half_width * (2.0 / u_viewport) * clip.w;
    }
    gl_Position = segment_visible ? clip : vec4(2.0, 2.0, 2.0, 1.0);
    bool centre = u_cap_segments != 0 && gl_VertexID == 0;
    v_relative = centre ? 0.5 * (start + end) : (at_end ? end : start);
    float depth_start = -view_start.z;
    float depth_end = -view_end.z;
    v_depth = centre ? 0.5 * (depth_start + depth_end) : (at_end ? depth_end : depth_start);
    v_node_scalar = centre ? 0.5*(start_scalar+end_scalar) : (at_end ? end_scalar : start_scalar);
    v_primitive = uint(gl_InstanceID);
}
"""

_LINE_FRAGMENT = """
#version 330
uniform vec3 u_color;
uniform vec3 u_invalid_color;
uniform vec3 u_selection_color;
uniform bool u_has_scalars;
uniform bool u_node_scalars;
uniform bool u_has_line_mapping;
uniform vec2 u_scalar_range;
uniform uint u_element_width;
uniform usampler2D u_line_to_element;
uniform usampler2D u_active_elements;
uniform usampler2D u_selected_elements;
uniform usampler2D u_semantic_elements;
uniform sampler2D u_element_scalars;
uniform sampler2D u_colormap;
uniform vec3 u_highlight_color;
uniform vec3 u_preselection_color;
uniform float u_opacity;
uniform usampler2D u_semantic_lines;
uniform uint u_line_width;
uniform bool u_section_enabled;
uniform vec3 u_section_normal;
uniform float u_section_relative_offset;
in vec3 v_relative;
in float v_depth;
in float v_node_scalar;
flat in uint v_primitive;
//@OIT_EMIT@
ivec2 address(uint index, uint width) {
    return ivec2(int(index % width), int(index / width));
}
void main() {
    if (u_section_enabled && dot(u_section_normal, v_relative) < u_section_relative_offset) discard;
    uint element = u_has_line_mapping ? texelFetch(u_line_to_element,address(v_primitive,u_line_width),0).r : 0u;
    if (u_has_line_mapping && texelFetch(u_active_elements,address(element,u_element_width),0).r == 0u) discard;
    vec3 color = u_color;
    if (u_has_scalars && (u_node_scalars || u_has_line_mapping)) {
        float value = u_node_scalars ? v_node_scalar : texelFetch(u_element_scalars,address(element,u_element_width),0).r;
        if (isnan(value)) color = u_invalid_color;
        else {
            float span = max(1.0e-20,u_scalar_range.y-u_scalar_range.x);
            color = texture(u_colormap,vec2(clamp((value-u_scalar_range.x)/span,0.0,1.0),0.5)).rgb;
        }
    }
    uint semantic = texelFetch(
        u_semantic_lines, address(v_primitive, u_line_width), 0
    ).r;
    if (u_has_line_mapping) semantic |= texelFetch(u_semantic_elements,address(element,u_element_width),0).r;
    if ((semantic & 4u) != 0u) discard;
    if ((semantic & 2u) != 0u)
        color = mix(color, u_preselection_color, 0.85);
    else if ((semantic & 1u) != 0u)
        color = mix(color, u_highlight_color, 0.80);
    else if (u_has_line_mapping && texelFetch(u_selected_elements,address(element,u_element_width),0).r != 0u)
        color = mix(color,u_selection_color,0.65);
    emit_color(color, u_opacity, v_depth);
}
"""

_LINE_PICK_FRAGMENT = """
#version 330
uniform bool u_has_line_mapping;
uniform uint u_element_width;
uniform usampler2D u_line_to_element;
uniform usampler2D u_active_elements;
uniform usampler2D u_semantic_elements;
uniform bool u_section_enabled;
uniform vec3 u_section_normal;
uniform float u_section_relative_offset;
uniform uint u_pick_base;
uniform bool u_pick_enabled;
uniform usampler2D u_semantic_lines;
uniform uint u_line_width;
in vec3 v_relative;
flat in uint v_primitive;
layout(location = 0) out uint pick_id;
ivec2 address(uint index, uint width) {
    return ivec2(int(index % width), int(index / width));
}
void main() {
    uint semantic = texelFetch(
        u_semantic_lines, address(v_primitive, u_line_width), 0
    ).r;
    if (u_has_line_mapping) {
        uint element = texelFetch(u_line_to_element,address(v_primitive,u_line_width),0).r;
        if (texelFetch(u_active_elements,address(element,u_element_width),0).r == 0u) discard;
        semantic |= texelFetch(u_semantic_elements,address(element,u_element_width),0).r;
    }
    if ((semantic & 4u) != 0u) discard;
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
// Extra sprite pixels around the disc so its antialiased fringe is not clipped.
uniform float u_point_margin;
in vec3 in_position;
in vec3 in_displacement;
out vec3 v_relative;
out float v_depth;
void main() {
    vec3 local = in_position + u_deformation_scale * in_displacement;
    v_relative = u_linear * local + u_origin_from_camera;
    v_depth = -(u_view_rotation * vec4(v_relative, 1.0)).z;
    gl_Position = u_projection * u_view_rotation * vec4(v_relative, 1.0);
    gl_PointSize = u_point_size + u_point_margin;
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
uniform float u_point_margin;
uniform float u_opacity;
uniform bool u_section_enabled;
uniform vec3 u_section_normal;
uniform float u_section_relative_offset;
uniform usampler2D u_semantic_points;
uniform uint u_point_width;
in vec3 v_relative;
in float v_depth;
//@OIT_EMIT@
ivec2 address(uint index, uint width) {
    return ivec2(int(index % width), int(index / width));
}
void main() {
    // Analytic coverage: distance to the disc edge in pixels, so the rim is
    // smooth with or without multisampling (which cannot smooth a discard).
    vec2 delta = gl_PointCoord * 2.0 - 1.0;
    float radial = length(delta) * 0.5 * (u_point_size + u_point_margin);
    float radius = 0.5 * u_point_size;
    float coverage = clamp(radius - radial + 0.5, 0.0, 1.0);
    if (coverage <= 0.0) discard;
    if (u_section_enabled && dot(u_section_normal, v_relative) < u_section_relative_offset) discard;
    float border_width = min(0.375 * u_point_size, 1.0);
    float outline_mix = clamp(radial - (radius - border_width) + 0.5, 0.0, 1.0);
    vec3 color = u_has_outline ? mix(u_color, u_outline_color, outline_mix) : u_color;
    uint semantic = texelFetch(
        u_semantic_points, address(uint(gl_PrimitiveID), u_point_width), 0
    ).r;
    if ((semantic & 4u) != 0u) discard;
    if ((semantic & 2u) != 0u)
        color = mix(color, u_preselection_color, 0.90);
    else if ((semantic & 1u) != 0u)
        color = mix(color, u_highlight_color, 0.85);
    emit_color(color, u_opacity * coverage, v_depth);
}
"""

_POINT_PICK_FRAGMENT = """
#version 330
uniform bool u_section_enabled;
uniform vec3 u_section_normal;
uniform float u_section_relative_offset;
uniform uint u_pick_base;
uniform bool u_pick_enabled;
uniform usampler2D u_semantic_points;
uniform uint u_point_width;
in vec3 v_relative;
layout(location = 0) out uint pick_id;
ivec2 address(uint index, uint width) {
    return ivec2(int(index % width), int(index / width));
}
void main() {
    vec2 delta = gl_PointCoord * 2.0 - 1.0;
    if (dot(delta, delta) > 1.0) discard;
    if (u_section_enabled && dot(u_section_normal, v_relative) < u_section_relative_offset) discard;
    if ((texelFetch(
        u_semantic_points, address(uint(gl_PrimitiveID), u_point_width), 0
    ).r & 4u) != 0u) discard;
    pick_id = u_pick_enabled ? u_pick_base + uint(gl_PrimitiveID) + 1u : 0u;
}
"""


for _name in ("_FRAGMENT", "_LINE_FRAGMENT", "_POINT_FRAGMENT"):
    globals()[_name] = globals()[_name].replace("//@OIT_EMIT@", _OIT_EMIT_GLSL)
del _name

def _matrix_bytes(matrix: np.ndarray) -> bytes:
    return np.asarray(matrix, dtype=np.float32).T.tobytes()


_PROGRAM_NAMES: dict[int, tuple[Any, frozenset[str]]] = {}


def _has_uniform(program: Any, name: str) -> bool:
    """Membership test that does not iterate every program member each call.

    ``name in program`` falls back to iterating all uniforms and attributes;
    done several times per draw call it dominated scenes with many meshes.  A
    linked program's members never change, so remember them per program.
    """

    entry = _PROGRAM_NAMES.get(id(program))
    if entry is None or entry[0] is not program:
        try:
            names = frozenset(program)
        except TypeError:  # test doubles that only implement ``in``
            return name in program
        if len(_PROGRAM_NAMES) >= 64:  # renderers are recreated rarely
            _PROGRAM_NAMES.clear()
        entry = _PROGRAM_NAMES[id(program)] = (program, names)
    return name in entry[1]


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


def _surface_stipple_windows(
    value: object, opacity: float, phase: int = 0
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Stipple windows for drawing a surface group.

    Only an explicit stipple pattern is dithered.  Plain opacity is blended
    (order-independently) rather than turned into a Bayer mask, so a
    translucent surface keeps its colour and shading instead of showing dots;
    it is also pickable over its whole area.
    """

    if str(value or ""):
        return _stipple_windows(value, opacity, phase)
    window = (0, 64, (int(phase) % 4) * 16)
    return window, window


# Clip-space depth bias for the translucent pass (fraction of the depth range).
_OIT_DEPTH_BIAS = 2.0e-5

# Lines at least this many device pixels wide are drawn as round-capped
# capsules so polyline joints are filled; thinner lines keep the cheaper
# butt-ended quad.  ANY3DVIEW_ROUND_JOINS=0 forces the quad everywhere.
_ROUND_JOIN_MIN_HALF_WIDTH = 1.5
# Pixels of margin around a point sprite for its antialiased rim.
_POINT_MARGIN_PX = 2.0
_ROUND_JOINS = os.environ.get("ANY3DVIEW_ROUND_JOINS", "").strip().casefold() not in {
    "0", "false", "off", "no",
}

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
    hidden: np.ndarray,
) -> bytes:
    """Pack highlight/preselection into one byte per GPU primitive.

    Bit zero is persistent semantic highlight, bit one is transient
    preselection, and bit two discards hidden primitives.  The dense byte
    texture lets a changed semantic set be one
    bounded buffer upload, independent of camera movement.
    """

    values = np.zeros(max(0, int(count)), dtype=np.uint8)
    if len(values):
        highlighted_valid = highlighted[highlighted < len(values)]
        preselected_valid = preselected[preselected < len(values)]
        hidden_valid = hidden[hidden < len(values)]
        values[highlighted_valid.astype(np.intp, copy=False)] |= np.uint8(1)
        values[preselected_valid.astype(np.intp, copy=False)] |= np.uint8(2)
        values[hidden_valid.astype(np.intp, copy=False)] |= np.uint8(4)
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
    normal_buffer: Any = None
    has_normals: bool = False
    # True when some vertex is shared by two or more lines, i.e. the chunk has
    # joints whose overlap must be unioned when it is drawn translucent.
    lines_joined: bool = False
    line_endpoint_buffer: Any = None
    line_element_map: Any = None
    node_scalar_texture: Any = None
    node_scalar_width: int = 1
    displacement_buffer: Any = None
    line_displacement_buffer: Any = None

    @classmethod
    @_release_failed_allocations
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
        # Static positions and dynamic displacement use separate attributes in
        # every visible/pick VAO. Deformation never reuploads static geometry.
        vertex_payload = np.ascontiguousarray(relative).tobytes()
        vertex_buffer = (
            ctx.buffer(vertex_payload) if vertex_payload else ctx.buffer(reserve=12)
        )
        displacement_payload = np.ascontiguousarray(displacement).tobytes()
        displacement_buffer = (
            ctx.buffer(displacement_payload) if displacement_payload else ctx.buffer(reserve=12)
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
        geometry_content = [
            (vertex_buffer, "3f", "in_position"),
            (displacement_buffer, "3f", "in_displacement"),
        ]
        render_content = [
            *geometry_content,
            (node_scalar_buffer, "1f", "in_node_scalar"),
        ]
        normal_buffer = None
        if mesh.normals is not None and len(mesh.normals):
            normal_buffer = ctx.buffer(
                np.ascontiguousarray(mesh.normals, dtype=np.float32).tobytes()
            )
            render_content.append((normal_buffer, "3f", "in_normal"))
        render_vao = ctx.vertex_array(render_program, render_content, index_buffer)
        pick_vao = ctx.vertex_array(pick_program, geometry_content, index_buffer)
        line_buffer = None
        line_displacement_buffer = None
        line_endpoint_buffer = None
        node_scalar_texture = None
        node_scalar_width = 1
        line_render_vao = None
        line_pick_vao = None
        if mesh.lines is not None and len(mesh.lines):
            line_buffer = ctx.buffer(np.ascontiguousarray(relative[mesh.lines]).tobytes())
            line_displacement_buffer = ctx.buffer(
                np.ascontiguousarray(displacement[mesh.lines]).tobytes()
            )
            line_content = [
                (line_buffer, "3f 3f /i", "in_start", "in_end"),
                (line_displacement_buffer, "3f 3f /i",
                 "in_start_displacement", "in_end_displacement"),
            ]
            line_endpoint_buffer = ctx.buffer(np.ascontiguousarray(mesh.lines,dtype=np.uint32).tobytes())
            node_scalar_texture,node_scalar_width = _packed_texture(ctx,node_scalar_values,"f4")
            line_render_vao = ctx.vertex_array(line_program,[*line_content,(line_endpoint_buffer,"2u /i","in_endpoint_nodes")])
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
        line_mapping = np.zeros(0 if mesh.lines is None else len(mesh.lines),dtype=np.uint32) if mesh.line_to_element is None else mesh.line_to_element
        line_element_map,_ = _packed_texture(ctx,line_mapping,"u4")
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
        # A scalar palette is a bounded scale, not a repeating surface texture.
        # Linear filtering at 0/1 otherwise blends opposite endpoint texels and
        # gives the minimum and maximum the same colour.
        colormap.repeat_x = False
        colormap.repeat_y = False
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
            normal_buffer,
            normal_buffer is not None,
            bool(
                mesh.lines is not None
                and len(mesh.lines) > 1
                and np.bincount(
                    np.asarray(mesh.lines, dtype=np.int64).ravel(),
                    minlength=mesh.node_count,
                ).max() > 1
            ),
            line_endpoint_buffer,
            line_element_map,
            node_scalar_texture,
            node_scalar_width,
            displacement_buffer,
            line_displacement_buffer,
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
            self.line_endpoint_buffer,
            self.line_element_map,
            self.node_scalar_texture,
            self.point_render_vao,
            self.point_pick_vao,
            self.point_index_buffer,
            self.vertex_buffer,
            self.displacement_buffer,
            self.line_displacement_buffer,
            self.node_scalar_buffer,
            self.normal_buffer,
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
        if self.has_face_colors and _has_uniform(program, "u_face_colors"):
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
            if _has_uniform(program, name):
                if set_uniform is None:
                    program[name].value = value
                else:
                    set_uniform(program, name, value)

    def bind_line_semantics(self, program: Any, set_uniform: Any = None) -> None:
        self.active.use(location=1)
        self.selected.use(location=2)
        self.scalars.use(location=3)
        self.colormap.use(location=4)
        self.semantic_elements.use(location=7)
        self.line_element_map.use(location=8)
        if self.node_scalar_texture is not None:self.node_scalar_texture.use(location=9)
        self.semantic_lines.use(location=6)
        for name, value in (
            ("u_semantic_lines", 6),
            ("u_line_width", self.line_semantic_width),
            ("u_line_to_element",8),
            ("u_active_elements",1),
            ("u_selected_elements",2),
            ("u_element_scalars",3),
            ("u_colormap",4),
            ("u_semantic_elements",7),
            ("u_element_width",self.element_width),
            ("u_has_line_mapping",self.mesh.line_to_element is not None),
            ("u_node_values",9),
            ("u_node_width",self.node_scalar_width),
            ("u_has_scalars",self.has_scalars),
            ("u_node_scalars",self.node_scalars),
            ("u_scalar_range",self.scalar_range),
        ):
            if _has_uniform(program, name):
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
            if _has_uniform(program, name):
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
    colormap: object
    highlighted_elements: np.ndarray
    preselected_elements: np.ndarray
    hidden_elements: np.ndarray
    highlighted_lines: np.ndarray
    preselected_lines: np.ndarray
    hidden_lines: np.ndarray
    highlighted_points: np.ndarray
    preselected_points: np.ndarray
    hidden_points: np.ndarray
    chunk_pickable: set[object]
    chunk_semantic_masks: dict[object, dict[str, np.ndarray]]
    generations: DirtyGenerations
    chunks: list[tuple[object, _Chunk]]
    palette_dirty: bool = False
    visible: bool = True


@dataclass(slots=True)
class _CameraFrame:
    """Camera values shared by every program and chunk in one render pass."""

    viewport: tuple[int, int]
    position: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    view: np.ndarray
    projection: np.ndarray
    near: float
    far: float
    frustum_tangent: float
    section_enabled: bool
    section_normal: tuple[float, float, float]
    section_relative_offset: float


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
        # Device pixels per logical pixel; line widths and point sizes are
        # specified in logical pixels.
        self.pixel_ratio = 1.0
        self._diagnostics: list[str] = []
        self._msaa_targets = None
        self._msaa_program = None
        self._msaa_vao = None
        self._oit_program = None
        self._oit_vao = None
        self._oit_layer_program = None
        self._oit_layer_vao = None
        self._oit_targets = None
        self._oit_failed = os.environ.get("ANY3DVIEW_OIT", "").strip().casefold() in {
            "0", "false", "off", "no",
        }
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
        self._frame_stamp: dict[int, _CameraFrame] = {}
        self._world_bounds_cache: dict[int, tuple[int, np.ndarray, float]] = {}
        self.pick_dirty = True
        self.draw_calls = 0
        self.frame_count = 0
        self.geometry_uploads = 0
        self.semantic_buffer_updates = 0
        self.scalar_buffer_updates = 0
        self.palette_updates = 0
        self.appearance_updates = 0
        self.palette_write_failures = 0

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
        colormap: object = None,
    ) -> None:
        appearance=validated_appearance(scalar_range,colormap)
        # Reject invalid scalar options before replacing published resources.
        group_style = (
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
        )
        back_color = str(back_color)
        invalid_color = str(invalid_color)
        self.remove_mesh(handle)
        chunks = []
        try:
            for chunk_id, mesh in [(None, handle.mesh), *handle.chunks]:
                chunk = _Chunk.create(
                    self.ctx, mesh, handle, self.render_program, self.pick_program,
                    self.line_program, self.line_pick_program, self.point_program,
                    self.point_pick_program, color,
                    tuple(str(value) for value in face_colors)
                    if face_colors is not None and chunk_id is None else None,
                )
                chunks.append((chunk_id, chunk))
        except BaseException:
            for _chunk_id, chunk in chunks:
                try:chunk.release()
                except BaseException:pass
            raise
        try:
            self._groups[id(handle)] = _Group(
                handle,
                color,
                line_color,
                point_color,
                *group_style,
                None if face_colors is None else tuple(str(value) for value in face_colors),
                back_color,
                invalid_color,
                appearance.scalar_range,
                appearance.colormap,
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
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
            self._groups[id(handle)].visible = bool(handle.visible)
        except BaseException:
            self._groups.pop(id(handle),None)
            for _chunk_id,chunk in chunks:
                try:chunk.release()
                except BaseException:pass
            raise
        self.pick_dirty = True
        try:
            if appearance.colormap is not None:
                data=palette_data(appearance.colormap)
                for _chunk_id,chunk in chunks:chunk.colormap.write(data);self.palette_updates+=1
            # Primary indices do not name unrelated incremental chunks.
            self._write_selection(self._groups[id(handle)])
        except BaseException:
            try:self.remove_mesh(handle)
            except BaseException:pass  # retain the primary initialization error
            raise
        self.geometry_uploads += len(chunks)
        self.pick_dirty = True

    def set_mesh_opacity(self, handle: MeshHandle, opacity: float) -> None:
        """Update draw uniforms only; retained geometry and fields stay intact."""
        handle._check()
        group = self._groups.get(id(handle))
        if group is None or group.handle is not handle:
            raise ValueError("mesh handle does not belong to this renderer")
        group.opacity = opacity
        # Explicit stipple uses opacity to determine coverage in the pick pass.
        # Plain blended opacity has no effect on integer picking.
        if group.stipple:
            self.pick_dirty = True

    def set_mesh_style(self, handle: MeshHandle, **values) -> None:
        """Apply viewer-validated draw options without changing buffer bindings."""
        handle._check()
        group = self._groups.get(id(handle))
        if group is None or group.handle is not handle:
            raise ValueError("mesh handle does not belong to this renderer")
        for name, value in values.items():
            setattr(group, name, value)
            if name == "color":
                for _chunk_id, chunk in group.chunks:
                    chunk.color = value
        if ("opacity" in values and group.stipple) or (
            "line_width" in values and any(chunk.mesh.lines is not None and len(chunk.mesh.lines)
                                         for _, chunk in group.chunks)
        ):
            # Width changes line pick coverage; lighting and colours do not.
            self.pick_dirty = True

    def remove_mesh(self, handle: MeshHandle) -> None:
        group = self._groups.pop(id(handle), None)
        if group is not None:
            self.pick_dirty = True
            first_error = None
            for _chunk_id, chunk in group.chunks:
                self._world_bounds_cache.pop(id(chunk), None)
                try:chunk.release()
                except BaseException as error:
                    if first_error is None:first_error=error
            if first_error is not None:raise first_error

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
        hidden_elements: Sequence[int] = (),
        selected_lines: Sequence[int] = (),
        preselected_lines: Sequence[int] = (),
        hidden_lines: Sequence[int] = (),
        selected_points: Sequence[int] = (),
        preselected_points: Sequence[int] = (),
        hidden_points: Sequence[int] = (),
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
            ("hidden_elements", hidden_elements, "element"),
            ("highlighted_lines", selected_lines, "line"),
            ("preselected_lines", preselected_lines, "line"),
            ("hidden_lines", hidden_lines, "line"),
            ("highlighted_points", selected_points, "point"),
            ("preselected_points", preselected_points, "point"),
            ("hidden_points", hidden_points, "point"),
        )
        changed: set[str] = set()
        for attribute, supplied, kind in specifications:
            values = _normalized_indices(supplied)
            if np.array_equal(values, state[attribute]):
                continue
            state[attribute] = values
            changed.add(kind)
            if attribute.startswith("hidden_"):
                self.pick_dirty = True
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
        if any(len(state[name]) for name in ("hidden_elements", "hidden_lines", "hidden_points")):
            self.pick_dirty = True
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
                "hidden_elements",
                "highlighted_lines",
                "preselected_lines",
                "hidden_lines",
                "highlighted_points",
                "preselected_points",
                "hidden_points",
            )
        }

    def set_semantic_masks(
        self,
        handle: MeshHandle,
        *,
        selected_elements: Sequence[int] = (),
        preselected_elements: Sequence[int] = (),
        hidden_elements: Sequence[int] = (),
        selected_lines: Sequence[int] = (),
        preselected_lines: Sequence[int] = (),
        hidden_lines: Sequence[int] = (),
        selected_points: Sequence[int] = (),
        preselected_points: Sequence[int] = (),
        hidden_points: Sequence[int] = (),
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
            hidden_elements=hidden_elements,
            highlighted_lines=selected_lines,
            preselected_lines=preselected_lines,
            hidden_lines=hidden_lines,
            highlighted_points=selected_points,
            preselected_points=preselected_points,
            hidden_points=hidden_points,
        )

    def _set_semantic_masks_partial(
        self,
        handle: MeshHandle,
        *,
        highlighted_elements: Any = _MISSING,
        preselected_elements: Any = _MISSING,
        hidden_elements: Any = _MISSING,
        highlighted_lines: Any = _MISSING,
        preselected_lines: Any = _MISSING,
        hidden_lines: Any = _MISSING,
        highlighted_points: Any = _MISSING,
        preselected_points: Any = _MISSING,
        hidden_points: Any = _MISSING,
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
            ("hidden_elements", hidden_elements, "element"),
            ("highlighted_lines", highlighted_lines, "line"),
            ("preselected_lines", preselected_lines, "line"),
            ("hidden_lines", hidden_lines, "line"),
            ("highlighted_points", highlighted_points, "point"),
            ("preselected_points", preselected_points, "point"),
            ("hidden_points", hidden_points, "point"),
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
            if attribute.startswith("hidden_"):
                self.pick_dirty = True
        for kind in changed:
            self._write_semantic_mask(group, kind)

    def _write_semantic_mask(self, group: _Group, kind: str) -> None:
        if kind == "element":
            highlighted = group.highlighted_elements
            preselected = group.preselected_elements
            hidden = group.hidden_elements
        elif kind == "line":
            highlighted = group.highlighted_lines
            preselected = group.preselected_lines
            hidden = group.hidden_lines
        elif kind == "point":
            highlighted = group.highlighted_points
            preselected = group.preselected_points
            hidden = group.hidden_points
        else:  # pragma: no cover - internal invariant
            raise ValueError(f"unknown semantic primitive kind: {kind}")

        chunk = next(
            (value for value_id, value in group.chunks if value_id is None),
            None,
        )
        if chunk is not None:
            self._write_chunk_mask_texture(chunk, kind, highlighted, preselected, hidden)

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
        hidden = empty if state is None else state[f"hidden_{kind}s"]
        self._write_chunk_mask_texture(
            chunk, kind, highlighted, preselected, hidden
        )

    def _write_chunk_mask_texture(
        self,
        chunk: _Chunk,
        kind: str,
        highlighted: np.ndarray,
        preselected: np.ndarray,
        hidden: np.ndarray,
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
            _semantic_mask_payload(count, width, highlighted, preselected, hidden)
        )
        self.semantic_buffer_updates += 1

    def _sync(self, group: _Group) -> None:
        current = group.handle.generations
        previous = group.generations
        if bool(group.handle.visible) != group.visible or any(
                getattr(current,name) != getattr(previous,name)
                for name in current.__dataclass_fields__ if name not in {"scalar","appearance"}):
            # Fallible scalar/palette writes must not leave a previously clean
            # ID target representing stale visibility, deformation or topology.
            self.pick_dirty = True
        if current.topology != previous.topology or current.position != previous.position:
            semantic_masks = {
                "selected_elements": group.highlighted_elements.copy(),
                "preselected_elements": group.preselected_elements.copy(),
                "hidden_elements": group.hidden_elements.copy(),
                "selected_lines": group.highlighted_lines.copy(),
                "preselected_lines": group.preselected_lines.copy(),
                "hidden_lines": group.hidden_lines.copy(),
                "selected_points": group.highlighted_points.copy(),
                "preselected_points": group.preselected_points.copy(),
                "hidden_points": group.hidden_points.copy(),
            }
            chunk_semantic_masks = {
                chunk_id: {
                    name: values.copy() for name, values in state.items()
                }
                for chunk_id, state in group.chunk_semantic_masks.items()
            }
            chunk_pickable = set(group.chunk_pickable)
            if current.selection != previous.selection:
                owned_chunk_ids={chunk_id for chunk_id,_mesh,owners,_resolver in group.handle.chunk_records if owners is not None}
                chunk_semantic_masks={chunk_id:state for chunk_id,state in chunk_semantic_masks.items() if chunk_id in owned_chunk_ids}
            # Detach, but retain the complete previous group until both the
            # replacement and its semantic state have been built successfully.
            self._groups.pop(id(group.handle), None)
            try:
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
                    scalar_range=group.handle.appearance.scalar_range,
                    colormap=group.handle.appearance.colormap,
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
                        hidden_elements=state["hidden_elements"],
                        selected_lines=state["highlighted_lines"],
                        preselected_lines=state["preselected_lines"],
                        hidden_lines=state["hidden_lines"],
                        selected_points=state["highlighted_points"],
                        preselected_points=state["preselected_points"],
                        hidden_points=state["hidden_points"],
                    )
            except BaseException:
                try:self.remove_mesh(group.handle)
                except BaseException:pass  # keep the primary rebuild error
                self._groups[id(group.handle)] = group
                self.pick_dirty = True
                raise
            for _chunk_id, old_chunk in group.chunks:
                self._world_bounds_cache.pop(id(old_chunk), None)
                old_chunk.release()
            return
        sources = {chunk_id: mesh for chunk_id, mesh in [(None, group.handle.mesh), *group.handle.chunks]}
        if current.displacement != previous.displacement:
            # A displacement array or scale changed.  Scale is uniform-only;
            # upload only if the underlying array object changed.
            uploads = 0
            for chunk_id, chunk in group.chunks:
                mesh = sources[chunk_id]
                if mesh.displacements is not chunk.mesh.displacements:
                    displacement = (
                        np.zeros((mesh.node_count, 3), dtype=np.float32)
                        if mesh.displacements is None
                        else np.ascontiguousarray(mesh.displacements, dtype=np.float32)
                    )
                    chunk.displacement_buffer.write(displacement.tobytes())
                    if chunk.line_displacement_buffer is not None:
                        chunk.line_displacement_buffer.write(
                            np.ascontiguousarray(displacement[mesh.lines]).tobytes()
                        )
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
                self.scalar_buffer_updates += 2
                if chunk.node_scalar_texture is not None:
                    size = chunk.node_scalar_width*max(1,math.ceil(len(node_values)/chunk.node_scalar_width))
                    padded_nodes = np.zeros(size,dtype=np.float32)
                    padded_nodes[:len(node_values)] = node_values
                    chunk.node_scalar_texture.write(padded_nodes.tobytes())
                    self.scalar_buffer_updates += 1
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
        if current.appearance != previous.appearance or group.palette_dirty:
            appearance=group.handle.appearance
            if appearance.colormap!=group.colormap or group.palette_dirty:
                data=palette_data(appearance.colormap)
                rollback=palette_data(group.colormap)
                try:
                    for _chunk_id,chunk in group.chunks:
                        chunk.colormap.write(data);self.palette_updates+=1
                except BaseException:
                    # A failed write may already have changed any texture.
                    # Restore best-effort now and force a complete next sync;
                    # preserve the primary GPU error if restoration also fails.
                    group.palette_dirty=True;self.palette_write_failures+=1
                    for _chunk_id,chunk in group.chunks:
                        try:chunk.colormap.write(rollback);self.palette_updates+=1
                        except BaseException:pass
                    raise
                group.palette_dirty=False
            group.scalar_range=appearance.scalar_range;group.colormap=appearance.colormap
            self.appearance_updates+=1
        for chunk_id, chunk in group.chunks:
            chunk.mesh = sources[chunk_id]
        group.generations = current
        visible = bool(group.handle.visible)
        if visible != group.visible or any(getattr(current,name)!=getattr(previous,name) for name in current.__dataclass_fields__ if name not in {"scalar","appearance"}):
            self.pick_dirty = True
        group.visible = visible

    @staticmethod
    def _camera_frame(
        camera: Camera3D,
        viewport: tuple[int, int],
        section_plane: Optional[SectionPlane],
    ) -> _CameraFrame:
        """Calculate camera-dependent values once for a complete render pass."""

        width, height = viewport
        right, up, forward = camera.basis()
        right_vector = np.asarray(right.to_tuple(), dtype=np.float64)
        up_vector = np.asarray(up.to_tuple(), dtype=np.float64)
        forward_vector = np.asarray(forward.to_tuple(), dtype=np.float64)
        position = np.asarray(camera.position.to_tuple(), dtype=np.float64)
        view = np.eye(4, dtype=np.float64)
        view[:3, :3] = np.asarray((right_vector, up_vector, -forward_vector))
        aspect = width / max(1.0, float(height))
        scale = 1.0 / math.tan(camera.fov / 2.0)
        near, far = float(camera.near), float(camera.far)
        projection = np.zeros((4, 4), dtype=np.float64)
        projection[0, 0] = scale / aspect
        projection[1, 1] = scale
        projection[2, 2] = (far + near) / (near - far)
        projection[2, 3] = 2.0 * far * near / (near - far)
        projection[3, 2] = -1.0
        section_enabled = section_plane is not None and section_plane.enabled
        section_normal = (
            section_plane.normal.to_tuple()
            if section_enabled
            else (1.0, 0.0, 0.0)
        )
        return _CameraFrame(
            (int(width), int(height)),
            position,
            right_vector,
            up_vector,
            forward_vector,
            view,
            projection,
            near,
            far,
            math.tan(camera.fov * 0.5),
            section_enabled,
            section_normal,
            (
                float(section_plane.offset - np.dot(section_normal, position))
                if section_enabled
                else 0.0
            ),
        )

    def _uniforms(
        self,
        program: Any,
        chunk: _Chunk,
        frame: _CameraFrame,
    ) -> None:
        transform = chunk.handle.transform
        transformed_origin = transform[:3, :3] @ chunk.origin + transform[:3, 3]
        if self._frame_stamp.get(id(program)) is not frame:
            # Camera, viewport and section state are identical for every chunk
            # of one frame; write them once per program, not once per chunk.
            # The stamp holds the frame object itself so its id cannot recur.
            program["u_view_rotation"].write(_matrix_bytes(frame.view))
            program["u_projection"].write(_matrix_bytes(frame.projection))
            if _has_uniform(program, "u_viewport"):
                program["u_viewport"].value = tuple(float(value) for value in frame.viewport)
            if _has_uniform(program, "u_near"):
                program["u_near"].value = frame.near
            program["u_section_enabled"].value = frame.section_enabled
            program["u_section_normal"].value = frame.section_normal
            program["u_section_relative_offset"].value = frame.section_relative_offset
            self._frame_stamp[id(program)] = frame
        program["u_linear"].write(_matrix_bytes(transform[:3, :3]))
        program["u_origin_from_camera"].value = tuple(
            np.asarray(transformed_origin - frame.position, dtype=np.float32)
        )
        program["u_deformation_scale"].value = chunk.handle.deformation_scale

    def _surface_uniforms(
        self,
        group: _Group,
        chunk: _Chunk,
        frame: _CameraFrame,
        camera: Camera3D,
        light: Optional[Light],
        shading_enabled: bool,
        selection_color: str,
        preselection_color: str,
        *,
        oit: bool = False,
        oit_depth_range: tuple[float, float] = (0.0, 1.0),
        depth_only: bool = False,
    ) -> None:
        """Bind the surface program's per-chunk state (shared by every pass).

        Every OIT-related uniform is written on every call, so an aborted pass
        can never leave a stale ``u_oit_depth_only`` behind for the next one.
        """

        self._uniforms(self.render_program, chunk, frame)
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
        front_stipple, back_stipple = _surface_stipple_windows(
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
            self.render_program, "u_has_normals", chunk.has_normals
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
        self._set_uniform_value(self.render_program, "u_oit", oit)
        self._set_uniform_value(self.render_program, "u_oit_depth_only", depth_only)
        self._set_uniform_value(
            self.render_program, "u_oit_depth_range", oit_depth_range
        )


    def _recover_after_failed_pass(self, framebuffer: Any, viewport: tuple[int, int]) -> None:
        """Best-effort restore of GL and cached uniform state after an aborted pass.

        Cached uniform values are dropped before being rewritten, because the
        failure may have left the cache saying one thing and the program
        another.  Nothing here may raise: it runs inside an error handler.
        """

        ctx = self.ctx
        steps = (
            lambda: setattr(ctx, "blend_equation", moderngl.FUNC_ADD),
            lambda: ctx.disable(moderngl.BLEND),
            lambda: setattr(framebuffer, "depth_mask", True),
            lambda: setattr(framebuffer, "color_mask", (True, True, True, True)),
            framebuffer.use,
            lambda: setattr(ctx, "viewport", (0, 0, int(viewport[0]), int(viewport[1]))),
            lambda: ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE),
            lambda: setattr(ctx, "depth_func", "<="),
        )
        for step in steps:
            try:
                step()
            except Exception:
                pass
        for program, names in (
            (self.render_program, (("u_oit", False), ("u_oit_depth_only", False), ("u_depth_bias", 0.0))),
            (self.line_program, (("u_oit", False), ("u_oit_depth_only", False), ("u_cap_segments", 0))),
            (self.point_program, (("u_oit", False), ("u_oit_depth_only", False))),
        ):
            for name, value in names:
                self._uniform_value_cache.pop((id(program), name), None)
                try:
                    self._set_uniform_value(program, name, value)
                except Exception:
                    pass

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Quality features this renderer had to give up, and why."""

        return tuple(self._diagnostics)

    def _diagnose(self, message: str) -> None:
        """Record (once) that a feature degraded, so the host can report it."""

        if message not in self._diagnostics:
            self._diagnostics.append(message)

    def _release_msaa(self) -> None:
        targets, self._msaa_targets = self._msaa_targets, None
        for resource in () if targets is None or targets[2] is None else targets[2]:
            try:
                resource.release()
            except Exception:
                pass
        for resource in (self._msaa_vao, self._msaa_program):
            if resource is not None:
                resource.release()
        self._msaa_vao = self._msaa_program = None

    def _msaa_framebuffer(self, size: tuple[int, int], samples: int):
        """Return a multisampled framebuffer, or ``None`` if one cannot be made."""

        targets = self._msaa_targets
        if targets is not None and targets[:2] == (size, samples):
            return None if targets[2] is None else targets[2][-2:]
        self._release_msaa()
        try:
            color = self.ctx.renderbuffer(size, samples=samples)
            depth = self.ctx.depth_renderbuffer(size, samples=samples)
            multisampled = self.ctx.framebuffer(color, depth)
            resolved_color = self.ctx.texture(size, 4)
            resolved_color.filter = (moderngl.NEAREST, moderngl.NEAREST)
            resolved_depth = self.ctx.depth_texture(size)
            resolved_depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
            resolved_depth.compare_func = ""
            resolved = self.ctx.framebuffer(resolved_color, resolved_depth)
            if self._msaa_program is None:
                self._msaa_program = self.ctx.program(
                    vertex_shader=_OIT_COMPOSITE_VERTEX,
                    fragment_shader=_MSAA_RESOLVE_FRAGMENT,
                )
                self._msaa_vao = self.ctx.vertex_array(self._msaa_program, [])
        except Exception as error:
            # Remember the failure for this size so it is not retried per frame.
            self._msaa_targets = (size, samples, None)
            self._diagnose(
                f"multisampling unavailable at {samples}x ({type(error).__name__}: {error});"
                " rendering without antialiasing"
            )
            return None
        self._msaa_targets = (
            size, samples,
            (color, depth, resolved_color, resolved_depth, multisampled, resolved),
        )
        return multisampled, resolved

    def _resolve_msaa(
        self, multisampled: Any, resolved: Any, destination: Any, size: tuple[int, int]
    ) -> None:
        """Resolve samples to a plain framebuffer, then draw it to ``destination``.

        Blitting straight into a window's default framebuffer is not honoured
        by every driver, so the resolve goes through a texture-backed
        framebuffer and a fullscreen pass that also restores scene depth for
        the depth-tested HUD.
        """

        ctx = self.ctx
        ctx.copy_framebuffer(resolved, multisampled)
        resolved_color, resolved_depth = self._msaa_targets[2][2:4]
        destination.use()
        ctx.viewport = (0, 0, size[0], size[1])
        ctx.disable(moderngl.BLEND | moderngl.CULL_FACE)
        ctx.enable(moderngl.DEPTH_TEST)
        destination.depth_mask = True
        resolved_color.use(location=0)
        resolved_depth.use(location=1)
        self._set_uniform_value(self._msaa_program, "u_color", 0)
        self._set_uniform_value(self._msaa_program, "u_depth", 1)
        ctx.depth_func = "1"  # the pass writes the resolved depth unconditionally
        try:
            self._msaa_vao.render(mode=moderngl.TRIANGLES, vertices=3)
        finally:
            ctx.depth_func = "<="

    def _line_cap_segments(self, group: _Group) -> int:
        """Segments per round cap, or 0 for the butt-ended quad."""

        half = group.line_width * 0.5 * self.pixel_ratio
        if not _ROUND_JOINS or half < _ROUND_JOIN_MIN_HALF_WIDTH:
            return 0
        return int(min(12, max(3, math.ceil(half))))

    def _draw_lines(
        self,
        group: _Group,
        chunk: _Chunk,
        frame: _CameraFrame,
        selection_color: str,
        preselection_color: str,
        occlude_lines: bool,
        show_mesh_lines: bool,
        *,
        oit: bool = False,
        depth_only: bool = False,
        depth_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """Draw one chunk's lines (``depth_only`` replays occluders for OIT)."""

        if (
            group.depth_only
            or not (show_mesh_lines or not group.mesh_lines)
            or chunk.line_render_vao is None
        ):
            return
        ctx = self.ctx
        without_depth = group.line_overlay or not occlude_lines
        if depth_only and without_depth:
            return  # overlay lines are never in front of anything
        program = self.line_program
        ctx.disable(moderngl.CULL_FACE)
        if without_depth:
            ctx.disable(moderngl.DEPTH_TEST)
        self._uniforms(program, chunk, frame)
        chunk.bind_line_semantics(program, self._set_uniform_value)
        if "u_scalar_range" in program:self._set_uniform_value(program,"u_scalar_range",group.scalar_range or chunk.scalar_range)
        self._set_uniform_value(program, "u_color", _rgb(group.line_color))
        self._set_uniform_value(program,"u_invalid_color",_rgb(group.invalid_color))
        self._set_uniform_value(program,"u_selection_color",_rgb(selection_color))
        self._set_uniform_value(program, "u_highlight_color", _rgb(selection_color))
        self._set_uniform_value(
            program, "u_preselection_color", _rgb(preselection_color)
        )
        self._set_uniform_value(program, "u_opacity", group.opacity)
        self._set_uniform_value(
            program, "u_half_width", group.line_width * 0.5 * self.pixel_ratio
        )
        self._set_uniform_value(program, "u_oit", oit)
        self._set_uniform_value(program, "u_oit_depth_only", depth_only)
        self._set_uniform_value(program, "u_oit_depth_range", depth_range)
        caps = self._line_cap_segments(group)
        self._set_uniform_value(program, "u_cap_segments", caps)
        if caps:
            chunk.line_render_vao.render(
                mode=moderngl.TRIANGLE_FAN,
                vertices=2 * (caps + 1) + 2,
                instances=len(chunk.mesh.lines),
            )
        else:
            chunk.line_render_vao.render(
                mode=moderngl.TRIANGLE_STRIP,
                vertices=4,
                instances=len(chunk.mesh.lines),
            )
        if not depth_only:
            self.draw_calls += 1
        if without_depth:
            ctx.enable(moderngl.DEPTH_TEST)

    def _lines_drawable(
        self, group: _Group, chunk: _Chunk, show_mesh_lines: bool
    ) -> bool:
        return (
            not group.depth_only
            and (show_mesh_lines or not group.mesh_lines)
            and chunk.line_render_vao is not None
        )

    def _draw_lines_union(
        self,
        group: _Group,
        chunk: _Chunk,
        frame: _CameraFrame,
        selection_color: str,
        preselection_color: str,
        occlude_lines: bool,
        show_mesh_lines: bool,
        depth_range: tuple[float, float],
        size: tuple[int, int],
    ) -> None:
        """Add one translucent line chunk to the accumulation as a union.

        Segments of a polyline overlap at their joints (and round caps overlap
        the neighbouring body), so summing every fragment would darken each
        joint.  The chunk is drawn into a scratch layer with MAX blending, which
        keeps one contribution per pixel, and the layer is then added once.
        """

        if not self._lines_drawable(group, chunk, show_mesh_lines):
            return
        if not chunk.lines_joined:
            # No shared vertices, so nothing overlaps within the chunk.
            self._draw_lines(
                group, chunk, frame, selection_color, preselection_color,
                occlude_lines, show_mesh_lines, oit=True, depth_range=depth_range,
            )
            return
        ctx = self.ctx
        _accum, _reveal, _depth, oit_fbo, layer_accum, layer_reveal, layer_fbo = (
            self._oit_targets
        )
        layer_fbo.depth_mask = False  # keep the shared opaque depth when clearing
        layer_fbo.use()
        ctx.viewport = (0, 0, size[0], size[1])
        layer_fbo.clear(0.0, 0.0, 0.0, 0.0)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        ctx.blend_equation = moderngl.MAX
        try:
            self._draw_lines(
                group, chunk, frame, selection_color, preselection_color,
                occlude_lines, show_mesh_lines, oit=True, depth_range=depth_range,
            )
        finally:
            ctx.blend_equation = moderngl.FUNC_ADD
        oit_fbo.use()
        ctx.viewport = (0, 0, size[0], size[1])
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        layer_accum.use(location=0)
        layer_reveal.use(location=1)
        self._set_uniform_value(self._oit_layer_program, "u_layer_accum", 0)
        self._set_uniform_value(self._oit_layer_program, "u_layer_reveal", 1)
        self._oit_layer_vao.render(mode=moderngl.TRIANGLES, vertices=3)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    def _draw_points(
        self,
        group: _Group,
        chunk: _Chunk,
        frame: _CameraFrame,
        selection_color: str,
        preselection_color: str,
        *,
        oit: bool = False,
        depth_only: bool = False,
        depth_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """Draw one chunk's points (``depth_only`` replays occluders for OIT)."""

        if group.depth_only or chunk.point_render_vao is None:
            return
        ctx = self.ctx
        without_depth = group.point_overlay
        if depth_only and without_depth:
            return
        program = self.point_program
        ctx.disable(moderngl.CULL_FACE)
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        if without_depth:
            ctx.disable(moderngl.DEPTH_TEST)
        self._uniforms(program, chunk, frame)
        chunk.bind_point_semantics(program, self._set_uniform_value)
        self._set_uniform_value(program, "u_color", _rgb(group.point_color))
        self._set_uniform_value(program, "u_highlight_color", _rgb(selection_color))
        self._set_uniform_value(
            program, "u_preselection_color", _rgb(preselection_color)
        )
        self._set_uniform_value(
            program, "u_outline_color", _rgb(group.point_outline or group.point_color)
        )
        self._set_uniform_value(program, "u_has_outline", bool(group.point_outline))
        self._set_uniform_value(program, "u_opacity", group.opacity)
        self._set_uniform_value(
            program, "u_point_size", group.point_size * self.pixel_ratio
        )
        self._set_uniform_value(program, "u_oit", oit)
        self._set_uniform_value(program, "u_oit_depth_only", depth_only)
        self._set_uniform_value(program, "u_oit_depth_range", depth_range)
        self._set_uniform_value(program, "u_point_margin", _POINT_MARGIN_PX)
        if not oit and not depth_only:
            # The disc's rim carries fractional alpha; blend it over the scene.
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        chunk.point_render_vao.render(mode=moderngl.POINTS)
        if not depth_only:
            self.draw_calls += 1
        if without_depth:
            ctx.enable(moderngl.DEPTH_TEST)

    def _release_oit_targets(self) -> None:
        targets, self._oit_targets = self._oit_targets, None
        if targets is not None:
            for resource in targets:
                try:
                    resource.release()
                except Exception:
                    pass

    def _release_oit(self) -> None:
        self._release_oit_targets()
        for resource in (
            self._oit_vao, self._oit_program,
            self._oit_layer_vao, self._oit_layer_program,
        ):
            if resource is not None:
                resource.release()
        self._oit_vao = self._oit_program = None
        self._oit_layer_vao = self._oit_layer_program = None

    def _oit_framebuffer(self, size: tuple[int, int]):
        """Return the (re)sized accumulation targets, or ``None`` if unusable.

        Order-independent transparency needs float render targets.  Any
        failure disables it for this renderer and the blended, depth-sorted
        path draws instead; it never prevents the scene from rendering.
        """

        if self._oit_failed:
            return None
        try:
            if self._oit_program is None:
                self._oit_program = self.ctx.program(
                    vertex_shader=_OIT_COMPOSITE_VERTEX,
                    fragment_shader=_OIT_COMPOSITE_FRAGMENT,
                )
                self._oit_vao = self.ctx.vertex_array(self._oit_program, [])
                self._oit_layer_program = self.ctx.program(
                    vertex_shader=_OIT_COMPOSITE_VERTEX,
                    fragment_shader=_OIT_LAYER_FRAGMENT,
                )
                self._oit_layer_vao = self.ctx.vertex_array(
                    self._oit_layer_program, []
                )
            targets = self._oit_targets
            if targets is None or targets[0].size != size:
                self._release_oit_targets()
                accum = self.ctx.texture(size, 4, dtype="f2")
                reveal = self.ctx.texture(size, 1, dtype="f2")
                for texture in (accum, reveal):
                    texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
                    texture.repeat_x = texture.repeat_y = False
                depth = self.ctx.depth_renderbuffer(size)
                framebuffer = self.ctx.framebuffer([accum, reveal], depth)
                # A scratch layer sharing the same depth buffer: translucent
                # lines are rendered here with MAX blending (their union) and
                # then added to the accumulation, so overlapping parts of one
                # polyline are not counted twice.
                layer_accum = self.ctx.texture(size, 4, dtype="f2")
                layer_reveal = self.ctx.texture(size, 1, dtype="f2")
                for texture in (layer_accum, layer_reveal):
                    texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
                    texture.repeat_x = texture.repeat_y = False
                layer_framebuffer = self.ctx.framebuffer(
                    [layer_accum, layer_reveal], depth
                )
                targets = self._oit_targets = (
                    accum, reveal, depth, framebuffer,
                    layer_accum, layer_reveal, layer_framebuffer,
                )
            return targets
        except Exception as error:
            self._oit_failed = True
            self._release_oit_targets()
            self._diagnose(
                "order-independent transparency unavailable"
                f" ({type(error).__name__}: {error}); translucent surfaces use"
                " depth-sorted blending"
            )
            return None

    def _render_oit(
        self,
        framebuffer: Any,
        viewport: tuple[int, int],
        frame: _CameraFrame,
        camera: Camera3D,
        light: Optional[Light],
        shading_enabled: bool,
        selection_color: str,
        preselection_color: str,
        opaque_items: list,
        items: list,
        occlude_lines: bool = True,
        show_mesh_lines: bool = True,
    ) -> bool:
        """Draw translucent surfaces with weighted blended OIT.

        Returns ``False`` when float targets are unavailable so the caller can
        use the depth-sorted blended path instead.
        """

        size = (int(viewport[0]), int(viewport[1]))
        targets = self._oit_framebuffer(size)
        if targets is None:
            return False
        accum, reveal, _depth, oit_fbo = targets[:4]
        layer_accum, layer_reveal, layer_fbo = targets[4:]
        ctx = self.ctx
        depths = []
        for _group, chunk in items:
            center, radius = self._chunk_world_bounds(chunk)
            centre_depth = float(np.dot(center - frame.position, frame.forward))
            depths.extend((centre_depth - radius, centre_depth + radius))
        depth_range = (min(depths), max(depths))

        oit_fbo.use()
        ctx.viewport = (0, 0, size[0], size[1])
        oit_fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.disable(moderngl.BLEND)
        # The opaque scene's depth lives in the caller's target (often a
        # default framebuffer that cannot be attached here), so replay the
        # opaque surfaces depth-only to occlude translucent fragments.
        oit_fbo.depth_mask = True
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        for group, chunk in opaque_items:
            if chunk.mesh.triangle_count:
                if not group.cull_backface:
                    ctx.disable(moderngl.CULL_FACE)
                self._surface_uniforms(
                    group, chunk, frame, camera, light, shading_enabled,
                    selection_color, preselection_color, depth_only=True,
                )
                chunk.render_vao.render(mode=moderngl.TRIANGLES)
                ctx.enable(moderngl.CULL_FACE)
            # Opaque lines and points occlude translucent surfaces behind them.
            self._draw_lines(
                group, chunk, frame, selection_color, preselection_color,
                occlude_lines, show_mesh_lines, depth_only=True,
            )
            self._draw_points(
                group, chunk, frame, selection_color, preselection_color,
                depth_only=True,
            )
            ctx.enable(moderngl.CULL_FACE)
        oit_fbo.depth_mask = False  # translucent layers must not occlude each other
        self._set_uniform_value(self.render_program, "u_depth_bias", _OIT_DEPTH_BIAS)
        for group, chunk in items:
            if chunk.mesh.triangle_count:
                if not group.cull_backface:
                    ctx.disable(moderngl.CULL_FACE)
                self._surface_uniforms(
                    group, chunk, frame, camera, light, shading_enabled,
                    selection_color, preselection_color,
                    oit=True, oit_depth_range=depth_range,
                )
                chunk.render_vao.render(mode=moderngl.TRIANGLES)
                self.draw_calls += 1
            self._draw_lines_union(
                group, chunk, frame, selection_color, preselection_color,
                occlude_lines, show_mesh_lines, depth_range, size,
            )
            self._draw_points(
                group, chunk, frame, selection_color, preselection_color,
                oit=True, depth_range=depth_range,
            )
            ctx.enable(moderngl.CULL_FACE)
        self._set_uniform_value(self.render_program, "u_oit", False)
        self._set_uniform_value(self.line_program, "u_oit", False)
        self._set_uniform_value(self.point_program, "u_oit", False)
        self._set_uniform_value(self.render_program, "u_depth_bias", 0.0)

        framebuffer.use()
        ctx.viewport = (0, 0, size[0], size[1])
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        accum.use(location=0)
        reveal.use(location=1)
        self._set_uniform_value(self._oit_program, "u_accum", 0)
        self._set_uniform_value(self._oit_program, "u_reveal", 1)
        self._oit_vao.render(mode=moderngl.TRIANGLES, vertices=3)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.disable(moderngl.BLEND)
        oit_fbo.depth_mask = True
        framebuffer.depth_mask = True
        return True

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
        samples: int = 0,
    ) -> None:
        """Draw the scene.

        ``samples`` > 1 antialiases the render by drawing into an internal
        multisampled framebuffer and resolving colour and depth into
        ``target`` (the screen when omitted).  Leave it at 0 when the target is
        already multisampled, as Qt's widget framebuffer is.
        """

        self.frame_count += 1
        for group in list(self._groups.values()):
            self._sync(group)
        resolve_to = destination = None
        if int(samples) > 1:
            size = (int(viewport[0]), int(viewport[1]))
            pair = self._msaa_framebuffer(size, int(samples))
            if pair is not None:
                resolve_to = pair[1]
                destination = target or self.ctx.screen
                target = pair[0]
        framebuffer = target or self.ctx.screen
        # Write masks belong to the framebuffer (moderngl's Context has none),
        # and a colour mask takes effect when the framebuffer is bound.  Reset
        # them first: a stale mask would stop the clear below.
        framebuffer.depth_mask = True
        framebuffer.color_mask = (True, True, True, True)
        framebuffer.use()
        self.ctx.viewport = (0, 0, int(viewport[0]), int(viewport[1]))
        self.ctx.depth_func = "<="  # never inherit a stale function from an aborted frame
        framebuffer.clear(*clear_color, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.draw_calls = 0
        frame = self._camera_frame(camera, viewport, section_plane)
        draw_items = [
            (group, chunk)
            for group in self._groups.values()
            if not group.handle.removed and group.handle.visible
            for _chunk_id, chunk in group.chunks
            if self._chunk_visible(chunk, frame)
        ]
        transparent = sorted(
            (item for item in draw_items if item[0].opacity < 0.999),
            key=lambda item: (
                item[0].layer,
                -float(
                    np.linalg.norm(
                        self._chunk_world_bounds(item[1])[0] - frame.position
                    )
                ),
            ),
        )
        opaque_items = sorted(
            (item for item in draw_items if item[0].opacity >= 0.999),
            key=lambda item: item[0].layer,
        )
        ordered = opaque_items + transparent
        # Translucent surfaces, lines and points without an explicit stipple
        # pattern are drawn order-independently once the opaque scene is in.
        oit_items = [
            item for item in transparent
            if not item[0].stipple
            and not item[0].depth_only
            and (
                item[1].mesh.triangle_count
                or (
                    item[1].line_render_vao is not None
                    and (show_mesh_lines or not item[0].mesh_lines)
                )
                or item[1].point_render_vao is not None
            )
        ]
        oit_ids: set[int] = set()
        oit_pending = bool(oit_items)
        # Opaque points are drawn after every other opaque primitive.  Their
        # antialiased rim writes depth at fractional coverage; drawn earlier it
        # would hide farther geometry behind a ring of background.
        deferred_points: list[tuple[_Group, _Chunk]] = []

        def flush_points() -> None:
            if not deferred_points:
                return
            framebuffer.depth_mask = True
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.disable(moderngl.BLEND)
            for point_group, point_chunk in deferred_points:
                self._draw_points(
                    point_group, point_chunk, frame, selection_color,
                    preselection_color,
                )
            deferred_points.clear()
            self.ctx.enable(moderngl.CULL_FACE)

        for group, chunk in ordered:
            is_transparent = group.opacity < 0.999
            if is_transparent:
                flush_points()
            if is_transparent and oit_pending:
                oit_pending = False
                try:
                    translucent_done = self._render_oit(
                        framebuffer, viewport, frame, camera, light, shading_enabled,
                        selection_color, preselection_color, opaque_items, oit_items,
                        occlude_lines, show_mesh_lines,
                    )
                except Exception as error:
                    # A driver error inside the pass must not take the viewer
                    # down: report it, restore state and draw this frame's
                    # translucent geometry with sorted blending instead.
                    self._oit_failed = True
                    self._release_oit_targets()
                    self._diagnose(
                        "order-independent transparency disabled after a render"
                        f" error ({type(error).__name__}: {error}); translucent"
                        " surfaces use depth-sorted blending"
                    )
                    self._recover_after_failed_pass(framebuffer, viewport)
                    translucent_done = False
                if translucent_done:
                    oit_ids = {id(item[1]) for item in oit_items}
            if is_transparent:
                self.ctx.enable(moderngl.BLEND)
                self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                framebuffer.depth_mask = False
            else:
                self.ctx.disable(moderngl.BLEND)
                framebuffer.depth_mask = True
            if not group.cull_backface:
                self.ctx.disable(moderngl.CULL_FACE)
            if group.depth_only:
                framebuffer.color_mask = (False, False, False, False)
                framebuffer.use()  # applies the colour mask
            drawn_by_oit = id(chunk) in oit_ids
            if not drawn_by_oit:
                self._surface_uniforms(
                    group, chunk, frame, camera, light, shading_enabled,
                    selection_color, preselection_color,
                )
                if chunk.mesh.triangle_count:
                    chunk.render_vao.render(mode=moderngl.TRIANGLES)
                    self.draw_calls += 1
                self._draw_lines(
                    group, chunk, frame, selection_color, preselection_color,
                    occlude_lines, show_mesh_lines,
                )
                if is_transparent:
                    self._draw_points(
                        group, chunk, frame, selection_color, preselection_color
                    )
                elif not group.depth_only and chunk.point_render_vao is not None:
                    deferred_points.append((group, chunk))
            if not group.cull_backface or chunk.line_render_vao is not None or chunk.point_render_vao is not None:
                self.ctx.enable(moderngl.CULL_FACE)
            if group.depth_only:
                framebuffer.color_mask = (True, True, True, True)
                framebuffer.use()
        flush_points()
        framebuffer.depth_mask = True
        self.ctx.disable(moderngl.BLEND)
        if resolve_to is not None:
            self._resolve_msaa(
                framebuffer, resolve_to, destination,
                (int(viewport[0]), int(viewport[1])),
            )

    def _chunk_world_bounds(self, chunk: _Chunk) -> tuple[np.ndarray, float]:
        """Return exact transformed bounds, caching until the transform changes."""

        transform_generation = chunk.handle.generations.transform
        key = id(chunk)
        cached = self._world_bounds_cache.get(key)
        if cached is None or cached[0] != transform_generation:
            transform = chunk.handle.transform
            center = transform[:3, :3] @ chunk.origin + transform[:3, 3]
            radius = chunk.radius * float(np.linalg.norm(transform[:3, :3], ord=2))
            self._world_bounds_cache[key] = (transform_generation, center, radius)
            return center, radius
        return cached[1], cached[2]

    def _chunk_visible(self, chunk: _Chunk, frame: _CameraFrame) -> bool:
        center, radius = self._chunk_world_bounds(chunk)
        relative = center - frame.position
        x = float(np.dot(relative, frame.right))
        y = float(np.dot(relative, frame.up))
        depth = float(np.dot(relative, frame.forward))
        if depth + radius < frame.near or depth - radius > frame.far:
            return False
        visible_depth = max(frame.near, depth)
        half_height = visible_depth * frame.frustum_tangent
        half_width = half_height * frame.viewport[0] / max(1.0, float(frame.viewport[1]))
        return abs(x) <= half_width + radius and abs(y) <= half_height + radius

    def _ensure_pick_target(self, viewport: tuple[int, int]) -> None:
        if viewport == self._pick_size:
            return
        self.pick_dirty = True
        stage=_ResourceStage(self.ctx)
        try:
            texture=stage.texture(viewport,1,dtype="u4")
            depth=stage.depth_renderbuffer(viewport)
            framebuffer=stage.framebuffer(color_attachments=[texture],depth_attachment=depth)
        except BaseException:
            stage.release()
            raise
        previous=(self._pick_fbo,self._pick_texture,self._pick_depth)
        self._pick_texture,self._pick_depth,self._pick_fbo=texture,depth,framebuffer
        self._pick_size = viewport
        for resource in previous:
            if resource is not None:resource.release()
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
        # Resize, clear, sync and draw are fallible. Only a complete pass below
        # may mark the target clean, including an explicitly requested redraw.
        self.pick_dirty = True
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
        frame = self._camera_frame(camera, viewport, section_plane)
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
                self._uniforms(self.pick_program, chunk, frame)
                chunk.bind_fields(self.pick_program, self._set_uniform_value)
                self._set_uniform_value(self.pick_program, "u_pick_base", base)
                front_stipple, back_stipple = _surface_stipple_windows(
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
                    self._uniforms(self.line_pick_program, chunk, frame)
                    chunk.bind_line_semantics(
                        self.line_pick_program, self._set_uniform_value
                    )
                    self._set_uniform_value(
                        self.line_pick_program,
                        "u_half_width",
                        max(2.5, group.line_width * 0.5) * self.pixel_ratio,
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
                    self._uniforms(self.point_pick_program, chunk, frame)
                    chunk.bind_point_semantics(
                        self.point_pick_program, self._set_uniform_value
                    )
                    self._set_uniform_value(
                        self.point_pick_program,
                        "u_point_size",
                        max(7.0, group.point_size) * self.pixel_ratio,
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
        self._release_oit()
        self._release_msaa()
        self._empty_face_color.release()
