"""Original WGSL indexed rasterizer; no CPU image rasterization or GL fallback."""
from __future__ import annotations
import math
import sys
import numpy as np

SHADER = """
struct Uniforms { mvp: mat4x4<f32>, color: vec4<f32>, invalid: vec4<f32>, params: vec4<f32>, model: mat4x4<f32>, normal_matrix: mat4x4<f32>,
    plane: vec4<f32>, light: vec4<f32>, light_params: vec4<f32>,
    flags: vec4<f32>, back_color: vec4<f32>,
    display: vec4<f32>, viewport: vec4<f32>, outline: vec4<f32>, planes: array<vec4<f32>,7>, contour: vec4<f32>,
    levels: array<vec4<f32>,4>, emphasis: vec4<f32>, selection: vec4<f32>, };
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> displacement: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> node_values: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> element_values: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> element_active: array<u32>;
@group(0) @binding(6) var<storage, read> palette: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> face_colors: array<vec4<f32>>;
struct Varying {
    @builtin(position) position: vec4<f32>,
    @location(0) scalar: f32,
    @location(1) valid: f32,
    @location(2) @interpolate(flat) element: u32,
    @location(3) relative: vec3<f32>,
    @location(4) normal: vec3<f32>,
    @location(5) @interpolate(flat) primitive: u32,
    @location(6) clip_distance: f32,
    @location(7) marker: vec2<f32>,
    @location(8) local: vec3<f32>,
    @location(9) barycentric: vec3<f32>,
    @location(10) @interpolate(flat) edge_mask: u32,
};
@vertex fn vs(@location(0) node: u32, @location(1) element: u32, @location(2) primitive: u32, @location(3) other: u32, @location(4) corner: u32) -> Varying {
    var out: Varying;
    // Original reference coordinates plus supplied displacement: display only.
    let p = positions[node].xyz + u.params.x * displacement[node].xyz;
    out.position = u.mvp * vec4<f32>(p, 1.0);
    out.marker = vec2<f32>(0.0);
    out.local=p;
    out.edge_mask=other;
    out.barycentric=vec3<f32>(select(0.,1.,corner==0u),select(0.,1.,corner==1u),select(0.,1.,corner==2u));
    if (u.display.z == 1.0) {
        let other_p = positions[other].xyz + u.params.x * displacement[other].xyz;
        let clip_other = u.mvp * vec4<f32>(other_p,1.0);
        var delta = (clip_other.xy / clip_other.w - out.position.xy / out.position.w) * u.viewport.xy;
        if (corner >= 2u) { delta = -delta; }
        var perpendicular = vec2<f32>(0.0,1.0);
        if (dot(delta,delta)>0.0) { perpendicular=vec2<f32>(-delta.y,delta.x)/length(delta); }
        let sign = select(-1.0,1.0,corner == 1u || corner == 2u);
        out.position = vec4<f32>(out.position.xy + perpendicular*sign*u.display.y/u.viewport.xy*out.position.w,out.position.zw);
    } else if (u.display.z == 2.0) {
        var corners = array<vec2<f32>,4>(vec2<f32>(-1.,-1.),vec2<f32>(1.,-1.),vec2<f32>(1.,1.),vec2<f32>(-1.,1.));
        let offset=corners[corner];
        out.marker=offset;
        out.position=vec4<f32>(out.position.xy+offset*u.display.y/u.viewport.xy*out.position.w,out.position.zw);
    }
    out.relative = (u.model * vec4<f32>(p, 1.0)).xyz;
    out.normal = (u.normal_matrix * vec4<f32>(normals[node].xyz, 0.0)).xyz;
    out.primitive = primitive;
    out.clip_distance = dot(u.plane.xyz, p) - u.plane.w;
    var value = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    if (u.params.y == 1.0) { value = node_values[node]; }
    if (u.params.y == 2.0 && element != 0xffffffffu) { value = element_values[element]; }
    out.scalar = value.x; out.valid = value.y; out.element = element;
    return out;
}
fn shaded(v: Varying, front: bool) -> vec4<f32> {
    // Derivatives precede every potentially divergent discard.
    let flat_normal = cross(dpdy(v.relative), dpdx(v.relative));
    let scalar_width=max(fwidth(v.scalar),1e-7);
    let edge_width=max(fwidth(v.barycentric),vec3<f32>(1e-6));
    if (u.display.w>0.0 && u.display.z==0.0) {
        let code=u32(u.display.w);
        let start=code&63u;let count=(code>>6u)&127u;let rotation=code>>13u;
        var bayer=array<u32,64>(0u,32u,8u,40u,2u,34u,10u,42u,48u,16u,56u,24u,50u,18u,58u,26u,12u,44u,4u,36u,14u,46u,6u,38u,60u,28u,52u,20u,62u,30u,54u,22u,3u,35u,11u,43u,1u,33u,9u,41u,51u,19u,59u,27u,49u,17u,57u,25u,15u,47u,7u,39u,13u,45u,5u,37u,63u,31u,55u,23u,61u,29u,53u,21u);
        let x=u32(v.position.x)&7u;let y=u32(u.viewport.y-1.-floor(v.position.y))&7u;
        let threshold=(bayer[y*8u+x]+64u-rotation)&63u;
        if (threshold<start || threshold>=start+count) { discard; }
    }
    if (u.display.x <= 0.0) { discard; }
    if (u.display.z == 2.0 && length(v.marker)>1.0) { discard; }
    for (var i=0u; i<u32(u.flags.x); i=i+1u) {
        if (dot(u.planes[i].xyz,v.local)-u.planes[i].w<0.) { discard; }
    }
    if (u.selection.x>0. && u.display.z==0.) {
        var edges=v.barycentric/edge_width;
        if ((v.edge_mask&1u)==0u) { edges.x=10000.; }
        if ((v.edge_mask&2u)==0u) { edges.y=10000.; }
        if ((v.edge_mask&4u)==0u) { edges.z=10000.; }
        if (min(edges.x,min(edges.y,edges.z))>1.6) { discard; }
    }
    if (v.element != 0xffffffffu) {
        if (element_active[v.element] == 0u) { discard; }
    }
    var base = u.color.rgb;
    if (u.flags.w != 0.0 && !front) { base = u.back_color.rgb; }
    if (u.flags.z != 0.0 && (front || u.flags.w == 0.0)) {
        base = face_colors[v.primitive].rgb;
    } else if (u.params.y != 0.0 && (u.params.y == 1.0 || v.element != 0xffffffffu)) {
        if (v.valid < 0.999999) { base = u.invalid.rgb; }
        else {
            var value=v.scalar;
            var line=0.;
            var band=0.;
            for (var j=0u;j<u32(u.contour.y);j=j+1u) {
                let level=u.levels[j/4u][j%4u];
                band+=select(0.,1.,value>=level);
                line=max(line,1.-smoothstep(.4,1.3,abs(value-level)/scalar_width));
            }
            if (u.contour.x==1.) { value=(band+.5)/(u.contour.y+1.); }
            let x = clamp(value, 0.0, 1.0) * 256.0 - 0.5;
            let i = i32(floor(x));
            let a = palette[u32(clamp(i, 0, 255))].rgb;
            let b = palette[u32(clamp(i + 1, 0, 255))].rgb;
            base = mix(a, b, fract(x));
            if (u.contour.x==2.) { base=mix(vec3<f32>(.94),vec3<f32>(.08),line); }
            if (u.contour.z!=0.) {
                let zero=1.-smoothstep(.5,1.5,abs(v.scalar-u.contour.w)/scalar_width);
                base=mix(base,vec3<f32>(.02),zero);
            }
            if (u.emphasis.x!=0.) {
                let distance=(v.scalar-u.emphasis.y)*u.emphasis.x;
                base=mix(base*.25+vec3<f32>(.65),base,smoothstep(-scalar_width,scalar_width,distance));
                base=mix(base,vec3<f32>(.9,.1,.5),1.-smoothstep(.5,1.5,abs(distance)/scalar_width));
            }
        }
    }
    if (u.display.z == 2.0 && u.outline.w != 0.0 && length(v.marker)>1.0-2.0/u.display.y) { base=u.outline.rgb; }
    if (u.params.z>0. && u.display.z==0.) {
        let hatch=abs(fract((v.position.x+v.position.y)/9.)-.5);
        base=mix(base,vec3<f32>(.36,.32,.26),1.-smoothstep(.04,.13,hatch));
    }
    var normal = vec3<f32>(0.0);
    if (u.flags.y != 0.0 && dot(v.normal, v.normal) > 1e-12) {
        normal = normalize(v.normal);
    } else if (dot(flat_normal, flat_normal) > 0.0) {
        normal = normalize(flat_normal);
        if (!front) { normal = -normal; }
    }
    var shade = 1.0;
    if (u.light_params.w != 0.0) {
        let direction = normalize(u.light.xyz);
        let lambert = max(0.0, dot(normal, direction));
        shade = u.light.w + u.light_params.x * lambert;
        if (u.light_params.y > 0.0 && lambert > 0.0) {
            let view = normalize(-v.relative);
            let sum = direction + view;
            if (dot(sum,sum) > 0.0) {
                shade += u.light_params.y * pow(max(0.0,dot(normal,normalize(sum))),u.light_params.z);
            }
        }
    }
    if (shade <= 1.0) { return vec4<f32>(base * shade, u.display.x); }
    return vec4<f32>(mix(base,vec3<f32>(1.0),clamp((shade-1.0)/0.5,0.0,1.0)),u.display.x);
}
@fragment fn fs(v: Varying, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> { return shaded(v,front); }
@fragment fn fs_ao_depth(v: Varying, @builtin(front_facing) front: bool) -> @location(0) f32 {
    _ = shaded(v,front);
    return 1.0/v.position.w;
}
struct OitOutput { @location(0) accumulation: vec4<f32>, @location(1) reveal: vec4<f32>, };
@fragment fn fs_oit(v: Varying,@builtin(front_facing) front: bool) -> OitOutput {
    let color=shaded(v,front);
    let alpha=min(color.a,0.999);
    let depth=1.0/v.position.w;
    let nearness=1.0-clamp((depth-u.viewport.z)/max(1e-6,u.viewport.w-u.viewport.z),0.0,1.0);
    let weight=1.0+3.0*nearness;
    var out: OitOutput;
    out.accumulation=vec4<f32>(color.rgb*alpha*weight,alpha*weight);
    out.reveal=vec4<f32>(alpha);
    return out;
}

"""


def validate_gpu_mesh(mesh, scale=0.):
    """Refuse unrepresentable relative coordinates before retained publication."""
    limit=np.finfo(np.float32).max
    origin=mesh.positions[0] if mesh.node_count else np.zeros(3)
    relative=np.asarray(mesh.positions,dtype=np.float64)-origin
    displacement=np.zeros_like(relative)
    if mesh.displacements is not None and mesh.node_count:
        displacement=np.asarray(mesh.displacements,dtype=np.float64)-mesh.displacements[0]
    with np.errstate(over='ignore',invalid='ignore'):
        deformed=relative+float(scale)*displacement
    domains=[("positions",relative),("displacements",displacement),("deformed positions",deformed)]
    if mesh.normals is not None: domains.append(("normals",mesh.normals))
    for name,values in domains:
        if not np.isfinite(values).all() or np.any(np.abs(values)>limit):
            raise ValueError(f"WebGPU relative {name} exceed float32")


def scalar_payload(mesh, appearance):
    """Float64 rebasing commutes with interpolation; supplied fields are untouched."""
    source = mesh.element_scalars if mesh.element_scalars is not None else mesh.node_scalars
    mode = 2 if mesh.element_scalars is not None else 1 if mesh.node_scalars is not None else 0
    finite = np.empty(0) if source is None else source[np.isfinite(source)]
    bounds = appearance.scalar_range
    if bounds is None:
        bounds = (float(finite.min()), float(finite.max())) if len(finite) else (0., 1.)
    low, high = bounds
    span = max(1e-20, high - low)
    if not np.isfinite(span):
        raise ValueError("WebGPU scalar range span is not representable")
    def packed(values, count):
        result = np.zeros((max(1, count), 4), dtype=np.float32)
        result[:, 1] = 1
        if values is not None:
            valid = np.isfinite(values)
            with np.errstate(over='ignore', invalid='ignore'):
                normalized = (np.asarray(values, dtype=np.float64) - low) / span
            if not np.isfinite(normalized[valid]).all() or np.any(np.abs(normalized[valid]) > np.finfo(np.float32).max):
                raise ValueError("WebGPU normalized scalar domain exceeds float32")
            result[:count, 0] = np.where(valid, normalized, 0).astype(np.float32)
            result[:count, 1] = valid
        return result
    return mode, bounds, packed(mesh.node_scalars, mesh.node_count), packed(mesh.element_scalars, mesh.element_count)


COMPOSITE = """
const GRADIENT:bool = false;
@group(0) @binding(0) var opaque_color: texture_2d<f32>;
@group(0) @binding(1) var accumulation: texture_2d<f32>;
@group(0) @binding(2) var revealage: texture_2d<f32>;
@group(0) @binding(3) var occlusion: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) index:u32)->@builtin(position) vec4<f32> {
    var corners=array<vec2<f32>,3>(vec2<f32>(-1.,-1.),vec2<f32>(3.,-1.),vec2<f32>(-1.,3.));
    return vec4<f32>(corners[index],0.,1.);
}
@fragment fn fs(@builtin(position) p:vec4<f32>)->@location(0) vec4<f32> {
    let xy=vec2<i32>(p.xy);
    var background=textureLoad(opaque_color,xy,0);
    let coverage=background.a;
    if (GRADIENT && background.a < 1.0) {
        let t=f32(xy.y)/f32(textureDimensions(opaque_color).y);
        let gradient=mix(background.rgb*0.91,mix(background.rgb,vec3<f32>(1.),0.22),t);
        background=vec4<f32>(mix(gradient,background.rgb,background.a),1.);
    }
    let sum=textureLoad(accumulation,xy,0);
    let reveal=textureLoad(revealage,xy,0).r;
    let ao_size=textureDimensions(occlusion);
    let ao_xy=min(vec2<i32>(vec2<f32>(xy)*vec2<f32>(ao_size)/vec2<f32>(textureDimensions(opaque_color))),vec2<i32>(ao_size)-vec2<i32>(1));
    background=vec4<f32>(background.rgb*mix(1.,textureLoad(occlusion,ao_xy,0).r,coverage),background.a);
    return vec4<f32>(background.rgb*reveal+sum.rgb/max(sum.a,1e-6)*(1.-reveal),1.);
}
"""



def camera_matrix(camera, width, height):
    """Shared camera convention mapped to WebGPU's [0,1] clip depth."""
    right, up, forward = camera.basis()
    basis = np.array([v.to_tuple() for v in (right, up, forward)])
    view = np.eye(4)
    view[:3, :3] = basis
    view[:3, 3] = -basis @ np.array(camera.position.to_tuple())
    near, far = camera.near, camera.far
    scale = 1 / math.tan(camera.fov / 2)
    projection = np.zeros((4, 4))
    projection[0, 0] = scale / (width / height)
    projection[1, 1] = scale
    projection[2, 2] = far / (far - near)
    projection[2, 3] = -near * far / (far - near)
    projection[3, 2] = 1
    return projection @ view


class WebGPURenderer:
    """Indexed opaque meshes, depth-tested GPU rasterization and RGBA readback.

    Reference positions, indexed node/element references and result storage have
    independent dependencies. Scalar, range, palette and scale updates retain
    position/index buffers. Uniforms are written per object; rasterization and
    perspective clipping run on the GPU, never a CPU-projected image upload.
    """
    def __init__(self):
        import wgpu
        self.wgpu = wgpu
        wanted = "Metal" if sys.platform == "darwin" else "Vulkan"
        adapters = [a for a in wgpu.gpu.enumerate_adapters_sync()
                    if a.info.get("backend_type") == wanted and a.info.get("adapter_type") != "CPU"]
        if not adapters:
            raise RuntimeError(f"WebGPU requires a hardware {wanted} adapter; none available")
        self.adapter = adapters[0]
        self.device = self.adapter.request_device_sync()
        self.info = dict(self.adapter.info)
        self.cache = {}
        self.fields = {}
        self.chunk_views = {}
        self.index_uploads = self.field_uploads = self.palette_updates = 0
        self.pipelines = {}
        self.depth = None
        self.target = None
        self.opaque = self.accumulation = self.reveal = None
        self.size = None
        self.samples = 1
        self.gradient = False
        self.feature_edges = False
        self.multisampled = {}
        from .ssao import AmbientOcclusion
        self.ao=AmbientOcclusion(self.device,wgpu)
        self.ao_quality='Off';self.ao_strength=.8;self.ao_radius=24.;self.ao_effective='Off'
        self.closed = False
        self.frame_count = self.draw_calls = self.geometry_uploads = 0
        try:
            self.shader = self.device.create_shader_module(code=SHADER)
            visibility = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
            self.layout = self.device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": visibility, "buffer": {"type": "uniform"}},
                *[{"binding": i, "visibility": visibility, "buffer": {"type": "read-only-storage"}} for i in range(1,9)]])
            self.pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[self.layout])
        except BaseException:
            self.close()
            raise

    def _pipeline(self, topology, cull, mode="opaque"):
        samples=1 if mode=='ao-depth' else self.samples
        key = topology,cull,mode,samples
        if key not in self.pipelines:
            targets=[{"format":"rgba8unorm","write_mask":0 if mode=="depth" else 15}]
            if mode=='ao-depth':targets=[{'format':'r32float'}]
            if mode=="oit":
                targets=[{"format":"rgba16float","blend":{"color":{"src_factor":"one","dst_factor":"one","operation":"add"},"alpha":{"src_factor":"one","dst_factor":"one","operation":"add"}}},
                         {"format":"rgba16float","blend":{"color":{"src_factor":"zero","dst_factor":"one-minus-src","operation":"add"},"alpha":{"src_factor":"zero","dst_factor":"one-minus-src","operation":"add"}}}]
            if mode=="overlay":
                targets[0]["blend"]={"color":{"src_factor":"src-alpha","dst_factor":"one-minus-src-alpha","operation":"add"},"alpha":{"src_factor":"one","dst_factor":"one-minus-src-alpha","operation":"add"}}
            self.pipelines[key]=self.device.create_render_pipeline(layout=self.pipeline_layout,
                multisample={"count":samples},
                vertex={"module":self.shader,"entry_point":"vs","buffers":[{"array_stride":20,"step_mode":"vertex","attributes":[{"format":"uint32","offset":i*4,"shader_location":i} for i in range(5)]}]},
                primitive={"topology":"triangle-list","front_face":"ccw","cull_mode":"back" if cull else "none"},
                depth_stencil={"format":"depth24plus","depth_write_enabled":mode in {"opaque","depth","ao-depth"},"depth_compare":"always" if mode=="overlay" else "less-equal"},
                fragment={"module":self.shader,"entry_point":"fs_ao_depth" if mode=='ao-depth' else "fs_oit" if mode=="oit" else "fs","targets":targets})
        return self.pipelines[key]

    def _drop_draws(self, key):
        record = self.cache.pop(key, None)
        errors = []
        if record:
            for buffer in record[1]:
                try: buffer.destroy()
                except Exception as error: errors.append(str(error))
        if errors: raise RuntimeError("WebGPU buffer cleanup: " + "; ".join(errors))

    def remove(self, key):
        errors = []
        try: self._drop_draws(key)
        except Exception as error: errors.append(str(error))
        fields = getattr(self, "fields", {}).pop(key, None)
        if fields:
            for buffer in fields["buffers"]:
                try: buffer.destroy()
                except Exception as error: errors.append(str(error))
        if errors: raise RuntimeError("WebGPU cleanup: " + "; ".join(errors))

    def _sync_fields(self, handle, appearance):
        from ..appearance import palette_data
        key, mesh = id(handle), handle.mesh
        record = self.fields.get(key)
        shape=(mesh.node_count,mesh.element_count,mesh.triangle_count)
        if record is not None and record.get("shape")!=shape:
            self.remove(key);record=None
        created = record is None
        dependency=(id(mesh.positions),id(mesh.displacements),id(mesh.node_scalars),
                    id(mesh.element_scalars),id(mesh.active_elements),id(mesh.normals),
                    appearance.get("face_colors"),handle.appearance)
        if not created and record["dependency"] == dependency:
            return record
        if created:
            record = {"buffers": [], "signatures": {}}
        signatures = record["signatures"]
        try:
            origin = np.array(mesh.positions[0] if mesh.node_count else [0., 0., 0.], dtype=np.float64)
            position = np.zeros((max(1,mesh.node_count),4),dtype=np.float32)
            position[:mesh.node_count,:3] = mesh.positions-origin
            disp = np.zeros_like(position)
            displacement_origin = np.zeros(3,dtype=np.float64)
            if mesh.displacements is not None and mesh.node_count:
                displacement_origin = np.array(mesh.displacements[0],dtype=np.float64)
                disp[:mesh.node_count,:3] = mesh.displacements-displacement_origin
            mode, bounds, nodes, elements = scalar_payload(mesh,handle.appearance)
            active = np.ones(max(1,mesh.element_count),dtype=np.uint32)
            if mesh.active_elements is not None: active[:mesh.element_count] = mesh.active_elements
            palette = np.ones((256,4),dtype=np.float32)
            palette[:,:3] = np.frombuffer(palette_data(handle.appearance.colormap),dtype=np.float32).reshape(256,3)
            normal_values=np.zeros_like(position)
            if mesh.normals is not None: normal_values[:mesh.node_count,:3]=mesh.normals
            face_values=np.ones((max(1,mesh.triangle_count),4),dtype=np.float32)
            colors=appearance.get("face_colors")
            if colors:
                from ..face_colors import normalize_face_colors
                _,rgb=normalize_face_colors(colors)
                if len(rgb)==mesh.triangle_count: face_values[:mesh.triangle_count,:3]=rgb
                else:
                    mapping=np.arange(mesh.triangle_count) if mesh.triangle_to_element is None else mesh.triangle_to_element
                    face_values[:mesh.triangle_count,:3]=rgb[mapping]
            data = [position,disp,nodes,elements,active,palette,normal_values,face_values]
            identity = [id(mesh.positions),id(mesh.displacements),
                (id(mesh.node_scalars),bounds),(id(mesh.element_scalars),bounds),
                id(mesh.active_elements),handle.appearance.colormap,id(mesh.normals),colors]
            for index,(array,signature) in enumerate(zip(data,identity)):
                if created:
                    buffer = self.device.create_buffer_with_data(data=array,usage=self.wgpu.BufferUsage.STORAGE | self.wgpu.BufferUsage.COPY_DST)
                    record["buffers"].append(buffer)
                elif signatures[index] != signature:
                    self.device.queue.write_buffer(record["buffers"][index],0,array)
                else: continue
                signatures[index] = signature
                if index == 0: self.geometry_uploads += 1
                elif index == 5: self.palette_updates += 1
                else: self.field_uploads += 1
            record.update(shape=shape,origin=origin,displacement_origin=displacement_origin,mode=mode,bounds=bounds,dependency=dependency,
                # Keep identity dependencies alive: Python object IDs may be
                # reused after consecutive retained updates before a frame.
                references=(mesh.positions,mesh.displacements,mesh.node_scalars,
                    mesh.element_scalars,mesh.active_elements,mesh.triangles,mesh.lines,mesh.normals,mesh.point_indices))
            self.fields[key] = record
            return record
        except BaseException:
            if created:
                for buffer in record["buffers"]:
                    try: buffer.destroy()
                    except Exception: pass
            raise

    def _prepare(self, handle, masks, appearance, highlight):
        from ..core import parse_color
        key, mesh = id(handle), handle.mesh
        previous_fields=self.fields.get(key)
        fields = self._sync_fields(handle,appearance)
        created_fields=fields is not previous_fields
        mask_identity = tuple(sorted((name,np.asarray(values).dtype.str,np.asarray(values).shape,np.asarray(values).tobytes()) for name,values in masks.items()))
        styles = tuple(sorted((name,value) for name,value in appearance.items() if name not in {"scalar_range","colormap","face_colors","lit","back_color","opacity","line_width","point_size","point_outline","line_overlay","point_overlay","depth_only","layer","stipple","stipple_phase"}))
        signature = (id(mesh.triangles),id(mesh.lines),id(mesh.point_indices),mask_identity,handle.selected_elements.tobytes(),styles,highlight,getattr(self,'feature_edges',False))
        if key in self.cache and self.cache[key][0] == signature:
            return self.cache[key][2]
        self._drop_draws(key)
        owned,draws = [],[]
        try:
            for topology,indices,suffix,color,mapping in (
                ("triangle-list",mesh.triangles,"elements",appearance["color"],mesh.triangle_to_element),
                ("line-list",mesh.lines,"lines",appearance["line_color"],mesh.line_to_element),
                ("point-list",None if mesh.point_indices is None else mesh.point_indices[:,None],"points",appearance.get("point_color","#2563eb"),None)):
                if indices is None or not len(indices): continue
                ids = np.arange(len(indices),dtype=np.uint32) if mapping is None else mapping
                element_ids = ids if topology == "triangle-list" or mapping is not None else np.full(len(indices),0xffffffff,dtype=np.uint32)
                semantic_ids = ids if suffix == "elements" else np.arange(len(indices))
                visible = ~np.isin(semantic_ids,masks.get("hidden_"+suffix,()))
                if (getattr(self,'feature_edges',False) and suffix=='lines' and appearance.get('mesh_lines')
                        and mesh.triangles is not None and not len(handle.selected_elements)
                        and not any(len(v) for v in masks.values())):
                    from .quality import feature_line_mask
                    visible &= feature_line_mask(mesh.positions,mesh.triangles,indices)
                chosen = np.union1d(masks.get("selected_"+suffix,()),masks.get("preselected_"+suffix,()))
                if suffix == "elements": chosen = np.union1d(chosen,handle.selected_elements)
                selected = np.isin(semantic_ids,chosen)
                if suffix=="lines" and mapping is not None:
                    visible &= ~np.isin(mapping,masks.get("hidden_elements",()))
                    selected |= np.isin(mapping,np.union1d(handle.selected_elements,np.union1d(masks.get("selected_elements",()),masks.get("preselected_elements",()))))
                # Static corner references point into shared indexed scientific
                # arrays. Element identity is constant for every primitive.
                if topology == "triangle-list":
                    # Mark boundaries of the visible selected region, not fill diagonals.
                    edge_masks=np.zeros(len(indices),dtype=np.uint32)
                    selected_rows=np.flatnonzero(visible & selected)
                    if len(selected_rows):
                        # Flat/smooth shading can split coincident display vertices.
                        # Outline geometric boundaries, not those storage seams.
                        _,outline_vertices=np.unique(mesh.positions,axis=0,return_inverse=True)
                        triangles=outline_vertices[indices[selected_rows]]
                        edges=np.sort(triangles[:,[[1,2],[2,0],[0,1]]].reshape(-1,2),axis=1)
                        _,inverse,counts=np.unique(edges,axis=0,return_inverse=True,return_counts=True)
                        boundary=(counts[inverse]!=2).reshape(-1,3)
                        edge_masks[selected_rows]=(boundary*np.array([1,2,4],dtype=np.uint32)).sum(axis=1)
                    references=np.column_stack((indices.ravel(),np.repeat(element_ids,3),np.repeat(np.arange(len(indices)),3),np.repeat(edge_masks,3),np.tile([0,1,2],len(indices))))
                    corners=np.arange(indices.size,dtype=np.uint32).reshape(indices.shape)
                elif topology == "line-list":
                    nodes=indices[:,[0,0,1,0,1,1]]
                    others=indices[:,[1,1,0,1,0,0]]
                    references=np.column_stack((nodes.ravel(),np.repeat(element_ids,6),np.repeat(np.arange(len(indices)),6),others.ravel(),np.tile([0,1,2,0,2,3],len(indices))))
                    corners=np.arange(len(indices)*6,dtype=np.uint32).reshape(-1,6)
                else:
                    nodes=np.repeat(indices[:,0],6)
                    references=np.column_stack((nodes,np.repeat(element_ids,6),np.repeat(np.arange(len(indices)),6),nodes,np.tile([0,1,2,0,2,3],len(indices))))
                    corners=np.arange(len(indices)*6,dtype=np.uint32).reshape(-1,6)
                references=references.astype(np.uint32)
                vertex = self.device.create_buffer_with_data(data=references,usage=self.wgpu.BufferUsage.VERTEX)
                owned.append(vertex)
                for admit,tint,is_selected in ((visible & ~selected,color,False),(visible & selected,highlight,True)):
                    data = np.ascontiguousarray(corners[admit].ravel())
                    if not len(data): continue
                    index = self.device.create_buffer_with_data(data=data,usage=self.wgpu.BufferUsage.INDEX)
                    owned.append(index);self.index_uploads += 1
                    uniform = self.device.create_buffer(size=592,usage=self.wgpu.BufferUsage.UNIFORM | self.wgpu.BufferUsage.COPY_DST)
                    owned.append(uniform)
                    entries = [{"binding":0,"resource":{"buffer":uniform,"offset":0,"size":592}}]
                    entries += [{"binding":i+1,"resource":{"buffer":buffer}} for i,buffer in enumerate(fields["buffers"])]
                    bind = self.device.create_bind_group(layout=self.layout,entries=entries)
                    rgb = np.asarray(parse_color(tint),dtype=np.float32)/255
                    draws.append((vertex,index,uniform,bind,len(data),topology,
                        bool(appearance["cull_backface"] and topology=="triangle-list"),rgb,is_selected))
            self.cache[key] = signature,owned,draws
            return draws
        except BaseException:
            for buffer in owned:
                try: buffer.destroy()
                except Exception: pass
            if created_fields: self.remove(key)
            raise

    def render(self, entries, masks, camera, size, background, highlight="#ff8c00", *, light=None, shading=False, section=None, pixel_ratio=1., occlude_lines=True, chunk_masks=None, mesh_lines=True,interactive=False,contours=None,selection_outline=(True,False),readback=True,shadows=("Off",.5)):
        if self.closed:
            raise RuntimeError("WebGPU renderer is closed")
        from ..core import parse_color
        if self.chunk_views or any(entry["handle"].chunks for entry in entries.values()):
            from .scene import expand_chunks
            entries,masks=expand_chunks(self,entries,masks,chunk_masks or {})
        width, height = max(1, int(size[0])), max(1, int(size[1]))
        if self.size != (width, height,self.samples):
            for texture in self.multisampled.values():texture.destroy()
            self.multisampled={}
            for texture in (self.target, self.depth, getattr(self,"opaque",None), getattr(self,"accumulation",None), getattr(self,"reveal",None)):
                if texture is not None:
                    texture.destroy()
            self.target = self.device.create_texture(size=(width, height, 1), format="rgba8unorm", usage=self.wgpu.TextureUsage.RENDER_ATTACHMENT | self.wgpu.TextureUsage.COPY_SRC | self.wgpu.TextureUsage.TEXTURE_BINDING)
            self.depth = self.device.create_texture(size=(width, height, 1), sample_count=self.samples, format="depth24plus", usage=self.wgpu.TextureUsage.RENDER_ATTACHMENT)
            self.opaque=self.device.create_texture(size=(width,height,1),format="rgba8unorm",usage=self.wgpu.TextureUsage.RENDER_ATTACHMENT|self.wgpu.TextureUsage.TEXTURE_BINDING)
            self.accumulation=self.device.create_texture(size=(width,height,1),format="rgba16float",usage=self.wgpu.TextureUsage.RENDER_ATTACHMENT|self.wgpu.TextureUsage.TEXTURE_BINDING)
            self.reveal=self.device.create_texture(size=(width,height,1),format="rgba16float",usage=self.wgpu.TextureUsage.RENDER_ATTACHMENT|self.wgpu.TextureUsage.TEXTURE_BINDING)
            if self.samples>1:
                for name,fmt in (('target','rgba8unorm'),('opaque','rgba8unorm'),('accumulation','rgba16float'),('reveal','rgba16float')):
                    self.multisampled[name]=self.device.create_texture(size=(width,height,1),sample_count=self.samples,format=fmt,usage=self.wgpu.TextureUsage.RENDER_ATTACHMENT)
            self.size = width, height,self.samples
        for buffer in getattr(self,"_frame_buffers",[]): buffer.destroy()
        self._frame_buffers=[]
        jobs=[]
        matrix = camera_matrix(camera, width, height)
        active = set()
        for entry in sorted(entries.values(),key=lambda item:item.get("layer",0)):
            handle = entry["handle"]
            active.add(id(handle))
            if handle.removed or not handle.visible:
                continue
            for vertex, index, uniform, bind, count, topology, cull, rgb, selected in self._prepare(handle, masks.get(id(handle), {}), entry["appearance"], highlight):
                # Recenter before float32 conversion. Absolute world/camera and
                # affine origin composition remain float64, preserving small
                # geometry at large engineering-coordinate translations.
                if topology=="line-list" and entry["appearance"].get("mesh_lines") and not mesh_lines:continue
                from .scene import stipple_window
                pattern=entry["appearance"].get("stipple","") if topology=="triangle-list" else ""
                start,pattern_count,rotation=stipple_window(pattern,entry["appearance"].get("stipple_phase",0))
                pattern_code=start+(pattern_count<<6)+(rotation<<13) if pattern else 0
                # Surface opacity controls the fill, not its mesh-outline lines.
                # Keep outlines visible in Wireframe (fill opacity 0) and when
                # surfaces are transparent; explicit non-outline lines retain
                # their own opacity policy.
                outline_line=topology=='line-list' and entry['appearance'].get('mesh_lines')
                opacity=1. if outline_line or (pattern and pattern_count<64) else entry["appearance"]["opacity"]
                fields = self.fields[id(handle)]
                origin = fields["origin"] + handle.deformation_scale*fields["displacement_origin"]
                model = np.array(handle.transform, dtype=np.float64, copy=True)
                model[:3, 3] += model[:3, :3] @ origin
                invalid = np.asarray(parse_color(entry["appearance"]["invalid_color"]),dtype=np.float32)/255
                from ..shading import Light
                configured=Light() if light is None else light
                relative_model=np.array(model,copy=True)
                relative_model[:3,3]-=np.array(camera.position.to_tuple(),dtype=np.float64)
                normal_matrix=np.eye(4)
                smooth=handle.mesh.normals is not None
                if smooth:
                    try:
                        inverse=np.linalg.inv(model[:3,:3]).T
                        smooth=bool(np.isfinite(inverse).all())
                        if smooth:
                            magnitude=np.max(np.abs(inverse))
                            normal_matrix[:3,:3]=inverse/magnitude if magnitude>np.finfo(np.float32).max else inverse
                    except np.linalg.LinAlgError: smooth=False
                plane=np.zeros(4,dtype=np.float64)
                planes=np.zeros((7,4),dtype=np.float64)
                sections=() if section is None or not section.enabled else getattr(section,'planes',(section,))
                clipping=len(sections)
                for i,part in enumerate(sections):
                    normal=np.array(part.normal.to_tuple(),dtype=np.float64)
                    planes[i,:3]=model[:3,:3].T@normal
                    planes[i,3]=part.offset-normal@model[:3,3]
                from ..display import contour_uniforms
                contour,levels,emphasis=contour_uniforms(contours,fields['bounds'])
                lit=bool(shading and configured.enabled and entry["appearance"]["lit"] and topology=="triangle-list" and not selected)
                direction=configured.world_direction(camera.basis())
                colors=entry["appearance"].get("face_colors")
                back=entry["appearance"].get("back_color","")
                back_rgb=np.asarray(parse_color(back) if back else (0,0,0),dtype=np.float32)/255
                values = np.concatenate(((matrix @ model).T.ravel(), rgb, [1.], invalid, [1.],
                    [handle.deformation_scale, 0 if selected or topology=="point-list" else fields["mode"], float(entry["appearance"].get("_cut_cap_hatch",False)), 0.],
                    relative_model.T.ravel(),normal_matrix.T.ravel(),plane,
                    direction,[configured.ambient],[configured.diffuse,configured.specular,configured.shininess,float(lit)],
                    [float(clipping),float(smooth and topology=="triangle-list"),float(bool(colors) and topology=="triangle-list" and not selected),float(bool(back) and topology=="triangle-list" and not selected)],
                    back_rgb,[1.],
                    [opacity,(entry["appearance"]["point_size"] if topology=="point-list" else entry["appearance"]["line_width"])*pixel_ratio,2. if topology=="point-list" else 1. if topology=="line-list" else 0.,pattern_code],
                    [width,height,camera.near,camera.far],
                    np.asarray(parse_color(entry["appearance"].get("point_outline") or "black"),dtype=np.float32)/255,[float(bool(entry["appearance"].get("point_outline")))],planes.ravel(),contour,levels,emphasis,[0.,0.,0.,0.])).astype(np.float32)
                self.device.queue.write_buffer(uniform, 0, values)
                appearance=entry["appearance"]
                overlay=appearance.get("line_overlay") if topology=="line-list" else appearance.get("point_overlay") if topology=="point-list" else False
                overlay=bool(overlay or topology=="line-list" and not occlude_lines)
                mode="depth" if appearance.get("depth_only") else "overlay" if overlay else "oit" if opacity<.999 else "opaque"
                jobs.append((mode,topology,cull,vertex,index,uniform,bind,count))
                if selected and topology=='triangle-list' and selection_outline[0]:
                    outline_values=values.copy()
                    outline_values[16:19]=np.asarray(parse_color('#181818'),dtype=np.float32)/255
                    outline_values[-4]=1.
                    outline_uniform=self.device.create_buffer_with_data(data=outline_values,usage=self.wgpu.BufferUsage.UNIFORM)
                    outline_bind=self.device.create_bind_group(layout=self.layout,entries=[{'binding':0,'resource':{'buffer':outline_uniform}},*({'binding':i+1,'resource':{'buffer':buffer}} for i,buffer in enumerate(fields['buffers']))])
                    self._frame_buffers.append(outline_uniform)
                    jobs.append(('overlay' if selection_outline[1] else 'outline',topology,False,vertex,index,outline_uniform,outline_bind,count))
        self.ao_effective='Low' if interactive and self.ao_quality=='High' else self.ao_quality
        if self.ao_strength==0 or not any(job[0]=='opaque' and job[1]=='triangle-list' for job in jobs):self.ao_effective='Off'
        self.shadow_effective=shadows[0]
        self._shadow_settings=shadows
        self._shadow_light=(light.world_direction(camera.basis()) if light is not None else (1.,1.,1.))
        self._depth_effective=self.ao_effective if self.ao_effective!='Off' else shadows[0]
        self._execute(jobs,width,height,background,camera,pixel_ratio)
        for key in (set(self.cache) | set(self.fields)) - active:
            self.remove(key)
        self.frame_count += 1
        if not readback: return self.target
        data = self.device.queue.read_texture({"texture": self.target}, {"offset": 0, "bytes_per_row": width * 4, "rows_per_image": height}, (width, height, 1))
        return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4).copy()

    def _execute(self,jobs,width,height,background,camera,pixel_ratio):
        from ..core import parse_color
        encoder=self.device.create_command_encoder()
        def attachment(texture,color,load="clear"):
            multisampled=next((self.multisampled.get(name) for name in ('target','opaque','accumulation','reveal') if getattr(self,name) is texture),None)
            return {"view":(multisampled or texture).create_view(),"resolve_target":texture.create_view() if multisampled is not None else None,"clear_value":color,"load_op":load,"store_op":"store"}
        def depth(load="load"):
            return {"view":self.depth.create_view(),"depth_clear_value":1.,"depth_load_op":load,"depth_store_op":"store"}
        def draw(render_pass,modes):
            for mode,topology,cull,vertex,index,uniform,bind,count in jobs:
                if mode not in modes:continue
                render_pass.set_pipeline(self._pipeline(topology,cull,mode))
                render_pass.set_bind_group(0,bind);render_pass.set_vertex_buffer(0,vertex);render_pass.set_index_buffer(index,"uint32")
                render_pass.draw_indexed(count);self.draw_calls+=1
        color=tuple(c/255 for c in parse_color(background))+(0.,)
        render_pass=encoder.begin_render_pass(color_attachments=[attachment(self.opaque,color)],depth_stencil_attachment=depth("clear"))
        draw(render_pass,{"depth"});draw(render_pass,{"opaque"});render_pass.end()
        render_pass=encoder.begin_render_pass(color_attachments=[attachment(self.accumulation,(0.,0.,0.,0.)),attachment(self.reveal,(1.,1.,1.,1.))],depth_stencil_attachment=depth())
        draw(render_pass,{"oit"});render_pass.end()
        ao_texture=self.ao.prepare(width,height,self._depth_effective)
        if self._depth_effective!='Off':
            render_pass=encoder.begin_render_pass(color_attachments=[{'view':self.ao.textures['linear'].create_view(),'resolve_target':None,'clear_value':(0.,0.,0.,0.),'load_op':'clear','store_op':'store'}],
                depth_stencil_attachment={'view':self.ao.textures['depth'].create_view(),'depth_clear_value':1.,'depth_load_op':'clear','depth_store_op':'store'})
            for mode,topology,cull,vertex,index,uniform,bind,count in jobs:
                if mode!='opaque' or topology!='triangle-list':continue
                render_pass.set_pipeline(self._pipeline(topology,cull,'ao-depth'))
                render_pass.set_bind_group(0,bind);render_pass.set_vertex_buffer(0,vertex);render_pass.set_index_buffer(index,'uint32')
                render_pass.draw_indexed(count);self.draw_calls+=1
            render_pass.end()
            self.ao.execute(encoder,camera,self._depth_effective,self.ao_strength if self.ao_effective!="Off" else 0.,self.ao_radius,pixel_ratio,self._shadow_settings,self._shadow_light)
        if not hasattr(self,"composite_pipeline") or self.composite_samples!=(self.samples,self.gradient):
            module=self.device.create_shader_module(code=COMPOSITE.replace('const GRADIENT:bool = false;',f'const GRADIENT:bool = {str(self.gradient).lower()};'))
            self.composite_pipeline=self.device.create_render_pipeline(layout="auto",multisample={"count":self.samples},vertex={"module":module,"entry_point":"vs"},fragment={"module":module,"entry_point":"fs","targets":[{"format":"rgba8unorm"}]})
            self.composite_samples=(self.samples,self.gradient)
        bind=self.device.create_bind_group(layout=self.composite_pipeline.get_bind_group_layout(0),entries=[{"binding":i,"resource":texture.create_view()} for i,texture in enumerate((self.opaque,self.accumulation,self.reveal,ao_texture))])
        render_pass=encoder.begin_render_pass(color_attachments=[attachment(self.target,color)])
        render_pass.set_pipeline(self.composite_pipeline);render_pass.set_bind_group(0,bind);render_pass.draw(3);render_pass.end()
        render_pass=encoder.begin_render_pass(color_attachments=[attachment(self.target,color,"load")],depth_stencil_attachment=depth())
        draw(render_pass,{"outline"});draw(render_pass,{"overlay"});render_pass.end()
        self.device.queue.submit([encoder.finish()])

    def close(self):
        if self.closed:
            return
        self.closed = True
        errors = []
        def attempt(callback):
            try:
                callback()
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
        for buffer in getattr(self,"_frame_buffers",[]):attempt(buffer.destroy)
        self._frame_buffers=[]
        for key in set(self.cache) | set(getattr(self,"fields",{})):
            attempt(lambda key=key: self.remove(key))
        if hasattr(self,'ao'):attempt(self.ao.close)
        for texture in getattr(self,'multisampled',{}).values():attempt(texture.destroy)
        self.multisampled={}
        for texture in (self.target, self.depth, getattr(self,"opaque",None), getattr(self,"accumulation",None), getattr(self,"reveal",None)):
            if texture is not None:
                attempt(texture.destroy)
        self.target = self.depth = self.opaque = self.accumulation = self.reveal = None
        self.pipelines.clear()
        getattr(self,"chunk_views",{}).clear()
        attempt(self.device.destroy)
        if errors:
            raise RuntimeError("WebGPU cleanup failures: " + "; ".join(errors))
