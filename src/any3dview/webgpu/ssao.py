"""Opaque screen-space ambient occlusion with depth-aware separable filtering."""
import math
import numpy as np

SHADER = """
struct Params { size:vec4<f32>, settings:vec4<f32>, light:vec4<f32>, shadow:vec4<f32>, };
@group(0) @binding(0) var<uniform> u:Params;
@group(0) @binding(1) var depths:texture_2d<f32>;
@group(0) @binding(2) var source:texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32> {
    var corners=array<vec2<f32>,3>(vec2<f32>(-1.,-1.),vec2<f32>(3.,-1.),vec2<f32>(-1.,3.));
    return vec4<f32>(corners[i],0.,1.);
}
fn depth(p:vec2<i32>)->f32 {
    if (any(p<vec2<i32>(0)) || any(p>=vec2<i32>(u.size.xy))) { return 0.; }
    return textureLoad(depths,p,0).r;
}
fn position(p:vec2<i32>,z:f32)->vec3<f32> {
    let xy=(vec2<f32>(p)+vec2<f32>(.5))/u.size.xy;
    return vec3<f32>((xy.x*2.-1.)*u.size.z*z,(1.-xy.y*2.)*u.size.w*z,z);
}
@fragment fn fs_ao(@builtin(position) pixel:vec4<f32>)->@location(0) f32 {
    let p=vec2<i32>(pixel.xy); let z=depth(p);
    if (z<=0.) { return 1.; }
    let c=position(p,z);
    let r=p+vec2<i32>(1,0); let l=p-vec2<i32>(1,0);
    let b=p+vec2<i32>(0,1); let t=p-vec2<i32>(0,1);
    let zr=depth(r);let zl=depth(l);let zb=depth(b);let zt=depth(t);
    if ((zr<=0. && zl<=0.) || (zb<=0. && zt<=0.)) { return 1.; }
    var dx=position(r,zr)-c;var dy=position(b,zb)-c;
    if (zr<=0. || (zl>0. && abs(zl-z)<abs(zr-z))) { dx=c-position(l,zl); }
    if (zb<=0. || (zt>0. && abs(zt-z)<abs(zb-z))) { dy=c-position(t,zt); }
    var n=cross(dx,dy);
    if (dot(n,n)<1e-18) { n=vec3<f32>(0.,0.,-1.); } else { n=normalize(n); }
    if (n.z>0.) { n = -n; }
    let radius=u.settings.x;
    let world_radius=max(2.*z*u.size.w*radius/u.size.y,1e-6);
    let rotation=fract(sin(dot(vec2<f32>(p),vec2<f32>(12.9898,78.233)))*43758.5453)*6.2831853;
    var sum=0.;let count=u32(u.settings.z);
    for (var i=0u;i<count;i=i+1u) {
        let angle=f32(i)*2.39996323+rotation;
        let step=radius*sqrt((f32(i)+.5)/f32(count));
        let q=p+vec2<i32>(round(vec2<f32>(cos(angle),sin(angle))*step));
        let other=depth(q);
        if (other<=0.) { continue; }
        let delta=position(q,other)-c;let distance=length(delta);
        if (distance<1e-6 || distance>=world_radius) { continue; }
        let hemisphere=max(dot(n,delta/distance)-.025,0.);
        sum+=hemisphere*(1.-distance/world_radius);
    }
    var visibility=1.;
    if (u.shadow.x>0.) {
        let distance=max(2.*z*u.size.w*96./u.size.y,1e-5);
        let bias=max(distance*.004,z*.0001);
        var blocked=0.;
        let steps=u32(u.shadow.y);
        for (var i=1u;i<=steps;i=i+1u) {
            let ray=c+u.light.xyz*distance*f32(i)/f32(steps);
            if (ray.z<=0.) { continue; }
            let uv=vec2<f32>(ray.x/(ray.z*u.size.z)*.5+.5,.5-ray.y/(ray.z*u.size.w)*.5);
            let sample_depth=depth(vec2<i32>(uv*u.size.xy));
            let delta=ray.z-sample_depth;
            if (sample_depth>0. && delta>bias && delta<distance*.15) { blocked+=1.; }
        }
        visibility=1.-u.shadow.x*min(1.,blocked/2.);
    }
    return clamp(1.-2.*u.settings.y*sum/f32(count),0.,1.)*visibility;
}
@fragment fn fs_blur(@builtin(position) pixel:vec4<f32>)->@location(0) f32 {
    let p=vec2<i32>(pixel.xy);let z=depth(p);
    if (z<=0.) { return 1.; }
    let direction=select(vec2<i32>(0,1),vec2<i32>(1,0),u.settings.w==1.);
    var weights=array<f32,5>(1.,4.,6.,4.,1.);var sum=0.;var total=0.;
    let range=max(2.*z*u.size.w*u.settings.x/u.size.y*.2,1e-6);
    for (var i=0;i<5;i=i+1) {
        let q=p+direction*(i-2);let other=depth(q);
        if (other<=0.) { continue; }
        let weight=weights[u32(i)]*exp(-abs(other-z)/range);
        sum+=weight*textureLoad(source,q,0).r;total+=weight;
    }
    return sum/max(total,1e-6);
}
"""


def validated_settings(quality,strength,radius):
    if quality not in ('Off','Low','High'):raise ValueError('ambient occlusion must be Off, Low or High')
    strength=float(strength);radius=float(radius)
    if not math.isfinite(strength) or not 0<=strength<=2:raise ValueError('AO strength must be between 0 and 2')
    if not math.isfinite(radius) or not 1<=radius<=256:raise ValueError('AO radius must be between 1 and 256 pixels')
    return quality,strength,radius


class AmbientOcclusion:
    def __init__(self,device,wgpu):
        self.device=device;self.wgpu=wgpu;self.textures={};self.buffers=[];self.size=None;self.pipeline=None
        self.white=None;self.passes=0;self.pool={};self.viewport=None

    def white_texture(self):
        if self.white is None:
            self.white=self.device.create_texture(size=(1,1,1),format='r8unorm',usage=self.wgpu.TextureUsage.TEXTURE_BINDING|self.wgpu.TextureUsage.COPY_DST)
            self.device.queue.write_texture({'texture':self.white},bytes([255]),{'bytes_per_row':1,'rows_per_image':1},(1,1,1))
        return self.white

    def prepare(self,width,height,quality):
        if quality=='Off':return self.white_texture()
        if self.viewport!=(width,height):
            for textures in self.pool.values():
                for texture in textures.values():texture.destroy()
            self.pool={};self.viewport=(width,height)
        divisor=2 if quality=='Low' else 1
        size=(max(1,(width+divisor-1)//divisor),max(1,(height+divisor-1)//divisor))
        self.aspect=width/height;self.scale=size[1]/height
        if size not in self.pool:
            self.textures={}
            for name,fmt in (('linear','r32float'),('depth','depth24plus'),('raw','r8unorm'),('horizontal','r8unorm'),('filtered','r8unorm')):
                usage=self.wgpu.TextureUsage.RENDER_ATTACHMENT
                if name!='depth':usage|=self.wgpu.TextureUsage.TEXTURE_BINDING
                self.textures[name]=self.device.create_texture(size=(*size,1),format=fmt,usage=usage)
            self.pool[size]=self.textures
        self.textures=self.pool[size];self.size=size
        return self.textures['filtered']

    def execute(self,encoder,camera,quality,strength,radius,pixel_ratio,shadow=("Off",.5),light=None):
        if self.pipeline is None:
            visibility=self.wgpu.ShaderStage.FRAGMENT
            self.layout=self.device.create_bind_group_layout(entries=[
                {'binding':0,'visibility':visibility,'buffer':{'type':'uniform'}},
                *[{'binding':i,'visibility':visibility,'texture':{'sample_type':'unfilterable-float'}} for i in (1,2)]])
            layout=self.device.create_pipeline_layout(bind_group_layouts=[self.layout]);shader=self.device.create_shader_module(code=SHADER)
            self.pipeline={name:self.device.create_render_pipeline(layout=layout,
                vertex={'module':shader,'entry_point':'vs'},fragment={'module':shader,'entry_point':name,'targets':[{'format':'r8unorm'}]}) for name in ('fs_ao','fs_blur')}
            self.buffers=[self.device.create_buffer(size=64,usage=self.wgpu.BufferUsage.UNIFORM|self.wgpu.BufferUsage.COPY_DST) for _ in range(3)]
        width,height=self.size;focal=1/math.tan(camera.fov/2)
        radius=radius*pixel_ratio*self.scale
        direction=np.array((1.,1.,1.) if light is None else light)
        direction=direction/max(np.linalg.norm(direction),1e-12)
        basis=np.array([v.to_tuple() for v in camera.basis()]);direction=basis@direction
        light_direction=direction
        shadow_strength=shadow[1] if shadow[0]!='Off' else 0.
        for i,(name,source,target,direction) in enumerate((('fs_ao',self.white_texture(),self.textures['raw'],0),
                ('fs_blur',self.textures['raw'],self.textures['horizontal'],1),
                ('fs_blur',self.textures['horizontal'],self.textures['filtered'],2))):
            values=np.array([width,height,self.aspect/focal,1/focal,radius,strength,8 if quality=='Low' else 16,direction,*light_direction,0.,shadow_strength,8 if shadow[0]=='Low' else 24,0.,0.],dtype=np.float32)
            self.device.queue.write_buffer(self.buffers[i],0,values)
            bind=self.device.create_bind_group(layout=self.layout,entries=[
                {'binding':0,'resource':{'buffer':self.buffers[i],'size':64}},
                {'binding':1,'resource':self.textures['linear'].create_view()},
                {'binding':2,'resource':source.create_view()}])
            render=encoder.begin_render_pass(color_attachments=[{'view':target.create_view(),'resolve_target':None,'clear_value':(1.,1.,1.,1.),'load_op':'clear','store_op':'store'}])
            render.set_pipeline(self.pipeline[name]);render.set_bind_group(0,bind);render.draw(3);render.end();self.passes+=1

    def close(self):
        for textures in self.pool.values():
            for texture in textures.values():texture.destroy()
        for buffer in self.buffers:buffer.destroy()
        if self.white is not None:self.white.destroy()
        self.textures={};self.pool={};self.buffers=[];self.white=None
