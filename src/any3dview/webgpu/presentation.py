"""Explicit bitmap and optional rendercanvas screen presentation targets.

Native is a prototype and is never selected automatically without promotion.
Scene pixels remain on the GPU; only the shared QPainter HUD is uploaded.
"""
PRESENT_SHADER="""
const OUTPUT_SRGB: bool = false;
@group(0) @binding(0) var scene:texture_2d<f32>;
@group(0) @binding(1) var hud:texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32> {
 var p=array<vec2<f32>,3>(vec2<f32>(-1.,-1.),vec2<f32>(3.,-1.),vec2<f32>(-1.,3.));
 return vec4<f32>(p[i],0.,1.);
}
@fragment fn fs(@builtin(position) p:vec4<f32>)->@location(0) vec4<f32> {
 let a=textureLoad(scene,vec2<i32>(p.xy),0);
 let b=textureLoad(hud,vec2<i32>(p.xy),0);
 // Scene and QPainter HUD textures contain encoded display RGB bytes.
 // sRGB render attachments encode shader output automatically, so undo
 // that encoding first to retain the exact bitmap palette/overlay colors.
 var rgb=mix(a.rgb,b.rgb,b.a);
 if (OUTPUT_SRGB) {
   rgb=select(pow((rgb+vec3<f32>(.055))/1.055,vec3<f32>(2.4)),rgb/12.92,rgb<=vec3<f32>(.04045));
 }
 return vec4<f32>(rgb,1.);
}
"""

def presentation_shader(format):
    """Select output transfer for the actual configured surface attachment."""
    srgb=str(format).endswith('-srgb')
    return PRESENT_SHADER.replace('const OUTPUT_SRGB: bool = false;',f'const OUTPUT_SRGB: bool = {str(srgb).lower()};')


class BitmapTarget:
    method='bitmap'
    def __init__(self,owner): self.owner=owner
    def render(self): return self.owner._render_pixels()
    def close(self): pass


def render_hud_image(owner,width,height):
    """Raster shared logical-pixel HUD into a physical texture at actual DPR."""
    from PySide6.QtGui import QImage,QPainter
    from PySide6.QtCore import Qt
    ratio=float(owner.pixel_ratio)
    image=QImage(width,height,QImage.Format_RGBA8888)
    image.setDevicePixelRatio(ratio)
    image.fill(Qt.transparent)
    painter=QPainter(image)
    try:
        # Match QWidget bitmap painting: its device transform supplies DPR,
        # inverse scale cancels it before _PainterHud applies the shared DPR.
        painter.scale(1/ratio,1/ratio)
        owner._hud.painter=painter
        owner._render_hud((width,height))
    finally:
        owner._hud.painter=None
        painter.end()
    return image


def create_native_surface(owner):
    from rendercanvas.qt import QRenderWidget
    from ..qt import _InputSurface
    from PySide6.QtGui import QImage,QPainter
    from PySide6.QtCore import Qt,QTimer
    import numpy as np

    class NativeSurface(_InputSurface,QRenderWidget):
        def __init__(self):
            self.owner=owner
            super().__init__(parent=owner,update_mode='manual',present_method='screen')
            self._released=False
            try:
                self.gpu=owner._webgpu
                self.context=self.get_context('wgpu')
                self.format=self.context.get_preferred_format(self.gpu.adapter)
                self.context.configure(device=self.gpu.device,format=self.format)
                shader=self.gpu.device.create_shader_module(code=presentation_shader(self.format))
                self.pipeline=self.gpu.device.create_render_pipeline(layout='auto',vertex={'module':shader,'entry_point':'vs'},fragment={'module':shader,'entry_point':'fs','targets':[{'format':self.format}]})
                self.hud_texture=None;self.hud_size=None;self.presented_frames=0
                self.request_draw(self.draw_frame)
            except BaseException:
                self._released=True
                self.close()
                self.deleteLater()
                raise

        def _rc_get_present_info(self,methods):
            return super()._rc_get_present_info(['screen']) if 'screen' in methods else None

        def update(self,*args):
            if hasattr(self,'context'): self.request_draw()
            else: super().update(*args)

        def draw_frame(self):
            if owner._closed or self._released:return
            try:
                owner._sync_pixel_ratio()
                texture=owner._render_pixels(readback=False)
                width,height=owner.framebuffer_size()
                image=render_hud_image(owner,width,height)
                if self.hud_size!=(width,height):
                    if self.hud_texture is not None:self.hud_texture.destroy()
                    self.hud_texture=self.gpu.device.create_texture(size=(width,height,1),format='rgba8unorm',usage=self.gpu.wgpu.TextureUsage.TEXTURE_BINDING|self.gpu.wgpu.TextureUsage.COPY_DST)
                    self.hud_size=(width,height)
                self.gpu.device.queue.write_texture({'texture':self.hud_texture},np.asarray(image.constBits()),{'bytes_per_row':width*4,'rows_per_image':height},(width,height,1))
                bind=self.gpu.device.create_bind_group(layout=self.pipeline.get_bind_group_layout(0),entries=[{'binding':0,'resource':texture.create_view()},{'binding':1,'resource':self.hud_texture.create_view()}])
                target=self.context.get_current_texture()
                encoder=self.gpu.device.create_command_encoder()
                render=encoder.begin_render_pass(color_attachments=[{'view':target.create_view(),'resolve_target':None,'clear_value':(1.,1.,1.,1.),'load_op':'clear','store_op':'store'}])
                render.set_pipeline(self.pipeline);render.set_bind_group(0,bind);render.draw(3);render.end()
                self.gpu.device.queue.submit([encoder.finish()]);self.presented_frames+=1
            except Exception as error:
                owner._present_error=f'{type(error).__name__}: {error}'
                QTimer.singleShot(0,owner,lambda: not owner._closed and not self._released and owner.set_present_method('bitmap'))

        def release(self):
            if self._released:return
            self._released=True
            try:
                if self.hud_texture is not None:self.hud_texture.destroy();self.hud_texture=None
            finally:
                # BaseRenderCanvas.close tears down its context/events and marks
                # the Qt canvas closed before its parent schedules deletion.
                self.close()

    return NativeSurface()
