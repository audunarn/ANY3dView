"""CPU contracts and opt-in actual renderer checks for viewer roadmap."""
import math,os
import numpy as np
import pytest
from any3dview import SectionPlane,SectionBox
from any3dview.clipping import ClipIntersection
from any3dview.display import Contours,contour_uniforms

def test_rotated_box_and_plane_intersection():
    q=(0,0,math.sin(math.pi/4),math.cos(math.pi/4))
    box=SectionBox((10,0,0),(2,1,1),q)
    assert box.contains((10,1.9,0)) and not box.contains((11.1,0,0))
    clip=ClipIntersection((*box.planes,SectionPlane((0,1,0),0)))
    segment=clip.clip_segment((10,-3,0),(10,3,0))
    np.testing.assert_allclose([v.to_tuple() for v in segment],[(10,0,0),(10,2,0)],atol=1e-12)
    polygon=clip.clip_polygon([(8,-3,0),(12,-3,0),(12,3,0),(8,3,0)])
    assert polygon and all(clip.contains(v) for v in polygon)

@pytest.mark.parametrize('args',[((0,0,0),(-1,1,1)),((0,0,0),(1,1,1),(0,0,0,0)),((float('nan'),0,0),(1,1,1))])
def test_box_refusal(args):
    with pytest.raises(ValueError):SectionBox(*args)

def test_contour_original_domain_large_offset_and_validation():
    settings=Contours('Banded',(1e12+.25,1e12+.75),True,('above',1e12+.5))
    control,levels,emphasis=contour_uniforms(settings,(1e12,1e12+1))
    assert control[0]==1 and control[1]==2
    np.testing.assert_array_equal(levels[:2],[.25,.75]);assert emphasis[:2]==[1.,.5]
    for candidate in [('oops',()),('Banded',(1.,1.)),('Isolines',(float('nan'),))]:
        with pytest.raises(ValueError):Contours(*candidate)

@pytest.mark.skipif(os.environ.get('ANY3DVIEW_RUN_QT_TESTS')!='1',reason='Qt opt-in')
def test_software_box_query_and_portable_state():
    from PySide6.QtWidgets import QApplication
    from any3dview.qt import QtViewer
    app=QApplication.instance() or QApplication([])
    v=QtViewer(backend='software')
    try:
        v.set_section_box((0,0,0),(1,2,3),(0,0,0,2))
        v.set_section_plane((1,0,0),.5)
        assert v._effective_section.contains((.75,0,0)) and not v._effective_section.contains((0,0,0))
        state=v.export_view_state()
        assert state.section_box==v.section_box
        v.clear_section_box();v.apply_view_state(state)
        assert v.section_box==state.section_box
    finally:v.destroy();app.processEvents()

@pytest.mark.skipif(os.environ.get('ANY3DVIEW_TEST_WEBGPU')!='1',reason='GPU opt-in')
def test_gpu_box_contours_outlines_shadows_and_source_immutability(tmp_path):
    from PySide6.QtWidgets import QApplication
    from any3dview.qt_webgpu import QtWebGPUViewer
    from any3dview.arrays import MeshArrays
    from PIL import Image
    app=QApplication.instance() or QApplication([]);v=QtWebGPUViewer(width=240,height=200)
    try:
        v.surface.resize(240,200);v.set_top_view()
        mesh=MeshArrays([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],[[0,1,2],[0,2,3]],node_scalars=[-1,1,1,-1])
        before=mesh.owned_copy()
        h=v.add_mesh_arrays(mesh,cull_backface=False,scalar_range=(-1,1));v.fit_to_scene(redraw=False)
        continuous=np.asarray(v.capture_image())
        v.set_contours('Banded',(-.5,0,.5),zero_contour=True,threshold={'direction':'above','value':.2})
        bands=np.asarray(v.capture_image());assert not np.array_equal(continuous,bands)
        v.set_contours('Isolines',(-.5,0,.5));isolines=np.asarray(v.capture_image());assert not np.array_equal(bands,isolines)
        v.set_section_box((0,0,0),(.5,2,1),(0,0,math.sin(.2),math.cos(.2)))
        v.set_render_quality(ambient_occlusion='High');clipped=np.asarray(v.capture_image());assert not np.array_equal(isolines,clipped)
        h.set_selected_elements([0]);v.set_selection_outline(True,True);v.capture_image()
        v.set_shadows('High',.5);v.capture_image()
        with pytest.raises(NotImplementedError):v.set_cut_caps()
        np.testing.assert_array_equal(mesh.positions,before.positions);np.testing.assert_array_equal(mesh.node_scalars,before.node_scalars)
        Image.fromarray(clipped).save(tmp_path/'roadmap.png')
    finally:v.destroy();app.processEvents()


@pytest.mark.skipif(os.environ.get('ANY3DVIEW_TEST_WEBGPU')!='1',reason='GPU opt-in')
def test_qualified_caps_gizmo_high_resolution_and_refusal():
    from PySide6.QtWidgets import QApplication
    from any3dview.qt_webgpu import QtWebGPUViewer
    from any3dview.arrays import MeshArrays
    from any3dview.selection import PickBinding
    from any3dview.ownership import PackedOwnerTable
    from any3dview.webgpu.caps import box_volume_cap_provider
    import sys
    sys.path.insert(0,'C:/Github/ANYgeometry/src')
    from anygeometry.volume_cut import BoxVolume
    from types import SimpleNamespace
    app=QApplication.instance() or QApplication([]);v=QtWebGPUViewer(width=240,height=200)
    try:
        v.surface.resize(240,200);v.set_top_view()
        vertices=[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
        triangles=[[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
        owner=PickBinding.one('volume:box')
        h=v.add_mesh_arrays(MeshArrays(vertices,triangles),owners=PackedOwnerTable.from_owners(triangles=[owner]*12),cull_backface=False)
        v.fit_to_scene(redraw=False);v.set_section_plane((0,0,-1),0)
        v.set_cut_caps(provider=box_volume_cap_provider(BoxVolume((-1,-1,-1),(1,1,1)),h,owner))
        assert v._cap_handles and all(cap.mesh.node_scalars is None for cap in v._cap_handles)
        v.capture_image();original_target=v._webgpu.target
        image=v.capture_image(width=480,height=320);assert image.size==(480,320)
        assert v.framebuffer_size()==(240,200)
        assert v._webgpu.size[:2]==(240,200) and v._webgpu.target is original_target
        v.capture_image();assert v._webgpu.size[:2]==(240,200)
        v.set_section_box((0,0,0),(1,1,1));v.capture_image()
        gizmo=v._section_gizmo;assert len(gizmo.handles)==10
        handle=next(h for h in gizmo.handles if h[1]=='translate');x,y=handle[0]
        e=lambda dx,dy:SimpleNamespace(x=(x+dx)*v.pixel_ratio,y=(y+dy)*v.pixel_ratio)
        before=v.section_box;assert gizmo.press(e(0,0));assert gizmo.motion(e(10,5))
        assert v.section_box!=before;gizmo.cancel();assert v.section_box==before
        v.clear_cut_cap_provider()
        assert not v._cap_handles and v._cap_provider is None and v._cap_key is None
        with pytest.raises(NotImplementedError):v.set_cut_caps()
        with pytest.raises(ValueError):v.capture_image(width=0)
        with pytest.raises(TypeError):v.set_cut_caps(provider=lambda viewer,planes:[object()])
    finally:v.destroy();app.processEvents()

@pytest.mark.skipif(os.environ.get('ANY3DVIEW_TEST_NATIVE_PRESENT')!='1',reason='native prototype opt-in')
def test_native_present_prototype():
    from PySide6.QtWidgets import QApplication
    from any3dview.qt_webgpu import QtWebGPUViewer
    from any3dview.arrays import MeshArrays
    app=QApplication.instance() or QApplication([]);v=QtWebGPUViewer(width=240,height=200)
    try:
        v.add_mesh_arrays(MeshArrays([[-1,-1,0],[1,-1,0],[0,1,0]],[[0,1,2]]),cull_backface=False)
        v.fit_to_scene(redraw=False);v.set_axis_indicator(True);v.show();app.processEvents()
        v.set_present_method('native');app.processEvents();v.surface.force_draw();app.processEvents()
        snapshot=v.diagnostic_snapshot()
        assert snapshot['present_method']=='native',snapshot
        assert snapshot['native_presented_frames']>0,snapshot
        v.set_present_method('bitmap');assert v.diagnostic_snapshot()['present_method']=='bitmap'
    finally:v.destroy();app.processEvents()


@pytest.mark.skipif(os.environ.get('ANY3DVIEW_TEST_WEBGPU')!='1',reason='GPU opt-in')
def test_selection_through_depth_and_directional_shadow_oracles():
    from PySide6.QtWidgets import QApplication
    from any3dview.qt_webgpu import QtWebGPUViewer
    from any3dview.arrays import MeshArrays
    app=QApplication.instance() or QApplication([]);v=QtWebGPUViewer(width=300,height=240)
    try:
        v.surface.resize(300,240);v.set_top_view()
        def quad(z,extent,color):
            return v.add_mesh_arrays(MeshArrays([[-extent,-extent,z],[extent,-extent,z],[extent,extent,z],[-extent,extent,z]],[[0,1,2],[0,2,3]]),color=color,cull_backface=False)
        back=quad(0,.6,'#3366cc');front=quad(.2,1.,'#dddddd');v.fit_to_scene(redraw=False)
        back.set_selected_elements([0,1]);v.set_selection_outline(True,False);occluded=np.asarray(v.capture_image())
        v.set_selection_outline(True,True);through=np.asarray(v.capture_image());assert not np.array_equal(occluded,through)
        v.clear();v.set_selection_outline(False);quad(0,2.,'#dddddd');quad(.65,.6,'#dddddd')
        v.set_iso_view();v.fit_to_scene(redraw=False);v.set_light(direction=(1,0,1),follow_camera=False)
        off=np.asarray(v.capture_image());v.set_shadows('High',.8);shadow=np.asarray(v.capture_image())
        assert np.max(off[:,:,:3].astype(int)-shadow[:,:,:3].astype(int))>5
        v.set_shadows('Off');np.testing.assert_array_equal(off,np.asarray(v.capture_image()))
    finally:v.destroy();app.processEvents()


@pytest.mark.skipif(os.environ.get('ANY3DVIEW_RUN_QT_TESTS')!='1',reason='Qt opt-in')
@pytest.mark.parametrize('ratio',[1.,2.])
def test_local_axis_rotation_gestures_cancel_and_native_hud_dpi(ratio):
    from PySide6.QtWidgets import QApplication
    from any3dview.core import Point3D
    from any3dview.section_gizmo import SectionBoxGizmo
    from any3dview.qt import _PainterHud
    from any3dview.webgpu.presentation import render_hud_image
    from types import SimpleNamespace
    app=QApplication.instance() or QApplication([])
    class Viewer:
        pixel_ratio=ratio
        section_box=SectionBox((0,0,0),(1,1,1),(0,0,math.sin(.23),math.cos(.23)))
        camera=SimpleNamespace(position=Point3D(0,0,10),near=.1,fov=1.,basis=lambda:(Point3D(1,0,0),Point3D(0,1,0),Point3D(0,0,-1)))
        _hud=_PainterHud()
        def project_point(self,point):return ((100+point.x*15)*ratio,(100-point.y*15)*ratio,10-point.z)
        def framebuffer_size(self):return (round(200*ratio),round(200*ratio))
        def set_section_box(self,center,extents,orientation,enabled=True):self.section_box=SectionBox(center,extents,orientation,enabled)
        def _render_hud(self,viewport):
            self._hud.begin(viewport,ratio)
            # Fixed HUD anchor and clipped region distinguish physical/logical pixels.
            self._hud.quad((12,14,18,20),'#ff0000')
            self._hud.quad((195,195,210,210),'#00ff00')
            self.gizmo.draw(self._hud)
    v=Viewer();v.gizmo=SectionBoxGizmo(v)
    image=render_hud_image(v,*v.framebuffer_size())
    pixels=np.asarray(image.constBits()).reshape(image.height(),image.width(),4)
    assert tuple(pixels[round(16*ratio),round(15*ratio),:3])==(255,0,0)
    assert pixels[round(25*ratio),round(25*ratio),3]==0
    assert tuple(pixels[-1,-1,:3])==(0,255,0)
    handles=[h for h in v.gizmo.handles if h[1]=='rotate']
    assert {h[2] for h in handles}=={0,1,2}
    original=v.section_box
    start_axes=np.array([original.planes[2*i].normal.to_tuple() for i in range(3)])
    for handle in handles:
        start=np.array(handle[0]);pivot=np.array([100.,100.]);direction=start-pivot
        end=pivot+np.array([-direction[1],direction[0]])
        event=lambda p:SimpleNamespace(x=p[0]*ratio,y=p[1]*ratio)
        assert v.gizmo.press(event(start)) and v.gizmo.motion(event(end))
        axes=np.array([v.section_box.planes[2*i].normal.to_tuple() for i in range(3)])
        axis=handle[2]
        np.testing.assert_allclose(axes[axis],start_axes[axis],atol=1e-12)
        np.testing.assert_allclose(axes[(axis+1)%3],start_axes[(axis+2)%3],atol=1e-12)
        np.testing.assert_allclose(axes[(axis+2)%3],-start_axes[(axis+1)%3],atol=1e-12)
        v.gizmo.cancel();assert v.section_box==original


def test_clear_cut_cap_provider_releases_owner_closure_and_derived_handles():
    import gc,weakref
    from any3dview.qt_webgpu import QtWebGPUViewer
    from any3dview.arrays import MeshArrays
    from any3dview.retained import MeshHandle
    class Owner:pass
    class Host:
        clear_cut_cap_provider=QtWebGPUViewer.clear_cut_cap_provider
        set_cut_caps=QtWebGPUViewer.set_cut_caps
        _refresh_caps=QtWebGPUViewer._refresh_caps
        _effective_section=ClipIntersection((SectionPlane((1,0,0),0),))
        def make_current(self):pass
        def redraw(self):pass
    owner=Owner();reference=weakref.ref(owner)
    provider=lambda viewer,planes,source=owner:()
    h=Host();h._cap_provider=provider;h._cut_caps=(True,False);h._cap_key=object()
    cap=MeshHandle(MeshArrays([[0,0,0],[0,1,0],[0,0,1]],[[0,1,2]]))
    h._cap_handles=[cap];h._entries={id(cap):{'handle':cap}}
    del owner,provider
    assert reference() is not None
    h.clear_cut_cap_provider();gc.collect()
    assert cap.removed and h._cap_handles==[] and h._cap_provider is None and h._cap_key is None
    assert reference() is None and h._cut_caps==(False,False)
    with pytest.raises(NotImplementedError):h.set_cut_caps()
    reusable=lambda viewer,planes:()
    h.set_cut_caps(False,provider=reusable)
    h.set_cut_caps(False,provider=None)
    assert h._cap_provider is reusable


@pytest.mark.skipif(os.environ.get('ANY3DVIEW_TEST_WEBGPU')!='1',reason='GPU opt-in')
def test_native_presentation_unorm_and_srgb_preserve_display_bytes():
    from any3dview.webgpu.renderer import WebGPURenderer
    from any3dview.webgpu.presentation import presentation_shader
    renderer=WebGPURenderer();device=renderer.device;wgpu=renderer.wgpu;owned=[]
    try:
        def texture(format,usage):
            result=device.create_texture(size=(2,1,1),format=format,usage=usage);owned.append(result);return result
        source=texture('rgba8unorm',wgpu.TextureUsage.TEXTURE_BINDING|wgpu.TextureUsage.COPY_DST)
        hud=texture('rgba8unorm',wgpu.TextureUsage.TEXTURE_BINDING|wgpu.TextureUsage.COPY_DST)
        values=np.array([[128,64,32,255],[4,8,10,255]],dtype=np.uint8)
        device.queue.write_texture({'texture':source},values,{'bytes_per_row':8,'rows_per_image':1},(2,1,1))
        for format in ('rgba8unorm','rgba8unorm-srgb'):
            shader=device.create_shader_module(code=presentation_shader(format))
            pipeline=device.create_render_pipeline(layout='auto',vertex={'module':shader,'entry_point':'vs'},fragment={'module':shader,'entry_point':'fs','targets':[{'format':format}]})
            target=texture(format,wgpu.TextureUsage.RENDER_ATTACHMENT|wgpu.TextureUsage.COPY_SRC)
            for overlay in (np.zeros((2,4),dtype=np.uint8),np.tile(np.array([32,128,64,128],dtype=np.uint8),(2,1))):
                device.queue.write_texture({'texture':hud},overlay,{'bytes_per_row':8,'rows_per_image':1},(2,1,1))
                bind=device.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=[{'binding':0,'resource':source.create_view()},{'binding':1,'resource':hud.create_view()}])
                encoder=device.create_command_encoder()
                render=encoder.begin_render_pass(color_attachments=[{'view':target.create_view(),'resolve_target':None,'clear_value':(0.,0.,0.,1.),'load_op':'clear','store_op':'store'}])
                render.set_pipeline(pipeline);render.set_bind_group(0,bind);render.draw(3);render.end()
                device.queue.submit([encoder.finish()])
                raw=device.queue.read_texture({'texture':target},{'bytes_per_row':8,'rows_per_image':1},(2,1,1))
                actual=np.frombuffer(raw,dtype=np.uint8).reshape(2,4)
                alpha=overlay[:,3:4]/255.
                expected=np.rint(values[:,:3]*(1-alpha)+overlay[:,:3]*alpha).astype(int)
                assert np.max(np.abs(actual[:,:3].astype(int)-expected))<=1,(format,actual,expected)
                np.testing.assert_array_equal(actual[:,3],[255,255])
    finally:
        for texture in owned:texture.destroy()
        renderer.close()


def test_capture_device_and_temporary_attachment_admission_without_allocation():
    from any3dview.qt_webgpu import validate_capture_admission
    from types import SimpleNamespace
    def no_allocation(*args,**kwargs):raise AssertionError('Admission must not allocate GPU textures')
    device=SimpleNamespace(limits={'max_texture_dimension_2d':16384},create_texture=no_allocation)
    renderer=SimpleNamespace(device=device,samples=4,ao_quality='Off')
    assert validate_capture_admission(renderer,3840,2160)==3840*2160*136
    assert validate_capture_admission(renderer,480,320)==480*320*136
    renderer.ao_quality='High'
    assert validate_capture_admission(renderer,3840,2160)==3840*2160*168
    renderer.ao_quality='Off'
    assert validate_capture_admission(renderer,3840,2160,shadows=True)==3840*2160*168
    with pytest.raises(ValueError,match='2 GiB.*reduce resolution or anti-aliasing'):
        validate_capture_admission(renderer,16384,16384)
    device.limits={'max-texture-dimension-2d':2048}
    with pytest.raises(ValueError,match='device 2D texture limit 2048'):
        validate_capture_admission(renderer,3840,2160)
    assert validate_capture_admission(renderer,480,320)==480*320*136
