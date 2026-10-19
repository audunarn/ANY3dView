from dataclasses import FrozenInstanceError
from threading import Thread
import json
from pathlib import Path
import numpy as np
import pytest
from any3dview import MeshArrays,MeshHandle,Camera3D,Point3D
from any3dview.appearance import scalar_color,validated_appearance

STOPS=((0.,'#00f'),(.5,'#0f0'),(1.,'#f00'))
def mesh():return MeshArrays(np.array([[-1.,-.6,0.],[1.,-.6,0.],[-1.,.6,0.],[1.,.6,0.]],np.float32),np.array([[0,1,2]],np.uint32),lines=np.array([[2,3]],np.uint32),triangle_to_element=[0],line_to_element=[1],element_ids=[91,92],element_scalars=[0.,1.])

def test_appearance_only_generations_owner_mutation_and_initial_state():
    source=[[0.,'#000'],[1.,'#fff']];m=mesh();h=MeshHandle(m,scalar_range=(0.,2.),colormap=source)
    source[0][1]='#f00';assert h.appearance.colormap[0]==(0.,'#000000')
    with pytest.raises(FrozenInstanceError):h.appearance.scalar_range=(1.,2.)
    before=h.generations;h.update_appearance(scalar_range=(-1.,3.),colormap=STOPS)
    assert h.mesh is m and h.generations.appearance==before.appearance+1
    assert all(getattr(h.generations,n)==getattr(before,n) for n in before.__dataclass_fields__ if n!='appearance')
    errors=[]
    def wrong_thread():
        try:h.update_appearance(scalar_range=(0.,1.))
        except RuntimeError as e:errors.append(e)
    t=Thread(target=wrong_thread);t.start();t.join();assert errors
    h.remove()
    with pytest.raises(RuntimeError):h.update_appearance(colormap=STOPS)

@pytest.mark.parametrize('kwargs',[{'scalar_range':(2.,1.)},{'scalar_range':(0.,np.inf)},{'scalar_range':(True,1.)},{'colormap':()},{'colormap':((0.,'#nope'),)},{'colormap':((np.nan,'#000'),)},{'colormap':((.8,'#000'),(.2,'#fff'))}])
def test_refusal_is_pre_mutation(kwargs):
    h=MeshHandle(mesh());before=h.appearance;g=h.generations
    with pytest.raises((TypeError,ValueError)):h.update_appearance(**kwargs)
    assert h.appearance is before and h.generations==g

def test_synchronous_callback_failure_rolls_back_and_next_generation_is_fresh():
    consumed=[]
    def callback(h,generation):consumed.append((h.appearance,h.generations.appearance));raise RuntimeError('redraw failed')
    h=MeshHandle(mesh(),on_change=callback);old=h.appearance
    with pytest.raises(RuntimeError):h.update_appearance(colormap=STOPS)
    assert h.appearance is old and h.generations.appearance==2 and consumed[0][1]==1
    h._callback=lambda handle,change:consumed.append((handle.appearance,handle.generations.appearance))
    h.update_appearance(colormap=STOPS);assert consumed[-1][1]==3

def test_cpu_color_stops_match_endpoints_clamping_and_explicit_range():
    stops=validated_appearance((0.,1.),STOPS).colormap
    assert scalar_color(-1.,0.,1.,stops)=='#0000ff'
    assert scalar_color(2.,0.,1.,stops)=='#ff0000'
    assert scalar_color(.5,0.,1.,stops)=='#00ff00'
    assert scalar_color(.5,0.,2.,stops)=='#008080'

@pytest.fixture
def renderer():
    import moderngl
    from any3dview.gpu.renderer import ModernGLRenderer
    ctx=moderngl.create_standalone_context(require=330)
    e=Path('C:/ANYworkspaceWork/anypost-20261012/retained-appearance-adoption-20261013-001')
    (e/'gpu-runtime.json').write_text(json.dumps({k:ctx.info.get(k) for k in ('GL_VENDOR','GL_RENDERER','GL_VERSION')}))
    r=ModernGLRenderer(ctx);yield r;r.release();ctx.release()

def pixel(fb,camera,point):
    xy=camera.project_point(Point3D(*point),160,120);x,y=[int(round(v)) for v in xy]
    return np.array(tuple(fb.read(viewport=(x,119-y,1,1),components=3,alignment=1)))

def test_real_gpu_palette_range_updates_retain_buffers_pick_and_pixels(renderer):
    m=mesh();h=MeshHandle(m,scalar_range=(0.,1.),colormap=STOPS)
    renderer.add_mesh(h,scalar_range=h.appearance.scalar_range,colormap=h.appearance.colormap,line_width=8.,lit=False,cull_backface=False)
    fb=renderer.ctx.simple_framebuffer((160,120),components=4);camera=Camera3D()
    renderer.render(camera,(160,120),target=fb,shading_enabled=False)
    chunk=renderer._groups[id(h)].chunks[0][1]
    names=('vertex_buffer','triangle_index_buffer','triangle_vao','line_buffer','line_vao','scalars','colormap')
    names=tuple(n for n in names if hasattr(chunk,n));before={n:id(getattr(chunk,n)) for n in names}
    uploads=renderer.geometry_uploads;scalar_updates=renderer.scalar_buffer_updates
    # Explicitline is at the upper scalar endpoint.
    first=pixel(fb,camera,(0.,.6,0.));assert first[0]>240 and first[1]<20 and first[2]<20
    xy=camera.project_point(Point3D(0.,.6,0.),160,120);pick=renderer.pick(*[int(round(v)) for v in xy],camera,(160,120));assert pick==(h,'line',0)
    assert renderer.pick_dirty is False
    h.update_appearance(colormap=((0.,'#ff0000'),(1.,'#0000ff')),scalar_range=(0.,1.))
    renderer.render(camera,(160,120),target=fb,shading_enabled=False)
    second=pixel(fb,camera,(0.,.6,0.));assert second[2]>240 and second[0]<20
    assert before=={n:id(getattr(chunk,n)) for n in names}
    assert renderer.geometry_uploads==uploads and renderer.scalar_buffer_updates==scalar_updates and renderer.pick_dirty is False
    assert renderer.pick(*[int(round(v)) for v in xy],camera,(160,120))==pick
    palette_updates=renderer.palette_updates
    h.update_appearance(scalar_range=(0.,2.));renderer.render(camera,(160,120),target=fb,shading_enabled=False)
    midpoint=pixel(fb,camera,(0.,.6,0.));assert abs(int(midpoint[0])-int(midpoint[2]))<12
    assert renderer.palette_updates==palette_updates and renderer.geometry_uploads==uploads
    assert h.mesh is m;fb.release()

def test_recreated_gpu_resources_use_persistent_handle_appearance(renderer):
    h=MeshHandle(mesh());renderer.add_mesh(h)
    h.update_appearance(scalar_range=(-2.,2.),colormap=STOPS)
    renderer._sync(renderer._groups[id(h)])
    renderer.remove_mesh(h)
    renderer.add_mesh(h,scalar_range=h.appearance.scalar_range,colormap=h.appearance.colormap)
    group=renderer._groups[id(h)];assert group.scalar_range==(-2.,2.) and group.colormap==h.appearance.colormap
    palette=np.frombuffer(group.chunks[0][1].colormap.read(),dtype=np.float32).reshape(256,3)
    np.testing.assert_array_equal(palette[[0,-1]],[[0.,0.,1.],[1.,0.,0.]])


def test_callback_and_backend_replay_use_authoritative_rolled_back_appearance():
    from types import SimpleNamespace
    from any3dview.retained_viewer import RetainedViewer
    calls=[];fake=SimpleNamespace(_entries={},_renderer=SimpleNamespace(pick_dirty=False,add_mesh=lambda h,**kw:calls.append(kw)),_selection_index=object(),_animation_handles=[],_animation_entries={})
    original_index=fake._selection_index
    fake.redraw=lambda:(_ for _ in ()).throw(RuntimeError('synchronous redraw'))
    h=MeshHandle(mesh(),on_change=lambda handle,change:RetainedViewer._mesh_changed(fake,handle,change))
    fake._entries[id(h)]={'handle':h,'appearance':{'scalar_range':None,'colormap':None}}
    with pytest.raises(RuntimeError):h.update_appearance(colormap=STOPS)
    assert not fake._renderer.pick_dirty and fake._selection_index is original_index
    RetainedViewer._restore_live_renderer(fake)
    assert calls[-1]['colormap'] is None and calls[-1]['scalar_range'] is None

def test_actual_qt_software_scalar_palette_updates_without_geometry_upload():
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from any3dview.qt import QtViewer
    app=QApplication.instance() or QApplication([])
    viewer=QtViewer(backend='software');viewer.resize(240,180)
    surface=MeshArrays(np.array([[-1.,-.6,0.],[1.,-.6,0.],[0.,.6,0.]],np.float32),np.array([[0,1,2]],np.uint32),element_scalars=[0.])
    handle=viewer.add_mesh_arrays(surface,colormap=((0.,'#ff0000'),(1.,'#0000ff')),scalar_range=(0.,1.),lit=False,cull_backface=False)
    viewer.show();app.processEvents()
    first=viewer.grab().toImage();uploads=viewer.renderer_diagnostics['geometry_uploads']
    handle.update_appearance(colormap=((0.,'#00ff00'),(1.,'#0000ff')));app.processEvents()
    second=viewer.grab().toImage()
    assert first!=second and viewer.renderer_diagnostics['geometry_uploads']==uploads
    assert handle.mesh is surface
    viewer.close();app.processEvents()


def test_partial_gpu_lut_write_restores_and_retries_after_callback_rollback(renderer):
    h=MeshHandle(mesh());renderer.add_mesh(h);group=renderer._groups[id(h)];chunk=group.chunks[0][1]
    original=chunk.colormap;baseline=original.read();vertex=chunk.vertex_buffer;uploads=renderer.geometry_uploads
    class WriteThenFail:
        failed=False
        def write(self,data):
            original.write(data)
            if not self.failed:self.failed=True;raise RuntimeError('GPU write completed then failed')
        def __getattr__(self,name):return getattr(original,name)
    chunk.colormap=WriteThenFail()
    h._callback=lambda handle,change:renderer._sync(group)
    renderer.pick_dirty=False
    with pytest.raises(RuntimeError,match='completed then failed'):h.update_appearance(colormap=STOPS)
    assert h.appearance.colormap is None and group.palette_dirty
    assert original.read()==baseline and not renderer.pick_dirty
    renderer._sync(group)
    assert not group.palette_dirty and original.read()==baseline
    assert chunk.vertex_buffer is vertex and renderer.geometry_uploads==uploads
    assert renderer.palette_write_failures==1


def test_initial_gpu_palette_failure_removes_unpublished_resources(renderer,monkeypatch):
    import any3dview.gpu.renderer as module
    h=MeshHandle(mesh())
    def failure(stops):raise RuntimeError('initial palette failed')
    monkeypatch.setattr(module,'palette_data',failure)
    with pytest.raises(RuntimeError,match='initial palette failed'):renderer.add_mesh(h,colormap=STOPS)
    assert id(h) not in renderer._groups


def test_visibility_keeps_original_picking_invalidation_while_palette_does_not():
    from types import SimpleNamespace
    from any3dview.retained_viewer import RetainedViewer
    fake=SimpleNamespace(_entries={},_renderer=SimpleNamespace(pick_dirty=False),_selection_index=object(),redraw=lambda:None)
    h=MeshHandle(mesh(),on_change=lambda handle,change:RetainedViewer._mesh_changed(fake,handle,change))
    fake._entries[id(h)]={'handle':h,'appearance':{'scalar_range':None,'colormap':None}}
    h.update_appearance(colormap=STOPS);assert not fake._renderer.pick_dirty and fake._selection_index is not None
    h.set_visible(False);assert fake._renderer.pick_dirty and fake._selection_index is None
