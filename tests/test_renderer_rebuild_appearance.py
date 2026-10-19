"""Direct real-GL regressions without a RetainedViewer change callback."""
import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from any3dview import Camera3D,MeshArrays,MeshHandle,Point3D
from any3dview.appearance import palette_data

STOPS=((0.,'#0000ff'),(1.,'#ff0000'))

def mesh():
    return MeshArrays(np.array([[-1.,-.6,0.],[1.,-.6,0.],[-1.,.6,0.],[1.,.6,0.]],np.float32),
        np.array([[0,1,2]],np.uint32),lines=np.array([[2,3]],np.uint32),
        triangle_to_element=[0],line_to_element=[1],element_ids=[91,92],element_scalars=[0.,1.])

@pytest.fixture
def renderer(tmp_path):
    import moderngl
    from any3dview.gpu.renderer import ModernGLRenderer
    from any3dview.gpu import renderer as module
    ctx=moderngl.create_standalone_context(require=330)
    runtime={key:ctx.info.get(key) for key in ('GL_VENDOR','GL_RENDERER','GL_VERSION')}
    runtime.update(renderer_path=str(Path(module.__file__).resolve()),renderer_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
    (tmp_path/'gpu-runtime.json').write_text(json.dumps(runtime))
    result=ModernGLRenderer(ctx)
    try:yield result
    finally:result.release();ctx.release()

@pytest.mark.parametrize('rebuild',['position','topology'])
def test_coalesced_rebuild_uses_pending_handle_appearance(renderer,rebuild):
    handle=MeshHandle(mesh());renderer.add_mesh(handle)
    if rebuild=='position':handle.update_positions(handle.mesh.positions+.1)
    else:handle.add_chunk('extra',mesh())
    handle.update_appearance(scalar_range=(-2.,3.),colormap=STOPS)
    renderer._sync(renderer._groups[id(handle)])
    group=renderer._groups[id(handle)]
    assert group.scalar_range==handle.appearance.scalar_range and group.colormap==handle.appearance.colormap
    assert group.generations==handle.generations and not group.palette_dirty
    for _,chunk in group.chunks:assert chunk.colormap.read()==palette_data(handle.appearance.colormap)

def test_rebuild_retries_failed_pending_palette(renderer):
    handle=MeshHandle(mesh());renderer.add_mesh(handle)
    group=renderer._groups[id(handle)];original=group.chunks[0][1].colormap
    class FailedTexture:
        def write(self,data):original.write(data);raise RuntimeError('partial LUT failure')
        def __getattr__(self,name):return getattr(original,name)
    group.chunks[0][1].colormap=FailedTexture()
    handle.update_appearance(scalar_range=(0.,2.),colormap=STOPS)
    with pytest.raises(RuntimeError,match='partial LUT'):renderer._sync(group)
    assert group.palette_dirty and group.colormap is None
    handle.update_positions(handle.mesh.positions+.1)
    renderer._sync(group)
    replacement=renderer._groups[id(handle)]
    assert not replacement.palette_dirty and replacement.colormap==handle.appearance.colormap
    assert replacement.scalar_range==(0.,2.)
    assert replacement.chunks[0][1].colormap.read()==palette_data(STOPS)

def test_standalone_visibility_invalidates_picks_but_palette_does_not(renderer):
    handle=MeshHandle(mesh());renderer.add_mesh(handle,line_width=8.,cull_backface=False)
    camera=Camera3D();viewport=(160,120)
    xy=camera.project_point(Point3D(0.,.6,0.),*viewport);x,y=(int(round(v)) for v in xy)
    assert renderer.pick(x,y,camera,viewport)==(handle,'line',0)
    assert not renderer.pick_dirty
    group=renderer._groups[id(handle)];uploads=renderer.geometry_uploads
    buffers=tuple((id(chunk.vertex_buffer),id(chunk.line_buffer)) for _,chunk in group.chunks)
    handle.update_appearance(scalar_range=(0.,2.),colormap=STOPS);renderer._sync(group)
    assert not renderer.pick_dirty
    handle.set_visible(False);renderer._sync(group)
    assert renderer.pick_dirty and renderer.pick(x,y,camera,viewport) is None
    handle.set_visible(True);renderer._sync(group)
    assert renderer.pick_dirty and renderer.pick(x,y,camera,viewport)==(handle,'line',0)
    assert renderer.geometry_uploads==uploads
    assert buffers==tuple((id(chunk.vertex_buffer),id(chunk.line_buffer)) for _,chunk in group.chunks)

@pytest.mark.parametrize('change',['visibility','displacement','active'])
def test_failed_palette_cannot_consume_other_pick_changes(renderer,monkeypatch,change):
    import any3dview.gpu.renderer as module
    handle=MeshHandle(mesh());renderer.add_mesh(handle)
    group=renderer._groups[id(handle)];renderer.pick_dirty=False
    if change=='visibility':handle.set_visible(False)
    elif change=='displacement':handle.update_displacements(np.zeros_like(handle.mesh.positions))
    else:handle.set_active_elements([False,True])
    handle.update_appearance(colormap=STOPS)
    monkeypatch.setattr(module,'palette_data',lambda *a:(_ for _ in ()).throw(RuntimeError('failed palette')))
    with pytest.raises(RuntimeError,match='failed palette'):renderer._sync(group)
    assert renderer.pick_dirty

@pytest.mark.parametrize('viewport',[(160,120),(180,140)])
def test_failed_pick_pass_never_leaves_clean_target(renderer,monkeypatch,viewport):
    import any3dview.gpu.renderer as module
    handle=MeshHandle(mesh());renderer.add_mesh(handle)
    camera=Camera3D();renderer.render_pick(camera,(160,120));assert not renderer.pick_dirty
    handle.update_appearance(colormap=STOPS)
    monkeypatch.setattr(module,'palette_data',lambda *a:(_ for _ in ()).throw(RuntimeError('failed pick sync')))
    with pytest.raises(RuntimeError,match='failed pick sync'):renderer.render_pick(camera,viewport)
    assert renderer.pick_dirty

def test_failed_rebuild_preserves_resources_semantics_and_retries(renderer,monkeypatch):
    import any3dview.gpu.renderer as module
    handle=MeshHandle(mesh());renderer.add_mesh(handle)
    renderer.set_semantic_masks(handle,selected_elements=np.array([0],np.uint32))
    group=renderer._groups[id(handle)];chunk=group.chunks[0][1]
    vertex=chunk.vertex_buffer;before=vertex.read();generations=group.generations
    handle.update_positions(handle.mesh.positions+.1);handle.update_appearance(colormap=STOPS)
    original=module.palette_data
    monkeypatch.setattr(module,'palette_data',lambda *a:(_ for _ in ()).throw(RuntimeError('rebuild refused')))
    with pytest.raises(RuntimeError,match='rebuild refused'):renderer._sync(group)
    assert renderer._groups[id(handle)] is group and group.generations==generations
    assert chunk.vertex_buffer is vertex and vertex.read()==before
    np.testing.assert_array_equal(group.highlighted_elements,[0])
    monkeypatch.setattr(module,'palette_data',original);renderer._sync(group)
    replacement=renderer._groups[id(handle)]
    assert replacement is not group and replacement.generations==handle.generations
    np.testing.assert_array_equal(replacement.highlighted_elements,[0])
    assert replacement.chunks[0][1].colormap.read()==palette_data(STOPS)

def test_failed_pick_resize_preserves_old_target_and_releases_new_texture(renderer):
    handle=MeshHandle(mesh());renderer.add_mesh(handle,line_width=8.,cull_backface=False)
    camera=Camera3D();renderer.render_pick(camera,(160,120));old=renderer._pick_fbo
    real=renderer.ctx;created=[]
    class FaultContext:
        failed=False
        def texture(self,*a,**k):
            texture=real.texture(*a,**k);created.append(texture);return texture
        def depth_renderbuffer(self,*a,**k):
            if not self.failed:self.failed=True;raise RuntimeError('pick resize allocation')
            return real.depth_renderbuffer(*a,**k)
        def __getattr__(self,name):return getattr(real,name)
    renderer.ctx=FaultContext()
    with pytest.raises(RuntimeError,match='resize allocation'):renderer.render_pick(camera,(180,140))
    assert renderer._pick_fbo is old and renderer._pick_size==(160,120) and renderer.pick_dirty
    assert created and type(created[0].mglo).__name__=='InvalidObject'
    renderer.render_pick(camera,(160,120))
    xy=camera.project_point(Point3D(0.,.6,0.),160,120)
    assert renderer.pick(*(int(round(v)) for v in xy),camera,(160,120))==(handle,'line',0)
    renderer.render_pick(camera,(180,140));assert renderer._pick_size==(180,140)

def test_partial_chunk_allocation_releases_returned_gl_objects(renderer):
    real=renderer.ctx;created=[]
    class FaultContext:
        def buffer(self,*a,**k):
            buffer=real.buffer(*a,**k);created.append(buffer);return buffer
        def vertex_array(self,*a,**k):raise RuntimeError('partial chunk allocation')
        def __getattr__(self,name):return getattr(real,name)
    renderer.ctx=FaultContext();handle=MeshHandle(mesh())
    with pytest.raises(RuntimeError,match='partial chunk'):renderer.add_mesh(handle)
    assert created and all(type(resource.mglo).__name__=='InvalidObject' for resource in created)
    assert id(handle) not in renderer._groups

def test_later_chunk_failure_releases_completed_chunks_and_preserves_old_group(renderer,monkeypatch):
    import any3dview.gpu.renderer as module
    handle=MeshHandle(mesh());handle.add_chunk('extra',mesh());renderer.add_mesh(handle)
    old=renderer._groups[id(handle)];old_buffer=old.chunks[0][1].vertex_buffer;before=old_buffer.read()
    create=module._Chunk.create;completed=[]
    def fail_later(*args,**kwargs):
        if completed:raise RuntimeError('second chunk allocation')
        chunk=create(*args,**kwargs);completed.append(chunk);return chunk
    monkeypatch.setattr(module._Chunk,'create',fail_later)
    handle.update_positions(handle.mesh.positions+.1)
    with pytest.raises(RuntimeError,match='second chunk'):renderer._sync(old)
    assert renderer._groups[id(handle)] is old and old_buffer.read()==before
    assert type(completed[0].vertex_buffer.mglo).__name__=='InvalidObject'
    monkeypatch.setattr(module._Chunk,'create',create);renderer._sync(old)
    assert renderer._groups[id(handle)] is not old

def test_coalesced_selection_rebuild_matches_in_place_unowned_mask_clearing(renderer):
    states=[]
    for rebuild in (False,True):
        handle=MeshHandle(mesh());handle.add_chunk('extra',mesh());renderer.add_mesh(handle)
        renderer.set_chunk_semantic_masks(handle,'extra',hidden_elements=[0])
        handle.set_selected_elements([0])
        if rebuild:handle.update_positions(handle.mesh.positions+.1)
        renderer._sync(renderer._groups[id(handle)])
        group=renderer._groups[id(handle)]
        states.append(tuple(group.chunk_semantic_masks))
        assert 'extra' not in group.chunk_semantic_masks
        renderer.remove_mesh(handle)
    assert states[0]==states[1]

def test_actual_partial_palette_write_and_visibility_leave_picking_dirty(renderer):
    handle=MeshHandle(mesh());renderer.add_mesh(handle)
    group=renderer._groups[id(handle)];original=group.chunks[0][1].colormap;baseline=original.read();writes=[]
    class WriteThenFail:
        def write(self,data):
            original.write(data);writes.append(data)
            if len(writes)==1:raise RuntimeError('actual partial write')
        def __getattr__(self,name):return getattr(original,name)
    group.chunks[0][1].colormap=WriteThenFail();renderer.pick_dirty=False
    handle.set_visible(False);handle.update_appearance(colormap=STOPS)
    with pytest.raises(RuntimeError,match='actual partial'):renderer._sync(group)
    assert writes[0]!=baseline and original.read()==baseline
    assert renderer.pick_dirty and group.palette_dirty
    renderer._sync(group);assert not group.palette_dirty and not group.visible


def test_add_mesh_failed_selection_rolls_back_and_releases_chunks(renderer,monkeypatch):
    handle=MeshHandle(mesh());renderer.pick_dirty=False;chunks=[]
    from any3dview.gpu.renderer import _Chunk
    create=_Chunk.create.__func__
    def tracked(cls,*args,**kw):
        chunk=create(cls,*args,**kw);chunks.append(chunk);return chunk
    monkeypatch.setattr(_Chunk,'create',classmethod(tracked))
    monkeypatch.setattr(renderer,'_write_selection',lambda group:(_ for _ in ()).throw(RuntimeError('selection write failed')))
    with pytest.raises(RuntimeError,match='selection write'):renderer.add_mesh(handle)
    assert id(handle) not in renderer._groups and renderer.pick_dirty
    for chunk in chunks:
        with pytest.raises(Exception):chunk.vertex_buffer.read()


def test_successful_chunk_keeps_real_context_and_no_stage(renderer):
    handle=MeshHandle(mesh());renderer.add_mesh(handle)
    for _,chunk in renderer._groups[id(handle)].chunks:assert chunk.ctx is renderer.ctx


def test_remove_dirty_before_release_failure_and_continues_other_chunks(renderer):
    handle=MeshHandle(mesh());handle.add_chunk('second',mesh());renderer.add_mesh(handle)
    group=renderer._groups[id(handle)];events=[]
    chunks=list(group.chunks)
    class Broken:
        def release(self):events.append(('failed',renderer.pick_dirty));raise RuntimeError('release failed')
    class Remaining:
        def release(self):events.append(('remaining',renderer.pick_dirty))
    group.chunks=[('a',Broken()),('b',Remaining())]
    renderer.pick_dirty=False
    with pytest.raises(RuntimeError,match='release failed'):renderer.remove_mesh(handle)
    assert id(handle) not in renderer._groups and events==[('failed',True),('remaining',True)]
    for _,chunk in chunks:chunk.release()


def test_group_coercion_failure_releases_all_created_chunks(renderer,monkeypatch):
    from any3dview.gpu.renderer import _Chunk
    handle=MeshHandle(mesh());chunks=[];create=_Chunk.create.__func__
    def tracked(cls,*args,**kw):
        chunk=create(cls,*args,**kw);chunks.append(chunk);return chunk
    monkeypatch.setattr(_Chunk,'create',classmethod(tracked))
    try:
        with pytest.raises(TypeError):renderer.add_mesh(handle,layer=None)
        assert id(handle) not in renderer._groups
        for chunk in chunks:
            with pytest.raises(Exception):chunk.vertex_buffer.read()
    finally:
        for chunk in chunks:chunk.release()


def test_rebuild_double_fault_restores_old_group_and_preserves_primary(renderer,monkeypatch):
    from any3dview.gpu.renderer import _Chunk
    handle=MeshHandle(mesh());renderer.add_mesh(handle);old=renderer._groups[id(handle)]
    original_bytes=old.chunks[0][1].vertex_buffer.read();release=_Chunk.release
    handle.update_positions(handle.mesh.positions+.1)
    def fail_release(chunk):
        release(chunk)
        if all(chunk is not previous for _,previous in old.chunks):raise RuntimeError('secondary release failure')
    monkeypatch.setattr(_Chunk,'release',fail_release)
    semantic=renderer.set_semantic_masks
    monkeypatch.setattr(renderer,'set_semantic_masks',lambda *a,**kw:(_ for _ in ()).throw(ValueError('primary semantic failure')))
    with pytest.raises(ValueError,match='primary semantic failure'):renderer._sync(old)
    assert renderer._groups[id(handle)] is old and renderer.pick_dirty
    assert old.chunks[0][1].vertex_buffer.read()==original_bytes
    monkeypatch.setattr(renderer,'set_semantic_masks',semantic);monkeypatch.setattr(_Chunk,'release',release)
    renderer._sync(old)
    assert renderer._groups[id(handle)] is not old


@pytest.mark.parametrize('options', [
    {'layer': None}, {'line_width': 'x'},
    {'stipple_phase': float('inf')}, {'layer': float('nan')},
])
def test_invalid_replacement_style_preserves_published_group_without_allocations(renderer,monkeypatch,options):
    from any3dview.gpu.renderer import _Chunk
    handle=MeshHandle(mesh());renderer.add_mesh(handle)
    old=renderer._groups[id(handle)];chunk=old.chunks[0][1]
    before=chunk.vertex_buffer.read();renderer.pick_dirty=False
    def unexpected_allocation(*args,**kwargs):
        pytest.fail('invalid style reached GPU allocation')
    monkeypatch.setattr(_Chunk,'create',unexpected_allocation)
    with pytest.raises((TypeError,ValueError,OverflowError)):
        renderer.add_mesh(handle,**options)
    assert renderer._groups[id(handle)] is old
    assert chunk.vertex_buffer.read()==before
    assert not renderer.pick_dirty
