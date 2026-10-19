"""GPU streaming, immutable frame reuse and pick-cache regressions."""
import numpy as np
import pytest
from any3dview import MeshArrays, MeshHandle
from any3dview.retained_viewer import _animation_snapshot
from test_performance_review_oracle import Scene, mesh as animation_mesh


@pytest.fixture
def gpu():
    import moderngl
    from any3dview.gpu.renderer import ModernGLRenderer
    ctx = moderngl.create_standalone_context(require=330)
    renderer = ModernGLRenderer(ctx)
    try:
        yield renderer
    finally:
        renderer.release()
        ctx.release()


def geometry():
    return MeshArrays(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),
                      np.array([[0,1,2]],dtype=np.uint32),
                      lines=np.array([[0,1],[1,2]],dtype=np.uint32),
                      point_indices=np.array([0],dtype=np.uint32))


def test_deformation_streams_only_dynamic_buffers_and_retries(gpu):
    handle=MeshHandle(geometry()); gpu.add_mesh(handle)
    group=gpu._groups[id(handle)]; chunk=group.chunks[0][1]
    static={name:(getattr(chunk,name),getattr(chunk,name).read())
            for name in ('vertex_buffer','line_buffer','index_buffer','line_endpoint_buffer')}
    displacement=np.arange(9,dtype=np.float64).reshape(3,3)/10
    handle.update_displacements(displacement)
    original=chunk.line_displacement_buffer
    class Fail:
        def write(self,*args): raise RuntimeError('stream failure')
    chunk.line_displacement_buffer=Fail()
    with pytest.raises(RuntimeError,match='stream failure'): gpu._sync(group)
    assert group.generations.displacement != handle.generations.displacement
    chunk.line_displacement_buffer=original
    gpu._sync(group)
    for name,(buffer,data) in static.items():
        assert getattr(chunk,name) is buffer and buffer.read()==data
    np.testing.assert_array_equal(np.frombuffer(chunk.displacement_buffer.read(),np.float32).reshape(3,3),displacement.astype(np.float32))
    np.testing.assert_array_equal(np.frombuffer(original.read(),np.float32).reshape(2,2,3),displacement.astype(np.float32)[handle.mesh.lines])
    handle.update_displacements(None);gpu._sync(group)
    assert not np.frombuffer(chunk.displacement_buffer.read(),np.float32).any()
    assert not np.frombuffer(original.read(),np.float32).any()
    gpu.remove_mesh(handle)
    with pytest.raises(Exception): original.read()
    with pytest.raises(Exception): chunk.displacement_buffer.read()


@pytest.mark.parametrize('chunked',[False,True])
@pytest.mark.parametrize('kind',['elements','lines','points'])
def test_emphasis_keeps_pick_cache_but_hidden_masks_invalidate(gpu,chunked,kind):
    handle=MeshHandle(geometry())
    if chunked: handle.add_chunk('part',geometry())
    gpu.add_mesh(handle)
    def masks(**kw):
        if chunked: gpu.set_chunk_semantic_masks(handle,'part',**kw)
        else: gpu.set_semantic_masks(handle,**kw)
    gpu.pick_dirty=False
    masks(**{'selected_'+kind:[0]});assert not gpu.pick_dirty
    masks(**{'preselected_'+kind:[0]});assert not gpu.pick_dirty
    masks();assert not gpu.pick_dirty
    masks(**{'hidden_'+kind:[0]});assert gpu.pick_dirty
    gpu.pick_dirty=False
    masks(**{'hidden_'+kind:[0],'selected_'+kind:[0]});assert not gpu.pick_dirty
    masks();assert gpu.pick_dirty


def test_snapshots_share_equal_arrays_but_are_permanently_immutable():
    source=animation_mesh(); first=_animation_snapshot(source)
    second=_animation_snapshot(source,first)
    for name in source.__dataclass_fields__:
        a=getattr(first,name);b=getattr(second,name)
        if a is not None:
            assert a is b
            with pytest.raises(ValueError): a.flags.writeable=True
    source.positions[0,0]=9
    assert first.positions[0,0]==0
    third=_animation_snapshot(source,first)
    assert third.positions is not first.positions
    assert third.triangles is first.triangles
    mutable=animation_mesh()
    other=_animation_snapshot(mutable,mutable)
    assert other.positions is not mutable.positions


def test_cached_playback_rejects_replaced_normals_and_live_chunk_changes():
    view=Scene();h=view.add_mesh_arrays(animation_mesh());h.add_chunk('part',animation_mesh())
    view.begin_animation_cache();view.capture_animation_frame();view.capture_animation_frame()
    a,b=(f['meshes'][0] for f in view._animation_cache)
    assert a['mesh'].normals is b['mesh'].normals
    assert a['chunks'][0][1].positions is b['chunks'][0][1].positions
    view._show_animation_frame(0);original=view._animation_handles[0]
    view._show_animation_frame(1);assert view._animation_handles[0] is original
    original.replace_chunk('part',animation_mesh(shift=3))
    view._show_animation_frame(0);assert view._animation_handles[0] is not original
    b['mesh']=animation_mesh(normal=-1)
    view._show_animation_frame(1)
    np.testing.assert_array_equal(view._animation_handles[0].mesh.normals,animation_mesh(normal=-1).normals)


def test_clearing_hidden_chunk_masks_invalidates_before_writing(gpu):
    h=MeshHandle(geometry());h.add_chunk('part',geometry());gpu.add_mesh(h)
    gpu.set_chunk_semantic_masks(h,'part',selected_elements=[0]);gpu.pick_dirty=False
    gpu.clear_chunk_semantic_masks(h,'part');assert not gpu.pick_dirty
    gpu.set_chunk_semantic_masks(h,'part',hidden_elements=[0]);gpu.pick_dirty=False
    gpu.clear_chunk_semantic_masks(h,'part');assert gpu.pick_dirty
