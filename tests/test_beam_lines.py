import json
from pathlib import Path
import numpy as np
import pytest
from any3dview import MeshArrays,MeshHandle,Camera3D,Point3D


def beam(**fields):
    values=dict(positions=np.array([[-1.,-.6,0.],[1.,-.6,0.],[-1.,.6,0.],[1.,.6,0.]],dtype=np.float32),triangles=np.empty((0,3),dtype=np.uint32),lines=np.array([[0,1],[2,3]],dtype=np.uint32),line_to_element=np.array([1,0],dtype=np.uint32),element_ids=np.array([901,305],dtype=np.uint64))
    values.update(fields);return MeshArrays(**values)


def test_trailing_line_mapping_mixed_counts_and_retained_generations():
    mesh=beam(element_scalars=np.array([.2,.8],dtype=np.float64));assert mesh.element_count==2
    copied=mesh.owned_copy();assert not np.shares_memory(copied.line_to_element,mesh.line_to_element)
    handle=MeshHandle(mesh);positions=mesh.positions;lines=mesh.lines;mapping=mesh.line_to_element
    handle.update_element_scalars([.8,.2]);handle.update_node_scalars([0.,1.,0.,1.]);handle.update_element_scalars(None)
    assert handle.mesh.positions is positions and handle.mesh.lines is lines and handle.mesh.line_to_element is mapping
    assert handle.generations.scalar==3 and handle.generations.topology==0 and handle.generations.position==0
    mixed=MeshArrays(mesh.positions,np.array([[0,1,2]],dtype=np.uint32),lines=mesh.lines,line_to_element=[2,1],element_ids=[10,20,30]);assert mixed.element_count==3
    old=MeshArrays(mesh.positions,np.empty((0,3),dtype=np.uint32),mesh.lines);assert old.line_to_element is None and old.element_count==0


@pytest.mark.parametrize("mutation",[{"line_to_element":[0]},{"line_to_element":[-1,0]},{"line_to_element":[0,2]},{"line_to_element":[[0],[1]]},{"line_to_element":[.5,0.]},{"element_scalars":[[1.],[2.]]},{"node_scalars":[[1.]]},{"element_scalars":[1.,np.inf]}])
def test_invalid_line_mapping_scalar_shape_and_infinite_values_refused(mutation):
    with pytest.raises(ValueError):beam(**mutation)


@pytest.fixture
def renderer():
    import moderngl
    from any3dview.gpu.renderer import ModernGLRenderer
    context=moderngl.create_standalone_context(require=330)
    root=Path(r"C:/ANYworkspaceWork/anypost-20261012/beam-viewer-evidence")
    (root/"gpu-runtime.json").write_text(json.dumps({key:context.info.get(key) for key in ("GL_VENDOR","GL_RENDERER","GL_VERSION")}),encoding="utf-8")
    renderer=ModernGLRenderer(context)
    yield renderer
    renderer.release();context.release()


def sample(framebuffer,camera,point):
    projected=camera.project_point(Point3D(*point),160,120);assert projected is not None
    x,y=(int(round(v)) for v in projected)
    return np.array(tuple(framebuffer.read(viewport=(x,119-y,1,1),components=3,alignment=1)))


def pick(renderer,camera,point):
    projected=camera.project_point(Point3D(*point),160,120);assert projected is not None
    x,y=(int(round(v)) for v in projected)
    return renderer.pick(x,y,camera,(160,120))


def draw(renderer,handle):
    renderer.add_mesh(handle,line_width=8.,lit=False,cull_backface=False,invalid_color="#ff00ff")
    framebuffer=renderer.ctx.simple_framebuffer((160,120),components=4);camera=Camera3D()
    renderer.render(camera,(160,120),target=framebuffer,shading_enabled=False)
    return framebuffer,camera


def test_gpu_line_element_endpoint_colours_missing_active_and_primitive_ids(renderer):
    handle=MeshHandle(beam(element_scalars=np.array([1.,0.],dtype=np.float32)))
    framebuffer,camera=draw(renderer,handle)
    lower=sample(framebuffer,camera,(0.,-.6,0.));upper=sample(framebuffer,camera,(0.,.6,0.))
    assert lower[2]>lower[0] and upper[0]>upper[2]
    assert pick(renderer,camera,(0.,-.6,0.))==(handle,"line",0)
    assert pick(renderer,camera,(0.,.6,0.))==(handle,"line",1)
    handle.update_element_scalars([np.nan,0.]);renderer.render(camera,(160,120),target=framebuffer,shading_enabled=False)
    invalid=sample(framebuffer,camera,(0.,.6,0.));assert invalid[0]>240 and invalid[2]>240 and invalid[1]<20
    handle.set_active_elements([False,True]);renderer.render(camera,(160,120),target=framebuffer)
    assert pick(renderer,camera,(0.,.6,0.)) is None
    assert pick(renderer,camera,(0.,-.6,0.))==(handle,"line",0)
    framebuffer.release()


def test_gpu_node_endpoint_interpolation_scalar_only_buffers(renderer):
    handle=MeshHandle(beam(node_scalars=np.array([0.,1.,0.,1.],dtype=np.float32)))
    framebuffer,camera=draw(renderer,handle)
    low=sample(framebuffer,camera,(-.95,-.6,0.));high=sample(framebuffer,camera,(.95,-.6,0.));middle=sample(framebuffer,camera,(0.,-.6,0.))
    assert low[2]>low[0] and high[0]>high[2] and not np.array_equal(middle,low) and not np.array_equal(middle,high)
    renderer.pick(80,60,camera,(160,120));assert renderer.pick_dirty is False
    chunk=renderer._groups[id(handle)].chunks[0][1]
    resources={name:(getattr(chunk,name).glo,getattr(chunk,name).read()) for name in ("vertex_buffer","line_buffer","line_endpoint_buffer","index_buffer")}
    vaos=(chunk.line_render_vao.glo,chunk.line_pick_vao.glo,chunk.pick_vao.glo)
    pick_buffer=(renderer._pick_fbo.glo,renderer._pick_texture.glo) if hasattr(renderer,"_pick_texture") else (renderer._pick_fbo.glo,)
    uploads=renderer.geometry_uploads;scalar_uploads=renderer.scalar_buffer_updates
    handle.update_node_scalars([1.,0.,1.,0.]);renderer.render(camera,(160,120),target=framebuffer,shading_enabled=False)
    assert renderer.geometry_uploads==uploads and renderer.scalar_buffer_updates==scalar_uploads+3
    assert renderer.pick_dirty is False
    for name,expected in resources.items():assert (getattr(chunk,name).glo,getattr(chunk,name).read())==expected
    assert (chunk.line_render_vao.glo,chunk.line_pick_vao.glo,chunk.pick_vao.glo)==vaos
    assert renderer._pick_fbo.glo==pick_buffer[0]
    swapped=sample(framebuffer,camera,(-.95,-.6,0.));assert swapped[0]>swapped[2]
    framebuffer.release()


def test_gpu_mapped_element_semantic_overlays_and_hidden_selection(renderer):
    handle=MeshHandle(beam(element_scalars=[1.,0.]));framebuffer,camera=draw(renderer,handle)
    renderer.set_semantic_masks(handle,hidden_elements=(0,),preselected_elements=(1,))
    renderer.render(camera,(160,120),target=framebuffer,preselection_color="#00ff00")
    assert pick(renderer,camera,(0.,.6,0.)) is None
    lower=sample(framebuffer,camera,(0.,-.6,0.));assert lower[1]>lower[0] and lower[1]>lower[2]
    assert pick(renderer,camera,(0.,-.6,0.))==(handle,"line",0)
    framebuffer.release()


def test_fallback_pick_identity_preserves_line_key_and_semantic_owners():
    from any3dview.retained_viewer import RetainedViewer
    from any3dview import ModelOwner,PackedOwnerTable
    viewer=object.__new__(RetainedViewer);handle=MeshHandle(beam());entry={"tags":(),"owners":None}
    binding=viewer._selection_binding(handle,entry,"line",0)
    assert binding.owners[0].key==f"mesh:{id(handle)}:line:0" and binding.owners[0].kind=="mesh.line"
    assert binding.owners[0].identity==305
    supplied=ModelOwner("3e0d7a91-e385-4905-885e-b5a554294e91","edge",7)
    entry["owners"]=PackedOwnerTable.from_owners(lines=((supplied,),()))
    actual=viewer._selection_binding(handle,entry,"line",0)
    assert actual.owners[0].identity==supplied and actual.owners[0].key=="3e0d7a91-e385-4905-885e-b5a554294e91:edge:7"


def test_software_line_field_refusal_is_atomic_for_updates_and_chunks():
    from any3dview.retained_viewer import RetainedViewer
    from types import SimpleNamespace
    viewer=SimpleNamespace(backend_name="software")
    validate=lambda mesh:RetainedViewer._validate_mesh_support(viewer,mesh)
    handle=MeshHandle(beam(),validate_mesh=validate);before=handle.mesh;generations=handle.generations
    for update in (lambda:handle.update_element_scalars([1.,2.]),lambda:handle.update_node_scalars([0.,1.,0.,1.]),lambda:handle.set_active_elements([True,False]),lambda:handle.add_chunk("bad",beam(element_scalars=[1.,2.]))):
        with pytest.raises(NotImplementedError,match="line"):update()
        assert handle.mesh is before and handle.generations==generations and not handle.chunks
    with pytest.raises(NotImplementedError):MeshHandle(beam(element_scalars=[1.,2.]),validate_mesh=validate)


def test_qt_automatic_fallback_refuses_before_releasing_or_attaching_lines():
    from types import SimpleNamespace
    from any3dview.qt import QtViewer
    from any3dview.retained_viewer import RetainedViewer
    messages=[];changes=[];handle=MeshHandle(beam(element_scalars=[1.,2.]))
    fake=SimpleNamespace(_closed=False,_backend="gpu",backend_name="gpu",_backend_diagnostics=(),_allow_fallback=True,
        surface=SimpleNamespace(error=RuntimeError("context lost"),hide=lambda:changes.append("hide")),
        render_error=SimpleNamespace(emit=messages.append),setToolTip=lambda value:None,_release_gl=lambda:changes.append("release"),
        _display_entries=lambda:{1:{"handle":handle}})
    fake._validate_mesh_support=lambda mesh,**kw:RetainedViewer._validate_mesh_support(fake,mesh,**kw)
    QtViewer._render_failed(fake)
    assert not changes and fake._backend=="gpu" and len(messages)==2
    assert "fallback refused" in messages[-1] and "line" in fake._backend_diagnostics[-1]


def test_gpu_mixed_triangles_and_lines_keep_distinct_element_rows(renderer):
    mesh=MeshArrays(np.array([[-1.4,-.5,0.],[-.3,-.5,0.],[-.85,.7,0.],[.25,0.,0.],[1.3,0.,0.]],dtype=np.float32),np.array([[0,1,2]],dtype=np.uint32),lines=np.array([[3,4]],dtype=np.uint32),triangle_to_element=[0],line_to_element=[2],element_ids=[101,102,103],element_scalars=[0.,.5,1.])
    handle=MeshHandle(mesh);framebuffer,camera=draw(renderer,handle)
    triangle=sample(framebuffer,camera,(-.85,-.1,0.));line=sample(framebuffer,camera,(.7,0.,0.))
    assert triangle[2]>triangle[0] and line[0]>line[2]
    assert pick(renderer,camera,(-.85,-.1,0.))==(handle,"triangle",0)
    assert pick(renderer,camera,(.7,0.,0.))==(handle,"line",0)
    handle.set_active_elements([True,True,False]);renderer.render(camera,(160,120),target=framebuffer)
    assert pick(renderer,camera,(.7,0.,0.)) is None
    assert pick(renderer,camera,(-.85,-.1,0.))==(handle,"triangle",0)
    framebuffer.release()
