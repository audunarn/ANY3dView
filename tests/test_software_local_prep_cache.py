"""Local software preparation reuse and forced-cold raster equivalence."""
from types import SimpleNamespace
import numpy as np
import pytest
pytest.importorskip('PySide6')
from PySide6.QtGui import QImage,QPainter
from any3dview import MeshArrays
from any3dview import qt
from any3dview.retained_viewer import RetainedViewer


class Viewer(RetainedViewer):
    def __init__(self):
        self.initialize_state(width=280,height=180)
        self._renderer=qt._SoftwareResources();self._software_prep={}
        self._hud=qt._PainterHud();self._render_hud=lambda _:None
        self.surface=SimpleNamespace(devicePixelRatioF=lambda:1.)
        self._host=SimpleNamespace(make_current=lambda:None,framebuffer_size=lambda:(280,180))
        for name in ('_software_prep_signature','_software_chunk_prep','_prune_software_prep'):
            setattr(self,name,getattr(qt.QtViewer,name).__get__(self))
    def redraw(self):pass


def mesh(offset=0.):
    return MeshArrays(np.asarray(((offset,0.,0.),(offset+1.,0.,0.),(offset+1.,1.,0.),(offset,1.,0.))),
        np.asarray(((0,1,2),(0,2,3)),dtype=np.uint32),
        lines=np.asarray(((0,1),(1,2)),dtype=np.uint32),point_indices=np.asarray((0,2),dtype=np.uint32),
        displacements=np.asarray(((0.,0.,.1),)*4))


def scene():
    viewer=Viewer()
    first=viewer.add_mesh_arrays(mesh(),tags='first',color='#123456',cull_backface=False)
    second=viewer.add_mesh_arrays(mesh(2),tags='second',color='#abcdef',cull_backface=False)
    viewer.set_top_view();viewer.fit_to_scene()
    return viewer,first,second


def render(viewer):
    image=QImage(280,180,QImage.Format_RGBA8888)
    qt.QtViewer._paint_software(viewer,QPainter(image))
    return bytes(image.constBits())


def assert_cold_parity(viewer):
    warm=render(viewer)
    viewer._software_prep.clear()
    assert render(viewer)==warm
    return warm


def test_selection_and_hover_preserve_unaffected_warm_prep_and_pixels():
    viewer,first,second=scene();before=render(viewer)
    initial=dict(viewer._software_prep)
    viewer.set_highlight(['first'])
    selected=render(viewer)
    assert selected!=before
    assert viewer._software_prep[id(first),None] is not initial[id(first),None]
    assert viewer._software_prep[id(second),None] is initial[id(second),None]
    assert_cold_parity(viewer)
    warm=dict(viewer._software_prep)
    viewer.set_preselection('second');hover=render(viewer)
    assert hover!=selected
    assert viewer._software_prep[id(first),None] is warm[id(first),None]
    assert viewer._software_prep[id(second),None] is not warm[id(second),None]
    assert_cold_parity(viewer)
    warm=dict(viewer._software_prep)
    # Repeated semantic publication can advance renderer.mask_revision without
    # changing even the selected/preselected handle's effective masks.
    viewer._apply_highlight_masks();render(viewer)
    assert all(viewer._software_prep[key] is value for key,value in warm.items())


@pytest.mark.parametrize('masks',[
    {'hidden_elements':(0,)},{'selected_elements':(0,)},{'preselected_elements':(0,)},
    {'hidden_lines':(0,)},{'selected_lines':(0,)},{'hidden_points':(0,)},{'selected_points':(0,)},
    {'preselected_lines':(0,)},{'preselected_points':(0,)},
])
def test_all_mask_channels_are_local_and_match_cold_raster(masks):
    viewer,first,second=scene();render(viewer);warm=dict(viewer._software_prep)
    viewer._renderer.set_semantic_masks(first,**masks)
    render(viewer)
    assert viewer._software_prep[id(first),None] is not warm[id(first),None]
    assert viewer._software_prep[id(second),None] is warm[id(second),None]
    assert_cold_parity(viewer)


def test_chunk_masks_do_not_invalidate_base_or_sibling_chunks():
    viewer,first,second=scene();first.add_chunk('one',mesh(4));first.add_chunk('two',mesh(6))
    viewer.fit_to_scene();render(viewer);warm=dict(viewer._software_prep)
    viewer._renderer.set_chunk_semantic_masks(first,'one',hidden_elements=(0,),selected_lines=(0,))
    render(viewer)
    assert viewer._software_prep[id(first),'one'] is not warm[id(first),'one']
    for key in ((id(first),None),(id(first),'two'),(id(second),None)):
        assert viewer._software_prep[key] is warm[key]
    assert_cold_parity(viewer)
    before=viewer._software_prep[id(first),'one']
    viewer._renderer.clear_chunk_semantic_masks(first,'one');render(viewer)
    assert viewer._software_prep[id(first),'one'] is not before
    assert_cold_parity(viewer)
    first.remove_chunk('one');render(viewer)
    assert (id(first),'one') not in viewer._software_prep


@pytest.mark.parametrize('change',[
    lambda v,h:h.update_positions(h.mesh.positions+.1),
    lambda v,h:h.update_displacements(h.mesh.displacements+.1),
    lambda v,h:h.set_deformation_scale(3.),
    lambda v,h:h.set_selected_elements([0]),
    lambda v,h:h.set_active_elements([True,False]),
    lambda v,h:v.set_mesh_style(h,color='#ef1234'),
    lambda v,h:v.set_section_plane((1,0,0),.5),
    lambda v,h:h.set_transform(np.asarray(((1.,0,0,.1),(0,1.,0,0),(0,0,1.,0),(0,0,0,1.)))),
])
def test_geometry_style_section_and_handle_selection_stay_invalidated(change):
    viewer,first,second=scene();render(viewer);warm=viewer._software_prep[id(first),None]
    change(viewer,first);render(viewer)
    assert viewer._software_prep[id(first),None] is not warm
    assert_cold_parity(viewer)


def test_removed_and_hidden_handles_release_preparation():
    viewer,first,second=scene();render(viewer)
    first.set_visible(False);render(viewer)
    assert (id(first),None) not in viewer._software_prep
    first.set_visible(True);render(viewer)
    assert (id(first),None) in viewer._software_prep
    first.remove();render(viewer)
    assert (id(first),None) not in viewer._software_prep
    assert_cold_parity(viewer)


@pytest.mark.parametrize('ratio',[1.,1.5,2.])
@pytest.mark.parametrize('mode',['plain','selected','hover','alpha','mixed','masked','section','deformed'])
def test_painter_state_reuse_matches_prechange_reference_exactly(ratio,mode):
    from _software_painter_reference import reference_paint_software
    viewer,first,second=scene()
    viewer.surface.devicePixelRatioF=lambda:ratio
    viewer._host.framebuffer_size=lambda:(round(280*ratio),round(180*ratio))
    if mode=='selected':viewer.set_highlight(['first','second'])
    if mode=='hover':viewer.set_preselection('second')
    if mode=='alpha':viewer.set_mesh_style(first,opacity=.37,color='#ed1234',outline='#7d01ef',width=2.)
    if mode=='mixed':
        viewer.set_mesh_style(first,color='#ed1234',outline='#7d01ef',width=3.)
        viewer.set_mesh_style(second,color='#ed1234',outline='#7d01ef',width=1.)
    if mode=='masked':viewer._renderer.set_semantic_masks(first,hidden_elements=(1,),selected_lines=(0,),hidden_points=(1,))
    if mode=='section':viewer.set_section_plane((1,0,0),.5)
    if mode=='deformed':first.set_deformation_scale(5.)
    optimized=render(viewer)
    expected=QImage(280,180,QImage.Format_RGBA8888)
    reference_paint_software(viewer,QPainter(expected))
    assert optimized==bytes(expected.constBits())


def test_painter_state_reuse_reduces_calls_without_changing_draw_operations():
    from _software_painter_reference import reference_paint_software
    class Counter:
        def __init__(self):
            self.image=QImage(280,180,QImage.Format_RGBA8888);self.actual=QPainter(self.image);self.calls={}
        def __getattr__(self,name):
            original=getattr(self.actual,name)
            def counted(*args,**kwargs):
                self.calls[name]=self.calls.get(name,0)+1
                return original(*args,**kwargs)
            return counted
    viewer,first,second=scene()
    optimized=Counter();qt.QtViewer._paint_software(viewer,optimized)
    reference=Counter();reference_paint_software(viewer,reference)
    assert bytes(optimized.image.constBits())==bytes(reference.image.constBits())
    for name in ('drawPolygon','drawPolyline','drawLine','drawEllipse'):
        assert optimized.calls.get(name,0)==reference.calls.get(name,0)
    for name in ('setRenderHint','setPen','setBrush'):
        assert optimized.calls[name]<reference.calls[name]
