"""Smooth shading must retain explicit indexed polygon boundary identity."""
import numpy as np
import pytest
from any3dview import MeshArrays
from any3dview.retained_viewer import _surface_outline_loops
from test_numeric_face_batches import Capture,LegacyCapture

@pytest.mark.parametrize('capture',[Capture,LegacyCapture])
@pytest.mark.parametrize('outline',['','black'])
def test_smoothed_quad_boundary_ignores_fan_diagonal(capture,outline):
    viewer=capture();polygon=np.array([[0,0,0],[2,0,0],[2,1,0],[0,1,0]],float)
    viewer.add_faces([polygon],smooth_angle=35,outline=outline)
    mesh,options=viewer.outputs[0]
    loops=list(_surface_outline_loops(mesh,range(mesh.triangle_count),options['_outline_groups'],options['_outline_vertex_ids']))
    assert len(loops)==1 and len(loops[0])==4
    assert mesh.normals is not None and len(mesh.positions)==6
    # No inference from equal coordinates: IDs come from the pre-expansion fan.
    assert options['_outline_vertex_ids'].tolist()==[0,1,2,0,2,3]


def test_true_triangle_elements_keep_shared_diagonal():
    viewer=Capture();q=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],float)
    viewer.add_faces([q[[0,1,2]],q[[0,2,3]]],smooth_angle=35)
    mesh,options=viewer.outputs[0]
    loops=list(_surface_outline_loops(mesh,range(mesh.triangle_count),options['_outline_groups'],options['_outline_vertex_ids']))
    assert len(loops)==2 and all(len(loop)==3 for loop in loops)


def test_animation_entry_carries_outline_identity():
    from any3dview.retained_viewer import RetainedViewer
    from types import SimpleNamespace
    viewer=Capture();viewer.add_faces([np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],float)],smooth_angle=35)
    mesh,options=viewer.outputs[0];handle=SimpleNamespace(mesh=mesh)
    item=dict(owners=None,owner_resolver=None,appearance={},outline_groups=options['_outline_groups'],outline_vertex_ids=options['_outline_vertex_ids'],layer=5,tags=frozenset(),item=1,semantic_elements={},hit_elements={},semantic_lines={},hit_lines={},semantic_points={},hit_points={},chunk_semantics={})
    entry=RetainedViewer._animation_entry(viewer,handle,item)
    assert entry['outline_vertex_ids'] is item['outline_vertex_ids']
    loops=list(_surface_outline_loops(mesh,range(mesh.triangle_count),entry['outline_groups'],entry['outline_vertex_ids']))
    assert len(loops)==1 and len(loops[0])==4

def test_captured_animation_frame_restores_smoothed_outline_topology(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from any3dview.qt import QtViewer
    app=QApplication.instance() or QApplication([])
    viewer=QtViewer(backend='software');viewer.resize(240,180)
    try:
        viewer.add_faces([np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],float)],smooth_angle=35)
        original=next(iter(viewer._entries.values()));normals=original['handle'].mesh.normals.copy()
        viewer.capture_animation_frame();viewer.capture_animation_frame()
        first,second=(frame['meshes'][0] for frame in viewer._animation_cache)
        assert first['outline_vertex_ids'] is not second['outline_vertex_ids']
        np.testing.assert_array_equal(first['outline_vertex_ids'],[0,1,2,0,2,3])
        original['outline_vertex_ids'][0]=99
        viewer._show_animation_frame(0)
        restored=next(iter(viewer._animation_entries.values()));mesh=restored['handle'].mesh
        np.testing.assert_array_equal(mesh.normals,normals)
        loops=list(_surface_outline_loops(mesh,range(mesh.triangle_count),restored['outline_groups'],restored['outline_vertex_ids']))
        assert len(loops)==1 and len(loops[0])==4
        viewer._show_animation_frame(1)  # In-place frame update keeps topology metadata.
        restored=next(iter(viewer._animation_entries.values()));mesh=restored['handle'].mesh
        loops=list(_surface_outline_loops(mesh,range(mesh.triangle_count),restored['outline_groups'],restored['outline_vertex_ids']))
        assert len(loops)==1 and len(loops[0])==4
    finally:
        viewer.close();app.processEvents()
