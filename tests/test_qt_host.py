"""Qt host contracts and shared retained semantics; real GPU is separately opt-in."""
import os
import subprocess
import sys

import numpy as np
import pytest

if os.environ.get("ANY3DVIEW_RUN_QT_TESTS")!="1":
    pytest.skip("set ANY3DVIEW_RUN_QT_TESTS=1",allow_module_level=True)
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMainWindow
from any3dview import create_viewer, MeshArrays, ViewerBackend
from any3dview.qt import QtViewerHostAdapter
from any3dview.errors import GPUUnavailableError

@pytest.fixture(scope="module")
def qapp():return QApplication.instance() or QApplication([])

def triangle():
    return MeshArrays(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),np.array([[0,1,2]],dtype=np.uint32))

def test_qt_software_retained_render_selection_capture(qapp):
    viewer=create_viewer(None,"software",host=QtViewerHostAdapter())
    try:
        assert isinstance(viewer,ViewerBackend)
        viewer.show();qapp.processEvents()
        handle=viewer.add_mesh_arrays(triangle(),color="#ff0000",tags="triangle",cull_backface=False)
        viewer.set_top_view();viewer.fit_to_scene();qapp.processEvents()
        projected=viewer.project_point((0.25,0.25,0))
        assert viewer.pick_at(int(projected[0]),int(projected[1]))=="triangle"
        image=viewer.capture_image();assert image.width>0
        assert len(np.unique(np.asarray(image).reshape(-1,4),axis=0))>1
        viewer.set_highlight(["triangle"]);assert "triangle" in viewer.highlighted_tags()
        handle.set_visible(False);qapp.processEvents()
        viewer.set_section_plane((1,0,0),0.5);handle.set_visible(True);qapp.processEvents()
        viewer.capture_image()
    finally:viewer.destroy();qapp.processEvents()

def test_disabled_gpu_auto_fallback_and_explicit_failure(qapp,monkeypatch):
    monkeypatch.setenv("ANY3DVIEW_DISABLE_GPU","1")
    with pytest.raises(GPUUnavailableError):create_viewer(None,"gpu",host=QtViewerHostAdapter())
    viewer=create_viewer(None,"auto",host=QtViewerHostAdapter())
    assert viewer.backend_name=="software"
    assert viewer.backend_diagnostics
    viewer.destroy();qapp.processEvents()


def _center_color(viewer, qapp, point=(.5, .5, 0)):
    qapp.processEvents()
    image = np.asarray(viewer.capture_image())
    x, y, _ = viewer.project_point(point)
    return image[int(y), int(x), :3].astype(int)


def _center_edge_contrast(viewer, qapp):
    qapp.processEvents()
    image = np.asarray(viewer.capture_image()).astype(int)
    x, y, _ = viewer.project_point((.5,.5,0))
    x, y = round(x), round(y)
    return np.max(np.abs(image[y-2:y+3,x-2:x+3,:3] - image[y,x+9,:3]))


@pytest.mark.parametrize("state", ["plain", "highlight", "preselection"])
def test_software_quad_has_boundary_without_fill_diagonal(qapp, state):
    viewer = create_viewer(None, "software", host=QtViewerHostAdapter())
    try:
        viewer.resize(640, 480); viewer.show()
        viewer.add_faces([[(0,0,0), (1,0,0), (1,1,0), (0,1,0)]],
                         colors="#e08040", outline="#000000", width=3, tags="quad")
        viewer.set_top_view(); viewer.fit_to_scene()
        if state == "highlight": viewer.set_highlight(["quad"])
        if state == "preselection": viewer.set_preselection("quad")
        diagonal = _center_color(viewer, qapp)
        interior = _center_color(viewer, qapp, (.55,.45,0))
        assert np.max(np.abs(diagonal - interior)) < 20
        if state == "highlight":
            viewer.begin_animation_cache(); viewer.capture_animation_frame()
            viewer._show_animation_frame(0)
            assert np.max(np.abs(_center_color(viewer, qapp) - _center_color(viewer, qapp, (.55,.45,0)))) < 20
        assert viewer.pick_at(*map(int, viewer.project_point((.5,.5,0))[:2])) == "quad"
        viewer.set_section_plane((1,0,0), .25)
        diagonal = _center_color(viewer, qapp)
        interior = _center_color(viewer, qapp, (.55,.45,0))
        assert np.max(np.abs(diagonal - interior)) < 20
    finally:
        viewer.destroy(); qapp.processEvents()


def test_outline_topology_preserves_element_boundaries_and_holes():
    from any3dview.retained_viewer import RetainedViewer, _surface_outline_loops
    positions = np.zeros((8,3))
    triangles = np.array([(0,1,5),(0,5,4),(1,2,6),(1,6,5),
                          (2,3,7),(2,7,6),(3,0,4),(3,4,7)], dtype=np.uint32)
    ring = MeshArrays(positions, triangles, triangle_to_element=np.zeros(8,dtype=np.uint32))
    loops = list(_surface_outline_loops(ring, range(8)))
    assert {frozenset(loop) for loop in loops} == {frozenset(range(4)), frozenset(range(4,8))}
    # Separate actual triangles retain shared edges even with identical positions.
    actual = MeshArrays(positions, triangles)
    assert len(list(_surface_outline_loops(actual, range(8)))) == 8
    entry = {"outline_groups": np.zeros(8,dtype=np.uint32), "outline_topology": triangles}
    assert RetainedViewer._outline_groups(entry,None,actual) is entry["outline_groups"]
    replacement = MeshArrays(positions,triangles[::-1])
    assert RetainedViewer._outline_groups(entry,None,replacement) is None
    assert entry["outline_groups"] is None


def test_software_clipped_hole_chunk_transform_deformation_and_visibility(qapp):
    viewer = create_viewer(None, "software", host=QtViewerHostAdapter())
    positions = np.array([(0.,0.,0.),(3.,0.,0.),(3.,3.,0.),(0.,3.,0.),
                          (1.,1.,0.),(2.,1.,0.),(2.,2.,0.),(1.,2.,0.)])
    triangles = np.array([(0,1,5),(0,5,4),(1,2,6),(1,6,5),
                          (2,3,7),(2,7,6),(3,0,4),(3,4,7)],dtype=np.uint32)
    mesh = MeshArrays(positions, triangles, triangle_to_element=np.zeros(8,dtype=np.uint32),
                      displacements=np.tile((.1,0,0),(8,1)))
    try:
        viewer.resize(640,480); viewer.show()
        handle = viewer.add_mesh_arrays(MeshArrays(np.empty((0,3)),np.empty((0,3),dtype=np.uint32)),
                                        color="#e08040", tags="ring", cull_backface=False)
        handle.add_chunk("ring",mesh)
        transform = np.eye(4); transform[0,3] = .2
        handle.set_transform(transform); handle.set_deformation_scale(2)
        viewer.set_top_view(); viewer.fit_to_scene()
        viewer.set_highlight(["ring"])
        outside = _center_color(viewer,qapp,(3.7,3.7,0))
        hole = _center_color(viewer,qapp,(1.9,1.5,0))
        assert np.max(np.abs(hole-outside)) < 20
        viewer.set_section_plane((1,0,0),1.9)
        # The section cut through the hole must remain empty, including its cut line.
        assert np.max(np.abs(_center_color(viewer,qapp,(1.9,1.5,0))-outside)) < 20
        # A cut through filled material remains an actual outlined boundary.
        assert np.max(np.abs(_center_color(viewer,qapp,(1.9,.5,0))-outside)) > 30
        assert viewer.pick_at(*map(int,viewer.project_point((2.9,1.5,0))[:2])) == "ring"
        handle.set_visible(False)
        assert np.max(np.abs(_center_color(viewer,qapp,(2.9,1.5,0))-outside)) < 20
    finally:
        viewer.destroy(); qapp.processEvents()


def test_software_polygon_visibility_and_per_triangle_colors(qapp):
    viewer = create_viewer(None,"software",host=QtViewerHostAdapter())
    try:
        viewer.resize(640,480); viewer.show()
        positions = np.array([(0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.)])
        mesh = MeshArrays(positions,np.array([(0,1,2),(0,2,3)],dtype=np.uint32),
                          triangle_to_element=np.array([0,0],dtype=np.uint32))
        handle = viewer.add_mesh_arrays(mesh,face_colors=("#ff0000","#0000ff"),
                                        line_color="#000000",tags="quad",cull_backface=False)
        viewer.set_top_view(); viewer.fit_to_scene()
        first = _center_color(viewer,qapp,(.75,.25,0))
        second = _center_color(viewer,qapp,(.25,.75,0))
        assert first[0] > 200 and first[2] < 50
        assert second[2] > 200 and second[0] < 50
        outside = _center_color(viewer,qapp,(1.2,1.2,0))
        handle.set_active_elements(np.array([False]))
        viewer.set_highlight(["quad"])
        assert np.max(np.abs(_center_color(viewer,qapp)-outside)) < 20
        handle.set_active_elements(np.array([True]))
        viewer._renderer.masks[id(handle)] = {"hidden_elements": np.array([0])}
        assert np.max(np.abs(_center_color(viewer,qapp)-outside)) < 20
    finally:
        viewer.destroy(); qapp.processEvents()


def test_software_explicit_element_topology_and_mesh_toggle(qapp):
    viewer = create_viewer(None, "software", host=QtViewerHostAdapter())
    positions = np.array([(0.,0.,0.), (1.,0.,0.), (1.,1.,0.), (0.,1.,0.)])
    triangles = np.array([(0,1,2), (0,2,3)], dtype=np.uint32)
    try:
        viewer.resize(640,480); viewer.show()
        # These are two actual FE triangles, even though the scene tag is shared.
        viewer.add_mesh_arrays(MeshArrays(positions, triangles), color="#e08040",
                               line_color="#000000", line_width=3, tags="mesh", cull_backface=False)
        viewer.set_top_view(); viewer.fit_to_scene()
        assert _center_edge_contrast(viewer, qapp) > 30
        viewer.set_mesh_lines(False)
        assert np.max(_center_color(viewer, qapp)) > 100
        viewer.set_highlight(["mesh"])
        assert np.max(np.abs(_center_color(viewer, qapp) - _center_color(viewer, qapp, (.55,.45,0)))) > 30
        viewer.clear(); viewer.set_mesh_lines(True)
        viewer.add_mesh_arrays(MeshArrays(positions, triangles, triangle_to_element=np.array([0,0],dtype=np.uint32)),
                               color="#e08040", line_color="#000000", line_width=3, tags="quad", cull_backface=False)
        assert np.max(np.abs(_center_color(viewer, qapp) - _center_color(viewer, qapp, (.55,.45,0)))) < 20
        viewer.set_highlight(["quad"])
        assert np.max(np.abs(_center_color(viewer, qapp) - _center_color(viewer, qapp, (.55,.45,0)))) < 20
    finally:
        viewer.destroy(); qapp.processEvents()


def test_software_nearer_fill_occludes_farther_raw_mesh_boundaries(qapp):
    viewer = create_viewer(None,"software",host=QtViewerHostAdapter())
    try:
        viewer.resize(640,480); viewer.show()
        back = np.array([(-1.,-1.,-.5),(2.,-1.,-.5),(.5,2.,-.5)])
        front = np.array([(0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.)])
        viewer.add_mesh_arrays(MeshArrays(back,np.array([(0,1,2)],dtype=np.uint32)),
                               color="#e08040",line_color="#000000",line_width=5,cull_backface=False)
        viewer.add_mesh_arrays(MeshArrays(front,np.array([(0,1,2),(0,2,3)],dtype=np.uint32),
                                          triangle_to_element=np.array([0,0],dtype=np.uint32)),
                               color="#e08040",line_color="#000000",line_width=5,cull_backface=False)
        viewer.set_top_view(); viewer.fit_to_scene()
        # A rear triangle edge crosses the interior of the nearer quad.
        assert np.max(np.abs(_center_color(viewer,qapp,(.1,.5,0)) - np.array([224,128,64]))) < 20
    finally:
        viewer.destroy(); qapp.processEvents()


@pytest.mark.parametrize("backend", ["software", "gpu"])
def test_qt_resize_capture_and_view_state_replacement(qapp, backend):
    if backend == "gpu" and os.environ.get("ANY3DVIEW_RUN_QT_GPU_TESTS") != "1":
        pytest.skip("real GPU acceptance is explicitly opt-in")
    viewer = create_viewer(None, backend, host=QtViewerHostAdapter())
    replacement = None
    try:
        viewer.show()
        viewer.add_mesh_arrays(triangle(), tags="triangle", cull_backface=False)
        viewer.set_top_view()
        viewer.fit_to_scene()
        viewer.set_light(ambient=0.2)
        viewer.update_selection_config(drag_threshold_px=9)
        for size in ((480, 320), (240, 180)):
            viewer.resize(*size)
            qapp.processEvents()
            image = viewer.capture_image()
            assert image.size == viewer.viewport_size
            assert len(np.unique(np.asarray(image).reshape(-1, 4), axis=0)) > 1
        state = viewer.export_view_state()
        replacement = create_viewer(None, "software", host=QtViewerHostAdapter())
        replacement.apply_view_state(state)
        # Camera setters normalize the orbit representation on restoration.
        assert replacement.camera.position.to_tuple() == pytest.approx(
            viewer.camera.position.to_tuple(), rel=1e-12, abs=1e-12
        )
        assert replacement.camera.target.to_tuple() == pytest.approx(
            viewer.camera.target.to_tuple(), rel=1e-12, abs=1e-12
        )
        assert replacement.selection_config.drag_threshold_px == 9
        assert replacement.light.ambient == pytest.approx(0.2)
        replacement.set_light(ambient=0.7)
        assert viewer.light.ambient == pytest.approx(0.2)
    finally:
        if replacement is not None:
            replacement.destroy()
        viewer.destroy()
        qapp.processEvents()


@pytest.mark.skipif(os.environ.get("ANY3DVIEW_RUN_QT_GPU_TESTS") != "1", reason="real GPU acceptance is explicitly opt-in")
def test_qt_gpu_reparent_retains_scene_and_selection(qapp):
    first, second = QMainWindow(), QMainWindow()
    viewer = create_viewer(None, "gpu", host=QtViewerHostAdapter())
    try:
        first.setCentralWidget(viewer)
        first.resize(400, 300)
        first.show()
        viewer.add_mesh_arrays(triangle(), tags="triangle", cull_backface=False)
        viewer.set_top_view()
        viewer.fit_to_scene()
        viewer.set_highlight(["triangle"])
        qapp.processEvents()
        second.setCentralWidget(first.takeCentralWidget())
        second.resize(480, 320)
        second.show()
        qapp.processEvents()
        assert viewer.surface.error is None
        assert "triangle" in viewer.highlighted_tags()
        projected = viewer.project_point((0.25, 0.25, 0))
        assert viewer.pick_at(int(projected[0]), int(projected[1])) == "triangle"
        image = viewer.capture_image()
        assert image.size == viewer.viewport_size
        assert len(np.unique(np.asarray(image).reshape(-1, 4), axis=0)) > 1
    finally:
        viewer.destroy()
        first.close()
        second.close()
        qapp.processEvents()


@pytest.mark.parametrize("backend",["software","gpu"])
@pytest.mark.parametrize("fps,interval",[(0.5,2000),(2,500),(8,125)])
def test_animation_fractional_rate_schedules_actual_qt_timer(qapp,backend,fps,interval):
    if backend=="gpu" and os.environ.get("ANY3DVIEW_RUN_QT_GPU_TESTS")!="1":
        pytest.skip("real GPU acceptance is explicitly opt-in")
    viewer=create_viewer(None,backend,host=QtViewerHostAdapter())
    try:
        viewer.add_mesh_arrays(triangle(),tags="triangle",cull_backface=False)
        viewer.show();qapp.processEvents()
        viewer.begin_animation_cache();viewer.capture_animation_frame()
        viewer.play_animation(fps=fps)
        timer=viewer._animation_after_id
        assert timer.isActive() and timer.interval()==interval
        assert viewer.animation_frames==1
        for invalid in (0,-1,float("nan"),float("inf"),float("-inf")):
            with pytest.raises(ValueError,match="positive and finite"):
                viewer.play_animation(fps=invalid)
            assert viewer._animation_after_id is timer and timer.isActive()
        for invalid in (1e-310,1e-8):
            with pytest.raises(ValueError,match="timer interval range"):
                viewer.play_animation(fps=invalid)
            assert viewer._animation_after_id is timer and timer.isActive()
        viewer.stop_animation();assert not timer.isActive()
    finally:viewer.destroy();qapp.processEvents()

@pytest.mark.skipif(os.environ.get("ANY3DVIEW_RUN_QT_GPU_TESTS")!="1",reason="real GPU acceptance is explicitly opt-in")
def test_real_qt_gpu(qapp):
    viewer=create_viewer(None,"gpu",host=QtViewerHostAdapter())
    try:
        viewer.add_mesh_arrays(triangle(),tags="triangle",cull_backface=False)
        viewer.show();viewer.set_top_view();viewer.fit_to_scene();qapp.processEvents()
        assert viewer.surface.error is None
        projected=viewer.project_point((0.25,0.25,0))
        assert viewer.pick_at(int(projected[0]),int(projected[1]))=="triangle"
        assert len(np.unique(np.asarray(viewer.capture_image()).reshape(-1,4),axis=0))>1
    finally:viewer.destroy();qapp.processEvents()


@pytest.mark.skipif(os.environ.get("ANY3DVIEW_RUN_QT_GPU_TESTS")!="1",reason="real GPU acceptance is explicitly opt-in")
def test_qt_gpu_windows_have_independent_contexts(qapp):
    first=create_viewer(None,"gpu",host=QtViewerHostAdapter())
    second=create_viewer(None,"gpu",host=QtViewerHostAdapter())
    try:
        assert first._renderer.ctx is not second._renderer.ctx
        for viewer in (first,second):
            viewer.show();viewer.add_mesh_arrays(triangle(),tags="triangle",cull_backface=False)
            viewer.set_top_view();viewer.fit_to_scene()
        first.set_highlight(["triangle"]);qapp.processEvents()
        assert first.surface.error is None and second.surface.error is None
        first.destroy();qapp.processEvents()
        second.clear();second.add_mesh_arrays(triangle(),cull_backface=False);qapp.processEvents()
        assert second.surface.error is None
        assert len(np.unique(np.asarray(second.capture_image()).reshape(-1,4),axis=0))>1
    finally:first.destroy();second.destroy();qapp.processEvents()


@pytest.mark.skipif(os.environ.get("ANY3DVIEW_RUN_QT_GPU_TESTS")!="1",reason="real GPU acceptance is explicitly opt-in")
def test_qt_runtime_gpu_failure_preserves_scene_on_auto_fallback(qapp):
    viewer=create_viewer(None,"auto",host=QtViewerHostAdapter())
    try:
        assert viewer.backend_name=="gpu"
        viewer.show();handle=viewer.add_mesh_arrays(triangle(),tags="triangle",cull_backface=False)
        viewer.set_top_view();viewer.fit_to_scene();qapp.processEvents()
        viewer.surface.error=RuntimeError("injected context loss")
        viewer._render_failed();qapp.processEvents()
        assert viewer.backend_name=="software"
        assert viewer._entries[id(handle)]["handle"] is handle
        assert "injected context loss" in " ".join(viewer.backend_diagnostics)
        assert len(np.unique(np.asarray(viewer.capture_image()).reshape(-1,4),axis=0))>1
    finally:viewer.destroy();qapp.processEvents()


@pytest.mark.skipif(os.environ.get("ANY3DVIEW_RUN_QT_GPU_TESTS")!="1",reason="real GPU acceptance is explicitly opt-in")
def test_qt_gpu_hud_composes_into_window_without_alpha_artifacts(qapp):
    window=QMainWindow()
    viewer=create_viewer(window,"gpu",host=QtViewerHostAdapter())
    window.setCentralWidget(viewer);window.resize(800,600);window.show()
    try:
        viewer.set_thickness_legend([0,.5,1],title="Readable legend",unit="m")
        qapp.processEvents()
        image=window.grab().toImage()
        ratio=image.devicePixelRatio()
        color=image.pixelColor(int((window.width()-20)*ratio),int(20*ratio))
        assert min(color.red(),color.green(),color.blue())>230
        assert color.alpha()==255
    finally:viewer.destroy();window.close();qapp.processEvents()
