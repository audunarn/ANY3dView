"""Independent regressions for the reviewed performance boundaries; no GUI."""
from types import SimpleNamespace
import numpy as np
import pytest
from any3dview import MeshArrays
from any3dview.retained_viewer import RetainedViewer
from any3dview.scheduler import ViewerScheduler
from any3dview.qt import QtViewer


class Scene(RetainedViewer):
    stop_animation=QtViewer.stop_animation
    def __init__(self):
        self.initialize_state()
        self._host=SimpleNamespace(make_current=lambda:None)
        self._renderer=SimpleNamespace(add_mesh=lambda *a,**k:None,remove_mesh=lambda *a:None,pick_dirty=True)
    def redraw(self): pass
    def _apply_highlight_masks(self): pass
    def after_cancel(self, token): pass


def mesh(shift=0., value=1., normal=1.):
    return MeshArrays(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])+shift,
                      np.array([[0,1,2]],dtype=np.uint32),
                      element_scalars=np.array([value]),
                      normals=np.tile([0.,0.,normal],(3,1)))


def test_animation_changed_chunk_data_reaches_actual_display_handle():
    view=Scene(); handle=view.add_mesh_arrays(mesh())
    handle.add_chunk('part',mesh())
    view.begin_animation_cache(); view.capture_animation_frame()
    handle.replace_chunk('part',mesh(shift=.25,value=7.))
    view.capture_animation_frame()
    view._show_animation_frame(0); view._show_animation_frame(1)
    actual=dict(view._animation_handles[0].chunks)['part']
    np.testing.assert_array_equal(actual.positions,mesh(shift=.25).positions)
    np.testing.assert_array_equal(actual.element_scalars,[7.])
    view.stop_animation()
    assert not view._animation_frame_active
    assert handle in [entry['handle'] for entry in view._display_entries().values()]


def test_animation_changed_normals_reach_actual_display_handle():
    view=Scene(); view.add_mesh_arrays(mesh())
    view.begin_animation_cache(); view.capture_animation_frame()
    view.capture_animation_frame()
    view._animation_cache[1]['meshes'][0]['mesh']=mesh(normal=-1.)
    view._show_animation_frame(0); view._show_animation_frame(1)
    np.testing.assert_array_equal(view._animation_handles[0].mesh.normals,mesh(normal=-1.).normals)


def test_scheduler_error_remains_fail_fast_and_pending_work_can_resume():
    scheduler=ViewerScheduler(); seen=[]; wakes=[]
    scheduler.set_notifier(lambda:wakes.append(1))
    def fail(): raise ValueError('original failure')
    scheduler.submit(fail); scheduler.submit(seen.append,'later')
    with pytest.raises(ValueError,match='original failure'): scheduler.drain()
    assert seen==[], 'Later callbacks must not execute before the failure is observed'
    assert scheduler.pending()
    assert scheduler.drain()==1
    assert seen==['later']


def test_compatible_animation_reuses_handle_and_updates_values():
    view=Scene();handle=view.add_mesh_arrays(mesh())
    view.begin_animation_cache();view.capture_animation_frame()
    handle.update_element_scalars([8.]);handle.update_displacements(np.ones((3,3))*.1)
    view.capture_animation_frame();view._show_animation_frame(0)
    previous=view._animation_handles[0];view._show_animation_frame(1)
    assert view._animation_handles[0] is previous
    np.testing.assert_array_equal(previous.mesh.element_scalars,[8.])
    np.testing.assert_array_equal(previous.mesh.displacements,np.ones((3,3))*.1)


def test_selection_key_reuses_scalar_but_invalidates_geometry():
    view=Scene();handle=view.add_mesh_arrays(mesh())
    before=view._selection_key((100,100))
    handle.update_element_scalars([8.]);handle.update_appearance(scalar_range=(0.,10.))
    assert view._selection_key((100,100))==before
    handle.update_positions(handle.mesh.positions+.1)
    assert view._selection_key((100,100))!=before


def test_gpu_hover_latest_payload_does_not_restart_interval():
    from any3dview import qt
    calls=[];active=[False]
    def start(interval):calls.append(interval);active[0]=True
    timer=SimpleNamespace(start=start,isActive=lambda:active[0],stop=lambda:active.__setitem__(0,False))
    hits=[];surface=SimpleNamespace(underMouse=lambda:True)
    owner=SimpleNamespace(_closed=False,surface=surface,_hover_timer=timer,_pending_hover=None,
                          _drag=False,_selection_dragging=False,_hover_select=lambda e:hits.append(e.x))
    for i in range(100):qt.QtViewer._queue_hover(owner,SimpleNamespace(x=i,y=10),interval=16,restart=False)
    assert calls==[16]
    qt.QtViewer._deliver_hover(owner);assert hits==[99]
    qt.QtViewer._queue_hover(owner,SimpleNamespace(x=101,y=10),interval=16,restart=False)
    qt.QtViewer._cancel_hover(owner);qt.QtViewer._deliver_hover(owner)
    assert hits==[99]


def test_software_preparation_reuses_camera_changes_invalidates_and_evicts():
    from any3dview import qt
    view=Scene();view._renderer=qt._SoftwareResources();view._software_prep={}
    for name in ['_software_prep_signature','_software_chunk_prep','_prune_software_prep']:
        setattr(view,name,getattr(qt.QtViewer,name).__get__(view))
    handle=view.add_mesh_arrays(mesh());entry=view._entries[id(handle)]
    first=view._software_chunk_prep(entry,handle,None,handle.mesh,None)
    view.camera.orbit(.2,.1)
    assert view._software_chunk_prep(entry,handle,None,handle.mesh,None) is first
    handle.update_positions(handle.mesh.positions+.1)
    second=view._software_chunk_prep(entry,handle,None,handle.mesh,None)
    assert second is not first
    np.testing.assert_allclose(second.world,first.world+.1)
    handle.remove();view._prune_software_prep()
    assert not view._software_prep
