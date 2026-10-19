from types import SimpleNamespace
import importlib.util
from pathlib import Path
import pytest
from any3dview.core import Camera3D
spec=importlib.util.spec_from_file_location('trace_camera',Path(__file__).parents[1]/'tools/trace_qt_camera.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class Signal:
    def connect(self, callback):self.callback=callback
    def disconnect(self, callback):self.callback=None
    def emit(self):self.callback()


def viewer():
    v=SimpleNamespace(camera=Camera3D(),_drag='orbit',_mouse=(0,0),backend_name='gpu')
    v.event_widget=SimpleNamespace(frameSwapped=Signal())
    v._renderer=SimpleNamespace(render=lambda *a,**k:'drawn')
    def motion(event):
        if v._drag=='orbit':v.camera.orbit(.008*(event.x-v._mouse[0]),.008*(event.y-v._mouse[1]))
        v._mouse=(event.x,event.y)
    v._motion=motion
    v._press=lambda *a:None
    v._release=lambda *a:setattr(v,'_drag','')
    v._cancel_interaction=v._release
    return v


def test_pose_identity_and_cancelled_drag_are_distinguishable():
    v=viewer();original=v._motion;t=module.CameraPresentationTrace(v)
    v._motion(SimpleNamespace(x=20,y=10))
    assert v._renderer.render(v.camera,(800,600))=='drawn'
    v.event_widget.frameSwapped.emit()
    records=t.snapshot()['events'];m,r,s=records
    assert m['before']!=m['after'] and m['after']==r['pose']==s['last_submission']['pose']
    assert r['motion_id']==m['motion_id']==1
    v._cancel_interaction();v._motion(SimpleNamespace(x=40,y=20))
    m=t.snapshot()['events'][-1]
    assert m['mode_before']=='' and m['before']==m['after']
    t.close();t.close();assert v._motion is original


def test_trace_bounded_and_does_not_restore_over_new_owner():
    v=viewer();t=module.CameraPresentationTrace(v,capacity=2)
    for i in range(4):v._motion(SimpleNamespace(x=i,y=0))
    assert t.snapshot()['dropped_events']==2
    replacement=lambda event:None;v._motion=replacement;t.close()
    assert v._motion is replacement


def test_failed_render_propagates_without_claiming_submission():
    v=viewer()
    def fail(*args,**kw):raise RuntimeError('draw failed')
    v._renderer.render=fail;t=module.CameraPresentationTrace(v)
    with pytest.raises(RuntimeError):v._renderer.render(v.camera,(1,1))
    v.event_widget.frameSwapped.emit()
    assert t.snapshot()['events'][-1]['last_submission'] is None
    t.close()


def test_current_renderer_uploads_new_camera_frame_uniforms():
    import numpy as np
    from any3dview.gpu.renderer import ModernGLRenderer, _matrix_bytes
    class Uniform:
        def __init__(self):self.writes=[];self.value=None
        def write(self,data):self.writes.append(data)
    program={name:Uniform() for name in ('u_view_rotation','u_projection','u_viewport','u_near',
        'u_section_enabled','u_section_normal','u_section_relative_offset','u_linear',
        'u_origin_from_camera','u_deformation_scale')}
    renderer=ModernGLRenderer.__new__(ModernGLRenderer);renderer._frame_stamp={}
    chunk=SimpleNamespace(handle=SimpleNamespace(transform=np.eye(4),deformation_scale=0.),origin=np.zeros(3))
    camera=Camera3D()
    first=renderer._camera_frame(camera,(1932,1069),None)
    renderer._uniforms(program,chunk,first)
    renderer._uniforms(program,chunk,first)
    assert len(program['u_view_rotation'].writes)==1
    camera.orbit(.4,.2)
    second=renderer._camera_frame(camera,(1932,1069),None)
    renderer._uniforms(program,chunk,second)
    assert len(program['u_view_rotation'].writes)==2
    assert program['u_view_rotation'].writes[0]!=program['u_view_rotation'].writes[1]
    assert program['u_view_rotation'].writes[-1]==_matrix_bytes(second.view)
    np.testing.assert_allclose(program['u_origin_from_camera'].value,-np.array(camera.position.to_tuple()))
    camera.fov *= .8
    third=renderer._camera_frame(camera,(1932,1069),None)
    renderer._uniforms(program,chunk,third)
    assert program['u_projection'].writes[-1]==_matrix_bytes(third.projection)
    assert program['u_projection'].writes[-1]!=program['u_projection'].writes[-2]
