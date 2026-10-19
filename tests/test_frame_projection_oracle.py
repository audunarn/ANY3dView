"""Independent parity oracle for the frame-local projection optimization."""
from types import SimpleNamespace
import numpy as np
import pytest
from any3dview.core import Camera3D, Point3D
from any3dview.retained_viewer import RetainedViewer, _FrameProjection


@pytest.mark.parametrize('size', [(800, 600), (320, 900), (1920, 1080)])
@pytest.mark.parametrize('azimuth,elevation', [(0., 0.), (1.2, -.4)])
def test_frame_projection_matches_public_point_projection_after_camera_changes(size, azimuth, elevation):
    camera = Camera3D()
    viewer = SimpleNamespace(camera=camera, viewport_size=size,
                             _host=SimpleNamespace(framebuffer_size=lambda: size))
    for distance in (7., 19.):
        camera.set_orbit(azimuth=azimuth, elevation=elevation, distance=distance)
        frame = _FrameProjection(viewer)
        coordinates = np.array([[-2., 1., .5], [3., -1., 2.], [0., 0., 0.]])
        forward = np.array(camera.basis()[2].to_tuple())
        origin = np.array(camera.position.to_tuple())
        coordinates = np.vstack((coordinates, origin-forward, origin+forward*camera.far*2))
        actual, visible = RetainedViewer._project_world_points(viewer, coordinates, frame)
        for i, point in enumerate(coordinates):
            expected = RetainedViewer.project_point(viewer, Point3D(*point))
            assert bool(visible[i]) == (expected is not None)
            scalar = RetainedViewer._project_point_with_frame(viewer, point, frame)
            if expected is not None:
                np.testing.assert_allclose(actual[i], expected, rtol=1e-12, atol=1e-10)
                np.testing.assert_allclose(scalar, expected, rtol=1e-12, atol=1e-10)
            else:
                assert scalar is None


def test_frame_projection_retains_strict_depth_boundaries_and_empty_arrays():
    camera = Camera3D(); camera.set_orbit(azimuth=0., elevation=0., distance=10.)
    camera.near=1.; camera.far=20.
    viewer=SimpleNamespace(camera=camera,viewport_size=(800,600),
                           _host=SimpleNamespace(framebuffer_size=lambda:(800,600)))
    frame=_FrameProjection(viewer)
    points=np.array([[9.,0.,0.],[-10.,0.,0.],[8.,0.,0.]])
    _, visible=RetainedViewer._project_world_points(viewer,points,frame)
    assert visible.tolist()==[False,False,True]
    projected,visible=RetainedViewer._project_world_points(viewer,[],frame)
    assert projected.shape==(0,3) and visible.shape==(0,)


def test_batched_outline_projection_preserves_all_semantic_edges(monkeypatch):
    from any3dview import MeshArrays
    from any3dview.retained_viewer import _surface_outline_loops
    camera = Camera3D(); camera.set_orbit(azimuth=.6, elevation=.4, distance=10.)
    viewer = SimpleNamespace(camera=camera, viewport_size=(800,600),
        _host=SimpleNamespace(framebuffer_size=lambda:(1600,1200)),
        _section_plane=None, _outline_groups=lambda *args:None)
    mesh = MeshArrays(np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]]),
        np.array([[0,1,2],[0,2,3]],dtype=np.uint32),
        triangle_to_element=np.array([0,0],dtype=np.uint32))
    frame = _FrameProjection(viewer)
    expected=[]
    for loop in _surface_outline_loops(mesh,[0,1],None):
        edges=[loop] if len(loop)==2 else zip(loop,(*loop[1:],loop[0]))
        for edge in edges:
            expected.append(RetainedViewer._project_outline_segment(viewer,mesh.positions[list(edge)],frame))
    def unexpected(*args):
        raise AssertionError('Unclipped outlines must use batched projection')
    viewer._project_outline_segment=unexpected
    actual=list(RetainedViewer._surface_outline_segments(viewer,{},None,mesh,[0,1],mesh.positions,frame))
    assert len(actual)==len(expected)==4
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-10)
