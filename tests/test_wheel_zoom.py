"""Wheel zoom is a fixed world step per notch and never stalls at the orbit target."""
import math
import os

import numpy as np
import pytest

from any3dview.core import Camera3D, Point3D


def _gap(a, b):
    return math.dist(a.to_tuple(), b.to_tuple())


def test_dolly_moves_a_constant_step_and_pushes_target_through():
    camera = Camera3D()
    camera.set_target(Point3D(0.0, 0.0, 0.0))
    camera.set_orbit(distance=10.0)
    forward = camera.basis()[2]
    positions = [camera.position]
    for _ in range(15):  # 15 m of travel: passes the target at 10 m
        camera.dolly(1.0, min_distance=1.0)
        positions.append(camera.position)
    steps = [_gap(a, b) for a, b in zip(positions, positions[1:])]
    assert steps == pytest.approx([1.0] * 15, abs=1e-9)
    # The camera went straight through the old target along the view direction.
    travelled = positions[-1] - positions[0]
    assert travelled.dot(forward) == pytest.approx(15.0, abs=1e-9)
    assert camera.distance == pytest.approx(1.0)
    # Zooming back out is the same step and leaves the target where it is.
    target = camera.target
    camera.dolly(-1.0, min_distance=1.0)
    assert camera.distance == pytest.approx(2.0)
    assert _gap(camera.target, target) == pytest.approx(0.0, abs=1e-12)


def test_dolly_ignores_zero_and_non_finite_steps():
    camera = Camera3D()
    before = camera.position.to_tuple()
    for amount in (0.0, math.nan, math.inf):
        camera.dolly(amount)
    assert camera.position.to_tuple() == before


qt_only = pytest.mark.skipif(os.environ.get("ANY3DVIEW_RUN_QT_TESTS") != "1",
                             reason="set ANY3DVIEW_RUN_QT_TESTS=1")


class _Wheel:
    def __init__(self, notches):
        self._y = int(round(120 * notches))

    def angleDelta(self):
        class _D:
            def __init__(self, y):
                self._y = y

            def y(self):
                return self._y
        return _D(self._y)

    def accept(self):
        pass


TRI = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
FACES = np.array([[0, 1, 2]], dtype=np.uint32)


@pytest.fixture
def qt_viewer():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from any3dview import create_viewer
    from any3dview.qt import QtViewerHostAdapter

    app = QApplication.instance() or QApplication([])
    viewer = create_viewer(None, "software", host=QtViewerHostAdapter())
    viewer.show(); app.processEvents()
    try:
        yield viewer, app
    finally:
        viewer.destroy()


def _notch_moves(viewer, notches, count):
    positions = [viewer.camera.position]
    for _ in range(count):
        viewer.event_widget.wheelEvent(_Wheel(notches))
        positions.append(viewer.camera.position)
    return [_gap(a, b) for a, b in zip(positions, positions[1:])]


@qt_only
def test_qt_wheel_zoom_into_one_of_two_far_apart_objects(qt_viewer):
    from any3dview import MeshArrays
    viewer, app = qt_viewer
    viewer.add_mesh_arrays(MeshArrays(TRI, FACES), cull_backface=False)
    viewer.add_mesh_arrays(MeshArrays(TRI + [1000., 0., 0.], FACES), cull_backface=False)
    viewer.set_top_view(); viewer.fit_to_scene(); app.processEvents()
    step = 0.05 * math.dist((0., 0., 0.), (1001., 1., 0.))
    notches = int(viewer.camera.distance / step) + 5  # enough to pass the orbit centre
    # Every notch moves the same distance, also right at the orbit centre.
    assert _notch_moves(viewer, 1, notches) == pytest.approx([step] * notches, rel=1e-9)
    assert viewer.camera.distance == pytest.approx(step)
    # Scrolling back out is the same step.
    assert _notch_moves(viewer, -1, 1) == pytest.approx([step], rel=1e-9)


@qt_only
def test_qt_wheel_step_follows_the_latest_fit_and_never_scans_geometry(qt_viewer):
    from any3dview import MeshArrays
    viewer, app = qt_viewer
    big = viewer.add_mesh_arrays(MeshArrays(TRI * 1000.0, FACES), cull_backface=False)
    viewer.set_top_view(); viewer.fit_to_scene(); app.processEvents()

    def no_scan(*_args, **_kwargs):
        raise AssertionError("wheel zoom must not scan scene geometry")

    viewer._scene_bounds = no_scan
    assert _notch_moves(viewer, 1, 2) == pytest.approx([0.05 * 1000.0 * math.sqrt(2)] * 2, rel=1e-9)
    del viewer._scene_bounds
    # Replace the scene by a small one and fit again within the same burst.
    big.remove()
    viewer.add_mesh_arrays(MeshArrays(TRI * 10.0, FACES), cull_backface=False)
    viewer.fit_to_scene()
    viewer._scene_bounds = no_scan
    assert _notch_moves(viewer, 1, 2) == pytest.approx([0.05 * 10.0 * math.sqrt(2)] * 2, rel=1e-9)
    # Half notches (high-resolution wheels) move half a step.
    assert _notch_moves(viewer, -0.5, 1) == pytest.approx([0.025 * 10.0 * math.sqrt(2)], rel=1e-9)


@qt_only
def test_qt_wheel_step_before_any_fit_is_fixed_at_first_use(qt_viewer):
    viewer, _app = qt_viewer
    viewer.camera.set_orbit(distance=20.0)
    assert _notch_moves(viewer, 1, 12) == pytest.approx([2.0] * 12, rel=1e-9)


def test_tk_wheel_delta_keeps_fractional_and_multiple_notches():
    from any3dview.core import tk_wheel_notches

    assert tk_wheel_notches(120, "win32") == 1.0
    assert tk_wheel_notches(-240, "win32") == -2.0
    assert tk_wheel_notches(15, "win32") == pytest.approx(0.125)
    assert tk_wheel_notches(3, "darwin") == 3.0
    for bad in (0, None, "x", math.nan):
        assert tk_wheel_notches(bad, "win32") == 0.0
