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


@qt_only
def test_qt_wheel_zoom_into_one_of_two_far_apart_objects():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from any3dview import MeshArrays, create_viewer
    from any3dview.qt import QtViewerHostAdapter

    app = QApplication.instance() or QApplication([])
    viewer = create_viewer(None, "software", host=QtViewerHostAdapter())
    try:
        viewer.show(); app.processEvents()
        tri = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        viewer.add_mesh_arrays(MeshArrays(tri, faces), cull_backface=False)
        viewer.add_mesh_arrays(MeshArrays(tri + [1000., 0., 0.], faces), cull_backface=False)
        viewer.set_top_view(); viewer.fit_to_scene(); app.processEvents()
        surface = viewer.event_widget
        step = 0.05 * math.dist((0., 0., 0.), (1001., 1., 0.))
        start_distance = viewer.camera.distance
        positions = [viewer.camera.position]
        notches = int(start_distance / step) + 5  # enough to pass the orbit centre
        for _ in range(notches):
            surface.wheelEvent(_Wheel(1))
            positions.append(viewer.camera.position)
        moves = [_gap(a, b) for a, b in zip(positions, positions[1:])]
        # Every notch moves the same distance, also right at the orbit centre.
        assert moves == pytest.approx([step] * notches, rel=1e-9)
        assert viewer.camera.distance == pytest.approx(step)
        # Scrolling back out is the same step.
        before = viewer.camera.position
        surface.wheelEvent(_Wheel(-1))
        assert _gap(before, viewer.camera.position) == pytest.approx(step, rel=1e-9)
    finally:
        viewer.destroy()
