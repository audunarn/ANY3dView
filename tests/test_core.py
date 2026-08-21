import math

import numpy as np
import pytest

import any3dview
from any3dview import Camera3D, Light, Mesh, Point3D
from any3dview.shading import face_shade


def test_public_version_and_modules_are_available():
    assert any3dview.__version__ == "0.4.0"
    for name in any3dview.__all__:
        assert hasattr(any3dview, name), name


def test_point_camera_and_mesh_are_backend_neutral():
    point = Point3D(1, 2, 3)
    assert (point + Point3D(2, 0, -1)).to_tuple() == (3.0, 2.0, 2.0)
    assert point.cross(Point3D(0, 1, 0)).to_tuple() == (-3.0, 0.0, 1.0)

    camera = Camera3D()
    projected = camera.project_point(camera.target, 400, 300)
    assert projected == pytest.approx((200.0, 150.0))

    mesh = Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)])
    assert len(mesh.vertices) == 3
    assert mesh.bounds()[1].to_tuple() == (1.0, 1.0, 0.0)


def test_shading_stays_numpy_vectorized():
    normals = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    light = Light(direction=Point3D(0, 0, 1), ambient=0.2, diffuse=0.8, specular=0)
    values = face_shade(normals, light, light.world_direction())
    assert values.tolist() == pytest.approx([1.0, 0.2])


def test_screen_ray_hits_camera_target_plane():
    camera = Camera3D()
    point = camera.unproject_to_plane(
        200, 150, 400, 300, camera.target, Point3D(0, 0, 1)
    )
    assert point is not None
    assert math.isclose(point.z, camera.target.z, abs_tol=1.0e-8)
