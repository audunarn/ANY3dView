import numpy as np
import pytest

from any3dview.demo import build_demo_mesh


def test_demo_mesh_exercises_retained_result_fields() -> None:
    mesh = build_demo_mesh(4)

    assert mesh.positions.shape == (25, 3)
    assert mesh.triangles.shape == (32, 3)
    assert mesh.lines is not None and mesh.lines.shape == (40, 2)
    assert mesh.point_indices is not None and mesh.point_indices.shape == (4,)
    assert mesh.triangle_to_element is not None
    assert mesh.element_count == 16
    assert mesh.displacements is not None and np.any(mesh.displacements[:, 2] > 0.0)
    assert mesh.element_scalars is not None and np.isfinite(mesh.element_scalars).all()


def test_demo_mesh_rejects_degenerate_grid() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        build_demo_mesh(1)
