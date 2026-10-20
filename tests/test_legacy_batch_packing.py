"""Public legacy batch compatibility when numeric packing is accelerated."""
import numpy as np
import pytest

from any3dview import PickBinding, PickOwner, Point3D
from any3dview.retained_viewer import RetainedViewer


class Capture(RetainedViewer):
    def __init__(self):
        self.results = []

    def add_mesh_arrays(self, mesh, **options):
        self.results.append((mesh, options))


def batch(view, vertices, faces, **extra):
    view._legacy_mesh_batch(vertices, faces, color="gray", outline="black",
                            width=1, layer=5, cull_backface=False, opacity=1,
                            stipple="", tags="", lit=True, two_sided_shell=False,
                            bindings=extra.pop("bindings", None), **extra)


@pytest.mark.parametrize("kind", ["array", "tuple", "point", "mixed", "generator"])
def test_coordinate_coercion_preserves_values_and_triangle_order(kind):
    xyz = [[0., 0., 0.], [2., 0., 0.], [2., 1., 0.], [0., 1., 0.]]
    if kind == "array":
        vertices = np.asarray(xyz, dtype=np.float32)
    elif kind == "tuple":
        vertices = tuple(tuple(row) for row in xyz)
    elif kind == "point":
        vertices = [Point3D(*row) for row in xyz]
    elif kind == "mixed":
        vertices = [None, Point3D(*xyz[1]), tuple(xyz[2]), np.array(xyz[3])]
    else:
        vertices = (row for row in xyz)
    view = Capture()
    batch(view, vertices, [[0, 1, 2, 3]])
    mesh, options = view.results[0]
    np.testing.assert_array_equal(mesh.positions, xyz)
    assert mesh.positions.dtype == np.float64
    np.testing.assert_array_equal(mesh.triangles, [[0, 1, 2], [0, 2, 3]])
    np.testing.assert_array_equal(mesh.lines, [[0, 1], [0, 3], [1, 2], [2, 3]])
    np.testing.assert_array_equal(options["_outline_groups"], [0, 0])


def test_edge_dedup_keeps_exact_lexicographic_order_and_owner_rows():
    view = Capture()
    a, b = PickBinding((PickOwner("a"),)), PickBinding((PickOwner("b"),))
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = [(0, 1, 2), (2, 1, 0), (2, 2, 3), (0, 1)]
    batch(view, vertices, faces, bindings=[a, b, a, None],
          face_colors=["red", "blue", "green", "white"])
    mesh, options = view.results[0]
    expected_edges = sorted({tuple(sorted((face[i - 1], face[i])))
                             for face in faces if len(face) >= 3 for i in range(len(face))})
    np.testing.assert_array_equal(mesh.lines, expected_edges)
    np.testing.assert_array_equal(mesh.triangles, faces[:3])
    np.testing.assert_array_equal(options["_outline_groups"], [0, 1, 2])
    assert options["face_colors"] == ("red", "blue", "green")
    owners = options["owners"]
    np.testing.assert_array_equal(owners.triangle_offsets, [0, 1, 2, 3])
    np.testing.assert_array_equal(owners.triangle_indices, [0, 1, 0])
    assert owners.string_keys == ("a", "b")
    assert not owners.line_indices.size


def test_public_add_faces_keeps_numeric_and_point_inputs_equivalent():
    xyz = [[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
           [[0, 0, 1], [1, 0, 1], [0, 1, 1]]]
    numeric, objects = Capture(), Capture()
    kwargs = dict(colors=["red", "blue"], outline="black",
                  bindings=[PickOwner("a"), PickOwner("b")])
    numeric.add_faces(xyz, **kwargs)
    objects.add_faces([[Point3D(*point) for point in face] for face in xyz], **kwargs)
    assert len(numeric.results) == len(objects.results) == 1
    for name in ("positions", "triangles", "lines"):
        np.testing.assert_array_equal(getattr(numeric.results[0][0], name),
                                      getattr(objects.results[0][0], name))
    assert numeric.results[0][1]["face_colors"] == objects.results[0][1]["face_colors"]


def test_numeric_array_input_still_has_independent_retained_storage():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    view = Capture()
    batch(view, vertices, [[0, 1, 2]])
    vertices[:] = 99
    np.testing.assert_array_equal(view.results[0][0].positions,
                                  [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])


@pytest.mark.parametrize("container", ["array", "rows", "public_faces"])
def test_masked_coordinates_keep_legacy_nonfinite_refusal(container):
    vertices = np.ma.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                           mask=[[True, False, False], [False] * 3, [False] * 3])
    view = Capture()
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="finite"):
        if container == "public_faces":
            view.add_faces([vertices], outline="black")
        else:
            batch(view, vertices if container == "array" else list(vertices), [[0, 1, 2]])
    assert view.results == []


def test_public_faces_preserve_custom_scalar_float_coercion():
    class Shifted(float):
        def __float__(self):
            return super().__float__() + 3.

    view = Capture()
    view.add_faces([[[Shifted(0.), 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    assert view.results[0][0].positions[0, 0] == 3.


def test_array_subclass_preserves_legacy_row_iteration():
    class Reversed(np.ndarray):
        def __iter__(self):
            return iter(np.asarray(self)[::-1])

    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]).view(Reversed)
    view = Capture()
    batch(view, vertices, [[0, 1, 2]])
    np.testing.assert_array_equal(view.results[0][0].positions, np.asarray(vertices)[::-1])


@pytest.mark.parametrize("vertices, faces", [
    ([[0., 0., np.nan], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]]),
    ([[0., 0., 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 3]]),
    ([[0., 0.], [1, 0], [0, 1]], [[0, 1, 2]]),
])
def test_invalid_input_never_publishes_a_batch(vertices, faces):
    view = Capture()
    with pytest.raises((ValueError, TypeError)):
        batch(view, vertices, faces)
    assert view.results == []
