import numpy as np
import pytest

from any3dview import MeshArrays
from any3dview.benchmarks import member_lattice, plate_grid


def triangle_mesh(**fields):
    values = {
        "positions": np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        "triangles": np.asarray([[0, 1, 2]], dtype=np.uint32),
    }
    values.update(fields)
    return MeshArrays(**values)


def test_compatible_arrays_are_retained_zero_copy():
    positions = np.zeros((3, 3), dtype=np.float32)
    triangles = np.asarray([[0, 1, 2]], dtype=np.uint32)
    mesh = MeshArrays(positions, triangles)

    assert mesh.positions is positions
    assert mesh.triangles is triangles


def test_owned_copy_isolated_every_supplied_array():
    original = triangle_mesh(node_scalars=np.asarray([1.0, 2.0, np.nan], np.float32))
    copied = original.owned_copy()

    assert copied.positions is not original.positions
    assert copied.triangles is not original.triangles
    assert copied.node_scalars is not original.node_scalars
    assert np.isnan(copied.node_scalars[-1])


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"positions": [[0.0, 0.0]]}, "positions must have shape"),
        ({"positions": [[0.0, 0.0, np.inf]]}, "finite"),
        ({"triangles": [[0, 1, 3]]}, "missing position"),
        ({"triangles": [[0, 1.5, 2]]}, "integer indices"),
        ({"displacements": np.zeros((2, 3))}, "displacements must have shape"),
        ({"node_scalars": [1.0, 2.0]}, "node_scalars must have shape"),
        ({"node_scalars": [1.0, 2.0, np.inf]}, "infinite"),
        ({"element_scalars": [1.0, 2.0]}, "triangle_to_element is required"),
    ],
)
def test_invalid_mesh_arrays_fail_closed(fields, message):
    with pytest.raises(ValueError, match=message):
        triangle_mesh(**fields)


def test_element_mapping_validates_dynamic_fields():
    positions = np.zeros((4, 3), dtype=np.float64)
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    mesh = MeshArrays(
        positions,
        triangles,
        triangle_to_element=np.asarray([0, 0], dtype=np.uint32),
        element_ids=np.asarray([42], dtype=np.uint64),
        element_scalars=np.asarray([np.nan], dtype=np.float32),
        active_elements=np.asarray([True]),
    )

    assert mesh.element_count == 1
    assert mesh.element_ids.tolist() == [42]


def test_benchmark_generators_are_deterministic_and_indexed():
    first = plate_grid(3, 2)
    second = plate_grid(3, 2)
    lattice = member_lattice(3, 2)

    assert np.array_equal(first.mesh.positions, second.mesh.positions)
    assert np.array_equal(first.mesh.triangles, second.mesh.triangles)
    assert first.mesh.triangle_count == 12
    assert first.mesh.element_count == 6
    assert lattice.mesh.lines.shape == (17, 2)
