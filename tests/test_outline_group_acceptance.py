"""Opt-in surface outlines must not change ordinary element wireframes."""
import numpy as np
import pytest
from any3dview.retained_viewer import _surface_outline_loops
from test_numeric_face_batches import Capture, LegacyCapture


@pytest.mark.parametrize('factory', [Capture, LegacyCapture])
@pytest.mark.parametrize('smooth', [None, 35])
def test_grouped_shell_omits_shared_strip_edge_but_default_mesh_keeps_it(factory, smooth):
    polygons = [
        np.array(((0,0,0), (1,0,1), (1,1,1), (0,1,0)), float),
        np.array(((1,0,1), (2,0,0), (2,1,0), (1,1,1)), float),
    ]
    for labels, expected in ((None, [4,4]), (['shell','shell'], [6])):
        viewer = factory()
        viewer.add_faces(polygons, smooth_angle=smooth, outline_groups=labels)
        mesh, options = viewer.outputs[0]
        loops = list(_surface_outline_loops(mesh, range(mesh.triangle_count),
            options['_outline_groups'], options['_outline_vertex_ids']))
        assert sorted(map(len, loops)) == expected
    viewer = factory()
    with pytest.raises(ValueError, match='outline_groups'):
        viewer.add_faces(polygons, outline_groups=['only one'])
    assert not viewer.outputs
