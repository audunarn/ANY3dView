import numpy as np
import pytest
from test_retained import mesh
from any3dview import MeshHandle


def test_combined_candidate_validated_before_reservation_or_generation():
    calls=[]
    def validate(candidate):
        assert candidate.node_scalars is None
        if candidate.element_scalars is not None:raise NotImplementedError('backend')
    handle=MeshHandle(mesh(),validate_mesh=validate)
    before=handle.mesh;generation=handle.generations
    with pytest.raises(NotImplementedError):handle.update_scalar_fields(element_scalars=[2.],before_publish=calls.append)
    assert handle.mesh is before and handle.generations==generation and not calls


def test_reservation_and_redraw_failure_roll_back_mesh_and_generation():
    handle=MeshHandle(mesh());before=handle.mesh;generation=handle.generations
    def fail(*_):raise ValueError('reserve/redraw')
    with pytest.raises(ValueError):handle.update_scalar_fields(node_scalars=[1.,2.,3.],before_publish=fail)
    assert handle.mesh is before and handle.generations==generation
    handle._callback=fail
    with pytest.raises(ValueError):handle.update_scalar_fields(node_scalars=[1.,2.,3.])
    assert handle.mesh is before and handle.generations.scalar==generation.scalar+2


def test_switch_is_one_scalar_upload_and_geometry_deformation_stay_identical():
    changes=[];handle=MeshHandle(mesh(),on_change=lambda h,k:changes.append(k))
    handle.update_displacements(np.ones((3,3)));handle.set_deformation_scale(2)
    before=handle.mesh;generation=handle.generations;changes.clear()
    handle.update_scalar_fields(node_scalars=[1.,2.,3.]);handle.update_scalar_fields(element_scalars=[4.])
    assert changes==['scalar','scalar'] and handle.mesh.node_scalars is None
    assert handle.generations.scalar==generation.scalar+2
    assert handle.mesh.positions is before.positions and handle.mesh.triangles is before.triangles and handle.mesh.displacements is before.displacements
    assert handle.deformation_scale==2
