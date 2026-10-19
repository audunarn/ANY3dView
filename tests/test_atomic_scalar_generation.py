import numpy as np
import pytest
from test_retained import mesh
from any3dview import MeshHandle


def test_synchronous_consumer_failure_never_reuses_uploaded_generation():
    uploaded=[];state={'generation':None,'fail':True}
    def render(handle,kind):
        if state['generation']!=handle.generations.scalar:
            uploaded.append((handle.generations.scalar,handle.mesh.node_scalars.copy()));state['generation']=handle.generations.scalar
        if state['fail']:state['fail']=False;raise ValueError('synchronous redraw failed after upload')
    handle=MeshHandle(mesh(),on_change=render);before=handle.mesh;generation=handle.generations.scalar
    with pytest.raises(ValueError):handle.update_scalar_fields(node_scalars=[1.,2.,3.])
    assert handle.mesh is before and handle.generations.scalar==generation+2
    assert len(uploaded)==1 and state['generation']==generation+1
    handle.update_scalar_fields(node_scalars=[4.,5.,6.])
    assert len(uploaded)==2 and uploaded[-1][0]==generation+3
    np.testing.assert_array_equal(uploaded[-1][1],[4.,5.,6.])


def test_prepublication_refusals_still_leave_generation_and_mesh_untouched():
    handle=MeshHandle(mesh());before=handle.mesh;generations=handle.generations
    def reserve(_candidate):raise MemoryError('reservation')
    with pytest.raises(MemoryError):handle.update_scalar_fields(node_scalars=[1.,2.,3.],before_publish=reserve)
    assert handle.mesh is before and handle.generations==generations
    with pytest.raises(ValueError):handle.update_scalar_fields(node_scalars=[1.])
    assert handle.mesh is before and handle.generations==generations
