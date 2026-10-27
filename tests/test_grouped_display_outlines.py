"""Opt-in display grouping removes tessellation edges without changing picking."""
import numpy as np
import pytest
from any3dview import Point3D
from any3dview.retained_viewer import RetainedViewer
from any3dview.qt import _surface_outline_loops


def draw(groups=None,legacy=False,smooth=None,colors='#ffffff',back_colors=None):
    viewer=RetainedViewer.__new__(RetainedViewer)
    results=[]
    viewer.add_mesh_arrays=lambda mesh,**options:results.append((mesh,options))
    polygons=[np.array(((0,0,0),(1,0,0),(1,1,0),(0,1,0)),dtype=float),
              np.array(((1,0,0),(2,0,0),(2,1,0),(1,1,0)),dtype=float)]
    if legacy:polygons=[[Point3D(*p) for p in polygon] for polygon in polygons]
    viewer.add_faces(polygons,outline='#222222',outline_groups=groups,smooth_angle=smooth,colors=colors,back_colors=back_colors)
    return results


@pytest.mark.parametrize('legacy',[False,True])
@pytest.mark.parametrize('smooth',[None,35])
def test_shared_display_group_has_only_real_outer_edges(legacy,smooth):
    mesh,options=draw(['face','face'],legacy,smooth)[0]
    assert len(mesh.lines)==6
    loops=list(_surface_outline_loops(mesh,np.arange(mesh.triangle_count),options['_outline_groups'],options['_outline_vertex_ids']))
    assert len(loops)==1 and len(loops[0])==6
    default,_=draw(None,legacy,smooth)[0]
    assert len(default.lines)==8 # true separate mesh polygons retain their boundary
    separate,_=draw(['first','second'],legacy,smooth)[0]
    assert len(separate.lines)==8


def test_group_arity_is_validated_before_submission():
    with pytest.raises(ValueError,match='one label'):draw(['face'])
    with pytest.raises(TypeError,match='hashable'):draw([[],[]])


def test_color_batch_slices_keep_original_group_labels():
    results=draw(['first','second'],colors=['red','blue'],back_colors=['white','black'])
    assert len(results)==2
    assert all(len(options['_outline_groups'])==2 for mesh,options in results)
