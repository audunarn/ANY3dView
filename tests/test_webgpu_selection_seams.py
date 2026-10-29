"""Selection outlines must not expose split-normal storage seams."""
import os
import numpy as np
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('ANY3DVIEW_TEST_WEBGPU')!='1',reason='opt-in hardware')

def test_selected_split_vertices_keep_boundary_without_fill_diagonal():
    from PySide6.QtWidgets import QApplication
    from any3dview.arrays import MeshArrays
    from any3dview.qt_webgpu import QtWebGPUViewer
    app=QApplication.instance() or QApplication([])
    viewer=QtWebGPUViewer(width=400,height=300,shading=False,bg='white')
    try:
        viewer.surface.resize(400,300);viewer.set_top_view()
        positions=np.array([[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],
                            [-1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]])
        mesh=MeshArrays(positions,[[0,1,2],[3,4,5]])
        viewer.add_mesh_arrays(mesh,color='#7099b4',cull_backface=False,tags='plate')
        viewer.set_highlight(['plate']);viewer.fit_to_scene(redraw=False)
        pixels=np.asarray(viewer.capture_image().convert('RGB'),dtype=float)
        x,y,_=viewer.project_point((0.,0.,0.));x,y=round(x),round(y)
        interior=pixels[y,x+12]
        assert np.max(np.abs(pixels[y-2:y+3,x-2:x+3]-interior))<8
        x,y,_=viewer.project_point((0.,-1.,0.));x,y=round(x),round(y)
        assert np.max(np.abs(pixels[y-2:y+3,x-2:x+3]-interior))>20
    finally:
        viewer.destroy();app.processEvents()
