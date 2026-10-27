"""GPU regression: transparent surface fill must not erase its outline."""
import os
import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get('ANY3DVIEW_TEST_WEBGPU')!='1',reason='opt-in hardware')
@pytest.mark.parametrize('opacity',[0.,.4])
def test_surface_outline_stays_visible_without_changing_explicit_line_opacity(opacity):
    from PySide6.QtWidgets import QApplication
    from any3dview.arrays import MeshArrays
    from any3dview.qt_webgpu import QtWebGPUViewer

    app=QApplication.instance() or QApplication([])
    viewer=QtWebGPUViewer(width=320,height=240,shading=False,bg='white')
    try:
        viewer.surface.resize(320,240);viewer.set_top_view()
        positions=np.array([[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]])
        mesh=MeshArrays(positions,np.array([[0,1,2],[0,2,3]],dtype=np.uint32),
                        lines=np.array([[0,1],[1,2],[2,3],[3,0]],dtype=np.uint32))
        viewer.add_mesh_arrays(mesh,color='#808080',line_color='#ff0000',line_width=3,
                               opacity=opacity,_mesh_outline=True)
        viewer.fit_to_scene(redraw=False)
        image=np.asarray(viewer.capture_image())[:,:,:3]
        red=(image[:,:,0]>230)&(image[:,:,1]<40)&(image[:,:,2]<40)
        assert np.count_nonzero(red)>100
        if opacity==0:
            x,y,_=viewer.project_point((0,0,0))
            np.testing.assert_array_equal(image[round(y),round(x)],[255,255,255])

        # An ordinary line remains governed by its own opacity.
        viewer.clear()
        line=MeshArrays([[-1.,0.,0.],[1.,0.,0.]],np.empty((0,3),dtype=np.uint32),
                        lines=np.array([[0,1]],dtype=np.uint32))
        viewer.add_mesh_arrays(line,color='#ff0000',line_color='#ff0000',line_width=3,opacity=0.)
        viewer.fit_to_scene(redraw=False)
        image=np.asarray(viewer.capture_image())[:,:,:3]
        red=(image[:,:,0]>230)&(image[:,:,1]<40)&(image[:,:,2]<40)
        assert not np.any(red)
    finally:
        viewer.destroy();app.processEvents()
