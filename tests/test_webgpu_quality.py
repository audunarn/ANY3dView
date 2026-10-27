"""Display quality, scientific colors, and geometry lifetime contracts."""
import os
import numpy as np
import pytest
from any3dview.webgpu.quality import feature_line_mask


def test_feature_edges_keep_boundary_crease_nonmanifold_and_split_normals():
    points=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,0],[1,1,0]])
    triangles=np.array([[0,1,2],[4,5,3]])
    lines=np.array([[0,1],[1,2],[2,0],[2,3],[3,0]])
    before=points.copy()
    assert feature_line_mask(points,triangles,lines).tolist()==[True,True,False,True,True]
    points[3,2]=2
    assert feature_line_mask(points,triangles,lines).all()
    assert feature_line_mask(before,np.vstack((triangles,triangles[0])),lines)[2]
    np.testing.assert_array_equal(triangles,[[0,1,2],[4,5,3]])


@pytest.mark.skipif(os.environ.get('ANY3DVIEW_TEST_WEBGPU')!='1',reason='opt-in hardware')
def test_msaa_gradient_transparency_resize_and_quality_reuse(tmp_path):
    from PySide6.QtWidgets import QApplication
    from any3dview.qt_webgpu import QtWebGPUViewer
    app=QApplication.instance() or QApplication([])
    viewer=QtWebGPUViewer(width=320,height=240,shading=False)
    try:
        viewer.surface.resize(320,240);viewer.set_top_view()
        viewer.add_faces([[[-1,-1,0],[1,-.7,0],[.7,1,0],[-1,1,0]]],colors=['#804020'],outline='#334455',lit=False)
        viewer.fit_to_scene(redraw=False)
        one=np.asarray(viewer.capture_image());uploads=viewer._webgpu.geometry_uploads
        viewer.set_render_quality(samples=4,gradient=True,feature_edges=True)
        four=np.asarray(viewer.capture_image())
        assert viewer._webgpu.depth.sample_count==4 and len(viewer._webgpu.multisampled)==4
        assert not np.array_equal(four[5,5,:3],four[-5,5,:3])
        p=viewer.project_point((0,0,0));np.testing.assert_array_equal(four[round(p[1]),round(p[0]),:3],[128,64,32])
        assert viewer._webgpu.geometry_uploads==uploads
        handle=next(iter(viewer._entries.values()))['handle']
        handle.update_element_scalars(np.zeros(handle.mesh.element_count))
        handle.update_appearance(scalar_range=(0,1),colormap=((0,'#0099ee'),(1,'#0099ee')))
        scalar=np.asarray(viewer.capture_image())
        np.testing.assert_array_equal(scalar[round(p[1]),round(p[0]),:3],[0,153,238])
        viewer.add_faces([[[-.7,-.7,.2],[.7,-.7,.2],[.7,.7,.2],[-.7,.7,.2]]],colors=['#00ff00'],opacity=.4)
        transparent=np.asarray(viewer.capture_image())
        assert transparent[round(p[1]),round(p[0]),1]>64
        viewer.set_section_plane((1,0,0),0)
        cut=np.asarray(viewer.capture_image())
        left=viewer.project_point((-.5,0,0));right=viewer.project_point((.5,0,0))
        assert not np.array_equal(cut[round(left[1]),round(left[0]),:3],cut[round(right[1]),round(right[0]),:3])
        viewer.clear_section_plane()
        viewer.surface.resize(400,280);viewer.capture_image()
        viewer.set_render_quality(samples=1,gradient=False,feature_edges=False)
        viewer.capture_image();assert viewer._webgpu.depth.sample_count==1 and not viewer._webgpu.multisampled
        with pytest.raises(ValueError):viewer.set_render_quality(samples=2)
        from PIL import Image
        Image.fromarray(one).save(tmp_path/'quality-before.png');Image.fromarray(four).save(tmp_path/'quality-after.png')
    finally:
        viewer.destroy();app.processEvents()
