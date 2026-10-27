import os
import unittest
import numpy as np
from any3dview.arrays import MeshArrays
from any3dview.core import Point3D

@unittest.skipUnless(os.environ.get('ANY3DVIEW_TEST_WEBGPU')=='1','opt-in hardware WebGPU check')
class PrimitiveContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        from any3dview.qt_webgpu import QtWebGPUViewer
        self.viewer=QtWebGPUViewer(width=320,height=240);self.viewer.surface.resize(320,240);self.viewer.set_top_view()
    def tearDown(self):self.viewer.destroy();self.app.processEvents()
    def image(self):return np.asarray(self.viewer.capture_image())
    def test_stipple_pixels_and_masked_picks(self):
        mesh=MeshArrays([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],[[0,1,2],[0,2,3]])
        self.viewer.add_mesh_arrays(mesh,color='red',stipple='gray50',tags='stipple',cull_backface=False)
        self.viewer.fit_to_scene(redraw=False);image=self.image()
        center=self.viewer.project_point((0,0,0));x,y=round(center[0]),round(center[1])
        square=image[y-8:y+8,x-8:x+8,:3]
        red=np.all(square==[255,0,0],axis=2)
        self.assertEqual(np.count_nonzero(red),128)
        for dy,dx in ((0,0),(0,1),(1,0),(1,1)):
            colored=bool(np.array_equal(image[y+dy,x+dx,:3],[255,0,0]))
            self.assertEqual(bool(self.viewer.query_point(x+dx,y+dy,radius=0)),colored)
    def test_wide_hit_depth_overlay_point_outline_and_pixel_ratio(self):
        plane=MeshArrays([[-1,-1,.2],[1,-1,.2],[1,1,.2],[-1,1,.2]],[[0,1,2],[0,2,3]])
        self.viewer.add_mesh_arrays(plane,color='green',cull_backface=False)
        line=self.viewer.add_mesh_arrays(MeshArrays([[-1,0,0],[1,0,0]],np.empty((0,3),np.uint32),lines=[[0,1]]),line_color='red',line_width=25,tags='wide')
        self.viewer.fit_to_scene(redraw=False);image=self.image()
        x,y,_=self.viewer.project_point((0,0,0));np.testing.assert_array_equal(image[round(y),round(x),:3],[0,255,0])
        self.viewer.set_occlude_lines(False);image=self.image()
        np.testing.assert_array_equal(image[round(y),round(x),:3],[255,0,0])
        hits=self.viewer.query_point(round(x),round(y)+10,radius=0)
        self.assertIn('wide',[hit.key for hit in hits])
        self.viewer.add_markers([Point3D(0,.6,.3)],colors=['blue'],size=[21],outline=['black'],tags='point')
        image=self.image();x,y,_=self.viewer.project_point((0,.6,.3))
        np.testing.assert_array_equal(image[round(y),round(x),:3],[0,0,255])
        self.assertTrue(np.any(np.all(image[round(y)-12:round(y)+13,round(x)-12:round(x)+13,:3]==0,axis=2)))
    def test_point_only_fit_and_deformed_clip_bounds(self):
        mesh=MeshArrays([[10,0,0],[20,2,0]],np.empty((0,3),np.uint32),point_indices=[0,1],displacements=[[1,0,0],[1,0,0]])
        handle=self.viewer.add_mesh_arrays(mesh,point_color='blue',point_size=11,tags='markers')
        handle.set_deformation_scale(2)
        np.testing.assert_array_equal(self.viewer._scene_bounds()[0],[12,0,0])
        np.testing.assert_array_equal(self.viewer._scene_bounds()[1],[22,2,0])
        self.viewer.fit_to_scene(redraw=False)
        self.assertAlmostEqual(self.viewer.camera.target.x,17)
        image=self.image();self.assertGreater(np.count_nonzero(np.all(image[:,:,:3]==[0,0,255],axis=2)),20)
        self.viewer.set_section_plane((1,0,0),15)
        np.testing.assert_array_equal(self.viewer._scene_bounds()[0],[22,2,0])

    def test_chunk_replacement_resizes_and_fits_and_clears(self):
        first=MeshArrays([[-1,-1,0],[0,-1,0],[-.5,1,0]],[[0,1,2]])
        handle=self.viewer.add_mesh_arrays(first,color='red',cull_backface=False)
        handle.add_chunk('other',first)
        self.image()
        bigger=MeshArrays([[10,-1,0],[12,-1,0],[12,1,0],[10,1,0]],[[0,1,2],[0,2,3]])
        handle.replace_chunk('other',bigger)
        self.assertEqual(self.viewer._scene_bounds()[1][0],12)
        self.viewer.fit_to_scene(redraw=False);self.image()
        handle.remove_chunk('other');self.image()
        self.assertEqual(len(self.viewer._webgpu.fields),1)

if __name__=='__main__':unittest.main()
