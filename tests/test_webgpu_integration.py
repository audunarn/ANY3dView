import os
import unittest
import numpy as np
from any3dview.arrays import MeshArrays
from any3dview.core import Point3D

@unittest.skipUnless(os.environ.get('ANY3DVIEW_TEST_WEBGPU')=='1','opt-in hardware WebGPU check')
class FullViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        from any3dview.qt_webgpu import QtWebGPUViewer
        self.viewer=QtWebGPUViewer(width=320,height=240)
        self.viewer.surface.resize(320,240);self.viewer.set_top_view()
    def tearDown(self):self.viewer.destroy();self.app.processEvents()
    def image(self):return np.asarray(self.viewer.capture_image())
    def test_weighted_transparency_input_order_and_opaque_depth(self):
        def add(z,color,opacity):
            return self.viewer.add_mesh_arrays(MeshArrays([[-1,-1,z],[1,-1,z],[1,1,z],[-1,1,z]],[[0,1,2],[0,2,3]]),color=color,opacity=opacity,cull_backface=False)
        add(0,'#ff0000',.5);add(.1,'#0000ff',.5)
        self.viewer.fit_to_scene(redraw=False)
        first=self.image();center=first[first.shape[0]//2,first.shape[1]//2,:3]
        self.assertGreater(center[0],90);self.assertGreater(center[2],90)
        self.viewer._entries=dict(reversed(tuple(self.viewer._entries.items())))
        np.testing.assert_array_equal(self.image(),first)
        add(.2,'#00ff00',1)
        image=self.image();np.testing.assert_array_equal(image[image.shape[0]//2,image.shape[1]//2,:3],[0,255,0])
    def test_depth_only_prepass_is_order_independent_and_clears_foreground(self):
        def add(z,color,**options):
            return self.viewer.add_mesh_arrays(MeshArrays([[-1,-1,z],[1,-1,z],[1,1,z],[-1,1,z]],[[0,1,2],[0,2,3]]),color=color,cull_backface=False,**options)
        add(0,'blue');add(.2,'red',depth_only=True,pickable=False)
        self.viewer.fit_to_scene(redraw=False)
        image=self.image();np.testing.assert_array_equal(image[image.shape[0]//2,image.shape[1]//2,:3],[255,255,255])
        self.viewer._entries=dict(reversed(tuple(self.viewer._entries.items())))
        np.testing.assert_array_equal(self.image(),image)

    def test_wide_lines_points_hud_and_legacy_shapes(self):
        self.viewer.add_line(Point3D(-1,0,0),Point3D(1,0,0),color='red',width=9,tags='line')
        self.viewer.add_markers([Point3D(0,.6,0)],colors=['blue'],size=[15],outline=['black'],tags='point')
        self.viewer.fit_to_scene(redraw=False)
        image=self.image()
        x,y,_=self.viewer.project_point((0,0,0));column=image[:,round(x),:3]
        self.assertGreaterEqual(np.count_nonzero(np.all(column==[255,0,0],axis=1)),8)
        x,y,_=self.viewer.project_point((0,.6,0));np.testing.assert_array_equal(image[round(y),round(x),:3],[0,0,255])
        self.assertTrue(self.viewer.query_point(round(x),round(y)))
        before=self.image()
        self.viewer.add_text(Point3D(0,0,0),'GPU geometry + shared HUD',color='black')
        self.viewer.set_axis_indicator(True)
        self.assertFalse(np.array_equal(self.image(),before))
        self.viewer.add_box(1,1,1,color='green',outline='black')
        self.image()
    def test_chunk_replace_and_animation_shared_facade(self):
        base=MeshArrays([[-1,-1,0],[0,-1,0],[-.5,1,0]],[[0,1,2]])
        handle=self.viewer.add_mesh_arrays(base,color='red',cull_backface=False)
        other=MeshArrays([[0,-1,0],[1,-1,0],[.5,1,0]],[[0,1,2]])
        handle.add_chunk('right',other)
        self.viewer.fit_to_scene(redraw=False)
        image=self.image();self.assertGreater(np.count_nonzero(image[:,:,1]<20),30)
        handle.remove_chunk('right');self.image()
        self.viewer.begin_animation_cache();self.viewer.capture_animation_frame()
        handle.update_positions(np.asarray(base.positions)+[.2,0,0]);self.viewer.capture_animation_frame()
        self.assertEqual(self.viewer.animation_frames,2)
        self.viewer.play_animation(fps=10);self.viewer.stop_animation();self.image()

if __name__=='__main__':unittest.main()
