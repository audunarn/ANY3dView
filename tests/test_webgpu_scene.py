"""Opt-in real-GPU opaque scene feature contracts."""
import os
import unittest
from dataclasses import replace
import numpy as np

@unittest.skipUnless(os.environ.get('ANY3DVIEW_TEST_WEBGPU')=='1','opt-in hardware WebGPU check')
class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        from any3dview.qt_webgpu import QtWebGPUViewer
        self.viewer=QtWebGPUViewer(shading=True,width=320,height=240)
        self.viewer.surface.resize(320,240);self.viewer.set_top_view()
    def tearDown(self):
        self.viewer.destroy();self.app.processEvents()
    def add(self,normals=None,**options):
        from any3dview.arrays import MeshArrays
        mesh=MeshArrays([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],[[0,1,2],[0,2,3]],normals=normals)
        handle=self.viewer.add_mesh_arrays(mesh,color='#804020',lit=True,cull_backface=False,**options)
        self.viewer.fit_to_scene(redraw=False)
        return handle
    def image(self):return np.asarray(self.viewer.capture_image())
    def pixel(self,world):
        p=self.viewer.project_point(world);return self.image()[round(p[1]),round(p[0]),:3]
    def test_light_flat_ambient_diffuse_whitening_and_unlit(self):
        handle=self.add()
        self.viewer.set_light((0,0,1),.2,.3,0)
        np.testing.assert_allclose(self.pixel((0,0,0)),[64,32,16],atol=1)
        before=self.viewer.renderer_diagnostics.copy()
        self.viewer.set_light(ambient=1.25,diffuse=0)
        np.testing.assert_allclose(self.pixel((0,0,0)),np.array([128,64,32])*.5+127.5,atol=1)
        self.viewer.set_mesh_style(handle,lit=False)
        np.testing.assert_array_equal(self.pixel((0,0,0)),[128,64,32])
        self.assertEqual(self.viewer.renderer_diagnostics['geometry_uploads'],before['geometry_uploads'])
        self.assertEqual(self.viewer.renderer_diagnostics['index_uploads'],before['index_uploads'])
    def test_smooth_inverse_transpose_and_singular_flat_fallback(self):
        normals=np.tile([1.,0.,1.],(4,1));handle=self.add(normals=normals)
        matrix=np.diag([2.,1.,.5,1.]);handle.set_transform(matrix)
        self.viewer.fit_to_scene(redraw=False)
        self.viewer.set_light(direction=(0,0,1),ambient=0,diffuse=1,specular=0)
        expected=np.array([128,64,32])*2/np.sqrt(4.25)
        np.testing.assert_allclose(self.pixel((0,0,0)),expected,atol=1)
        matrix[2,2]=0;handle.set_transform(matrix)
        np.testing.assert_array_equal(self.pixel((0,0,0)),[128,64,32])
    def test_face_colors_update_priority_and_no_geometry_upload(self):
        handle=self.add(face_colors=np.array([[255,0,0],[0,0,255]],dtype=np.uint8))
        self.viewer.set_shading(False)
        np.testing.assert_array_equal(self.pixel((.5,-.5,0)),[255,0,0])
        handle.update_element_scalars([1,1])
        np.testing.assert_array_equal(self.pixel((-.5,.5,0)),[0,0,255])
        renderer=self.viewer._webgpu
        before=renderer.geometry_uploads,renderer.index_uploads
        handle.update_face_colors([[0.,1.,0.,1.],[1.,1.,0.,1.]])
        np.testing.assert_array_equal(self.pixel((.5,-.5,0)),[0,255,0])
        self.assertEqual((renderer.geometry_uploads,renderer.index_uploads),before)
        mesh,generations=handle.mesh,handle.generations
        with self.assertRaises(ValueError):handle.update_face_colors([[0.,1.,0.,.5],[1.,1.,0.,1.]])
        self.assertIs(handle.mesh,mesh);self.assertEqual(handle.generations,generations)
    def test_zero_normals_flat_deformed_response_and_specular(self):
        handle=self.add(normals=np.zeros((4,3)))
        self.viewer.set_light(direction=(0,0,1),ambient=0,diffuse=0,specular=.5,shininess=1)
        value=self.pixel((0,0,0))
        self.assertGreater(value[0],60) # Lambert gate + Blinn specular on flat fallback
        self.viewer.set_light(ambient=.2,diffuse=.8,specular=0)
        undeformed=self.pixel((0,0,0))
        handle.update_displacements([[0,0,-1],[0,0,1],[0,0,1],[0,0,-1]])
        handle.set_deformation_scale(1)
        deformed=self.pixel((0,0,0))
        expected=np.array([128,64,32])*(.2+.8/np.sqrt(2))
        np.testing.assert_allclose(deformed,expected,atol=1)
        self.assertLess(deformed[0],undeformed[0])

    def test_backface_color_scalar_branch_and_camera_light(self):
        from any3dview.arrays import MeshArrays
        mesh=MeshArrays([[-1,-1,0],[1,-1,0],[0,1,0]],[[0,2,1]])
        handle=self.viewer.add_mesh_arrays(mesh,color='#aa0000',back_color='#0000ff',
            face_colors=['#00ff00'],lit=False,cull_backface=False)
        self.viewer.fit_to_scene(redraw=False)
        np.testing.assert_array_equal(self.pixel((0,0,0)),[0,0,255])
        handle.update_element_scalars([0])
        handle.update_appearance(colormap=((0,'#ffff00'),(1,'#ffff00')))
        np.testing.assert_array_equal(self.pixel((0,0,0)),[255,255,0]) # exact existing else-if precedence
        handle.remove()
        handle=self.add()
        self.viewer.set_light(direction=(0,0,1),ambient=0,diffuse=1,specular=0,follow_camera=True)
        np.testing.assert_allclose(self.pixel((0,0,0)),[128,64,32],atol=1)
        self.viewer.set_side_view()
        # The geometry is now edge-on; inspect uniform direction from canonical Light instead of inventing a pixel.
        self.assertGreater(abs(self.viewer._light.world_direction(self.viewer.camera.basis())[0]),.99)

    def test_invalid_policy_and_closed_setters_leave_state(self):
        self.add()
        state=self.viewer.export_view_state()
        with self.assertRaises(ValueError):self.viewer.set_light(ambient=float('nan'),diffuse=.9)
        self.assertEqual(self.viewer._light.key(),state.light.key())
        with self.assertRaises(ValueError):self.viewer.set_section_plane((0,0,0),0)
        self.assertIsNone(self.viewer._section_plane)
        with self.assertRaises(ValueError):self.viewer.apply_view_state(replace(state,background='unsupported'))
        self.assertEqual(self.viewer._light.key(),state.light.key())
        self.assertIsNone(self.viewer._section_plane)
        from threading import Thread
        failures=[]
        def wrong_thread():
            for call in (lambda:self.viewer.set_light(ambient=.9),lambda:self.viewer.set_shading(False),
                         lambda:self.viewer.set_section_plane(),self.viewer.capture_image,self.viewer.destroy):
                try:call()
                except RuntimeError as error:failures.append(str(error))
            self.viewer.submit_update(self.viewer.set_light,ambient=.7)
        worker=Thread(target=wrong_thread);worker.start();worker.join()
        self.assertEqual(len(failures),5)
        self.assertEqual(self.viewer._light.key(),state.light.key())
        self.assertIsNone(self.viewer._section_plane)
        self.assertTrue(self.viewer._shading_enabled)
        self.assertFalse(self.viewer._closed)
        self.viewer._poll_updates()
        self.assertEqual(self.viewer._light.ambient,.7)
        self.viewer.destroy()
        for call in (lambda:self.viewer.set_light(ambient=.8),lambda:self.viewer.set_shading(False),
                     lambda:self.viewer.set_section_plane(),self.viewer.clear_section_plane):
            with self.assertRaises(RuntimeError):call()

    def test_clipping_pick_fit_large_offset_and_state(self):
        handle=self.add();self.viewer.set_shading(False)
        self.image();before=self.viewer.renderer_diagnostics.copy()
        self.viewer.set_section_plane((1,0,0),0)
        np.testing.assert_array_equal(self.pixel((-.5,0,0)),[255,255,255])
        self.assertFalse(self.viewer.query_point(*[round(x) for x in self.viewer.project_point((-.5,0,0))[:2]],radius=0))
        self.assertGreaterEqual(self.viewer._scene_bounds()[0][0],0)
        shift=np.array([1e9,-1e9,1e9]);matrix=np.eye(4);matrix[:3,3]=shift;handle.set_transform(matrix)
        self.viewer.camera.target=self.viewer.camera.target+type(self.viewer.camera.target)(*shift)
        self.viewer.camera.position=self.viewer.camera.position+type(self.viewer.camera.position)(*shift)
        self.viewer.set_section_plane((1,0,0),1e9)
        np.testing.assert_array_equal(self.pixel(shift+[-.5,0,0]),[255,255,255])
        np.testing.assert_array_equal(self.pixel(shift+[.5,0,0]),[128,64,32])
        state=self.viewer.export_view_state();self.viewer.apply_view_state(state,redraw=False)
        self.assertEqual(self.viewer.export_view_state().section_plane,state.section_plane)
        self.assertEqual(self.viewer.export_view_state().light.key(),state.light.key())
        self.assertEqual(self.viewer.renderer_diagnostics['geometry_uploads'],before['geometry_uploads'])
        self.assertEqual(self.viewer.renderer_diagnostics['index_uploads'],before['index_uploads'])
        self.viewer.clear_section_plane()
        np.testing.assert_array_equal(self.pixel(shift+[-.5,0,0]),[128,64,32])

if __name__=='__main__':unittest.main()
