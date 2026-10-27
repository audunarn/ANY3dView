"""Focused optional WebGPU development checks, run directly or with pytest."""
import os
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from any3dview.arrays import MeshArrays
from any3dview.core import Camera3D, Point3D
from any3dview.retained import MeshHandle
from any3dview.webgpu import camera_matrix


class MatrixTests(unittest.TestCase):
    def test_camera_matches_shared_projection(self):
        camera=Camera3D()
        for point in (Point3D(0,0,0), Point3D(.3,.2,.1)):
            expected=camera.project_point(point, 300, 200)
            clip=camera_matrix(camera,300,200) @ np.array((*point.to_tuple(),1))
            projected=(clip[:2]/clip[3]+1)*[150,-100]+[0,200]
            np.testing.assert_allclose(projected, expected, atol=1e-10)
            self.assertTrue(0 < clip[2]/clip[3] < 1)


class OwnershipTests(unittest.TestCase):
    def test_failed_constructor_destroys_device(self):
        from any3dview.webgpu import WebGPURenderer
        class Device:
            destroyed=False
            def create_shader_module(self, **kwargs):raise RuntimeError("shader failure")
            def destroy(self):self.destroyed=True
        device=Device()
        adapter=SimpleNamespace(info={"backend_type":"Vulkan","adapter_type":"DiscreteGPU"},request_device_sync=lambda:device)
        fake=SimpleNamespace(gpu=SimpleNamespace(enumerate_adapters_sync=lambda:[adapter]))
        with patch.dict('sys.modules', wgpu=fake), patch('sys.platform', 'win32'):
            with self.assertRaisesRegex(RuntimeError,'shader failure'):WebGPURenderer()
        self.assertTrue(device.destroyed)

    def test_failed_prepare_and_attempt_all_cleanup(self):
        from any3dview.webgpu import WebGPURenderer
        renderer=object.__new__(WebGPURenderer)
        buffers=[]
        class Buffer:
            destroyed=False
            def destroy(self):self.destroyed=True
        class Device:
            destroyed=False
            def create_buffer_with_data(self,**kwargs):
                buffer=Buffer();buffers.append(buffer);return buffer
            def create_buffer(self,**kwargs):raise RuntimeError('uniform failure')
            def destroy(self):self.destroyed=True
        renderer.device=Device();renderer.cache={};renderer.fields={};renderer.closed=False
        renderer.geometry_uploads=renderer.field_uploads=renderer.palette_updates=renderer.index_uploads=0
        renderer.pipelines={};renderer.target=renderer.depth=None
        renderer.wgpu=SimpleNamespace(BufferUsage=SimpleNamespace(VERTEX=1,INDEX=2,UNIFORM=4,COPY_DST=8,STORAGE=16))
        mesh=MeshArrays([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]])
        with self.assertRaisesRegex(RuntimeError,'uniform failure'):
            renderer._prepare(MeshHandle(mesh),{},dict(color='red',line_color='blue',cull_backface=False), 'orange')
        self.assertGreater(len(buffers),0)
        self.assertTrue(all(buffer.destroyed for buffer in buffers))
        class Bad:
            def destroy(self):raise RuntimeError('injected destroy')
        renderer.target=Bad();renderer.depth=Buffer()
        depth=renderer.depth
        with self.assertRaisesRegex(RuntimeError,'injected destroy'):renderer.close()
        self.assertTrue(depth.destroyed)
        self.assertTrue(renderer.device.destroyed)
        renderer.close()



@unittest.skipUnless(os.environ.get("ANY3DVIEW_TEST_WEBGPU") == "1", "opt-in hardware WebGPU check")
class QtAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])

    def test_dispatch_refuses_explicit_disabled_and_unknown_api(self):
        from any3dview.qt import QtViewerHostAdapter
        from any3dview.errors import GPUUnavailableError
        with patch.dict(os.environ, ANY3DVIEW_GPU_API='webgpu', ANY3DVIEW_DISABLE_GPU='1'):
            with self.assertRaises(GPUUnavailableError):QtViewerHostAdapter().create_viewer(None,backend='auto')
        with patch.dict(os.environ, ANY3DVIEW_GPU_API='typo'):
            with self.assertRaises(ValueError):QtViewerHostAdapter().create_viewer(None,backend='auto')

    def test_nontruncated_semantic_mask_identity(self):
        from any3dview.qt_webgpu import QtWebGPUViewer
        viewer=QtWebGPUViewer()
        mesh=MeshArrays([[0,0,0],[1,0,0],[0,1,0]], np.tile([[0,1,2]], (1500,1)))
        try:
            handle=viewer.add_mesh_arrays(mesh)
            appearance=viewer._entries[id(handle)]['appearance']
            first=np.arange(1500);second=first.copy();second[750]=99999
            self.assertEqual(repr(first),repr(second))
            draws1=viewer._webgpu._prepare(handle,{'selected_elements':first},appearance,'orange')
            uploads=viewer._webgpu.index_uploads
            draws2=viewer._webgpu._prepare(handle,{'selected_elements':second},appearance,'orange')
            self.assertEqual(len(draws1),1)
            self.assertEqual(len(draws2),2)
            self.assertEqual(viewer._webgpu.index_uploads,uploads+2)
        finally:viewer.destroy()

    def test_surviving_handle_refuses_after_viewer_close_without_mutation(self):
        from any3dview.qt_webgpu import QtWebGPUViewer
        viewer=QtWebGPUViewer()
        mesh=MeshArrays([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]])
        handle=viewer.add_mesh_arrays(mesh)
        before_mesh=handle.mesh
        before_transform=handle.transform.copy()
        before_visible=handle.visible
        before_generations=handle.generations
        viewer.destroy()
        changed_transform=np.eye(4);changed_transform[0,3]=7
        for action in (lambda:handle.update_positions(np.asarray(mesh.positions)+1),
                       lambda:handle.set_transform(changed_transform),
                       lambda:handle.set_visible(False),
                       lambda:handle.set_selected_elements([0]),
                       lambda:handle.remove()):
            with self.assertRaisesRegex(RuntimeError,'closed WebGPU viewer'):action()
            self.assertIs(handle.mesh,before_mesh)
            np.testing.assert_array_equal(handle.transform,before_transform)
            self.assertEqual(handle.visible,before_visible)
            self.assertEqual(handle.generations,before_generations)
            self.assertFalse(handle.removed)
        self.app.processEvents()

    def test_refusal_retained_updates_and_close(self):
        from any3dview.qt_webgpu import QtWebGPUViewer
        viewer=QtWebGPUViewer(width=160,height=120)
        mesh=MeshArrays([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]])
        try:
            handle=viewer.add_mesh_arrays(mesh,color='#ff0000',cull_backface=False,tags='one')
            self.assertIsInstance(handle,MeshHandle)
            self.assertIs(viewer._entries[id(handle)]['handle'],handle)
            viewer.fit_to_scene(redraw=False)
            image=np.asarray(viewer.capture_image())
            self.assertGreater(np.count_nonzero((image[:,:,0]>200)&(image[:,:,1]<30)),20)
            baseline=image.copy()
            shift=np.array([1e9, -1e9, 1e9])
            handle.update_positions(np.asarray(mesh.positions)+shift)
            viewer.camera.target=viewer.camera.target+Point3D(*shift)
            viewer.camera.position=viewer.camera.position+Point3D(*shift)
            shifted=np.asarray(viewer.capture_image())
            np.testing.assert_array_equal(shifted,baseline)
            handle.update_positions(mesh.positions)
            viewer.camera.target=viewer.camera.target-Point3D(*shift)
            viewer.camera.position=viewer.camera.position-Point3D(*shift)
            before_state=viewer.export_view_state()
            viewer.apply_view_state(before_state,redraw=False)
            # Shared Camera3D roundtrip recomputes spherical coordinates;
            # compare the camera numerically, not dataclass float identity.
            np.testing.assert_allclose(viewer.camera.position.to_tuple(),before_state.camera_position.to_tuple(),rtol=0,atol=1e-12)
            self.assertEqual(viewer.export_view_state().light.key(),before_state.light.key())
            matrix=np.eye(4);matrix[:3,3]=shift
            handle.set_transform(matrix)
            viewer.camera.target=viewer.camera.target+Point3D(*shift)
            viewer.camera.position=viewer.camera.position+Point3D(*shift)
            np.testing.assert_array_equal(np.asarray(viewer.capture_image()),baseline)
            handle.set_transform(np.eye(4))
            viewer.camera.target=viewer.camera.target-Point3D(*shift)
            viewer.camera.position=viewer.camera.position-Point3D(*shift)
            initial=handle.mesh
            for action in (lambda:handle.update_node_scalars([1,2]),
                           lambda:handle.add_chunk('x',mesh,owners='bad'),
                           lambda:viewer.set_mesh_opacity(handle,1.5),
                           lambda:handle.update_appearance(scalar_range=(1,0)),
                           lambda:viewer.add_mesh_arrays(mesh,opacity=1.5),
                           lambda:viewer.add_mesh_arrays(mesh,unsupported_option=True)):
                with self.assertRaises((NotImplementedError,ValueError,TypeError)):action()
                self.assertIs(handle.mesh,initial)
                self.assertEqual(len(viewer._entries),1)
            old=viewer._webgpu.geometry_uploads
            handle.update_positions(np.asarray(mesh.positions)+[.1,0,0])
            viewer.capture_image()
            self.assertGreater(viewer._webgpu.geometry_uploads,old)
            handle.set_visible(False)
            image=np.asarray(viewer.capture_image())
            self.assertTrue((image[:,:,:3]==255).all())
            handle.remove()
            self.assertEqual(len(viewer._entries),0)
        finally:
            viewer.destroy();viewer.destroy()
        self.assertTrue(viewer._webgpu.closed)
        self.app.processEvents()

if __name__=='__main__':
    unittest.main()
