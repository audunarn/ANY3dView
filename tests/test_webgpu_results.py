"""Optional real-GPU result-field contract checks; no native Qt windows."""
import os
import unittest
from dataclasses import replace
import numpy as np
from any3dview.arrays import MeshArrays
from any3dview.appearance import validated_appearance,palette_data
from any3dview.webgpu.renderer import scalar_payload

PALETTE=((0.,'#000000'),(.5,'#ff0000'),(1.,'#ffffff'))

class PayloadTests(unittest.TestCase):
    def test_precedence_range_nan_and_large_offset(self):
        mesh=MeshArrays([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]],node_scalars=[1e12,1e12+.5,1e12+1],element_scalars=[17.])
        mode,bounds,nodes,elements=scalar_payload(mesh,validated_appearance())
        self.assertEqual((mode,bounds),(2,(17.,17.)))
        mode,bounds,nodes,elements=scalar_payload(replace(mesh,element_scalars=None),validated_appearance((1e12,1e12+1)))
        np.testing.assert_array_equal(nodes[:,0],[0,.5,1])
        self.assertEqual(mode,1)
        mesh=replace(mesh,node_scalars=[np.nan,0.,1.],element_scalars=None)
        _,_,nodes,_=scalar_payload(mesh,validated_appearance())
        np.testing.assert_array_equal(nodes[:,1],[0,1,1])


@unittest.skipUnless(os.environ.get('ANY3DVIEW_TEST_WEBGPU')=='1','opt-in hardware WebGPU check')
class ResultGPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        from any3dview.qt_webgpu import QtWebGPUViewer
        self.viewer=QtWebGPUViewer(width=240,height=200,bg='white')
        self.viewer.surface.resize(240,200)
        self.viewer.set_top_view()

    def tearDown(self):
        self.viewer.destroy();self.app.processEvents()

    def add(self,**fields):
        mesh=MeshArrays([[-1,-1,0],[1,-1,0],[0,1,0]],[[0,1,2]],**fields)
        handle=self.viewer.add_mesh_arrays(mesh,cull_backface=False,scalar_range=(0,1),colormap=PALETTE,invalid_color='#00ff00')
        self.viewer.fit_to_scene(redraw=False)
        return handle

    def image(self):return np.asarray(self.viewer.capture_image())

    def test_interpolation_precedes_palette_and_field_cache_identity(self):
        handle=self.add(node_scalars=[0,1,.5])
        before=handle.mesh.owned_copy()
        image=self.image()
        center=image[image.shape[0]//2,image.shape[1]//2,:3]
        self.assertGreater(center[0],240)
        self.assertLess(center[1],15) # precolored endpoint RGB interpolation would be grey
        renderer=self.viewer._webgpu
        position=renderer.fields[id(handle)]['buffers'][0]
        index=[draw[1] for draw in renderer.cache[id(handle)][2]]
        uploads=renderer.geometry_uploads,renderer.index_uploads
        handle.update_node_scalars([1e12,1e12+1,1e12+.5])
        handle.update_appearance(scalar_range=(1e12,1e12+1))
        np.testing.assert_array_equal(self.image(),image)
        self.assertIs(renderer.fields[id(handle)]['buffers'][0],position)
        self.assertEqual([draw[1] for draw in renderer.cache[id(handle)][2]],index)
        self.assertEqual((renderer.geometry_uploads,renderer.index_uploads),uploads)
        handle.update_appearance(colormap=((0,'#0000ff'),(1,'#0000ff')))
        self.assertTrue((self.image()[:,:,2]==255).all())
        self.assertEqual((renderer.geometry_uploads,renderer.index_uploads),uploads)
        np.testing.assert_array_equal(before.positions,handle.mesh.positions)
        np.testing.assert_array_equal(before.triangles,handle.mesh.triangles)

    def test_element_priority_clear_nan_and_active_deformation(self):
        handle=self.add(node_scalars=[0,0,0],element_scalars=[1],displacements=[[0,.5,0]]*3)
        image=self.image()
        center=image[image.shape[0]//2,image.shape[1]//2,:3]
        np.testing.assert_array_equal(center,[255,255,255])
        handle.update_element_scalars(None)
        center=self.image()[image.shape[0]//2,image.shape[1]//2,:3]
        np.testing.assert_array_equal(center,[0,0,0])
        handle.update_node_scalars([np.nan,0,1])
        center=self.image()[image.shape[0]//2,image.shape[1]//2,:3]
        np.testing.assert_array_equal(center,[0,255,0])
        handle.update_scalar_fields()
        baseline=self.image()
        renderer=self.viewer._webgpu;identity=tuple(renderer.fields[id(handle)]['buffers'])
        uploads=renderer.geometry_uploads,renderer.index_uploads,renderer.field_uploads
        handle.set_deformation_scale(1)
        moved=self.image()
        self.assertFalse(np.array_equal(moved,baseline))
        self.assertEqual(tuple(renderer.fields[id(handle)]['buffers']),identity)
        self.assertEqual((renderer.geometry_uploads,renderer.index_uploads,renderer.field_uploads),uploads)
        old_world=(-.8,-.8,0)
        projected=self.viewer.project_point(old_world)
        self.assertEqual(self.viewer.query_point(round(projected[0]),round(projected[1]),radius=0),())
        handle.set_active_elements([False])
        self.assertTrue((self.image()[:,:,:3]==255).all())
        self.assertIsNone(self.viewer._scene_bounds())
        self.assertEqual(len(self.viewer._projected_selection_index().primitives),0)
        handle.set_active_elements([True]);handle.set_deformation_scale(0)
        np.testing.assert_array_equal(self.image(),baseline)

    def test_mapped_and_unmapped_lines_active_and_element_mapping(self):
        positions=np.array([[-1,-.6,0],[1,-.6,0],[-1,.6,0],[1,.6,0]],float)
        mesh=MeshArrays(positions,np.empty((0,3),dtype=np.uint32),lines=[[0,1],[2,3]],
            line_to_element=[1,0],element_scalars=[0,1],active_elements=[True,False])
        handle=self.viewer.add_mesh_arrays(mesh,line_color='#ff0000',scalar_range=(0,1),
            colormap=((0,'#0000ff'),(1,'#00ff00')))
        self.viewer.fit_to_scene(redraw=False)
        image=self.image()
        self.assertGreater(np.count_nonzero(np.all(image[:,:,:3]==[0,0,255],axis=2)),10)
        self.assertEqual(np.count_nonzero(np.all(image[:,:,:3]==[0,255,0],axis=2)),0)
        handle.set_active_elements([False,True])
        image=self.image()
        self.assertGreater(np.count_nonzero(np.all(image[:,:,:3]==[0,255,0],axis=2)),10)
        self.assertEqual(np.count_nonzero(np.all(image[:,:,:3]==[0,0,255],axis=2)),0)
        np.testing.assert_array_equal(self.viewer._scene_bounds()[0],positions[:2].min(axis=0))
        handle.remove()
        unmapped=MeshArrays(positions,np.empty((0,3),dtype=np.uint32),lines=[[0,1],[2,3]])
        self.viewer.add_mesh_arrays(unmapped,line_color='#ff0000')
        self.assertGreater(np.count_nonzero(np.all(self.image()[:,:,:3]==[255,0,0],axis=2)),10)

    def test_unmapped_line_keeps_base_color_and_ignores_element_activity(self):
        mesh=MeshArrays([[-1,-1,0],[1,-1,0],[0,1,0],[-1,2,0],[1,2,0]],[[0,1,2]],
            lines=[[3,4]],element_scalars=[.7],active_elements=[False])
        handle=self.viewer.add_mesh_arrays(mesh,line_color='#ff0000',colormap=PALETTE)
        self.viewer.fit_to_scene(redraw=False)
        pixels=self.image()
        self.assertGreater(np.count_nonzero(np.all(pixels[:,:,:3]==[255,0,0],axis=2)),20)
        np.testing.assert_array_equal(self.viewer._scene_bounds()[0],[-1,2,0])
        np.testing.assert_array_equal(self.viewer._scene_bounds()[1],[1,2,0])

    def test_nontrivial_triangle_mapping_and_source_immutability(self):
        positions=np.array([[-2,-1,0],[-.2,-1,0],[-1,1,0],[.2,-1,0],[2,-1,0],[1,1,0]],float)
        mesh=MeshArrays(positions,[[0,1,2],[3,4,5]],triangle_to_element=[1,0],
            node_scalars=[.5]*6,element_scalars=[0,1],active_elements=[True,True])
        snapshots={name:getattr(mesh,name).tobytes() for name in ('positions','triangles','triangle_to_element','node_scalars','element_scalars','active_elements')}
        handle=self.viewer.add_mesh_arrays(mesh,cull_backface=False,scalar_range=(0,1),
            colormap=((0,'#0000ff'),(1,'#ff0000')))
        self.viewer.fit_to_scene(redraw=False)
        image=self.image()
        self.assertGreater(np.count_nonzero(np.all(image[:,:,:3]==[255,0,0],axis=2)),50)
        self.assertGreater(np.count_nonzero(np.all(image[:,:,:3]==[0,0,255],axis=2)),50)
        handle.set_active_elements([True,False])
        image=self.image()
        self.assertEqual(np.count_nonzero(np.all(image[:,:,:3]==[255,0,0],axis=2)),0)
        for name,raw in snapshots.items():self.assertEqual(getattr(mesh,name).tobytes(),raw)

    def test_bad_updates_refuse_before_mutation_and_hook(self):
        handle=self.add(node_scalars=[0,1,.5])
        mesh,generations,appearance=handle.mesh,handle.generations,handle.appearance
        hook=[]
        for action in (lambda:handle.update_node_scalars([0,1]),
                       lambda:handle.update_displacements([[0,0,0]]),
                       lambda:handle.set_active_elements([True,False]),
                       lambda:handle.update_appearance(scalar_range=(2,1)),
                       lambda:handle.update_appearance(colormap=((0,'nonsense'),))):
            with self.assertRaises(ValueError):action()
            self.assertIs(handle.mesh,mesh)
            self.assertEqual(handle.generations,generations)
            self.assertEqual(handle.appearance,appearance)
        def cancel(candidate):hook.append(candidate);raise RuntimeError('reservation refused')
        with self.assertRaisesRegex(RuntimeError,'reservation refused'):
            handle.update_scalar_fields(node_scalars=[1,2,3],before_publish=cancel)
        self.assertEqual(len(hook),1)
        self.assertIs(handle.mesh,mesh)
        self.assertEqual(handle.generations,generations)

if __name__=='__main__':unittest.main()
