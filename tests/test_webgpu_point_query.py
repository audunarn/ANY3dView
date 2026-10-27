"""Canonical parity and bounded candidate construction for accelerated point picks."""
import os
import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
from any3dview.arrays import MeshArrays
from any3dview.retained_viewer import RetainedViewer
from any3dview.selection import SelectionFilter,PickBinding
from any3dview.ownership import PackedOwnerTable
from any3dview.semantic import VisibilityState,SemanticRef

@unittest.skipUnless(os.environ.get('ANY3DVIEW_TEST_WEBGPU')=='1','opt-in hardware WebGPU check')
class PointQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        from any3dview.qt_webgpu import QtWebGPUViewer
        self.viewer=QtWebGPUViewer(width=320,height=240);self.viewer.surface.resize(320,240);self.viewer.set_top_view()
    def tearDown(self):self.viewer.destroy();self.app.processEvents()
    def assert_same(self,x,y,filter=None):
        expected=RetainedViewer.query_point(self.viewer,x,y,selection_filter=filter,radius=3)
        actual=self.viewer.query_point(x,y,selection_filter=filter,radius=3)
        self.assertEqual([(h.key,h.primitive,h.item,h.visible) for h in actual],[(h.key,h.primitive,h.item,h.visible) for h in expected])
        np.testing.assert_allclose([[h.depth,h.screen_distance] for h in actual],[[h.depth,h.screen_distance] for h in expected],rtol=1e-12,atol=1e-10)
    def test_canonical_ids_chunks_clip_visibility_active_deformation_and_filters(self):
        mesh=MeshArrays([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],[[0,1,2],[0,2,3]],lines=[[0,2]],point_indices=[0,1],
            triangle_to_element=[1,0],element_ids=[71,90],active_elements=[True,False],displacements=[[.1,0,.2]]*4)
        owners=PackedOwnerTable.from_owners(triangles=[PickBinding.one('face:a'),PickBinding.one('face:b')],lines=[PickBinding.one('line:a')],points=[PickBinding.one('point:a'),PickBinding.one('point:b')])
        handle=self.viewer.add_mesh_arrays(mesh,owners=owners,cull_backface=False,line_width=17,point_size=13)
        handle.add_chunk('part',replace(mesh,positions=np.asarray(mesh.positions)+[0,0,-.3]),owners=owners)
        # Mixed winding must contribute only canonically admitted faces to dense IDs.
        self.viewer.add_mesh_arrays(MeshArrays(np.asarray(mesh.positions)+[0,0,-.5],
            [[0,1,2],[0,2,1],[0,2,3],[0,3,2]]),tags='culled',cull_backface=True)
        self.viewer.fit_to_scene(redraw=False)
        for state in range(5):
            if state==1:handle.set_active_elements(None);handle.set_deformation_scale(1.3)
            if state==2:self.viewer.set_section_plane((1,.2,.1),-.1)
            if state==3:
                self.viewer.clear_section_plane();self.viewer.set_visibility_state(VisibilityState(hidden=[SemanticRef('application','application','face:a')]))
            if state==4:self.viewer.set_visibility_state(VisibilityState());self.viewer.set_view(-65,30)
            for x,y in ((160,120),(120,100),(210,150),(0,0),(300,200)):
                for flt in (None,SelectionFilter(key_prefixes=('line:',)),SelectionFilter(key_prefixes=('missing',))):self.assert_same(x,y,flt)
        # A replacement with new dimensions must invalidate every geometry view.
        handle.replace_chunk('part',MeshArrays([[-1,-1,0],[1,-1,0],[0,1,0]],[[0,1,2]]),owners=PackedOwnerTable.from_owners(triangles=[PickBinding.one('new:chunk')]))
        self.assert_same(160,120)
    def test_near_far_straddlers_and_backface_ordinals_match(self):
        self.viewer.camera.near=.5;self.viewer.camera.far=8
        self.viewer.set_front_view()
        mesh=MeshArrays([[-1,-1,0],[1,-1,0],[0,1,0],[-.4,-20,0],[.4,20,0],[0,0,1]],[[0,1,2],[3,4,5]],lines=[[3,4]],point_indices=[0,5])
        self.viewer.add_mesh_arrays(mesh,tags='straddler',cull_backface=False,line_width=11)
        self.viewer.camera.set_target(type(self.viewer.camera.target)(0,0,0));self.viewer.camera.set_orbit(distance=4)
        for plane in (None,((1,0,0),0),((0,1,0),0)):
            if plane:self.viewer.set_section_plane(*plane)
            else:self.viewer.clear_section_plane()
            for point in ((160,120),(140,120),(180,100),(10,10)):self.assert_same(*point)
    def test_large_scene_does_not_construct_full_projection_or_tile_index(self):
        n=161
        positions=np.array([(x,y,0.) for y in np.linspace(-10,10,n) for x in np.linspace(-10,10,n)])
        a=np.array([j*n+i for j in range(n-1) for i in range(n-1)],dtype=np.uint32)
        triangles=np.column_stack((np.repeat(a,2),np.column_stack((a+1,a+n+1)).ravel(),np.column_stack((a+n+1,a+n)).ravel()))
        self.viewer.add_mesh_arrays(MeshArrays(positions,triangles),tags='grid',cull_backface=False)
        self.viewer.fit_to_scene(redraw=False)
        for delta in (0,.05,.1):
            self.viewer.camera.orbit(delta,delta)
            with patch.object(self.viewer,'_projected_selection_index',side_effect=AssertionError('full scene projection invoked')):
                self.assertTrue(self.viewer.query_point(160,120))
            query=self.viewer._point_query
            self.assertLess(query.candidate_primitives,len(triangles)//50)
            self.assertLess(query.projected_primitives,len(triangles)//50)
        builds=self.viewer._point_query.camera_builds
        self.viewer.query_point(161,120)
        self.assertEqual(self.viewer._point_query.camera_builds,builds)
        self.viewer.destroy()
        self.assertIsNone(self.viewer._point_query)

if __name__=='__main__':unittest.main()
