"""Small real Windows host integration, explicitly enabled by the owned check."""
import json
import os
from pathlib import Path
import time
import sys
import any3dview
import numpy as np
import pytest
from any3dview import MeshArrays

if os.environ.get("ANY3DVIEW_RUN_BEAM_HOST_TESTS")!="1":
    pytest.skip("set ANY3DVIEW_RUN_BEAM_HOST_TESTS=1",allow_module_level=True)


def mesh(**fields):
    values=dict(positions=np.array([[-1.,-.6,0.],[1.,-.6,0.],[-1.,.6,0.],[1.,.6,0.]],dtype=np.float32),triangles=np.empty((0,3),dtype=np.uint32),lines=np.array([[0,1],[2,3]],dtype=np.uint32),line_to_element=[1,0],element_ids=[901,305])
    values.update(fields);return MeshArrays(**values)


@pytest.mark.parametrize("toolkit",["tk","qt"])
def test_real_host_line_pixels_picking_and_scalar_only_uploads(toolkit):
    if toolkit=="tk":
        import tkinter as tk
        from any3dview.gpu import Any3DView
        root=tk.Tk();root.geometry("640x480+20+20")
        viewer=Any3DView(root,width=640,height=480);viewer.pack(fill="both",expand=True)
        pump=root.update
        close=root.destroy
    else:
        from PySide6.QtWidgets import QApplication
        from any3dview import create_viewer
        from any3dview.qt import QtViewerHostAdapter
        application=QApplication.instance() or QApplication([])
        viewer=create_viewer(None,"gpu",host=QtViewerHostAdapter());viewer.resize(640,480);viewer.show()
        pump=application.processEvents
        close=lambda:(viewer.destroy(),application.processEvents())
    try:
        if toolkit=="qt":pump()
        handle=viewer.add_mesh_arrays(mesh(element_scalars=[1.,0.]),line_width=8,lit=False,cull_backface=False,invalid_color="#ff00ff")
        viewer.set_top_view();viewer.fit_to_scene();pump()
        image=np.asarray(viewer.capture_image())
        def pixel(point):
            x,y,_=viewer.project_point(point)
            return image[int(round(y)),int(round(x)),:3].astype(int)
        def hits(point):
            x,y,_=viewer.project_point(point)
            return viewer.query_point(round(x),round(y),radius=1)
        lower=pixel((0.,-.6,0.));upper=pixel((0.,.6,0.))
        assert lower[2]>lower[0] and upper[0]>upper[2]
        lower_hits=hits((0.,-.6,0.));upper_hits=hits((0.,.6,0.))
        assert lower_hits[0].identity==305 and upper_hits[0].identity==901
        assert lower_hits[0].key.endswith(":line:0") and upper_hits[0].key.endswith(":line:1")
        renderer=viewer._renderer;chunk=renderer._groups[id(handle)].chunks[0][1]
        resources={name:(getattr(chunk,name).glo,getattr(chunk,name).read()) for name in ("vertex_buffer","line_buffer","line_endpoint_buffer","index_buffer")}
        topology=handle.generations.topology;positions=handle.generations.position;geometry_uploads=renderer.geometry_uploads
        scalar_uploads=renderer.scalar_buffer_updates;pick_fbo=renderer._pick_fbo.glo
        assert not renderer.pick_dirty
        handle.update_element_scalars([0.,1.]);pump();image=np.asarray(viewer.capture_image())
        assert pixel((0.,-.6,0.))[0]>pixel((0.,-.6,0.))[2]
        assert renderer.geometry_uploads==geometry_uploads and renderer.scalar_buffer_updates==scalar_uploads+3
        assert handle.generations.topology==topology and handle.generations.position==positions
        assert not renderer.pick_dirty and renderer._pick_fbo.glo==pick_fbo
        for name,value in resources.items():assert (getattr(chunk,name).glo,getattr(chunk,name).read())==value
        handle.update_element_scalars([np.nan,1.]);pump();image=np.asarray(viewer.capture_image())
        invalid=pixel((0.,.6,0.));assert invalid[0]>240 and invalid[2]>240 and invalid[1]<20
        handle.set_active_elements([False,True]);pump();image=np.asarray(viewer.capture_image())
        assert not hits((0.,.6,0.)) and hits((0.,-.6,0.))[0].identity==305
        handle.set_active_elements(None);handle.update_element_scalars(None);handle.update_node_scalars([0.,1.,0.,1.]);pump();image=np.asarray(viewer.capture_image())
        left=pixel((-.85,-.6,0.));right=pixel((.85,-.6,0.));assert left[2]>left[0] and right[0]>right[2]
        evidence=Path("C:/ANYworkspaceWork/anypost-20261012/beam-viewer-evidence")
        (evidence/f"host-{toolkit}-runtime.json").write_text(json.dumps({"python":sys.version,"executable":sys.executable,"viewer_module":any3dview.__file__,"backend":viewer.backend_name,"GL":{k:renderer.ctx.info.get(k) for k in ("GL_VENDOR","GL_RENDERER","GL_VERSION")},"initial_lower":lower.tolist(),"initial_upper":upper.tolist(),"node_left":left.tolist(),"node_right":right.tolist(),"geometry_uploads":renderer.geometry_uploads,"scalar_buffer_updates":renderer.scalar_buffer_updates,"lines":2,"frames":renderer.frame_count},indent=2))
    finally:close()


def test_real_qt_software_mapped_field_refusal_before_attach_and_update():
    from PySide6.QtWidgets import QApplication
    from any3dview import create_viewer
    from any3dview.qt import QtViewerHostAdapter
    app=QApplication.instance() or QApplication([]);viewer=create_viewer(None,"software",host=QtViewerHostAdapter())
    try:
        with pytest.raises(NotImplementedError,match="line"):viewer.add_mesh_arrays(mesh(element_scalars=[1.,2.]))
        assert not viewer._entries
        handle=viewer.add_mesh_arrays(mesh());original=handle.mesh;generations=handle.generations
        with pytest.raises(NotImplementedError,match="line"):handle.update_element_scalars([1.,2.])
        assert handle.mesh is original and handle.generations==generations
    finally:viewer.destroy();app.processEvents()
