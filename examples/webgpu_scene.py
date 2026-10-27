"""Experimental opaque scene controls; supplied illustrative display arrays only."""
import json
import sys
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QCheckBox,QSlider,QPushButton,QComboBox,QLineEdit
from any3dview.arrays import MeshArrays
from any3dview.qt_webgpu import QtWebGPUViewer


def main():
    app=QApplication.instance() or QApplication(sys.argv)
    window=QWidget();window.resize(1150,850)
    window.setWindowTitle('Experimental WebGPU opaque scenes — lit / normals / face colors / section')
    layout=QVBoxLayout(window)
    layout.addWidget(QLabel('Right drag orbit · Middle drag pan · Wheel zoom · Left click select · Supplied normals describe reference geometry'))
    toolbar=QHBoxLayout();layout.addLayout(toolbar)
    lit=QCheckBox('Lighting');lit.setChecked(True)
    smooth=QCheckBox('Supplied smooth normals')
    faces=QCheckBox('Face colors override scalars')
    scalar=QCheckBox('Nodal contour');scalar.setChecked(True)
    follow=QCheckBox('Camera-follow light')
    clip=QCheckBox('Section x ≥ offset')
    fit=QPushButton('Fit')
    for widget in (lit,smooth,faces,scalar,follow,clip,fit):toolbar.addWidget(widget)
    sliders=QHBoxLayout();layout.addLayout(sliders)
    sliders.addWidget(QLabel('Section offset'))
    plane=QSlider(Qt.Horizontal);plane.setRange(-100,100);plane.setValue(0);sliders.addWidget(plane)
    sliders.addWidget(QLabel('Deformation scale'))
    deformation=QSlider(Qt.Horizontal);deformation.setRange(0,200);sliders.addWidget(deformation)
    views=QComboBox();views.addItems(['Isometric','Top','Front','Side']);sliders.addWidget(views)
    edit=QLineEdit();edit.setPlaceholderText('Desktop input check');sliders.addWidget(edit)
    viewer=QtWebGPUViewer(window,shading=True,bg='#f0f4f8',interaction_profile='commercial')
    layout.addWidget(viewer,1)
    status=QLabel('Experimental GPU opaque scene; CPU semantic picking; Qt RGBA bitmap readback')
    layout.addWidget(status)
    n=11
    positions=np.array([(x,y,.25*(1-x*x)*(1-y*y)) for y in np.linspace(-1,1,n) for x in np.linspace(-1,1,n)])
    triangles=[]
    for j in range(n-1):
        for i in range(n-1):
            a=j*n+i;triangles.extend(([a,a+1,a+n+1],[a,a+n+1,a+n]))
    triangles=np.asarray(triangles,dtype=np.uint32)
    normals=np.array([(.5*x*(1-y*y),.5*y*(1-x*x),1.) for x,y,z in positions])
    displacement=np.zeros_like(positions);displacement[:,2]=.3*(1-positions[:,0]**2)*(1-positions[:,1]**2)
    node_values=positions[:,0]+1
    face_values=['#4385b8' if i%2 else '#d8a555' for i in range(len(triangles))]
    handles={}
    for has_normals in (False,True):
        for has_faces in (False,True):
            mesh=MeshArrays(positions,triangles,normals=normals if has_normals else None,
                node_scalars=node_values,displacements=displacement)
            handles[has_normals,has_faces]=viewer.add_mesh_arrays(mesh,lit=True,cull_backface=False,
                color='#4385b8',face_colors=face_values if has_faces else None,tags='illustrative-scene')
    def visibility():
        for key,handle in handles.items():handle.set_visible(key==(smooth.isChecked(),faces.isChecked()))
    def section():
        if clip.isChecked():viewer.set_section_plane((1,0,0),plane.value()/100)
        else:viewer.clear_section_plane()
    def scalars():
        for handle in handles.values():handle.update_node_scalars(node_values if scalar.isChecked() else None)
    def deform(value):
        for handle in handles.values():handle.set_deformation_scale(value/100)
    smooth.toggled.connect(visibility);faces.toggled.connect(visibility)
    lit.toggled.connect(viewer.set_shading)
    follow.toggled.connect(lambda enabled:viewer.set_light(follow_camera=enabled))
    scalar.toggled.connect(scalars);deformation.valueChanged.connect(deform)
    clip.toggled.connect(section);plane.valueChanged.connect(section)
    fit.clicked.connect(lambda:viewer.fit_to_scene())
    methods=[viewer.set_iso_view,viewer.set_top_view,viewer.set_front_view,viewer.set_side_view]
    views.currentIndexChanged.connect(lambda index:methods[index]())
    viewer.set_pick_callback(lambda pick:status.setText(f'Selected {pick.tag} — shared CPU semantic identity'))
    visibility();viewer.set_iso_view();viewer.fit_to_scene();window.show()
    print(json.dumps(viewer.diagnostic_snapshot(),indent=2),flush=True)
    app.aboutToQuit.connect(viewer.destroy)
    return app.exec()

if __name__=='__main__':raise SystemExit(main())
