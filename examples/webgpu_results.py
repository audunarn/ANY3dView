"""Illustrative supplied fields; no solver/results recovery is performed."""
import json
import sys
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication,QWidget,QVBoxLayout,QHBoxLayout,QLabel,
    QComboBox,QCheckBox,QSlider,QPushButton,QLineEdit)
from any3dview.arrays import MeshArrays
from any3dview.qt_webgpu import QtWebGPUViewer


def main():
    app=QApplication.instance() or QApplication(sys.argv)
    window=QWidget();window.resize(1100,800)
    window.setWindowTitle('Experimental WebGPU retained fields — illustrative arrays')
    layout=QVBoxLayout(window)
    layout.addWidget(QLabel('Supplied illustrative values only · Right drag orbit · Middle drag pan · Wheel zoom · Left click select'))
    controls=QHBoxLayout();layout.addLayout(controls)
    fields=QComboBox();fields.addItems(['Nodal scalar','Element scalar','No scalar'])
    palette=QComboBox();palette.addItems(['Default palette','Black / red / white','Blue / white'])
    fixed=QCheckBox('Fixed range 0…2')
    active=QCheckBox('Only alternating elements')
    fit=QPushButton('Fit')
    for widget in (fields,palette,fixed,active,fit):controls.addWidget(widget)
    scale_row=QHBoxLayout();layout.addLayout(scale_row)
    scale_label=QLabel('Deformation display scale: 0.00')
    scale=QSlider(Qt.Horizontal);scale.setRange(0,300);scale.setValue(0)
    edit=QLineEdit();edit.setPlaceholderText('Desktop input responsiveness check')
    scale_row.addWidget(scale_label);scale_row.addWidget(scale,1);scale_row.addWidget(edit)
    viewer=QtWebGPUViewer(window,bg='#f4f6fa',interaction_profile='commercial')
    layout.addWidget(viewer,1)
    status=QLabel('GPU field interpolation + indexed depth rendering; CPU semantic picking; Qt bitmap readback')
    layout.addWidget(status)
    n=13
    positions=np.array([(x,y,0.) for y in np.linspace(-1,1,n) for x in np.linspace(-1,1,n)])
    triangles=[]
    for j in range(n-1):
        for i in range(n-1):
            a=j*n+i;triangles.extend(([a,a+1,a+n+1],[a,a+n+1,a+n]))
    triangles=np.array(triangles,dtype=np.uint32)
    node_values=positions[:,0]+1
    element_values=(positions[triangles].mean(axis=1)[:,1]+1)
    displacements=np.zeros_like(positions)
    displacements[:,2]=.4*(1-positions[:,0]**2)*(1-positions[:,1]**2)
    mesh=MeshArrays(positions,triangles,node_scalars=node_values,displacements=displacements)
    handle=viewer.add_mesh_arrays(mesh,color='#3c88bc',cull_backface=False,tags='illustrative-patch')
    def update_fields():
        index=fields.currentIndex()
        handle.update_scalar_fields(node_scalars=node_values if index==0 else None,
                                   element_scalars=element_values if index==1 else None)
    def update_palette():
        maps=[None,((0,'#000000'),(.5,'#ff0000'),(1,'#ffffff')),((0,'#174bb8'),(1,'#ffffff'))]
        handle.update_appearance(colormap=maps[palette.currentIndex()])
    def update_scale(value):
        handle.set_deformation_scale(value/100)
        scale_label.setText(f'Deformation display scale: {value/100:.2f}')
    fields.currentIndexChanged.connect(update_fields)
    palette.currentIndexChanged.connect(update_palette)
    fixed.toggled.connect(lambda checked:handle.update_appearance(scalar_range=(0.,2.) if checked else None))
    active.toggled.connect(lambda checked:handle.set_active_elements(np.arange(len(triangles))%2==0 if checked else None))
    scale.valueChanged.connect(update_scale)
    fit.clicked.connect(lambda:viewer.fit_to_scene())
    viewer.set_pick_callback(lambda pick:status.setText(f'Selected {pick.tag}; source arrays retained, GPU fields displayed'))
    viewer.set_iso_view();viewer.fit_to_scene();window.show()
    print(json.dumps(viewer.diagnostic_snapshot(),indent=2),flush=True)
    app.aboutToQuit.connect(viewer.destroy)
    return app.exec()

if __name__=='__main__':raise SystemExit(main())
