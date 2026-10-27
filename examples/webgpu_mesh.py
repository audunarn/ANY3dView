"""Experimental WebGPU desktop/input demo: python examples/webgpu_mesh.py."""
import json
import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit
from any3dview.arrays import MeshArrays
from any3dview.qt_webgpu import QtWebGPUViewer


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("ANY3dView experimental WebGPU — opaque arrays / bitmap readback")
    layout = QVBoxLayout(window)
    layout.addWidget(QLabel("Right drag: orbit · Middle drag: pan · Wheel: zoom · Left click: select"))
    controls = QHBoxLayout()
    fit = QPushButton("Fit")
    view = QComboBox()
    view.addItems(["Isometric", "Top", "Front", "Side"])
    edit = QLineEdit()
    edit.setPlaceholderText("Type here to check desktop input responsiveness")
    controls.addWidget(fit)
    controls.addWidget(view)
    controls.addWidget(edit, 1)
    layout.addLayout(controls)
    viewer = QtWebGPUViewer(window, width=900, height=650, bg="#edf2f7", interaction_profile="commercial")
    layout.addWidget(viewer, 1)
    status = QLabel("Experimental: unlit colors, CPU semantic picking, GPU depth rendering + RGBA readback")
    layout.addWidget(status)
    positions = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1.]], dtype=float)
    for name, triangle, color in zip(("base", "front", "left", "slope"),
            ([0,2,1],[0,1,3],[0,3,2],[1,2,3]),
            ("#2878c8", "#dc7040", "#48a078", "#b884c4")):
        viewer.add_mesh_arrays(MeshArrays(positions,np.array([triangle])),
            color=color, tags=name, cull_backface=False)
    def selected(pick):
        viewer.set_highlight([pick.tag])
        status.setText(f"Selected: {pick.tag} — shared CPU semantic identity")
    viewer.set_pick_callback(selected)
    fit.clicked.connect(lambda: viewer.fit_to_scene())
    methods = [viewer.set_iso_view,viewer.set_top_view,viewer.set_front_view,viewer.set_side_view]
    view.currentIndexChanged.connect(lambda index:methods[index]())
    viewer.fit_to_scene()
    window.resize(1000,800)
    window.show()
    print(json.dumps(viewer.diagnostic_snapshot(), indent=2), flush=True)
    app.aboutToQuit.connect(viewer.destroy)
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
