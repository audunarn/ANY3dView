"""Working-copy construction and face sketches on the shared viewport."""
from dataclasses import replace,asdict
from copy import deepcopy
import json

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QComboBox,
    QLineEdit, QCheckBox, QPushButton, QLabel, QPlainTextEdit)
from anygeometry import face_sketch_plane, SketchConstraint,SketchDefinition

from ... import commands
from ...geometry.construction import ConstructionTask, ConstructionMode
from ...geometry.sketching import FaceSketchTask
from ...geometry.snapping import geometry_snap_data
from ...model.coordinates import CoordinateSystem
from ...model.workplanes import Workplane


class ConstructionPanel(QWidget):
    def __init__(self, app):
        super().__init__();self.app=app;self.support=None;self.editing_feature_id=None
        layout=QVBoxLayout(self);form=QFormLayout();layout.addLayout(form)
        self.form=form;self.length_labels={}
        self.system=QComboBox();form.addRow("Coordinate system",self.system)
        self.offset=QLineEdit("0");self.length_row("Plane offset",self.offset)
        self.spacing=QLineEdit("1 m");self.length_row("Grid spacing",self.spacing)
        self.tolerance=QLineEdit("0.05 m");self.length_row("Snap tolerance",self.tolerance)
        self.grid=QCheckBox("Snap to grid and axes");self.grid.setChecked(True);layout.addWidget(self.grid)
        self.objects=QCheckBox("Snap to endpoints, midpoints and intersections");self.objects.setChecked(True);layout.addWidget(self.objects)
        self.mode=QComboBox();self.mode.addItems([item.value for item in ConstructionMode]);form.addRow("Construction",self.mode)
        self.closed=QCheckBox("Close polyline");layout.addWidget(self.closed)
        self.coordinates=QLineEdit();self.length_row("Add local point (u, v)",self.coordinates)
        self.extrusion=QLineEdit("0");self.length_row("Sketch extrusion",self.extrusion)
        self.constraints=QPlainTextEdit("[]");form.addRow("Sketch constraints",self.constraints)
        self.constraints.setToolTip("Constraint records use SI metres for dimensional values.")
        self.pair=QLineEdit("1, 2");self.distance=QLineEdit("1 m")
        form.addRow("Constraint point numbers",self.pair);self.length_row("Constraint distance",self.distance)
        for label,callback in [("Construct plate from existing vertices",lambda:app.geometry_actions.start_plate()),("Start construction",self.start),("Sketch selected flat plate",self.sketch),
                               ("Add local point",self.add_point),("Remove last point",self.backspace),
                               ("Add distance constraint",lambda:self.add_constraint("distance")),("Add coincidence constraint",lambda:self.add_constraint("coincident")),("Remove last constraint",self.remove_constraint),
                               ("Apply",self.apply),("Cancel",app.viewport.cancel_construction)]:
            button=QPushButton(label);button.clicked.connect(app.guarded(callback));layout.addWidget(button)
        self.summary=QLabel("Click the viewport or enter local coordinates. Enter applies; Escape cancels.")
        self.summary.setWordWrap(True);layout.addWidget(self.summary);layout.addStretch()

    def length_row(self,label,field):
        self.form.addRow(f"{label} ({self.app.project.units.symbol('length')})",field)
        self.length_labels[field]=label

    def length(self,text):
        return self.app.project.units.parse(text,"length")

    def workplane(self):
        objects=self.objects.isChecked()
        return Workplane(self.system.currentData(),self.length(self.offset.text()),self.length(self.spacing.text()),
                         self.length(self.tolerance.text()),self.grid.isChecked(),self.grid.isChecked(),objects,objects,objects)

    def begin(self,task,plane,systems):
        self.app.viewport.cancel_construction()
        self.app.session.active_workplane=plane
        self.app.viewport.begin_construction(task,plane,systems,
            snap_data=lambda:geometry_snap_data(self.app.project.geometry),
            update_handler=self.updated,apply_handler=self.app.guarded(self.apply))
        self.updated(task,None)

    def start(self):
        self.support=None;self.editing_feature_id=None
        self.begin(ConstructionTask(self.mode.currentText(),close=self.closed.isChecked()),
                   self.workplane(),self.app.project.coordinate_systems)

    def sketch(self):
        refs=self.app.selection.ordered_items
        if len(refs)!=1 or refs[0].kind!="face":raise ValueError("Select one flat plate for its sketch")
        self.start_sketch(refs[0])

    def start_sketch(self,support,definition=None,feature_id=None):
        self.support=support;self.editing_feature_id=feature_id
        plane=face_sketch_plane(self.app.project.geometry,self.support.id)
        system=CoordinateSystem("Sketch plane",origin=plane.origin,axis=plane.normal,reference=plane.x_axis)
        systems=dict(self.app.project.coordinate_systems);systems[system.id]=system
        workplane=replace(self.workplane(),coordinate_system_id=system.id,offset=0)
        self.closed.setChecked(definition.closed if definition is not None else True)
        self.extrusion.setText(self.app.project.units.format(definition.extrusion if definition is not None else 0,"length",precision=17))
        self.begin(FaceSketchTask(plane,workplane.snap_tolerance,definition),workplane,systems)

    def edit_sketch(self,feature_id):
        record=self.app.project.geometry.features.get(feature_id)
        if record.kind!="geometry.sketch.extrude":return False
        refs=record.inputs.get("support_face",())
        if len(refs)!=1:raise ValueError("Sketch support is invalid")
        resolved=self.app.project.geometry.features.resolve(refs[0],self.app.project.geometry)
        if len(resolved)!=1 or resolved[0].kind!="face":raise ValueError("Sketch support is unresolved")
        self.start_sketch(resolved[0],SketchDefinition.from_parameters(record.parameters),record.feature_id)
        self.app.notebook.select("Construction");return True

    def add_constraint(self,kind):
        task=self.app.viewport.construction_task
        if not isinstance(task,FaceSketchTask):raise ValueError("Start a face sketch first")
        indices=tuple(int(value.strip())-1 for value in self.pair.text().split(","))
        if len(indices)!=2 or indices[0]==indices[1] or min(indices)<0 or max(indices)>=len(task.point_keys):raise ValueError("Enter two different existing point numbers")
        extrusion=self.length(self.extrusion.text()) if task.ready else None
        previous_count=len(task.constraints)
        if kind=="distance":task.add_distance(*indices,self.length(self.distance.text()))
        else:task.add_coincidence(*indices)
        try:
            if task.ready:task.solve_preview(extrusion)
        except Exception:
            # Owner validation leaves points unchanged on failure. Discard only
            # the attempted constraint so the visible working copy stays valid.
            del task.constraints[previous_count:]
            raise
        self.app.viewport.refresh_construction_overlay();self.updated(task,None)

    def remove_constraint(self):
        task=self.app.viewport.construction_task
        if not isinstance(task,FaceSketchTask):raise ValueError("Start a face sketch first")
        task.remove_last_constraint();self.app.viewport.refresh_construction_overlay();self.updated(task,None)

    def add_point(self):
        task=self.app.viewport.construction_task
        if task is None:raise ValueError("Start construction before adding a point")
        u,v=(self.length(value) for value in self.coordinates.text().split(","))
        frame=self.app.viewport._construction_workplane.resolve(self.app.viewport._construction_coordinate_systems)
        task.add_plane_coordinates(u,v,frame)
        self.app.viewport.refresh_construction_overlay();self.updated(task,None)

    def backspace(self):
        task=self.app.viewport.construction_task
        if task is not None:
            task.backspace();self.app.viewport.refresh_construction_overlay();self.updated(task,None)

    def apply(self):
        if self.app.session.read_only:raise PermissionError("This project is read-only")
        task=self.app.viewport.construction_task
        if isinstance(task,FaceSketchTask):
            constraints=json.loads(self.constraints.toPlainText())
            # Validate the proposed edit separately. A failed owner solve or
            # project command must leave the displayed working preview intact.
            preview=deepcopy(task)
            preview.constraints[:]=[SketchConstraint(**item) for item in constraints]
            preview.close=self.closed.isChecked()
            definition=preview.solve_preview(self.length(self.extrusion.text()))
            if self.editing_feature_id is None:feature=self.app.run(commands.AddSketch(self.support,definition))
            else:feature=self.app.run(commands.EditFeature(self.editing_feature_id,parameters=definition.to_parameters()))
            self.app.viewport.end_construction()
            self.editing_feature_id=None
            self.support=None
            made=tuple(ref for key,ref in feature.outputs.items() if key.startswith("extrusion/face/"))
            if made:
                self.app.selection.set_mode("face")
                self.app.selection.restore(made)
        else:
            result=self.app.viewport.finish_construction(self.app.run)
            made=result.edges or result.vertices
            if made:
                self.app.selection.set_mode("edge" if result.edges else "vertex")
                self.app.selection.restore(made)
        self.app.set_status("Construction applied")

    def updated(self,task,snap):
        if isinstance(task,FaceSketchTask):self.constraints.setPlainText(json.dumps([asdict(item) for item in task.constraints]))
        self.summary.setText(f"{len(task.points)} point(s) · {'ready to apply' if task.ready else 'add more points'}"+
                             (f" · {snap.kind.value} snap" if snap is not None and snap.kind is not None else ""))

    def refresh(self):
        for field,label in self.length_labels.items():
            self.form.labelForField(field).setText(f"{label} ({self.app.project.units.symbol('length')})")
        values=[(system.name,identifier) for identifier,system in self.app.project.coordinate_systems.items()]
        if values!=[(self.system.itemText(i),self.system.itemData(i)) for i in range(self.system.count())]:
            current=self.system.currentData();self.system.clear()
            for name,identifier in values:self.system.addItem(name,identifier)
            index=self.system.findData(current)
            if index>=0:self.system.setCurrentIndex(index)
