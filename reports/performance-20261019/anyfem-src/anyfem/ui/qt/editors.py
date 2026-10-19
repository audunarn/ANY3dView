"""Command-backed Qt editors. Model changes always enter the shared command stack."""
from __future__ import annotations

import dataclasses
import inspect
import json
import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QComboBox, QFormLayout, QLabel, QLineEdit, QCheckBox,
    QPlainTextEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from ... import commands
from anyloads import Envelope
from ...model.attributes import PressureTable, Support
from ...model.materials import MaterialSpec
from ...model.sections import PlateSection, BeamSection
from ...model.coordinates import CoordinateSystem
from ...model.imperfections import Imperfection
from ...model.records import OutputRequest
from ...model.regions import region_from_dict, RegionRef
from ...model.units import UnitProfile
from anygeometry import SketchDefinition


class CodeEdit(QPlainTextEdit):
    """A monospace multi-line field that reads and writes like a QLineEdit."""

    def __init__(self):
        super().__init__()
        self.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.setTabChangesFocus(False)
        self.setMinimumHeight(150)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)

    def text(self):return self.toPlainText()
    def setText(self,value):self.setPlainText(value)


PRESSURE_CODE_HINT="\n".join((
    "def pressure(case, x, y, z):",
    "    # case: load case number; x, y, z: element centre in metres.",
    "    # Optional arguments: t (case time), name, parameters.",
    "    return 1025 * 9.81 * max(0.0, 2.0 - z)",
))


def title(name):
    if name == "frame_indices":return "Frame indices (zero based)"
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).replace("_", " ").capitalize()


def reference(app, text):
    """Resolve explicit or selected project-bound topology; never guess coordinates."""
    if not text.strip():
        refs = app.selection.ordered_items
        if len(refs) != 1:
            raise ValueError("Select one entity or enter its type and ID, for example vertex:1")
        return refs[0]
    kind, identifier = text.strip().split(":", 1)
    if kind=="group":
        if app.imported is None:raise ValueError("Group references require an imported model")
        return app.imported.groups[identifier]
    if kind in {"node","element","element_face"}:
        from ...selection import MeshEntityRef
        value=tuple(int(item) for item in identifier.split(",")) if kind=="element_face" else int(identifier)
        return MeshEntityRef(kind,value)
    return app.project.geometry.entity_ref(kind, int(identifier))


class CommandEditor(QWidget):
    """Readable forms for public commands, including nested engineering records."""
    RECORD_TYPES = {"MaterialSpec": MaterialSpec, "PlateSection": PlateSection,
                    "BeamSection": BeamSection, "Support": Support,
                    "CoordinateSystem": CoordinateSystem, "Imperfection": Imperfection}

    def __init__(self, app, names, parent=None):
        super().__init__(parent)
        self.app = app
        layout = QVBoxLayout(self)
        self.choice = QComboBox()
        for name in names:
            if hasattr(commands,name):
                self.choice.addItem(title(name), name)
        layout.addWidget(self.choice)
        guidance=QLabel("Engineering quantities use the project unit profile; explicit unit suffixes are accepted. Entity IDs follow the model tree.")
        guidance.setWordWrap(True);layout.addWidget(guidance)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)
        self.feedback = QLabel()
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        self.apply = QPushButton("Apply")
        self.apply.setProperty("primary",True)
        self.apply.clicked.connect(app.guarded(self.execute))
        layout.addWidget(self.apply)
        self.choice.currentIndexChanged.connect(self.rebuild)
        self.rebuild()

    def rebuild(self):
        self.loaded_ids={};self.optional_records={}
        self.dimensions={}
        self.fields = {}
        self.record_fields = {}
        self.operation = getattr(commands, self.choice.currentData())
        container = QWidget()
        form = QFormLayout(container)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.signature = inspect.signature(self.operation)
        for name, parameter in self.signature.parameters.items():
            if name in {"label","id"} or parameter.kind in {parameter.VAR_POSITIONAL,parameter.VAR_KEYWORD}:
                continue
            annotation = str(parameter.annotation).strip("'")
            record = next((t for key,t in self.RECORD_TYPES.items() if annotation.startswith(key)),None)
            if name=="replacement":record=getattr(self,"replacement_type",None)
            if record:
                form.addRow(QLabel(title(name)))
                if "None" in annotation:
                    enabled=QCheckBox("Include "+title(name));form.addRow(enabled)
                    self.optional_records[name]=enabled
                fields={}
                for f in dataclasses.fields(record):
                    if not f.init or f.name == "id":
                        continue
                    default = f.default
                    if default is dataclasses.MISSING and f.default_factory is not dataclasses.MISSING:
                        default=f.default_factory()
                    field=self.make_field(default, str(f.type), f.name)
                    self.unit_field(field,default,f.name,record.__name__)
                    form.addRow(title(f.name),field)
                    fields[f.name]=(field,str(f.type),default)
                self.record_fields[name]=(record,fields)
            else:
                field=self.make_field(parameter.default,annotation,name)
                self.unit_field(field,parameter.default,name,self.operation.__name__)
                form.addRow(title(name),field)
                self.fields[name]=(field,annotation,parameter.default)
        previous=self.scroll.takeWidget()
        if previous is not None:
            previous.deleteLater()
        self.scroll.setWidget(container)
        self.feedback.clear()

    @staticmethod
    def make_field(default, annotation, name):
        field=CodeEdit() if name=="code" else QLineEdit()
        missing=default is inspect.Parameter.empty or default is dataclasses.MISSING or repr(default)=="<factory>"
        if not missing and default is not None:
            field.setText(default if isinstance(default,str) else json.dumps(default,default=lambda value:value.tolist() if hasattr(value,"tolist") else str(value)))
        if name=="code":
            field.setPlaceholderText(PRESSURE_CODE_HINT)
        elif "Envelope" in annotation:
            field.setPlaceholderText('When it acts, e.g. {"start": 20, "stop": 300, "by": "number", "expression": "sin(omega * t)"}; empty = always')
        elif "EntityRef" in annotation:
            field.setPlaceholderText("Selection, vertex:1 / edge:1 / face:1, or group:name")
        elif "Sequence" in annotation:
            field.setPlaceholderText("Comma-separated values")
        elif "Mapping" in annotation or "Dict" in annotation or "dict" in annotation:
            field.setPlaceholderText('For example {"ux": 0, "uy": 0, "uz": 0}')
        elif missing:
            field.setPlaceholderText("Required")
        return field

    def value(self, field, annotation, default):
        text=field.text().strip()
        if text=="null" and ("None" in annotation or "Optional" in annotation):return None
        if annotation.startswith("RegionRef"):
            return RegionRef(text.removeprefix("region:"))
        if text.startswith("region:") and "RegionRef" in annotation:return RegionRef(text.split(":",1)[1])
        if "EntityRef" in annotation and any(word in annotation for word in ("Sequence", "List", "list", "tuple")):
            if not text:return tuple(self.app.selection.ordered_items)
            values=json.loads(text) if text.startswith("[") else text.split(",")
            return tuple(reference(self.app,value) for value in values)
        if "EntityRef" in annotation:
            return reference(self.app,text)
        if not text:
            if default is None or "None" in annotation:
                return None
            if default is not inspect.Parameter.empty and default is not dataclasses.MISSING and repr(default)!="<factory>":
                return default
            raise ValueError("A required field is empty")
        dimension=self.dimensions.get(field) if hasattr(self,"dimensions") else None
        if dimension is not None:
            return self.parse_quantity(text,annotation,dimension,self.app.project.units)
        if annotation.startswith("str") or annotation.startswith("Literal") or annotation=="<class 'str'>" or annotation.startswith("Optional[str]"):
            return text
        if annotation.startswith("UnitProfile") and not text.startswith("{"):
            return text
        if annotation.startswith("float") or annotation=="<class 'float'>":
            return float(text)
        if annotation.startswith("int") or annotation=="<class 'int'>":
            return int(text)
        if annotation.startswith("bool"):
            if text.casefold() not in {"true","false","1","0"}:
                raise ValueError("Boolean fields accept true or false")
            return text.casefold() in {"true","1"}
        if any(word in annotation for word in ("Sequence","Tuple","tuple[","list[","ndarray")):
            if "str" in annotation and not text.startswith("["):
                return tuple(value.strip() for value in text.split(",") if value.strip())
            if not text.startswith("[") and not text.startswith("("):
                text="["+text+"]"
        try:
            value=json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"Enter a valid value: {text}") from None
        converters={"SketchDefinition":SketchDefinition.from_parameters,
                    "Region":region_from_dict,"OutputRequest":OutputRequest.from_dict,
                    "UnitProfile":UnitProfile.from_dict,
                    "PressureTable":lambda data:PressureTable(**data),
                    "Envelope":Envelope.from_dict}
        converter=converters.get(annotation.split(" | ")[0])
        return converter(value) if converter is not None else value

    @staticmethod
    def dimension(name,owner):
        if name in {"plane_point", "axis_point", "spacing"}:return "length"
        if name=="vector" and owner=="Extrude":return "length"
        if name in {"x","y","z","origin","centre","center","position","translation","reference_point","offset","eccentricity","radius","radius_start","radius_end","height","length","width","thickness","web_height","web_thickness","flange_width","flange_thickness","longitudinal_spacing","ring_spacing","transverse_spacing","extrusion","amplitude","target_size"}:return "length"
        if name in {"angle","angle_step"}:return "angle"
        if name=="rotation_deg":return "angle_degrees"
        if name=="force":return "force"
        if name=="moment":return "moment"
        if name=="force_per_length":return "line_load"
        if name in {"traction","force_per_area"}:return "pressure"
        if name=="constraints":return "constraints"
        if name=="value":return {"AddPressure":"pressure","Pressure":"pressure","AddWetPressure":"pressure","AddMass":"mass","Mass":"mass"}.get(owner)
        if name=="acceleration":return "acceleration"
        if name=="density":return "density"
        if name=="yield_stress":return "pressure"
        return None

    @staticmethod
    def format_quantity(value,dimension,profile):
        import numpy as np
        if dimension=="constraints":return json.dumps({key:profile.format(number,"length" if key.startswith("u") else "angle",precision=17) for key,number in value.items()})
        if dimension=="angle_degrees":value=np.deg2rad(value);dimension="angle"
        if isinstance(value,(list,tuple,np.ndarray)):return ", ".join(profile.format(float(number),dimension,precision=17) for number in value)
        return profile.format(float(value),dimension,precision=17)

    @staticmethod
    def parse_quantity(text,annotation,dimension,profile):
        import numpy as np
        if dimension=="constraints":return {key:profile.parse(value,"length" if key.startswith("u") else "angle") for key,value in json.loads(text).items()}
        actual="angle" if dimension=="angle_degrees" else dimension
        if any(word in annotation for word in ("Sequence","Tuple","tuple[","list[","ndarray")):
            values=json.loads(text) if text.startswith("[") else text.split(",")
            result=tuple(profile.parse(value,actual) for value in values)
        else:result=profile.parse(text,actual)
        return float(np.rad2deg(result)) if dimension=="angle_degrees" else result

    def unit_field(self,field,value,name,owner):
        dimension=self.dimension(name,owner)
        if dimension is None:return
        self.dimensions[field]=dimension
        missing=value is inspect.Parameter.empty or value is dataclasses.MISSING or repr(value)=="<factory>"
        if not missing and value is not None:field.setText(self.format_quantity(value,dimension,self.app.project.units))

    def command_options(self,overrides=None):
        overrides=overrides or {}
        options={}
        for name,(field,annotation,default) in self.fields.items():
            if name in overrides:
                options[name]=overrides[name];continue
            if not field.text().strip() and repr(default)=="<factory>":
                continue
            options[name]=self.value(field,annotation,default)
        if self.operation in {commands.AddFeature,commands.EditFeature} and options.get("inputs") is not None:
            from anygeometry import FeatureOutputRef
            def input_ref(value):
                if isinstance(value,str):return reference(self.app,value)
                if isinstance(value,dict):
                    if "entity" in value:return reference(self.app,":".join(map(str,value["entity"])))
                    if "feature" in value:return FeatureOutputRef(value["feature"],value["output"],value["kind"])
                    if "feature_id" in value:return FeatureOutputRef(**value)
                    return reference(self.app,f"{value['kind']}:{value['id']}")
                return value
            options["inputs"]={key:tuple(input_ref(value) for value in refs) for key,refs in options["inputs"].items()}
        for name,(record,fields) in self.record_fields.items():
            if name in self.optional_records and not self.optional_records[name].isChecked():
                options[name]=None;continue
            values={key:self.value(field,annotation,default) for key,(field,annotation,default) in fields.items()}
            if name in self.loaded_ids:values["id"]=self.loaded_ids[name]
            options[name]=record(**values)
        return options

    def execute(self):
        options=self.command_options()
        command=self.operation(**options)
        result=self.app.commands.query(command) if self.operation is commands.MeasureGeometry else self.app.run(command)
        self.feedback.setText(f"Applied {title(self.operation.__name__)}"+(f": {result}" if isinstance(result,(int,str)) else ""))
        return result

    @staticmethod
    def encoded(value):
        if isinstance(value,RegionRef):return value.id
        if hasattr(value,"kind") and hasattr(value,"id"):return f"{value.kind}:{value.id}"
        if isinstance(value,str):return value
        if hasattr(value,"to_dict"):value=value.to_dict()
        return json.dumps(value,default=lambda item:item.tolist() if hasattr(item,"tolist") else dataclasses.asdict(item) if dataclasses.is_dataclass(item) else str(item))

    def load_values(self,operation,**values):
        index=self.choice.findData(operation)
        if index<0:raise ValueError(f"{operation} is unavailable in this task")
        replacement=values.get("replacement")
        self.replacement_type=type(replacement) if replacement is not None else None
        self.choice.setCurrentIndex(index);self.rebuild()
        for name,value in values.items():
            if name in self.record_fields:
                if name in self.optional_records:self.optional_records[name].setChecked(value is not None)
                if value is None:continue
                if hasattr(value,"id"):self.loaded_ids[name]=value.id
                for key,(field,*_) in self.record_fields[name][1].items():
                    number=getattr(value,key)
                    field.setText(self.format_quantity(number,self.dimensions[field],self.app.project.units) if field in self.dimensions and number is not None else self.encoded(number))
                    if key=="name":field.setReadOnly(True)
            elif name in self.fields:
                field=self.fields[name][0]
                field.setText(self.format_quantity(value,self.dimensions[field],self.app.project.units) if field in self.dimensions and value is not None else self.encoded(value))
        self.feedback.setText("Loaded existing values. Apply creates one undoable edit.")

    def refresh(self):
        self.apply.setEnabled(not self.app.session.read_only and not self.app._closing)

    def case_name(self):
        field=self.fields.get("case")
        return field[0].text() if field else "default"


class GeometryTask(CommandEditor):
    def __init__(self,app,names):
        super().__init__(app,names)
        app.selection.add_listener(self.populate_selection)
        self.populate_selection()
        button=QPushButton("Check selected plates for mapped meshing")
        button.clicked.connect(app.guarded(self.check_mappable));self.layout().addWidget(button)

    def rebuild(self):
        super().rebuild()
        self._selection_signature = None
        self.populate_selection()
        help_text = {
            "AddFace": "Closed boundary edge IDs create a geometry face. Assign material and thickness separately.",
            "AddPlate": "Boundary-ordered vertex IDs create boundary lines and a geometry face. Review the order; assign material and thickness separately.",
        }.get(self.choice.currentData(), "IDs are filled from matching selected geometry. Explicit edits remain until selection changes.")
        self.feedback.setText(help_text)
        self.apply.setToolTip(f"Apply {self.choice.currentText()} using the entered values as an undoable model edit. {help_text}")

    def populate_selection(self):
        refs = tuple(self.app.selection.ordered_items)
        signature = tuple((ref.kind, ref.id) for ref in refs)
        if signature == getattr(self, "_selection_signature", None):return
        self._selection_signature = signature
        from anygeometry.entities import EntityRef
        for name, (field, _, _) in self.fields.items():
            kind = {"vertex_ids":"vertex", "edge_ids":"edge", "face_ids":"face"}.get(name)
            if kind is None:continue
            ids = [ref.id for ref in refs if isinstance(ref, EntityRef) and ref.kind == kind]
            field.setText(", ".join(map(str, ids)))
        vertex_ids = [ref.id for ref in refs if isinstance(ref, EntityRef) and ref.kind == "vertex"]
        names = {"AddLine": ("start", "end"), "AddArc": ("start", "via", "end")}.get(self.choice.currentData(), ())
        for index, name in enumerate(names):
            self.fields[name][0].setText(str(vertex_ids[index]) if index < len(vertex_ids) else "")
        for name, (field, _, _) in self.fields.items():
            kind = {"vertex_id":"vertex", "edge_id":"edge", "face_id":"face"}.get(name)
            if kind:
                ids = [ref.id for ref in refs if isinstance(ref, EntityRef) and ref.kind == kind]
                field.setText(str(ids[0]) if len(ids) == 1 else "")

    def check_mappable(self):
        from anymesher.decomposition import check_mappable
        refs=self.app.selection.ordered_items
        if not refs or any(ref.kind!="face" for ref in refs):raise ValueError("Select plates to check")
        reports=[check_mappable(self.app.project.geometry,ref.id) for ref in refs]
        bad=[report for report in reports if not report.ok]
        self.feedback.setText(str(bad[0]) if bad else f"{len(reports)} plate(s): all mappable")
        return reports
