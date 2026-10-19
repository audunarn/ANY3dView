"""Engineering presets using shared material and boundary-condition APIs."""
import importlib.util
from PySide6.QtWidgets import QFormLayout,QComboBox,QLineEdit,QCheckBox,QPushButton
from anymaterial import available_grades
from ... import commands
from ...model import attributes
from ...model.materials import dnv_steel_material
from ...model.sections import PlateSection
from .editors import CommandEditor


class SectionTask(CommandEditor):
    def __init__(self,app):
        super().__init__(app,["AddMaterial","AddPlateSection","AddBeamSection","AssignPlate","AssignBeam","AddImperfection","AddCoordinateSystem","AddRegion"])
        layout=self.layout();form=QFormLayout();layout.addLayout(form)
        self.grade=QComboBox();self.grade.addItems(list(available_grades()));self.grade.setCurrentText("S355")
        self.thickness=QLineEdit("0.01");self.section_name=QLineEdit("plate")
        self.nonlinear=QCheckBox("DNV RP-C208 nonlinear hardening");self.nonlinear.setChecked(True)
        form.addRow("DNV grade",self.grade);form.addRow("DNV plate thickness (m)",self.thickness);form.addRow("Plate section name",self.section_name);layout.addWidget(self.nonlinear)
        for label,callback in [("Create DNV material",self.create_material),("Create plate section with DNV material",self.create_plate)]:
            button=QPushButton(label);button.clicked.connect(app.guarded(callback));layout.addWidget(button)
        # Standard beam sections come from ANYsection, an optional sibling that is imported lazily
        # so ANYfem starts and works without it.
        self.library_button=QPushButton("Browse section library…");self.library_button.clicked.connect(app.guarded(self.browse_library));layout.addWidget(self.library_button)
        if importlib.util.find_spec("anysection") is None:
            self.library_button.setEnabled(False)
            self.library_button.setToolTip("Install ANYsection (pip install ANYsection[gui]) to browse standard beam sections.")

    def browse_library(self):
        """Pick a standard beam section and load it into the beam form for review."""
        try:from anysection.gui import choose_section
        except ImportError as error:raise ValueError(f"The section library needs ANYsection with PySide6: {error}") from error
        choice=choose_section(parent=self.app)
        return None if choice is None else self.load_library_choice(choice)

    def load_library_choice(self,choice):
        """Fill the Add beam section form from a library choice; nothing is applied here.

        The engineer reviews the values, picks the material and applies, which is one undoable command
        like every other edit.  A material is only suggested when the project has exactly one, because
        guessing between several would be a silent decision.  Dimensions arrive in SI and are shown in
        the project unit profile; the name is qualified by series and standard because catalogue
        designations repeat and ANYfem keys beam sections by name.
        """
        index=self.choice.findData("AddBeamSection")
        if index<0:raise ValueError("Add beam section is unavailable in this task")
        self.choice.setCurrentIndex(index);self.rebuild()
        fields=self.record_fields["section"][1]
        for key,value in choice.anyfem_kwargs().items():
            field=fields[key][0]
            field.setText(self.library_quantity(value,self.dimensions[field]) if field in self.dimensions else self.encoded(value))
        section=choice.section;materials=list(self.app.project.materials)
        if len(materials)==1:fields["material"][0].setText(materials[0])
        notes=[f"Loaded {section.qualified_name} from the section library."]
        if len(materials)!=1:notes.append("Choose the material.")
        if section.profile in {"Angle","L-bulb"}:notes.append("Set the web direction: this profile is asymmetric.")
        if section.derived:notes.append("Equivalent angle of a bulb flat: area and strong-axis inertia match the published section.")
        notes.append("Apply creates one undoable edit.")
        self.feedback.setText(" ".join(notes))
        return section

    def library_quantity(self,value,dimension):
        """Shortest project-unit text that reads back as exactly this SI value.

        Stored values are shown with 17 digits so they round-trip; a catalogue dimension such as 92 mm
        would read 0.091999999999999998 m.  This keeps the same exactness (no rounding) and drops the noise.
        """
        profile=self.app.project.units
        shown=abs(float(profile.to_display(value,dimension)))
        for precision in range(max(3,len(str(int(shown)))),18):
            text=profile.format(value,dimension,precision=precision)
            if profile.parse(text,dimension)==value:return text
        return self.format_quantity(value,dimension,profile)

    def material(self):return dnv_steel_material(self.grade.currentText(),float(self.thickness.text()),nonlinear=self.nonlinear.isChecked())
    def create_material(self):return self.app.run(commands.AddMaterial(self.material()))
    def create_plate(self):
        material=self.material();name=self.section_name.text().strip();previous=self.app.project.plate_sections.get(name)
        options={"id":previous.id} if previous is not None else {}
        return self.app.run(commands.AddPlateSection(PlateSection(name,float(self.thickness.text()),material.name,**options),material))


class LoadTask(CommandEditor):
    SELECTION_KINDS={commands.AddPointLoad:"vertex",commands.AddPressure:"face",
        commands.AddLineLoad:"edge",commands.AddSurfaceTraction:"face",commands.AddMass:None,
        commands.AddWetSurface:"face"}
    def __init__(self,app):
        super().__init__(app,["AddPointLoad","AddPressure","AddSupport","AddLineLoad","AddSurfaceTraction","AddConnection","AddConnectionLoad","AddWetSurface","AddWetPressure","AddWetPressureTable","AddMass","SetAcceleration","SetFollowerPressure","AddLoadCase","AddLoadSeries","SetLoadSeriesTimes","AddLoadCaseSeries","AddLoadCasesFromTable","DeleteLoadCase","AddCombination","EditAttribute","DeleteAttribute"])
        form=QFormLayout();self.layout().addLayout(form)
        self.case=QComboBox();form.addRow("Active load case",self.case)
        self.case.currentTextChanged.connect(self.case_changed)
        self.preset=QComboBox();self.preset.addItems(["fixed","pinned","simply_supported","symmetry","antisymmetry"]);form.addRow("Support preset",self.preset)
        self.normal=QLineEdit("z");form.addRow("Symmetry normal (x/y/z)",self.normal)
        self.dof=QComboBox();self.dof.addItems(["ux","uy","uz"]);form.addRow("Simply supported component",self.dof)
        button=QPushButton("Apply support preset to selection");button.clicked.connect(app.guarded(self.support_preset));self.layout().addWidget(button)

    def support_preset(self):
        refs=list(self.app.selection.ordered_items)
        if not refs:raise ValueError("Select geometry for the support preset")
        kind=self.preset.currentText();options={}
        if kind in {"symmetry","antisymmetry"}:options["normal"]=self.normal.text()
        elif kind=="simply_supported":options["normal"]=self.dof.currentText()
        self.app.run_many(commands.AddSupport(getattr(attributes,kind)(ref,**options)) for ref in refs)

    def execute(self):
        refs=tuple(self.app.selection.ordered_items)
        if (self.operation not in self.SELECTION_KINDS or len(refs)<2
                or self.fields["ref"][0].text().strip()):
            return super().execute()
        kind=self.SELECTION_KINDS[self.operation]
        if self.app.selection.domain.value!="geometry" or (kind is not None and any(ref.kind!=kind for ref in refs)):
            raise ValueError(f"Select {kind or 'geometry'} entities for this load")
        options=self.command_options({"ref":refs[0]})
        if options.get("distribution_policy")=="total_distributed":
            if self.operation is commands.AddPointLoad:
                for key in ("force","moment"):options[key]=tuple(value/len(refs) for value in options[key])
            elif self.operation is commands.AddMass:options["value"]/=len(refs)
        result=self.app.run_many(self.operation(**dict(options,ref=ref)) for ref in refs)
        self.feedback.setText(f"Applied to {len(refs)} selected entities")
        return result

    def refresh(self):
        super().refresh()
        current=self.case.currentText();choices=list(self.app.project.load_cases)
        if choices!=[self.case.itemText(i) for i in range(self.case.count())]:
            # Never fall back to a series definition or template: it cannot be solved.
            solvable=[name for name in choices if not self.app.project.load_cases[name].is_template]
            fallback="default" if "default" in solvable else (solvable[0] if solvable else (choices[0] if choices else "default"))
            self.case.clear();self.case.addItems(choices);self.case.setCurrentText(current if current in choices else fallback)

    def case_name(self):return self.case.currentText() or "default"

    def case_changed(self,value):
        if "case" in self.fields:self.fields["case"][0].setText(value)

    def rebuild(self):
        super().rebuild()
        if hasattr(self,"case"):self.case_changed(self.case_name())
