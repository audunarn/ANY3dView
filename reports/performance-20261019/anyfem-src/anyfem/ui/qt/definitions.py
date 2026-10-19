"""Typed persistent region, output and unit tasks for the Qt workbench."""
from PySide6.QtWidgets import QFormLayout,QLineEdit,QComboBox,QPushButton,QListWidget,QAbstractItemView,QLabel
from ... import commands
from ...model.units import UNIT_PROFILES,UnitProfile
from ...model.records import OutputRequest
from ...presentation.definitions import region_from_selection,boolean_region,_UNIT_CHOICES
from .editors import CommandEditor


class DefinitionsTask(CommandEditor):
    def __init__(self,app):
        super().__init__(app,["AddCoordinateSystem","AddOutputRequest","EditOutputRequest","DeleteOutputRequest","SetUnitProfile","DeleteProjectRecord","SetElementOrder","RefineForImpact"])
        self.RECORD_TYPES=dict(self.RECORD_TYPES,OutputRequest=OutputRequest)
        layout=self.layout();form=QFormLayout();layout.addLayout(form)
        self.region_name=QLineEdit("Region-1");form.addRow("Region name",self.region_name)
        button=QPushButton("Create region from selection");button.clicked.connect(app.guarded(self.create_region));layout.addWidget(button)
        self.boolean_operation=QComboBox();self.boolean_operation.addItems(["union","intersection","subtract"]);form.addRow("Boolean operation",self.boolean_operation)
        self.operands=QListWidget();self.operands.setSelectionMode(QAbstractItemView.ExtendedSelection);layout.addWidget(self.operands)
        button=QPushButton("Create Boolean region");button.clicked.connect(app.guarded(self.create_boolean));layout.addWidget(button)
        self.profile=QComboBox();self.profile.addItems(list(UNIT_PROFILES)+["Custom"]);form.addRow("Unit profile",self.profile)
        self.unit_fields={};self.unit_name=QLineEdit("Project units");form.addRow("Custom profile name",self.unit_name)
        for dimension,values in _UNIT_CHOICES.items():
            choice=QComboBox();choice.addItems(values);self.unit_fields[dimension]=choice;form.addRow(dimension.capitalize(),choice)
        self.profile.currentTextChanged.connect(self.profile_selected)
        self.profile_selected()
        button=QPushButton("Apply unit profile");button.clicked.connect(app.guarded(self.apply_units));layout.addWidget(button)
        self._region_ids=()
        self._operand_signature=()
        self._profile_signature=None

    def create_region(self):
        mesh_id=self.app.mesh_record_id;pending=[]
        if self.app.selection.domain.value=="mesh" and mesh_id not in self.app.project.mesh_records:
            from uuid import uuid4
            from anymesher.serialize import mesh_to_dict
            from ...document import canonical_hash
            from ...model.records import MeshRecord
            mesh=self.app.mesh
            if mesh is None:raise ValueError("Generate or import a mesh before creating a mesh region")
            mesh_id=mesh_id or str(uuid4())
            record=MeshRecord(id=mesh_id,name="Imported mesh" if self.app.imported is not None else "Mesh",kind="imported" if self.app.imported is not None else "generated",source_model_hash=self.app.session.revision.model_hash,mesh_input_hash="",mesh_hash=canonical_hash(mesh_to_dict(mesh)),summary={"nodes":mesh.num_nodes,"elements":mesh.num_elements})
            pending.append(commands.AddMeshRecord(record))
        region=region_from_selection(self.app.project,self.region_name.text(),self.app.selection.ordered_items,self.app.selection.mode,mesh_id=mesh_id)
        pending.append(commands.AddRegion(region))
        self.app.run_many(pending)
        if self.app.selection.domain.value=="mesh":
            self.app.mesh_record_id=mesh_id
            self.app._meshes[mesh_id]=self.app.mesh
        return region

    def create_boolean(self):
        regions=[self.app.project.regions[self._region_ids[self.operands.row(item)]] for item in self.operands.selectedItems()]
        region=boolean_region(self.region_name.text(),self.boolean_operation.currentText(),regions)
        self.app.run(commands.AddRegion(region));return region

    def apply_units(self):
        name=self.profile.currentText()
        profile=UNIT_PROFILES[name] if name!="Custom" else UnitProfile(self.unit_name.text(),{dimension:choice.currentText() for dimension,choice in self.unit_fields.items()})
        self.app.run(commands.SetUnitProfile(profile))

    def profile_selected(self):
        builtin=UNIT_PROFILES.get(self.profile.currentText())
        self.unit_name.setEnabled(builtin is None)
        for dimension,field in self.unit_fields.items():
            if builtin is not None:field.setCurrentText(builtin.symbol(dimension))
            field.setEnabled(builtin is None)

    def refresh(self):
        super().refresh()
        profile=self.app.project.units;signature=(profile.name,tuple(sorted(profile.units.items())))
        if signature!=self._profile_signature:
            self._profile_signature=signature
            builtin=UNIT_PROFILES.get(profile.name)
            self.profile.setCurrentText(profile.name if builtin is not None and dict(builtin.units)==dict(profile.units) else "Custom")
            self.unit_name.setText(profile.name)
            for dimension,field in self.unit_fields.items():field.setCurrentText(profile.symbol(dimension))
            self.profile_selected()
        regions=[region for region in self.app.project.regions if not region.hidden]
        identifiers=tuple(region.id for region in regions)
        signature=tuple((region.id,region.name,region.domain.value,region.entity_kind,region.mesh_id)
                        for region in regions)
        if signature!=self._operand_signature:
            selected={self._region_ids[self.operands.row(item)] for item in self.operands.selectedItems()
                      if self.operands.row(item)<len(self._region_ids)}
            if identifiers!=self._region_ids:
                self.operands.clear()
                self.operands.addItems([""]*len(regions))
            self._region_ids=identifiers
            for index,region in enumerate(regions):
                mesh_suffix=f" / {region.mesh_id[:8]}" if region.mesh_id else ""
                item=self.operands.item(index)
                item.setText(f"{region.name}  [{region.domain.value}/{region.entity_kind}{mesh_suffix}]")
                item.setSelected(region.id in selected)
            self._operand_signature=signature
