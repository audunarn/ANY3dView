"""Commercial-style Details tasks for regions, coordinates and units.

The form widgets live here, while the constructors below remain Tk-free so
scripts and focused tests use exactly the same validation as the desktop UI.
All project changes are dispatched as commands; previews and form edits never
mutate the live document.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np
from anygeometry.entities import EntityRef

from .. import commands as cmd
from ..model.coordinates import CoordinateSystem
from ..model.records import MeshRecord, OutputRequest
from ..model.regions import (
    BooleanRegion,
    ElementFaceRef,
    ManualRegion,
    MeshEntityRef as RegionMeshEntityRef,
    Region,
    RegionDomain,
)
from ..model.units import UNIT_PROFILES, UnitProfile
from ..selection import MeshEntityRef as SelectionMeshEntityRef, mode_label
from .panels import StagePanel

__all__ = [
    "DefinitionsPanel",
    "MAX_REGION_OPERANDS",
    "boolean_region",
    "coordinate_system_from_values",
    "output_request_from_values",
    "region_from_selection",
    "unit_profile_from_values",
]


from ..presentation.definitions import (
    MAX_REGION_OPERANDS, _UNIT_CHOICES, _UNIT_LABELS, _feature_anchor, _next_name,
    region_from_selection, boolean_region, coordinate_system_from_values,
    unit_profile_from_values, output_request_from_values,
)


class DefinitionsPanel(StagePanel):
    """Bounded task editor for persistent reusable model definitions."""

    title = "Definitions"
    TASKS = (
        "Region from selection",
        "Boolean region",
        "Coordinate system",
        "Output request",
        "Units",
    )

    def build(self) -> None:
        chooser = ttk.Frame(self)
        chooser.pack(fill="x", pady=(0, 8))
        ttk.Label(chooser, text="Task", width=10).pack(side="left")
        self._task = tk.StringVar(value=self.TASKS[0])
        task_box = ttk.Combobox(
            chooser,
            textvariable=self._task,
            values=self.TASKS,
            state="readonly",
        )
        task_box.pack(side="left", fill="x", expand=True)
        task_box.bind("<<ComboboxSelected>>", self._show_task)

        self._task_frames: dict[str, ttk.LabelFrame] = {}
        self._build_selection_region()
        self._build_boolean_region()
        self._build_coordinates()
        self._build_output_request()
        self._build_units()
        self._show_task()

        self._operand_signature: tuple[object, ...] | None = None
        self._profile_signature: tuple[object, ...] | None = None
        self._output_signature: tuple[object, ...] | None = None
        self.refresh()

    def _task_frame(self, task: str, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(self, text=title, padding=8)
        self._task_frames[task] = frame
        return frame

    @staticmethod
    def _entry(parent: tk.Misc, label: str, default: str = ""):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=17).pack(side="left")
        variable = tk.StringVar(value=default)
        widget = ttk.Entry(row, textvariable=variable)
        widget.pack(side="left", fill="x", expand=True)
        return variable, widget

    @staticmethod
    def _vector_input(
        parent: tk.Misc,
        label: str,
        defaults: Sequence[str],
    ) -> tuple[ttk.Label, list[tk.StringVar]]:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        label_widget = ttk.Label(row, text=label, width=17)
        label_widget.pack(side="left")
        variables: list[tk.StringVar] = []
        for value in defaults:
            variable = tk.StringVar(value=value)
            ttk.Entry(row, textvariable=variable, width=7).pack(
                side="left", fill="x", expand=True, padx=1
            )
            variables.append(variable)
        return label_widget, variables

    def _build_selection_region(self) -> None:
        frame = self._task_frame(self.TASKS[0], "Named region")
        self._region_name, _widget = self._entry(frame, "Name", "Region-1")
        self._selection_summary = ttk.Label(
            frame,
            text="No selection",
            foreground="#555555",
            wraplength=310,
            justify="left",
        )
        self._selection_summary.pack(fill="x", pady=(6, 8))
        ttk.Label(
            frame,
            text=(
                "The active geometry or mesh filter defines the region type. "
                "All selected owners become one reusable scope."
            ),
            foreground="#666666",
            wraplength=310,
            justify="left",
        ).pack(fill="x", pady=(0, 8))
        self.button(frame, "Create region from selection", self._create_region)

    def _build_boolean_region(self) -> None:
        frame = self._task_frame(self.TASKS[1], "Boolean region")
        self._boolean_name, _widget = self._entry(frame, "Name", "Region-1")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Operation", width=17).pack(side="left")
        self._boolean_operation = tk.StringVar(value="Union")
        ttk.Combobox(
            row,
            textvariable=self._boolean_operation,
            values=("Union", "Intersection", "Subtract"),
            state="readonly",
        ).pack(side="left", fill="x", expand=True)

        ttk.Label(frame, text="Operands (Ctrl/Shift selects multiple)").pack(
            anchor="w", pady=(7, 2)
        )
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill="both", expand=True)
        self._operand_list = tk.Listbox(
            list_frame,
            selectmode="extended",
            exportselection=False,
            height=9,
        )
        scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self._operand_list.yview
        )
        self._operand_list.configure(yscrollcommand=scrollbar.set)
        self._operand_list.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self._operand_ids: tuple[str, ...] = ()
        self._operand_note = ttk.Label(frame, text="", foreground="#666666")
        self._operand_note.pack(fill="x", pady=(4, 4))
        self.button(frame, "Create Boolean region", self._create_boolean)

    def _build_coordinates(self) -> None:
        frame = self._task_frame(self.TASKS[2], "Named coordinate system")
        self._coordinate_name, _widget = self._entry(frame, "Name", "CSYS-1")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Type", width=17).pack(side="left")
        self._coordinate_kind = tk.StringVar(value="Cartesian")
        ttk.Combobox(
            row,
            textvariable=self._coordinate_kind,
            values=("Cartesian", "Cylindrical"),
            state="readonly",
        ).pack(side="left", fill="x", expand=True)
        self._origin_label, self._coordinate_origin = self._vector_input(
            frame, "Origin", ("0", "0", "0")
        )
        _axis_label, self._coordinate_axis = self._vector_input(
            frame, "z axis", ("0", "0", "1")
        )
        _reference_label, self._coordinate_reference = self._vector_input(
            frame, "x reference", ("1", "0", "0")
        )
        ttk.Label(
            frame,
            text=(
                "Axes are normalized and checked for a finite right-handed "
                "basis. Cylindrical x is the local radial direction."
            ),
            foreground="#666666",
            wraplength=310,
            justify="left",
        ).pack(fill="x", pady=(6, 8))
        self.button(frame, "Create coordinate system", self._create_coordinate)

    def _build_units(self) -> None:
        frame = self._task_frame(self.TASKS[4], "Entry and display units")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Profile", width=17).pack(side="left")
        self._profile_choice = tk.StringVar(value="SI-m-N-Pa")
        self._profile_box = ttk.Combobox(
            row,
            textvariable=self._profile_choice,
            values=tuple(UNIT_PROFILES) + ("Custom",),
            state="readonly",
        )
        self._profile_box.pack(side="left", fill="x", expand=True)
        self._profile_box.bind("<<ComboboxSelected>>", self._profile_selected)
        self._custom_profile_name, self._custom_name_entry = self._entry(
            frame, "Custom name", "Project units"
        )

        editor = ttk.Frame(frame)
        editor.pack(fill="x", pady=(5, 6))
        self._unit_variables: dict[str, tk.StringVar] = {}
        self._unit_boxes: dict[str, ttk.Combobox] = {}
        for index, dimension in enumerate(UnitProfile.REQUIRED):
            column = 0 if index < 5 else 2
            row_index = index if index < 5 else index - 5
            ttk.Label(
                editor,
                text=_UNIT_LABELS[dimension],
                width=15,
            ).grid(row=row_index, column=column, sticky="w", padx=(0, 2), pady=1)
            variable = tk.StringVar(value=_UNIT_CHOICES[dimension][0])
            box = ttk.Combobox(
                editor,
                textvariable=variable,
                values=_UNIT_CHOICES[dimension],
                state="readonly",
                width=8,
            )
            box.grid(row=row_index, column=column + 1, sticky="ew", padx=(0, 8))
            self._unit_variables[dimension] = variable
            self._unit_boxes[dimension] = box
        editor.columnconfigure(1, weight=1)
        editor.columnconfigure(3, weight=1)
        self._active_profile = ttk.Label(frame, text="", foreground="#555555")
        self._active_profile.pack(fill="x", pady=(1, 6))
        ttk.Label(
            frame,
            text="Stored and solver values remain SI; this changes entry and display only.",
            foreground="#666666",
            wraplength=310,
            justify="left",
        ).pack(fill="x", pady=(0, 8))
        self.button(frame, "Apply unit profile", self._apply_units)

    def _build_output_request(self) -> None:
        frame = self._task_frame(self.TASKS[3], "Typed output request")
        self._output_label, _widget = self._entry(
            frame, "Label", "Displacement"
        )
        self._output_quantities, _widget = self._entry(
            frame, "Quantities", "displacement"
        )

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Region", width=17).pack(side="left")
        self._output_region = tk.StringVar(value="")
        self._output_region_box = ttk.Combobox(
            row, textvariable=self._output_region, state="readonly"
        )
        self._output_region_box.pack(side="left", fill="x", expand=True)
        self._output_region_ids: dict[str, str] = {}

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Analysis", width=17).pack(side="left")
        self._output_analysis = tk.StringVar(value="Unassigned")
        self._output_analysis_box = ttk.Combobox(
            row, textvariable=self._output_analysis, state="readonly"
        )
        self._output_analysis_box.pack(side="left", fill="x", expand=True)
        self._output_analysis_ids: dict[str, str] = {}

        for attribute, label, values, default in (
            (
                "_output_location",
                "Location",
                ("node", "element", "element_face", "integration_point", "global", "history"),
                "node",
            ),
            ("_output_recovery", "Recovery", ("native", "nodal", "patch", "raw"), "native"),
            ("_output_reduction", "Reduction", ("none", "average", "max", "min", "abs_max", "envelope"), "none"),
            ("_output_basis", "Basis", ("global", "local", "element", "material"), "global"),
            ("_output_frames", "Frames", ("all", "first", "last", "selected", "envelope"), "all"),
        ):
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=17).pack(side="left")
            variable = tk.StringVar(value=default)
            setattr(self, attribute, variable)
            ttk.Combobox(
                row, textvariable=variable, values=values, state="readonly"
            ).pack(side="left", fill="x", expand=True)
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Frame indices (0+)", width=17).pack(side="left")
        self._output_frame_indices = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self._output_frame_indices).pack(
            side="left", fill="x", expand=True
        )
        ttk.Label(
            frame,
            text=(
                "Only quantities produced by the attached analysis are accepted. "
                "Unavailable fields are never created as zero-valued results."
            ),
            foreground="#666666",
            wraplength=310,
            justify="left",
        ).pack(fill="x", pady=(6, 8))
        self.button(frame, "Create output request", self._create_output_request)

    def _show_task(self, _event=None) -> None:
        selected = self._task.get()
        for name, frame in self._task_frames.items():
            if name == selected:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()

    def refresh(self) -> None:
        selection = self.app.selection
        count = len(selection.items)
        self._selection_summary.configure(
            text=(
                f"{count} {mode_label(selection.mode).lower()}"
                f"{'s' if count != 1 else ''} selected in "
                f"{selection.domain.value}"
            )
        )
        self._origin_label.configure(
            text=f"Origin [{self.app.project.units.symbol('length')}]"
        )
        self._refresh_operands()
        self._refresh_output_requests()
        self._refresh_profile()

    def _refresh_output_requests(self) -> None:
        regions = tuple(
            item for item in self.app.project.regions if not item.hidden
        )
        analyses = tuple(self.app.project.analyses.values())
        signature = (
            tuple((item.id, item.name) for item in regions),
            tuple((item.id, item.name, item.type) for item in analyses),
        )
        if signature == self._output_signature:
            return
        region_labels = tuple(
            f"{item.name} [{item.domain.value}/{item.entity_kind}]"
            for item in regions
        )
        self._output_region_ids = {
            label: item.id for label, item in zip(region_labels, regions)
        }
        self._output_region_box.configure(values=region_labels)
        if self._output_region.get() not in self._output_region_ids:
            self._output_region.set(region_labels[0] if region_labels else "")

        analysis_labels = tuple(
            f"{item.name} [{item.type.replace('_', ' ')}]" for item in analyses
        )
        self._output_analysis_ids = {
            label: item.id for label, item in zip(analysis_labels, analyses)
        }
        choices = ("Unassigned",) + analysis_labels
        self._output_analysis_box.configure(values=choices)
        if self._output_analysis.get() not in choices:
            self._output_analysis.set("Unassigned")
        self._output_signature = signature

    def _refresh_operands(self) -> None:
        regions = tuple(
            region for region in self.app.project.regions if not region.hidden
        )
        signature = tuple(
            (item.id, item.name, item.domain.value, item.entity_kind, item.mesh_id)
            for item in regions
        )
        if signature == self._operand_signature:
            return
        selected_ids = {
            self._operand_ids[index]
            for index in self._operand_list.curselection()
            if index < len(self._operand_ids)
        }
        visible = regions[:MAX_REGION_OPERANDS]
        self._operand_ids = tuple(item.id for item in visible)
        self._operand_list.delete(0, "end")
        for index, region in enumerate(visible):
            mesh_suffix = f" / {region.mesh_id[:8]}" if region.mesh_id else ""
            self._operand_list.insert(
                "end",
                f"{region.name}  [{region.domain.value}/{region.entity_kind}{mesh_suffix}]",
            )
            if region.id in selected_ids:
                self._operand_list.selection_set(index)
        omitted = len(regions) - len(visible)
        self._operand_note.configure(
            text=(
                f"Showing first {len(visible)} of {len(regions)} regions"
                if omitted
                else f"{len(regions)} reusable region(s)"
            )
        )
        self._operand_signature = signature

    def _refresh_profile(self) -> None:
        profile = self.app.project.units
        signature = (profile.name, tuple(sorted(profile.units.items())))
        self._active_profile.configure(text=f"Active: {profile.name}")
        if signature == self._profile_signature:
            return
        builtin = UNIT_PROFILES.get(profile.name)
        if builtin is not None and dict(builtin.units) == dict(profile.units):
            self._profile_choice.set(profile.name)
        else:
            self._profile_choice.set("Custom")
            self._custom_profile_name.set(profile.name)
        for dimension, variable in self._unit_variables.items():
            variable.set(profile.symbol(dimension))
        self._profile_signature = signature
        self._profile_selected()

    def _profile_selected(self, _event=None) -> None:
        choice = self._profile_choice.get()
        builtin = UNIT_PROFILES.get(choice)
        custom = builtin is None
        if builtin is not None:
            for dimension, variable in self._unit_variables.items():
                variable.set(builtin.symbol(dimension))
        self._custom_name_entry.configure(state="normal" if custom else "disabled")
        for box in self._unit_boxes.values():
            box.configure(state="readonly" if custom else "disabled")

    def _mesh_identity(self) -> tuple[str, cmd.AddMeshRecord | None]:
        project = self.app.project
        requested = str(getattr(self.app, "mesh_record_id", "") or "")
        if requested and requested in project.mesh_records:
            return requested, None
        if not requested and project.mesh_records:
            return next(reversed(project.mesh_records)), None
        mesh = getattr(self.app, "mesh", None)
        if mesh is None:
            raise ValueError("generate or import a mesh before creating a mesh region")

        from anymesher.serialize import mesh_to_dict
        from ..document import canonical_hash

        identifier = requested or str(uuid4())
        record = MeshRecord(
            id=identifier,
            name="Imported mesh" if getattr(self.app, "imported", None) is not None else "Mesh",
            kind="imported" if getattr(self.app, "imported", None) is not None else "generated",
            source_model_hash=str(
                getattr(
                    getattr(getattr(self.app, "session", None), "revision", None),
                    "model_hash",
                    "",
                )
            ),
            mesh_input_hash="",
            mesh_hash=canonical_hash(mesh_to_dict(mesh)),
            summary={"nodes": mesh.num_nodes, "elements": mesh.num_elements},
        )
        return identifier, cmd.AddMeshRecord(record)

    def _create_region(self) -> None:
        selection = self.app.selection
        mesh_id = None
        mesh_command = None
        if selection.domain.value == "mesh":
            mesh_id, mesh_command = self._mesh_identity()
        region = region_from_selection(
            self.app.project,
            self._region_name.get().strip(),
            selection.items,
            selection.mode,
            mesh_id=mesh_id,
        )
        region_command = cmd.AddRegion(region)
        if mesh_command is None:
            self.app.run(region_command)
        else:
            self.app.run(
                cmd.CompositeCommand(
                    (mesh_command, region_command), label="create mesh region"
                )
            )
        if mesh_id is not None:
            self.app.mesh_record_id = mesh_id
        self._region_name.set(
            _next_name(
                "Region", (item.name for item in self.app.project.regions)
            )
        )
        self.app.set_status(
            f"created region {region.name!r} with {len(selection.items)} target(s)"
        )

    def _create_boolean(self) -> None:
        indexes = tuple(int(value) for value in self._operand_list.curselection())
        operands = tuple(
            self.app.project.regions[self._operand_ids[index]] for index in indexes
        )
        region = boolean_region(
            self._boolean_name.get().strip(),
            self._boolean_operation.get().lower(),
            operands,
        )
        self.app.run(cmd.AddRegion(region, label="create Boolean region"))
        self._boolean_name.set(
            _next_name(
                "Region", (item.name for item in self.app.project.regions)
            )
        )
        self.app.set_status(
            f"created {self._boolean_operation.get().lower()} region {region.name!r}"
        )

    def _create_coordinate(self) -> None:
        system = coordinate_system_from_values(
            self._coordinate_name.get(),
            self._coordinate_kind.get(),
            [item.get() for item in self._coordinate_origin],
            [item.get() for item in self._coordinate_axis],
            [item.get() for item in self._coordinate_reference],
            self.app.project.units,
        )
        self.app.run(cmd.AddCoordinateSystem(system))
        self._coordinate_name.set(
            _next_name(
                "CSYS",
                (item.name for item in self.app.project.coordinate_systems.values()),
            )
        )
        self.app.set_status(f"created {system.kind} coordinates {system.name!r}")

    def _apply_units(self) -> None:
        choice = self._profile_choice.get()
        if choice in UNIT_PROFILES:
            profile = UNIT_PROFILES[choice]
        else:
            profile = unit_profile_from_values(
                self._custom_profile_name.get(),
                {
                    dimension: variable.get()
                    for dimension, variable in self._unit_variables.items()
                },
            )
        self.app.run(cmd.SetUnitProfile(profile))
        self.app.set_status(f"unit profile: {profile.name}")

    def _create_output_request(self) -> None:
        region_label = self._output_region.get()
        if region_label not in self._output_region_ids:
            raise ValueError("create a named region before requesting output")
        request = output_request_from_values(
            self._output_label.get(),
            self._output_quantities.get(),
            self._output_region_ids[region_label],
            self._output_location.get(),
            recovery=self._output_recovery.get(),
            reduction=self._output_reduction.get(),
            basis=self._output_basis.get(),
            frame_policy=self._output_frames.get(),
            frame_indices=self._output_frame_indices.get(),
        )
        analysis_id = self._output_analysis_ids.get(self._output_analysis.get())
        self.app.run(
            cmd.AddOutputRequest(
                request,
                () if analysis_id is None else (analysis_id,),
            )
        )
        self._output_label.set(
            _next_name(
                "Output",
                (item.label for item in self.app.project.output_requests.values()),
            )
        )
        suffix = "" if analysis_id is None else " and attached it to the analysis"
        self.app.set_status(f"created output request {request.label!r}{suffix}")
