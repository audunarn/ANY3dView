"""The stage panels: Geometry, Mesh, Loads & BC, Solve, Results, Visualization.

Every panel acts through the command stack, never directly on the project, so
everything the user does here is undoable and scriptable by the same calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from queue import Empty, Queue
from threading import Thread
import json
import tkinter as tk
import traceback
from tkinter import colorchooser, filedialog, ttk
from typing import Callable, List, Optional, Sequence

import numpy as np
from anymaterial import available_grades
from anygeometry.entities import EntityRef
from anygeometry import SketchDefinition, face_sketch_plane

from .. import commands as cmd
from ..geometry.construction import ConstructionMode, ConstructionTask
from ..geometry.sketching import FaceSketchTask
from ..geometry.snapping import (
    GeometrySnapData,
    SnapPoint,
    SnapSegment,
    geometry_snap_data,
)
from anymesher.decomposition import check_mappable
from ..mesh.mapped import ELEMENT_ORDERS
from ..mesh_controls import (
    MeshControls,
    StructuredMeshControls,
    native_v2_options_type,
)
from ..mesh.refinement import refine_around
from ..mesh.seeding import SeedingConflict
from ..model.attributes import (
    DOF_NAMES,
    LineLoad,
    Mass,
    PointLoad,
    Pressure,
    Support,
    SurfaceTraction,
)
from ..model.imperfections import Imperfection
from ..model.materials import Material, dnv_steel_material
from ..model.sections import PROFILES, BeamSection, PlateSection
from ..model.project import ProjectError
from ..model.workplanes import Workplane
from ..model.coordinates import CoordinateSystem
from ..post.extract import along_line, envelope, probe
from ..post.fields import available_fields
from ..post.history import history_series
from ..post.report import (
    field_to_csv,
    result_report_context,
    write_csv,
    write_report,
    write_result_report,
)
from ..selection import SELECTION_MODES, mode_label
from .plot import HistoryPlot
from .live_progress import GRAPH_CHOICES, LiveProgressData
from .result_display import (
    DISPLAY_UNIT_SYSTEMS,
    ENGINEERING_DISPLAY,
    converted_series,
    unit_transform,
)
from .result_export import lazy_field_to_csv, save_gif
from .result_summary import (
    nonlinear_path_summary,
    prescribed_path_progress,
    submitted_target_load_factor,
)
from .visualization import GEOMETRY_DETAILS, RENDER_MODES, VisualizationStyle
from .scene import (
    COLOR_LOAD,
    COLOR_MASS,
    COLOR_MOMENT,
    COLOR_PRESSURE,
    COLOR_ROTATION,
    COLOR_SUPPORT,
    RESULT_COLORMAPS,
)

__all__ = [
    "GeometryPanel",
    "LoadPanel",
    "MeshPanel",
    "ResultsPanel",
    "SectionPanel",
    "SolvePanel",
    "StagePanel",
    "VisualizationPanel",
    "mapped_mesh_eligibility",
]


def mapped_mesh_eligibility(geometry) -> tuple[bool, str]:
    """Return a conservative, user-facing mapped-method qualification.

    This is an early UI check over the live geometry.  The mesh worker repeats
    the authoritative checks after creating its detached structural closure,
    because intersection imprinting can partition otherwise mappable plates.
    """

    face_ids = tuple(sorted(int(item) for item in geometry.faces))
    if not face_ids:
        return (
            True,
            "There are no plate faces to qualify. A beam-only model is "
            "supported when section-assigned members exist; they are seeded "
            "along their model lines.",
        )

    problems: list[str] = []
    for face_id in face_ids:
        face = geometry.faces[face_id]
        holes = tuple(getattr(face, "holes", ()))
        if holes:
            problems.append(
                f"plate face {face_id} contains {len(holes)} hole loop(s)"
            )
            continue
        report = check_mappable(geometry, face_id)
        if not report.ok:
            detail = (
                report.messages[0]
                if report.messages
                else "its boundary cannot be arranged as four mapped sides"
            )
            problems.append(f"plate face {face_id}: {detail}")

    if problems:
        shown = "; ".join(problems[:3])
        remaining = len(problems) - 3
        if remaining > 0:
            shown += f"; and {remaining} more"
        return (
            False,
            "Mapped is unavailable for the current model: " + shown + ". "
            "Partition the indicated plates into four-sided regions, or choose "
            "Automatic/Unstructured.",
        )
    return (
        True,
        f"Mapped is available: all {len(face_ids)} plate face(s) have four "
        "mapped sides and no holes. Final eligibility is checked again after "
        "automatic intersection preparation.",
    )


class StagePanel(ttk.Frame):
    """Base class: a titled panel with small form helpers."""

    title = "Stage"

    def __init__(self, master: tk.Misc, app) -> None:
        super().__init__(master, padding=8)
        self.app = app
        workbench = getattr(app, "workbench", None)
        self.presenter = (
            None
            if workbench is None
            else workbench.presenter(
                self.title,
                status=getattr(app, "_status_port", None),
            )
        )
        self.build()

    # -- construction ---------------------------------------------------
    def build(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def refresh(self) -> None:
        """Called when the model or selection changes."""

    # -- form helpers ---------------------------------------------------
    def section(
        self, text: str, parent: tk.Misc | None = None
    ) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(self if parent is None else parent, text=text, padding=6)
        frame.pack(fill="x", pady=(0, 8))
        return frame

    @staticmethod
    def labelled_entry(
        parent: tk.Misc,
        label: str,
        default: str = "",
        width: int = 10,
        *,
        label_width: int = 16,
    ) -> tuple[ttk.Frame, tk.StringVar]:
        """An entry row, returning its frame so it can be shown or hidden."""

        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=label_width).pack(side="left")
        variable = tk.StringVar(value=default)
        ttk.Entry(row, textvariable=variable, width=width).pack(
            side="left", fill="x", expand=True
        )
        return row, variable

    @classmethod
    def entry_row(
        cls, parent: tk.Misc, label: str, default: str = "", width: int = 10
    ) -> tk.StringVar:
        return cls.labelled_entry(parent, label, default, width)[1]

    @staticmethod
    def labelled_vector(
        parent: tk.Misc, label: str, default: Sequence[str] = ("0", "0", "0")
    ) -> tuple[ttk.Frame, List[tk.StringVar]]:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=16).pack(side="left")
        variables = []
        for value in default:
            variable = tk.StringVar(value=value)
            ttk.Entry(row, textvariable=variable, width=7).pack(
                side="left", padx=1
            )
            variables.append(variable)
        return row, variables

    @classmethod
    def vector_row(
        cls, parent: tk.Misc, label: str, default: Sequence[str] = ("0", "0", "0")
    ) -> List[tk.StringVar]:
        return cls.labelled_vector(parent, label, default)[1]

    def button(
        self, parent: tk.Misc, text: str, action: Callable[[], None]
    ) -> ttk.Button:
        widget = ttk.Button(parent, text=text, command=self.guarded(action))
        widget.pack(fill="x", pady=2)
        return widget

    def guarded(self, action: Callable[[], None]) -> Callable[[], None]:
        """Run an action, turning any modelling error into a status message."""

        def wrapped() -> None:
            try:
                action()
            except Exception as error:
                # GeometryError and SeedingConflict both derive from
                # ValueError, so a modelling refusal reaches the status bar
                # rather than a traceback.  Unexpected callback errors are
                # captured too, so Copy diagnosis contains enough evidence to
                # reproduce them instead of losing them in stderr.
                if self.presenter is not None:
                    self.presenter.report_error(error)
                else:
                    self.app.set_status(
                        f"{type(error).__name__}: {error}",
                        error=True,
                        diagnostic={
                            "type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        },
                    )

        return wrapped

    # -- parsing --------------------------------------------------------
    @staticmethod
    def number(variable: tk.StringVar, label: str) -> float:
        text = variable.get().strip()
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{label} must be a number, got {text!r}") from None

    def vector(self, variables: Sequence[tk.StringVar], label: str) -> np.ndarray:
        return np.array(
            [self.number(variable, label) for variable in variables], dtype=float
        )

    # -- selection ------------------------------------------------------
    def require_selection(self, kind: str, count: Optional[int] = None) -> List[EntityRef]:
        if self.presenter is not None:
            return self.presenter.require_selection(kind, count)
        selection = self.app.selection
        if selection.mode != kind:
            raise ValueError(
                f"switch to {mode_label(kind)} mode first "
                f"(currently {mode_label(selection.mode)})"
            )
        items = selection.items
        if not items:
            raise ValueError(f"select at least one {mode_label(kind).lower()} first")
        if count is not None and len(items) != count:
            raise ValueError(
                f"select exactly {count} {mode_label(kind).lower()}s "
                f"({len(items)} selected)"
            )
        return items


# ----------------------------------------------------------------------
class GeometryPanel(StagePanel):
    title = "Geometry"

    def build(self) -> None:
        ttk.Label(
            self,
            text=(
                "Create model geometry here. Point / Line / Plate selection is "
                "controlled by the persistent selection strip above."
            ),
            foreground="#555555",
            wraplength=360,
        ).pack(fill="x", padx=6, pady=(5, 3))

        self._geometry_tabs = ttk.Notebook(self)
        self._geometry_tabs.pack(fill="both", expand=True, padx=3, pady=(0, 4))
        guiding_page = ttk.Frame(self._geometry_tabs)
        operations_page = ttk.Frame(self._geometry_tabs)
        plates_page = ttk.Frame(self._geometry_tabs)
        beams_page = ttk.Frame(self._geometry_tabs)
        self._geometry_pages = {
            "Guiding geometry": guiding_page,
            "Operations": operations_page,
            "Plates": plates_page,
            "Beams": beams_page,
        }
        for name, page in self._geometry_pages.items():
            self._geometry_tabs.add(page, text=name)
        ttk.Label(
            guiding_page,
            text=(
                "Points, curves and workplane sketches define construction intent; "
                "they become structural members only after a section is assigned."
            ),
            foreground="#666666",
            wraplength=330,
        ).pack(fill="x", padx=6, pady=(5, 1))

        point = self.section("Points", guiding_page)
        self._point = self.vector_row(point, "x, y, z")
        self.button(point, "Add point", self._add_point)

        construction = self.section("Workplanes / sketch construction", guiding_page)
        row = ttk.Frame(construction)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="coordinates", width=16).pack(side="left")
        self._workplane_coordinates = tk.StringVar(value="Global")
        self._workplane_coordinates_box = ttk.Combobox(
            row,
            textvariable=self._workplane_coordinates,
            state="readonly",
            width=16,
        )
        self._workplane_coordinates_box.pack(side="left", fill="x", expand=True)
        self._workplane_offset = self.entry_row(construction, "plane offset", "0 m")
        self._workplane_grid = self.entry_row(construction, "grid spacing", "1 m")
        self._workplane_tolerance = self.entry_row(
            construction, "snap tolerance", "50 mm"
        )
        snaps = ttk.Frame(construction)
        snaps.pack(fill="x", pady=1)
        self._snap_grid = tk.BooleanVar(value=True)
        self._snap_object = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            snaps, text="Grid/axes", variable=self._snap_grid
        ).pack(side="left")
        ttk.Checkbutton(
            snaps, text="End/mid/intersection", variable=self._snap_object
        ).pack(side="left", padx=(8, 0))
        row = ttk.Frame(construction)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="construct", width=16).pack(side="left")
        self._construction_mode = tk.StringVar(value="Polyline")
        ttk.Combobox(
            row,
            textvariable=self._construction_mode,
            values=("Point", "Line", "Polyline"),
            state="readonly",
            width=12,
        ).pack(side="left", fill="x", expand=True)
        self._construction_close = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row, text="Close", variable=self._construction_close
        ).pack(side="left", padx=(6, 0))
        self.button(construction, "Start click construction", self._start_construction)
        actions = ttk.Frame(construction)
        actions.pack(fill="x", pady=(2, 0))
        ttk.Button(
            actions, text="Apply / Enter", command=self.guarded(self._apply_construction)
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            actions, text="Cancel / Esc", command=self._cancel_construction
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))
        ttk.Label(
            construction,
            text="LMB adds snapped points; numeric x/y/z entry above remains available.",
            foreground="#666666",
            wraplength=270,
        ).pack(anchor="w", pady=(3, 0))

        face_sketch = self.section("Editable sketch on a flat plate", guiding_page)
        ttk.Label(
            face_sketch,
            text=(
                "Select one Model Geometry Plate, then click Start. The plate "
                "becomes an unlimited gridded sketch plane; points may lie "
                "inside, outside or coincident with its boundary."
            ),
            foreground="#555555",
            wraplength=330,
        ).pack(fill="x", pady=(0, 3))
        self._face_sketch_grid = self.entry_row(face_sketch, "grid spacing", "0.25 m")
        self._face_sketch_extrusion = self.entry_row(face_sketch, "extrusion", "1 m")
        self._face_sketch_closed = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            face_sketch, text="Closed profile", variable=self._face_sketch_closed
        ).pack(anchor="w")
        self.button(face_sketch, "Start sketch on selected plate", self._start_face_sketch)
        self._face_sketch_pair = self.entry_row(face_sketch, "point numbers", "1 2")
        self._face_sketch_distance = self.entry_row(face_sketch, "distance", "1 m")
        constraint_actions = ttk.Frame(face_sketch)
        constraint_actions.pack(fill="x", pady=1)
        ttk.Button(
            constraint_actions,
            text="Set distance",
            command=self.guarded(self._add_sketch_distance),
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            constraint_actions,
            text="Coincident",
            command=self.guarded(self._add_sketch_coincidence),
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))
        edit_actions = ttk.Frame(face_sketch)
        edit_actions.pack(fill="x", pady=1)
        ttk.Button(
            edit_actions, text="Remove last point", command=self._remove_sketch_point
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            edit_actions,
            text="Remove last constraint",
            command=self._remove_sketch_constraint,
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))
        apply_actions = ttk.Frame(face_sketch)
        apply_actions.pack(fill="x", pady=(2, 0))
        ttk.Button(
            apply_actions,
            text="Apply sketch / extrusion",
            command=self.guarded(self._apply_face_sketch),
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            apply_actions, text="Cancel", command=self._cancel_face_sketch
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))
        self._face_sketch_status = ttk.Label(
            face_sketch,
            text="No face sketch active",
            foreground="#666666",
            wraplength=330,
        )
        self._face_sketch_status.pack(fill="x", pady=(3, 0))
        ttk.Label(
            face_sketch,
            text=(
                "Point numbers follow click order. For an open path whose last "
                "point is coincident with point 1, clear Closed profile."
            ),
            foreground="#777777",
            wraplength=330,
        ).pack(fill="x", pady=(2, 0))
        self._editing_sketch_feature_id: int | None = None
        self._sketch_support_ref: EntityRef | None = None

        line = self.section("Guiding curves", guiding_page)
        self.button(line, "Line through 2 points", self._add_line)
        self.button(line, "Arc through 3 points", self._add_arc)
        self.button(line, "Polyline through selected points", self._add_polyline)
        ttk.Label(
            line,
            text="Arc order: start, via, end",
            foreground="#666666",
        ).pack(anchor="w")

        plate = self.section("Plate boundaries", plates_page)
        self.button(plate, "Plate from selected lines", self._add_face)
        self.button(plate, "Plate from selected points", self._add_plate)

        extrude = self.section("Swept plates", plates_page)
        self._extrude = self.vector_row(extrude, "extrude vector", ("0", "0", "1"))
        self.button(extrude, "Extrude selected lines", self._extrude_lines)
        self._axis_point = self.vector_row(extrude, "axis point")
        self._axis_direction = self.vector_row(
            extrude, "axis direction", ("0", "0", "1")
        )
        self._angle = self.entry_row(extrude, "angle [deg]", "360")
        self.button(extrude, "Revolve selected lines", self._revolve_lines)

        duplicate = self.section("Copy, mirror and pattern", operations_page)
        self._copy_offset = self.vector_row(duplicate, "copy offset", ("1", "0", "0"))
        self.button(duplicate, "Copy selected", self._copy_selected)
        self._mirror_point = self.vector_row(duplicate, "plane point")
        self._mirror_normal = self.vector_row(
            duplicate, "plane normal", ("1", "0", "0")
        )
        self.button(duplicate, "Mirror selected", self._mirror_selected)
        self._linear_direction = self.vector_row(
            duplicate, "linear direction", ("1", "0", "0")
        )
        self._linear_spacing = self.entry_row(duplicate, "spacing [m]", "1")
        self._linear_count = self.entry_row(duplicate, "additional copies", "2")
        self.button(duplicate, "Linear pattern", self._linear_pattern)
        self._pattern_axis_point = self.vector_row(duplicate, "axis point")
        self._pattern_axis_direction = self.vector_row(
            duplicate, "axis direction", ("0", "0", "1")
        )
        self._pattern_angle = self.entry_row(duplicate, "angle step [deg]", "90")
        self._pattern_count = self.entry_row(duplicate, "additional copies", "3")
        self.button(duplicate, "Circular pattern", self._circular_pattern)

        generators = self.section("Plate and shell generators", plates_page)
        row = ttk.Frame(generators)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="type", width=16).pack(side="left")
        self._generator_type = tk.StringVar(value="Plate")
        self._generator_type_box = ttk.Combobox(
            row,
            textvariable=self._generator_type,
            values=("Plate", "Bulkhead", "Frame", "Stiffened panel", "Cylinder", "Cone"),
            state="readonly",
            width=18,
        )
        self._generator_type_box.pack(side="left", fill="x", expand=True)
        self._generator_rows: dict[str, ttk.Frame] = {}

        def generator_entry(key: str, label: str, default: str) -> tk.StringVar:
            made_row, variable = self.labelled_entry(generators, label, default)
            self._generator_rows[key] = made_row
            return variable

        def generator_vector(
            key: str, label: str, default: Sequence[str] = ("0", "0", "0")
        ) -> List[tk.StringVar]:
            made_row, variables = self.labelled_vector(generators, label, default)
            self._generator_rows[key] = made_row
            return variables

        self._generator_length = generator_entry("length", "length / height [m]", "4")
        self._generator_width = generator_entry("width", "panel width [m]", "2")
        self._generator_radius_start = generator_entry("radius_start", "radius [m]", "1")
        self._generator_radius_end = generator_entry("radius_end", "end radius [m]", "0.5")
        self._generator_longitudinal = generator_entry(
            "longitudinal", "longitudinal spacing", "0.5"
        )
        self._generator_transverse = generator_entry(
            "transverse", "transverse / ring spacing", "1"
        )
        self._generator_segments = generator_entry(
            "segments", "circumferential segments", "12"
        )
        self._generator_group = generator_entry("group", "semantic group", "shell")
        self._generator_origin = generator_vector("origin", "origin")
        self._generator_u = generator_vector(
            "u", "length direction", ("1", "0", "0")
        )
        self._generator_v = generator_vector(
            "v", "width direction", ("0", "1", "0")
        )
        self._generator_axis = generator_vector(
            "axis", "axis", ("0", "0", "1")
        )
        self._generator_action = self.button(
            generators, "Create generator feature", self._add_generator
        )
        self._generator_explanation = ttk.Label(
            generators, foreground="#555555", wraplength=330, justify="left"
        )
        self._generator_explanation.pack(fill="x", pady=(5, 2))
        self._generator_diagram = tk.Canvas(
            generators,
            height=125,
            background="#f7f8fa",
            highlightthickness=1,
            highlightbackground="#c8ccd2",
        )
        self._generator_diagram.pack(fill="x", pady=(0, 2))
        self._generator_type_box.bind(
            "<<ComboboxSelected>>", self._generator_type_changed, add="+"
        )
        self._generator_diagram.bind(
            "<Configure>", lambda _event: self._draw_generator_diagram(), add="+"
        )
        self._update_generator_form()

        beam_paths = self.section("Beam centre-lines", beams_page)
        ttk.Label(
            beam_paths,
            text=(
                "Edges are beam centre-lines. Existing plate edges may also receive "
                "a beam section, so connected beams and plates share topology."
            ),
            foreground="#666666",
            wraplength=330,
        ).pack(fill="x", pady=(0, 3))
        self.button(beam_paths, "Straight beam through 2 points", self._add_line)
        self.button(beam_paths, "Curved beam through 3 points", self._add_arc)
        self._beam_path_close = tk.BooleanVar(value=False)
        close_row = ttk.Frame(beam_paths)
        close_row.pack(fill="x", pady=1)
        ttk.Checkbutton(
            close_row, text="Close centre-line loop", variable=self._beam_path_close
        ).pack(side="left")
        self.button(
            beam_paths,
            "Beam chain through selected points",
            self._add_beam_polyline,
        )

        beam_generator = self.section("Girder / stiffener generator", beams_page)
        row = ttk.Frame(beam_generator)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="type", width=16).pack(side="left")
        self._beam_generator_type = tk.StringVar(value="Girder")
        ttk.Combobox(
            row,
            textvariable=self._beam_generator_type,
            values=("Girder", "Stiffener"),
            state="readonly",
            width=18,
        ).pack(side="left", fill="x", expand=True)
        self._beam_generator_length = self.entry_row(
            beam_generator, "length [m]", "4"
        )
        self._beam_generator_origin = self.vector_row(beam_generator, "origin")
        self._beam_generator_direction = self.vector_row(
            beam_generator, "direction", ("1", "0", "0")
        )
        self.button(
            beam_generator,
            "Create beam generator feature",
            self._add_beam_generator,
        )

        divide = self.section("Cut and partition", operations_page)
        self._fraction = self.entry_row(divide, "fraction", "0.5")
        self.button(divide, "Split selected lines", self._split_lines)
        row = ttk.Frame(divide)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="direction", width=16).pack(side="left")
        self._axis = tk.StringVar(value="0")
        for value, text in (("0", "sides 0-2"), ("1", "sides 1-3")):
            ttk.Radiobutton(
                row, text=text, value=value, variable=self._axis
            ).pack(side="left", expand=True)
        self.button(divide, "Split selected plates", self._split_plates)
        self._strips = self.entry_row(divide, "strips", "3")
        self.button(divide, "Strip selected plates", self._strip_plates)
        self.button(divide, "Three-sided region to plates", self._triangle)
        ttk.Label(
            divide,
            text=(
                "Coplanar overlap: select plates in order. The union is split "
                "into separate A-only, common and B-only plates; the first "
                "selected plate owns each common fragment."
            ),
            foreground="#666666",
            wraplength=330,
        ).pack(fill="x", pady=(5, 1))
        self.button(
            divide, "Fragment plate overlaps", self._fragment_plate_overlaps
        )

        trim = self.section("Modeling: neutral trim hole", operations_page)
        self._trim_centre = self.vector_row(trim, "centre")
        self._trim_radius = self.entry_row(trim, "radius [m]", "0.25")
        self.button(trim, "Create trimmed face hole", self._trim_hole)
        ttk.Label(
            trim,
            text="Keeps one structural face; creates an inner boundary.",
            foreground="#666666",
            wraplength=270,
        ).pack(anchor="w")

        hole = self.section("Mesh: butterfly decomposition", operations_page)
        self._hole_centre = self.vector_row(hole, "centre")
        self._hole_radius = self.entry_row(hole, "radius [m]", "0.25")
        self.button(hole, "Create butterfly patches", self._punch)

        edit = self.section("Inspect and edit", operations_page)
        self._corners = self.entry_row(edit, "corners", "0 1 2 3")
        self.button(edit, "Set plate corners", self._set_corners)
        self.button(edit, "Reverse line / plate orientation", self._reverse)
        row = ttk.Frame(edit)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="measurement", width=16).pack(side="left")
        self._measurement = tk.StringVar(value="Auto")
        ttk.Combobox(
            row,
            textvariable=self._measurement,
            values=(
                "Auto", "Position", "Coordinates", "Distance", "Angle",
                "Radius", "Length", "Area", "Perimeter", "Centroid", "Normal",
            ),
            state="readonly",
            width=14,
        ).pack(side="left", fill="x", expand=True)
        self.button(edit, "Measure selection", self._measure)
        self.button(edit, "Check selected plates", self._check)
        self.button(edit, "Delete selection", self._delete)
        self._geometry_tabs.bind(
            "<<NotebookTabChanged>>", self._geometry_tab_changed, add="+"
        )

    def refresh(self) -> None:
        systems = sorted(
            self.app.project.coordinate_systems.values(),
            key=lambda item: (item.id != "global", item.name.lower(), item.id),
        )
        values = tuple(item.name for item in systems)
        self._workplane_coordinates_box.configure(values=values)
        if self._workplane_coordinates.get() not in values and values:
            self._workplane_coordinates.set(values[0])

    @staticmethod
    def generator_form_spec(kind: str) -> tuple[tuple[str, ...], str]:
        """Applicable generator fields and the engineer-facing explanation."""

        specifications = {
            "Plate": (
                ("length", "width", "group", "origin", "u", "v"),
                "Creates one flat rectangular structural plate from the origin and two in-plane directions.",
            ),
            "Bulkhead": (
                ("length", "width", "group", "origin", "u", "v"),
                "Creates one flat bulkhead plate. Length and width follow the two displayed direction vectors.",
            ),
            "Frame": (
                ("length", "width", "group", "origin", "u", "v"),
                "Creates a flat frame-plane plate region; assign structural sections after creation.",
            ),
            "Stiffened panel": (
                ("length", "width", "longitudinal", "transverse", "group", "origin"),
                "Creates a stiffened plate with semantic longitudinal stiffener and transverse-member lines at the requested spacing.",
            ),
            "Cylinder": (
                ("length", "radius_start", "longitudinal", "transverse", "segments", "origin", "axis"),
                "Creates a segmented cylindrical shell around the axis. Circumferential segments control the faceted ring.",
            ),
            "Cone": (
                ("length", "radius_start", "radius_end", "longitudinal", "transverse", "segments", "origin", "axis"),
                "Creates a segmented conical shell from the start radius to the end radius along the axis.",
            ),
        }
        try:
            return specifications[str(kind)]
        except KeyError:
            raise ValueError(f"unknown structural generator {kind!r}") from None

    def _generator_type_changed(self, _event=None) -> None:
        self._update_generator_form()

    def _update_generator_form(self) -> None:
        visible, explanation = self.generator_form_spec(self._generator_type.get())
        for row in self._generator_rows.values():
            row.pack_forget()
        for key in visible:
            self._generator_rows[key].pack(
                fill="x", pady=1, before=self._generator_action
            )
        self._generator_explanation.configure(text=explanation)
        self._draw_generator_diagram()

    def _draw_generator_diagram(self) -> None:
        if not hasattr(self, "_generator_diagram"):
            return
        canvas = self._generator_diagram
        canvas.delete("all")
        width = max(int(canvas.winfo_width()), 300)
        height = max(int(canvas.winfo_height()), 125)
        kind = self._generator_type.get()
        ink = "#35566f"
        fill = "#a9c8dd"
        accent = "#d36b32"
        origin_colour = "#c62828"

        def origin_mark(x: float, y: float) -> None:
            size = 5
            canvas.create_line(
                x - size, y - size, x + size, y + size,
                fill=origin_colour, width=2,
            )
            canvas.create_line(
                x - size, y + size, x + size, y - size,
                fill=origin_colour, width=2,
            )
            canvas.create_text(
                x + 8, y - 7, text="origin", anchor="sw", fill=origin_colour
            )

        def dimension(
            start: tuple[float, float],
            end: tuple[float, float],
            text: str,
            *,
            offset: tuple[float, float] = (0, 0),
        ) -> None:
            canvas.create_line(*start, *end, fill=ink, arrow="both")
            middle = (
                0.5 * (start[0] + end[0]) + offset[0],
                0.5 * (start[1] + end[1]) + offset[1],
            )
            canvas.create_text(*middle, text=text, fill=ink)

        if kind in {"Plate", "Bulkhead", "Frame", "Stiffened panel"}:
            polygon = (45, 88, width - 65, 88, width - 30, 35, 80, 35)
            canvas.create_polygon(*polygon, fill=fill, outline=ink, width=2)
            if kind == "Frame":
                canvas.create_polygon(
                    91, 70, width - 85, 70, width - 66, 49, 106, 49,
                    fill="#f7f8fa", outline=ink,
                )
            elif kind == "Stiffened panel":
                for fraction in (0.25, 0.5, 0.75):
                    x0 = 45 + fraction * (width - 110)
                    x1 = 80 + fraction * (width - 110)
                    canvas.create_line(x0, 88, x1, 35, fill=accent, width=2)
                canvas.create_line(62, 62, width - 47, 62, fill=accent, width=2)
            if kind == "Bulkhead":
                canvas.create_line(45, 88, 45, 105, fill=accent, width=2)
                canvas.create_text(52, 105, text="vertical", anchor="w", fill=accent)
            canvas.create_text(12, 12, text=kind, anchor="nw", fill=ink)
            origin_mark(45, 88)
            dimension(
                (45, 104), (width - 65, 104), "length", offset=(0, 8)
            )
            dimension(
                (35, 86), (70, 33), "width", offset=(-16, 0)
            )
        else:
            top_y, bottom_y = 32, 92
            start_half = 48
            end_half = 48 if kind == "Cylinder" else 28
            centre = width // 2
            canvas.create_polygon(
                centre - start_half, bottom_y,
                centre + start_half, bottom_y,
                centre + end_half, top_y,
                centre - end_half, top_y,
                fill=fill, outline=ink, width=2,
            )
            canvas.create_oval(
                centre - start_half, bottom_y - 10,
                centre + start_half, bottom_y + 10,
                outline=ink, width=2,
            )
            canvas.create_oval(
                centre - end_half, top_y - 8,
                centre + end_half, top_y + 8,
                outline=ink, width=2,
            )
            canvas.create_line(centre, 108, centre, 15, fill=accent, arrow="last")
            canvas.create_text(centre + 8, 16, text="axis", anchor="nw", fill=accent)
            canvas.create_text(12, 12, text=kind, anchor="nw", fill=ink)
            origin_mark(centre, bottom_y)
            dimension(
                (centre - start_half - 16, bottom_y),
                (centre - start_half - 16, top_y),
                "height",
                offset=(-20, 0),
            )
            dimension(
                (centre, bottom_y + 14),
                (centre + start_half, bottom_y + 14),
                "start radius" if kind == "Cone" else "radius",
                offset=(0, 8),
            )
            if kind == "Cone":
                dimension(
                    (centre, top_y - 13),
                    (centre + end_half, top_y - 13),
                    "end radius",
                    offset=(0, -7),
                )

    def _fragment_plate_overlaps(self) -> None:
        faces = self.require_selection("face")
        if len(faces) < 2:
            raise ValueError("select at least two overlapping plates in ownership order")
        feature = self.app.run(
            cmd.FragmentPlateOverlaps(tuple(item.id for item in faces))
        )
        overlap_count = sum(key.startswith("overlap/") for key in feature.outputs)
        self.app.set_status(
            f"fragmented overlap into {len(feature.outputs)} non-overlapping "
            f"plate(s); {overlap_count} common plate(s) owned by the first selection"
        )

    def _geometry_tab_changed(self, _event=None) -> None:
        selected = self._geometry_tabs.select()
        if not selected:
            return
        name = self._geometry_tabs.tab(selected, "text")
        if name != "Guiding geometry" and self.app.viewport.cancel_construction():
            self.app.set_status(
                "click construction cancelled; Geometry tab changed"
            )

    def _geometry_selection(self) -> List[EntityRef]:
        mode = self.app.selection.mode
        if mode not in SELECTION_MODES:
            raise ValueError(
                "this modeling tool needs geometry selection; switch the "
                "selection domain from Mesh to Geometry"
            )
        return self.require_selection(mode)

    @staticmethod
    def _whole(value: float, label: str, *, minimum: int = 1) -> int:
        integer = int(value)
        if float(integer) != float(value) or integer < minimum:
            raise ValueError(f"{label} must be a whole number at least {minimum}")
        return integer

    def _optional_number(self, variable: tk.StringVar, label: str) -> float | None:
        return None if not variable.get().strip() else self.number(variable, label)

    def _copy_selection(self, result, originals: Sequence[EntityRef]) -> None:
        made = [result.entity_map[item] for item in originals if item in result.entity_map]
        if made:
            self.app.selection.restore(made)

    def _copy_selected(self) -> None:
        items = self._geometry_selection()
        result = self.app.run(
            cmd.CopyEntities(items, translation=self.vector(self._copy_offset, "copy offset"))
        )
        self._copy_selection(result, items)
        self.app.set_status(f"copied {len(items)} selected entity(ies)")

    def _mirror_selected(self) -> None:
        items = self._geometry_selection()
        result = self.app.run(
            cmd.MirrorEntities(
                items,
                plane_point=self.vector(self._mirror_point, "mirror plane point"),
                plane_normal=self.vector(self._mirror_normal, "mirror plane normal"),
            )
        )
        self._copy_selection(result, items)
        self.app.set_status(f"mirrored {len(items)} selected entity(ies)")

    def _linear_pattern(self) -> None:
        items = self._geometry_selection()
        count = self._whole(
            self.number(self._linear_count, "additional copies"),
            "additional copies",
        )
        result = self.app.run(
            cmd.LinearPattern(
                items,
                direction=self.vector(self._linear_direction, "pattern direction"),
                spacing=self.number(self._linear_spacing, "pattern spacing"),
                count=count,
            )
        )
        made = [
            instance.entity_map[item]
            for instance in result.instances
            for item in items
            if item in instance.entity_map
        ]
        if made:
            self.app.selection.restore(made)
        self.app.set_status(f"created {count} linear pattern instance(s)")

    def _circular_pattern(self) -> None:
        items = self._geometry_selection()
        count = self._whole(
            self.number(self._pattern_count, "additional copies"),
            "additional copies",
        )
        result = self.app.run(
            cmd.CircularPattern(
                items,
                axis_point=self.vector(self._pattern_axis_point, "pattern axis point"),
                axis_direction=self.vector(
                    self._pattern_axis_direction, "pattern axis direction"
                ),
                angle_step=float(
                    np.deg2rad(self.number(self._pattern_angle, "pattern angle step"))
                ),
                count=count,
            )
        )
        made = [
            instance.entity_map[item]
            for instance in result.instances
            for item in items
            if item in instance.entity_map
        ]
        if made:
            self.app.selection.restore(made)
        self.app.set_status(f"created {count} circular pattern instance(s)")

    def _add_generator(self) -> None:
        kind = self._generator_type.get()
        length = self.number(self._generator_length, "length / height")
        origin = self.vector(self._generator_origin, "generator origin")
        spacing = self._optional_number(
            self._generator_longitudinal, "longitudinal spacing"
        )
        transverse = self._optional_number(
            self._generator_transverse, "transverse / ring spacing"
        )
        if kind in {"Plate", "Bulkhead", "Frame"}:
            requested_group = self._generator_group.get().strip()
            semantic_group = (
                kind.casefold()
                if kind in {"Bulkhead", "Frame"}
                and requested_group in {"", "shell"}
                else (requested_group or "shell")
            )
            parameters = {
                "length": length,
                "width": self.number(self._generator_width, "plate width"),
                "origin": tuple(origin),
                "u_direction": tuple(
                    self.vector(self._generator_u, "length direction")
                ),
                "v_direction": tuple(
                    self.vector(self._generator_v, "width direction")
                ),
                "semantic_group": semantic_group,
            }
            command = cmd.AddFeature(
                f"generator.{kind.casefold()}",
                name=kind,
                parameters=parameters,
                label=f"add {kind.casefold()}",
            )
        elif kind == "Stiffened panel":
            if spacing is None:
                raise ValueError("a stiffened panel needs longitudinal spacing")
            command = cmd.AddStiffenedPanel(
                length,
                self.number(self._generator_width, "panel width"),
                spacing,
                transverse,
                origin=origin,
                semantic_group=self._generator_group.get().strip() or "shell",
            )
        else:
            segments = self._whole(
                self.number(self._generator_segments, "circumferential segments"),
                "circumferential segments",
                minimum=3,
            )
            axis = self.vector(self._generator_axis, "generator axis")
            shared = dict(
                height=length,
                circumferential_segments=segments,
                longitudinal_spacing=spacing,
                ring_spacing=transverse,
                origin=origin,
                axis=axis,
            )
            if kind == "Cylinder":
                command = cmd.AddCylinder(
                    self.number(self._generator_radius_start, "radius"), **shared
                )
            elif kind == "Cone":
                command = cmd.AddCone(
                    self.number(self._generator_radius_start, "start radius"),
                    self.number(self._generator_radius_end, "end radius"),
                    **shared,
                )
            else:  # defensive against a scripted variable value
                raise ValueError(f"unknown structural generator {kind!r}")
        feature = self.app.run(command)
        made = len(feature.outputs)
        self.app.set_status(
            f"created {feature.name} as editable feature {feature.feature_id} "
            f"({made} semantic outputs)"
        )

    def _add_beam_generator(self) -> None:
        kind = self._beam_generator_type.get().strip()
        if kind not in {"Girder", "Stiffener"}:
            raise ValueError(f"unknown beam generator {kind!r}")
        command = cmd.AddFeature(
            f"generator.{kind.casefold()}",
            name=kind,
            parameters={
                "length": self.number(
                    self._beam_generator_length, "beam generator length"
                ),
                "origin": tuple(
                    self.vector(self._beam_generator_origin, "beam generator origin")
                ),
                "direction": tuple(
                    self.vector(
                        self._beam_generator_direction,
                        "beam generator direction",
                    )
                ),
            },
            label=f"add {kind.casefold()}",
        )
        feature = self.app.run(command)
        self.app.set_status(
            f"created {kind.casefold()} centre-line as editable feature "
            f"{feature.feature_id}; assign a beam section in Sections"
        )

    def _add_point(self) -> None:
        x, y, z = self.vector(self._point, "point coordinate")
        vertex = self.app.run(cmd.AddPoint(float(x), float(y), float(z)))
        self.app.set_status(f"added point {vertex}")

    def _selected_coordinate_system_id(self) -> str:
        name = self._workplane_coordinates.get().strip()
        matches = [
            item.id
            for item in self.app.project.coordinate_systems.values()
            if item.name == name
        ]
        if len(matches) != 1:
            raise ValueError(f"coordinate system {name!r} is unavailable")
        return matches[0]

    def _workplane(self) -> Workplane:
        units = self.app.project.units
        objects = bool(self._snap_object.get())
        return Workplane(
            coordinate_system_id=self._selected_coordinate_system_id(),
            offset=units.parse(self._workplane_offset.get(), "length"),
            grid_spacing=units.parse(self._workplane_grid.get(), "length"),
            snap_tolerance=units.parse(self._workplane_tolerance.get(), "length"),
            snap_grid=bool(self._snap_grid.get()),
            snap_axes=bool(self._snap_grid.get()),
            snap_endpoints=objects,
            snap_midpoints=objects,
            snap_intersections=objects,
        )

    def _start_construction(self) -> None:
        close = bool(self._construction_close.get())
        mode = ConstructionMode(self._construction_mode.get().strip().lower())
        task = ConstructionTask(mode, close=close)
        workplane = self._workplane()
        if self.app.viewport.construction_active:
            self.app.viewport.cancel_construction()
        self.app.session.active_workplane = workplane
        self.app.viewport.begin_construction(
            task,
            workplane,
            self.app.project.coordinate_systems,
            snap_data=lambda: geometry_snap_data(self.app.project.geometry),
            update_handler=self._construction_updated,
            apply_handler=self.guarded(self._apply_construction),
        )
        self.app.set_status(
            f"{mode.value} construction active: click in the viewport; "
            "Enter applies and Escape cancels"
        )

    def _construction_updated(self, task: ConstructionTask, snap) -> None:
        if isinstance(task, FaceSketchTask):
            self._update_face_sketch_status(task, snap)
            return
        if task.cancelled:
            self.app.set_status("construction cancelled; model unchanged")
            return
        kind = "free"
        if snap is not None and snap.kind is not None:
            kind = snap.kind.value
        ready = "; ready to Apply" if task.ready else ""
        self.app.set_status(
            f"construction point {len(task.points)} ({kind} snap){ready}"
        )

    def _apply_construction(self) -> None:
        task = self.app.viewport.construction_task
        if task is None:
            raise ValueError("start click construction first")
        result = self.app.viewport.finish_construction(self.app.run)
        made = tuple(result.vertices) + tuple(result.edges)
        if made:
            # Prefer newly made lines when the active geometry filter allows
            # them; otherwise keep the new points selected.
            mode = "edge" if result.edges else "vertex"
            self.app.selection.set_mode(mode)
            self.app.selection.restore(result.edges or result.vertices)
        self.app.set_status(
            f"created {len(result.vertices)} point(s) and {len(result.edges)} line(s)"
        )

    def _cancel_construction(self) -> None:
        if not self.app.viewport.cancel_construction():
            self.app.set_status("no construction task is active")

    def _face_snap_data(self, plane) -> GeometrySnapData:
        vertices = [tuple(float(item) for item in plane.world(value)) for value in plane.boundary_vertices]
        endpoints = tuple(
            SnapPoint(f"support:vertex:{index}", point)
            for index, point in enumerate(vertices)
        )
        segments = tuple(
            SnapSegment(
                f"support:edge:{index}",
                vertices[index],
                vertices[(index + 1) % len(vertices)],
                "support-face",
            )
            for index in range(len(vertices))
        )
        midpoints = tuple(
            SnapPoint(
                f"support:edge:{index}:mid",
                tuple(float(item) for item in 0.5 * (np.asarray(item.start) + item.end)),
            )
            for index, item in enumerate(segments)
        )
        return GeometrySnapData(endpoints, midpoints, segments)

    def _start_face_sketch(
        self,
        *,
        support: EntityRef | None = None,
        definition: SketchDefinition | None = None,
        feature_id: int | None = None,
    ) -> None:
        if support is None:
            support = self.require_selection("face", 1)[0]
        if support.kind != "face":
            raise ValueError("select one Model Geometry Plate as sketch support")
        plane = face_sketch_plane(self.app.project.geometry, support.id)
        units = self.app.project.units
        spacing = units.parse(self._face_sketch_grid.get(), "length")
        tolerance = units.parse(self._workplane_tolerance.get(), "length")
        if self.app.viewport.construction_active:
            self.app.viewport.cancel_construction()
        task = FaceSketchTask(plane, tolerance, definition)
        system_id = "__active_face_sketch__"
        system = CoordinateSystem(
            name="Active face sketch",
            id=system_id,
            origin=plane.origin,
            axis=plane.normal,
            reference=plane.x_axis,
        )
        systems = dict(self.app.project.coordinate_systems)
        systems[system_id] = system
        workplane = Workplane(
            system_id,
            grid_spacing=spacing,
            snap_tolerance=tolerance,
            snap_grid=True,
            snap_axes=True,
            snap_endpoints=True,
            snap_midpoints=True,
            snap_intersections=True,
        )
        boundary = np.asarray(plane.boundary_vertices, dtype=float)
        lower = boundary.min(axis=0)
        upper = boundary.max(axis=0)
        margin = max(0.25 * float(np.max(upper - lower)), 2.0 * spacing)
        extent = (
            float(lower[0] - margin), float(upper[0] + margin),
            float(lower[1] - margin), float(upper[1] + margin),
        )
        self._editing_sketch_feature_id = feature_id
        self._sketch_support_ref = support
        if definition is not None:
            self._face_sketch_closed.set(definition.closed)
            self._face_sketch_extrusion.set(units.format(definition.extrusion, "length"))
        self.app.viewport.begin_construction(
            task,
            workplane,
            systems,
            snap_data=self._face_snap_data(plane),
            update_handler=self._construction_updated,
            apply_handler=self.guarded(self._apply_face_sketch),
            grid_extent=extent,
            length_formatter=lambda value: units.format(value, "length"),
        )
        self._update_face_sketch_status(task, None)
        self.app.set_status(
            "face sketch active: click points on the unlimited grid; Enter applies"
        )

    def _sketch_task(self) -> FaceSketchTask:
        task = self.app.viewport.construction_task
        if not isinstance(task, FaceSketchTask):
            raise ValueError("start a sketch on one selected flat plate first")
        return task

    def _sketch_pair(self, task: FaceSketchTask) -> tuple[int, int]:
        try:
            values = tuple(int(item) - 1 for item in self._face_sketch_pair.get().replace(",", " ").split())
        except ValueError as error:
            raise ValueError("point numbers must contain two whole numbers") from error
        if len(values) != 2 or values[0] == values[1]:
            raise ValueError("enter two different point numbers, for example 1 2")
        if min(values) < 0 or max(values) >= len(task.point_keys):
            raise ValueError(f"point numbers must be between 1 and {len(task.point_keys)}")
        return values

    def _add_sketch_distance(self) -> None:
        task = self._sketch_task()
        first, second = self._sketch_pair(task)
        value = self.app.project.units.parse(self._face_sketch_distance.get(), "length")
        task.add_distance(first, second, value)
        if task.ready:
            task.solve_preview(self.app.project.units.parse(self._face_sketch_extrusion.get(), "length"))
        self.app.viewport.refresh_construction_overlay()
        self._update_face_sketch_status(task, None)

    def _add_sketch_coincidence(self) -> None:
        task = self._sketch_task()
        first, second = self._sketch_pair(task)
        task.add_coincidence(first, second)
        if task.ready:
            task.solve_preview(self.app.project.units.parse(self._face_sketch_extrusion.get(), "length"))
        self.app.viewport.refresh_construction_overlay()
        self._update_face_sketch_status(task, None)

    def _remove_sketch_point(self) -> None:
        task = self._sketch_task()
        task.backspace()
        self.app.viewport.refresh_construction_overlay()
        self._update_face_sketch_status(task, None)

    def _remove_sketch_constraint(self) -> None:
        task = self._sketch_task()
        task.remove_last_constraint()
        self.app.viewport.refresh_construction_overlay()
        self._update_face_sketch_status(task, None)

    def _update_face_sketch_status(self, task: FaceSketchTask, snap) -> None:
        boundary = sum(item.kind in ("on_edge", "on_vertex") for item in task.constraints)
        dimensional = sum(item.kind == "distance" for item in task.constraints)
        coincidence = sum(item.kind == "coincident" for item in task.constraints)
        text = (
            f"{len(task.points)} point(s) • {dimensional} distance • "
            f"{coincidence + boundary} coincidence constraint(s)"
        )
        self._face_sketch_status.configure(text=text)
        self.app.set_status(text)

    def _apply_face_sketch(self) -> None:
        task = self._sketch_task()
        task.close = bool(self._face_sketch_closed.get())
        extrusion = self.app.project.units.parse(
            self._face_sketch_extrusion.get(), "length"
        )
        definition = task.solve_preview(extrusion)
        feature_id = self._editing_sketch_feature_id
        if feature_id is None:
            support = self._sketch_support_ref
            if support is None:
                raise ValueError("the sketch support plate is unavailable")
            feature = self.app.run(cmd.AddSketch(support, definition))
            action = "created"
        else:
            feature = self.app.run(
                cmd.EditFeature(feature_id, parameters=definition.to_parameters())
            )
            action = "updated"
        self.app.viewport.end_construction()
        self._editing_sketch_feature_id = None
        self._sketch_support_ref = None
        made = tuple(
            reference for key, reference in feature.outputs.items()
            if key.startswith("extrusion/face/")
        )
        if made:
            self.app.selection.set_mode("face")
            self.app.selection.restore(made)
        self._face_sketch_status.configure(text="No face sketch active")
        self.app.set_status(
            f"{action} editable sketch {feature.feature_id}: "
            f"{len(task.points)} points, {len(made)} extruded plate(s)"
        )

    def _cancel_face_sketch(self) -> None:
        if self.app.viewport.cancel_construction():
            self._editing_sketch_feature_id = None
            self._sketch_support_ref = None
            self._face_sketch_status.configure(text="No face sketch active")
            self.app.set_status("face sketch cancelled; model unchanged")

    def edit_sketch(self, feature_id: int) -> bool:
        record = self.app.project.geometry.features.get(int(feature_id))
        if record.kind != "geometry.sketch.extrude":
            return False
        references = record.inputs.get("support_face", ())
        if len(references) != 1:
            raise ValueError("the sketch support definition is invalid")
        resolved = self.app.project.geometry.features.resolve(
            references[0], self.app.project.geometry
        )
        if len(resolved) != 1 or resolved[0].kind != "face":
            raise ValueError("the sketch support plate is unresolved")
        self._geometry_tabs.select(self._geometry_pages["Guiding geometry"])
        self._start_face_sketch(
            support=resolved[0],
            definition=SketchDefinition.from_parameters(record.parameters),
            feature_id=record.feature_id,
        )
        return True

    def _add_line(self) -> None:
        points = self.require_selection("vertex", 2)
        edge = self.app.run(cmd.AddLine(points[0].id, points[1].id))
        self.app.set_status(f"added line {edge}")

    def _add_arc(self) -> None:
        points = self.require_selection("vertex", 3)
        edge = self.app.run(
            cmd.AddArc(points[0].id, points[1].id, points[2].id)
        )
        self.app.set_status(f"added arc {edge}")

    def _add_polyline(self) -> None:
        points = self.require_selection("vertex")
        if len(points) < 2:
            raise ValueError("a polyline needs at least two ordered points")
        edges = self.app.run(cmd.AddPolyline([ref.id for ref in points]))
        self.app.set_status(f"added guiding polyline with {len(edges)} line(s)")

    def _add_beam_polyline(self) -> None:
        points = self.require_selection("vertex")
        minimum = 3 if self._beam_path_close.get() else 2
        if len(points) < minimum:
            raise ValueError(
                f"this beam chain needs at least {minimum} ordered points"
            )
        edges = self.app.run(
            cmd.AddPolyline(
                [ref.id for ref in points], close=bool(self._beam_path_close.get())
            )
        )
        self.app.set_status(
            f"created {len(edges)} beam centre-line(s); assign a beam section "
            "in Sections"
        )

    def _add_face(self) -> None:
        edges = self.require_selection("edge")
        face = self.app.run(cmd.AddFace([ref.id for ref in edges]))
        self.app.set_status(f"added plate {face}")

    def _add_plate(self) -> None:
        points = self.require_selection("vertex")
        if len(points) < 4:
            raise ValueError("a plate needs at least four points")
        face = self.app.run(cmd.AddPlate([ref.id for ref in points]))
        self.app.set_status(f"added plate {face}")

    def _extrude_lines(self) -> None:
        edges = self.require_selection("edge")
        vector = self.vector(self._extrude, "extrusion vector")
        faces = self.app.run(cmd.Extrude([ref.id for ref in edges], vector))
        self.app.set_status(f"extruded into {len(faces)} plate(s)")

    def _revolve_lines(self) -> None:
        edges = self.require_selection("edge")
        angle = np.deg2rad(self.number(self._angle, "angle"))
        faces = self.app.run(
            cmd.Revolve(
                edge_ids=[ref.id for ref in edges],
                axis_point=self.vector(self._axis_point, "axis point"),
                axis_direction=self.vector(self._axis_direction, "axis direction"),
                angle=float(angle),
            )
        )
        self.app.set_status(f"revolved into {len(faces)} plate(s)")

    def _split_lines(self) -> None:
        edges = self.require_selection("edge")
        fraction = self.number(self._fraction, "fraction")
        self.app.run_many(cmd.SplitEdge(ref.id, fraction) for ref in edges)
        self.app.set_status(f"split {len(edges)} line(s)")

    def _split_plates(self) -> None:
        faces = self.require_selection("face")
        axis = int(self._axis.get())
        fraction = self.number(self._fraction, "fraction")
        self.app.run_many(
            cmd.SplitFace(ref.id, axis, fraction) for ref in faces
        )
        self.app.set_status(f"split {len(faces)} plate(s)")

    def _strip_plates(self) -> None:
        faces = self.require_selection("face")
        axis = int(self._axis.get())
        count = int(self.number(self._strips, "strips"))
        results = self.app.run_many(
            cmd.StripFace(ref.id, axis, count) for ref in faces
        )
        made = sum(len(strips) for strips, _dividers in results)
        self.app.set_status(f"made {made} strip(s)")

    def _triangle(self) -> None:
        edges = self.require_selection("edge", 3)
        faces = self.app.run(
            cmd.TriangleToQuads(edge_ids=[ref.id for ref in edges])
        )
        self.app.selection.clear()
        self.app.set_status(f"three-sided region became {len(faces)} plates")

    def _trim_hole(self) -> None:
        faces = self.require_selection("face", 1)
        centre = self.vector(self._trim_centre, "trim-hole centre")
        radius = self.number(self._trim_radius, "trim-hole radius")
        _face, arcs = self.app.run(
            cmd.NeutralTrimHole(faces[0].id, tuple(centre), radius)
        )
        self.app.set_status(
            f"created neutral trim hole with {len(arcs)} boundary arc(s); "
            "the structural plate was not decomposed"
        )

    def _punch(self) -> None:
        faces = self.require_selection("face", 1)
        centre = self.vector(self._hole_centre, "hole centre")
        radius = self.number(self._hole_radius, "hole radius")
        patches, _arcs = self.app.run(
            cmd.ButterflyHoleDecomposition(faces[0].id, tuple(centre), radius)
        )
        self.app.set_status(
            f"mesh-decomposition hole punched into {len(patches)} "
            "butterfly patches (not a neutral trim)"
        )

    def _set_corners(self) -> None:
        faces = self.require_selection("face", 1)
        try:
            corners = [int(part) for part in self._corners.get().split()]
        except ValueError:
            raise ValueError(
                "corners must be four whole numbers, e.g. 0 1 2 3"
            ) from None
        self.app.run(cmd.SetFaceCorners(faces[0].id, corners))
        self.app.set_status(f"corners set on plate {faces[0].id}")

    def _reverse(self) -> None:
        if self.app.selection.mode not in ("edge", "face"):
            raise ValueError("select one or more lines or plates to reverse")
        items = self.require_selection(self.app.selection.mode)
        self.app.run_many(cmd.ReverseEntity(item) for item in items)
        noun = "normal" if items[0].kind == "face" else "direction"
        self.app.set_status(f"reversed {noun} on {len(items)} entity(ies)")

    def _measure(self) -> None:
        items = self._geometry_selection()
        quantity = self._measurement.get().strip().lower()
        result = self.app.commands.query(cmd.MeasureGeometry(items, quantity))
        value = result.value
        if isinstance(value, tuple):
            rendered = "(" + ", ".join(f"{item:.7g}" for item in value) + ")"
        else:
            rendered = f"{value:.7g}"
        unit = "" if result.unit == "1" else f" {result.unit}"
        if result.kind == "angle":
            rendered += f" ({np.rad2deg(float(value)):.7g} deg)"
        self.app.set_status(f"{result.kind}: {rendered}{unit}")

    def _check(self) -> None:
        faces = self.require_selection("face")
        reports = [
            check_mappable(self.app.project.geometry, ref.id) for ref in faces
        ]
        bad = [report for report in reports if not report.ok]
        if not bad:
            self.app.set_status(f"{len(reports)} plate(s): all mappable")
        else:
            self.app.set_status(str(bad[0]), error=True)

    def _delete(self) -> None:
        items = self.require_selection(self.app.selection.mode)
        self.app.selection.clear()
        self.app.run_many(cmd.DeleteEntity(ref) for ref in items)
        self.app.set_status(f"deleted {len(items)} entity(ies)")


# ----------------------------------------------------------------------
class MeshPanel(StagePanel):
    title = "Mesh"
    _PLACEMENT_LABELS = {
        "legacy_lattice": "Legacy lattice (default)",
        "frontal_delaunay": "Frontal Delaunay (Alpha; planar only)",
    }
    _METRIC_LABELS = {
        "legacy": "Legacy sizing",
        "isotropic_spatial": "Spatial isotropic sizing (Alpha)",
    }
    _CERTIFICATION_LABELS = {
        "interactive": "Interactive (default)",
        "strict": "Strict full geometry audit",
    }
    _METHOD_LABELS = {
        "auto": "Automatic (recommended)",
        "mapped": "Mapped quadrilateral",
        "native": "Unstructured / native",
        "quad_first": "Quad-first",
    }
    _METHOD_VALUES = {label: value for value, label in _METHOD_LABELS.items()}
    _METHOD_HELP = {
        "auto": (
            "Plans the selected Sheet/component globally: mapped blocks and "
            "shared seed equations first, then qualified radial/O-grid partitions, "
            "then explicit metric-aligned native residual regions. Preview is read-only."
        ),
        "mapped": (
            "Uses the transfinite mapped quadrilateral method for every plate. "
            "Each plate must have four mapped sides and no holes; beams are "
            "seeded on their model lines."
        ),
        "native": (
            "Uses unstructured surface meshing for every plate, with optional "
            "quad recombination. This is appropriate for holes and general "
            "plate boundaries."
        ),
        "quad_first": (
            "Explicit quad-first meshing with Python triangulation. Advanced "
            "changes leave the SG1-tested configuration."
        ),
    }
    _STRUCTURE_LABELS = {
        "balanced": "Balanced (recommended)",
        "quad_first": "Prefer mapped quads",
        "size_first": "Prefer target size",
    }
    _STRUCTURE_VALUES = {
        label: value for value, label in _STRUCTURE_LABELS.items()
    }
    _NATIVE_BACKEND_LABELS = {
        "auto": "Automatic",
        "python": "Python compatibility",
        "native": "Compiled native",
    }
    _NATIVE_BACKEND_VALUES = {
        label: value for value, label in _NATIVE_BACKEND_LABELS.items()
    }

    def build(self) -> None:
        controls = self.section("Mesh generation")
        method_row = ttk.Frame(controls)
        method_row.pack(fill="x", pady=1)
        ttk.Label(method_row, text="meshing method", width=16).pack(side="left")
        self._method = tk.StringVar(value=self._METHOD_LABELS["auto"])
        self._method_combo = ttk.Combobox(
            method_row,
            textvariable=self._method,
            values=list(self._METHOD_VALUES),
            state="readonly",
            width=24,
        )
        self._method_combo.pack(side="left", fill="x", expand=True)
        self._method_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._method_changed()
        )
        self._method_dirty = False
        shortcuts = ttk.Frame(controls)
        shortcuts.pack(fill="x", pady=(2, 3))
        for value, label in (("auto", "Automatic"), ("mapped", "Mapped"), ("native", "Unstructured"), ("quad_first", "Quad-first")):
            ttk.Radiobutton(
                shortcuts, text=label, variable=self._method,
                value=self._METHOD_LABELS[value], command=self._method_changed,
            ).pack(side="left")
        self._method_help = ttk.Label(
            controls,
            text=self._METHOD_HELP["auto"],
            foreground="#444444",
            justify="left",
            wraplength=300,
        )
        self._method_help.pack(anchor="w", pady=(1, 3))
        self._structure_options = ttk.Frame(controls)
        self._structure_options.pack(fill="x", pady=(0, 3))
        ttk.Label(
            self._structure_options, text="automatic priority", width=16
        ).pack(side="left")
        self._structure_preference = tk.StringVar(
            value=self._STRUCTURE_LABELS["balanced"]
        )
        self._structure_combo = ttk.Combobox(
            self._structure_options,
            textvariable=self._structure_preference,
            values=list(self._STRUCTURE_VALUES),
            state="readonly",
            width=24,
        )
        self._structure_combo.pack(side="left", fill="x", expand=True)
        self._preference_dirty = False
        self._structure_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._preference_changed()
        )
        self._quality_options = ttk.LabelFrame(
            controls, text="Automatic hard quality gate", padding=(4, 2)
        )
        self._quality_options.pack(fill="x", pady=(0, 3))
        self._quality_jacobian = self.entry_row(
            self._quality_options, "min scaled J", "0.20"
        )
        self._quality_aspect = self.entry_row(
            self._quality_options, "max aspect", "4.0"
        )
        self._quality_min_angle = self.entry_row(
            self._quality_options, "min angle [deg]", "30"
        )
        self._quality_max_angle = self.entry_row(
            self._quality_options, "max angle [deg]", "150"
        )
        self._quality_warpage = self.entry_row(
            self._quality_options, "max warpage", "0.10"
        )
        ttk.Label(
            self._quality_options,
            text=(
                "A proposed structured layout that misses these limits is rejected "
                "before publication and Automatic uses the documented residual route."
            ),
            foreground="#666666",
            justify="left",
            wraplength=280,
        ).pack(anchor="w")
        self._mapped_status = ttk.Label(
            controls,
            text="",
            foreground="#666666",
            justify="left",
            wraplength=300,
        )
        self._mapped_status.pack(anchor="w", pady=(0, 4))
        self._size = self.entry_row(controls, "element size [m]", "0.25")
        row = ttk.Frame(controls)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="element order", width=16).pack(side="left")
        self._order = tk.StringVar(value="linear")
        self._order_combo = ttk.Combobox(
            row,
            textvariable=self._order,
            values=list(ELEMENT_ORDERS),
            state="readonly",
            width=12,
        )
        self._order_combo.pack(side="left", fill="x", expand=True)
        self._order_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._order_changed()
        )
        self._order_dirty = False
        self._show_method_advanced = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls, text="Advanced method settings",
            variable=self._show_method_advanced,
            command=self._toggle_method_advanced,
        ).pack(anchor="w")
        self._method_advanced = ttk.Frame(controls)
        self._strict_method = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self._method_advanced,
            text="Use only the selected method",
            variable=self._strict_method,
        ).pack(anchor="w")
        self._native_options = ttk.Frame(controls)
        self._native_options.pack(fill="x")
        self._quad_advanced = ttk.LabelFrame(controls, text="Advanced quad-first settings", padding=(4, 2))
        self._quad_layout = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self._quad_advanced, text="Automatic geometry layout",
            variable=self._quad_layout,
            command=self._update_method_controls,
        ).pack(anchor="w")
        self._quad_orientation = tk.StringVar(value="cross_4theta")
        self._quad_quality = tk.StringVar(value="shape_jacobian")
        self._quad_optimization = tk.StringVar(value="safeguarded")
        for label, variable, choices in (
            ("orientation", self._quad_orientation, ("cross_4theta", "boundary_tangent")),
            ("quality model", self._quad_quality, ("shape_jacobian", "shape")),
            ("optimization", self._quad_optimization, ("safeguarded", "off")),
        ):
            quad_row = ttk.Frame(self._quad_advanced)
            quad_row.pack(fill="x")
            ttk.Label(quad_row, text=label, width=16).pack(side="left")
            ttk.Combobox(quad_row, textvariable=variable, values=choices, state="readonly", width=20).pack(side="left")
        self._quad_front = self.entry_row(self._quad_advanced, "front iterations", "100000")
        self._quad_local = self.entry_row(self._quad_advanced, "local optimizations", "64")
        self._quad_cancel = self.entry_row(self._quad_advanced, "cancel interval", "256")
        self._quad_preset_status = ttk.Label(self._quad_advanced, text="SG1-compatible preset", foreground="#267326")
        self._quad_preset_status.pack(anchor="w")
        backend_row = ttk.Frame(self._native_options)
        backend_row.pack(fill="x", pady=1)
        ttk.Label(backend_row, text="triangulator", width=16).pack(
            side="left"
        )
        self._native_backend = tk.StringVar(value="Automatic")
        ttk.Combobox(
            backend_row,
            textvariable=self._native_backend,
            values=list(self._NATIVE_BACKEND_VALUES),
            state="readonly",
            width=20,
        ).pack(side="left", fill="x", expand=True)
        ttk.Label(
            self._native_options,
            text=(
                "Used for unstructured faces only. Automatic may use it as a "
                "fallback; Mapped does not use a triangulator."
            ),
            foreground="#666666",
            justify="left",
            wraplength=220,
        ).pack(anchor="w", pady=(0, 3))
        self._controls_dirty = False
        self._loading_mesh_controls = False
        self._recombine = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self._native_options, text="Recombine triangles into quads where admissible",
            variable=self._recombine,
        ).pack(anchor="w")
        self._show_structured_controls = tk.BooleanVar(value=False)
        self._structured_disclosure = ttk.Checkbutton(
            controls,
            text="Show advanced structured planning controls",
            variable=self._show_structured_controls,
            command=self._toggle_structured_controls,
        )
        self._advanced_structured = ttk.LabelFrame(
            controls, text="Automatic structured planning", padding=(4, 2)
        )
        self._allow_detached_partition = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self._advanced_structured,
            text="Allow detached mesh-only partitions",
            variable=self._allow_detached_partition,
        ).pack(anchor="w")
        self._structured_entries = {}
        for name, label, default in (
            ("max_element_growth", "max element growth", "1.5"),
            ("minimum_size_ratio", "minimum size ratio", "0.67"),
            ("maximum_size_ratio", "maximum size ratio", "1.5"),
            ("maximum_radial_sides", "max radial sides", "8"),
            (
                "maximum_candidates_per_component",
                "candidates / component",
                "256",
            ),
            ("maximum_face_records", "max face records", "100000"),
            ("maximum_blocks", "max blocks", "100000"),
            ("maximum_edge_records", "max edge records", "200000"),
            (
                "maximum_estimated_elements",
                "max estimated elements",
                "2000000",
            ),
            (
                "maximum_divisions_per_edge",
                "max divisions / edge",
                "100000",
            ),
        ):
            self._structured_entries[name] = self.entry_row(
                self._advanced_structured, label, default
            )
        ttk.Label(
            self._advanced_structured,
            text=(
                "These are hard planning limits from ANYmesher 0.5. "
                "Preview and Generate use the same values. Lower limits fail "
                "with a diagnostic instead of silently changing the layout."
            ),
            foreground="#666666",
            justify="left",
            wraplength=300,
        ).pack(anchor="w", pady=(2, 0))
        self._advanced_native = ttk.LabelFrame(
            self._native_options, text="Native surface filling — explicit Alpha opt-in",
            padding=(4, 2),
        )
        self._show_native_controls = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self._native_options, text="Show advanced native / Alpha controls",
            variable=self._show_native_controls, command=self._toggle_native_controls,
        ).pack(anchor="w")
        self._native_filling_status = ttk.Label(self._native_options, foreground="#666666")
        self._native_filling_status.pack(anchor="w")
        self._point_placement = tk.StringVar(value=self._PLACEMENT_LABELS["legacy_lattice"])
        self._metric_mode = tk.StringVar(value=self._METRIC_LABELS["legacy"])
        for label, variable, values in (
            ("point placement", self._point_placement, self._PLACEMENT_LABELS),
            ("size metric", self._metric_mode, self._METRIC_LABELS),
        ):
            row = ttk.Frame(self._advanced_native)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=label, width=16).pack(side="left")
            combo = ttk.Combobox(row, textvariable=variable, values=list(values.values()), state="readonly")
            combo.pack(side="left", fill="x", expand=True)
            if variable is self._metric_mode:
                self._metric_combo = combo
            else:
                self._placement_combo = combo
        self._native_v2_available = native_v2_options_type() is not None
        if not self._native_v2_available:
            self._placement_combo.configure(state="disabled")
            ttk.Label(self._advanced_native, text="Native-v2 controls unavailable in installed ANYmesher.",
                      foreground="#8a5a00").pack(anchor="w")
        self._point_placement.trace_add("write", self._placement_changed)
        self._native_budget_frame = ttk.Frame(self._advanced_native)
        self._native_budget_frame.pack(fill="x")
        self._native_max_insertions = self.entry_row(self._native_budget_frame, "max insertions", "10000")
        self._native_max_operations = self.entry_row(self._native_budget_frame, "max topology ops", "1000000")
        self._native_cancel_interval = self.entry_row(self._native_budget_frame, "cancel check interval", "256")
        ttk.Label(self._advanced_native, wraplength=300, justify="left", foreground="#666666",
                  text="Metric and budgets apply to Frontal Delaunay only. When inactive, saved budgets are retained.").pack(anchor="w")
        ttk.Label(
            self._advanced_native, wraplength=300, justify="left", foreground="#8a5a00",
            text=("Alpha: Frontal Delaunay is planar-only; curved native faces are rejected. "
                  "Spatial sizing uses the target size and local refinements. Qualification "
                  "is incomplete; compiled availability is not a quality/performance guarantee. "
                  "Budgets bound insertion/topology work; cancellation checks are cooperative. "
                  "Field-guided quad fronts are not available."),
        ).pack(anchor="w", pady=2)
        audit_row = self._audit_row = ttk.Frame(controls)
        audit_row.pack(fill="x", pady=2)
        ttk.Label(audit_row, text="geometry validation", width=16).pack(side="left")
        self._certification = tk.StringVar(value=self._CERTIFICATION_LABELS["interactive"])
        ttk.Combobox(audit_row, textvariable=self._certification,
                     values=list(self._CERTIFICATION_LABELS.values()), state="readonly").pack(
                         side="left", fill="x", expand=True)
        ttk.Label(
            controls, wraplength=300, justify="left", foreground="#666666",
            text=("Interactive audits changed regions when supplied; it does not run a full "
                  "geometry audit for every job. Strict audits the full working geometry and "
                  "can take longer. Element quality/admission checks remain enabled in both."),
        ).pack(anchor="w")
        for variable in (
            self._native_backend, self._recombine, self._point_placement, self._metric_mode,
            self._native_max_insertions, self._native_max_operations, self._native_cancel_interval,
            self._certification, self._quality_jacobian, self._quality_aspect,
            self._quality_min_angle, self._quality_max_angle, self._quality_warpage,
            self._allow_detached_partition, *self._structured_entries.values(),
        ):
            variable.trace_add("write", self._mesh_control_edited)
        self._placement_changed()
        preview_actions = ttk.Frame(controls)
        preview_actions.pack(fill="x", pady=(2, 1))
        self._preview_button = ttk.Button(
            preview_actions, text="Preview layout", command=self._preview_layout
        )
        self._preview_button.pack(side="left", fill="x", expand=True)
        self._commit_layout_button = ttk.Button(
            preview_actions,
            text="Commit partitions",
            command=self._commit_layout,
        )
        self._commit_layout_button.pack(side="left", fill="x", expand=True, padx=(3, 0))
        self._discard_layout_button = ttk.Button(
            preview_actions,
            text="Discard preview",
            command=self._discard_layout,
        )
        self._discard_layout_button.pack(side="left", fill="x", expand=True, padx=(3, 0))
        self._preview_status = ttk.Label(
            controls,
            text="Preview shows proposed blocks and residual regions; committing is optional.",
            foreground="#666666",
            justify="left",
            wraplength=300,
        )
        self._preview_status.pack(anchor="w", pady=(0, 3))
        self._generate_button = self.button(controls, "Generate mesh", self._generate)
        self._cancel_button = self.button(controls, "Cancel mesh", self._cancel)
        self.button(controls, "Open standalone ANYmesher...", self._open_mesher)
        ttk.Label(
            controls,
            text=(
                "The standalone editor creates neutral meshes. It does not "
                "change the meshing method or model associations in this project."
            ),
            foreground="#666666",
            justify="left",
            wraplength=300,
        ).pack(anchor="w", pady=(0, 3))

        seeding = self.section("Seeding")
        ttk.Label(
            seeding,
            text=(
                "Pin an exact number of divisions on selected model lines. "
                "The same edge nodes are reused at mapped/unstructured and "
                "beam/shell interfaces."
            ),
            foreground="#666666",
            justify="left",
            wraplength=300,
        ).pack(anchor="w", pady=(0, 3))
        self._divisions = self.entry_row(seeding, "divisions", "4")
        self.button(seeding, "Pin selected lines", self._pin)
        self.button(seeding, "Clear pins", self._clear_pins)
        self._pin_label = ttk.Label(seeding, text="no pinned lines", foreground="#666666")
        self._pin_label.pack(anchor="w")

        refine = self.section("Local refinement")
        ttk.Label(
            refine,
            text=(
                "Applies a smaller target size near selected model entities. "
                "Unstructured plates can grade locally; a mapped plate remains "
                "a structured grid whose compatible edge divisions may grow."
            ),
            foreground="#666666",
            justify="left",
            wraplength=300,
        ).pack(anchor="w", pady=(0, 3))
        self._refine_size = self.entry_row(refine, "element size [m]", "0.05")
        self._refine_radius = self.entry_row(refine, "radius [m]", "0.1")
        self.button(refine, "Refine around selection", self._refine)
        self.button(refine, "Clear refinements", self._clear_refinements)
        self._refine_label = ttk.Label(
            refine, text="no refinement zones", foreground="#666666",
            justify="left", wraplength=220,
        )
        self._refine_label.pack(anchor="w")

        self._copy_mesh_diagnosis_button = self.button(
            self, "Copy mesh diagnosis", self._copy_mesh_diagnosis
        )
        self._stats = ttk.Label(self, text="no mesh", justify="left")
        self._stats.pack(anchor="w")
        self._update_method_controls()

    def _toggle_native_controls(self):
        if self._show_native_controls.get():
            self._advanced_native.pack(fill="x", pady=3)
        else:
            self._advanced_native.pack_forget()

    def _toggle_method_advanced(self) -> None:
        if self._show_method_advanced.get():
            self._method_advanced.pack(fill="x", pady=(0, 3))
        else:
            self._method_advanced.pack_forget()

    def _toggle_structured_controls(self):
        if (
            self._show_structured_controls.get()
            and self._method_value() == "auto"
        ):
            self._advanced_structured.pack(
                fill="x", pady=(0, 3), before=self._mapped_status
            )
        else:
            self._advanced_structured.pack_forget()

    def reset_mesh_drafts(self):
        """An explicit document open/new discards drafts, even for the same ID."""
        self._controls_dirty = self._method_dirty = self._preference_dirty = self._order_dirty = False
        self._controls_project_id = None

    def _mesh_control_edited(self, *_args):
        if not self._loading_mesh_controls:
            self._controls_dirty = True

    def _placement_changed(self, *_args):
        frontal = self._point_placement.get() == self._PLACEMENT_LABELS["frontal_delaunay"]
        self._native_filling_status.configure(
            text="Alpha Frontal Delaunay (planar only)" if frontal else "Legacy lattice (default)",
            foreground="#8a5a00" if frontal else "#666666",
        )
        enabled = frontal and self._native_v2_available
        self._metric_combo.configure(state="readonly" if enabled else "disabled")
        if (not frontal and not getattr(self, "_loading_mesh_controls", False)
                and self._metric_mode.get() != self._METRIC_LABELS["legacy"]):
            self._metric_mode.set(self._METRIC_LABELS["legacy"])
        if hasattr(self, "_native_budget_frame"):
            for row in self._native_budget_frame.winfo_children():
                for widget in row.winfo_children():
                    if isinstance(widget, ttk.Entry):
                        widget.configure(state="normal" if enabled else "disabled")

    @staticmethod
    def _control_choice(labels, variable, name):
        for key, label in labels.items():
            if label == variable.get():
                return key
        raise ValueError(f"unknown {name}: {variable.get()!r}")

    def _mesh_controls_value(self):
        if getattr(self, "_controls_error", None):
            raise ValueError(self._controls_error)
        certification = self._control_choice(self._CERTIFICATION_LABELS, self._certification, "geometry validation")
        if self._method_value() == "mapped":
            # Hidden native entries must not block a mapped-only submission.
            return replace(MeshControls.from_settings(self.app.project.native_mesh_settings),
                           certification_mode=certification)
        def integer(variable, name):
            try:
                value = int(variable.get())
            except (ValueError, TypeError):
                raise ValueError(f"{name} must be a positive integer") from None
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
            return value
        placement = self._control_choice(self._PLACEMENT_LABELS, self._point_placement, "point placement")
        saved = MeshControls.from_settings(self.app.project.native_mesh_settings)
        frontal = placement == "frontal_delaunay"
        return MeshControls(
            recombine=self._recombine.get(),
            certification_mode=certification,
            point_placement=placement,
            metric_mode=self._control_choice(self._METRIC_LABELS, self._metric_mode, "size metric"),
            max_insertions=(integer(self._native_max_insertions, "max insertions") if frontal else saved.max_insertions),
            max_topology_operations=(integer(self._native_max_operations, "max topology operations") if frontal else saved.max_topology_operations),
            cancellation_interval=(integer(self._native_cancel_interval, "cancellation interval") if frontal else saved.cancellation_interval),
        )

    def _structured_controls_value(self) -> StructuredMeshControls:
        if self._method_value() != "auto":
            # Hidden structured-planner drafts must not block a mapped-only or
            # native-only submission. Preserve the last saved Automatic policy.
            return StructuredMeshControls.from_settings(
                self.app.project.native_mesh_settings
            )
        integer_fields = {
            "maximum_radial_sides",
            "maximum_candidates_per_component",
            "maximum_face_records",
            "maximum_blocks",
            "maximum_edge_records",
            "maximum_estimated_elements",
            "maximum_divisions_per_edge",
        }
        values = {}
        for name, variable in self._structured_entries.items():
            label = name.replace("_", " ")
            if name in integer_fields:
                try:
                    numeric = float(variable.get())
                except (TypeError, ValueError):
                    raise ValueError(f"{label} must be a positive whole number") from None
                value = int(numeric)
                if numeric != value or value < 1:
                    raise ValueError(f"{label} must be a positive whole number")
            else:
                value = self.number(variable, label)
            values[name] = value
        return StructuredMeshControls(
            allow_detached_partition=bool(self._allow_detached_partition.get()),
            **values,
        )

    def _refresh_mesh_controls(self):
        project_id = self.app.project.document_id
        new_document = getattr(self, "_controls_project_id", None) != project_id
        if new_document:
            self._controls_dirty = False
            self._method_dirty = False
            self._preference_dirty = False
        self._controls_project_id = project_id
        if self._controls_dirty:
            return
        settings = self.app.project.native_mesh_settings
        parameters = {} if settings is None else settings.to_dict()["parameters"]
        quad_token = (project_id, repr(parameters.get("quad_options")),
                      parameters.get("layout_policy", "existing"))
        if (hasattr(self, "_quad_orientation")
                and getattr(self, "_quad_loaded_token", None) != quad_token):
            from ..quad_first import effective_quad_options

            quad = effective_quad_options(parameters.get("quad_options"))
            self._quad_layout.set(parameters.get("layout_policy", "existing") == "adaptive")
            self._quad_orientation.set(quad.orientation)
            self._quad_quality.set(quad.quality_model)
            self._quad_optimization.set(quad.line_search)
            for variable, value in (
                (self._quad_front, quad.max_front_iterations),
                (self._quad_local, quad.max_local_optimizations),
                (self._quad_cancel, quad.cancellation_interval),
            ):
                variable.set(str(value))
            self._quad_loaded_token = quad_token
        try:
            controls = MeshControls.from_settings(settings)
            structured_controls = StructuredMeshControls.from_settings(settings)
        except ValueError as error:
            # Display-only saved intent; never pass this stand-in to execution.
            from types import SimpleNamespace
            controls = SimpleNamespace(**MeshControls.display_values(settings))
            structured_controls = SimpleNamespace(
                **StructuredMeshControls.display_values(settings)
            )
            self._controls_error = (
                f"Saved mesh controls cannot run: {error}. "
                "Use a compatible ANYmesher installation; saved settings are unchanged."
            )
        else:
            self._controls_error = None
        placement_label = self._PLACEMENT_LABELS.get(controls.point_placement, str(controls.point_placement))
        changed_placement = self._point_placement.get() != placement_label
        self._loading_mesh_controls = True
        try:
            self._native_backend.set(self._NATIVE_BACKEND_LABELS[self.app.project.native_triangulation_backend])
            self._recombine.set(controls.recombine)
            self._point_placement.set(placement_label)
            self._metric_mode.set(self._METRIC_LABELS.get(controls.metric_mode, str(controls.metric_mode)))
            self._certification.set(self._CERTIFICATION_LABELS[controls.certification_mode])
            self._native_max_insertions.set(str(controls.max_insertions))
            self._native_max_operations.set(str(controls.max_topology_operations))
            self._native_cancel_interval.set(str(controls.cancellation_interval))
            self._allow_detached_partition.set(
                structured_controls.allow_detached_partition
            )
            for name, variable in self._structured_entries.items():
                variable.set(str(getattr(structured_controls, name)))
            if controls.point_placement == "frontal_delaunay" and (new_document or changed_placement):
                self._show_native_controls.set(True)
                self._toggle_native_controls()
            from anymesher import MeshQualityPolicy
            parameters = {} if settings is None else dict(settings.parameters)
            quality = MeshQualityPolicy.create({
                key.removeprefix("mesh_quality_"): value for key, value in parameters.items()
                if key.startswith("mesh_quality_")
            }).to_dict()
            for key, variable in (
                ("minimum_scaled_jacobian", self._quality_jacobian),
                ("maximum_aspect_ratio", self._quality_aspect),
                ("minimum_angle", self._quality_min_angle),
                ("maximum_angle", self._quality_max_angle),
                ("maximum_warpage", self._quality_warpage),
            ):
                variable.set(str(quality[key]))
            if self._controls_error:
                self._native_filling_status.configure(text=self._controls_error, foreground="#a03020", wraplength=300)
        finally:
            self._loading_mesh_controls = False

    def _method_value(self) -> str:
        try:
            return self._METHOD_VALUES[self._method.get()]
        except KeyError as error:
            raise ValueError(
                f"unknown meshing method {self._method.get()!r}"
            ) from error

    def _stored_method_value(self) -> str:
        settings = self.app.project.native_mesh_settings
        if settings is None:
            return "auto"
        backend = str(getattr(settings.backend, "value", settings.backend))
        return {
            "automatic": "auto",
            "auto": "auto",
            "mapped": "mapped",
            "native": "native",
            "quad_first": "quad_first",
        }.get(backend, "auto")

    def _structure_preference_value(self) -> str:
        try:
            return self._STRUCTURE_VALUES[self._structure_preference.get()]
        except KeyError as error:
            raise ValueError(
                f"unknown automatic priority {self._structure_preference.get()!r}"
            ) from error

    def _stored_structure_preference(self) -> str:
        settings = self.app.project.native_mesh_settings
        parameters = {} if settings is None else dict(settings.parameters)
        value = str(parameters.get("structure_preference", "balanced"))
        return value if value in self._STRUCTURE_LABELS else "balanced"

    def _quality_policy_value(self) -> dict[str, float]:
        from anymesher import MeshQualityPolicy

        values = {
            "minimum_scaled_jacobian": self.number(
                self._quality_jacobian, "minimum scaled Jacobian"
            ),
            "maximum_aspect_ratio": self.number(
                self._quality_aspect, "maximum aspect ratio"
            ),
            "minimum_angle": self.number(
                self._quality_min_angle, "minimum angle"
            ),
            "maximum_angle": self.number(
                self._quality_max_angle, "maximum angle"
            ),
            "maximum_warpage": self.number(
                self._quality_warpage, "maximum warpage"
            ),
        }
        return MeshQualityPolicy.create(values).to_dict()

    def _method_changed(self) -> None:
        self._method_dirty = True
        self._update_method_controls()

    def _order_changed(self) -> None:
        self._order_dirty = True

    def _preference_changed(self) -> None:
        self._preference_dirty = True
        self._update_method_controls()

    def _update_method_controls(self, *, busy: bool | None = None) -> None:
        method = self._method_value()
        self._method_help.configure(text=self._METHOD_HELP[method])
        eligible, diagnostic = mapped_mesh_eligibility(self.app.project.geometry)
        if method == "mapped":
            self._mapped_status.configure(
                text=diagnostic,
                foreground=("#267326" if eligible else "#a03020"),
            )
        elif method == "auto":
            self._mapped_status.configure(
                text=(
                    diagnostic
                    if eligible
                    else "Automatic remains available and chooses the safe "
                    "method per plate. Mapped-only precheck: " + diagnostic
                ),
                foreground=("#267326" if eligible else "#8a5a00"),
            )
        else:
            self._mapped_status.configure(
                text=(
                    "Mapped eligibility is not required because every plate "
                    "will use the unstructured method."
                ),
                foreground="#666666",
            )

        if method == "auto":
            if not self._structure_options.winfo_manager():
                self._structure_options.pack(
                    fill="x", pady=(0, 3), before=self._mapped_status
                )
            if not self._quality_options.winfo_manager():
                self._quality_options.pack(
                    fill="x", pady=(0, 3), before=self._mapped_status
                )
            if not self._structured_disclosure.winfo_manager():
                self._structured_disclosure.pack(
                    anchor="w", pady=(0, 2), before=self._mapped_status
                )
            self._toggle_structured_controls()
        else:
            self._structure_options.pack_forget()
            self._quality_options.pack_forget()
            self._structured_disclosure.pack_forget()
            self._advanced_structured.pack_forget()

        if method in ("mapped", "quad_first"):
            self._native_options.pack_forget()
        elif not self._native_options.winfo_manager():
            self._native_options.pack(fill="x", before=self._audit_row)
        if method == "quad_first":
            if not self._quad_advanced.winfo_manager():
                self._quad_advanced.pack(fill="x", before=self._audit_row)
            try:
                from ..quad_first import sg1_compatible
                compatible = sg1_compatible(self._quad_options_value())
                compatible = compatible and not self._quad_layout.get()
                self._quad_preset_status.configure(
                    text="SG1-compatible preset" if compatible else "Settings differ from SG1-tested configuration",
                    foreground="#267326" if compatible else "#8a5a00",
                )
            except (ValueError, TypeError) as error:
                self._quad_preset_status.configure(text=str(error), foreground="#a03020")
        else:
            self._quad_advanced.pack_forget()

        if busy is None:
            busy = bool(getattr(self.app, "mesh_job_running", False))
        blocked = (method == "mapped" and not eligible) or bool(getattr(self, "_controls_error", None))
        if getattr(self, "_controls_error", None):
            self._mapped_status.configure(text=self._controls_error, foreground="#a03020")
        self._generate_button.configure(
            state="disabled" if busy or blocked else "normal"
        )
        preview_available = method == "auto" and not busy
        self._preview_button.configure(
            state="normal" if preview_available else "disabled"
        )
        current_preview = getattr(self.app, "_mesh_layout_preview", None)
        can_commit = bool(
            preview_available
            and current_preview is not None
            and current_preview[0].requires_working_clone
        )
        self._commit_layout_button.configure(
            state="normal" if can_commit else "disabled"
        )
        self._discard_layout_button.configure(
            state="normal" if current_preview is not None else "disabled"
        )

    def refresh(self) -> None:
        self._refresh_mesh_controls()
        pins = self.app.seeding_overrides
        self._pin_label.configure(
            text="no pinned lines"
            if not pins
            else "pinned: "
            + ", ".join(f"{edge}→{count}" for edge, count in sorted(pins.items()))
        )
        if not self._order_dirty and self._order.get() != self.app.project.element_order:
            self._order.set(self.app.project.element_order)
        if not self._method_dirty:
            method_label = self._METHOD_LABELS[self._stored_method_value()]
            if self._method.get() != method_label:
                self._method.set(method_label)
        if not self._preference_dirty:
            preference_label = self._STRUCTURE_LABELS[
                self._stored_structure_preference()
            ]
            if self._structure_preference.get() != preference_label:
                self._structure_preference.set(preference_label)

        zones = self.app.project.refinements
        self._refine_label.configure(
            text="no refinement zones"
            if not zones
            else "\n".join(
                f"{zone.size:g} m within {zone.radius:g} m of "
                f"{zone.ref if zone.ref is not None else 'a point'}"
                for zone in zones
            )
        )

        busy = bool(getattr(self.app, "mesh_job_running", False))
        self._update_method_controls(busy=busy)
        self._cancel_button.configure(state="normal" if busy else "disabled")

        record_id = (
            getattr(self.app, "_active_mesh_task_id", None)
            or getattr(self.app, "_mesh_details_record_id", None)
            or getattr(self.app, "mesh_record_id", None)
        )
        record = self.app.project.mesh_records.get(record_id or "")
        if record is None and self.app.project.mesh_records:
            record = next(reversed(self.app.project.mesh_records.values()))
        state = None if record is None else self.app.mesh_record_state(record)
        mesh = (
            self.app._inspection_mesh
            if record is not None and state == "inspection_only"
            else self.app.mesh
        )
        lines: list[str] = []
        if record is not None:
            lines.append(f"{record.name}: {state}")
            automation = record.summary.get("automation")
            if isinstance(automation, Mapping):
                method_used = automation.get("selected_method")
                if method_used:
                    lines.append(f"meshing method used: {method_used}")
                admission = automation.get("solver_admission")
                if admission:
                    lines.append(f"solver admission: {admission}")
                attempts = automation.get("attempts", ())
                if isinstance(attempts, (list, tuple)):
                    for attempt in attempts:
                        if isinstance(attempt, Mapping) and attempt.get("status") == "rejected":
                            lines.append(
                                f"recovery from {attempt.get('method')}: "
                                f"{attempt.get('reason')}"
                            )
            requested_order = record.summary.get("element_order")
            if requested_order is not None:
                lines.append(f"element order requested: {requested_order}")
            requested_strategy = record.summary.get("strategy_requested")
            if requested_strategy is not None:
                method = self._METHOD_LABELS.get(
                    str(requested_strategy), str(requested_strategy)
                )
                lines.append(f"meshing method requested: {method}")
            strategy_by_face = record.summary.get("strategy_by_face", {})
            requested_controls = record.summary.get("controls_requested")
            if isinstance(requested_controls, Mapping):
                lines.append("requested surface controls: " + ", ".join(
                    f"{key}={value}" for key, value in requested_controls.items()
                ))
            requested_structured = record.summary.get(
                "structured_controls_requested"
            )
            if isinstance(requested_structured, Mapping):
                lines.append("requested structured controls: " + ", ".join(
                    f"{key}={value}"
                    for key, value in requested_structured.items()
                ))
            if isinstance(strategy_by_face, dict) and strategy_by_face:
                mapped = sum(
                    str(value) == "mapped" for value in strategy_by_face.values()
                )
                native = sum(
                    str(value) == "native" for value in strategy_by_face.values()
                )
                lines.append(
                    f"plate method used: {mapped} mapped, {native} unstructured"
                )
            structured = record.summary.get("structured_layout", {})
            if isinstance(structured, Mapping) and structured:
                metrics = structured.get("metrics", {})
                plan = structured.get("plan", {})
                status = structured.get("status")
                if status:
                    lines.append(f"structured layout: {status}")
                if isinstance(plan, Mapping) and plan:
                    options = plan.get("options", {})
                    if isinstance(options, Mapping):
                        preference = options.get("preference")
                        if preference:
                            lines.append(f"automatic priority: {preference}")
                    blocks = plan.get("blocks", ())
                    interfaces = plan.get("interfaces", ())
                    blockers = plan.get("blockers", ())
                    lines.append(
                        f"global layout: {len(blocks) if isinstance(blocks, list) else 0} "
                        f"block/residual region(s), "
                        f"{len(interfaces) if isinstance(interfaces, list) else 0} interface(s), "
                        f"{len(blockers) if isinstance(blockers, list) else 0} blocker(s)"
                    )
                    if plan.get("plan_hash"):
                        lines.append(f"layout hash: {str(plan['plan_hash'])[-16:]}")
                if structured.get("plan_hash"):
                    lines.append(
                        f"layout hash: {str(structured['plan_hash'])[-16:]}"
                    )
                seed_solution = structured.get("seed_solution", {})
                if isinstance(seed_solution, Mapping) and seed_solution:
                    values = [int(value) for value in seed_solution.values()]
                    lines.append(
                        f"global seed solution: {len(values)} edge(s), "
                        f"{min(values)}–{max(values)} division(s)"
                    )
                structured_quality = structured.get("quality", {})
                if isinstance(structured_quality, Mapping) and structured_quality:
                    worst = structured_quality.get("worst", structured_quality)
                    accepted = bool(structured_quality.get("accepted", False))
                    lines.append(
                        "structured quality gate: "
                        + ("accepted" if accepted else "rejected / native fallback")
                    )
                    if isinstance(worst, Mapping):
                        lines.append(
                            "layout worst: scaled J "
                            f"{float(worst.get('minimum_scaled_jacobian', 0.0)):.3g}, "
                            f"aspect {float(worst.get('maximum_aspect_ratio', 0.0)):.3g}, "
                            f"angles {float(worst.get('minimum_angle', 0.0)):.3g}°–"
                            f"{float(worst.get('maximum_angle', 0.0)):.3g}°"
                        )
                if isinstance(metrics, Mapping) and metrics:
                    lines.append(
                        "mesh regularity: "
                        f"score {100.0 * float(metrics.get('regularity_score', 0.0)):.1f}%, "
                        f"{100.0 * float(metrics.get('quad_fraction', 0.0)):.1f}% quads, "
                        f"{100.0 * float(metrics.get('mapped_area_fraction', 0.0)):.1f}% mapped area, "
                        f"{int(metrics.get('structured_block_count', 0))} structured block(s)"
                    )
                    lines.append(
                        "size regularity: CV "
                        f"{float(metrics.get('size_coefficient_of_variation', 0.0)):.3g}; "
                        f"{int(metrics.get('edges_outside_balanced_size_band', 0))} edge(s) "
                        "outside 0.67–1.50× target"
                    )
                    lines.append(
                        "grid continuity "
                        f"{100.0 * float(metrics.get('grid_line_continuity_score', 0.0)):.1f}%; "
                        f"{int(metrics.get('irregular_interior_valence_count', 0))} irregular node(s)"
                    )
                diagnostics = structured.get("diagnostics", ())
                if isinstance(diagnostics, (list, tuple)):
                    lines.extend(
                        f"structure note: {message}" for message in diagnostics[:4]
                    )
            complex_geometry = record.summary.get("complex_geometry", {})
            if isinstance(complex_geometry, Mapping) and complex_geometry:
                components = complex_geometry.get("components", ())
                component_count = complex_geometry.get("component_count")
                if (
                    component_count is None
                    and "components" in complex_geometry
                    and isinstance(components, (list, tuple))
                ):
                    component_count = len(components)
                if component_count is not None:
                    lines.append(
                        f"complex geometry plan: {int(component_count)} component(s)"
                    )
                if isinstance(components, (list, tuple)):
                    for index, component in enumerate(components):
                        if not isinstance(component, Mapping):
                            continue
                        component_id = component.get("component_id", index)
                        selected = component.get("selected_strategy")
                        attempted = component.get("attempted_strategies", ())
                        details: list[str] = []
                        if selected:
                            details.append(f"selected {selected}")
                        if isinstance(attempted, (list, tuple)) and attempted:
                            details.append(
                                "attempted " + ", ".join(str(item) for item in attempted)
                            )
                        fallback = component.get("fallback_reason")
                        if fallback:
                            details.append(f"fallback {fallback}")
                        if details:
                            lines.append(
                                f"component {component_id}: " + "; ".join(details)
                            )
                source_faces = complex_geometry.get("source_face_count")
                final_faces = complex_geometry.get("final_face_count")
                if source_faces is not None and final_faces is not None:
                    lines.append(
                        f"complex geometry faces: {int(source_faces)} source, "
                        f"{int(final_faces)} final"
                    )
                junction_edges = complex_geometry.get("declared_junction_edge_count")
                if junction_edges is not None:
                    lines.append(
                        f"declared junction edges: {int(junction_edges)}"
                    )
                strategies = complex_geometry.get("source_strategy_by_face", {})
                if isinstance(strategies, Mapping) and strategies:
                    counts: dict[str, int] = {}
                    for strategy in strategies.values():
                        name = str(strategy)
                        counts[name] = counts.get(name, 0) + 1
                    lines.append(
                        "component strategy selection: "
                        + ", ".join(
                            f"{name}={counts[name]}" for name in sorted(counts)
                        )
                    )
                rejection = complex_geometry.get("alignment_candidate_rejected")
                if isinstance(rejection, Mapping):
                    lines.append(
                        "boundary alignment: attempted, rejected; "
                        + str(rejection.get("reason", "whole-mesh quality"))
                    )
            if "automatic_intersections" in record.summary:
                intersections = int(record.summary["automatic_intersections"])
                lines.append(
                    f"{intersections} automatic plate intersection(s) imprinted"
                )
            if "automatic_beam_connections" in record.summary:
                connections = int(record.summary["automatic_beam_connections"])
                lines.append(
                    f"{connections} automatic beam joint/shell connection(s) created"
                )
            if "automatic_shell_connections" in record.summary:
                connections = int(record.summary["automatic_shell_connections"])
                lines.append(
                    f"{connections} automatic shell T-junction tie(s) created"
                )
            requested_backend = record.summary.get("native_backend_requested")
            if requested_backend is not None:
                lines.append(f"native triangulator requested: {requested_backend}")
            provenance = record.summary.get("triangulation_backend_by_face", {})
            if isinstance(provenance, dict):
                routes = sorted(
                    {
                        (
                            str(values.get("requested_backend", "unknown")),
                            str(values.get("selected_backend", "unknown")),
                            str(values.get("actual_backend", "unknown")),
                        )
                        for values in provenance.values()
                        if isinstance(values, dict)
                    }
                )
                lines.extend(
                    "triangulator route: "
                    f"requested {requested}, selected {selected}, actual {actual}"
                    for requested, selected, actual in routes
                )
            optimisation_by_face = record.summary.get(
                "quality_optimization_by_face", {}
            )
            if isinstance(optimisation_by_face, dict):
                for face_id, optimisation in sorted(
                    optimisation_by_face.items(), key=lambda item: int(item[0])
                ):
                    if not isinstance(optimisation, Mapping):
                        continue
                    selected_strategy = optimisation.get("selected_strategy")
                    if selected_strategy:
                        lines.append(
                            f"face {face_id} native strategy: {selected_strategy}"
                        )
                    face_complex = optimisation.get("complex_geometry", {})
                    if isinstance(face_complex, Mapping):
                        alignment = face_complex.get("alignment_evaluation")
                        if alignment:
                            lines.append(
                                f"face {face_id} boundary alignment: {alignment}"
                            )
                    collar_skip = optimisation.get("boundary_collar_skip_reason")
                    if collar_skip:
                        lines.append(
                            f"face {face_id} collar fallback: {collar_skip}"
                        )
                    scalars = [
                        f"{key}={value}"
                        for key, value in optimisation.items()
                        if isinstance(value, (str, int, float, bool))
                        and key
                        not in {"selected_strategy", "boundary_collar_skip_reason"}
                    ]
                    if scalars:
                        lines.append(
                            f"face {face_id} quality optimization: "
                            + ", ".join(scalars[:6])
                        )
                    final_quality = optimisation.get("final_quality", {})
                    if isinstance(final_quality, Mapping) and final_quality:
                        final_parts: list[str] = []
                        if final_quality.get("min_scaled_jacobian") is not None:
                            final_parts.append(
                                "min Jacobian "
                                f"{float(final_quality['min_scaled_jacobian']):.4g}"
                            )
                        if final_quality.get("min_angle") is not None:
                            final_parts.append(
                                f"angles {float(final_quality['min_angle']):.4g}°–"
                                f"{float(final_quality.get('max_angle', 0.0)):.4g}°"
                            )
                        poor = final_quality.get("poor_element_ids", ())
                        if poor:
                            final_parts.append(f"{len(poor)} poor element(s)")
                        if final_parts:
                            lines.append(
                                f"face {face_id} final quality: "
                                + ", ".join(final_parts)
                            )
            quality = record.summary.get("quality", {})
            if quality:
                lines.extend(
                    (
                        f"max aspect ratio {float(quality.get('max_aspect_ratio', 0.0)):.4g}",
                        f"mean aspect ratio {float(quality.get('mean_aspect_ratio', 0.0)):.4g}",
                        f"max warp {float(quality.get('max_warp', 0.0)):.4g}",
                    )
                )
                if quality.get("min_scaled_jacobian") is not None:
                    lines.append(
                        "min scaled Jacobian "
                        f"{float(quality['min_scaled_jacobian']):.4g}"
                    )
                if quality.get("min_angle") is not None:
                    lines.append(
                        f"angle range {float(quality['min_angle']):.4g}° to "
                        f"{float(quality.get('max_angle', 0.0)):.4g}°"
                    )
                poor_ids = quality.get("poor_element_ids", ())
                if poor_ids:
                    lines.append(f"poor-quality elements: {len(poor_ids)}")
                warnings = quality.get("warnings", ())
                lines.extend(f"warning: {warning}" for warning in warnings)
            if record.diagnostics:
                diagnostic = record.diagnostics[-1]
                message = (
                    diagnostic.get("message", str(diagnostic))
                    if isinstance(diagnostic, dict)
                    else str(diagnostic)
                )
                lines.append(f"error: {message}")
                details = (
                    diagnostic.get("details")
                    if isinstance(diagnostic, Mapping)
                    else None
                )
                admission = (
                    details.get("admission")
                    if isinstance(details, Mapping)
                    else None
                )
                if isinstance(admission, Mapping):
                    total = int(admission.get("element_count", 0))
                    failing = int(admission.get("failing_element_count", 0))
                    topology = admission.get("topology_violations", ())
                    lines.append(
                        f"qualified S3 admission: {failing}/{total} "
                        "triangle(s) failed"
                    )
                    if isinstance(topology, (list, tuple)) and topology:
                        lines.append(
                            f"topology violations: {len(topology)}; "
                            f"first: {topology[0]}"
                        )
                    failing_elements = admission.get("failing_elements", ())
                    if (
                        isinstance(failing_elements, (list, tuple))
                        and failing_elements
                        and isinstance(failing_elements[0], Mapping)
                    ):
                        first = failing_elements[0]
                        violations = first.get("violations", ())
                        lines.append(
                            f"first failing triangle {first.get('element_id')}: "
                            + "; ".join(str(item) for item in violations)
                        )
                repair = (
                    details.get("repair")
                    if isinstance(details, Mapping)
                    else None
                )
                if isinstance(repair, Mapping):
                    lines.append(
                        f"qualified S3 repair attempts: "
                        f"{int(repair.get('attempt_count', 0))}"
                    )
        if mesh is not None:
            beams = "3-node" if mesh.is_quadratic else "2-node"
            lines.extend(
                (
                    f"{mesh.num_nodes} nodes",
                    f"{len(mesh.shells)} shell elements",
                    f"{len(mesh.beams)} {beams} beam elements",
                )
            )
        self._stats.configure(text="\n".join(lines) if lines else "no mesh")

    def _copy_mesh_diagnosis(self) -> None:
        report = str(self._stats.cget("text")).strip()
        if not report or report == "no mesh":
            self.app.set_status("no mesh diagnosis is available")
            return
        payload = "ANYfem mesh diagnosis\n\n" + report + "\n"
        record_id = (
            getattr(self.app, "_active_mesh_task_id", None)
            or getattr(self.app, "_mesh_details_record_id", None)
            or getattr(self.app, "mesh_record_id", None)
        )
        record = self.app.project.mesh_records.get(record_id or "")
        if record is None and self.app.project.mesh_records:
            record = next(reversed(self.app.project.mesh_records.values()))
        if record is not None and record.diagnostics:
            payload += (
                "\nStructured error evidence\n"
                + json.dumps(
                    record.diagnostics[-1],
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
        try:
            self.app.clipboard.copy_text(payload)
        except tk.TclError as error:
            self.app.set_status(
                f"could not copy mesh diagnosis: {error}",
                error=True,
                diagnostic={"type": type(error).__name__, "message": str(error)},
            )
            return
        self.app.set_status(
            f"mesh diagnosis copied ({len(payload):,} characters)"
        )

    def _generate(self) -> None:
        size = self.number(self._size, "element size")
        if size <= 0:
            raise ValueError("element size must be positive")
        strategy = self._method_value()
        preference = self._structure_preference_value()
        quality_policy = self._quality_policy_value() if strategy == "auto" else None
        if strategy == "mapped" and self._strict_method.get():
            eligible, diagnostic = mapped_mesh_eligibility(
                self.app.project.geometry
            )
            if not eligible:
                raise ValueError(diagnostic)
        backend = (
            None
            if strategy == "mapped"
            else self._NATIVE_BACKEND_VALUES[self._native_backend.get()]
        )
        options = {
            "native_backend": backend,
            "strategy": strategy,
            "structure_preference": preference,
            "mesh_controls": self._mesh_controls_value(),
            "structured_controls": self._structured_controls_value(),
        }
        if strategy == "quad_first":
            options["quad_options"] = self._quad_options_value()
            options["layout_policy"] = "adaptive" if self._quad_layout.get() else "existing"
        if quality_policy is not None:
            options["quality_policy"] = quality_policy
        try:
            from anymesher import MeshAutomationOptions
        except ImportError as error:
            raise ValueError(
                "Automatic mesh recovery needs a compatible ANYmesher installation"
            ) from error

        options["automation"] = MeshAutomationOptions(
            strict_method=self._strict_method.get()
        )
        # Validate every active control before any undoable model mutation.
        if self._order.get() not in ELEMENT_ORDERS:
            raise ValueError(f"unknown element order {self._order.get()!r}")
        if self._order.get() != self.app.project.element_order:
            self.app.run(cmd.SetElementOrder(order=self._order.get()))
        self.app.generate_mesh_async(size, **options)
        self._order_dirty = False
        self._method_dirty = False
        self._preference_dirty = False
        self._controls_dirty = False

    def _quad_options_value(self):
        from anymesher.quad.options import QuadMeshingOptions

        optimization = self._quad_optimization.get()
        return QuadMeshingOptions(
            orientation=self._quad_orientation.get(),
            quality_model=self._quad_quality.get(),
            line_search=optimization,
            max_front_iterations=int(self.number(self._quad_front, "front iterations")),
            max_local_optimizations=(
                0 if optimization == "off" else int(self.number(self._quad_local, "local optimizations"))
            ),
            cancellation_interval=int(self.number(self._quad_cancel, "cancel interval")),
        )

    def _preview_layout(self) -> None:
        size = self.number(self._size, "element size")
        if size <= 0:
            raise ValueError("element size must be positive")
        if self._method_value() != "auto":
            raise ValueError("layout preview is available for Automatic meshing")
        plan = self.app.preview_mesh_layout(
            size,
            structure_preference=self._structure_preference_value(),
            quality_policy=self._quality_policy_value(),
            structured_controls=self._structured_controls_value(),
        )
        structured = sum(item.structured for item in plan.faces)
        residual = len(plan.faces) - structured
        self._preview_status.configure(
            text=(
                f"Preview {plan.plan_hash[-12:]}: {len(plan.blocks)} block(s), "
                f"{len(plan.interfaces)} shared interface(s), {structured} structured "
                f"face(s), {residual} native residual(s), approximately "
                f"{plan.estimated_element_count} element(s)."
            ),
            foreground="#16667a",
        )
        self._update_method_controls()

    def _commit_layout(self) -> None:
        report = self.app.commit_mesh_layout_preview()
        self._preview_status.configure(
            text=(
                f"Committed feature: {sum(len(value) for value in report.source_to_working_faces.values())} "
                "materialized plate region(s). Generate mesh when ready."
            ),
            foreground="#267326",
        )
        self._update_method_controls()

    def _discard_layout(self) -> None:
        self.app.clear_mesh_layout_preview()
        self._preview_status.configure(
            text="Preview discarded; the model was not changed.",
            foreground="#666666",
        )
        self._update_method_controls()

    def _cancel(self) -> None:
        self.app.cancel_mesh()

    def _open_mesher(self) -> None:
        """Open ANYmesher as a standalone neutral-mesh tool.

        Its parametric primitives are not associated with this project's BRep,
        so presenting its generic mesh as analysis-ready here would leave
        sections, loads and supports referring to unrelated entity IDs.
        """

        from anymesher.gui import open_mesher

        open_mesher(self.winfo_toplevel())
        self.app.set_status(
            "opened standalone ANYmesher; save its neutral mesh from that window"
        )

    def _refine(self) -> None:
        """Refine around whatever is selected, whatever kind it is."""

        selection = list(self.app.selection)
        if not selection:
            raise ValueError(
                "select a point, line or plate to refine around first"
            )
        size = self.number(self._refine_size, "element size")
        radius = self.number(self._refine_radius, "radius")
        self.app.run_many(
            cmd.AddRefinement(refinement=refine_around(ref, size, radius))
            for ref in selection
        )
        self.app.set_status(
            f"refining to {size:g} m around {len(selection)} entity(ies); "
            "generate the mesh to apply it"
        )
        self.app.refresh_panels()

    def _clear_refinements(self) -> None:
        self.app.project.refinements.clear()
        self.app.set_status("cleared refinement zones")
        self.app.refresh_panels()

    def _pin(self) -> None:
        edges = self.require_selection("edge")
        count = int(self.number(self._divisions, "divisions"))
        if count < 1:
            raise ValueError("divisions must be at least 1")
        for ref in edges:
            self.app.seeding_overrides[ref.id] = count
        self.app.set_status(f"pinned {len(edges)} line(s) to {count} divisions")
        self.app.refresh_panels()

    def _clear_pins(self) -> None:
        self.app.seeding_overrides.clear()
        self.app.set_status("cleared seeding pins")
        self.app.refresh_panels()


# ----------------------------------------------------------------------
class SectionPanel(StagePanel):
    title = "Sections"

    def build(self) -> None:
        self._editing_imperfection = None
        material = self.section("Material definition")
        self._material_name = self.entry_row(material, "material name", "auto")
        grade_row = ttk.Frame(material)
        grade_row.pack(fill="x", pady=1)
        ttk.Label(grade_row, text="DNV RP-C208 grade", width=16).pack(side="left")
        self._grade = tk.StringVar(value="S355")
        self._grade_box = ttk.Combobox(
            grade_row,
            textvariable=self._grade,
            values=tuple(available_grades()),
            state="readonly",
            width=12,
        )
        self._grade_box.pack(side="left", fill="x", expand=True)
        self._grade_thickness = self.entry_row(material, "thickness [mm]", "10")
        self.button(material, "Add DNV steel", self._add_material)
        ttk.Label(
            material,
            text=(
                "Name is an editable project label (for example S355_NL). "
                "Grade selects the DNV properties and is kept separate."
            ),
            foreground="#666666",
            wraplength=430,
            justify="left",
        ).pack(anchor="w", pady=(1, 3))
        self.button(material, "Open ANYmaterial...", self._open_material_editor)
        library_row = ttk.Frame(material)
        library_row.pack(fill="x", pady=(4, 1))
        ttk.Label(library_row, text="inspect material", width=16).pack(side="left")
        self._material_choice = tk.StringVar(value="S355")
        self._material_choice_box = ttk.Combobox(
            library_row,
            textvariable=self._material_choice,
            values=(),
            state="readonly",
            width=24,
        )
        self._material_choice_box.pack(side="left", fill="x", expand=True)
        self._material_choice_box.bind(
            "<<ComboboxSelected>>", lambda _event: self._refresh_material_details()
        )
        self._material_details = ttk.Label(
            material, text="", justify="left", foreground="#333333"
        )
        self._material_details.pack(anchor="w", fill="x", pady=(2, 4))

        plate = self.section("Plate section")
        self._plate_name = self.entry_row(plate, "name", "plate")
        self._plate_thickness = self.entry_row(plate, "thickness [mm]", "10")
        self._plate_material, self._plate_material_box = self._material_row(plate)
        self._auto_dnv_plate = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            plate,
            text="Auto DNV nonlinear material from thickness",
            variable=self._auto_dnv_plate,
            command=self._update_plate_material_mode,
        ).pack(anchor="w", pady=(2, 0))
        ttk.Label(
            plate,
            text=(
                "Uses the DNV grade above and this plate thickness; an automatic "
                "thickness-qualified material name is generated and reused."
            ),
            foreground="#666666",
            wraplength=430,
            justify="left",
        ).pack(anchor="w", pady=(0, 2))
        self.button(plate, "Create / update section", self._add_plate_section)
        self.button(plate, "Assign to selected plates", self._assign_plate)

        beam = self.section("Beam section")
        self._beam_name = self.entry_row(beam, "name", "stiffener")
        row = ttk.Frame(beam)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="profile", width=16).pack(side="left")
        self._profile = tk.StringVar(value="T-bar")
        ttk.Combobox(
            row, textvariable=self._profile, values=list(PROFILES),
            state="readonly", width=12,
        ).pack(side="left", fill="x", expand=True)
        self._beam_dims = self.vector_row(beam, "hw, tw, b [mm]", ("200", "10", "100"))
        self._beam_flange = self.entry_row(beam, "tf [mm]", "12")
        self._beam_material, self._beam_material_box = self._material_row(beam)
        offset_row = ttk.Frame(beam)
        offset_row.pack(fill="x", pady=1)
        ttk.Label(offset_row, text="neutral-axis offset", width=16).pack(side="left")
        self._beam_offset_mode = tk.StringVar(value="Automatic")
        self._beam_offset_box = ttk.Combobox(
            offset_row,
            textvariable=self._beam_offset_mode,
            values=("Automatic", "Manual", "Centreline"),
            state="readonly",
            width=12,
        )
        self._beam_offset_box.pack(side="left", fill="x", expand=True)
        self._beam_offset_box.bind(
            "<<ComboboxSelected>>", lambda _event: self._update_beam_offset_mode()
        )
        side_row = ttk.Frame(beam)
        side_row.pack(fill="x", pady=1)
        ttk.Label(side_row, text="plate side", width=16).pack(side="left")
        self._beam_side = tk.StringVar(value="Front")
        ttk.Combobox(
            side_row,
            textvariable=self._beam_side,
            values=("Front", "Back"),
            state="readonly",
            width=12,
        ).pack(side="left", fill="x", expand=True)
        manual_offset_row = ttk.Frame(beam)
        manual_offset_row.pack(fill="x", pady=1)
        ttk.Label(
            manual_offset_row, text="manual offset [mm]", width=16
        ).pack(side="left")
        self._beam_manual_offset = tk.StringVar(value="0")
        self._beam_manual_offset_entry = ttk.Entry(
            manual_offset_row,
            textvariable=self._beam_manual_offset,
            width=10,
        )
        self._beam_manual_offset_entry.pack(side="left", fill="x", expand=True)
        self._beam_rotation = self.entry_row(beam, "axial rotation [deg]", "0")
        ttk.Label(
            beam,
            text=(
                "Automatic places the neutral axis at the section centroid. "
                "Front follows the plate normal; Back mirrors it. Rotation is "
                "about the model line at the plate intersection."
            ),
            foreground="#666666",
            wraplength=430,
            justify="left",
        ).pack(anchor="w", pady=(1, 3))
        self.button(beam, "Create / update section", self._add_beam_section)
        self.button(beam, "Assign to selected lines", self._assign_beam)
        self._update_beam_offset_mode()

        usage = self.section("Section definitions and assigned model entities")
        self._section_usage = ttk.Treeview(
            usage,
            columns=("kind", "material", "assigned"),
            show="tree headings",
            height=6,
        )
        self._section_usage.heading("#0", text="Definition")
        self._section_usage.heading("kind", text="Type")
        self._section_usage.heading("material", text="Material")
        self._section_usage.heading("assigned", text="Assigned model entities")
        self._section_usage.column("#0", width=100, stretch=True)
        self._section_usage.column("kind", width=55, stretch=False)
        self._section_usage.column("material", width=165, stretch=True)
        self._section_usage.column("assigned", width=190, stretch=True)
        self._section_usage.pack(fill="x")

        imperfection = self.section("Imperfection")
        self._imperfection_amplitude = self.entry_row(
            imperfection, "amplitude [mm]", "auto"
        )
        self._waves = self.entry_row(imperfection, "waves", "1 1")
        self._imperfection_apply = self.button(
            imperfection, "Add to selection", self._add_imperfection
        )
        ttk.Label(
            imperfection,
            text=(
                "Purple wire shows the stress-free imperfect shape. Small "
                "amplitudes are automatically exaggerated for visibility; "
                "the tree and solver retain the stated physical amplitude."
            ),
            foreground="#6a1b9a",
            wraplength=430,
            justify="left",
        ).pack(anchor="w", pady=(2, 0))
        self._imperfection_label = ttk.Label(self, text="", foreground="#666666")
        self._imperfection_label.pack(anchor="w")
        self._update_plate_material_mode()

    def refresh(self) -> None:
        names = tuple(sorted(self.app.project.materials))
        self._material_choice_box.configure(values=names)
        if names and self._material_choice.get() not in names:
            self._material_choice.set(names[0])
        for variable, box in (
            (self._plate_material, self._plate_material_box),
            (self._beam_material, self._beam_material_box),
        ):
            box.configure(values=names)
            if names and variable.get() not in names:
                variable.set(names[0])
        count = len(self.app.project.imperfections)
        self._imperfection_label.configure(
            text="no imperfections" if not count else f"{count} imperfection(s)"
        )
        self._update_plate_material_mode()
        self._refresh_material_details()
        self._refresh_section_usage()

    def _refresh_material_details(self) -> None:
        material = self.app.project.materials.get(self._material_choice.get())
        if material is None:
            self._material_details.configure(text="No material selected")
            return
        lines = [f"{material.name}"]
        if material.symmetry == "isotropic":
            lines.append(
                f"Elastic: E = {material.elastic_modulus / 1e9:.6g} GPa, "
                f"nu = {material.poisson_ratio:.5g}"
            )
        else:
            lines.append(f"Elastic symmetry: {material.symmetry}")
        lines.append(
            f"Density = {material.density:.6g} kg/m3; "
            f"nominal yield = {material.yield_stress / 1e6:.6g} MPa"
        )
        hardening = material.hardening
        if hardening is None:
            lines.append("Response: ELASTIC ONLY (no plastic hardening curve)")
        else:
            kind = str(hardening.get("kind", "custom"))
            lines.append(f"Response: NONLINEAR PLASTICITY ACTIVE ({kind})")
            if kind == "dnv_c208":
                lines.append(
                    f"DNV grade {hardening.get('grade', '?')}; "
                    f"product thickness = "
                    f"{float(hardening.get('thickness', 0.0)) * 1000:.6g} mm"
                )
            try:
                curve = material.build().hardening_curve
                plastic_strain = np.array((0.0, 0.01, 0.10), dtype=float)
                flow = np.asarray(curve.flow_stress(plastic_strain), dtype=float) / 1e6
                lines.append(
                    "Flow stress [MPa] at plastic strain "
                    f"0/1%/10% = {flow[0]:.4g} / {flow[1]:.4g} / {flow[2]:.4g}"
                )
            except (AttributeError, TypeError, ValueError):
                lines.append("Hardening curve is stored and validated by ANYmaterial")
        self._material_details.configure(text="\n".join(lines))

    @staticmethod
    def _assigned_label(prefix: str, identifiers: Sequence[int]) -> str:
        ordered = sorted(int(value) for value in identifiers)
        shown = ", ".join(str(value) for value in ordered[:12])
        if len(ordered) > 12:
            shown += f", +{len(ordered) - 12} more"
        return "unassigned" if not ordered else f"{prefix} {shown}"

    def _refresh_section_usage(self) -> None:
        self.app.project.resolve_section_assignments(strict=False)
        self._section_usage.delete(*self._section_usage.get_children())
        faces_by_section: dict[str, list[int]] = {}
        for identifier, assigned in self.app.project.face_sections.items():
            faces_by_section.setdefault(assigned, []).append(identifier)
        edges_by_section: dict[str, list[int]] = {}
        for identifier, assigned in self.app.project.edge_sections.items():
            edges_by_section.setdefault(assigned, []).append(identifier)
        for name, section in sorted(self.app.project.plate_sections.items()):
            faces = faces_by_section.get(name, ())
            self._section_usage.insert(
                "", "end", iid=f"plate:{section.id}", text=name,
                values=("Plate", section.material, self._assigned_label("Plates", faces)),
            )
        for name, section in sorted(self.app.project.beam_sections.items()):
            edges = edges_by_section.get(name, ())
            self._section_usage.insert(
                "", "end", iid=f"beam:{section.id}", text=name,
                values=("Beam", section.material, self._assigned_label("Lines", edges)),
            )

    def _update_plate_material_mode(self) -> None:
        self._plate_material_box.configure(
            state="disabled" if self._auto_dnv_plate.get() else "readonly"
        )

    def _update_beam_offset_mode(self) -> None:
        self._beam_manual_offset_entry.configure(
            state="normal" if self._beam_offset_mode.get() == "Manual" else "disabled"
        )

    def edit_tree_item(self, key: str) -> bool:
        """Load a selected plate/beam definition into its editable controls."""

        if key.startswith("plate_section:"):
            name = key.split(":", 1)[1]
            section = self.app.project.plate_sections.get(name)
            if section is None:
                return False
            self._plate_name.set(section.name)
            self._plate_thickness.set(f"{section.thickness * 1000.0:.12g}")
            self._plate_material.set(section.material)
            self._auto_dnv_plate.set(False)
            self._update_plate_material_mode()
            return True
        if key.startswith("beam_section:"):
            name = key.split(":", 1)[1]
            section = self.app.project.beam_sections.get(name)
            if section is None:
                return False
            self._beam_name.set(section.name)
            self._profile.set(section.profile)
            for variable, value in zip(
                self._beam_dims,
                (
                    section.web_height,
                    section.web_thickness,
                    section.flange_width,
                ),
            ):
                variable.set(f"{value * 1000.0:.12g}")
            self._beam_flange.set(f"{section.flange_thickness * 1000.0:.12g}")
            self._beam_material.set(section.material)
            self._beam_offset_mode.set(
                {
                    "automatic": "Automatic",
                    "manual": "Manual",
                    "centerline": "Centreline",
                }[section.offset_mode]
            )
            self._beam_side.set(section.attachment_side.title())
            self._beam_manual_offset.set(f"{section.eccentricity * 1000.0:.12g}")
            self._beam_rotation.set(f"{section.rotation_deg:.12g}")
            self._update_beam_offset_mode()
            return True
        return False

    def _add_material(self) -> None:
        thickness = self.number(self._grade_thickness, "thickness") / 1000.0
        requested_name = self._material_name.get().strip()
        custom_name = None if requested_name.lower() in ("", "auto") else requested_name
        material = dnv_steel_material(
            self._grade.get().strip(),
            thickness,
            nonlinear=True,
            name=custom_name,
        )
        self.app.run(cmd.AddMaterial(material, label="add DNV material"))
        self._material_choice.set(material.name)
        self._plate_material.set(material.name)
        self._beam_material.set(material.name)
        self.app.set_status(
            f"added nonlinear DNV material {material.name}: "
            f"grade {material.hardening['grade']}, "
            f"product thickness {thickness * 1000:g} mm"
        )
        self.app.refresh_all()

    @staticmethod
    def _material_row(parent: tk.Misc) -> tuple[tk.StringVar, ttk.Combobox]:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="material", width=16).pack(side="left")
        variable = tk.StringVar(value="S355")
        box = ttk.Combobox(
            row, textvariable=variable, values=(), state="readonly", width=12
        )
        box.pack(side="left", fill="x", expand=True)
        return variable, box

    def _open_material_editor(self) -> None:
        from anymaterial.gui import open_material_editor

        selected = self._plate_material.get() or self._beam_material.get()
        initial = self.app.project.materials.get(selected)
        open_material_editor(
            self.winfo_toplevel(), initial_spec=initial, on_apply=self._use_material
        )

    def _use_material(self, material) -> None:
        """Add an editor result and select it for both new section types."""

        if not isinstance(material, Material):
            # Keep ANYfem's compatibility properties (elastic_modulus and
            # poisson_ratio) while storing the complete ANYmaterial schema.
            material = Material.from_dict(material.to_dict())
        self.app.run(cmd.AddMaterial(material, label="use ANYmaterial specification"))
        self._material_choice.set(material.name)
        self._auto_dnv_plate.set(False)
        self._update_plate_material_mode()
        self._plate_material.set(material.name)
        self._beam_material.set(material.name)
        self.app.set_status(f"using material {material.name}")
        self.app.refresh_all()

    def _plate_section_command(self) -> tuple[cmd.AddPlateSection, str]:
        name = self._plate_name.get().strip()
        thickness = self.number(self._plate_thickness, "thickness") / 1000.0
        automatic = self._auto_dnv_plate.get()
        material_name = self._plate_material.get().strip()
        material = None
        if automatic:
            material = dnv_steel_material(
                self._grade.get().strip(), thickness, nonlinear=True
            )
            material_name = material.name
        existing = self.app.project.plate_sections.get(name)
        section = PlateSection(
            name=name,
            thickness=thickness,
            material=material_name,
            **({"id": existing.id} if existing is not None else {}),
        )
        return cmd.AddPlateSection(section, material), material_name

    def _add_plate_section(self) -> None:
        command, material_name = self._plate_section_command()
        self.app.run(command)
        self._plate_material.set(material_name)
        self._material_choice.set(material_name)
        name = command.section.name
        detail = (
            "automatic DNV nonlinear"
            if command.material is not None
            else "selected"
        )
        self.app.set_status(
            f"added plate section {name} using {material_name} ({detail} material)"
        )
        self.app.refresh_all()

    def _assign_plate(self) -> None:
        faces = self.require_selection("face")
        name = self._plate_name.get().strip()
        definition, material_name = self._plate_section_command()
        self.app.run_many(
            (definition, *(cmd.AssignPlate(ref.id, name) for ref in faces))
        )
        self._plate_material.set(material_name)
        self._material_choice.set(material_name)
        self.app.set_status(
            f"updated {name} and assigned it to {len(faces)} plate(s); "
            f"material {material_name}"
        )

    def _beam_section_command(self) -> cmd.AddBeamSection:
        web_height, web_thickness, flange_width = (
            value / 1000.0 for value in self.vector(self._beam_dims, "beam dimension")
        )
        name = self._beam_name.get().strip()
        existing = self.app.project.beam_sections.get(name)
        offset_mode = {
            "Automatic": "automatic",
            "Manual": "manual",
            "Centreline": "centerline",
        }[self._beam_offset_mode.get()]
        section = BeamSection(
            name=name,
            profile=self._profile.get(),
            material=self._beam_material.get().strip(),
            web_height=web_height,
            web_thickness=web_thickness,
            flange_width=flange_width,
            flange_thickness=self.number(self._beam_flange, "tf") / 1000.0,
            eccentricity=(
                self.number(self._beam_manual_offset, "manual offset") / 1000.0
                if offset_mode == "manual"
                else 0.0
            ),
            offset_mode=offset_mode,
            attachment_side=self._beam_side.get().lower(),
            rotation_deg=self.number(self._beam_rotation, "axial rotation"),
            **({"id": existing.id} if existing is not None else {}),
        )
        return cmd.AddBeamSection(section)

    def _add_beam_section(self) -> None:
        command = self._beam_section_command()
        self.app.run(command)
        section = command.section
        self.app.set_status(
            f"created beam section {section.name}: {section.profile}, "
            f"{section.offset_mode} offset on {section.attachment_side}, "
            f"rotation {section.rotation_deg:g} deg"
        )
        self.app.refresh_all()

    def _assign_beam(self) -> None:
        edges = self.require_selection("edge")
        name = self._beam_name.get().strip()
        definition = self._beam_section_command()
        self.app.run_many((definition, *(cmd.AssignBeam(ref.id, name) for ref in edges)))
        self.app.set_status(
            f"updated {name} and assigned it to {len(edges)} line(s)"
        )

    def _add_imperfection(self) -> None:
        text = self._imperfection_amplitude.get().strip().lower()
        amplitude = None if text in ("", "auto") else (
            self.number(self._imperfection_amplitude, "amplitude") / 1000.0
        )
        try:
            waves = tuple(int(part) for part in self._waves.get().split())
        except ValueError:
            raise ValueError("waves must be two whole numbers, e.g. 1 1") from None
        if len(waves) != 2:
            raise ValueError("waves must be two whole numbers, e.g. 1 1")

        if isinstance(self._editing_imperfection, Imperfection):
            current = self._editing_imperfection
            self.app.run(
                cmd.EditAttribute(
                    replace(current, amplitude=amplitude, waves=waves)
                )
            )
            self._editing_imperfection = None
            self._imperfection_apply.configure(text="Add to selection")
            self.app.set_status(f"updated imperfection {current.name}")
            return

        items = self.require_selection(self.app.selection.mode)

        self.app.run_many(
            cmd.AddImperfection(
                Imperfection(ref=ref, amplitude=amplitude, waves=waves)
            )
            for ref in items
        )
        self.app.set_status(f"imperfection on {len(items)} entity(ies)")

    def edit_imperfection(self, identifier: str) -> bool:
        for item in self.app.project.imperfections:
            if getattr(item, "id", None) != identifier:
                continue
            self._editing_imperfection = item
            self._imperfection_amplitude.set(
                "auto" if item.amplitude is None else f"{item.amplitude * 1000:.12g}"
            )
            self._waves.set(f"{item.waves[0]} {item.waves[1]}")
            self._imperfection_apply.configure(text="Update selected imperfection")
            self.app.show_geometry()
            self.app.selection.set_mode(item.ref.kind)
            self.app.selection.select(item.ref)
            self.app.refresh_views()
            return True
        return False


class LoadPanel(StagePanel):
    title = "Loads & BC"

    def build(self) -> None:
        self._editing_attribute = None
        self._editing_case_name: str | None = None
        scope = self.section("Scope on model")
        ttk.Label(
            scope,
            text="Select model geometry for loads and boundary conditions.",
            foreground="#666666",
        ).pack(anchor="w")
        scope_row = ttk.Frame(scope)
        scope_row.pack(fill="x", pady=(3, 0))
        for text, kind in (
            ("Points", "vertex"),
            ("Lines", "edge"),
            ("Plates", "face"),
        ):
            ttk.Button(
                scope_row,
                text=text,
                command=lambda value=kind: self._scope_geometry(value),
            ).pack(side="left", fill="x", expand=True, padx=1)

        key = self.section("Viewport key")
        for text, colour in (
            ("blue arrow  pressure", COLOR_PRESSURE),
            ("red arrow  force / prescribed translation", COLOR_LOAD),
            ("orange arrow  moment", COLOR_MOMENT),
            ("green axes  translational restraint", COLOR_SUPPORT),
            ("teal axes  rotational restraint", COLOR_ROTATION),
            ("purple marker / arrow  mass / acceleration", COLOR_MASS),
        ):
            ttk.Label(key, text=text, foreground=colour).pack(anchor="w")

        cases = self.section("Load case")
        row = ttk.Frame(cases)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="case", width=16).pack(side="left")
        self._case = tk.StringVar(value="default")
        self._case_box = ttk.Combobox(
            row, textvariable=self._case, values=["default"], width=12
        )
        self._case_box.pack(side="left", fill="x", expand=True)
        self._case_box.bind(
            "<<ComboboxSelected>>", lambda _event: self.app.refresh_views()
        )
        coordinate_row = ttk.Frame(cases)
        coordinate_row.pack(fill="x", pady=1)
        ttk.Label(coordinate_row, text="coordinates", width=16).pack(side="left")
        self._coordinates = tk.StringVar(value="Global")
        self._coordinate_box = ttk.Combobox(
            coordinate_row,
            textvariable=self._coordinates,
            values=("Global",),
            state="readonly",
            width=18,
        )
        self._coordinate_box.pack(side="left", fill="x", expand=True)
        self.button(cases, "New case", self._add_case)
        self.button(cases, "Delete case", self._delete_case)
        self._follower = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            cases,
            text="pressure follows the deformed shape",
            variable=self._follower,
            command=self.guarded(self._set_follower),
        ).pack(anchor="w")

        support = self.section("Support")
        preset_row = ttk.Frame(support)
        preset_row.pack(fill="x", pady=1)
        ttk.Label(preset_row, text="preset", width=16).pack(side="left")
        self._support_preset = tk.StringVar(value="Pinned")
        preset = ttk.Combobox(
            preset_row,
            textvariable=self._support_preset,
            values=(
                "Fixed", "Pinned", "Simply supported Z", "Prescribed",
                "X symmetry", "Y symmetry", "Z symmetry",
                "X antisymmetry", "Y antisymmetry", "Z antisymmetry",
            ),
            state="readonly",
            width=20,
        )
        preset.pack(side="left", fill="x", expand=True)
        preset.bind("<<ComboboxSelected>>", self._apply_support_preset)
        grid = ttk.Frame(support)
        grid.pack(fill="x")
        self._dofs = {}
        self._values = {}
        self._component_values = {}
        for index, dof in enumerate(DOF_NAMES):
            held = tk.BooleanVar(value=dof in ("ux", "uy", "uz"))
            value = tk.StringVar(value="0")
            ttk.Checkbutton(grid, text=dof, variable=held).grid(
                row=index, column=0, sticky="w", padx=2
            )
            ttk.Entry(grid, textvariable=value, width=12).grid(
                row=index, column=1, sticky="ew", padx=2, pady=1
            )
            ttk.Label(
                grid, text="mm" if dof.startswith("u") else "mrad"
            ).grid(row=index, column=2, sticky="w")
            self._dofs[dof] = held
            self._component_values[dof] = value
        grid.columnconfigure(1, weight=1)
        # Historical integrations set this one value.  Keep it as an explicit
        # convenience while the six component boxes above are authoritative.
        self._prescribed = self.entry_row(
            support, "set checked [mm/mrad]", "0"
        )
        self._prescribed.trace_add("write", self._broadcast_prescribed)
        self._support_apply = self.button(
            support, "Apply to selection", self._add_support
        )

        load = self.section("Load")
        self._pressure = self.entry_row(load, "pressure [Pa]", "10000")
        self._pressure_apply = self.button(
            load, "Pressure on plates", self._add_pressure
        )
        self._force = self.vector_row(load, "force [N]", ("0", "0", "-1000"))
        self._moment = self.vector_row(load, "moment [Nm]", ("0", "0", "0"))
        policy_row = ttk.Frame(load)
        policy_row.pack(fill="x", pady=1)
        ttk.Label(policy_row, text="multi-point policy", width=16).pack(side="left")
        self._point_policy = tk.StringVar(value="Per target")
        ttk.Combobox(
            policy_row,
            textvariable=self._point_policy,
            values=("Per target", "Total distributed"),
            state="readonly",
            width=18,
        ).pack(side="left", fill="x", expand=True)
        self._point_apply = self.button(
            load, "Point load on points", self._add_point_load
        )
        self._line = self.vector_row(load, "line load [N/m]", ("0", "0", "-1000"))
        self._line_apply = self.button(
            load, "Line load on lines", self._add_line_load
        )
        self._traction = self.vector_row(load, "traction [Pa]", ("0", "0", "-1000"))
        self._traction_apply = self.button(
            load, "Traction on plates", self._add_traction
        )

        body = self.section("Body load")
        self._acceleration = self.vector_row(
            body, "accel [m/s2]", ("0", "0", "-9.81")
        )
        self.button(body, "Apply to case", self._set_acceleration)
        self._mass = self.entry_row(body, "mass [kg]", "100")
        self._mass_apply = self.button(
            body, "Mass on selection", self._add_mass
        )

        combination = self.section("Combination")
        self._combination_name = self.entry_row(combination, "name", "ULS")
        self._factors = self.entry_row(combination, "factors", "default 1.0")
        ttk.Label(
            combination,
            text="factors: case factor, case factor, ...",
            foreground="#666666",
        ).pack(anchor="w")
        self.button(combination, "Define", self._add_combination)
        self._combination_label = ttk.Label(self, text="", justify="left")
        self._combination_label.pack(anchor="w")

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        names = sorted(self.app.project.load_cases) or ["default"]
        self._case_box.configure(values=names)
        if self._case.get() not in names:
            self._case.set(names[0])
        case = self.app.project.load_cases.get(self._case.get())
        if case is not None:
            self._follower.set(case.follower_pressure)
        coordinate_names = [
            item.name
            for item in sorted(
                self.app.project.coordinate_systems.values(),
                key=lambda item: (item.id != "global", item.name.lower()),
            )
        ]
        self._coordinate_box.configure(values=coordinate_names)
        if self._coordinates.get() not in coordinate_names:
            self._coordinates.set(coordinate_names[0] if coordinate_names else "Global")
        combinations = self.app.project.combinations
        self._combination_label.configure(
            text="no combinations"
            if not combinations
            else "\n".join(
                f"{name}: "
                + ", ".join(
                    f"{case_name} x {factor:g}"
                    for case_name, factor in sorted(item.factors.items())
                )
                for name, item in sorted(combinations.items())
            )
        )

    def case_name(self) -> str:
        return self._case.get().strip() or "default"

    def _scope_geometry(self, kind: str) -> None:
        labels = {"vertex": "points", "edge": "lines", "face": "plates"}
        self.app.show_geometry()
        self.app.selection_strip.set_context(
            kind,
            f"Geometry scope • select {labels[kind]}",
        )
        self.app.details.set_hint(f"Select model {labels[kind]}")
        self.app.set_status(
            f"model geometry shown; select {labels[kind]} for loads and BCs"
        )

    def coordinate_system_id(self) -> str:
        selected = self._coordinates.get()
        for identifier, system in self.app.project.coordinate_systems.items():
            if system.name == selected or identifier == selected:
                return identifier
        raise ValueError(f"coordinate system {selected!r} does not exist")

    # ------------------------------------------------------------------
    def _add_case(self) -> None:
        name = self._case.get().strip()
        if not name:
            raise ValueError("give the load case a name")
        self.app.run(cmd.AddLoadCase(name))
        self.app.set_status(f"load case {name}")
        self.app.refresh_all()

    def _delete_case(self) -> None:
        name = self.case_name()
        self.app.run(cmd.DeleteLoadCase(name))
        self.app.set_status(f"deleted load case {name}")
        self.app.refresh_all()

    def _set_follower(self) -> None:
        self.app.run(
            cmd.SetFollowerPressure(self._follower.get(), case=self.case_name())
        )
        self.app.set_status(
            "pressure follows the deformed shape"
            if self._follower.get()
            else "pressure acts on the reference shape"
        )

    @staticmethod
    def _set_vector(variables: Sequence[tk.StringVar], values: Sequence[float]) -> None:
        for variable, value in zip(variables, values):
            variable.set(f"{float(value):.12g}")

    def _set_coordinate_system(self, identifier: str) -> None:
        system = self.app.project.coordinate_systems.get(identifier)
        self._coordinates.set(system.name if system is not None else identifier)

    def _begin_attribute_edit(self, item, case_name: str | None = None) -> None:
        """Hydrate the form from the UUID-selected tree object."""

        self._editing_attribute = item
        self._editing_case_name = case_name
        if case_name is not None:
            self._case.set(case_name)
        if hasattr(item, "coordinate_system_id"):
            self._set_coordinate_system(item.coordinate_system_id)
        self.app.show_geometry()
        self.app.selection.set_mode(item.ref.kind)
        self.app.selection.select(item.ref)

        if isinstance(item, Support):
            self._support_preset.set("Prescribed")
            for dof in DOF_NAMES:
                active = dof in item.constraints
                self._dofs[dof].set(active)
                self._component_values[dof].set(
                    f"{1000.0 * float(item.constraints.get(dof, 0.0)):.12g}"
                )
            self._support_apply.configure(text="Update selected support")
        elif isinstance(item, PointLoad):
            self._set_vector(self._force, item.force)
            self._set_vector(self._moment, item.moment)
            self._point_policy.set(
                "Per target"
                if item.distribution_policy == "per_target"
                else "Total distributed"
            )
            self._point_apply.configure(text="Update selected point load")
        elif isinstance(item, Pressure):
            self._pressure.set(f"{float(item.value):.12g}")
            self._pressure_apply.configure(text="Update selected pressure")
        elif isinstance(item, LineLoad):
            self._set_vector(self._line, item.force_per_length)
            self._line_apply.configure(text="Update selected line load")
        elif isinstance(item, SurfaceTraction):
            self._set_vector(self._traction, item.traction)
            self._traction_apply.configure(text="Update selected traction")
        elif isinstance(item, Mass):
            self._mass.set(f"{float(item.value):.12g}")
            self._point_policy.set(
                "Per target"
                if item.distribution_policy == "per_target"
                else "Total distributed"
            )
            self._mass_apply.configure(text="Update selected mass")
        self.app.refresh_views()

    def edit_tree_item(self, key: str) -> bool:
        """Open the exact support/load/mass selected in the model tree."""

        if key.startswith("support:"):
            identifier = key.split(":", 1)[1]
            for item in self.app.project.supports:
                if item.id == identifier:
                    self._begin_attribute_edit(item)
                    return True
        if key.startswith("mass:"):
            identifier = key.split(":", 1)[1]
            for item in self.app.project.masses:
                if item.id == identifier:
                    self._begin_attribute_edit(item)
                    return True
        if key.startswith("load:") and not key.endswith(":gravity"):
            identifier = key.rsplit(":", 1)[1]
            for case_name, case in self.app.project.load_cases.items():
                for attribute in (
                    "point_loads",
                    "pressures",
                    "line_loads",
                    "surface_tractions",
                ):
                    for item in getattr(case, attribute):
                        if item.id == identifier:
                            self._begin_attribute_edit(item, case_name)
                            return True
        return False

    def _finish_attribute_edit(self) -> None:
        self._editing_attribute = None
        self._editing_case_name = None
        self._support_apply.configure(text="Apply to selection")
        self._pressure_apply.configure(text="Pressure on plates")
        self._point_apply.configure(text="Point load on points")
        self._line_apply.configure(text="Line load on lines")
        self._traction_apply.configure(text="Traction on plates")
        self._mass_apply.configure(text="Mass on selection")

    def _add_support(self) -> None:
        constraints = {
            dof: self.number(self._component_values[dof], dof) / 1000.0
            for dof, held in self._dofs.items()
            if held.get()
        }
        if not constraints:
            raise ValueError("tick at least one degree of freedom")
        if isinstance(self._editing_attribute, Support):
            current = self._editing_attribute
            updated = replace(
                current,
                constraints=dict(constraints),
                coordinate_system_id=self.coordinate_system_id(),
            )
            self.app.run(cmd.EditAttribute(updated))
            self._finish_attribute_edit()
            self.app.set_status(f"updated {current.name}")
            return
        items = self.require_selection(self.app.selection.mode)
        self.app.run_many(
            cmd.AddSupport(
                Support(
                    name=f"support_{ref}",
                    ref=ref,
                    constraints=dict(constraints),
                    coordinate_system_id=self.coordinate_system_id(),
                )
            )
            for ref in items
        )
        what = "prescribed" if any(constraints.values()) else "supported"
        self.app.set_status(f"{what} {len(items)} entity(ies)")

    def _broadcast_prescribed(self, *_args) -> None:
        value = self._prescribed.get()
        for dof, held in self._dofs.items():
            if held.get():
                self._component_values[dof].set(value)

    def _apply_support_preset(self, _event=None) -> None:
        name = self._support_preset.get()
        mapping = {
            "Fixed": set(DOF_NAMES),
            "Pinned": {"ux", "uy", "uz"},
            "Simply supported Z": {"uz"},
            "X symmetry": {"ux", "ry", "rz"},
            "Y symmetry": {"uy", "rx", "rz"},
            "Z symmetry": {"uz", "rx", "ry"},
            "X antisymmetry": {"uy", "uz", "rx"},
            "Y antisymmetry": {"ux", "uz", "ry"},
            "Z antisymmetry": {"ux", "uy", "rz"},
        }
        if name == "Prescribed":
            return
        selected = mapping.get(name, set())
        for dof, held in self._dofs.items():
            held.set(dof in selected)
            if dof in selected:
                self._component_values[dof].set("0")

    def _add_pressure(self) -> None:
        value = self.number(self._pressure, "pressure")
        if isinstance(self._editing_attribute, Pressure):
            current = self._editing_attribute
            self.app.run(cmd.EditAttribute(replace(current, value=value)))
            self._finish_attribute_edit()
            self.app.set_status("updated pressure")
            return
        faces = self.require_selection("face")
        self.app.run_many(
            cmd.AddPressure(ref, value, case=self.case_name())
            for ref in faces
        )
        self.app.set_status(f"pressure on {len(faces)} plate(s)")

    def _add_point_load(self) -> None:
        force = self.vector(self._force, "force")
        moment = self.vector(self._moment, "moment")
        policy = (
            "per_target"
            if self._point_policy.get() == "Per target"
            else "total_distributed"
        )
        if isinstance(self._editing_attribute, PointLoad):
            current = self._editing_attribute
            self.app.run(
                cmd.EditAttribute(
                    replace(
                        current,
                        force=force,
                        moment=moment,
                        coordinate_system_id=self.coordinate_system_id(),
                        distribution_policy=policy,
                    )
                )
            )
            self._finish_attribute_edit()
            self.app.set_status("updated point force/moment")
            return
        points = self.require_selection("vertex")
        scale = 1.0 if policy == "per_target" else 1.0 / len(points)
        self.app.run_many(
            cmd.AddPointLoad(
                ref,
                tuple(value * scale for value in force),
                tuple(value * scale for value in moment),
                case=self.case_name(),
                coordinate_system_id=self.coordinate_system_id(),
                distribution_policy=policy,
            )
            for ref in points
        )
        self.app.set_status(
            f"point force/moment on {len(points)} point(s), "
            f"{self._point_policy.get().lower()}"
        )

    def _add_line_load(self) -> None:
        intensity = self.vector(self._line, "line load")
        if isinstance(self._editing_attribute, LineLoad):
            current = self._editing_attribute
            self.app.run(
                cmd.EditAttribute(
                    replace(
                        current,
                        force_per_length=intensity,
                        coordinate_system_id=self.coordinate_system_id(),
                    )
                )
            )
            self._finish_attribute_edit()
            self.app.set_status("updated line load")
            return
        edges = self.require_selection("edge")
        self.app.run_many(
            cmd.AddLineLoad(
                ref,
                tuple(intensity),
                case=self.case_name(),
                coordinate_system_id=self.coordinate_system_id(),
            )
            for ref in edges
        )
        self.app.set_status(f"line load on {len(edges)} line(s)")

    def _add_traction(self) -> None:
        traction = self.vector(self._traction, "traction")
        if isinstance(self._editing_attribute, SurfaceTraction):
            current = self._editing_attribute
            self.app.run(
                cmd.EditAttribute(
                    replace(
                        current,
                        traction=traction,
                        coordinate_system_id=self.coordinate_system_id(),
                    )
                )
            )
            self._finish_attribute_edit()
            self.app.set_status("updated surface traction")
            return
        faces = self.require_selection("face")
        self.app.run_many(
            cmd.AddSurfaceTraction(
                ref,
                tuple(traction),
                case=self.case_name(),
                coordinate_system_id=self.coordinate_system_id(),
            )
            for ref in faces
        )
        self.app.set_status(f"traction on {len(faces)} plate(s)")

    def _set_acceleration(self) -> None:
        vector = self.vector(self._acceleration, "acceleration")
        self.app.run(
            cmd.SetAcceleration(
                tuple(vector),
                case=self.case_name(),
                coordinate_system_id=self.coordinate_system_id(),
            )
        )
        self.app.set_status(f"acceleration on case {self.case_name()}")

    def _add_mass(self) -> None:
        value = self.number(self._mass, "mass")
        policy = (
            "per_target"
            if self._point_policy.get() == "Per target"
            else "total_distributed"
        )
        if isinstance(self._editing_attribute, Mass):
            current = self._editing_attribute
            self.app.run(
                cmd.EditAttribute(
                    replace(current, value=value, distribution_policy=policy)
                )
            )
            self._finish_attribute_edit()
            self.app.set_status(f"updated {current.name}")
            return
        items = self.require_selection(self.app.selection.mode)
        scale = 1.0 if policy == "per_target" else 1.0 / len(items)
        self.app.run_many(
            cmd.AddMass(
                ref,
                value * scale,
                name=f"mass_{ref}",
                distribution_policy=policy,
            )
            for ref in items
        )
        self.app.set_status(
            f"mass on {len(items)} entity(ies), {self._point_policy.get().lower()}"
        )

    def _add_combination(self) -> None:
        name = self._combination_name.get().strip()
        if not name:
            raise ValueError("give the combination a name")
        parts = self._factors.get().replace(",", " ").split()
        if not parts or len(parts) % 2:
            raise ValueError(
                "factors must be pairs of case name and factor, e.g. "
                "'dead 1.2 live 1.5'"
            )
        factors = {}
        for case_name, factor in zip(parts[0::2], parts[1::2]):
            try:
                factors[case_name] = float(factor)
            except ValueError:
                raise ValueError(f"{factor!r} is not a number") from None
        self.app.run(cmd.AddCombination(name, factors))
        self.app.set_status(f"combination {name}")
        self.app.refresh_all()


# ----------------------------------------------------------------------
class SolvePanel(StagePanel):
    title = "Solve"

    # Each analysis names the option rows it uses, so the panel only shows
    # what the chosen one actually takes.
    ANALYSES = {
        "Linear static": (),
        "Batch linear static": ("batch_cases",),
        "Modal": ("modes",),
        "Buckling": ("modes",),
        "Nonlinear static": ("steps", "factor"),
        "Arc length": ("arc_steps",),
        "Transient": ("dt", "t_end", "damping"),
        "Impact": ("mass", "radius", "speed", "start", "direction"),
        "Capacity": (
            "modes", "buckling_mode", "steps", "factor", "imperfection"
        ),
    }

    # Advanced rows remain discoverable without making the normal Run form
    # intimidating.  Keep this map next to ANALYSES so adding a public solver
    # option cannot accidentally leave it visible for an unrelated analysis.
    ADVANCED_ANALYSES = {
        "Linear static": (),
        "Batch linear static": (),
        "Modal": ("modal_shift",),
        "Buckling": (),
        "Nonlinear static": (
            "nonlinear_iterations", "nonlinear_tolerance",
            "nonlinear_layers", "nonlinear_min_step",
        ),
        "Arc length": (
            "nonlinear_iterations", "nonlinear_tolerance",
            "nonlinear_layers", "arc_tolerance", "arc_initial",
            "arc_minimum", "arc_maximum", "arc_load_scaling",
            "arc_rotation_scale", "arc_target_iterations",
            "arc_growth", "arc_cutback", "arc_retries", "arc_peak_steps",
            "arc_peak_tolerance", "arc_load_limit", "arc_post_peak",
            "arc_translation_limit", "arc_preload_steps",
        ),
        "Transient": (
            "transient_beta", "transient_gamma", "transient_hht",
            "rayleigh_beta", "save_every", "stress_history",
        ),
        "Impact": (
            "impact_dt", "impact_duration", "save_every",
            "steps_per_contact", "steps_per_radius", "post_contact_periods",
            "impact_rayleigh_alpha", "rayleigh_beta", "skip_approach",
            "nonlinear_impact",
            "contact_surface", "beam_contact",
        ),
        "Capacity": (
            "nonlinear_iterations", "nonlinear_tolerance",
            "nonlinear_layers", "capacity_half_wave",
        ),
    }

    def build(self) -> None:
        controls = self.section("Analysis")
        row = ttk.Frame(controls)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="analysis", width=16).pack(side="left")
        self._analysis = tk.StringVar(value="Linear static")
        analysis_box = ttk.Combobox(
            row, textvariable=self._analysis, values=list(self.ANALYSES),
            state="readonly", width=16,
        )
        analysis_box.pack(side="left", fill="x", expand=True)
        analysis_box.bind(
            "<<ComboboxSelected>>",
            lambda _e: (self._show_options(), self._refresh_material_response()),
        )

        row = ttk.Frame(controls)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="loads", width=16).pack(side="left")
        self._target = tk.StringVar(value="case: default")
        self._target_box = ttk.Combobox(
            row, textvariable=self._target, values=["case: default"],
            state="readonly", width=16,
        )
        self._target_box.pack(side="left", fill="x", expand=True)
        self._material_response = ttk.Label(
            controls, text="", justify="left", foreground="#555555"
        )
        self._material_response.pack(anchor="w", fill="x", pady=(3, 1))

        options = self.section("Options")
        # tkinter.Misc owns a callable named ``_options`` which pack/grid use.
        # Shadowing it with this dictionary makes opening the Solve page fail.
        self._analysis_options = {}
        self._option_frames = {}
        for name, label, default in (
            ("batch_cases", "case names", "*"),
            ("modes", "modes", "6"),
            ("buckling_mode", "imperfection mode", "1"),
            ("steps", "initial load increments", "10"),
            ("factor", "max load factor", "1.0"),
            ("arc_steps", "max arc steps", "60"),
            ("dt", "time step [s]", "0.0002"),
            ("t_end", "duration [s]", "0.02"),
            ("damping", "Rayleigh alpha", "0.0"),
            ("mass", "sphere mass [kg]", "500"),
            ("radius", "sphere radius [m]", "0.15"),
            ("speed", "speed [m/s]", "4.0"),
            ("start", "start x y z [m]", "0 0 1"),
            ("direction", "direction x y z", "0 0 -1"),
            ("imperfection", "imperfection [mm]", "5.0"),
        ):
            frame, variable = self.labelled_entry(
                options,
                label,
                default,
                label_width=20 if name == "steps" else 16,
            )
            self._analysis_options[name] = variable
            self._option_frames[name] = frame
        self._nominal_steps_hint = ttk.Label(
            options,
            text=(
                "Starting partition only: adaptive cutbacks and growth determine "
                "the actual number of converged increments."
            ),
            justify="left",
            foreground="#555555",
            wraplength=560,
        )

        self._advanced_open = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self,
            text="Advanced solver controls",
            variable=self._advanced_open,
            command=self._show_advanced,
        ).pack(anchor="w", pady=(5, 1))
        self._advanced = ttk.Frame(self)
        advanced_tabs = ttk.Notebook(self._advanced, height=260)
        advanced_tabs.pack(fill="x", expand=True)
        numerics = ttk.Frame(advanced_tabs, padding=4)
        resources = ttk.Frame(advanced_tabs, padding=4)
        advanced_tabs.add(numerics, text="Numerics")
        advanced_tabs.add(resources, text="Resources")

        self._advanced_options = {}
        self._advanced_option_frames = {}

        def advanced_entry(name: str, label: str, default: str) -> None:
            frame, variable = self.labelled_entry(
                numerics, label, default, label_width=23
            )
            self._advanced_options[name] = variable
            self._advanced_option_frames[name] = frame

        for name, label, default in (
            ("modal_shift", "spectral shift", "0"),
            ("nonlinear_iterations", "max Newton iterations", "25"),
            ("nonlinear_tolerance", "residual tolerance", "1e-6"),
            ("nonlinear_layers", "shell integration layers", "5"),
            ("nonlinear_min_step", "minimum step fraction", "0.0009765625"),
            ("arc_tolerance", "arc constraint tolerance", "1e-6"),
            ("arc_initial", "initial load increment", "0.05"),
            ("arc_minimum", "minimum load increment", "0.0005"),
            ("arc_maximum", "maximum load increment", "0.20"),
            ("arc_load_scaling", "load scaling", "auto"),
            ("arc_rotation_scale", "rotation length scale", "auto"),
            ("arc_target_iterations", "target Newton iterations", "5"),
            ("arc_growth", "arc growth factor", "1.25"),
            ("arc_cutback", "arc cutback factor", "0.5"),
            ("arc_retries", "retries per arc step", "8"),
            ("arc_peak_steps", "steps after peak", "4"),
            ("arc_peak_tolerance", "peak drop tolerance", "0.001"),
            ("arc_load_limit", "absolute load limit", "auto"),
            ("arc_post_peak", "stop at peak fraction", "auto"),
            ("arc_translation_limit", "translation limit [m]", "auto"),
            ("arc_preload_steps", "preload increments", "10"),
            ("transient_beta", "Newmark beta", "0.25"),
            ("transient_gamma", "Newmark gamma", "0.5"),
            ("transient_hht", "HHT alpha", "0"),
            ("rayleigh_beta", "Rayleigh beta", "0"),
            ("save_every", "save every N steps", "1"),
            ("impact_dt", "time step [s]", "auto"),
            ("impact_duration", "duration [s]", "auto"),
            ("impact_rayleigh_alpha", "Rayleigh alpha", "0"),
            ("steps_per_contact", "steps per contact period", "20"),
            ("steps_per_radius", "steps per sphere radius", "20"),
            ("post_contact_periods", "post-contact periods", "20"),
            ("capacity_half_wave", "elements per half-wave", "4"),
        ):
            advanced_entry(name, label, default)

        self._kinematics_frame, self._kinematics = self.labelled_entry(
            numerics, "kinematics", "von_karman", label_width=23
        )
        self._corotational_frame, self._corotational_tangent = self.labelled_entry(
            numerics, "corotational tangent", "auto", label_width=23
        )

        contact_row = ttk.Frame(numerics)
        contact_row.pack(fill="x", pady=1)
        ttk.Label(contact_row, text="contact surface", width=23).pack(side="left")
        self._contact_surface = tk.StringVar(value="midsurface")
        ttk.Combobox(
            contact_row,
            textvariable=self._contact_surface,
            values=("midsurface", "top", "bottom"),
            state="readonly",
            width=16,
        ).pack(side="left", fill="x", expand=True)
        self._advanced_options["contact_surface"] = self._contact_surface
        self._advanced_option_frames["contact_surface"] = contact_row

        self._advanced_checks = {}
        for name, text, default in (
            ("stress_history", "Save transient stress history", False),
            ("skip_approach", "Skip free-flight approach", True),
            ("nonlinear_impact", "Use nonlinear impact integration", False),
            ("beam_contact", "Include beam contact", False),
        ):
            frame = ttk.Frame(numerics)
            frame.pack(fill="x", pady=1)
            variable = tk.BooleanVar(value=default)
            ttk.Checkbutton(frame, text=text, variable=variable).pack(anchor="w")
            self._advanced_checks[name] = variable
            self._advanced_option_frames[name] = frame

        self._advanced_hint = ttk.Label(
            numerics,
            text=(
                "Auto keeps solver-selected timing or stopping limits. "
                "Only controls supported by the selected analysis are shown."
            ),
            foreground="#555555",
            justify="left",
            wraplength=520,
        )
        self._advanced_hint.pack(fill="x", pady=(4, 0))

        self._use_resources = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            resources,
            text="Override solver resource settings",
            variable=self._use_resources,
            command=self._toggle_resource_controls,
        ).pack(anchor="w", pady=(0, 3))
        self._resource_options = {}
        self._resource_entries = []
        for name, label in (
            ("solver_threads", "solver threads"),
            ("assembly_threads", "assembly threads"),
            ("recovery_threads", "recovery threads"),
            ("process_workers", "process workers"),
            ("memory_mib", "memory limit [MiB]"),
        ):
            frame, variable = self.labelled_entry(
                resources, label, "auto", label_width=20
            )
            self._resource_options[name] = variable
            self._resource_entries.extend(
                widget for widget in frame.winfo_children()
                if isinstance(widget, ttk.Entry)
            )
        self._deterministic = tk.BooleanVar(value=True)
        self._deterministic_check = ttk.Checkbutton(
            resources,
            text="Deterministic execution",
            variable=self._deterministic,
        )
        self._deterministic_check.pack(anchor="w", pady=(2, 0))
        ttk.Label(
            resources,
            text=(
                "Leave values on Auto to use the solver backend defaults. "
                "Resource controls change cost and reproducibility, not the model."
            ),
            foreground="#555555",
            justify="left",
            wraplength=520,
        ).pack(fill="x", pady=(3, 0))

        self._record_snapshots = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            numerics,
            text="save converged nonlinear increments",
            variable=self._record_snapshots,
        ).pack(anchor="w")
        self._toggle_resource_controls()

        self._run = ttk.Button(
            self, text="Run", command=self.guarded(self._solve)
        )
        self._run.pack(fill="x", pady=2)
        self._cancel = ttk.Button(
            self, text="Cancel", command=self.app.cancel_solve, state="disabled"
        )
        self._cancel.pack(fill="x", pady=2)

        self._progress = ttk.Label(self, text="", foreground="#555555")
        self._progress.pack(anchor="w", pady=(4, 0))

        self._transcript_tabs = ttk.Notebook(self)
        self._transcript_tabs.pack(fill="both", expand=True, pady=(6, 0))
        monitor_page = ttk.Frame(self._transcript_tabs)
        inputs_page = ttk.Frame(self._transcript_tabs)
        self._transcript_tabs.add(monitor_page, text="Run monitor")
        self._transcript_tabs.add(inputs_page, text="Submitted inputs")

        monitor = ttk.Panedwindow(monitor_page, orient=tk.VERTICAL)
        monitor.pack(fill="both", expand=True)
        graph_page = ttk.Frame(monitor, padding=(2, 2))
        log_page = ttk.Frame(monitor, padding=(2, 2))
        monitor.add(graph_page, weight=1)
        monitor.add(log_page, weight=1)

        graph_header = ttk.Frame(graph_page)
        graph_header.pack(fill="x")
        ttk.Label(graph_header, text="live graph", width=11).pack(side="left")
        self._live_graph_choice = tk.StringVar(value=GRAPH_CHOICES[0])
        graph_box = ttk.Combobox(
            graph_header,
            textvariable=self._live_graph_choice,
            values=GRAPH_CHOICES,
            state="readonly",
            width=30,
        )
        graph_box.pack(side="left", fill="x", expand=True)
        self._live_graph_choice_box = graph_box
        graph_box.bind(
            "<<ComboboxSelected>>", lambda _event: self._refresh_live_plot()
        )
        self._live_caption = ttk.Label(
            graph_page, text="Waiting for solver progress…", foreground="#555555"
        )
        self._live_caption.pack(fill="x", anchor="w")
        self._live_plot = HistoryPlot(graph_page, width=360, height=150)
        self._live_plot.pack(fill="both", expand=True)

        ttk.Label(log_page, text="Text log", foreground="#555555").pack(anchor="w")
        self._report = tk.Text(log_page, height=8, wrap="word", state="disabled")
        self._report.pack(fill="both", expand=True)
        self._submitted_inputs = tk.Text(
            inputs_page, height=12, wrap="none", state="disabled"
        )
        self._submitted_inputs.pack(fill="both", expand=True)
        self._live_line_count = 0
        self._last_live_line = ""
        self._live_data = LiveProgressData()
        self._live_plot_after = None

        self._show_options()
        self._show_advanced()

    # ------------------------------------------------------------------
    def _show_options(self) -> None:
        wanted = set(self.ANALYSES.get(self._analysis.get(), ()))
        for name, frame in self._option_frames.items():
            if name in wanted:
                frame.pack(fill="x", pady=1)
            else:
                frame.pack_forget()
        if self._analysis.get() in ("Nonlinear static", "Capacity"):
            self._nominal_steps_hint.pack(fill="x", anchor="w", pady=(3, 0))
        else:
            self._nominal_steps_hint.pack_forget()
        self._show_advanced_options()

    def _show_advanced_options(self) -> None:
        wanted = set(self.ADVANCED_ANALYSES.get(self._analysis.get(), ()))
        for name, frame in self._advanced_option_frames.items():
            if name in wanted:
                frame.pack(fill="x", pady=1, before=self._advanced_hint)
            else:
                frame.pack_forget()
        nonlinear = self._analysis.get() in ("Nonlinear static", "Arc length")
        for frame in (self._kinematics_frame, self._corotational_frame):
            if nonlinear:
                frame.pack(fill="x", pady=1, before=self._advanced_hint)
            else:
                frame.pack_forget()

    def _toggle_resource_controls(self) -> None:
        state = "normal" if self._use_resources.get() else "disabled"
        for entry in self._resource_entries:
            entry.configure(state=state)
        self._deterministic_check.configure(state=state)

    def _show_advanced(self) -> None:
        if self._advanced_open.get():
            self._advanced.pack(fill="x", pady=(0, 4), before=self._run)
        else:
            self._advanced.pack_forget()

    def refresh(self) -> None:
        running = self.app.worker.running
        self._run.configure(state="disabled" if running else "normal")
        self._cancel.configure(state="normal" if running else "disabled")

        project = self.app.project
        options = [f"case: {name}" for name in sorted(project.load_cases)]
        options += [
            f"combination: {name}" for name in sorted(project.combinations)
        ]
        self._target_box.configure(values=options or ["case: default"])
        if self._target.get() not in options and options:
            self._target.set(options[0])
        self._refresh_material_response()

    def _refresh_material_response(self) -> None:
        from ..presentation.engineering_summary import material_response
        text, colour = material_response(self.app.project, self._analysis.get())
        self._material_response.configure(text=text, foreground=colour)

    def show_progress(self, text: str) -> None:
        self._progress.configure(text=text)

    def begin_job(self, name: str, job_id: str, submitted_inputs: str = "") -> None:
        """Start a bounded live transcript for a newly submitted job."""

        self._report.configure(state="normal")
        self._report.delete("1.0", "end")
        self._report.insert("end", f"{name}  [{job_id[:8]}]\n")
        self._report.configure(state="disabled")
        self._live_line_count = 1
        self._last_live_line = ""
        self._live_data.clear()
        self._refresh_live_plot()
        self.set_submitted_inputs(submitted_inputs)
        self._transcript_tabs.select(0)
        self.append_progress("queued")

    def set_submitted_inputs(self, text: str) -> None:
        self._submitted_inputs.configure(state="normal")
        self._submitted_inputs.delete("1.0", "end")
        self._submitted_inputs.insert("1.0", str(text).strip() or "No input record available")
        self._submitted_inputs.configure(state="disabled")

    def append_progress(self, text: str, payload=None) -> None:
        """Append one live solver message without allowing unbounded growth."""

        line = str(text).strip()
        if self._live_data.ingest(line, payload):
            self._schedule_live_plot()
        if not line or line == self._last_live_line:
            return
        self._last_live_line = line
        self._report.configure(state="normal")
        self._report.insert("end", line + "\n")
        self._live_line_count += 1
        if self._live_line_count > 1000:
            excess = self._live_line_count - 1000
            self._report.delete("1.0", f"{excess + 1}.0")
            self._live_line_count -= excess
        self._report.see("end")
        self._report.configure(state="disabled")

    def _schedule_live_plot(self) -> None:
        """Coalesce worker bursts so graph redraw never floods Tk."""

        if self._live_plot_after is None:
            self._live_plot_after = self.after(50, self._refresh_live_plot)

    def _refresh_live_plot(self) -> None:
        self._live_plot_after = None
        self._live_graph_choice_box.configure(values=self._live_data.graph_choices)
        choice = self._live_graph_choice.get()
        series = self._live_data.series(choice)
        self._live_plot.show([] if series is None else [series])
        self._live_caption.configure(text=self._live_data.caption(choice))

    def append_report(self, text: str) -> None:
        report = str(text).strip()
        if not report:
            return
        self._report.configure(state="normal")
        self._report.insert("end", "\nResult summary\n--------------\n" + report + "\n")
        self._report.see("end")
        self._report.configure(state="disabled")

    def write(self, text: str) -> None:
        self._report.configure(state="normal")
        self._report.delete("1.0", "end")
        self._report.insert("1.0", text)
        self._report.configure(state="disabled")
        self._live_line_count = int(self._report.index("end-1c").split(".")[0])
        self._last_live_line = ""

    def destroy(self) -> None:
        if self._live_plot_after is not None:
            try:
                self.after_cancel(self._live_plot_after)
            except tk.TclError:
                pass
            self._live_plot_after = None
        super().destroy()

    # ------------------------------------------------------------------
    @staticmethod
    def _triple(variable: tk.StringVar, label: str) -> tuple:
        parts = variable.get().replace(",", " ").split()
        if len(parts) != 3:
            raise ValueError(f"{label} needs three numbers, e.g. '0 0 -1'")
        try:
            return tuple(float(part) for part in parts)
        except ValueError:
            raise ValueError(f"{label} must be three numbers") from None

    def _target_kwargs(self) -> dict:
        choice = self._target.get()
        if choice.startswith("combination: "):
            return {"combination": choice.split(": ", 1)[1]}
        name = choice.split(": ", 1)[-1] if ": " in choice else choice
        return {"load_case": name or "default"}

    def _advanced_number(self, name: str, label: str) -> float:
        return self.number(self._advanced_options[name], label)

    def _advanced_integer(self, name: str, label: str) -> int:
        value = self._advanced_number(name, label)
        integer = int(value)
        if value != integer:
            raise ValueError(f"{label} must be a whole number")
        return integer

    def _optional_advanced_number(self, name: str, label: str) -> float | None:
        value = self._advanced_options[name].get().strip()
        if value.casefold() in ("", "auto", "none"):
            return None
        return self._advanced_number(name, label)

    def _optional_resource_integer(self, name: str, label: str) -> int | None:
        value = self._resource_options[name].get().strip()
        if value.casefold() in ("", "auto", "none"):
            return None
        try:
            numeric = float(value)
        except ValueError:
            raise ValueError(
                f"{label} must be a positive whole number or Auto"
            ) from None
        integer = int(numeric)
        if numeric != integer or integer <= 0:
            raise ValueError(f"{label} must be a positive whole number or Auto")
        return integer

    def _resource_policy(self):
        if not self._use_resources.get():
            return None
        from ..solve import resource_policy

        memory_mib = self._optional_resource_integer(
            "memory_mib", "memory limit"
        )
        return resource_policy(
            solver_threads=self._optional_resource_integer(
                "solver_threads", "solver threads"
            ),
            assembly_threads=self._optional_resource_integer(
                "assembly_threads", "assembly threads"
            ),
            recovery_threads=self._optional_resource_integer(
                "recovery_threads", "recovery threads"
            ),
            process_workers=self._optional_resource_integer(
                "process_workers", "process workers"
            ),
            deterministic=bool(self._deterministic.get()),
            memory_limit_bytes=(
                None if memory_mib is None else memory_mib * 1024 * 1024
            ),
        )

    def _solve(self) -> None:
        analysis = self._analysis.get()
        kwargs = self._target_kwargs()

        if analysis == "Modal":
            kwargs = {
                "num_modes": int(
                    self.number(self._analysis_options["modes"], "modes")
                ),
                "shift": self._advanced_number("modal_shift", "spectral shift"),
            }
        elif analysis == "Batch linear static":
            value = self._analysis_options["batch_cases"].get().strip()
            names = (
                tuple(sorted(self.app.project.load_cases))
                if value in ("", "*")
                else tuple(value.replace(",", " ").split())
            )
            if not names:
                raise ValueError("a batch solve needs at least one load case")
            unknown = sorted(set(names) - set(self.app.project.load_cases))
            if unknown:
                raise ValueError(f"unknown batch load case(s): {unknown}")
            kwargs = {"load_cases": names}
        elif analysis == "Buckling":
            kwargs["num_modes"] = int(self.number(self._analysis_options["modes"], "modes"))
        elif analysis == "Nonlinear static":
            kwargs["num_steps"] = int(
                self.number(
                    self._analysis_options["steps"], "initial load increments"
                )
            )
            kwargs["max_load_factor"] = self.number(
                self._analysis_options["factor"], "max load factor"
            )
            kwargs.update(
                max_iterations=self._advanced_integer(
                    "nonlinear_iterations", "max Newton iterations"
                ),
                tolerance=self._advanced_number(
                    "nonlinear_tolerance", "residual tolerance"
                ),
                num_layers=self._advanced_integer(
                    "nonlinear_layers", "shell integration layers"
                ),
                min_step_fraction=self._advanced_number(
                    "nonlinear_min_step", "minimum step fraction"
                ),
            )
        elif analysis == "Arc length":
            from anysolver import ArcLengthControl

            kwargs["control"] = ArcLengthControl(
                initial_load_increment=self._advanced_number(
                    "arc_initial", "initial load increment"
                ),
                minimum_load_increment=self._advanced_number(
                    "arc_minimum", "minimum load increment"
                ),
                maximum_load_increment=self._advanced_number(
                    "arc_maximum", "maximum load increment"
                ),
                load_scaling=self._optional_advanced_number(
                    "arc_load_scaling", "load scaling"
                ),
                rotation_length_scale=self._optional_advanced_number(
                    "arc_rotation_scale", "rotation length scale"
                ),
                target_iterations=self._advanced_integer(
                    "arc_target_iterations", "target Newton iterations"
                ),
                growth_factor=self._advanced_number(
                    "arc_growth", "arc growth factor"
                ),
                cutback_factor=self._advanced_number(
                    "arc_cutback", "arc cutback factor"
                ),
                max_steps=int(
                    self.number(self._analysis_options["arc_steps"], "max arc steps")
                ),
                max_retries_per_step=self._advanced_integer(
                    "arc_retries", "retries per arc step"
                ),
                stop_after_peak_steps=self._advanced_integer(
                    "arc_peak_steps", "steps after peak"
                ),
                peak_drop_tolerance=self._advanced_number(
                    "arc_peak_tolerance", "peak drop tolerance"
                ),
                maximum_absolute_load_factor=self._optional_advanced_number(
                    "arc_load_limit", "absolute load limit"
                ),
                post_peak_load_fraction=self._optional_advanced_number(
                    "arc_post_peak", "post-peak load fraction"
                ),
                max_translation=self._optional_advanced_number(
                    "arc_translation_limit", "translation limit"
                ),
                preload_steps=self._advanced_integer(
                    "arc_preload_steps", "preload increments"
                ),
            )
            kwargs.update(
                max_iterations=self._advanced_integer(
                    "nonlinear_iterations", "max Newton iterations"
                ),
                tolerance=self._advanced_number(
                    "nonlinear_tolerance", "residual tolerance"
                ),
                arc_tolerance=self._advanced_number(
                    "arc_tolerance", "arc constraint tolerance"
                ),
                num_layers=self._advanced_integer(
                    "nonlinear_layers", "shell integration layers"
                ),
            )
        elif analysis == "Transient":
            kwargs["dt"] = self.number(self._analysis_options["dt"], "time step")
            kwargs["t_end"] = self.number(self._analysis_options["t_end"], "duration")
            kwargs["rayleigh_alpha"] = self.number(
                self._analysis_options["damping"], "Rayleigh alpha"
            )
            kwargs.update(
                beta=self._advanced_number("transient_beta", "Newmark beta"),
                gamma=self._advanced_number("transient_gamma", "Newmark gamma"),
                hht_alpha=self._advanced_number("transient_hht", "HHT alpha"),
                rayleigh_beta=self._advanced_number(
                    "rayleigh_beta", "Rayleigh beta"
                ),
                save_every=self._advanced_integer("save_every", "save interval"),
                include_stress_history=bool(
                    self._advanced_checks["stress_history"].get()
                ),
            )
        elif analysis == "Impact":
            from ..model.collision import Collision
            from anysolver import SphereContactConfig

            # An impact does not need a load case; the sphere is the load.
            kwargs.pop("combination", None)
            kwargs["load_case"] = None
            kwargs["collision"] = Collision(
                mass=self.number(self._analysis_options["mass"], "sphere mass"),
                radius=self.number(self._analysis_options["radius"], "sphere radius"),
                speed=self.number(self._analysis_options["speed"], "speed"),
                start=self._triple(self._analysis_options["start"], "start"),
                direction=self._triple(
                    self._analysis_options["direction"], "direction"
                ),
            )
            kwargs.update(
                dt=self._optional_advanced_number("impact_dt", "time step"),
                t_end=self._optional_advanced_number(
                    "impact_duration", "duration"
                ),
                save_every=self._advanced_integer("save_every", "save interval"),
                steps_per_contact=self._advanced_number(
                    "steps_per_contact", "steps per contact period"
                ),
                steps_per_radius=self._advanced_number(
                    "steps_per_radius", "steps per sphere radius"
                ),
                post_contact_periods=self._advanced_number(
                    "post_contact_periods", "post-contact periods"
                ),
                skip_approach=bool(self._advanced_checks["skip_approach"].get()),
                nonlinear=bool(self._advanced_checks["nonlinear_impact"].get()),
                rayleigh_alpha=self._advanced_number(
                    "impact_rayleigh_alpha", "Rayleigh alpha"
                ),
                rayleigh_beta=self._advanced_number(
                    "rayleigh_beta", "Rayleigh beta"
                ),
                contact=SphereContactConfig(
                    contact_surface=self._contact_surface.get(),
                    beam_contact=bool(self._advanced_checks["beam_contact"].get()),
                ),
            )
        elif analysis == "Capacity":
            kwargs["num_buckling_modes"] = int(
                self.number(self._analysis_options["modes"], "buckling modes")
            )
            kwargs["buckling_mode_number"] = int(
                self.number(
                    self._analysis_options["buckling_mode"], "imperfection mode"
                )
            )
            kwargs["num_steps"] = int(
                self.number(
                    self._analysis_options["steps"], "initial load increments"
                )
            )
            kwargs["max_load_factor"] = self.number(
                self._analysis_options["factor"], "max load factor"
            )
            kwargs["imperfection_amplitude"] = self.number(
                self._analysis_options["imperfection"], "imperfection"
            ) / 1000.0
            kwargs.update(
                nonlinear_max_iterations=self._advanced_integer(
                    "nonlinear_iterations", "max Newton iterations"
                ),
                nonlinear_tolerance=self._advanced_number(
                    "nonlinear_tolerance", "residual tolerance"
                ),
                nonlinear_num_layers=self._advanced_integer(
                    "nonlinear_layers", "shell integration layers"
                ),
                mesh_min_elements_per_half_wave=self._advanced_integer(
                    "capacity_half_wave", "elements per half-wave"
                ),
            )

        if analysis in ("Nonlinear static", "Arc length"):
            kwargs["kinematics"] = self._kinematics.get().strip() or "von_karman"
            kwargs["corotational_tangent"] = (
                self._corotational_tangent.get().strip() or "auto"
            )
        if analysis in ("Nonlinear static", "Arc length", "Capacity"):
            kwargs["record_increment_snapshots"] = self._record_snapshots.get()

        resources = self._resource_policy()
        if resources is not None:
            if analysis in ("Nonlinear static", "Capacity"):
                kwargs["resources"] = resources
            else:
                kwargs["resource_config"] = resources

        self.app.solve(analysis, **kwargs)


# ----------------------------------------------------------------------
class VisualizationPanel(StagePanel):
    """Application-wide viewport appearance and performance controls."""

    title = "Visualization"

    def build(self) -> None:
        ttk.Label(
            self,
            text=(
                "These settings apply to model, mesh and result views. "
                "Geometry detail changes display tessellation only; it does "
                "not change model geometry, mesh density or solver inputs."
            ),
            justify="left",
            wraplength=560,
        ).pack(fill="x", pady=(0, 8))

        style = self.app.viewport.visualization
        appearance = self.section("Viewport appearance")

        render_row = ttk.Frame(appearance)
        render_row.pack(fill="x", pady=1)
        ttk.Label(render_row, text="render", width=18).pack(side="left")
        self._render_mode = tk.StringVar(value=style.render_mode)
        ttk.Combobox(
            render_row,
            textvariable=self._render_mode,
            values=RENDER_MODES,
            state="readonly",
            width=22,
        ).pack(side="left", fill="x", expand=True)

        detail_row = ttk.Frame(appearance)
        detail_row.pack(fill="x", pady=1)
        ttk.Label(detail_row, text="geometry detail", width=18).pack(side="left")
        self._geometry_detail = tk.StringVar(value=style.geometry_detail)
        ttk.Combobox(
            detail_row,
            textvariable=self._geometry_detail,
            values=GEOMETRY_DETAILS,
            state="readonly",
            width=22,
        ).pack(side="left", fill="x", expand=True)
        ttk.Label(
            appearance,
            text=(
                "Auto reduces curved-face display density as cylinders and "
                "cones gain structural segments. Fast is useful for very "
                "large models; Fine preserves the densest preview."
            ),
            justify="left",
            foreground="#666666",
            wraplength=540,
        ).pack(fill="x", pady=(1, 4))

        self._surface_opacity = self.entry_row(
            appearance, "surface opacity [%]", f"{100.0 * style.surface_opacity:g}"
        )
        self._edge_width = self.entry_row(
            appearance, "edge width [px]", str(style.edge_width)
        )
        self._background_color = self._colour_row(
            appearance, "background", style.background
        )
        self._edge_color = self._colour_row(
            appearance, "model / mesh edges", style.edge_color
        )

        result_style = self.section("Result visualization")
        ttk.Label(
            result_style,
            text=(
                "These controls apply when a contour result is displayed and "
                "are synchronized with Results > Visualization."
            ),
            justify="left",
            foreground="#666666",
            wraplength=540,
        ).pack(fill="x", pady=(0, 4))
        palette_row = ttk.Frame(result_style)
        palette_row.pack(fill="x", pady=1)
        ttk.Label(palette_row, text="contour colours", width=18).pack(side="left")
        self._result_colormap = tk.StringVar(value=style.result_colormap)
        ttk.Combobox(
            palette_row,
            textvariable=self._result_colormap,
            values=tuple(RESULT_COLORMAPS),
            state="readonly",
            width=22,
        ).pack(side="left", fill="x", expand=True)
        self._legend_font_size = self.entry_row(
            result_style, "legend text [px]", str(style.legend_font_size)
        )
        self._legend_width = self.entry_row(
            result_style, "legend width [px]", str(style.legend_width)
        )
        self._show_legend = tk.BooleanVar(value=style.show_legend)
        self._show_result_nodes = tk.BooleanVar(value=style.show_result_nodes)
        self._show_result_supports = tk.BooleanVar(value=style.show_result_supports)
        self._show_result_loads = tk.BooleanVar(value=style.show_result_loads)
        self._show_result_masses = tk.BooleanVar(value=style.show_result_masses)
        self._show_imperfect_reference = tk.BooleanVar(
            value=style.show_imperfect_reference
        )
        self._show_beam_sections = tk.BooleanVar(value=style.show_beam_sections)
        for text, variable in (
            ("Show contour legend", self._show_legend),
            ("Render assigned beam sections in 3D", self._show_beam_sections),
            ("Show FE nodes", self._show_result_nodes),
            ("Show supports", self._show_result_supports),
            ("Show loads", self._show_result_loads),
            ("Show masses", self._show_result_masses),
            ("Show initial imperfect reference", self._show_imperfect_reference),
        ):
            ttk.Checkbutton(
                result_style, text=text, variable=variable
            ).pack(anchor="w")

        actions = ttk.Frame(result_style)
        actions.pack(fill="x", pady=(6, 0))
        ttk.Button(
            actions,
            text="Apply",
            command=self.guarded(self._apply),
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            actions,
            text="Reset defaults",
            command=self.guarded(self._reset),
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        view = self.section("View and renderer")
        self._renderer_status = ttk.Label(view, text="", justify="left", wraplength=540)
        self._renderer_status.pack(fill="x", pady=(0, 5))
        view_actions = ttk.Frame(view)
        view_actions.pack(fill="x")
        ttk.Button(
            view_actions, text="Model view", command=self.app.show_geometry
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            view_actions, text="Mesh view", command=self.app.show_mesh
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))
        ttk.Button(
            view_actions, text="Fit", command=self.app.viewport.fit
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.refresh()

    def _colour_row(self, parent, label: str, value: str) -> tk.StringVar:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=18).pack(side="left")
        variable = tk.StringVar(value=value)
        ttk.Entry(row, textvariable=variable, width=14).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(
            row,
            text="Choose…",
            width=9,
            command=lambda: self._choose_colour(variable),
        ).pack(side="left", padx=(4, 0))
        return variable

    def _choose_colour(self, variable: tk.StringVar) -> None:
        _rgb, colour = colorchooser.askcolor(
            color=variable.get(), parent=self, title="Choose viewport colour"
        )
        if colour:
            variable.set(colour)

    def _style(self) -> VisualizationStyle:
        try:
            opacity = float(self._surface_opacity.get()) / 100.0
        except ValueError:
            raise ValueError(
                "surface opacity must be a percentage from 0 to 100"
            ) from None
        try:
            edge_width = int(self._edge_width.get())
        except ValueError:
            raise ValueError("edge width must be a whole number") from None
        try:
            legend_font_size = int(self._legend_font_size.get())
        except ValueError:
            raise ValueError("legend text size must be a whole number") from None
        try:
            legend_width = int(self._legend_width.get())
        except ValueError:
            raise ValueError("legend width must be a whole number") from None
        return VisualizationStyle(
            render_mode=self._render_mode.get(),
            surface_opacity=opacity,
            background=self._background_color.get().strip(),
            edge_color=self._edge_color.get().strip(),
            edge_width=edge_width,
            show_legend=bool(self._show_legend.get()),
            legend_font_size=legend_font_size,
            legend_width=legend_width,
            result_colormap=self._result_colormap.get(),
            show_result_nodes=bool(self._show_result_nodes.get()),
            show_result_supports=bool(self._show_result_supports.get()),
            show_result_loads=bool(self._show_result_loads.get()),
            show_result_masses=bool(self._show_result_masses.get()),
            show_imperfect_reference=bool(
                self._show_imperfect_reference.get()
            ),
            show_beam_sections=bool(self._show_beam_sections.get()),
            geometry_detail=self._geometry_detail.get(),
        )

    def _load_style(self, style: VisualizationStyle) -> None:
        self._render_mode.set(style.render_mode)
        self._geometry_detail.set(style.geometry_detail)
        self._surface_opacity.set(f"{100.0 * style.surface_opacity:g}")
        self._edge_width.set(str(style.edge_width))
        self._background_color.set(style.background)
        self._edge_color.set(style.edge_color)
        self._show_legend.set(style.show_legend)
        self._legend_font_size.set(str(style.legend_font_size))
        self._legend_width.set(str(style.legend_width))
        self._result_colormap.set(style.result_colormap)
        self._show_result_nodes.set(style.show_result_nodes)
        self._show_result_supports.set(style.show_result_supports)
        self._show_result_loads.set(style.show_result_loads)
        self._show_result_masses.set(style.show_result_masses)
        self._show_imperfect_reference.set(style.show_imperfect_reference)
        self._show_beam_sections.set(style.show_beam_sections)

    def sync_from_viewport(self) -> None:
        """Load the last applied style when this task page is opened."""

        self._load_style(self.app.viewport.visualization)

    def _apply(self) -> None:
        style = self._style()
        self.app.viewport.set_visualization(style)
        results = self.app.panels.get("Results")
        if results is not None:
            results.sync_visualization_from_viewport()
        # Rebuild geometry scenes so a changed curved-detail policy takes
        # effect immediately instead of waiting for the next workflow page.
        self.app.refresh_views()
        self.app.set_status(
            f"visualization applied: {style.render_mode}, "
            f"geometry detail {style.geometry_detail}"
        )
        self.refresh()

    def _reset(self) -> None:
        self._load_style(VisualizationStyle())
        self._apply()

    def refresh(self) -> None:
        if not hasattr(self, "_renderer_status"):
            return
        requested = self.app.viewport.requested_backend
        active = self.app.viewport.active_backend
        diagnostics = self.app.viewport.backend_diagnostics
        text = f"Requested: {requested}   Active: {active.upper()}"
        if diagnostics:
            text += "\n" + "\n".join(str(item) for item in diagnostics)
        renderer = getattr(self.app.viewport.canvas, "renderer_diagnostics", None)
        if renderer:
            text += (
                f"\nRetained meshes: {renderer.get('retained_meshes', 0)}   "
                f"GPU uploads: {renderer.get('geometry_uploads', 0)}   "
                f"draw calls: {renderer.get('draw_calls', 0)}"
            )
        self._renderer_status.configure(text=text)


# ----------------------------------------------------------------------
class ResultsPanel(StagePanel):
    title = "Results"

    def build(self) -> None:
        retained = self.section("Result set")
        self._result_set_section = retained
        row = ttk.Frame(retained)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="job", width=16).pack(side="left")
        self._job = tk.StringVar(value="-")
        self._job_ids = {}
        self._job_box = ttk.Combobox(
            row, textvariable=self._job, values=("-",), state="readonly", width=22
        )
        self._job_box.pack(side="left", fill="x", expand=True)
        self._job_box.bind("<<ComboboxSelected>>", lambda _event: self._pick_job())

        outcome = self.section("Analysis outcome")
        outcome_head = ttk.Frame(outcome)
        outcome_head.pack(fill="x", pady=(1, 3))
        self._outcome_status = ttk.Label(
            outcome_head, text="NO RESULT", font=("TkDefaultFont", 9, "bold")
        )
        self._outcome_status.pack(side="left")
        self._outcome_progress_text = ttk.Label(outcome_head, text="")
        self._outcome_progress_text.pack(side="right")
        metrics = ttk.Frame(outcome)
        metrics.pack(fill="x")
        self._outcome_values = {}
        for column, name in enumerate(("Start", "First converged", "Peak", "Last / target")):
            cell = ttk.Frame(metrics)
            cell.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 5, 0))
            metrics.columnconfigure(column, weight=1)
            ttk.Label(cell, text=name, foreground="#666666").pack(anchor="w")
            value = tk.StringVar(value="—")
            ttk.Label(cell, textvariable=value, font=("TkDefaultFont", 8, "bold")).pack(anchor="w")
            self._outcome_values[name] = value
        self._outcome_progress = ttk.Progressbar(
            outcome, orient="horizontal", mode="determinate", maximum=100.0
        )
        self._outcome_progress.pack(fill="x", pady=(4, 2))
        self._outcome_reason = ttk.Label(
            outcome, text="Run an analysis to see its outcome.", justify="left",
            wraplength=560,
        )
        self._outcome_reason.pack(fill="x", anchor="w")

        # Bound the task controls so adding visualization options cannot
        # collapse the lower path/probe workspace on typical laptop screens.
        setup_tabs = ttk.Notebook(self, height=220)
        setup_tabs.pack(fill="x", pady=(0, 6))
        self._setup_tabs = setup_tabs
        display_page = ttk.Frame(setup_tabs, padding=4)
        visualization_page = ttk.Frame(setup_tabs, padding=4)
        increments_page = ttk.Frame(setup_tabs, padding=4)
        quantities_page = ttk.Frame(setup_tabs, padding=4)
        setup_tabs.add(display_page, text="Display")
        setup_tabs.add(visualization_page, text="Visualization")
        setup_tabs.add(increments_page, text="Increments")
        setup_tabs.add(quantities_page, text="Quantities & tools")
        # Display controls are primary result navigation, not an advanced
        # section below the job outcome.  Keep these tabs immediately below
        # the Details page navigation so Visualization is visible without
        # scrolling or hunting below Result set / Analysis outcome.
        setup_tabs.pack_configure(before=retained)

        controls = self.section("Contour display", parent=display_page)
        row = ttk.Frame(controls)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="field", width=16).pack(side="left")
        self._component = tk.StringVar(value="magnitude")
        self._field_box = ttk.Combobox(
            row, textvariable=self._component, values=available_fields(),
            state="readonly", width=14,
        )
        self._field_box.pack(side="left", fill="x", expand=True)
        self._field_box.bind("<<ComboboxSelected>>", lambda _event: self._show())

        self._scale = self.entry_row(controls, "deform scale", "auto")
        self._limits = self.entry_row(controls, "colour range", "auto")
        units_row = ttk.Frame(controls)
        units_row.pack(fill="x", pady=1)
        ttk.Label(units_row, text="display units", width=16).pack(side="left")
        self._display_units = tk.StringVar(value=DISPLAY_UNIT_SYSTEMS[0])
        ttk.Combobox(
            units_row,
            textvariable=self._display_units,
            values=DISPLAY_UNIT_SYSTEMS,
            state="readonly",
            width=22,
        ).pack(side="left", fill="x", expand=True)
        self._display_units.trace_add("write", lambda *_args: self._display_changed())
        self._envelope = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls,
            text="envelope over every shape",
            variable=self._envelope,
            command=self.guarded(self._show),
        ).pack(anchor="w")
        display_actions = ttk.Frame(controls)
        display_actions.pack(fill="x", pady=(3, 0))
        ttk.Button(display_actions, text="Update view", command=self.guarded(self._show)).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(
            display_actions,
            text="Model view",
            command=self._model_view,
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))
        ttk.Button(display_actions, text="Mesh view", command=self.app.show_mesh).pack(
            side="left", fill="x", expand=True, padx=(4, 0)
        )

        visual_columns = ttk.Frame(visualization_page)
        visual_columns.pack(fill="both", expand=True)
        visual_columns.columnconfigure(0, weight=3)
        visual_columns.columnconfigure(1, weight=2)
        appearance = ttk.LabelFrame(
            visual_columns, text="Viewport appearance", padding=6
        )
        appearance.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        initial_style = self.app.viewport.visualization
        render_row = ttk.Frame(appearance)
        render_row.pack(fill="x", pady=1)
        ttk.Label(render_row, text="render", width=16).pack(side="left")
        self._render_mode = tk.StringVar(value=initial_style.render_mode)
        ttk.Combobox(
            render_row,
            textvariable=self._render_mode,
            values=RENDER_MODES,
            state="readonly",
            width=22,
        ).pack(side="left", fill="x", expand=True)
        self._surface_opacity = self.entry_row(
            appearance,
            "surface opacity [%]",
            f"{100.0 * initial_style.surface_opacity:g}",
        )
        self._edge_width = self.entry_row(
            appearance, "edge width [px]", str(initial_style.edge_width)
        )
        geometry_detail_row = ttk.Frame(appearance)
        geometry_detail_row.pack(fill="x", pady=1)
        ttk.Label(geometry_detail_row, text="geometry detail", width=16).pack(
            side="left"
        )
        self._geometry_detail = tk.StringVar(value=initial_style.geometry_detail)
        ttk.Combobox(
            geometry_detail_row,
            textvariable=self._geometry_detail,
            values=GEOMETRY_DETAILS,
            state="readonly",
            width=22,
        ).pack(side="left", fill="x", expand=True)
        self._background_color = self._colour_row(
            appearance, "background", initial_style.background
        )
        self._edge_color = self._colour_row(
            appearance, "element edges", initial_style.edge_color
        )
        palette_row = ttk.Frame(appearance)
        palette_row.pack(fill="x", pady=1)
        ttk.Label(palette_row, text="contour colours", width=16).pack(side="left")
        self._colormap = tk.StringVar(value=initial_style.result_colormap)
        ttk.Combobox(
            palette_row,
            textvariable=self._colormap,
            values=tuple(RESULT_COLORMAPS),
            state="readonly",
            width=22,
        ).pack(side="left", fill="x", expand=True)
        self._colormap.trace_add("write", lambda *_args: self._show_if_available())

        layers = ttk.LabelFrame(
            visual_columns, text="Visible result layers", padding=6
        )
        layers.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self._show_result_nodes = tk.BooleanVar(
            value=initial_style.show_result_nodes
        )
        self._show_result_legend = tk.BooleanVar(value=initial_style.show_legend)
        self._show_result_supports = tk.BooleanVar(
            value=initial_style.show_result_supports
        )
        self._show_result_loads = tk.BooleanVar(
            value=initial_style.show_result_loads
        )
        self._show_result_masses = tk.BooleanVar(
            value=initial_style.show_result_masses
        )
        # The purple stress-free reference is useful while defining an
        # imperfection, but obscures contours and modes.  Results default to a
        # clean view and let the engineer opt it back in.
        self._show_imperfect_reference = tk.BooleanVar(
            value=initial_style.show_imperfect_reference
        )
        self._show_beam_sections = tk.BooleanVar(
            value=initial_style.show_beam_sections
        )
        for index, (text, variable) in enumerate((
            ("beam sections in 3D", self._show_beam_sections),
            ("FE nodes", self._show_result_nodes),
            ("contour legend", self._show_result_legend),
            ("supports", self._show_result_supports),
            ("loads", self._show_result_loads),
            ("masses", self._show_result_masses),
            ("initial imperfect reference (purple wire)", self._show_imperfect_reference),
        )):
            ttk.Checkbutton(
                layers,
                text=text,
                variable=variable,
                command=self.guarded(self._apply_visualization),
            ).grid(row=index, column=0, sticky="w")
        actions = ttk.Frame(visualization_page)
        actions.pack(fill="x", pady=(3, 0))
        ttk.Button(
            actions,
            text="Apply visualization",
            command=self.guarded(self._apply_visualization),
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            actions,
            text="Reset",
            command=self.guarded(self._reset_visualization),
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        if getattr(self.app.viewport, "supports_section_planes", False):
            clipping_page = ttk.Frame(setup_tabs, padding=4)
            setup_tabs.add(clipping_page, text="Section plane")
            clipping = self.section("Section plane", parent=clipping_page)
            self._clip_normal = self.vector_row(
                clipping, "normal", ("1", "0", "0")
            )
            self._clip_offset = self.entry_row(clipping, "offset [m]", "0")
            self.button(clipping, "Apply clipping", self._apply_section_plane)
            self.button(clipping, "Clear clipping", self._clear_section_plane)

        quantities = self.section("Available quantities", parent=quantities_page)
        self._quantities = ttk.Treeview(
            quantities, columns=("location", "unit"), show="tree headings", height=4
        )
        self._quantities.heading("#0", text="Quantity")
        self._quantities.heading("location", text="Location")
        self._quantities.heading("unit", text="Unit")
        self._quantities.column("#0", width=125, stretch=True)
        self._quantities.column("location", width=70, stretch=False)
        self._quantities.column("unit", width=70, stretch=False)
        self._quantities.pack(fill="x")
        self._quantities.bind(
            "<Double-1>", lambda _event: self.guarded(self._activate_quantity)()
        )

        browse = self.section("Increment / shape", parent=increments_page)
        row = ttk.Frame(browse)
        row.pack(fill="x", pady=1)
        ttk.Button(row, text="<", width=3, command=self.guarded(self._previous)).pack(
            side="left"
        )
        self._shape = tk.StringVar(value="-")
        self._shape_box = ttk.Combobox(
            row, textvariable=self._shape, values=["-"], state="readonly", width=18
        )
        self._shape_box.pack(side="left", fill="x", expand=True, padx=2)
        self._shape_box.bind(
            "<<ComboboxSelected>>", lambda _e: self.guarded(self._pick_shape)()
        )
        ttk.Button(row, text=">", width=3, command=self.guarded(self._next)).pack(
            side="left"
        )
        self._play_button = ttk.Button(
            row, text="▶ Play", width=8, command=self.guarded(self._animate)
        )
        self._play_button.pack(side="left", padx=(3, 0))
        ttk.Label(row, text="speed").pack(side="left", padx=(8, 2))
        self._playback_fps = tk.StringVar(value="4")
        self._playback_speed_box = ttk.Combobox(
            row,
            textvariable=self._playback_fps,
            values=("0.5", "1", "2", "4", "8", "12", "20", "30"),
            state="readonly",
            width=4,
        )
        self._playback_speed_box.pack(side="left")
        ttk.Label(row, text="fps").pack(side="left", padx=(2, 0))
        self._frame_details = ttk.Label(
            browse, text="No saved increments.", foreground="#555555", wraplength=560
        )
        self._frame_details.pack(fill="x", anchor="w", pady=(2, 0))

        query = self.section("Tools", parent=quantities_page)
        tool_specs = (
            ("Probe", self._probe),
            ("Along line", self._along_line),
            ("Report", self._export_report),
            ("CSV", self._export_field),
            ("PNG", self._export_png),
            ("GIF", self._export_gif),
        )
        tool_buttons = {}
        for index, (label, command) in enumerate(tool_specs):
            button = ttk.Button(query, text=label, command=self.guarded(command))
            button.grid(row=0, column=index, sticky="ew", padx=2, pady=2)
            query.columnconfigure(index, weight=1)
            tool_buttons[label] = button
        png_button = tool_buttons["PNG"]
        gif_button = tool_buttons["GIF"]
        if not getattr(self.app.viewport, "capture_available", False):
            png_button.configure(state="disabled")
            gif_button.configure(state="disabled")

        # The lower workspace owns the remaining height.  Path, detailed probe
        # text and submitted inputs no longer compete vertically in three tiny
        # canvases; each gets the full area when selected.
        self._readout_tabs = ttk.Notebook(self)
        self._readout_tabs.pack(fill="both", expand=True, pady=(6, 0))
        path_page = ttk.Frame(self._readout_tabs)
        readout_page = ttk.Frame(self._readout_tabs)
        inputs_page = ttk.Frame(self._readout_tabs)
        self._readout_tabs.add(path_page, text="Path / increments")
        self._readout_tabs.add(readout_page, text="Probe / details")
        self._readout_tabs.add(inputs_page, text="Submitted inputs")
        path_actions = ttk.Frame(path_page)
        path_actions.pack(fill="x")
        ttk.Label(
            path_actions,
            text=(
                "Green = unloaded start   Orange = peak   Purple = last converged   "
                "Cyan = displayed increment"
            ),
            foreground="#555555",
        ).pack(side="left")
        ttk.Button(
            path_actions, text="Plot selected point", command=self.guarded(self._plot_at_selection)
        ).pack(side="right")
        self.plot = HistoryPlot(path_page, width=380, height=260)
        self.plot.pack(fill="both", expand=True, pady=(3, 0))

        self._summary = ttk.Label(
            readout_page, text="no results", justify="left", wraplength=600
        )
        self._summary.pack(fill="x", anchor="w", padx=4, pady=4)
        self._readout = tk.Text(
            readout_page, height=16, wrap="none", state="disabled"
        )
        self._readout.pack(fill="both", expand=True)
        self._result_inputs = tk.Text(
            inputs_page, height=16, wrap="none", state="disabled"
        )
        self._result_inputs.pack(fill="both", expand=True)
        self._animation_after = None
        self._animation_index = 0
        self._gif_export = None
        self._gif_poll_after = None
        self._gif_results: Queue = Queue()

    # ------------------------------------------------------------------
    def field_name(self) -> str:
        return self._component.get()

    def ensure_compatible_field(self) -> str:
        """Keep artifact field keys from leaking into live result display.

        Persisted artifacts store the vector quantity as ``displacement``;
        live ShapeView contours expose its scalar magnitude as ``magnitude``.
        Switching result sources must therefore validate the retained choice
        before the viewport evaluates it.
        """

        solution = self.app.solution
        if solution is None:
            dataset = self.app.result_datasets.get(self.app.active_job_id)
            choices = tuple(dataset.field_keys) if dataset is not None else ()
        else:
            available = getattr(solution, "available_fields", None)
            choices = tuple(
                available() if callable(available) else available_fields()
            )
        if choices:
            self._field_box.configure(values=choices)
            current = self._component.get()
            aliases = {
                "displacement": "magnitude",
                "mode_shape": "magnitude",
                "stress": "von_mises",
            }
            replacement = aliases.get(current, current)
            if replacement not in choices:
                replacement = "magnitude" if "magnitude" in choices else choices[0]
            if replacement != current:
                self._component.set(replacement)
        return self._component.get()

    def display_units(self) -> str:
        return self._display_units.get()

    def colormap(self):
        return RESULT_COLORMAPS.get(
            self._colormap.get(), RESULT_COLORMAPS["Cool-warm"]
        )

    def show_result_nodes(self) -> bool:
        return bool(self._show_result_nodes.get())

    def show_result_supports(self) -> bool:
        return bool(self._show_result_supports.get())

    def show_result_loads(self) -> bool:
        return bool(self._show_result_loads.get())

    def show_result_masses(self) -> bool:
        return bool(self._show_result_masses.get())

    def show_imperfect_reference(self) -> bool:
        return bool(self._show_imperfect_reference.get())

    def _colour_row(self, parent, label: str, value: str) -> tk.StringVar:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=16).pack(side="left")
        variable = tk.StringVar(value=value)
        ttk.Entry(row, textvariable=variable, width=14).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(
            row,
            text="Choose…",
            width=9,
            command=lambda: self._choose_colour(variable),
        ).pack(side="left", padx=(4, 0))
        return variable

    def _choose_colour(self, variable: tk.StringVar) -> None:
        _rgb, colour = colorchooser.askcolor(
            color=variable.get(), parent=self, title="Choose viewport colour"
        )
        if colour:
            variable.set(colour)

    def _visualization_style(self) -> VisualizationStyle:
        try:
            opacity = float(self._surface_opacity.get()) / 100.0
        except ValueError:
            raise ValueError("surface opacity must be a percentage from 0 to 100") from None
        try:
            edge_width = int(self._edge_width.get())
        except ValueError:
            raise ValueError("edge width must be an integer from 1 to 8") from None
        return VisualizationStyle(
            background=self._background_color.get().strip(),
            render_mode=self._render_mode.get(),
            surface_opacity=opacity,
            edge_color=self._edge_color.get().strip(),
            edge_width=edge_width,
            show_legend=bool(self._show_result_legend.get()),
            legend_font_size=self.app.viewport.visualization.legend_font_size,
            legend_width=self.app.viewport.visualization.legend_width,
            result_colormap=self._colormap.get(),
            show_result_nodes=bool(self._show_result_nodes.get()),
            show_result_supports=bool(self._show_result_supports.get()),
            show_result_loads=bool(self._show_result_loads.get()),
            show_result_masses=bool(self._show_result_masses.get()),
            show_imperfect_reference=bool(
                self._show_imperfect_reference.get()
            ),
            show_beam_sections=bool(self._show_beam_sections.get()),
            geometry_detail=self._geometry_detail.get(),
        )

    def _apply_visualization(self) -> None:
        self.app.viewport.set_visualization(self._visualization_style())
        panel = self.app.panels.get("Visualization")
        if panel is not None:
            panel.sync_from_viewport()
        self._show_if_available()

    def sync_visualization_from_viewport(self) -> None:
        """Keep the result-local appearance controls consistent globally."""

        style = self.app.viewport.visualization
        self._render_mode.set(style.render_mode)
        self._surface_opacity.set(f"{100.0 * style.surface_opacity:g}")
        self._edge_width.set(str(style.edge_width))
        self._geometry_detail.set(style.geometry_detail)
        self._background_color.set(style.background)
        self._edge_color.set(style.edge_color)
        self._show_result_legend.set(style.show_legend)
        self._colormap.set(style.result_colormap)
        self._show_result_nodes.set(style.show_result_nodes)
        self._show_result_supports.set(style.show_result_supports)
        self._show_result_loads.set(style.show_result_loads)
        self._show_result_masses.set(style.show_result_masses)
        self._show_imperfect_reference.set(style.show_imperfect_reference)
        self._show_beam_sections.set(style.show_beam_sections)

    def _reset_visualization(self) -> None:
        style = VisualizationStyle()
        self._render_mode.set(style.render_mode)
        self._surface_opacity.set(f"{100.0 * style.surface_opacity:g}")
        self._edge_width.set(str(style.edge_width))
        self._geometry_detail.set(style.geometry_detail)
        self._background_color.set(style.background)
        self._edge_color.set(style.edge_color)
        self._colormap.set(style.result_colormap)
        self._show_result_nodes.set(style.show_result_nodes)
        self._show_result_legend.set(style.show_legend)
        self._show_result_supports.set(style.show_result_supports)
        self._show_result_loads.set(style.show_result_loads)
        self._show_result_masses.set(style.show_result_masses)
        self._show_imperfect_reference.set(style.show_imperfect_reference)
        self._show_beam_sections.set(style.show_beam_sections)
        self._apply_visualization()

    def _show_if_available(self) -> None:
        if self.app.solution is not None or (
            self.app.active_job_id in self.app.result_datasets
        ):
            self.guarded(self._show)()

    def _display_changed(self) -> None:
        self._show_if_available()
        self.refresh()

    def colour_limits(self):
        text = self._limits.get().strip().lower()
        if text in ("", "auto"):
            return None
        parts = text.replace(",", " ").split()
        if len(parts) != 2:
            raise ValueError(
                "colour range must be 'auto' or two numbers, e.g. '0 30e6'"
            )
        try:
            low, high = float(parts[0]), float(parts[1])
        except ValueError:
            raise ValueError(f"colour range {text!r} is not two numbers") from None
        return low, high

    def field_values(self):
        """The Field to colour by: this shape, or the envelope of all of them."""

        if not self._envelope.get():
            return None
        solution = self.app.solution
        if solution is None:
            return None
        return envelope(solution, self.field_name()).field

    def write(self, text: str) -> None:
        self._readout.configure(state="normal")
        self._readout.delete("1.0", "end")
        self._readout.insert("1.0", text)
        self._readout.configure(state="disabled")
        self._readout_tabs.select(1)

    @staticmethod
    def _factor(value) -> str:
        return "—" if value is None else f"λ = {float(value):.5g}"

    def _clear_outcome(self, message: str = "Run an analysis to see its outcome.") -> None:
        self._outcome_status.configure(text="NO RESULT", foreground="#666666")
        for value in self._outcome_values.values():
            value.set("—")
        self._outcome_progress.configure(value=0.0)
        self._outcome_progress_text.configure(text="")
        self._outcome_reason.configure(text=message, foreground="#555555")

    def _update_outcome(self, solution) -> None:
        job_id = getattr(self.app, "active_job_id", None)
        submitted = getattr(self.app, "submitted_input_reports", {}).get(job_id)
        submitted_target = submitted_target_load_factor(submitted)
        path = nonlinear_path_summary(
            solution, target_load_factor=submitted_target
        )
        if path is None:
            status = str(getattr(solution, "status", "available"))
            self._outcome_status.configure(
                text=status.replace("_", " ").upper(), foreground="#b71c1c" if status.lower() not in {"ok","complete","completed","success","converged"} else "#2e7d32"
            )
            for value in self._outcome_values.values():
                value.set("—")
            self._outcome_progress.configure(value=0.0 if status.lower() not in {"ok","complete","completed","success","converged"} else 100.0)
            self._outcome_progress_text.configure(text="")
            self._outcome_reason.configure(
                text="This analysis has no nonlinear equilibrium path.",
                foreground="#555555",
            )
            return
        colour = {
            "success": "#2e7d32",
            "warning": "#b26a00",
            "error": "#b00020",
        }[path.severity]
        self._outcome_status.configure(
            text=path.status.replace("_", " ").upper(), foreground=colour
        )
        self._outcome_values["Start"].set("λ = 0 (unloaded)")
        self._outcome_values["First converged"].set(
            self._factor(path.first_converged_load_factor)
        )
        self._outcome_values["Peak"].set(self._factor(path.peak_load_factor))
        last = self._factor(path.last_converged_load_factor)
        target = self._factor(path.target_load_factor)
        self._outcome_values["Last / target"].set(f"{last} / {target}")
        fraction = path.progress_fraction
        self._outcome_progress.configure(value=0.0 if fraction is None else 100.0 * fraction)
        self._outcome_progress_text.configure(
            text="" if fraction is None else f"{100.0 * fraction:.1f}% of target"
        )
        details = [
            path.stop_reason,
            f"{path.converged_steps} converged increments",
            f"{path.total_iterations} Newton iterations",
        ]
        if path.status.casefold() == "stopped_at_limit":
            details.append(
                "numerical path stop; not by itself a verified capacity point"
            )
        if path.first_failed_load_factor is not None:
            details.append(f"first failed trial λ={path.first_failed_load_factor:.5g}")
        if path.failed_iteration_reason:
            details.append(path.failed_iteration_reason)
        if path.max_peeq is not None:
            details.append(f"max PEEQ={path.max_peeq:.5g}")
        details.extend(
            prescribed_path_progress(
                submitted,
                last_load_factor=path.last_converged_load_factor,
                target_load_factor=path.target_load_factor,
            )
        )
        built = getattr(solution, "built", None)
        project = getattr(built, "project", None)
        if (
            path.status.casefold() == "stopped_at_limit"
            and project is not None
            and not (getattr(project, "imperfections", ()) or ())
        ):
            details.append(
                "perfect geometry submitted (no imperfection); a buckling branch "
                "can be difficult to follow"
            )
        details.append(
            f"{path.saved_increments} committed contour frame(s) saved"
            if path.saved_increments
            else "increment contours were not saved"
        )
        self._outcome_reason.configure(text="  •  ".join(details), foreground=colour)

    def _update_frame_details(self) -> None:
        solution = self.app.solution
        if solution is None:
            dataset = self.app.result_datasets.get(self.app.active_job_id)
            count = 0 if dataset is None else len(dataset.frames)
            self._frame_details.configure(
                text=(
                    f"Persisted frame {self.app.shape_index + 1} of {count}."
                    if count else "Static result; no frame sequence."
                )
            )
            return
        shapes = getattr(solution, "shapes", None)
        if not shapes:
            self.plot.set_active_index(None)
            if nonlinear_path_summary(solution) is not None:
                self._frame_details.configure(
                    text=(
                        "Final converged state only. To animate real load increments, "
                        "enable ‘Save converged increment snapshots’ in Solve and rerun."
                    ),
                    foreground="#9a5b00",
                )
            else:
                self._frame_details.configure(text="Single static shape.", foreground="#555555")
            return
        index = min(max(self.app.shape_index, 0), len(shapes) - 1)
        shape = shapes[index]
        step = getattr(shape, "step", None)
        if step is None:
            self.plot.set_active_index(None)
            self._frame_details.configure(
                text=f"Shape {index + 1} of {len(shapes)} • value {float(shape.value):.5g}",
                foreground="#555555",
            )
            return
        # The nonlinear history adds the unloaded origin before converged
        # increments, hence the +1 correspondence to the saved frame index.
        self.plot.set_active_index(index + 1)
        pieces = [
            f"Saved increment {index + 1} of {len(shapes)}",
            f"λ={float(shape.value):.5g}",
            f"{int(getattr(step, 'iterations', 0))} iterations",
            f"residual={float(getattr(step, 'residual_norm', 0.0)):.3g}",
            f"|u|={float(getattr(step, 'displacement_norm', 0.0)):.5g} m",
            f"PEEQ={float(getattr(step, 'max_equivalent_plastic_strain', 0.0)):.5g}",
        ]
        self._frame_details.configure(text="  •  ".join(pieces), foreground="#333333")

    def _refresh_submitted_inputs(self) -> None:
        job_id = getattr(self.app, "active_job_id", None)
        report = getattr(self.app, "submitted_input_reports", {}).get(job_id)
        if report is None:
            dataset = self.app.result_datasets.get(job_id)
            if dataset is not None:
                provenance = dataset.metadata("provenance")
                submitted = provenance.get("submitted_inputs")
                if submitted is not None:
                    import json

                    report = json.dumps(
                        submitted, indent=2, sort_keys=True, allow_nan=False
                    )
                elif provenance.get("submitted_inputs_text"):
                    report = str(provenance["submitted_inputs_text"])
        if report is None:
            report = (
                "Submitted input text is unavailable for this reopened/legacy result. "
                "Its hashes and provenance remain in the result artifact."
            )
        self._result_inputs.configure(state="normal")
        self._result_inputs.delete("1.0", "end")
        self._result_inputs.insert("1.0", report)
        self._result_inputs.configure(state="disabled")

    # ------------------------------------------------------------------
    def _require_solution(self):
        if self.app.solution is None:
            raise ValueError("run a solve first")
        return self.app.current_shape()

    def _model_view(self) -> None:
        """Show selectable design geometry while retaining this result task."""

        kind = self.app.selection.mode
        if kind not in ("vertex", "edge"):
            kind = "vertex"
        self.app.show_geometry()
        self.app.selection_strip.set_context(
            kind,
            "Result probing • select model Points or Lines, then use Probe / Along line",
        )
        self.app.details.set_hint("Model view for result probes")
        self.app.set_status(
            "model geometry shown; select points or lines, then use Probe or Along line"
        )

    def _probe(self) -> None:
        shape = self._require_solution()
        items = self.require_selection(self.app.selection.mode)
        engineering = self.display_units() == ENGINEERING_DISPLAY
        def load(cancel):
            readouts = [
                probe(shape, ref).text(
                    length_scale=1000.0 if engineering else 1.0,
                    length_unit="mm" if engineering else "m",
                    stress_scale=1.0e-6 if engineering else 1.0,
                    stress_unit="MPa" if engineering else "Pa",
                )
                for ref in items
            ]
            return (chr(10)*2).join(readouts)
        def publish(text):
            self.write(text);self.app.set_status(f"probed {len(items)} entity(ies)")
        self.app._request_result(load,publish,channel="inspection")

    def _along_line(self) -> None:
        shape = self._require_solution()
        edges = self.require_selection("edge", 1)
        name,units=self.field_name(),self.display_units()
        def load(cancel):
            result=along_line(shape,edges[0],name)
            xs,xunit=unit_transform("m",units);ys,yunit=unit_transform(result.unit,units)
            return (f"{name} along {edges[0]} [{xunit}, {yunit}]\n"+
                    "\n".join(f"{x*xs:.6g}, {y*ys:.6g}" for x,y in zip(result.distances,result.values)))
        self.app._request_result(load,lambda text:self.write(text),channel="inspection")

    def _export_report(self) -> None:
        active_job_id = getattr(self.app, "active_job_id", None)
        dataset = getattr(self.app, "result_datasets", {}).get(active_job_id)
        solution = self.app.solution
        if dataset is None and solution is None:
            raise ValueError("run or open a result first")
        path = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=(
                [
                    ("Markdown", "*.md"),
                    ("HTML", "*.html"),
                    ("All files", "*.*"),
                ]
                if dataset is not None
                else [("Markdown", "*.md"), ("All files", "*.*")]
            ),
            initialfile=f"{self.app.project.name}_report.md",
        )
        if not path:
            return
        project,session=self.app.project,self.app.session
        stale_method=getattr(self.app,"_job_is_stale",None)
        job=getattr(project,"jobs",{}).get(active_job_id)
        stale=stale_method(job) if callable(stale_method) and job is not None else None
        revision=getattr(session,"revision",None)
        document_hash=str(getattr(revision,"document_hash",""))
        def load(cancel):
            if dataset is None:
                if str(path).casefold().endswith((".html", ".htm")):
                    raise ValueError(
                        "HTML reporting requires the retained result artifact; "
                        "save the project and retry when result persistence completes"
                    )
                from ..post.report import report_markdown
                from .result_export import write_client_export
                written = write_client_export(path,report_markdown(solution),cancel,source=str(active_job_id or "current solved source"))
            else:
                context = result_report_context(
                    dataset,
                    project=project,
                    job=job,
                    current_document_hash=document_hash,
                    stale=stale,
                )
                from ..post.report import result_report_markdown,result_report_html
                from .result_export import write_client_export
                text=(result_report_html if str(path).casefold().endswith((".html",".htm")) else result_report_markdown)(dataset,context=context)
                written = write_client_export(path,text,cancel,source=str(active_job_id))
            return written
        self.app._request_result(load,lambda written:None,dataset=dataset,channel="export",ordered=True)

    def _export_field(self) -> None:
        dataset = None
        if self.app.solution is None and self.app.active_job_id is not None:
            dataset = self.app.result_datasets.get(self.app.active_job_id)
        if dataset is None:
            shape = self._require_solution()
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            initialfile=f"{self.app.project.name}_{self.field_name()}.csv",
        )
        if not path:
            return
        key,frame=self.field_name(),self.app.shape_index
        job_id=self.app.active_job_id
        def load(cancel):
            text=lazy_field_to_csv(dataset,key,frame=dataset.field(key).root_frame_index(frame)) if dataset is not None else field_to_csv(shape,key)
            from .result_export import write_client_export
            return write_client_export(path,text,cancel,source=str(job_id or "current solved source"))
        self.app._request_result(load,lambda written:None,dataset=dataset,channel="export",ordered=True)

    def _require_result_set(self) -> None:
        if self.app.solution is not None:
            return
        if (
            self.app.active_job_id is not None
            and self.app.active_job_id in self.app.result_datasets
        ):
            return
        raise ValueError("run or open a result first")

    def _require_capturable_result(self) -> None:
        """Reject table/history quantities instead of capturing stale geometry."""

        self._require_result_set()
        if self.app.solution is not None:
            return
        dataset = self.app.result_datasets[self.app.active_job_id]
        key = self.field_name()
        if key not in dataset.field_keys:
            raise ValueError(f"retained result has no field {key!r}")
        location = dataset.field(key).descriptor.location
        if location not in (
            "node", "element", "element_face", "integration_point"
        ):
            raise ValueError(
                f"{key!r} is a {location} table, not a spatial field; "
                "export it as CSV instead"
            )

    def _render_result_frame(self, index: int, *, on_ready=None, on_cancel=None) -> None:
        self.app.shape_index = int(index)
        if self.app.solution is None:
            self._show(on_ready=on_ready,on_cancel=on_cancel)
        else:
            self.app.show_results(on_ready=on_ready,on_cancel=on_cancel)

    def _export_png(self) -> None:
        self._require_capturable_result()
        if not getattr(self.app.viewport, "capture_available", False):
            raise ValueError("viewport PNG export is unavailable on this desktop")
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            initialfile=f"{self.app.project.name}_{self.field_name()}.png",
        )
        if not path:
            return
        def capture():
            written=self.app.viewport.capture_png(path)
            self.app.set_status(f"viewport image written to {written}")
        self._render_result_frame(self.app.shape_index,on_ready=capture)

    def _export_gif(self) -> None:
        """Capture result shapes incrementally so Tk remains responsive."""

        self._require_capturable_result()
        if not getattr(self.app.viewport, "capture_available", False):
            raise ValueError("viewport GIF export is unavailable on this desktop")
        count = self._shape_count()
        if count < 2:
            raise ValueError("this result has only one shape to export")
        if self._gif_export is not None or self._gif_poll_after is not None:
            raise ValueError("a GIF export is already in progress")
        path = filedialog.asksaveasfilename(
            defaultextension=".gif",
            filetypes=[("GIF animation", "*.gif"), ("All files", "*.*")],
            initialfile=f"{self.app.project.name}_{self.field_name()}.gif",
        )
        if not path:
            return
        if self._animation_after is not None:
            self.after_cancel(self._animation_after)
            self._animation_after = None
        self._gif_export = {
            "path": path,
            "index": 0,
            "count": count,
            "original": self.app.shape_index,
            "images": [],
        }
        self.app.set_status(f"capturing GIF frame 1 of {count}")
        self.after_idle(self._capture_gif_frame)

    def _capture_gif_frame(self) -> None:
        state = self._gif_export
        if state is None:
            return
        try:
            index = int(state["index"])
            count = int(state["count"])
            if index >= count:
                self._finish_gif_capture()
                return
            def capture():
                if self._gif_export is not state:return
                try:
                    state["images"].append(self.app.viewport.capture_image())
                    state["index"]=index+1
                    self.after(1,self._capture_gif_frame)
                except Exception as error:self._abort_gif_capture(error)
            self._render_result_frame(index,on_ready=capture,on_cancel=lambda:self._abort_gif_capture(RuntimeError("capture request cancelled")))
        except Exception as error:  # asynchronous callbacks cannot use guarded()
            self._abort_gif_capture(error)

    def _restore_gif_shape(self, state) -> None:
        try:
            self._render_result_frame(int(state["original"]))
            self.app.refresh_panels()
        except Exception:
            # Preserve the export diagnostic if a closing window cannot redraw.
            pass

    def _finish_gif_capture(self) -> None:
        state = self._gif_export
        if state is None:
            return
        self._gif_export = None
        self._restore_gif_shape(state)
        path = state["path"]
        images = list(state["images"])

        def encode() -> None:
            try:
                self._gif_results.put((save_gif(images, path), None))
            except Exception as error:
                self._gif_results.put((None, error))

        Thread(target=encode, name="anyfem-gif-export", daemon=True).start()
        self.app.set_status("encoding GIF animation")
        self._gif_poll_after = self.after(50, self._poll_gif_export)

    def _poll_gif_export(self) -> None:
        try:
            written, error = self._gif_results.get_nowait()
        except Empty:
            self._gif_poll_after = self.after(50, self._poll_gif_export)
            return
        self._gif_poll_after = None
        if error is not None:
            self.app.set_status(f"GIF export failed: {error}", error=True)
        else:
            self.app.set_status(f"GIF animation written to {written}")

    def _abort_gif_capture(self, error: Exception) -> None:
        state = self._gif_export
        self._gif_export = None
        if state is not None:
            self._restore_gif_shape(state)
        self.app.set_status(f"GIF export failed: {error}", error=True)

    def _apply_section_plane(self) -> None:
        normal = self.vector(self._clip_normal, "section-plane normal")
        offset = self.number(self._clip_offset, "section-plane offset")
        self.app.viewport.set_section_plane(normal, offset, enabled=True)
        self.app.set_status("section-plane clipping enabled")

    def _clear_section_plane(self) -> None:
        self.app.viewport.clear_section_plane()
        self.app.set_status("section-plane clipping cleared")

    def _plot_at_selection(self) -> None:
        """Trace the history at a selected point rather than the peak node."""

        solution = self.app.solution
        if solution is None:
            raise ValueError("run a solve first")
        points = self.require_selection("vertex", count=1)
        node = self.app.current_shape().built.mesh.node_of_vertex.get(points[0].id)
        if node is None:
            raise ValueError(f"{points[0]} has no node in the mesh")
        name=self.field_name() if self.field_name() in DOF_NAMES else "uz"
        units=self.display_units()
        def publish(series):
            if not series:raise ValueError("this result has no history to plot")
            self.plot.show(series);self._readout_tabs.select(0)
            self.app.set_status(f"plotting at node {node}")
        self.app._request_result(lambda cancel:converted_series(history_series(solution,probe=node,component=name),units),publish,channel="history")

    def refresh(self) -> None:
        records = list(getattr(self.app.project, "jobs", {}).values())
        labels = []
        self._job_ids = {}
        for record in reversed(records):
            stale = self.app._job_is_stale(record)
            status = getattr(record.status, "value", str(record.status))
            retained = getattr(self.app, "solutions", {}).get(record.id)
            analysis_status = str(getattr(retained, "status", "")).strip()
            if status == "completed" and analysis_status and analysis_status not in {
                "completed", "converged", "ok"
            }:
                status = (
                    f"job completed; analysis {analysis_status.replace('_', ' ')}"
                )
            label = f"{record.name} [{status}{', stale' if stale else ''}]"
            labels.append(label)
            self._job_ids[label] = record.id
        self._job_box.configure(values=labels or ["-"])
        active_id = getattr(self.app, "active_job_id", None)
        active_label = next(
            (label for label, identifier in self._job_ids.items() if identifier == active_id),
            None,
        )
        if active_label is not None:
            self._job.set(active_label)
        elif self._job.get() not in self._job_ids:
            self._job.set(labels[0] if labels else "-")
        self._refresh_submitted_inputs()

        solution = self.app.solution
        if solution is None:
            dataset = self.app.result_datasets.get(active_id)
            if dataset is not None and self.app.prepare_result_dataset(dataset):return
            if dataset is not None:
                keys = dataset.field_keys
                self._field_box.configure(values=keys)
                if keys and self._component.get() not in keys:
                    self._component.set(keys[0])
                self._refresh_quantities(dataset)
                frames = dataset.frames
                labels = (
                    [
                        f"frame {index + 1} ({value:.5g})"
                        for index, value in enumerate(frames)
                    ]
                    if len(frames)
                    else ["static"]
                )
                self._shape_box.configure(values=labels)
                self.app.shape_index = min(self.app.shape_index, len(labels) - 1)
                self._shape.set(labels[self.app.shape_index])
                summary = dataset.metadata("summary")
                text = (
                    summary.get("text")
                    or summary.get("solution_type")
                    or "Persisted result"
                )
                deformation = (
                    "deformation available"
                    if "displacement" in keys
                    else "deformation unavailable (undeformed display)"
                )
                self._summary.configure(text=f"{text}\n{deformation}")
                status = str(summary.get("status", "persisted result"))
                self._outcome_status.configure(
                    text=status.replace("_", " ").upper(), foreground="#b71c1c" if dataset.identity.get("partial",False) or status.lower() not in {"ok","complete","completed","success","converged"} else "#2e7d32"
                )
                for value in self._outcome_values.values():
                    value.set("—")
                self._outcome_progress.configure(value=0.0 if dataset.identity.get("partial",False) or status.casefold() not in {"ok","complete","completed","success","converged"} else 100.0)
                self._outcome_progress_text.configure(
                    text=f"{len(frames)} saved frame(s)" if len(frames) else "static"
                )
                self._outcome_reason.configure(
                    text=(
                        "Loaded from retained result artifact. Detailed nonlinear "
                        "termination metadata is shown when present in the report/inputs."
                    ),
                    foreground="#555555",
                )
                self._update_frame_details()
                self.plot.clear()
                return
            self._summary.configure(text="no results")
            self._shape_box.configure(values=["-"])
            self._shape.set("-")
            self.plot.clear()
            self._quantities.delete(*self._quantities.get_children())
            self._clear_outcome()
            self._update_frame_details()
            return

        # The plot follows the result: a transient or an incremental solve has
        # a history, a linear static does not, and an empty plot says so rather
        # than keeping the previous result's curve on screen.
        units=self.display_units()
        self.app._request_result(lambda cancel:converted_series(history_series(solution),units),self.plot.show,channel="history")
        self._update_outcome(solution)

        # An imported result names its own fields, which are the file's, not
        # ANYfem's list.
        self.ensure_compatible_field()
        self._refresh_quantities(solution)

        shapes = getattr(solution, "shapes", None)
        if shapes:
            labels = [shape.label for shape in shapes]
            self._shape_box.configure(values=labels)
            index = min(self.app.shape_index, len(labels) - 1)
            self._shape.set(labels[index])
        else:
            self._shape_box.configure(values=["static"])
            self._shape.set("static")
        self._update_frame_details()

        shape = self.app.current_shape()
        # Summary refresh must not scan every nodal DOF on the owner thread.
        displacement_text = "Translation statistics load with the requested result field."
        constitutive = self._constitutive_summary(shape)
        self._summary.configure(
            text=(
                f"{solution.summary()}\n"
                f"showing {getattr(shape, 'label', 'static')}\n"
                f"{displacement_text}\n{constitutive}"
            )
        )

    @staticmethod
    def _constitutive_summary(shape) -> str:
        from ..presentation.engineering_summary import constitutive_summary
        return constitutive_summary(shape)

    def _pick_job(self) -> None:
        job_id = self._job_ids.get(self._job.get())
        if job_id is None:
            return
        solution = self.app.solutions.get(job_id)
        if solution is None:
            try:
                solution = self.app.job_manager.result(job_id)
            except KeyError:
                dataset = self.app.result_datasets.get(job_id)
                if dataset is None:
                    raise ValueError(
                        "this retained job has no in-memory result and its "
                        "artifact is unavailable"
                    )
                self.app.active_job_id = job_id
                self.app.solution = None
                self.app.shape_index = 0
                self.app.refresh_panels()
                if dataset.field_keys and self.app.mesh is not None:
                    self._component.set(dataset.field_keys[0])
                    self._show()
                return
        self.app.active_job_id = job_id
        self.app.solution = solution
        self.app.shape_index = 0
        self.app.show_results()
        self.app.refresh_panels()

    def _refresh_quantities(self, solution) -> None:
        self._quantities.delete(*self._quantities.get_children())
        if hasattr(solution, "field_keys") and hasattr(solution, "field"):
            for key in solution.field_keys:
                descriptor = solution.field(key).descriptor
                self._quantities.insert(
                    "",
                    "end",
                    iid=f"artifact:{key}",
                    text=descriptor.label,
                    values=(descriptor.location, descriptor.unit),
                )
            return
        from ..post.solver_data import solution_quantity_descriptors

        descriptors = solution_quantity_descriptors(solution)
        if not descriptors:
            fields = getattr(solution, "available_fields", None)
            values = fields() if callable(fields) else available_fields()
            for key in values:
                self._quantities.insert(
                    "", "end", iid=f"field:{key}", text=str(key), values=("derived", "")
                )
            return
        for descriptor in descriptors:
            identifier = str(getattr(descriptor, "quantity_id", "quantity"))
            key = identifier
            suffix = 2
            while self._quantities.exists(key):
                key = f"{identifier}:{suffix}"
                suffix += 1
            self._quantities.insert(
                "", "end", iid=key,
                text=str(getattr(descriptor, "label", identifier)),
                values=(
                    str(getattr(descriptor, "location", "")),
                    str(getattr(descriptor, "unit", "")),
                ),
            )
    def _activate_quantity(self) -> None:
        selected = self._quantities.selection()
        if not selected:
            return
        quantity = selected[0].split(":", 1)[0]
        if quantity == "artifact":
            self._component.set(selected[0].split(":", 1)[1])
            self._show()
            return
        mapping = {
            "displacement": "magnitude",
            "mode_shape": "magnitude",
            "stress": "von_mises",
            "equivalent_plastic_strain": "equivalent_plastic_strain",
            "field": selected[0].split(":", 1)[-1],
        }
        field = mapping.get(quantity)
        values = tuple(self._field_box.cget("values"))
        if field in values:
            self._component.set(field)
            self._show()

    # ------------------------------------------------------------------
    def _shape_count(self) -> int:
        if self.app.solution is None and self.app.active_job_id is not None:
            dataset = self.app.result_datasets.get(self.app.active_job_id)
            if dataset is not None:
                return max(1, len(dataset.frames))
        shapes = getattr(self.app.solution, "shapes", None)
        return len(shapes) if shapes else 1

    def _pick_shape(self) -> None:
        values = list(self._shape_box.cget("values"))
        if self._shape.get() in values:
            self.app.shape_index = values.index(self._shape.get())
            if self.app.solution is None:
                self._show()
            else:
                self.app.show_results()
            self.app.refresh_panels()

    def _step(self, delta: int) -> None:
        if (
            self.app.solution is None
            and self.app.active_job_id not in self.app.result_datasets
        ):
            raise ValueError("run a solve first")
        count = self._shape_count()
        self.app.shape_index = (self.app.shape_index + delta) % count
        if self.app.solution is None:
            self._show()
        else:
            self.app.show_results()
        self.app.refresh_panels()

    def _previous(self) -> None:
        self._step(-1)

    def _next(self) -> None:
        self._step(1)

    def _animate(self) -> None:
        """Play available frames with Tk timers; unavailable frames stay absent."""

        if (
            self.app.solution is None
            and self.app.active_job_id not in self.app.result_datasets
        ):
            raise ValueError("run a solve first")
        count = self._shape_count()
        if count < 2:
            raise ValueError("this result has only one shape to show")
        if self._animation_after is not None:
            self.after_cancel(self._animation_after)
            self._animation_after = None
            self._play_button.configure(text="▶ Play")
            self.app.set_status("playback stopped")
            return
        self._animation_index = (
            0 if self.app.shape_index + 1 >= count else self.app.shape_index + 1
        )
        self._play_button.configure(text="■ Stop")
        self.app.set_status(f"playback started at {self.playback_fps():g} fps")
        self._animation_step()

    def playback_fps(self) -> float:
        """Selected visual playback rate; result time/load values stay unchanged."""

        value = float(self._playback_fps.get())
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("playback speed must be a positive number of frames/s")
        return value

    def _playback_delay_ms(self) -> int:
        return max(1, int(round(1000.0 / self.playback_fps())))

    def _animation_step(self) -> None:
        count = self._shape_count()
        if self._animation_index >= count:
            self._animation_after = None
            self._play_button.configure(text="▶ Play")
            self.app.refresh_panels()
            return
        self.app.shape_index = self._animation_index
        self._animation_index += 1
        values = list(self._shape_box.cget("values"))
        if self.app.shape_index < len(values):
            self._shape.set(values[self.app.shape_index])
        self._update_frame_details()
        if self.app.solution is None:
            self._show()
        else:
            self.app.show_results()
        self._animation_after = self.after(
            self._playback_delay_ms(), self._animation_step
        )

    def scale_value(self, solution) -> float:
        if self._scale.get().strip().lower() in ("", "auto"):
            from ..presentation.result_display import deformation_scale
            return deformation_scale(self._scale.get(),self.app.project,solution)
        return self.number(self._scale, "deform scale")

    def component(self) -> str:
        return self._component.get()

    def _show(self, *, on_ready=None,on_cancel=None) -> None:
        if self.app.solution is None:
            if self.app.active_job_id not in self.app.result_datasets:
                raise ValueError("run a solve first")
            dataset = self.app.result_datasets[self.app.active_job_id]
            key = self._component.get()
            if self.app.prepare_result_dataset(dataset,on_ready=lambda:self._show(on_ready=on_ready,on_cancel=on_cancel),on_cancel=on_cancel):return
            stored = dataset.field(key)
            descriptor = stored.descriptor
            if descriptor.location in ("global", "history", "support", "contact"):
                frame = stored.root_frame_index(self.app.shape_index)
                value_scale, display_unit = unit_transform(
                    descriptor.unit, self.display_units()
                )
                def load(cancel):
                    values=np.asarray(stored.read(frame))*value_scale
                    return f"{descriptor.label} [{display_unit}]\n"+np.array2string(values,precision=7,threshold=200)
                def publish(text):
                    self.write(text)
                    self.app.set_status(f"showing {descriptor.label} as a result table")
                    if on_ready is not None:on_ready()
                self.app._request_result(load,publish,dataset=dataset,on_cancel=on_cancel)
                return
            scale_text = self._scale.get().strip().lower()
            scale = (
                1.0
                if scale_text in ("", "auto")
                else self.number(self._scale, "deform scale")
            )
            self.app.show_persisted_result(
                key,
                frame=self.app.shape_index,
                scale=scale,
                limits=self.colour_limits(),
                on_ready=on_ready,on_cancel=on_cancel,
            )
            self.app.details.set_hint("Result contour")
            return
        self.app.show_results(on_ready=on_ready,on_cancel=on_cancel)
        self.app.details.set_hint("Result contour")

    def destroy(self) -> None:
        """Cancel panel-owned timers before Tcl discards their callbacks."""

        self._gif_export = None
        for attribute in ("_animation_after", "_gif_poll_after"):
            identifier = getattr(self, attribute, None)
            if identifier is not None:
                try:
                    self.after_cancel(identifier)
                except tk.TclError:
                    pass
                setattr(self, attribute, None)
        super().destroy()
