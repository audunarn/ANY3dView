"""The ANYfem application window.

Layout: model tree on the left, 3D viewport in the middle, stage panels on the
right, status bar along the bottom. Shared workbench state and commands belong
to the toolkit-neutral application controller; this module is the Tk adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
import tkinter as tk
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, Dict, Iterable, Optional

import numpy as np

from ..application import WorkbenchController, default_project as _default_project
from ..application.workflow import WorkbenchWorkflow, _solution_report, _job_progress_text, _record_settings, _submitted_input_report, _execute_analysis_job, _format_preflight_errors
from ..commands import Command
from ..document import canonical_hash
from ..diagnostics import ErrorDiagnostic, build_diagnostic_report
from ..jobs import analysis_hash
from ..mesh_controls import MeshControls, StructuredMeshControls
from ..mesh_jobs import (
    clone_mesh_for_job,
    MeshJobResult,
    MeshSettings,
    mesh_semantic_hash,
)
from ..io.decks import export_calculix_deck
from ..io.project_file import (
    load_project, project_from_dict, project_to_dict, save_project,
)
from ..io.recovery import (
    ProjectLock, discover_recoveries, load_recovery, write_autosave,
)
from ..io.results import import_calculix_results, import_sesam_results
from ..io.result_artifact import write_solution_artifact
from ..io.sesam import import_sesam
from ..solve.build import build_fe_model
from ..model.project import Project, ProjectError
from ..model.records import AnalysisDefinition, MeshRecord
from ..selection import MeshEntityRef, mode_label
from ..solve.run import (
    solve_arc_length,
    solve_buckling,
    solve_capacity,
    solve_impact,
    solve_linear_static,
    solve_linear_static_many,
    solve_modal,
    solve_nonlinear_static,
    solve_transient,
)

from .panels import (
    GeometryPanel,
    LoadPanel,
    MeshPanel,
    ResultsPanel,
    SectionPanel,
    SolvePanel,
    VisualizationPanel,
)
from .definitions import DefinitionsPanel
from .scene import (
    Polyline,
    Scene,
    build_attribute_overlay,
    build_collision_overlay,
    build_geometry_scene,
    build_mesh_scene,
    build_persisted_result_scene,
    build_result_scene,
    geometry_display_resolution,
)
from .tree import ModelTree
from .viewport import Viewport
from .worker import JobWorkerFacade
from .workspace import DetailsWorkspace, JobStatusView, SelectionStrip
from .scripting import ScriptingPanel
from .tk_adapters import (
    CallbackStatusPort,
    TkClipboardPort,
    TkDialogPort,
    TkSchedulerPort,
)

__all__ = ["ANALYSES", "AnyFemApp", "main"]

# Which function each analysis name on the Solve panel runs.
from ..application.workflow import ANALYSES

_RENDERER_LABELS = {
    "auto": "Automatic",
    "gpu": "GPU",
    "software": "Tk",
}
_RENDERER_BACKENDS = {label: backend for backend, label in _RENDERER_LABELS.items()}


from ..application.workflow import _WorkbenchField


class AnyFemApp(WorkbenchWorkflow, ttk.Frame):
    """The main window."""

    project = _WorkbenchField("project")
    selection = _WorkbenchField("selection")
    session = _WorkbenchField("session")
    commands = _WorkbenchField("commands")
    job_manager = _WorkbenchField("job_manager")
    mesh_task_manager = _WorkbenchField("mesh_task_manager")
    mesh = _WorkbenchField("mesh")
    _meshes = _WorkbenchField("meshes")
    mesh_record_id = _WorkbenchField("mesh_record_id")
    solution = _WorkbenchField("solution")
    solutions = _WorkbenchField("solutions")
    result_datasets = _WorkbenchField("result_datasets")
    submitted_input_reports = _WorkbenchField("submitted_input_reports")
    active_job_id = _WorkbenchField("active_job_id")
    analysis = _WorkbenchField("analysis")
    shape_index = _WorkbenchField("shape_index")
    imported = _WorkbenchField("imported")
    path = _WorkbenchField("path")
    seeding_overrides = _WorkbenchField("seeding_overrides")
    _view_mode = _WorkbenchField("view_mode")
    _geometry_selection_mode = _WorkbenchField("geometry_selection_mode")
    _active_model_hash = _WorkbenchField("active_model_hash")

    def switch_viewer_backend(self, backend: str) -> str:
        """Switch renderers without changing application or project state."""

        active = self.viewport.switch_backend(backend)
        self._viewer_backend = self.viewport.requested_backend
        self._renderer_choice.set(_RENDERER_LABELS[self._viewer_backend])
        self._update_renderer_label()
        diagnostics = self.viewport.backend_diagnostics
        detail = f"; {'; '.join(diagnostics)}" if diagnostics else ""
        self.set_status(f"renderer: {'GPU' if active == 'gpu' else 'Tk'}{detail}")
        return active


    def __init__(
        self,
        master: tk.Misc,
        project: Optional[Project] = None,
        *,
        viewer_backend: str = "auto",
        viewer_host=None,
    ) -> None:
        super().__init__(master)
        normalized_backend = str(viewer_backend).strip().casefold()
        normalized_backend = {"automatic": "auto", "tk": "software"}.get(
            normalized_backend, normalized_backend
        )
        if normalized_backend not in _RENDERER_LABELS:
            raise ValueError("viewer_backend must be 'auto', 'gpu', or 'software'")
        self._viewer_backend = normalized_backend
        self._viewer_host = viewer_host
        self.workbench = WorkbenchController(
            project if project is not None else _default_project()
        )
        self.scheduler = TkSchedulerPort(self)
        self.dialogs = TkDialogPort(
            lambda: self.winfo_toplevel(),
            messagebox=messagebox,
            filedialog=filedialog,
            simpledialog=simpledialog,
        )
        self.clipboard = TkClipboardPort(self)
        self._status_port = CallbackStatusPort(self.set_status)
        self._artifact_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="anyfem-artifact"
        )
        self._recovery_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="anyfem-recovery"
        )
        self._artifact_futures: Dict[str, Future] = {}
        self._artifact_destinations: Dict[str, Path] = {}
        self._log_futures: Dict[str, Future] = {}
        self._log_destinations: Dict[str, Path] = {}
        self._active_mesh_task_id: Optional[str] = None
        self._mesh_details_record_id: Optional[str] = None
        self._mesh_layout_preview: tuple[Any, Any] | None = None
        self._inspection_mesh = None
        self._geometry_scene_cache: tuple[object, Scene, Scene] | None = None
        self._closing = False
        self._refresh_suspended = 0
        self._project_lock: ProjectLock | None = None
        self._root_bindings: list[tuple[str, str]] = []
        self._autosave_after = None
        self._autosave_hard_after = None
        self._recovery_future: Future | None = None
        self._recovery_future_epoch = 0
        self._recovery_pending: tuple[int, dict[str, Any], dict[str, Any]] | None = None
        self._recovery_epoch = 0
        self._recent_paths = self._load_recent_paths()
        self._error_diagnostics: list[ErrorDiagnostic] = []
        from ..application.workflow import _loading_hash
        self._active_loading_hash = _loading_hash(self.project)

        self._build()
        self._build_menu()

        self.commands.add_listener(self.refresh_all)
        self.session.add_listener(self._on_revision_changed)
        self.selection.add_listener(self._on_selection_changed)
        self.worker = JobWorkerFacade(self.job_manager)
        self._job_poll = self.scheduler.call_later(
            self.worker.POLL_MS, self._poll_jobs
        )

        self.refresh_all()
        self.show_geometry(reset_view=True)
        self._update_window_title()
        try:
            self.winfo_toplevel().protocol("WM_DELETE_WINDOW", self.request_close)
            self.winfo_toplevel().bind("<Control-p>", self.show_command_palette)
            self.winfo_toplevel().bind("<Control-P>", self.show_command_palette)
            self._bind_root_shortcut("<Control-z>", self._undo_shortcut)
            self._bind_root_shortcut("<Control-Z>", self._undo_shortcut)
            self._bind_root_shortcut("<Control-y>", self._redo_shortcut)
            self._bind_root_shortcut("<Control-Y>", self._redo_shortcut)
        except tk.TclError:  # pragma: no cover - embedded frame
            pass

    # ------------------------------------------------------------------
    def _build(self) -> None:
        self.pack(fill="both", expand=True)

        toolbar = ttk.Frame(self, padding=(6, 4))
        toolbar.pack(fill="x")
        self._undo_button = ttk.Button(toolbar, text="Undo", command=self.undo)
        self._undo_button.pack(side="left")
        self._redo_button = ttk.Button(toolbar, text="Redo", command=self.redo)
        self._redo_button.pack(side="left", padx=(4, 12))
        for label, name in (
            ("Iso", "iso"), ("Top", "top"), ("Front", "front"), ("Side", "side")
        ):
            ttk.Button(
                toolbar, text=label, width=6,
                command=lambda n=name: self.viewport.set_view(n),
            ).pack(side="left", padx=1)
        ttk.Button(toolbar, text="Fit", width=6, command=self._fit).pack(
            side="left", padx=(8, 0)
        )
        ttk.Label(toolbar, text="Renderer").pack(side="left", padx=(12, 3))
        self._renderer_choice = tk.StringVar(
            value=_RENDERER_LABELS[self._viewer_backend]
        )
        self._renderer_selector = ttk.Combobox(
            toolbar,
            textvariable=self._renderer_choice,
            values=tuple(_RENDERER_BACKENDS),
            width=10,
            state="readonly",
        )
        self._renderer_selector.pack(side="left")
        self._renderer_selector.bind(
            "<<ComboboxSelected>>", self._on_renderer_selected, add="+"
        )
        self._renderer_active_label = ttk.Label(toolbar, text="")
        self._renderer_active_label.pack(side="left", padx=(3, 0))
        self._show_attributes = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            toolbar,
            text="Attributes / imperfections",
            variable=self._show_attributes,
            command=self.refresh_views,
        ).pack(side="left", padx=(12, 0))
        ttk.Separator(toolbar, orient="vertical").pack(
            side="left", fill="y", padx=8
        )
        for label, page in (
            ("Model", "Geometry"), ("Define", "Definitions"), ("Mesh", "Mesh"),
            ("Assign", "Sections"), ("Load/BC", "Loads & BC"),
            ("Run", "Solve"), ("Inspect", "Results"),
            ("Script", "Scripting"),
            ("Visual", "Visualization"),
        ):
            ttk.Button(
                toolbar,
                text=label,
                command=lambda value=page: self.details.select(value),
            ).pack(side="left", padx=1)
        ttk.Button(
            toolbar, text="Commands...", command=self.show_command_palette
        ).pack(side="left", padx=(8, 0))
        self._view_label = ttk.Label(toolbar, text="")
        self._view_label.pack(side="right")

        self.selection_strip = SelectionStrip(self, self)
        self.selection_strip.pack(fill="x")

        panes = ttk.PanedWindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True)

        self.tree = ModelTree(
            panes,
            self.project,
            self.selection,
            job_is_stale=self._job_is_stale,
            mesh_is_stale=lambda record: self.mesh_record_state(record) == "stale",
        )
        self.tree.set_action_handler(self._tree_action)
        panes.add(self.tree, weight=1)

        centre = ttk.Frame(panes)
        panes.add(centre, weight=4)
        self.viewport = Viewport(
            centre,
            selection=self.selection,
            backend=self._viewer_backend,
            viewer_host=self._viewer_host,
        )
        self.viewport.pack(fill="both", expand=True)
        self.viewport.set_pick_handler(self._on_pick)
        self.viewport.bind_event("<Control-a>", self._select_all)
        self.viewport.bind_event("<Control-A>", self._select_all)
        self.viewport.bind_event(
            "<KeyPress-f>", lambda _event: self.viewport.frame_selection()
        )
        self.viewport.bind_event(
            "<KeyPress-F>", lambda _event: self.viewport.frame_selection()
        )
        self.viewport.bind_event("<Delete>", self._delete_selection)
        self.viewport.bind_event("<Escape>", lambda _event: self.selection.clear())
        self._update_renderer_label()

        right = ttk.Frame(panes)
        panes.add(right, weight=2)
        self.details = DetailsWorkspace(right)
        self.details.pack(fill="both", expand=True)
        # Backward-compatible attribute for integrations that selected a
        # notebook page.  The object now implements the same ``select`` call
        # while using a persistent Details task workspace.
        self.notebook = self.details

        self.panels = {}
        for panel_class in (
            GeometryPanel, DefinitionsPanel, MeshPanel, SectionPanel, LoadPanel, SolvePanel,
            ResultsPanel, ScriptingPanel, VisualizationPanel,
        ):
            panel = panel_class(self.details._content, self)
            self.details.add(panel, text=panel_class.title)
            self.panels[panel_class.title] = panel
        self.details.set_select_handler(self._on_details_page_selected)

        self.job_status = JobStatusView(centre, self)
        self.job_status.pack(fill="x")

        status = ttk.Frame(self, padding=(6, 3))
        status.pack(fill="x")
        self._status = ttk.Label(status, text="ready", anchor="w")
        self._status.pack(side="left", fill="x", expand=True)
        self._selection_label = ttk.Label(status, text="")
        self._selection_label.pack(side="right")
        if self.viewport.backend_diagnostics:
            self.set_status(
                "renderer: Tk fallback; "
                + "; ".join(self.viewport.backend_diagnostics)
            )


    def _update_renderer_label(self) -> None:
        active = "GPU" if self.viewport.active_backend == "gpu" else "Tk"
        self._renderer_active_label.configure(text=f"({active})")


    def _on_renderer_selected(self, _event: tk.Event | None = None) -> None:
        previous = self.viewport.requested_backend
        requested = _RENDERER_BACKENDS[self._renderer_choice.get()]
        try:
            self.switch_viewer_backend(requested)
        except Exception as error:
            self._renderer_choice.set(_RENDERER_LABELS[previous])
            diagnostics = tuple(getattr(error, "diagnostics", ()))
            detail = "; ".join(str(item) for item in diagnostics if item)
            message = str(error) + (f"\n\n{detail}" if detail else "")
            self.set_status(f"renderer switch failed: {str(error)}", error=True)
            self.dialogs.show_error("Renderer unavailable", message)

    # ------------------------------------------------------------------
    # commands
    # ------------------------------------------------------------------


    def _bind_root_shortcut(self, sequence: str, callback) -> None:
        """Bind one application shortcut and retain its teardown token."""

        identifier = self.winfo_toplevel().bind(sequence, callback, add="+")
        if identifier:
            self._root_bindings.append((sequence, identifier))


    # ------------------------------------------------------------------
    # meshing and solving
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # views
    # ------------------------------------------------------------------


    def _on_details_page_selected(self, page: str) -> None:
        """Keep task navigation and the viewport in the same workflow context."""

        if page == "Geometry":
            self.show_geometry()
            self.selection_strip.set_context(
                self._geometry_selection_mode,
                "Model geometry • select Point, Line or Plate",
            )
            self.details.set_hint("Model geometry")
            self.set_status("model geometry shown; Point/Line/Plate selection is active")
            return
        if page == "Loads & BC":
            self.show_geometry()
            kind = (
                self.selection.mode
                if self.selection.mode in ("vertex", "edge", "face")
                else "edge"
            )
            self.selection_strip.set_context(
                kind,
                "Geometry scope • choose Point, Line or Plate",
            )
            self.details.set_hint("Scope on model geometry")
            self.set_status(
                "model geometry shown; select points, lines or plates for loads and BCs"
            )
            return
        if page == "Sections":
            self.show_geometry()
            self.details.set_hint("Assign on model geometry")
            return
        if page == "Visualization":
            self.details.set_hint("Viewport and result appearance")
            self.panels["Visualization"].sync_from_viewport()
            self.panels["Visualization"].refresh()
            return
        if page == "Mesh" and self._inspection_mesh is not None:
            self.show_inspection_mesh()
            self.details.set_hint("Inspection mesh — solver admission blocked")
        elif page in ("Mesh", "Solve") and self.mesh is not None:
            self.show_mesh()
            self.details.set_hint("Mesh view")


    def _update_view_label(self) -> None:
        text = f"showing: {self._view_mode}"
        if self._view_mode == "results":
            shape = self.current_shape()
            if shape is not None and getattr(shape, "label", ""):
                text += f" - {shape.label}"
        self._view_label.configure(text=text)

    # ------------------------------------------------------------------
    # refresh
    # ------------------------------------------------------------------
    def refresh_all(self) -> None:
        if self._closing or self._refresh_suspended:
            return
        self._geometry_scene_cache = None
        self.tree.refresh()
        self.refresh_panels()
        self._refresh_toolbar()
        if self._view_mode == "geometry":
            self.show_geometry()

    def refresh_panels(self) -> None:
        if self._closing:
            return
        for panel in self.panels.values():
            panel.refresh()
        self._refresh_toolbar()

    def _refresh_toolbar(self) -> None:
        self._undo_button.configure(
            state="normal" if self.commands.can_undo else "disabled",
            text=f"Undo {self.commands.undo_label or ''}".strip(),
        )
        self._redo_button.configure(
            state="normal" if self.commands.can_redo else "disabled",
            text=f"Redo {self.commands.redo_label or ''}".strip(),
        )

    def _on_selection_changed(self) -> None:
        if self._closing:
            return
        if getattr(self.selection.domain, "value", "geometry") == "geometry":
            self._geometry_selection_mode = self.selection.mode
        self._selection_label.configure(
            text=f"{mode_label(self.selection.mode)} mode - "
            f"{self.selection.describe()}"
        )
        self.refresh_panels()


    # ------------------------------------------------------------------
    # workspace commands, recovery and document ownership
    # ------------------------------------------------------------------


    def show_command_palette(self, _event=None):
        commands = {
            "New project": lambda: self.new_project(confirm=True),
            "Open project": lambda: self.open_project(confirm=True),
            "Save project": self.save_project,
            "Undo": self.undo,
            "Redo": self.redo,
            "Model geometry": lambda: self.details.select("Geometry"),
            "Regions, coordinates and units": lambda: self.details.select("Definitions"),
            "Mesh controls": lambda: self.details.select("Mesh"),
            "Materials and sections": lambda: self.details.select("Sections"),
            "Loads and boundary conditions": lambda: self.details.select("Loads & BC"),
            "Analysis and jobs": lambda: self.details.select("Solve"),
            "Results explorer": lambda: self.details.select("Results"),
            "Python scripting console": lambda: self.details.select("Scripting"),
            "Visualization settings": lambda: self.details.select("Visualization"),
            "Frame model": self._fit,
            "Isometric view": lambda: self.viewport.set_view("iso"),
            "Top view": lambda: self.viewport.set_view("top"),
        }
        dialog = tk.Toplevel(self)
        dialog.title("ANYfem commands")
        dialog.transient(self.winfo_toplevel())
        dialog.geometry("520x360")
        query = tk.StringVar(value="")
        entry = ttk.Entry(dialog, textvariable=query)
        entry.pack(fill="x", padx=10, pady=10)
        listing = tk.Listbox(dialog, activestyle="dotbox")
        listing.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        def refresh(*_args) -> None:
            needle = query.get().strip().casefold()
            listing.delete(0, "end")
            for label in commands:
                if needle in label.casefold():
                    listing.insert("end", label)
            if listing.size():
                listing.selection_set(0)

        def run_selected(_event=None) -> None:
            selected = listing.curselection()
            if not selected:
                return
            label = listing.get(selected[0])
            dialog.destroy()
            self.guarded(commands[label])()

        query.trace_add("write", refresh)
        listing.bind("<Double-1>", run_selected)
        listing.bind("<Return>", run_selected)
        entry.bind("<Return>", run_selected)
        dialog.bind("<Escape>", lambda _e: dialog.destroy())
        refresh()
        entry.focus_set()
        return "break"


    def request_close(self) -> None:
        if not self._confirm_discard():
            return
        root = self.winfo_toplevel()
        self.destroy()
        try:
            root.destroy()
        except tk.TclError:
            pass


    def _refresh_recent_menu(self) -> None:
        menu = getattr(self, "_recent_menu", None)
        if menu is None:
            return
        menu.delete(0, "end")
        if not self._recent_paths:
            menu.add_command(label="(none)", state="disabled")
            return
        for path in self._recent_paths:
            menu.add_command(
                label=path,
                command=self.guarded(
                    lambda value=path: self.open_project(value, confirm=True)
                ),
            )


    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # files
    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        root = self.winfo_toplevel()
        menu = tk.Menu(root)
        files = tk.Menu(menu, tearoff=0)
        files.add_command(
            label="New",
            command=self.guarded(lambda: self.new_project(confirm=True)),
        )
        files.add_command(
            label="Open...",
            command=self.guarded(lambda: self.open_project(confirm=True)),
        )
        self._recent_menu = tk.Menu(files, tearoff=0)
        files.add_cascade(label="Recent projects", menu=self._recent_menu)
        self._refresh_recent_menu()
        files.add_command(
            label="Recover autosave...",
            command=self.guarded(self.recover_autosave),
        )
        files.add_command(
            label="Open file inspector...",
            command=self.guarded(self.open_file_inspector),
        )
        files.add_separator()
        files.add_command(label="Save", command=self.guarded(self.save_project))
        files.add_command(
            label="Save As...", command=self.guarded(lambda: self.save_project(True))
        )
        files.add_separator()
        files.add_command(
            label="Import SESAM...", command=self.guarded(self.import_sesam_model)
        )
        files.add_command(
            label="Export CalculiX deck...", command=self.guarded(self.export_deck)
        )
        files.add_separator()
        files.add_command(
            label="Import CalculiX results...",
            command=self.guarded(self.import_calculix_result),
        )
        files.add_command(
            label="Import SESAM results...",
            command=self.guarded(self.import_sesam_result),
        )
        menu.add_cascade(label="File", menu=files)
        edit = tk.Menu(menu, tearoff=0)
        edit.add_command(label="Undo", accelerator="Ctrl+Z", command=self.undo)
        edit.add_command(label="Redo", accelerator="Ctrl+Y", command=self.redo)
        edit.add_separator()
        edit.add_command(
            label="Command palette...",
            accelerator="Ctrl+P",
            command=self.show_command_palette,
        )
        edit.add_command(
            label="Python scripting console...",
            command=lambda: self.details.select("Scripting"),
        )
        menu.add_cascade(label="Edit", menu=edit)
        view = tk.Menu(menu, tearoff=0)
        view.add_command(
            label="Visualization settings...",
            command=lambda: self.details.select("Visualization"),
        )
        view.add_separator()
        for label, name in (
            ("Isometric", "iso"), ("Top", "top"),
            ("Front", "front"), ("Side", "side"),
        ):
            view.add_command(
                label=label, command=lambda value=name: self.viewport.set_view(value)
            )
        view.add_command(label="Frame all", accelerator="F", command=self._fit)
        menu.add_cascade(label="View", menu=view)
        try:
            root.configure(menu=menu)
        except tk.TclError:  # pragma: no cover - embedded without a toplevel
            pass
        self._menu = menu


    def open_file_inspector(self, path: Optional[str] = None) -> None:
        """Open ANYfileio's inspector as a child of this application."""

        from anyfileio.gui import open_inspector

        open_inspector(self.winfo_toplevel(), path=path)


    def _update_window_title(self) -> None:
        root = self.winfo_toplevel()
        name = self.path.name if self.path is not None else self.project.name
        dirty = " *" if getattr(self, "session", None) and self.session.dirty else ""
        try:
            root.title(f"{name}{dirty} - ANYfem")
        except tk.TclError:  # pragma: no cover - embedded frame
            pass


    # ------------------------------------------------------------------
    def set_status(
        self,
        text: str,
        error: bool = False,
        *,
        diagnostic: Any = None,
    ) -> None:
        self._status_revision=getattr(self,'_status_revision',0)+1
        self._status.configure(
            text=text, foreground="#b00020" if error else "#222222"
        )
        if error:
            self._error_diagnostics.append(
                ErrorDiagnostic.capture(
                    text,
                    details=diagnostic,
                    project_id=str(getattr(self.project, "document_id", "")),
                    view=str(getattr(self, "_view_mode", "")),
                )
            )
            del self._error_diagnostics[:-10]


    def destroy(self) -> None:
        """Unhook everything before the widgets go.

        Tk destroys children in its own order, and a Treeview fires
        ``<<TreeviewSelect>>`` on the way out.  Without unhooking, that
        callback would drive a refresh into panels that no longer exist.
        """

        self.flush_project_writes()
        self._closing = True
        try:
            for attribute in ("_autosave_after", "_autosave_hard_after"):
                identifier = getattr(self, attribute, None)
                if identifier is not None:
                    try:
                        self.scheduler.cancel_call(identifier)
                    except tk.TclError:
                        pass
                    setattr(self, attribute, None)
            if getattr(self, "_job_poll", None) is not None:
                try:
                    self.scheduler.cancel_call(self._job_poll)
                except tk.TclError:
                    pass
                self._job_poll = None
            self.close_result_access()
            self.worker.stop()
            self.workbench.close()
            self.selection.remove_listener(self._on_selection_changed)
            self.selection.remove_listener(self.tree.sync_from_selection)
            self.selection.remove_listener(self.viewport._apply_highlight)
            self.commands.remove_listener(self.refresh_all)
            self.session.remove_listener(self._on_revision_changed)
            root = self.winfo_toplevel()
            for sequence, identifier in self._root_bindings:
                try:
                    root.unbind(sequence, identifier)
                except tk.TclError:
                    pass
            self._root_bindings.clear()
            if self._project_lock is not None:
                self._project_lock.release()
                self._project_lock = None
            self._recovery_pending = None
            self._recovery_executor.shutdown(wait=False, cancel_futures=True)
            self._artifact_executor.shutdown(wait=False, cancel_futures=True)
        finally:
            super().destroy()


def default_project() -> Project:
    """A new project with usable plate and beam defaults.

    The Sections panel exposes ``plate`` and ``stiffener`` as its initial
    definition names.  Both must exist in the document baseline: recorded GUI
    assignments may legitimately refer to either name before the user edits
    its properties, and replay must not depend on an unrecorded UI mutation.
    """

    return _default_project()


def main() -> None:  # pragma: no cover - entry point
    root = tk.Tk()
    root.title("ANYfem")
    root.geometry("1500x900")
    AnyFemApp(root)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()
