from collections.abc import Mapping
import json
import os
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from .controller import WorkbenchController, default_project as _default_project
from .controller import default_project
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

from ..presentation.scene import (Polyline, Scene, build_attribute_overlay, build_collision_overlay, build_geometry_scene, build_mesh_scene, build_persisted_result_scene, build_result_scene, geometry_display_resolution)
from .worker import JobWorkerFacade

ANALYSES = {
    "Linear static": solve_linear_static,
    "Batch linear static": solve_linear_static_many,
    "Modal": solve_modal,
    "Buckling": solve_buckling,
    "Nonlinear static": solve_nonlinear_static,
    "Arc length": solve_arc_length,
    "Transient": solve_transient,
    "Impact": solve_impact,
    "Capacity": solve_capacity,
}

class _WorkbenchField:
    """Data descriptor keeping legacy ``app.field`` access controller-owned."""

    def __init__(self, field: str) -> None:
        self.field = field

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        workbench = instance.__dict__.get("workbench")
        if workbench is None:
            raise AttributeError(self.field)
        return getattr(workbench, self.field)

    def __set__(self, instance, value) -> None:
        workbench = instance.__dict__.get("workbench")
        if workbench is None:
            raise AttributeError(
                f"workbench is not initialized; cannot set {self.field!r}"
            )
        if self.field == "result_datasets":
            previous=getattr(workbench,"result_datasets",{})
            from queue import SimpleQueue
            retired=instance.__dict__.setdefault("_retired_result_datasets",SimpleQueue())
            instance.__dict__["_result_dataset_snapshot"]=tuple(value.values())
            for dataset in previous.values():
                if not any(dataset is current for current in value.values()):retired.put(dataset)
        if self.field in {"solution", "project", "mesh", "result_datasets", "active_model_hash", "shape_index", "active_job_id"}:
            invalidate = getattr(instance, "invalidate_result_requests", None)
            if invalidate is not None:
                if self.field != "result_datasets":invalidate(source=self.field in {"solution","project","mesh","active_model_hash"},reads=self.field=="active_job_id")
        workbench.update(self.field, value)


class WorkbenchWorkflow:
    """Shared workflow with presentation hooks supplied by the desktop adapter.

    Hooks format views and dialogs; document and engineering operations stay here.
    """
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


    def initialize_workflow(self):
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
        self._autosave_after = None
        self._autosave_hard_after = None
        self._recovery_future: Future | None = None
        self._recovery_future_epoch = 0
        self._recovery_pending: tuple[int, dict[str, Any], dict[str, Any]] | None = None
        self._recovery_epoch = 0
        self._recent_paths = self._load_recent_paths()
        self._error_diagnostics: list[ErrorDiagnostic] = []
        self._active_loading_hash = _loading_hash(self.project)


    @property
    def requested_viewer_backend(self) -> str:
        return self.viewport.requested_backend


    @property
    def active_viewer_backend(self) -> str:
        return self.viewport.active_backend


    @property
    def viewer_backend_diagnostics(self) -> tuple[str, ...]:
        return self.viewport.backend_diagnostics




    def run(self, command: Command) -> Any:
        """Run a command; the stack notifies everything that must refresh."""

        return self.workbench.execute(command)


    def run_many(self, commands: Iterable[Command]) -> list[Any]:
        """Run independent commands with one final expensive GUI refresh.

        In the desktop session the commands form one atomic undo item.  A
        lightweight legacy harness without a DocumentSession retains the old
        individual-stack behavior for API compatibility.
        """

        pending = list(commands)
        if not pending:
            return []
        if getattr(self, "workbench", None) is not None:
            return self.workbench.execute_many(pending, label="batch edit")
        self._refresh_suspended += 1
        try:
            return [self.commands.run(command) for command in pending]
        finally:
            self._refresh_suspended -= 1
            if self._refresh_suspended == 0:
                self.refresh_all()


    def undo(self) -> None:
        if self.session.undo():
            self.set_status("undone")


    def redo(self) -> None:
        if self.session.redo():
            self.set_status("redone")


    def _undo_shortcut(self, _event: object | None = None) -> str:
        self.guarded(self.undo)()
        return "break"


    def _redo_shortcut(self, _event: object | None = None) -> str:
        self.guarded(self.redo)()
        return "break"


    def _job_is_stale(self, job: object) -> bool:
        """Compare a retained job's immutable inputs with the active inputs."""

        if getattr(job, "model_hash", "") != self.session.revision.model_hash:
            return True

        mesh_hash = ""
        mesh_record = self.project.mesh_records.get(
            getattr(self, "mesh_record_id", "")
        )
        if mesh_record is not None:
            mesh_hash = mesh_record.mesh_hash
        submitted_mesh_hash = str(getattr(job, "mesh_hash", ""))
        if submitted_mesh_hash and mesh_hash != submitted_mesh_hash:
            return True

        definition = self.project.analyses.get(
            str(getattr(job, "analysis_id", ""))
        )
        if definition is not None:
            # Hashing needs a full serialized project snapshot (tens of
            # milliseconds), and the job table asks on every poll tick.  The
            # answer can only change with the document, so key it on the
            # committed revision and drop every entry when that moves.
            revision = self.session.revision
            revision_key = (revision.id, revision.document_hash)
            cache = getattr(self, "_job_stale_cache", None)
            if cache is None or cache[0] != revision_key:
                cache = (revision_key, {})
                self._job_stale_cache = cache
            submitted = str(getattr(job, "analysis_hash", ""))
            key = (str(getattr(job, "analysis_id", "")), submitted)
            stale = cache[1].get(key)
            if stale is None:
                stale = analysis_hash(
                    definition,
                    getattr(self.project, "output_requests", None),
                    document=self.session.snapshot().document,
                ) != submitted
                cache[1][key] = stale
            return stale
        return False


    def _on_revision_changed(self, revision) -> None:
        """Invalidate derived data and expose dirty/stale state immediately."""
        model_changed = revision.model_hash != self._active_model_hash
        self._active_model_hash = revision.model_hash
        loading_hash = _loading_hash(self.project)
        loading_changed = loading_hash != getattr(self, "_active_loading_hash", loading_hash)
        self._active_loading_hash = loading_hash
        if model_changed or loading_changed:
            self.invalidate_result_requests(source=True)
            # Mesh-native documents have no editable generating geometry.
            # Attribute/region changes invalidate solves, not their source mesh.
            if model_changed:
                self.mesh = self.imported.mesh if self.imported is not None else None
                self._inspection_mesh = None
            self.solution = None
            self.shape_index = 0
            # Derived views belong to the previous immutable revision.  Move
            # back to editable geometry immediately instead of leaving a
            # result/mesh label around with no corresponding data.
            if hasattr(self, "viewport"):
                if self.imported is not None:self.show_mesh()
                else:
                    self._view_mode = "geometry"
                    self.show_geometry()
        if hasattr(self, "job_status"):
            self.job_status.refresh()
        if hasattr(self, "tree"):
            # CommandStack notifies before DocumentSession publishes its new
            # revision. Update only retained-job badges against the committed
            # hash; rebuilding the whole large tree here would duplicate work.
            self.tree.refresh_job_states()
            self.tree.refresh_mesh_states()
        self._update_window_title()
        if self.session.dirty:
            self._schedule_autosave()


    def _invalidate_downstream(self) -> None:
        """A geometry change makes any existing mesh and result stale."""

        self.mesh = None
        self._inspection_mesh = None
        self.solution = None
        self.show_geometry()


    @staticmethod
    def _triangulation_backend_summary(mesh) -> dict[str, dict[str, Any]]:
        diagnostics = getattr(mesh, "hybrid_diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            return {}
        by_face = diagnostics.get("triangulation_backend_by_face", {})
        if not isinstance(by_face, Mapping):
            return {}
        return {
            str(face_id): dict(values)
            for face_id, values in sorted(by_face.items(), key=lambda item: int(item[0]))
            if isinstance(values, Mapping)
        }


    @staticmethod
    def _meshing_strategy_summary(mesh) -> dict[str, str]:
        diagnostics = getattr(mesh, "hybrid_diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            return {}
        by_face = diagnostics.get("strategy_by_face", {})
        if not isinstance(by_face, Mapping):
            return {}
        return {
            str(face_id): str(strategy)
            for face_id, strategy in sorted(
                by_face.items(), key=lambda item: int(item[0])
            )
        }


    @staticmethod
    def _mesh_quality_optimization_summary(mesh) -> dict[str, dict[str, Any]]:
        """Retain ANYmesher per-face optimization provenance."""

        diagnostics = getattr(mesh, "hybrid_diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            return {}
        direct = diagnostics.get("quality_optimization_by_face", {})
        if isinstance(direct, Mapping) and direct:
            return {
                str(face_id): dict(values)
                for face_id, values in sorted(
                    direct.items(), key=lambda item: int(item[0])
                )
                if isinstance(values, Mapping)
            }
        by_face = diagnostics.get("triangulation_backend_by_face", {})
        if not isinstance(by_face, Mapping):
            return {}
        return {
            str(face_id): dict(values["quality_optimization"])
            for face_id, values in sorted(
                by_face.items(), key=lambda item: int(item[0])
            )
            if isinstance(values, Mapping)
            and isinstance(values.get("quality_optimization"), Mapping)
        }


    @staticmethod
    def _complex_geometry_summary(mesh) -> dict[str, Any]:
        """Retain ANYmesher's component strategy and alignment provenance."""

        diagnostics = getattr(mesh, "hybrid_diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            return {}
        report = diagnostics.get("complex_geometry", {})
        result = dict(report) if isinstance(report, Mapping) else {}
        alignment_rejection = diagnostics.get("alignment_candidate_rejected")
        if isinstance(alignment_rejection, Mapping):
            result["alignment_candidate_rejected"] = dict(alignment_rejection)
        return result


    @staticmethod
    def _structured_layout_summary(mesh) -> dict[str, Any]:
        """Retain the reproducible ANYmesher structure-first plan/report."""

        diagnostics = getattr(mesh, "hybrid_diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            return {}
        report = diagnostics.get("structured_layout", {})
        if isinstance(report, Mapping) and report:
            return dict(report)
        status = diagnostics.get("structured_layout_status")
        plan_hash = diagnostics.get("structured_plan_hash")
        quality = diagnostics.get("structured_quality")
        if status is None and plan_hash is None and not isinstance(quality, Mapping):
            return {}
        current: dict[str, Any] = {
            "status": status,
            "plan_hash": plan_hash,
        }
        if isinstance(quality, Mapping):
            current["quality"] = dict(quality)
        return current


    @staticmethod
    def _project_mesh_strategy(project: Project, requested: str | None) -> str:
        """Resolve the public hybrid strategy while preserving legacy auto."""

        from anymesher.hybrid import MeshingStrategy

        if requested is None:
            settings = project.native_mesh_settings
            backend = (
                "automatic"
                if settings is None
                else str(getattr(settings.backend, "value", settings.backend))
            )
            requested = {
                "automatic": "auto",
                "auto": "auto",
                "mapped": "mapped",
                "native": "native",
                "quad_first": "quad_first",
            }.get(backend, backend)
        if str(requested).strip().lower() == "quad_first":
            return "quad_first"
        try:
            return MeshingStrategy(str(requested).strip().lower()).value
        except ValueError as error:
            choices = ", ".join(item.value for item in MeshingStrategy)
            raise ValueError(
                f"unknown meshing strategy {requested!r}; expected one of {choices}"
            ) from error


    @staticmethod
    def _project_structure_preference(
        project: Project, requested: str | None
    ) -> str:
        from anymesher.structured import StructurePreference

        if requested is None:
            settings = project.native_mesh_settings
            parameters = {} if settings is None else dict(settings.parameters)
            requested = str(parameters.get("structure_preference", "balanced"))
        try:
            return StructurePreference(str(requested).strip().lower()).value
        except ValueError as error:
            choices = ", ".join(item.value for item in StructurePreference)
            raise ValueError(
                f"unknown structure preference {requested!r}; expected one of {choices}"
            ) from error


    def _store_mesh_strategy(
        self,
        strategy: str,
        *,
        target_size: float,
        element_order: str,
        structure_preference: str = "balanced",
        quality_policy: Mapping[str, float] | None = None,
        mesh_controls: MeshControls | None = None,
        structured_controls: StructuredMeshControls | None = None,
        quad_options: object | None = None,
        layout_policy: str = "existing",
    ) -> None:
        """Persist a UI strategy through the existing native-settings schema."""

        from ..native_meshing import NativeMeshSettings

        current = self.project.native_mesh_settings
        parameters = {} if current is None else current.to_dict()["parameters"]
        if strategy == "quad_first":
            from ..quad_first import effective_quad_options

            parameters["quad_options"] = effective_quad_options(quad_options).to_dict()
            parameters["layout_policy"] = layout_policy
        else:
            parameters.pop("quad_options", None)
            parameters.pop("layout_policy", None)
        parameters["structure_preference"] = structure_preference
        if mesh_controls is not None:
            parameters.update(mesh_controls.parameters())
        if structured_controls is not None:
            parameters.update(structured_controls.parameters())
        if quality_policy is not None:
            for key, value in quality_policy.items():
                parameters[f"mesh_quality_{key}"] = float(value)
        settings = NativeMeshSettings.create(
            target_size,
            element_order=element_order,
            backend="automatic" if strategy == "auto" else strategy,
            certification_mode=(
                mesh_controls.certification_mode if mesh_controls is not None
                else ("interactive" if current is None else current.certification_mode)
            ),
            controls=() if current is None else current.controls,
            parameters=parameters,
        )
        self.project.set_native_mesh_settings(settings)


    def generate_mesh(
        self,
        target_size: float,
        *,
        native_backend: str | None = None,
        strategy: str | None = None,
        structure_preference: str | None = None,
        quality_policy: Mapping[str, float] | None = None,
        mesh_controls: MeshControls | None = None,
        structured_controls: StructuredMeshControls | None = None,
        quad_options: object | None = None,
        layout_policy: str | None = None,
        automation: object | None = None,
        execution_policy: object | None = None,
        pool_lease: object | None = None,
    ):
        """Generate a mesh synchronously for scripts and legacy integrations.

        The desktop Mesh task uses :meth:`generate_mesh_async`; keeping this
        method synchronous preserves the established headless/test contract.
        ``execution_policy``/``pool_lease`` are execution-only (see
        :mod:`anyfem.mesh_execution`); a warm pool lease stays in this process
        and is never transported into the Qt workbench.
        """

        resolved_strategy = self._project_mesh_strategy(self.project, strategy)
        if resolved_strategy == "quad_first":
            from ..quad_first import effective_quad_options

            if quad_options is None and self.project.native_mesh_settings is not None:
                quad_options = self.project.native_mesh_settings.to_dict()["parameters"].get("quad_options")
            quad_options = effective_quad_options(quad_options)
            if layout_policy is None:
                layout_policy = (
                    self.project.native_mesh_settings.to_dict()["parameters"].get("layout_policy", "existing")
                    if self.project.native_mesh_settings is not None else "existing"
                )
        mesh_controls = mesh_controls or MeshControls.from_settings(self.project.native_mesh_settings)
        structured_controls = (
            structured_controls
            or StructuredMeshControls.from_settings(
                self.project.native_mesh_settings
            )
        )
        resolved_preference = self._project_structure_preference(
            self.project, structure_preference
        )
        with self.session.transaction("mesh settings"):
            self.project.target_size = float(target_size)
            self.project.seeding_overrides = dict(self.seeding_overrides)
            if native_backend is not None:
                self.project.set_native_triangulation_backend(native_backend)
            if resolved_strategy == "quad_first":
                self.project.set_native_triangulation_backend("python")
            if strategy is not None or structure_preference is not None or quality_policy is not None or mesh_controls is not None:
                self._store_mesh_strategy(
                    resolved_strategy,
                    target_size=float(target_size),
                    element_order=self.project.element_order,
                    structure_preference=resolved_preference,
                    quality_policy=quality_policy,
                    mesh_controls=mesh_controls,
                    structured_controls=structured_controls,
                    quad_options=quad_options,
                    layout_policy=layout_policy or "existing",
                )
        requested_backend = self.project.native_triangulation_backend
        effective_native_backend = (
            None if resolved_strategy == "mapped" else requested_backend
        )
        created_mesh = self.project.generate_mesh(
            target_size,
            overrides=self.seeding_overrides,
            strategy=resolved_strategy,
            structure_preference=resolved_preference,
            quality_policy=quality_policy,
            mesh_controls=mesh_controls,
            structured_controls=structured_controls,
            quad_options=quad_options,
            layout_policy=layout_policy,
            automation=automation,
            execution_policy=execution_policy,
            pool_lease=pool_lease,
        )
        automation_result = created_mesh.hybrid_diagnostics.get("automation", {})
        inspection_only = automation_result.get("status") == "inspection_only"
        from anymesher import verify_mesh_quality

        mesh_input_hash = canonical_hash(
            {
                "target_size": float(target_size),
                "overrides": dict(self.seeding_overrides),
                "element_order": self.project.element_order,
                "strategy": resolved_strategy,
                "quad_options": (
                    None if quad_options is None else quad_options.to_dict()
                ),
                "layout_policy": layout_policy if resolved_strategy == "quad_first" else None,
                "structure_preference": (
                    resolved_preference if resolved_strategy == "auto" else None
                ),
                "native_backend": effective_native_backend,
                "controls": mesh_controls.effective_dict(resolved_strategy),
                "quality_policy": (
                    dict(quality_policy or {}) if resolved_strategy == "auto" else None
                ),
                "structured_controls": (
                    structured_controls.effective_dict()
                    if resolved_strategy == "auto"
                    else None
                ),
                **({"automation": automation.to_dict()} if automation is not None else {}),
            }
        )
        preparation = dict(self.project._last_mesh_preparation)
        mesh_hash = mesh_semantic_hash(
            created_mesh,
            model_hash=self.session.revision.model_hash,
            mesh_input_hash=mesh_input_hash,
            structural_preparation=preparation,
        )
        quality = verify_mesh_quality(created_mesh).as_dict()
        record = MeshRecord(
            name=f"Mesh {len(self.project.mesh_records) + 1}",
            source_model_hash=self.session.revision.model_hash,
            mesh_input_hash=mesh_input_hash,
            mesh_hash=mesh_hash,
            status="inspection_only" if inspection_only else "completed",
            structural_preparation=preparation,
            summary={
                "nodes": created_mesh.num_nodes,
                "elements": created_mesh.num_elements,
                "native_backend_requested": effective_native_backend,
                "strategy_requested": resolved_strategy,
                "quad_options_requested": (
                    None if quad_options is None else quad_options.to_dict()
                ),
                "layout_policy_requested": layout_policy if resolved_strategy == "quad_first" else None,
                "automation_requested": (
                    None if automation is None else automation.to_dict()
                ),
                "strict_high_order_status": created_mesh.hybrid_diagnostics.get(
                    "high_order_geometry", {}
                ).get("status"),
                "controls_requested": mesh_controls.effective_dict(resolved_strategy),
                "structured_controls_requested": (
                    structured_controls.effective_dict()
                    if resolved_strategy == "auto"
                    else None
                ),
                "strategy_by_face": self._meshing_strategy_summary(created_mesh),
                "structured_layout": self._structured_layout_summary(created_mesh),
                "complex_geometry": self._complex_geometry_summary(created_mesh),
                "quality_optimization_by_face": (
                    self._mesh_quality_optimization_summary(created_mesh)
                ),
                "triangulation_backend_by_face": (
                    self._triangulation_backend_summary(created_mesh)
                ),
                "automatic_intersections": int(
                    getattr(created_mesh, "automatic_intersections", 0)
                ),
                "automatic_beam_connections": int(
                    getattr(created_mesh, "automatic_beam_connections", 0)
                ),
                "automatic_shell_connections": int(
                    getattr(created_mesh, "automatic_shell_connections", 0)
                ),
                "quality": quality,
                "automation": dict(automation_result),
                "execution": dict(
                    getattr(self.project, "_last_mesh_execution", {}) or {}
                ),
                "solver_admission": "BLOCKED" if inspection_only else "ADMITTED",
            },
        )
        with self.session.transaction("record mesh", solver_affecting=False):
            self.project.mesh_records[record.id] = record
        self._mesh_details_record_id = record.id
        self._meshes[record.id] = created_mesh
        if inspection_only:
            self._inspection_mesh = created_mesh
            self.show_inspection_mesh()
            self.set_status("Mesh created â€” inspection only; solver admission is blocked")
        else:
            self.mesh = created_mesh
            self._inspection_mesh = None
            self.solution = None
            self.mesh_record_id = record.id
            self.set_status(
                f"Mesh created â€” ready to solve: {created_mesh.num_nodes} nodes, "
                f"{created_mesh.num_elements} elements; "
                f"{getattr(created_mesh, 'automatic_intersections', 0)} plate "
                "intersection(s) imprinted; "
                f"{getattr(created_mesh, 'automatic_beam_connections', 0)} beam "
                "connection(s) created; "
                f"{getattr(created_mesh, 'automatic_shell_connections', 0)} shell "
                "T-junction tie(s) created; "
                f"max aspect {quality['max_aspect_ratio']:.3g}, "
                f"warp {quality['max_warp']:.3g}"
            )
            self.show_mesh()
        self.refresh_panels()
        return created_mesh


    def generate_mesh_async(
        self,
        target_size: float,
        *,
        native_backend: str | None = None,
        strategy: str | None = None,
        structure_preference: str | None = None,
        quality_policy: Mapping[str, float] | None = None,
        mesh_controls: MeshControls | None = None,
        structured_controls: StructuredMeshControls | None = None,
        quad_options: object | None = None,
        layout_policy: str | None = None,
        automation: object | None = None,
        execution_policy: object | None = None,
    ) -> MeshRecord:
        """Submit meshing from an immutable snapshot and return immediately.

        ``execution_policy`` is execution-only and crosses to the worker as a
        plain picklable value; warm pool leases are deliberately not accepted
        here (they stay local to a script's own process).
        """

        if self.mesh_task_manager.busy:
            raise ValueError("a mesh is already being generated")
        resolved_strategy = self._project_mesh_strategy(self.project, strategy)
        if resolved_strategy == "quad_first":
            from ..quad_first import effective_quad_options

            if quad_options is None and self.project.native_mesh_settings is not None:
                quad_options = self.project.native_mesh_settings.to_dict()["parameters"].get("quad_options")
            quad_options = effective_quad_options(quad_options)
            if layout_policy is None:
                layout_policy = (
                    self.project.native_mesh_settings.to_dict()["parameters"].get("layout_policy", "existing")
                    if self.project.native_mesh_settings is not None else "existing"
                )
        mesh_controls = mesh_controls or MeshControls.from_settings(self.project.native_mesh_settings)
        structured_controls = (
            structured_controls
            or StructuredMeshControls.from_settings(
                self.project.native_mesh_settings
            )
        )
        resolved_preference = self._project_structure_preference(
            self.project, structure_preference
        )
        settings = MeshSettings.create(
            target_size,
            element_order=self.project.element_order,
            overrides=self.seeding_overrides,
            strategy=resolved_strategy,
            structure_preference=resolved_preference,
            quality_policy=quality_policy,
            controls=mesh_controls,
            structured_controls=structured_controls,
            quad_options=quad_options,
            layout_policy=layout_policy or "existing",
            automation=automation,
            execution_policy=execution_policy,
        )
        with self.session.transaction("mesh settings"):
            self.project.target_size = settings.target_size
            self.project.seeding_overrides = dict(settings.overrides)
            if native_backend is not None:
                self.project.set_native_triangulation_backend(native_backend)
            if resolved_strategy == "quad_first":
                self.project.set_native_triangulation_backend("python")
            if strategy is not None or structure_preference is not None or quality_policy is not None or mesh_controls is not None:
                self._store_mesh_strategy(
                    resolved_strategy,
                    target_size=settings.target_size,
                    element_order=settings.element_order,
                    structure_preference=resolved_preference,
                    quality_policy=quality_policy,
                    mesh_controls=mesh_controls,
                    structured_controls=structured_controls,
                    quad_options=quad_options,
                    layout_policy=layout_policy or "existing",
                )
        requested_backend = self.project.native_triangulation_backend
        effective_native_backend = (
            None if settings.strategy == "mapped" else requested_backend
        )
        snapshot = self.session.snapshot()
        record = MeshRecord(
            name=f"Mesh {len(self.project.mesh_records) + 1}",
            source_model_hash=snapshot.revision.model_hash,
            mesh_input_hash=canonical_hash(
                {
                    "mesh_settings": settings.input_hash,
                    "native_backend": effective_native_backend,
                }
            ),
            mesh_hash="",
            status="running",
            summary={
                "target_size": settings.target_size,
                "element_order": settings.element_order,
                "native_backend_requested": effective_native_backend,
                "strategy_requested": settings.strategy,
                "quad_options_requested": (
                    None if settings.quad_options is None else settings.quad_options.to_dict()
                ),
                "layout_policy_requested": settings.layout_policy if settings.strategy == "quad_first" else None,
                "automation_requested": (
                    None if settings.automation is None else settings.automation.to_dict()
                ),
                "controls_requested": mesh_controls.effective_dict(settings.strategy),
                "structured_controls_requested": (
                    structured_controls.effective_dict()
                    if settings.strategy == "auto"
                    else None
                ),
                "structure_preference": (
                    settings.structure_preference
                    if settings.strategy == "auto"
                    else None
                ),
                "execution": {
                    "requested": (
                        None
                        if settings.execution_policy is None
                        else settings.execution_policy.describe()
                    )
                },
                "status": "running",
            },
        )
        with self.session.transaction("queue mesh", solver_affecting=False):
            self.project.mesh_records[record.id] = record
        try:
            self.mesh_task_manager.submit(record.id, snapshot, settings)
        except BaseException:
            with self.session.transaction("remove failed mesh submission", solver_affecting=False):
                self.project.mesh_records.pop(record.id, None)
            raise
        self._active_mesh_task_id = record.id
        self._mesh_details_record_id = record.id
        self.set_status("meshing in background from immutable model snapshot")
        self.refresh_panels()
        return record


    @property
    def mesh_job_running(self) -> bool:
        return self.mesh_task_manager.busy


    def cancel_mesh(self) -> bool:
        """Request cancellation at the next safe meshing phase."""

        job_id = self.mesh_task_manager.active_job_id
        if job_id is None or not self.mesh_task_manager.cancel(job_id):
            return False
        record = self.project.mesh_records.get(job_id)
        if record is not None:
            with self.session.transaction("cancel mesh", solver_affecting=False):
                record.status = "cancelling"
                record.summary["status"] = "cancelling"
        self.set_status("cancelling mesh; waiting for the current safe phase")
        self.refresh_panels()
        return True


    def mesh_record_state(self, record: MeshRecord) -> str:
        """Return persisted state with revision-based staleness applied."""

        if record.status in ("completed", "inspection_only", "stale") and (
            record.source_model_hash
            and record.source_model_hash != self.session.revision.model_hash
        ):
            return "stale"
        return str(record.status)


    def _poll_mesh_jobs(self) -> None:
        for event in self.mesh_task_manager.poll():
            record = self.project.mesh_records.get(event.job_id)
            if record is None:
                continue
            self._observe_run_event("mesh", event)
            if event.kind in ("started", "progress"):
                self.set_status(event.message)
                continue
            if event.kind == "cancelling":
                self.set_status(event.message)
                continue

            if event.kind == "completed":
                result = event.payload
                if not isinstance(result, MeshJobResult):
                    continue
                current = record.source_model_hash == self.session.revision.model_hash
                with self.session.transaction(
                    "record completed mesh", solver_affecting=False
                ):
                    record.mesh_hash = result.mesh_hash
                    record.structural_preparation = dict(
                        result.structural_preparation
                    )
                    automation_result = result.mesh.hybrid_diagnostics.get("automation", {})
                    inspection_only = (
                        isinstance(automation_result, Mapping)
                        and automation_result.get("status") == "inspection_only"
                    )
                    record.status = (
                        "stale" if not current else
                        "inspection_only" if inspection_only else "completed"
                    )
                    record.summary.update(
                        {
                            "status": record.status,
                            "nodes": result.mesh.num_nodes,
                            "elements": result.mesh.num_elements,
                            "strict_high_order_status": result.mesh.hybrid_diagnostics.get(
                                "high_order_geometry", {}
                            ).get("status"),
                            "strategy_by_face": self._meshing_strategy_summary(
                                result.mesh
                            ),
                            "structured_layout": self._structured_layout_summary(
                                result.mesh
                            ),
                            "complex_geometry": self._complex_geometry_summary(
                                result.mesh
                            ),
                            "quality_optimization_by_face": (
                                self._mesh_quality_optimization_summary(
                                    result.mesh
                                )
                            ),
                            "triangulation_backend_by_face": (
                                self._triangulation_backend_summary(result.mesh)
                            ),
                            "automatic_intersections": int(
                                getattr(result.mesh, "automatic_intersections", 0)
                            ),
                            "automatic_beam_connections": int(
                                getattr(result.mesh, "automatic_beam_connections", 0)
                            ),
                            "automatic_shell_connections": int(
                                getattr(result.mesh, "automatic_shell_connections", 0)
                            ),
                            "quality": dict(result.quality),
                            "automation": dict(automation_result),
                            "execution": dict(result.execution),
                            "mesh_validity": "VALID",
                            "solver_admission": (
                                "BLOCKED" if inspection_only else "ADMITTED"
                            ),
                        }
                    )
                self._meshes[record.id] = result.mesh
                if current:
                    self._mesh_details_record_id = record.id
                    if inspection_only:
                        self._inspection_mesh = result.mesh
                        self.show_inspection_mesh()
                        self.set_status(
                            "Mesh created â€” inspection only; solver admission is blocked"
                        )
                    else:
                        self._inspection_mesh = None
                        self.mesh = result.mesh
                        self.mesh_record_id = record.id
                        self.solution = None
                        self.show_mesh()
                        quality = result.quality
                        used_method = automation_result.get("selected_method")
                        method_note = (
                            f"; method used: {used_method}"
                            if used_method is not None else ""
                        )
                        self.set_status(
                            f"Mesh created â€” ready to solve: {result.mesh.num_nodes} nodes, "
                            f"{result.mesh.num_elements} elements{method_note}; "
                            f"{getattr(result.mesh, 'automatic_intersections', 0)} "
                            "plate intersection(s) imprinted; "
                            f"{getattr(result.mesh, 'automatic_beam_connections', 0)} "
                            "beam connection(s) created; "
                            f"{getattr(result.mesh, 'automatic_shell_connections', 0)} "
                            "shell T-junction tie(s) created; "
                            f"max aspect {float(quality['max_aspect_ratio']):.3g}, "
                            f"warp {float(quality['max_warp']):.3g}"
                        )
                else:
                    self.set_status(
                        "mesh completed for an older model revision and was retained as stale"
                    )
            elif event.kind == "cancelled":
                with self.session.transaction(
                    "record cancelled mesh", solver_affecting=False
                ):
                    record.status = "cancelled"
                    record.summary["status"] = "cancelled"
                self.set_status("mesh generation cancelled")
            elif event.kind == "failed":
                incomplete = (
                    isinstance(event.payload, Mapping)
                    and event.payload.get("type") == "MeshRecoveryIncomplete"
                )
                with self.session.transaction("record failed mesh", solver_affecting=False):
                    record.status = "incomplete" if incomplete else "failed"
                    record.summary["status"] = record.status
                    if event.payload:
                        record.diagnostics.append(event.payload)
                self.set_status(
                    f"mesh generation {record.status}: {event.message}",
                    error=True,
                    diagnostic=event.payload,
                )
            self._active_mesh_task_id = None
            self.refresh_all()


    def confirm_untrusted_code(self, items) -> bool:
        """Ask the user to approve pressure code that came from a file.

        ``items`` are ``(load case, pressure name, code)``. A frontend that can
        show the code overrides this; the default refuses, so headless use
        never runs code it was not told to trust.
        """

        return False

    def _require_trusted_pressure_code(self) -> None:
        pending = self.project.untrusted_code()
        if not pending:
            return
        if not self.confirm_untrusted_code(pending):
            names = ", ".join(f"{case}: {name}" for case, name, _code in pending)
            raise ValueError(
                "this project contains pressure code that has not been "
                f"approved on this machine ({names}). Review it and approve it "
                "(anyloads.trust_code), or remove it, before solving."
            )
        from anyloads import trust_code

        for _case, _name, code in pending:
            trust_code(code)

    def solve(
        self, analysis: str = "Linear static", *,
        output_request_ids: Iterable[str] = (), **options: Any,
    ) -> None:
        """Queue an analysis against an immutable document/mesh snapshot."""

        if self.mesh is None:
            if self._inspection_mesh is not None:
                raise ValueError("inspection-only mesh cannot be solved; generate an admitted mesh first")
            raise ValueError("generate a mesh first")
        if getattr(self.mesh, "hybrid_diagnostics", {}).get("automation", {}).get("status") == "inspection_only":
            raise ValueError("inspection-only mesh cannot be solved")
        try:
            function = ANALYSES[analysis]
        except KeyError:
            raise ValueError(f"unknown analysis {analysis!r}") from None
        self._require_trusted_pressure_code()

        self.analysis = analysis
        if isinstance(output_request_ids, str):
            raise ValueError("output request IDs must be a sequence, not a string")
        submitted_options = dict(options)
        target_kind = "none" if analysis == "Modal" else "load_case"
        target_id = str(options.get("load_case", "default"))
        if options.get("combination") is not None:
            target_kind = "combination"
            target_id = str(options["combination"])
        if analysis == "Batch linear static":
            # A batch has no single case; it solves every case it is given.
            target_kind = "load_cases"
            target_id = "batch"
        definition = AnalysisDefinition(
            name=f"{analysis} {len(self.project.analyses) + 1}",
            type=analysis.lower().replace(" ", "_"),
            target_kind=target_kind,
            target_id=target_id,
            settings=_record_settings(options),
            output_request_ids=tuple(output_request_ids),
        )
        for request_id in definition.output_request_ids:
            request = self.project.output_requests.get(request_id)
            if request is None:
                raise ValueError(f"output request {request_id!r} is unavailable")
            problems = request.problems_for_analysis(definition.type)
            if problems:
                raise ValueError("; ".join(problems))
        from .output_requests import freeze_output_requests
        output_scopes = freeze_output_requests(
            self.project, definition.output_request_ids, self.mesh,
            mesh_id=str(getattr(self, "mesh_record_id", "")),
        )
        job_options = dict(options)
        if self.imported is not None:
            # An imported model is already built and has no geometry to mesh,
            # so it goes straight to the analysis.
            # Public codecs own immutable geometry/mesh state (including
            # mapping proxies); generic deepcopy is not a valid job boundary.
            submitted_project=self.session.snapshot().thaw()
            from ..io.sesam import import_sesam_bytes
            _,source_contents=self.imported.artifact_embedding()
            submitted_import=import_sesam_bytes(source_contents,source_name=self.imported.source.name if self.imported.source is not None else "embedded.FEM")
            case=submitted_project.load_cases.get(str(options.get("load_case",self.active_case())))
            if options.get("combination") is not None:
                from ..solve.build import _build_combination
                case=_build_combination(submitted_project,submitted_import.mesh,submitted_import.fe_model,str(options["combination"]))
            options = {
                key: value
                for key, value in options.items()
                if key not in ("load_case", "combination")
            }
            built = submitted_import.built(case, project=submitted_project)
            job_options = {
                key: value
                for key, value in job_options.items()
                if key not in ("load_case", "combination")
            }
            job_options["built"] = built
        else:
            # Use ANYmesher's public versioned codec as the immutable job
            # boundary.  Mesh provenance can contain ANYgeometry read-only
            # mapping views, which deliberately cannot be deep-copied.
            job_options["mesh"] = clone_mesh_for_job(self.mesh)

        # Do not leave an analysis definition behind when preparation of the
        # immutable job input fails (for example, an invalid mesh artifact).
        with self.session.transaction("create analysis", solver_affecting=False):
            self.project.analyses[definition.id] = definition

        mesh_hash = ""
        mesh_record = self.project.mesh_records.get(
            getattr(self, "mesh_record_id", "")
        )
        if mesh_record is not None:
            mesh_hash = mesh_record.mesh_hash
        record = self.job_manager.submit(
            definition,
            self.session.snapshot(),
            _execute_analysis_job,
            mesh_hash=mesh_hash,
            kwargs={
                "solver_function": function,
                "analysis_name": analysis,
                "options": job_options,
                "requested_stress": any(
                    key.startswith("stress")
                    for scope in output_scopes for key in scope["request"]["quantity_keys"]
                ),
                "requested_global_stress": any(
                    scope["request"]["basis"] == "global" and scope["request"]["recovery"] != "patch" and
                    any(key.startswith("stress") for key in scope["request"]["quantity_keys"])
                    for scope in output_scopes
                ),
                "requested_patch_stress": any(
                    scope["request"]["recovery"] == "patch" and
                    any(key.startswith("stress") for key in scope["request"]["quantity_keys"])
                    for scope in output_scopes
                ),
            },
            name=definition.name,
        )
        self.active_job_id = record.id
        input_report = _submitted_input_report(
            self.project,
            definition,
            submitted_options,
            self.mesh,
            revision=self.session.revision.sequence,
            model_hash=self.session.revision.model_hash,
            mesh_hash=mesh_hash,
            output_scopes=output_scopes,
        )
        self.submitted_input_reports[record.id] = input_report
        solve_panel = self.panels.get("Solve")
        if solve_panel is not None:
            solve_panel.begin_job(
                definition.name,
                record.id,
                input_report,
            )
        self.set_status(f"queued {definition.name}")
        self.refresh_panels()
        self.job_status.refresh()


    def cancel_solve(self) -> None:
        self.worker.cancel()


    def _on_solved(
        self,
        solution,
        job_id: str | None = None,
        completion: str = "completed",
    ) -> None:
        if job_id is not None:
            job=self.project.jobs.get(job_id)
            if job is None:
                return  # Completion belongs to a replaced/removed document.
            self.solutions[job_id] = solution
            if self._job_is_stale(job):
                self.set_status(f"retained stale result for {job.name}; active model unchanged")
                self.panels["Solve"].append_progress(f"{completion}: retained stale result")
                self.refresh_panels()
                return
            self.active_job_id = job_id
        self.solution = solution
        shapes = getattr(solution, "shapes", None)
        # Nonlinear snapshots are a path to the submitted result; open on the
        # last converged state while retaining every real increment for the
        # navigator and playback controls.
        self.shape_index = (
            len(shapes) - 1 if shapes and hasattr(solution, "steps") else 0
        )
        self.set_status(solution.summary())
        panel = self.panels["Solve"]
        panel.append_progress(completion)
        panel.append_report(_solution_report(solution))
        panel.show_progress("")
        self.notebook.select(self.panels["Results"])
        self.refresh_panels()
        self.show_results()


    def _poll_jobs(self) -> None:
        if self._closing:
            return
        for event in self.job_manager.poll():
            if event.job_id in self.project.jobs:
                self._observe_run_event("solve", event)
            if event.kind in ("queued", "started", "progress", "status"):
                self._on_progress(event.message, event.payload)
            elif event.kind == "completed":
                try:
                    self._on_solved(event.payload, event.job_id)
                except Exception as error:
                    # A display/export adapter must never turn a successfully
                    # completed numerical job into an unhandled Tk callback.
                    self.set_status(
                        f"job completed, but opening Results failed: {error}",
                        error=True,
                    )
                    panel = self.panels.get("Solve")
                    if panel is not None:
                        panel.append_progress(f"Results display failed: {error}")
                    try:
                        self.refresh_panels()
                    except Exception as refresh_error:
                        if panel is not None:
                            panel.append_progress(
                                f"Results controls refresh failed: {refresh_error}"
                            )
                if self.path is not None:
                    self._schedule_result_artifact(event.job_id, self.path)
                    self._schedule_job_log_artifact(event.job_id, self.path)
                if self.session.dirty:
                    self._write_recovery()
            elif event.kind == "partial":
                try:
                    self._on_solved(
                        event.payload,
                        event.job_id,
                        completion=f"partial: {event.message}",
                    )
                except Exception as error:
                    self.set_status(
                        f"partial job retained, but opening Results failed: {error}",
                        error=True,
                    )
                if self.path is not None:
                    self._schedule_result_artifact(event.job_id, self.path)
                    self._schedule_job_log_artifact(event.job_id, self.path)
                if self.session.dirty:
                    self._write_recovery()
            elif event.kind == "cancelled":
                self.set_status(f"job {event.job_id[:8]} cancelled")
                if self.path is not None:
                    self._schedule_job_log_artifact(event.job_id, self.path)
                self.refresh_panels()
            elif event.kind == "failed":
                job_record = self.project.jobs.get(event.job_id)
                diagnostic = None
                if job_record is not None and job_record.diagnostics:
                    diagnostic = job_record.diagnostics[-1]
                elif isinstance(event.payload, BaseException):
                    diagnostic = {
                        "type": type(event.payload).__name__,
                        "message": str(event.payload),
                        "traceback": "".join(
                            traceback.format_exception(event.payload)
                        ),
                    }
                self.set_status(
                    f"job {event.job_id[:8]} failed: {event.message}",
                    error=True,
                    diagnostic=diagnostic,
                )
                panel = self.panels.get("Solve")
                if panel is not None:
                    panel.append_progress(f"failed: {event.message}")
                if self.path is not None:
                    self._schedule_job_log_artifact(event.job_id, self.path)
                self.refresh_panels()
        if hasattr(self, "job_status"):
            self.job_status.refresh()
        self._poll_result_artifacts()
        self._poll_job_log_artifacts()
        self._poll_mesh_jobs()
        self._poll_recovery_write()
        self._job_poll = self.scheduler.call_later(
            self.worker.POLL_MS, self._poll_jobs
        )


    def _result_mesh_id(self, job) -> str:
        """Identify the submitted mesh even after the active model changes."""
        return next((record.id for record in reversed(tuple(self.project.mesh_records.values()))
                     if record.mesh_hash==job.mesh_hash),"active-mesh")

    def retained_result_mesh(self, job_id):
        """Resolve the submitted mesh without replacing the editable active mesh."""
        job=self.project.jobs.get(job_id)
        solution=self.solutions.get(job_id)
        built=getattr(solution,"built",None)
        if built is not None:return built.mesh
        dataset=self.result_datasets.get(job_id)
        identity=dataset.identity if dataset is not None else {}
        mesh_id=identity.get("mesh_id") or (self._result_mesh_id(job) if job is not None else "")
        if mesh_id in self._meshes:return self._meshes[mesh_id]
        record=self.project.mesh_records.get(mesh_id)
        if record is not None and self.path is not None:
            artifact=self.project.artifacts.get(record.artifact_id or "")
            if artifact is not None:
                from ..io.artifacts import ArtifactStore
                mesh=ArtifactStore(self.path).read_mesh(artifact)
                self._meshes[mesh_id]=mesh;return mesh
        # Legacy sidecars may predate explicit mesh IDs. Hash matching still
        # prevents rendering old fields on an unrelated current remesh.
        current=self.project.mesh_records.get(self.mesh_record_id or "")
        if self.mesh is not None and (job is None or current is not None and current.mesh_hash==job.mesh_hash):return self.mesh
        raise ValueError("The submitted result mesh is unavailable; restore its mesh artifact")

    def _result_artifact_provenance(self, job_id: str) -> dict[str, Any]:
        submitted = self.submitted_input_reports.get(job_id)
        if not submitted:
            return {}
        try:
            return {"submitted_inputs": json.loads(submitted)}
        except json.JSONDecodeError:
            return {"submitted_inputs_text": submitted}

    def _schedule_result_artifact(self, job_id: str, destination: Path) -> None:
        """Write a completed result off the Tk event thread."""

        if self.session.read_only or job_id in self._artifact_futures:
            return
        solution = self.solutions.get(job_id)
        record = self.project.jobs.get(job_id)
        if solution is None or record is None:
            return
        mesh_id = self._result_mesh_id(record)
        from ..io.artifacts import ArtifactStore

        store = ArtifactStore(destination)
        future = self._artifact_executor.submit(
            write_solution_artifact,
            store,
            solution,
            job_id=record.id,
            document_id=self.project.document_id,
            mesh_id=mesh_id,
            model_hash=record.model_hash,
            mesh_hash=record.mesh_hash,
            analysis_hash=record.analysis_hash,
            provenance=self._result_artifact_provenance(job_id),
            summary=dict(record.summary),
            diagnostics=tuple(record.diagnostics),
            partial=bool(record.partial),
        )
        self._artifact_futures[job_id] = future
        self._artifact_destinations[job_id] = Path(destination)


    def _poll_result_artifacts(self) -> None:
        for job_id, future in tuple(self._artifact_futures.items()):
            if not future.done():
                continue
            self._artifact_futures.pop(job_id, None)
            destination = self._artifact_destinations.pop(job_id, None)
            try:
                artifact = future.result()
            except BaseException as error:  # persisted as a job diagnostic
                record = self.project.jobs.get(job_id)
                if record is not None:
                    record.diagnostics.append(
                        {"type": type(error).__name__, "message": str(error)}
                    )
                self.set_status(
                    f"result artifact for {job_id[:8]} failed: {error}", error=True
                )
                continue
            if job_id not in self.project.jobs or destination is None:
                continue
            self._record_result_artifact(job_id, artifact, destination)
            if self.session.dirty:
                self._write_recovery()


    def _record_result_artifact(self, job_id: str, artifact, destination: Path) -> None:
        from ..io.artifacts import ArtifactStore

        with self.session.transaction(
            "record result artifact", solver_affecting=False
        ):
            self.project.jobs[job_id].result_artifact_id = artifact.id
            self.project.artifacts[artifact.id] = artifact
        try:
            dataset = ArtifactStore(destination).open_result(
                artifact,defer=True
            )
            self.result_datasets=dict(self.result_datasets,**{job_id:dataset})
        except (OSError, ValueError):
            pass


    def _schedule_job_log_artifact(self, job_id: str, destination: Path) -> None:
        """Persist one terminal numerical-job log outside the Tk thread."""

        if self.session.read_only or job_id in self._log_futures:
            return
        record = self.project.jobs.get(job_id)
        if record is None:
            return
        try:
            entries = self.job_manager.log(job_id)
        except KeyError:
            return
        from ..io.artifacts import ArtifactStore

        store = ArtifactStore(destination)
        self._log_futures[job_id] = self._artifact_executor.submit(
            store.write_log, job_id, entries
        )
        self._log_destinations[job_id] = Path(destination)


    def _poll_job_log_artifacts(self) -> None:
        for job_id, future in tuple(self._log_futures.items()):
            if not future.done():
                continue
            self._log_futures.pop(job_id, None)
            self._log_destinations.pop(job_id, None)
            try:
                artifact = future.result()
            except BaseException as error:
                record = self.project.jobs.get(job_id)
                if record is not None:
                    with self.session.transaction(
                        "record job log failure", solver_affecting=False
                    ):
                        record.diagnostics.append(
                            {"type": type(error).__name__, "message": str(error)}
                        )
                self.set_status(
                    f"job log for {job_id[:8]} failed: {error}", error=True
                )
                continue
            if job_id not in self.project.jobs:
                continue
            self._record_job_log_artifact(job_id, artifact)
            if self.session.dirty:
                self._write_recovery()


    def _record_job_log_artifact(self, job_id: str, artifact) -> None:
        with self.session.transaction(
            "record job log artifact", solver_affecting=False
        ):
            self.project.jobs[job_id].log_artifact_id = artifact.id
            self.project.artifacts[artifact.id] = artifact


    def current_shape(self):
        """The displacement field currently being displayed.

        A static result is its own shape; a modal, buckling or transient
        result is browsed by index.  Everything downstream sees the same
        interface either way.
        """

        solution = self.solution
        if solution is None:
            return None
        shapes = getattr(solution, "shapes", None)
        if not shapes:
            return solution
        index = min(max(self.shape_index, 0), len(shapes) - 1)
        return shapes[index]


    def show_geometry(self, reset_view: bool = False) -> None:
        self._mesh_layout_preview = None
        self._view_mode = "geometry"
        resolution = geometry_display_resolution(
            self.project.geometry,
            self.viewport.visualization.geometry_detail,
        )
        entity_owners = self.tree.generated_entity_owners()
        exposed_features = self.tree.exploded_feature_ids
        cache_key = (
            id(self.project.geometry),
            int(self.project.geometry.revision),
            resolution,
            bool(self.viewport.visualization.show_beam_sections),
            tuple(sorted(exposed_features)),
        )
        cached = self._geometry_scene_cache
        if cached is None or cached[0] != cache_key:
            scene = build_geometry_scene(
                self.project,
                divisions=resolution.divisions,
                curve_samples=resolution.curve_samples,
                show_beam_sections=self.viewport.visualization.show_beam_sections,
                entity_owners=entity_owners,
                exposed_feature_ids=exposed_features,
            )
            interaction_scene = build_geometry_scene(
                self.project,
                divisions=resolution.interaction_divisions,
                curve_samples=resolution.interaction_curve_samples,
                show_beam_sections=self.viewport.visualization.show_beam_sections,
                entity_owners=entity_owners,
                exposed_feature_ids=exposed_features,
            )
            self._geometry_scene_cache = (cache_key, scene, interaction_scene)
        else:
            _key, scene, interaction_scene = cached
        self.viewport.show(
            self._with_attributes(scene.copy()),
            interaction_scene=self._with_attributes(interaction_scene.copy()),
            reset_view=reset_view,
        )
        self._update_view_label()


    def preview_mesh_layout(
        self,
        target_size: float,
        *,
        structure_preference: str = "balanced",
        quality_policy: Mapping[str, float] | None = None,
        structured_controls: StructuredMeshControls | None = None,
    ) -> Any:
        """Plan and display exact mesh-only partitions without model mutation."""

        from anymesher import apply_structured_layout, plan_structured_layout

        structured_controls = (
            structured_controls
            or StructuredMeshControls.from_settings(
                self.project.native_mesh_settings
            )
        )
        plan = plan_structured_layout(
            self.project.geometry,
            target_size=float(target_size),
            options=structured_controls.owner_options(
                preference=structure_preference,
                quality_policy=dict(quality_policy or {}),
            ),
            overrides=self.seeding_overrides,
            explicit_seeding=bool(self.seeding_overrides),
        )
        working, report = apply_structured_layout(self.project.geometry, plan)
        overlay = Scene()
        highlighted_edges = {
            int(oriented.edge)
            for item in plan.faces
            if item.structured
            for oriented in self.project.geometry.faces[item.source_face_id].loop
        }
        highlighted_edges.update(
            int(edge_id)
            for edge_id in working.edges
            if edge_id not in self.project.geometry.edges
        )
        parameters = np.linspace(0.0, 1.0, 17)
        for edge_id in sorted(highlighted_edges):
            owner = working if edge_id in working.edges else self.project.geometry
            if edge_id not in owner.edges:
                continue
            overlay.lines.append(
                Polyline(
                    None,
                    np.asarray(owner.sample_edge(edge_id, parameters), dtype=float),
                    color="#00a6d6",
                    width=3,
                    draw_overlay=True,
                )
            )
        scene = build_geometry_scene(
            self.project,
            exposed_feature_ids=self.tree.exploded_feature_ids,
        )
        scene.merge(overlay)
        self._mesh_layout_preview = (plan, report)
        self._view_mode = "mesh-layout-preview"
        self.viewport.show(self._with_attributes(scene))
        self._update_view_label()
        self.set_status(
            f"mesh layout preview: {len(plan.blocks)} block/residual region(s), "
            f"{len(plan.interfaces)} shared interface(s), "
            f"estimated {plan.estimated_element_count} element(s); no model change"
        )
        return plan


    def commit_mesh_layout_preview(self) -> Any:
        """Commit the currently displayed whole plan as one undoable feature."""

        if self._mesh_layout_preview is None:
            raise ValueError("preview an Automatic mesh layout first")
        from ..commands import CommitStructuredLayout

        plan, _preview = self._mesh_layout_preview
        if not plan.is_current(self.project.geometry):
            self._mesh_layout_preview = None
            raise ValueError("mesh layout preview is stale; preview it again")
        report = self.run(CommitStructuredLayout(plan))
        self._mesh_layout_preview = None
        self.show_geometry()
        self.set_status(
            "committed the complete structured mesh partition plan as one "
            "checksummed frozen feature; undo restores the exact prior design"
        )
        return report


    def clear_mesh_layout_preview(self) -> None:
        self._mesh_layout_preview = None
        self.show_geometry()


    def show_mesh(self) -> None:
        if self.mesh is None:
            if self._inspection_mesh is not None:self.show_inspection_mesh()
            else:self.show_geometry()
            return
        self._view_mode = "mesh"
        scene = build_mesh_scene(
            self.project,
            self.mesh,
            show_beam_sections=self.viewport.visualization.show_beam_sections,
        )
        self.viewport.show(self._with_attributes(scene))
        self._update_view_label()


    def show_inspection_mesh(self) -> None:
        """Display a valid candidate without making it available to solving."""

        if self._inspection_mesh is None:
            return
        self._view_mode = "inspection_mesh"
        scene = build_mesh_scene(
            self.project, self._inspection_mesh,
            show_beam_sections=self.viewport.visualization.show_beam_sections,
        )
        details = self._inspection_mesh.hybrid_diagnostics.get("automation", {})
        problem_ids = {
            int(element_id)
            for attempt in details.get("attempts", ())
            for element_id in attempt.get("problem_element_ids", ())
        }
        for face in scene.faces:
            if face.polygon_owners is None or face.colors is None:
                continue
            face.colors = [
                "#e67459" if any(
                    getattr(owner, "kind", None) == "element"
                    and getattr(owner, "id", None) in problem_ids
                    for owner in owners
                ) else color
                for color, owners in zip(face.colors, face.polygon_owners)
            ]
        self.viewport.show(self._with_attributes(scene))
        self._update_view_label()


    def prepare_result_dataset(self,dataset,*,on_ready=None,on_cancel=None):
        if getattr(dataset,"metadata_ready",True):return False
        self.set_status("Preparing result metadata...")
        def prepared(value):
            self.refresh_panels()
            if on_ready is not None:on_ready()
        self._request_result(lambda cancel:dataset.prepare_metadata(),prepared,dataset=dataset,on_cancel=on_cancel)
        return True

    def show_persisted_result(
        self,
        field_key: str,
        *,
        frame: int = 0,
        component: str | None = None,
        scale: float = 1.0,
        limits=None,
        on_ready=None,
        on_cancel=None,
    ) -> None:
        if self.active_job_id is None:
            raise ValueError("this persisted result has no available mesh")
        dataset = self.result_datasets.get(self.active_job_id)
        if dataset is None:
            raise ValueError("the result artifact is unavailable")
        if self.prepare_result_dataset(dataset,on_ready=lambda:self.show_persisted_result(
                field_key,frame=frame,scale=scale,component=component,limits=limits,on_ready=on_ready,on_cancel=on_cancel),on_cancel=on_cancel):return
        job=self.project.jobs.get(self.active_job_id)
        stale=job is not None and self._job_is_stale(job)
        panel = self.panels["Results"]
        cancel_pending=getattr(panel,"cancel_pending_display",None)
        if callable(cancel_pending):cancel_pending()
        options = dict(colormap=panel.colormap(), display_units=panel.display_units(),
                       show_nodes=panel.show_result_nodes(),
                       show_beam_sections=self.viewport.visualization.show_beam_sections and not stale)
        geometry_key=(getattr(self,"_result_revision",0),id(dataset),self.active_job_id,frame if scale else None,
                      scale,repr(self.viewport.visualization),self._show_attributes.get(),stale,
                      options["show_nodes"],panel.show_result_supports(),panel.show_result_loads(),
                      panel.show_result_masses(),panel.show_imperfect_reference())
        warm=(geometry_key==getattr(self,"_persisted_geometry_key",None) and
              getattr(self.viewport,'active_backend',None)=='gpu' and
              getattr(self.viewport,'_retained_result',None) is not None)
        def load(cancel):
            if warm:
                from ..ui.scene import read_persisted_display_field
                field,descriptor=read_persisted_display_field(dataset,field_key,frame=frame,component=component)
                return field,descriptor
            result_mesh=self.retained_result_mesh(self.active_job_id)
            scene = build_persisted_result_scene(
                self.project,
                result_mesh,
                dataset,
                field_key,
                frame=frame,
                component=component,
                scale=scale,
                limits=limits,
                colormap=options["colormap"],
                display_units=options["display_units"],
                show_nodes=options["show_nodes"],
                show_beam_sections=options["show_beam_sections"],
            )
            return scene
        def apply(scene):
            panel = self.panels["Results"]
            access=getattr(self,"_live_result_access",None)
            field=scene[0] if warm else scene.result_field
            scalar_bytes=0
            retained=getattr(self.viewport,"_retained_result",None)
            if warm and retained:scalar_bytes=sum(handle.mesh.element_scalars.nbytes for handle in retained[1])
            from ..presentation.retained_results import prepared_publication_bytes
            reservation=access.reserve_publication(field,prepared_publication_bytes(field,getattr(self.viewport,"_scene",None) if warm else scene,scalar_bytes)) if access is not None else None
            if warm:
                from ..ui.scene import persisted_display_metadata
                field,descriptor=scene
                range_,palette,legend=persisted_display_metadata(field,descriptor,component,limits=limits,display_units=options["display_units"],colormap=options["colormap"])
                if self.viewport.update_result_field(field,scalar_range=range_,colormap=palette,legend=legend):
                    if reservation is not None:access.commit_publication(reservation)
                    self._result_geometry_key=None
                    self._warn_persisted_status(dataset,stale)
                    self._update_view_label()
                    if on_ready is not None:on_ready()
                    return
                self._persisted_geometry_key=None
                if access is not None:access.discard_publication()
                self.show_persisted_result(field_key,frame=frame,component=component,scale=scale,limits=limits,on_ready=on_ready,on_cancel=on_cancel)
                return
            result_mesh=scene.result_mesh
            scene = scene if stale else self._with_attributes(
                scene,
                mesh=result_mesh,
                show_supports=panel.show_result_supports(),
                show_loads=panel.show_result_loads(),
                show_masses=panel.show_result_masses(),
                show_imperfections=panel.show_imperfect_reference(),
            )
            self._view_mode = "results"
            if access is not None and getattr(self.viewport,"active_backend",None)=="gpu":
                from ..presentation.retained_results import prepare
                prepared=prepare(self.viewport,scene)
                if prepared is not None:
                    access.discard_publication()
                    reservation=access.reserve_publication(field,prepared_publication_bytes(field,scene,sum(mesh.element_scalars.nbytes for mesh,options in prepared[1])))
                    scene._prepared_result_batches=prepared
            self.viewport.show(scene)
            from ..presentation.retained_results import discard_prepared_field
            discard_prepared_field(field)
            if reservation is not None:access.commit_publication(reservation)
            self._result_geometry_key=None
            self._persisted_geometry_key=geometry_key
            if on_ready is not None:on_ready()
            self._update_view_label()
            if stale:self.set_status("Stale retained result on its submitted mesh; current sections and model attributes are omitted")
            self._warn_persisted_status(dataset,stale)
        def publish(scene):
            try:return apply(scene)
            except BaseException:
                access=getattr(self,"_live_result_access",None)
                if access is not None:access.discard_publication()
                raise
        return self._request_result(load, publish,dataset=dataset,on_cancel=on_cancel,display_options=dict(limits=limits,display_units=options["display_units"],colormap=options["colormap"]))

    def _set_result_warning(self,text=None,*,error=False):
        previous=getattr(self,'_result_warning_stamp',None)
        current=WorkbenchWorkflow._result_status_stamp(self)
        if text:
            self.set_status(text,error=error);self._result_warning_text=text
            self._result_warning_stamp=WorkbenchWorkflow._result_status_stamp(self)
        elif previous and current==previous:
            self.set_status('Results ready');self._result_warning_text=None;self._result_warning_stamp=None

    def _result_status_stamp(self):
        return (getattr(self,'_status_revision',0),WorkbenchWorkflow._result_status_text(self))

    def _result_export_notice(self,message,request_status,*,error=False):
        outcomes=self.__dict__.setdefault('_result_export_outcomes',[])
        outcomes.append(message);del outcomes[:-10]
        if WorkbenchWorkflow._result_status_stamp(self)==request_status:
            self.set_status(message,error=error)
        else:
            log=getattr(self,'log',None)
            if hasattr(log,'appendPlainText'):log.appendPlainText(message)
            elif hasattr(log,'insert'):log.insert('end',message+'\n')
            else:
                solve=getattr(self,'panels',{}).get('Solve')
                if hasattr(solve,'append_report'):solve.append_report(message)
            if error:
                from ..diagnostics import ErrorDiagnostic
                diagnostics=self.__dict__.setdefault('_result_export_diagnostics',[])
                diagnostics.append(ErrorDiagnostic.capture(message));del diagnostics[:-10]

    def _result_status_text(self):
        status=getattr(self,'_status',None)
        if hasattr(status,'get'):return status.get()
        if hasattr(status,'cget'):return status.cget('text')
        return None

    def _warn_persisted_status(self,dataset,stale=False):
        warnings=[]
        if getattr(dataset,"identity",{}).get("partial",False):warnings.append("Partial analysis: explicit stored values; no completed-analysis or portable certification claim")
        if stale:warnings.append("Stale retained result shown on its submitted mesh")
        WorkbenchWorkflow._set_result_warning(self,"; ".join(warnings) if warnings else None,error=bool(warnings))



    def _with_attributes(
        self,
        scene,
        *,
        show_supports: bool = True,
        show_loads: bool = True,
        show_masses: bool = True,
        show_imperfections: bool = True,
        project=None,
        mesh=None,
        case_name=None,
    ):
        """Overlay supports and the active case's loads, if asked for."""

        if not self._show_attributes.get():
            return scene
        return scene.merge(
            build_attribute_overlay(
                self.project if project is None else project,
                case_name=self.active_case() if case_name is None else case_name,
                mesh=self.mesh if mesh is None else mesh,
                show_supports=show_supports,
                show_loads=show_loads,
                show_masses=show_masses,
                show_imperfections=show_imperfections,
            )
        )


    def _result_case_name(self, shape) -> str:
        """The load case whose loads a result shape should display.

        A batch result has one case per shape, so the case selected in the
        Loads panel says nothing about which shape is on screen.
        """

        solved = getattr(getattr(shape.built, "load_case", None), "name", None)
        if solved is not None and str(solved) in self.project.load_cases:
            return str(solved)
        return self.active_case()

    def active_case(self) -> str:
        panel = self.panels.get("Loads & BC")
        return "default" if panel is None else panel.case_name()


    def refresh_views(self) -> None:
        """Redraw whichever view is showing."""

        if self._closing:
            return
        if self._view_mode == "results":
            self.show_results()
        elif self._view_mode == "mesh":
            self.show_mesh()
        else:
            self.show_geometry()


    def invalidate_result_requests(self, *, source=False, reads=False):
        self._result_revision = getattr(self, "_result_revision", 0) + 1
        if source:self._result_source_epoch=getattr(self,'_result_source_epoch',0)+1
        loader=getattr(self,"_result_loader",None)
        if loader is not None:
            loader.invalidate(source=source,reads=reads)

    def close_result_access(self):
        """Owner returns immediately; the SAME worker retires query resources."""
        self.invalidate_result_requests(source=True)
        loader=getattr(self,'_result_loader',None)
        self._result_loader=None
        access=getattr(self,'_live_result_access',None)
        datasets=tuple(getattr(self,'result_datasets',{}).values())+self._take_retired_result_datasets()
        def finalizer():
            current=getattr(self,'_live_result_access',None)
            self._live_result_access=None
            unique=[]
            for resource in (current,access)+datasets:
                if resource is None or any(resource is previous for previous in unique):continue
                unique.append(resource)
                try:resource.close()
                except BaseException as error:
                    import logging
                    logging.getLogger(__name__).exception('Result resource retirement failed')
                    errors=self.__dict__.setdefault('_result_retirement_errors',[])
                    errors.append(f'{type(error).__name__}: {error}')
                    del errors[:-10]
        if loader is not None:
            self._result_completed_exports=loader.outcomes
            self._result_failed_exports=loader.failures
            self._retired_result_loader=loader
            self._retiring_result_access=True
            try:self._result_retirement=loader.retire(finalizer,report_pending=not self._closing)
            finally:self._retiring_result_access=False
        elif getattr(self,'_retired_result_loader',None) is not None:
            self._result_retirement=self._retired_result_loader.retire(finalizer,report_pending=not self._closing)
        elif getattr(self,'_result_retirement',None) is None:
            # No worker means no source recovery can be using these readers.
            finalizer()
        retired=getattr(self,'_retired_result_loader',None)
        if retired is not None:
            pending_future=self._result_retirement
            def report_retired_exports():
                from queue import Empty
                finished=pending_future.done()
                while True:
                    try:callback=retired.retirement_reports.get_nowait()
                    except Empty:break
                    if not self._closing:
                        try:callback()
                        except Exception:
                            import logging
                            logging.getLogger(__name__).exception('Retired export report failed')
                if not self._closing and not finished:self.scheduler.call_later(5,report_retired_exports)
            report_retired_exports()
        from queue import Empty
        queue=getattr(self,'_result_queue',None)
        if queue is not None:
            while True:
                try:callback=queue.get_nowait()
                except Empty:break
                callback()  # already-settled delivery releases its payload only

    def _take_retired_result_datasets(self):
        from queue import Empty
        queue=self.__dict__.get("_retired_result_datasets")
        if queue is None:return ()
        datasets=[]
        while True:
            try:datasets.append(queue.get_nowait())
            except Empty:return tuple(datasets)

    def _request_result(self, load, publish, *, dataset=None, channel="display", ordered=False, on_cancel=None, display_options=None, _retry_id=None, _status_stamp=None, _source_epoch=None):
        request_status=WorkbenchWorkflow._result_status_stamp(self) if _status_stamp is None else _status_stamp
        source_epoch=getattr(self,'_result_source_epoch',0) if _source_epoch is None else _source_epoch
        if getattr(self,'_retiring_result_access',False):
            if on_cancel is not None:on_cancel()
            return None
        retirement=getattr(self,'_result_retirement',None)
        if retirement is not None:
            if not retirement.done():
                pending=self.__dict__.setdefault('_result_retry_tokens',{})
                token=object() if _retry_id is None else _retry_id
                if not ordered and _retry_id is None:pending[channel]=token
                owner=(self.solution,self.project,self.mesh,self.active_job_id)
                def retry():
                    current=(self.solution,self.project,self.mesh,self.active_job_id)
                    valid=(all(a is b for a,b in zip(owner[:3],current[:3])) and
                           (ordered or owner[3]==current[3]) and source_epoch==getattr(self,'_result_source_epoch',0))
                    if self._closing or not valid or (not ordered and pending.get(channel) is not token):
                        if on_cancel is not None:on_cancel()
                        if ordered and channel=='export' and not valid and not self._closing:
                            WorkbenchWorkflow._result_export_notice(self,'Export cancelled before publication',request_status)
                        return
                    self._request_result(load,publish,dataset=dataset,channel=channel,ordered=ordered,on_cancel=on_cancel,display_options=display_options,_retry_id=token,_status_stamp=request_status,_source_epoch=source_epoch)
                if not self._closing:self.scheduler.call_later(5,retry)
                elif on_cancel is not None:on_cancel()
                return None
            self._result_retirement=None
            retired_loader=self.__dict__.pop('_retired_result_loader',None)
            if retired_loader is not None:retired_loader.shutdown()
            try:retirement.result()
            except BaseException as error:
                import logging
                logging.getLogger(__name__).exception('Result retirement failed: %s',error)
        from queue import SimpleQueue, Empty
        from .result_requests import ResultRequests
        if getattr(self, "_result_loader", None) is None:
            self._result_queue = SimpleQueue()
            self._result_loader = ResultRequests(self._result_queue.put)
            drain_loader=self._result_loader
            def drain():
                if self._closing or getattr(self,"_result_loader",None) is not drain_loader:
                    return
                try:
                    while True:
                        self._result_queue.get_nowait()()
                except Empty:
                    pass
                finally:
                    if not self._closing:self.scheduler.call_later(5, drain)
            self.scheduler.call_later(0, drain)
        revision = getattr(self, "_result_revision", 0)
        solution, project, mesh = self.solution, self.project, self.mesh
        display_scene=getattr(self.viewport,"_scene",None) if hasattr(self,"viewport") else None
        retained=getattr(self.viewport,"_retained_result",None) if hasattr(self,"viewport") else None
        panel=getattr(self,"panels",{}).get("Results")
        if display_options is None and channel=="display" and retained is not None and display_scene is not None and panel is not None:
            display_options=dict(limits=panel.colour_limits(),display_units=panel.display_units(),colormap=panel.colormap())
        datasets=tuple(getattr(self,"result_datasets",{}).values())
        cleanup_settled=False
        def cancel_capture():
            nonlocal cleanup_settled
            if not cleanup_settled:
                cleanup_settled=True
                if on_cancel is not None:on_cancel()
        def completed(result):
            if result.pending:
                cancel_capture()
                return
            if result.committed:
                if not self._closing:
                    publish(result.value)
                    message=f"Export completed from {result.source}: {result.value}" if result.source else f"Export completed: {result.value}"
                    WorkbenchWorkflow._result_export_notice(self,message,request_status)
                else:
                    cancel_capture()
                    import logging
                    logging.getLogger(__name__).info('Export completed from %s: %s',result.source,result.value)
                return
            if ordered and result.error is not None and not result.cancelled:
                cancel_capture()
                if not self._closing:
                    WorkbenchWorkflow._result_export_notice(self,f'Export failed: {type(result.error).__name__}: {result.error}',request_status,error=True)
                return
            if self._closing or result.cancelled or self.solution is not solution or self.project is not project or self.mesh is not mesh:
                cancel_capture()
                if ordered and channel=='export' and not self._closing:
                    message='Export cancelled before publication'
                    WorkbenchWorkflow._result_export_notice(self,message,request_status)
                return
            if result.error is not None:
                cancel_capture()
                self.set_status(f"Result query failed: {result.error}",error=True)
                return
            try:publish(result.value)
            except Exception as error:
                access=getattr(self,"_live_result_access",None)
                if access is not None:access.discard_publication()
                cancel_capture()
                self.set_status(f"Result publication refused: {error}",error=True)
        def run(cancel):
            for other in datasets:
                if other is not dataset:
                    release=getattr(other,"release_reader",None) or other.close
                    release()
            for retired_dataset in self._take_retired_result_datasets():
                if not any(retired_dataset is current for current in getattr(self,"_result_dataset_snapshot",datasets)):retired_dataset.close()
            from ..post.access import LiveResultAccess,result_access_scope
            access=getattr(self,"_live_result_access",None)
            accepted_bytes=getattr(access,"_publication_bytes",0)
            if dataset is not None and access is not None:
                access.close();access=None
            if access is None or getattr(self,"_live_access_revision",None)!=revision:
                if access is not None:access.close()
                access=self._live_result_access=LiveResultAccess(analysis_status=getattr(solution,"status",None))
                access.commit_publication(accepted_bytes)
                self._live_access_revision=revision
            with result_access_scope(access):
                value=load(cancel)
                field=getattr(value,"field",value[0] if isinstance(value,tuple) and value else value)
                if display_options is not None and retained is not None and display_scene is not None and hasattr(field,"node_values") and not cancel.is_set():
                    from ..presentation.retained_results import prepare_warm_field
                    prepare_warm_field(field,display_scene,retained,working_bytes=access.working_bytes,**display_options)
                return value
        return self._result_loader.request(run, completed,channel=channel,ordered=ordered)

    def _result_scale_text(self,panel):
        widget=getattr(panel,"scale",getattr(panel,"_scale",None))
        if widget is None:return None
        return widget.text() if hasattr(widget,"text") else widget.get()

    def show_results(self, *, on_ready=None, on_cancel=None) -> None:
        if self.solution is None:
            self.invalidate_result_requests()
            self.show_mesh()
            return
        panel = self.panels["Results"]
        cancel_pending=getattr(panel,'cancel_pending_display',None)
        if callable(cancel_pending):cancel_pending()
        panel.ensure_compatible_field()
        shape, solution, name = self.current_shape(), self.solution, panel.field_name()
        checkbox = getattr(panel, "envelope", getattr(panel, "_envelope", None))
        use_envelope = checkbox is not None and (checkbox.isChecked() if hasattr(checkbox, "isChecked") else checkbox.get())
        revision = getattr(self, "_result_revision", 0)
        scale_text=self._result_scale_text(panel)
        def load(cancel):
            from ..post.access import LiveResultAccess
            access = getattr(self, "_live_result_access", None)
            scale_key=(revision,id(shape),scale_text)
            if scale_text is not None and getattr(self,"_prepared_result_scale",(None,))[0] != scale_key:
                from ..presentation.result_display import deformation_scale
                if str(scale_text).strip().lower() in {"","auto"}:
                    try:
                        magnitude=access.field(shape,"magnitude",cancel_event=cancel)
                        from types import SimpleNamespace
                        proxy=SimpleNamespace(built=shape.built,max_translation=magnitude.extreme)
                        value=deformation_scale(scale_text,self.project,proxy)
                    except KeyError:value=0.0
                else:value=deformation_scale(scale_text,self.project,shape)
                self._prepared_result_scale=(scale_key,value)
            if use_envelope:
                from ..post.access import evaluate_envelope
                return evaluate_envelope(solution, name,cancel_event=cancel)
            return access.field(shape, name, cancel_event=cancel)
        def publish(field):
            coverage=field if hasattr(field,"admitted_case_ids") else None
            if coverage is not None:field=coverage.field
            accepted=self._publish_results_field(field)
            if accepted and coverage is not None:
                WorkbenchWorkflow._set_result_warning(self,f"Envelope coverage: {coverage.admitted_count}/{coverage.shape_count} admitted; {coverage.failed_count} failed {coverage.failed_case_ids}",error=bool(coverage.failed_count))
            if accepted is False:
                if self.solution is not None:self.show_results(on_ready=on_ready,on_cancel=on_cancel)
            elif on_ready is not None:on_ready()
        return self._request_result(load,publish,on_cancel=on_cancel)

    def _publish_results_field(self, result_field) -> bool:
        try:return WorkbenchWorkflow._apply_results_field(self,result_field)
        except BaseException:
            access=getattr(self,"_live_result_access",None)
            if access is not None:access.discard_publication()
            raise

    def _apply_results_field(self, result_field) -> bool:
        if self.solution is None:
            self.show_mesh()
            return False
        panel = self.panels["Results"]
        panel.ensure_compatible_field()
        shape = self.current_shape()
        if result_field.name not in {panel.field_name(),panel.field_name()+" envelope"}:
            raise ValueError("returned result field does not match the requested component")
        self._view_mode = "results"
        from ..post.access import source_case_metadata
        status=source_case_metadata(shape,getattr(self.solution,"status",None))["status"]
        if not result_field.name.endswith(" envelope") and status not in {"ok","complete","completed","success","succeeded","converged"}:
            WorkbenchWorkflow._set_result_warning(self,f"Stored source snapshot from {status}; no accepted-state or portable certification claim",error=True)
        elif not result_field.name.endswith(" envelope"):WorkbenchWorkflow._set_result_warning(self,)
        scale_text=self._result_scale_text(panel)
        if scale_text is None:scale=panel.scale_value(shape)
        else:
            key=(getattr(self,"_result_revision",0),id(shape),scale_text)
            prepared=getattr(self,"_prepared_result_scale",(None,None))
            if prepared[0] != key:
                return False
            scale=prepared[1]
        geometry_key = (getattr(self,"_result_revision",0),id(shape),scale,
                        panel.show_result_nodes(),repr(self.viewport.visualization),
                        panel.show_result_supports(),panel.show_result_loads(),
                        panel.show_result_masses(),panel.show_imperfect_reference(),
                        self._show_attributes.get(),self._result_case_name(shape))
        access=getattr(self,"_live_result_access",None)
        reservation=None
        if access is not None:
            retained=getattr(self.viewport,"_retained_result",None)
            scalar_bytes=0
            if retained and geometry_key == getattr(self,"_result_geometry_key",None):
                for handle in retained[1]:
                    mesh=handle.mesh
                    scalar_bytes += mesh.element_scalars.nbytes
            from ..presentation.retained_results import prepared_publication_bytes
            candidate_scene=getattr(self.viewport,"_scene",None)
            reservation=access.reserve_publication(result_field,scalar_bytes if candidate_scene is None else prepared_publication_bytes(result_field,candidate_scene,scalar_bytes))
        if geometry_key == getattr(self,"_result_geometry_key",None):
            from ..ui.scene import result_display_metadata
            range_,palette,legend=result_display_metadata(result_field,limits=panel.colour_limits(),
                                                          display_units=panel.display_units(),colormap=panel.colormap())
            if self.viewport.update_result_field(result_field,scalar_range=range_,colormap=palette,legend=legend):
                if reservation is not None:access.commit_publication(reservation)
                self._update_view_label()
                return True
        from ..presentation.retained_results import discard_prepared_field
        discard_prepared_field(result_field)
        scene = build_result_scene(
            shape,
            field=panel.field_name(),
            scale=scale,
            limits=panel.colour_limits(),
            values=result_field,
            colormap=panel.colormap(),
            display_units=panel.display_units(),
            show_nodes=panel.show_result_nodes(),
            show_beam_sections=self.viewport.visualization.show_beam_sections,
        )
        scene = self._with_attributes(
            scene,
            project=getattr(shape.built,"project",self.project),
            mesh=shape.built.mesh,
            case_name=self._result_case_name(shape),
            show_supports=panel.show_result_supports(),
            show_loads=panel.show_result_loads(),
            show_masses=panel.show_result_masses(),
            show_imperfections=panel.show_imperfect_reference(),
        )
        if getattr(self.solution, "sphere_positions", None) is not None:
            scene.merge(build_collision_overlay(self.solution, self.shape_index))
        if access is not None and getattr(self.viewport,"active_backend",None)=="gpu":
            from ..presentation.retained_results import prepare
            prepared=prepare(self.viewport,scene)
            if prepared is not None:
                scalar_bytes=sum(mesh.element_scalars.nbytes for mesh,options in prepared[1])
                access.discard_publication()
                reservation=access.reserve_publication(result_field,scalar_bytes+getattr(result_field,"_display_buffer_bytes",0))
                scene._prepared_result_batches=prepared
        from ..presentation.retained_results import prepared_publication_bytes,discard_prepared_field
        if access is not None and reservation is None:reservation=access.reserve_publication(result_field,prepared_publication_bytes(result_field,scene,0))
        if access is not None:
            access.discard_publication()
            reservation=access.reserve_publication(result_field,prepared_publication_bytes(result_field,scene,scalar_bytes))
        self.viewport.show(scene)
        discard_prepared_field(result_field)
        self._persisted_geometry_key=None
        self._result_geometry_key = geometry_key
        if reservation is not None:access.commit_publication(reservation)
        self._update_view_label()
        return True


    def _fit(self) -> None:
        self.viewport.fit()


    def _observe_run_event(self, kind: str, event: Any) -> None:
        """Optional UI-thread presentation hook; no worker or project mutation."""

    def _on_progress(self, text: str, payload: Any = None) -> None:
        line = _job_progress_text(text, payload)
        self.set_status(line)
        panel = self.panels.get("Solve")
        if panel is not None:
            panel.show_progress(line)
            panel.append_progress(line, payload=payload)


    def _on_pick(self, ref) -> None:
        if ref is None:
            return
        self.set_status(f"selected {ref}")
        # Clicking something while results are showing reads it out: that is
        # what a probe is for, and asking for it twice would be tedious.
        if self._view_mode == "results" and self.solution is not None:
            panel = self.panels.get("Results")
            if panel is not None:
                panel.guarded(panel._probe)()


    def _selection_universe(self):
        mode = self.selection.mode
        if mode == "vertex":
            return [self.project.geometry.entity_ref("vertex", key) for key in self.project.geometry.vertices]
        if mode == "edge":
            return [self.project.geometry.entity_ref("edge", key) for key in self.project.geometry.edges]
        if mode == "face":
            return [self.project.geometry.entity_ref("face", key) for key in self.project.geometry.faces]
        if self.mesh is None:
            return []
        if mode == "node":
            return [MeshEntityRef("node", key) for key in self.mesh.nodes]
        if mode == "element":
            identifiers = (
                *self.mesh.shells.keys(),
                *self.mesh.beams.keys(),
                *self.mesh.couplings.keys(),
            )
            return [MeshEntityRef("element", key) for key in identifiers]
        if mode == "element_face":
            return [MeshEntityRef("element_face", (key, 0)) for key in self.mesh.shells]
        return []


    def _select_all(self, _event=None):
        self.selection.select_all(self._selection_universe())
        return "break"


    def _delete_selection(self, _event=None):
        from ..commands import DeleteEntity

        geometry = [
            ref for ref in self.selection.items
            if getattr(ref, "kind", "") in ("vertex", "edge", "face")
            and not isinstance(ref, MeshEntityRef)
        ]
        if geometry:
            self.run_many(DeleteEntity(ref) for ref in geometry)
        else:
            self.set_status("mesh topology is immutable; delete its owning geometry or scope")
        return "break"


    def _tree_action(self, action: str, keys: tuple[str, ...]) -> None:
        if not keys:
            return
        if action == "delete":
            self._delete_tree_items(keys)
            return
        if action == "explode":
            if not all(key.startswith("feature:") for key in keys):
                self.set_status(
                    "Explode applies to geometry feature rows only",
                    error=True,
                )
                return
            try:
                exposed = self.tree.toggle_feature_topology(
                    int(key.split(":", 1)[1]) for key in keys
                )
            except ValueError as error:
                self.set_status(str(error), error=True)
                return
            self.show_geometry()
            self.set_status(
                f"{'exposed' if exposed else 'collapsed'} generated topology "
                f"for {len(keys)} feature(s); geometry was not changed"
            )
            return
        key = keys[0]
        prefix = key.split(":", 1)[0]
        page = {
            "feature": "Geometry",
            "material": "Sections",
            "plate_section": "Sections",
            "beam_section": "Sections",
            "imperfection": "Sections",
            "coordinate": "Definitions",
            "region": "Definitions",
            "unit": "Definitions",
            "mesh": "Mesh",
            "case": "Loads & BC",
            "load": "Loads & BC",
            "support": "Loads & BC",
            "mass": "Loads & BC",
            "analysis": "Solve",
            "job": "Solve",
            "result": "Results",
        }.get(prefix, "Geometry")
        if action == "edit":
            self.details.select(page)
            if prefix == "feature":
                geometry_panel = self.panels.get("Geometry")
                feature_id = int(key.split(":", 1)[1])
                edit_sketch = getattr(geometry_panel, "edit_sketch", None)
                if (
                    callable(edit_sketch)
                    and edit_sketch(feature_id)
                ):
                    self.details.set_hint(
                        f"Editing constrained sketch {self.tree.tree.item(key, 'text')}"
                    )
                    self.set_status("editable sketch loaded on its support plate")
                    return
            if prefix == "imperfection":
                section_panel = self.panels.get("Sections")
                identifier = key.split(":", 1)[1]
                if (
                    section_panel is not None
                    and section_panel.edit_imperfection(identifier)
                ):
                    self.details.set_hint(
                        f"Editing actual values of {self.tree.tree.item(key, 'text')}"
                    )
                    self.set_status("selected imperfection loaded into Details")
                    return
            if prefix in {"plate_section", "beam_section"}:
                section_panel = self.panels.get("Sections")
                if section_panel is not None and section_panel.edit_tree_item(key):
                    self.details.set_hint(
                        f"Editing {self.tree.tree.item(key, 'text')}"
                    )
                    self.set_status("selected section definition loaded into Details")
                    return
            if prefix in {"support", "mass", "load"}:
                load_panel = self.panels.get("Loads & BC")
                if load_panel is not None and load_panel.edit_tree_item(key):
                    self.details.set_hint(
                        f"Editing actual values of {self.tree.tree.item(key, 'text')}"
                    )
                    self.set_status("selected attribute loaded into Details")
                    return
            self.details.set_hint(f"Editing {self.tree.tree.item(key, 'text')}")
            return
        if action == "suppress" and prefix == "feature":
            feature_id = int(key.split(":", 1)[1])
            try:
                from ..commands import SuppressFeature
                record = self.project.geometry.features.get(feature_id)
                self.run(SuppressFeature(feature_id, not record.suppressed))
                record = self.project.geometry.features.get(feature_id)
            except ValueError as error:
                self.set_status(
                    f"feature {feature_id} unchanged: {error}",
                    error=True,
                    diagnostic={
                        "type": type(error).__name__,
                        "message": str(error),
                        "feature_id": feature_id,
                        "action": "suppress",
                    },
                )
                return
            self.set_status(
                f"feature {feature_id} "
                f"{'suppressed' if record.suppressed else 'resumed'}"
            )
            return
        if action == "rename" and prefix == "feature":
            feature_id = int(key.split(":", 1)[1])
            record = self.project.geometry.features.get(feature_id)
            name = self.dialogs.ask_text(
                "Rename feature", "Name", initial=record.name,
            )
            if name:
                from ..commands import RenameFeature
                self.workbench.execute(RenameFeature(feature_id, name), solver_affecting=False)
            return
        if action == "dependencies" and prefix == "feature":
            feature_id = int(key.split(":", 1)[1])
            dependencies = self.project.geometry.features.dependents(
                feature_id, transitive=True
            )
            self.set_status(
                f"feature {feature_id} dependents: "
                + (", ".join(map(str, dependencies)) or "none")
            )
            return
        if action == "isolate":
            isolate = getattr(self.viewport, "isolate", None)
            if callable(isolate):
                isolate(keys)
            else:
                self.set_status("isolate is available from the viewport display groups")
            return
        self.set_status(f"{action} is not available for this item")


    def _delete_tree_items(self, keys: tuple[str, ...]) -> None:
        """Delete every highlighted leaf as one dependency-audited undo item."""

        from ..commands import (
            DeleteAttribute,
            DeleteEntity,
            DeleteFeature,
            DeleteLoadCase,
            DeleteOutputRequest,
            DeleteProjectRecord,
        )
        from ..selection import parse_entity_tag

        selected = tuple(dict.fromkeys(str(key) for key in keys))
        categories: set[str] = set()
        commands = []
        mesh_ids: set[str] = set()
        job_ids: set[str] = set()
        result_ids: set[str] = set()
        cascaded_job_ids: set[str] = set()
        explicitly_selected_job_ids = {
            key.split(":", 1)[1]
            for key in selected
            if key.startswith("job:")
        }
        selected_analysis_ids = {
            key.split(":", 1)[1]
            for key in selected
            if key.startswith("analysis:")
        }
        analysis_dependent_job_ids = {
            job.id
            for job in self.project.jobs.values()
            if job.analysis_id in selected_analysis_ids
        }

        entity_refs = [parse_entity_tag(key) for key in selected]
        if any(ref is not None for ref in entity_refs):
            if not all(ref is not None for ref in entity_refs):
                self.set_status(
                    "Delete geometry separately from other tree item types",
                    error=True,
                )
                return
            categories.add("geometry")
            # Delete owners before their topology dependencies.  The whole
            # sequence remains one CompositeCommand and rolls back on failure.
            order = {"face": 0, "edge": 1, "vertex": 2}
            for ref in sorted(entity_refs, key=lambda item: order.get(item.kind, 9)):
                commands.append(DeleteEntity(ref))
        else:
            for key in selected:
                if ":" not in key:
                    self.set_status(
                        "Select individual tree items, not a branch heading",
                        error=True,
                    )
                    return
                prefix, identifier = key.split(":", 1)
                if prefix in {"support", "mass", "wet_surface", "connection", "imperfection", "load"}:
                    categories.add("attribute")
                    if key.endswith(":gravity"):
                        self.set_status(
                            "Gravity/acceleration is deleted by editing its load case",
                            error=True,
                        )
                        return
                    commands.append(DeleteAttribute(key.rsplit(":", 1)[1]))
                elif prefix == "feature":
                    categories.add("feature")
                    commands.append(DeleteFeature(int(identifier)))
                elif prefix == "case":
                    categories.add("case")
                    case = next(
                        (
                            (name, value)
                            for name, value in self.project.load_cases.items()
                            if str(getattr(value, "id", name)) == identifier
                        ),
                        None,
                    )
                    if case is None:
                        self.set_status(f"Load case {identifier!r} no longer exists", error=True)
                        return
                    commands.append(DeleteLoadCase(case[0]))
                elif prefix == "output_request":
                    categories.add("output request")
                    commands.append(DeleteOutputRequest(identifier))
                elif prefix in {"plate_section", "beam_section"}:
                    categories.add("section definition")
                    commands.append(DeleteProjectRecord(prefix, identifier))
                elif prefix in {
                    "mesh", "analysis", "job", "result", "material",
                    "coordinate", "region",
                }:
                    # A submitted analysis owns its retained job/result
                    # history.  Deleting only the definition would orphan
                    # those records, which the command layer correctly
                    # refuses.  At the tree/workflow level, explicit analysis
                    # deletion therefore removes finished dependants first in
                    # the same atomic, undoable edit.  Active jobs still fail
                    # closed in DeleteProjectRecord("job", ...).
                    if prefix == "analysis":
                        categories.add(prefix)
                        dependants = [
                            job
                            for job in self.project.jobs.values()
                            if job.analysis_id == identifier
                        ]
                        for job in dependants:
                            if job.id not in cascaded_job_ids:
                                commands.append(DeleteProjectRecord("job", job.id))
                                cascaded_job_ids.add(job.id)
                                job_ids.add(job.id)
                                if job.result_artifact_id is not None:
                                    result_ids.add(job.result_artifact_id)
                        commands.append(DeleteProjectRecord(prefix, identifier))
                        continue
                    if prefix == "job" and identifier in analysis_dependent_job_ids:
                        # The user selected both an analysis and its child job;
                        # the cascade already contains it exactly once.
                        continue
                    categories.add(prefix)
                    commands.append(DeleteProjectRecord(prefix, identifier))
                    if prefix == "mesh":
                        mesh_ids.add(identifier)
                    elif prefix == "job":
                        job_ids.add(identifier)
                    elif prefix == "result":
                        result_ids.add(identifier)
                else:
                    self.set_status(
                        f"Delete is not available for {prefix.replace('_', ' ')}",
                        error=True,
                    )
                    return

        # Homogeneous bulk edits are predictable.  Attributes and plate/beam
        # sections are intentionally normalized above so related sibling
        # types may be removed together.
        if len(categories) > 1:
            self.set_status(
                "Delete one kind of tree item at a time; nothing was changed",
                error=True,
            )
            return
        if not commands:
            return
        # Feature dependencies point from later monotonic IDs to earlier IDs.
        # Removing dependants first lets a multi-feature selection succeed
        # without implicit cascade deletion.
        if categories == {"feature"}:
            commands.sort(key=lambda command: int(command.feature_id), reverse=True)
        try:
            self.session.execute_many(
                commands,
                label=f"delete {len(commands)} tree item(s)",
                solver_affecting=categories not in ({"mesh"}, {"job"}, {"result"}),
            )
        except (ValueError, KeyError, ProjectError) as error:
            self.set_status(str(error), error=True)
            return

        if getattr(self, "mesh_record_id", None) in mesh_ids:
            self.mesh = None
            self.mesh_record_id = None
            self._mesh_details_record_id = None
            self.solution = None
            self.show_geometry()
        if self._inspection_mesh is not None and self._mesh_details_record_id in mesh_ids:
            self._inspection_mesh = None
        active_result_removed = self.active_job_id in job_ids
        if self.active_job_id is not None and result_ids:
            old_job = next(
                (
                    value for value in self.project.jobs.values()
                    if value.id == self.active_job_id
                ),
                None,
            )
            active_result_removed = active_result_removed or (
                old_job is not None and old_job.result_artifact_id is None
            )
        if active_result_removed:
            self.active_job_id = None
            self.solution = None
            self.shape_index = 0
            self.show_mesh()
        self.set_status(
            f"deleted {len(selected)} selected {next(iter(categories))} item(s)"
            + (
                f" and {len(cascaded_job_ids - explicitly_selected_job_ids)} "
                "dependent job(s)/result(s)"
                if cascaded_job_ids - explicitly_selected_job_ids
                else ""
            )
            + "; Undo restores all"
        )


    def _schedule_autosave(self) -> None:
        if self._autosave_after is not None:
            try:
                self.scheduler.cancel_call(self._autosave_after)
            except RuntimeError:
                pass
        self._autosave_after = self.scheduler.call_later(
            30_000, self._write_recovery
        )
        if self._autosave_hard_after is None:
            self._autosave_hard_after = self.scheduler.call_later(
                300_000, self._write_recovery
            )


    def _write_recovery(self) -> None:
        """Capture recovery state on the UI thread, then queue file I/O."""

        for attribute in ("_autosave_after", "_autosave_hard_after"):
            identifier = getattr(self, attribute, None)
            if identifier is not None:
                try:
                    self.scheduler.cancel_call(identifier)
                except RuntimeError:
                    pass
                setattr(self, attribute, None)
        if not getattr(self, "session", None) or not self.session.dirty:
            return
        snapshot = self.session.snapshot()
        request = (
            self._recovery_epoch,
            dict(snapshot.document),
            {
                "document_id": self.project.document_id,
                "revision_id": snapshot.revision.id,
                "revision_sequence": snapshot.revision.sequence,
                "document_hash": snapshot.revision.document_hash,
                "model_hash": snapshot.revision.model_hash,
                "saved_document_hash": self.session.saved_document_hash,
                "source_path": self.path,
                # Materialize on the UI thread; generators over a live project
                # are not immutable worker inputs.
                "artifact_refs": tuple(
                    artifact.to_dict() for artifact in self.project.artifacts.values()
                ),
            },
        )
        if self._recovery_future is None:
            self._start_recovery_write(request)
        else:
            # Only the newest dirty revision matters while the preceding
            # atomic bundle finishes.
            self._recovery_pending = request
        if self.session.dirty:
            self._autosave_hard_after = self.scheduler.call_later(
                300_000, self._write_recovery
            )


    def _start_recovery_write(
        self, request: tuple[int, dict[str, Any], dict[str, Any]]
    ) -> None:
        epoch, document, keywords = request
        self._recovery_future_epoch = epoch
        self._recovery_future = self._recovery_executor.submit(
            write_autosave, document, **keywords
        )


    def _poll_recovery_write(self) -> None:
        future = self._recovery_future
        if future is None or not future.done():
            return
        epoch = self._recovery_future_epoch
        self._recovery_future = None
        try:
            future.result()
        except BaseException as error:  # noqa: BLE001 - surface worker failures
            if epoch == self._recovery_epoch:
                self.set_status(f"autosave failed: {error}", error=True)
        else:
            if epoch == self._recovery_epoch:
                self.set_status("autosaved recovery snapshot")

        pending = self._recovery_pending
        self._recovery_pending = None
        if pending is not None and pending[0] == self._recovery_epoch:
            self._start_recovery_write(pending)


    def recover_autosave(self) -> None:
        candidates = discover_recoveries(latest_only=True)
        if not candidates:
            raise ValueError("no recoverable ANYfem autosaves were found")
        candidate = candidates[0]
        if not self.dialogs.confirm(
            "Recover ANYfem autosave",
            f"Recover snapshot from {candidate.created_utc}?\n"
            f"Recommended action: {candidate.recommendation.replace('_', ' ')}",
        ):
            return
        project = project_from_dict(load_recovery(candidate))
        self._set_project(project)
        self.session.dirty = True
        self.set_status("recovered autosave; use Save As to keep it")
        self.refresh_all()
        self.show_geometry(reset_view=True)


    def _load_recent_paths(self) -> list[str]:
        try:
            from platformdirs import user_config_path

            path = Path(user_config_path("ANYfem", appauthor=False)) / "recent.json"
            values = json.loads(path.read_text(encoding="utf-8"))
            return [str(value) for value in values if Path(value).suffix == ".anyfem"][:10]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return []


    def _remember_recent(self, path: Path) -> None:
        resolved = str(path.resolve())
        self._recent_paths = [
            resolved,
            *(item for item in self._recent_paths if item.casefold() != resolved.casefold()),
        ][:10]
        self._refresh_recent_menu()
        try:
            from platformdirs import user_config_path

            destination = Path(user_config_path("ANYfem", appauthor=False)) / "recent.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(self._recent_paths, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temporary, destination)
        except OSError:
            pass


    def _acquire_destination_lock(self, destination: Path) -> ProjectLock:
        current = self._project_lock
        if current is not None and current.project_path == destination.resolve(False):
            return current
        lock = ProjectLock(destination)
        decision = lock.acquire()
        if not decision.acquired:
            raise PermissionError(
                decision.reason or "project is locked by another ANYfem process"
            )
        return lock


    def guarded(self, action):
        """Turn a refusal into a status message rather than a traceback."""

        def wrapped() -> None:
            try:
                action()
            except Exception as error:
                self.set_status(
                    f"{type(error).__name__}: {error}",
                    error=True,
                    diagnostic={
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )

        return wrapped


    def new_project(self, *, confirm: bool = False) -> None:
        """Start a document.

        Direct callers keep the historical noninteractive API.  Commands
        reached through the File menu pass ``confirm=True`` and therefore
        protect unsaved work.
        """

        if confirm and not self._confirm_discard():
            return
        self._set_project(default_project())
        self.set_status("new model")
        self.refresh_all()
        self.show_geometry(reset_view=True)


    def open_project(
        self, path: Optional[str] = None, *, confirm: bool = False
    ) -> None:
        if confirm and not self._confirm_discard():
            return
        if path is None:
            path = self.dialogs.open_file(
                filetypes=[("ANYfem project", "*.anyfem"), ("All files", "*.*")]
            )
            if not path:
                return
        source = Path(path)
        current_lock = self._project_lock
        lock = (current_lock if current_lock is not None
                and current_lock.project_path == source.resolve(False)
                else ProjectLock(source))
        decision = lock.acquire()
        if decision.can_take_over:
            answer = self.dialogs.confirm_save(
                "Stale ANYfem project lock",
                "The previous ANYfem process no longer owns this project.\n\n"
                "Yes: take over the stale lock and edit the project.\n"
                "No: open the project read-only.\n"
                "Cancel: leave the current project open.",
            )
            if answer is None:
                return
            if answer:
                decision = lock.acquire(take_over_stale=True)
        held_lock = lock if decision.acquired else None
        try:
            loaded = load_project(source)
        except BaseException:
            if held_lock is not None and held_lock is not current_lock:
                held_lock.release()
            raise
        self._set_project(
            loaded,
            path=source,
            project_lock=held_lock,
            read_only=decision.read_only,
        )
        # Generated and imported meshes are optional artifacts; failure to
        # locate one never prevents the editable document from opening.
        try:
            from ..io.artifacts import ArtifactStore

            store = ArtifactStore(path)
            if loaded.mesh_records:
                # UUID-sorted project registries do not preserve creation order.
                # Prefer the saved active record, then the imported semantics
                # association, then a usable legacy artifact by creation time.
                candidates=list(loaded.mesh_records.values())
                def priority(record):
                    artifact=loaded.artifacts.get(record.artifact_id or "")
                    return (bool(record.summary.get("active_mesh")),bool(record.artifact_id and record.artifact_id==loaded.imported_semantics_artifact_id),artifact is not None,artifact.created_utc if artifact is not None else "",record.id)
                latest=max(candidates,key=priority)
                self._mesh_details_record_id = latest.id
                artifact = loaded.artifacts.get(latest.artifact_id or "")
                if artifact is not None:
                    loaded_mesh = store.read_mesh(artifact)
                    self._meshes[latest.id] = loaded_mesh
                    if self.mesh_record_state(latest) == "inspection_only":
                        self._inspection_mesh = loaded_mesh
                    elif latest.kind == "imported" or self.mesh_record_state(latest) != "stale":
                        self.mesh_record_id = latest.id
                        self.mesh = loaded_mesh
                    if loaded.mesh_only and loaded.imported_format == "sesam_fem":
                        from ..io.sesam import import_sesam_artifact

                        self.imported = import_sesam_artifact(store, artifact)
                        # Use the verified sidecar association map so mesh
                        # regions and result IDs are byte-for-byte those saved.
                        self.imported.mesh = loaded_mesh
                if self._inspection_mesh is not None:
                    for prior in reversed(list(loaded.mesh_records.values())[:-1]):
                        if self.mesh_record_state(prior) != "completed":
                            continue
                        prior_artifact = loaded.artifacts.get(prior.artifact_id or "")
                        if prior_artifact is None:
                            continue
                        self.mesh = store.read_mesh(prior_artifact)
                        self.mesh_record_id = prior.id
                        self._meshes[prior.id] = self.mesh
                        break
        except (OSError, ValueError):
            self.mesh = None
        try:
            from ..io.artifacts import ArtifactStore

            store = ArtifactStore(path)
            for job_id, record in loaded.jobs.items():
                artifact = loaded.artifacts.get(record.result_artifact_id or "")
                if artifact is None:
                    continue
                try:
                    self.result_datasets=dict(self.result_datasets,**{job_id:store.open_result(artifact,defer=True)})
                except (OSError, ValueError) as error:
                    record.diagnostics.append(
                        {
                            "type": type(error).__name__,
                            "message": f"result artifact unavailable: {error}",
                        }
                    )
            if self.result_datasets:
                self.active_job_id = next(reversed(self.result_datasets))
        except (OSError, ValueError):
            pass
        self._remember_recent(source)
        self.set_status(
            f"opened {self.path.name}"
            + (f" read-only: {decision.reason}" if decision.read_only else "")
        )
        self.refresh_all()
        if loaded.mesh_only and self.mesh is not None:
            self.show_mesh()
            self.viewport.fit()
        else:
            self.show_geometry(reset_view=True)


    def save_project(self, ask: bool = False, path: Optional[str] = None) -> None:
        if path is None:
            path = str(self.path) if self.path and not ask else ""
        if not path:
            path = self.dialogs.save_file(
                defaultextension=".anyfem",
                filetypes=[("ANYfem project", "*.anyfem"), ("All files", "*.*")],
                initialfile=f"{self.project.name}.anyfem",
            )
            if not path:
                return
        destination = Path(path)
        if not destination.suffix:
            destination = destination.with_suffix(".anyfem")
        if (
            self.session.read_only
            and self.path is not None
            and destination.resolve(False) == self.path.resolve(False)
        ):
            raise PermissionError("this project is locked; use Save As")
        destination_lock = self._acquire_destination_lock(destination)
        previous_lock = self._project_lock
        owns_new_lock = destination_lock is not previous_lock
        try:
            self._save_project_contents(destination)
            saved_path = save_project(self.project, destination)
        except BaseException:
            if owns_new_lock:
                destination_lock.release()
            raise
        if previous_lock is not None and previous_lock is not destination_lock:
            previous_lock.release()
        self._project_lock = destination_lock
        self.session.read_only = False
        self.path = saved_path
        self.session.mark_saved(self.path)
        self._remember_recent(self.path)
        self._update_window_title()
        self.set_status(f"saved {self.path.name}")


    def _save_project_contents(self, destination: Path) -> None:
        """Commit sidecars before the project index JSON."""

        saved_mesh_ids: set[str] = set()
        if self.mesh is not None:
            from ..io.artifacts import ArtifactStore
            from anymesher.serialize import mesh_to_dict

            store = ArtifactStore(destination)
            mesh_record = self.project.mesh_records.get(
                getattr(self, "mesh_record_id", "")
            )
            if mesh_record is None:
                mesh_record = MeshRecord(
                    name="Imported mesh" if self.imported is not None else "Mesh",
                    kind="imported" if self.imported is not None else "generated",
                    source_model_hash=self.session.revision.model_hash,
                    mesh_input_hash="",
                    mesh_hash=canonical_hash(mesh_to_dict(self.mesh)),
                    summary={
                        "nodes": self.mesh.num_nodes,
                        "elements": self.mesh.num_elements,
                    },
                )
            imported_metadata = None
            embedded_source = None
            if self.imported is not None:
                imported_metadata, embedded_source = self.imported.artifact_embedding()
            artifact = store.write_mesh(
                self.mesh,
                mesh_id=mesh_record.id,
                document_id=self.project.document_id,
                model_hash=mesh_record.source_model_hash,
                mesh_hash=mesh_record.mesh_hash,
                structural_preparation=(
                    mesh_record.structural_preparation
                ),
                imported_model=imported_metadata,
                embedded_source=embedded_source,
            )
            mesh_record.artifact_id = artifact.id
            with self.session.transaction("record saved mesh", solver_affecting=False):
                self.project.mesh_records[mesh_record.id] = mesh_record
                for candidate in self.project.mesh_records.values():
                    candidate.summary.pop("active_mesh",None)
                mesh_record.summary["active_mesh"]=True
                self.project.artifacts[artifact.id] = artifact
                if self.imported is not None:
                    self.project.imported_semantics_artifact_id = artifact.id
            self.mesh_record_id = mesh_record.id
            self._meshes[mesh_record.id] = self.mesh
            saved_mesh_ids.add(mesh_record.id)
        self._persist_cached_mesh_artifacts(destination, exclude=saved_mesh_ids)
        self._persist_result_artifacts(destination)


    def _persist_cached_mesh_artifacts(
        self, destination: Path, *, exclude: set[str]
    ) -> None:
        """Persist retained stale generated meshes so they remain inspectable."""

        if not self._meshes:
            return
        from ..io.artifacts import ArtifactStore

        store = ArtifactStore(destination)
        for mesh_id, mesh in tuple(self._meshes.items()):
            if mesh_id in exclude:
                continue
            record = self.project.mesh_records.get(mesh_id)
            if record is None or record.kind == "imported":
                continue
            artifact = store.write_mesh(
                mesh,
                mesh_id=record.id,
                document_id=self.project.document_id,
                model_hash=record.source_model_hash,
                mesh_hash=record.mesh_hash,
                structural_preparation=record.structural_preparation,
            )
            with self.session.transaction(
                "record retained mesh artifact", solver_affecting=False
            ):
                record.artifact_id = artifact.id
                self.project.artifacts[artifact.id] = artifact


    def _persist_result_artifacts(self, destination: Path) -> None:
        """Write in-memory results and copy unopened sidecars for Save As."""

        from ..io.artifacts import ArtifactStore

        # Do not race an automatic post-job write with the explicit save.
        for job_id, future in tuple(self._artifact_futures.items()):
            try:
                artifact = future.result()
            except BaseException as error:
                record = self.project.jobs.get(job_id)
                if record is not None:
                    record.diagnostics.append(
                        {"type": type(error).__name__, "message": str(error)}
                    )
            else:
                source_destination = self._artifact_destinations.get(job_id)
                if job_id in self.project.jobs and source_destination is not None:
                    self._record_result_artifact(
                        job_id, artifact, source_destination
                    )
            self._artifact_futures.pop(job_id, None)
            self._artifact_destinations.pop(job_id, None)

        store = ArtifactStore(destination)
        written: set[str] = set()
        for job_id, solution in tuple(self.solutions.items()):
            record = self.project.jobs.get(job_id)
            if record is None:
                continue
            mesh_id=self._result_mesh_id(record)
            artifact = write_solution_artifact(
                store,
                solution,
                job_id=record.id,
                document_id=self.project.document_id,
                mesh_id=mesh_id,
                model_hash=record.model_hash,
                mesh_hash=record.mesh_hash,
                analysis_hash=record.analysis_hash,
                provenance=self._result_artifact_provenance(job_id),
                summary=dict(record.summary),
                diagnostics=tuple(record.diagnostics),
                partial=bool(record.partial),
            )
            self._record_result_artifact(job_id, artifact, destination)
            written.add(artifact.id)

        written.update(self._persist_job_log_artifacts(destination))

        if self.path is None or self.path.resolve(False) == destination.resolve(False):
            return
        source = ArtifactStore(self.path)
        for record in self.project.jobs.values():
            for artifact_id, label in (
                (record.result_artifact_id, "result"),
                (record.log_artifact_id, "job log"),
            ):
                if not artifact_id or artifact_id in written:
                    continue
                artifact = self.project.artifacts.get(artifact_id)
                if artifact is None:
                    continue
                try:
                    copied = store.copy_from(source, artifact)
                except (OSError, ValueError) as error:
                    record.diagnostics.append(
                        {
                            "type": type(error).__name__,
                            "message": f"{label} artifact unavailable during Save As: {error}",
                        }
                    )
                    continue
                self.project.artifacts[copied.id] = copied


    def _persist_job_log_artifacts(self, destination: Path) -> set[str]:
        """Finish automatic log writes and reproduce current logs on save."""

        from ..io.artifacts import ArtifactStore

        written: set[str] = set()
        for job_id, future in tuple(self._log_futures.items()):
            try:
                artifact = future.result()
            except BaseException as error:
                record = self.project.jobs.get(job_id)
                if record is not None:
                    record.diagnostics.append(
                        {"type": type(error).__name__, "message": str(error)}
                    )
            else:
                if job_id in self.project.jobs:
                    self._record_job_log_artifact(job_id, artifact)
                    written.add(artifact.id)
            self._log_futures.pop(job_id, None)
            self._log_destinations.pop(job_id, None)

        store = ArtifactStore(destination)
        terminal = {"completed", "cancelled", "failed", "partial", "interrupted"}
        for job_id, record in self.project.jobs.items():
            status = getattr(record.status, "value", str(record.status))
            if status not in terminal:
                continue
            try:
                entries = self.job_manager.log(job_id)
            except KeyError:
                continue
            artifact = store.write_log(job_id, entries)
            self._record_job_log_artifact(job_id, artifact)
            written.add(artifact.id)
        return written


    def import_sesam_model(self, path: Optional[str] = None) -> None:
        if path is None:
            path = self.dialogs.open_file(
                filetypes=[("SESAM FEM", "*.FEM *.fem"), ("All files", "*.*")]
            )
            if not path:
                return
        model = import_sesam(path)
        if not self._confirm_discard():
            return
        project = model.project()
        project.mesh_only = True
        project.imported_format = "sesam_fem"
        self._set_project(project, imported=model)
        self.mesh = model.mesh
        # Jobs submitted before the first save still need a stable mesh
        # identity, so their retained answers can resolve that mesh on reopen.
        from anymesher.serialize import mesh_to_dict
        record = MeshRecord(
            name="Imported mesh", kind="imported",
            source_model_hash=self.session.revision.model_hash,
            mesh_input_hash="",
            mesh_hash=canonical_hash(mesh_to_dict(model.mesh)),
            summary={"nodes": model.mesh.num_nodes,
                     "elements": model.mesh.num_elements, "active_mesh": True},
        )
        with self.session.transaction("record imported mesh", solver_affecting=False):
            self.project.mesh_records[record.id] = record
        self.mesh_record_id = record.id
        self._meshes[record.id] = model.mesh
        note = (
            ""
            if not model.diagnostics
            else f"; {len(model.diagnostics)} diagnostic(s)"
        )
        self.set_status(
            f"imported {model.summary()}{note}. Geometry editing is off: an "
            "imported file has a mesh, not plates and lines."
        )
        self.refresh_all()
        self.show_mesh()
        self.viewport.fit()


    def _set_project(
        self,
        project: Project,
        *,
        path: Path | None = None,
        imported=None,
        project_lock: ProjectLock | None = None,
        read_only: bool = False,
    ) -> None:
        # Finish old-document writers while its lock and session still belong
        # to us. Their completion must never be applied to a new document.
        try:
            self.flush_project_writes()
        except BaseException:
            if project_lock is not None and project_lock is not self._project_lock:
                project_lock.release()
            raise
        try:
            self.viewport.cancel_construction()
            self.worker.cancel()
            self.commands.remove_listener(self.refresh_all)
            self.session.remove_listener(self._on_revision_changed)
        except (AttributeError, ValueError):
            pass
        if self._project_lock is not None and self._project_lock is not project_lock:
            self._project_lock.release()
        self._project_lock = project_lock
        self._recovery_epoch += 1
        self._recovery_pending = None
        self.workbench.replace_project(
            project,
            path=path,
            imported=imported,
            read_only=read_only,
        )
        self._active_loading_hash = _loading_hash(project)
        self._inspection_mesh = None
        self._geometry_scene_cache = None
        self._mesh_layout_preview = None
        cancel_construction=getattr(self.viewport,"cancel_construction",None)
        if callable(cancel_construction):cancel_construction()
        mesh_panel = self.panels.get("Mesh")
        if mesh_panel is not None:
            mesh_panel.reset_mesh_drafts()
        self.commands.add_listener(self.refresh_all)
        self.session.add_listener(self._on_revision_changed)
        self.worker = JobWorkerFacade(self.job_manager)
        self.tree.project = project
        self.tree.reset_feature_exposure()
        self._artifact_futures.clear()
        self._artifact_destinations.clear()
        self._log_futures.clear()
        self._log_destinations.clear()
        self._active_mesh_task_id = None
        self._mesh_details_record_id = None
        self.selection.clear()
        self._update_window_title()

    def flush_project_writes(self) -> None:
        """Drain owned sidecar/recovery writers before relinquishing a lock.

        These workers perform file I/O only and do not wait for UI callbacks.
        Record failures through the existing artifact diagnostic path.
        """
        for futures in (self._artifact_futures, self._log_futures):
            for future in tuple(futures.values()):
                try:
                    future.result()
                except Exception:
                    pass  # Pollers retain the failure on the owning job.
        self._poll_result_artifacts()
        self._poll_job_log_artifacts()
        while self._recovery_future is not None:
            try:
                self._recovery_future.result()
            except Exception:
                pass  # Recovery poller reports the failure.
            self._poll_recovery_write()


    def _confirm_discard(self) -> bool:
        if not getattr(self, "session", None) or not self.session.dirty:
            return True
        answer = self.dialogs.confirm_save(
            "Unsaved ANYfem project",
            "Save the current project before continuing?",
        )
        if answer is None:
            return False
        if answer:
            self.save_project()
            return not self.session.dirty
        return True


    def import_calculix_result(self, path: Optional[str] = None) -> None:
        """Read a CalculiX FRD onto the current model."""

        if path is None:
            path = self.dialogs.open_file(
                filetypes=[
                    ("CalculiX results", "*.frd *.FRD *.dat *.DAT"),
                    ("All files", "*.*"),
                ]
            )
            if not path:
                return
        self._attach_results(import_calculix_results(path))


    def import_sesam_result(self, path: Optional[str] = None) -> None:
        """Read SESAM SIF shell stresses onto the current model."""

        if path is None:
            path = self.dialogs.open_file(
                filetypes=[("SESAM SIF", "*.SIF *.sif"), ("All files", "*.*")]
            )
            if not path:
                return
        self._attach_results(import_sesam_results(path))


    def _attach_results(self, results) -> None:
        """Bind imported results to whatever model is loaded.

        Matching is by node ID, so this needs a model that has been meshed or
        imported.  A mismatch is reported rather than partially attached.
        """

        built = self.built()
        if built is None:
            raise ValueError(
                "generate or import a mesh first: results are matched to a "
                "model by node ID, so there has to be one to match against"
            )
        solution = results.attach(built)
        # Imported answers own a separate retained result identity. Reusing a
        # previous solve's active job would mislabel its quantity and provenance.
        from ..model.records import JobRecord,JobStatus
        from datetime import datetime,timezone
        now=datetime.now(timezone.utc).isoformat()
        mesh_record=self.project.mesh_records.get(self.mesh_record_id or "")
        if mesh_record is None:
            from anymesher.serialize import mesh_to_dict
            mesh_record=MeshRecord(name="Imported mesh" if self.imported is not None else "Mesh",kind="imported" if self.imported is not None else "generated",source_model_hash=self.session.revision.model_hash,mesh_input_hash="",mesh_hash=canonical_hash(mesh_to_dict(self.mesh)),summary={"nodes":self.mesh.num_nodes,"elements":self.mesh.num_elements})
        source_hash=results.source_hash
        record=JobRecord(analysis_id="",name=f"Imported {results.format}: {Path(results.source).name}",model_hash=self.session.revision.model_hash,mesh_hash=mesh_record.mesh_hash,analysis_hash=canonical_hash({"format":results.format,"source_hash":source_hash}),input_hash=source_hash,status=JobStatus.COMPLETED,created_utc=now,started_utc=now,finished_utc=now,summary={"source":str(results.source),"source_hash":source_hash,"format":results.format})
        with self.session.transaction("attach imported results",solver_affecting=False):
            self.project.mesh_records[mesh_record.id]=mesh_record
            self.project.jobs[record.id]=record
        self.mesh_record_id=mesh_record.id;self._meshes[mesh_record.id]=self.mesh
        self.solutions[record.id]=solution;self.active_job_id=record.id
        self.solution=solution;self.shape_index=0
        self.set_status(results.summary())
        self.refresh_all()
        self.show_results()


    def built(self):
        """The built model behind the current mesh, if there is one."""

        if self.imported is not None:
            return self.imported.built(project=self.project)
        if self.mesh is None:
            return None
        from ..solve.build import build_fe_model

        return build_fe_model(
            self.project, self.mesh,
            require_loads=False, require_supports=False,
        )


    def export_deck(self, path: Optional[str] = None) -> None:
        if self.mesh is None:
            raise ValueError("generate or import a mesh first")
        if path is None:
            path = self.dialogs.save_file(
                defaultextension=".inp",
                filetypes=[("CalculiX deck", "*.inp"), ("All files", "*.*")],
                initialfile=f"{self.project.name}.inp",
            )
            if not path:
                return
        built = self.build_current()
        written = export_calculix_deck(built, path)
        self.set_status(
            f"deck written to {Path(path).name}. A generated deck is a handoff, "
            "not evidence: it says nothing until it has been run and compared."
        )
        return written


    def build_current(self):
        """The built model for whatever is loaded, imported or modelled."""

        if self.imported is not None:
            case = self.project.load_cases.get(self.active_case())
            return self.imported.built(case, project=self.project)
        if self.mesh is None:
            raise ValueError("generate a mesh first")
        return build_fe_model(self.project, self.mesh)


    def diagnostic_report(self, *, recent_commands=()) -> str:
        """Return the current support report without exposing full model data."""

        selection = self.selection
        mesh_record = self.project.mesh_records.get(
            str(getattr(self, "mesh_record_id", ""))
        )
        context = {
            "project_path": None if self.path is None else str(self.path),
            "details_page": getattr(self.details, "_current", None),
            "view_mode": self._view_mode,
            "renderer_requested": getattr(self.viewport, "requested_backend", None),
            "renderer_active": getattr(self.viewport, "active_backend", None),
            "status_text": self._status.cget("text"),
            "revision": {
                "sequence": self.session.revision.sequence,
                "document_hash": self.session.revision.document_hash,
                "model_hash": self.session.revision.model_hash,
            },
            "selection": {
                "domain": selection.domain.value,
                "mode": selection.mode,
                "allowed_kinds": tuple(sorted(selection.allowed_kinds)),
                "items": tuple(repr(item) for item in selection.ordered_items),
            },
            "active_mesh_record": None if mesh_record is None else mesh_record.to_dict(),
        }
        return build_diagnostic_report(
            self.project,
            errors=self._error_diagnostics,
            context=context,
            recent_commands=recent_commands,
        )


def _solution_report(solution) -> str:
    """A text summary that suits whichever analysis produced the result."""

    mesh = solution.built.mesh
    lines = [
        solution.built.project.name,
        "",
        f"nodes                {mesh.num_nodes}",
        f"shell elements       {len(mesh.shells)}",
        f"beam elements        {len(mesh.beams)}",
        "",
        solution.summary(),
    ]

    shapes = getattr(solution, "shapes", None)
    if shapes:
        lines.append("")
        for shape in shapes[:12]:
            lines.append(f"  {shape.label:<16} {shape.value:.6g}")
        if len(shapes) > 12:
            lines.append(f"  ... {len(shapes) - 12} more")

    steps = getattr(solution, "steps", None)
    if steps:
        lines.append("")
        lines.append(f"steps                {len(steps)}")
        lines.append(f"status               {solution.status}")
    return "\n".join(lines)


def _job_progress_text(message: str, payload: Any = None) -> str:
    """Format strings and ANYsolver ProgressEvents for the live transcript."""

    base = str(message or "").strip()
    if payload is None or isinstance(payload, str):
        return base or "working"

    def value(name: str, default=None):
        attribute = getattr(payload, name, None)
        if attribute is not None:
            return attribute
        getter = getattr(payload, "get", None)
        return getter(name, default) if callable(getter) else default

    stage = str(value("stage", "")).strip().replace("_", " ")
    detail = str(value("message", base) or base).strip()
    qualifiers = []
    completed = value("completed")
    total = value("total")
    fraction = value("fraction")
    iteration = value("iteration")
    if completed is not None and total not in (None, 0, 0.0):
        qualifiers.append(f"{float(completed):g}/{float(total):g}")
    elif fraction is not None:
        qualifiers.append(f"{100.0 * float(fraction):.0f}%")
    if iteration is not None:
        qualifiers.append(f"iteration {int(iteration)}")
    for key, label in (
        ("load_factor", "load"),
        ("time_s", "t"),
        ("residual_norm", "residual"),
    ):
        item = value(key)
        if item is not None:
            suffix = " s" if key == "time_s" else ""
            qualifiers.append(f"{label} {float(item):.5g}{suffix}")
    prefix = stage or "solver"
    line = f"{prefix}: {detail}" if detail and detail != stage else prefix
    return line + (f"  [{' | '.join(qualifiers)}]" if qualifiers else "")


def _loading_hash(project: Project) -> str:
    """Hash persisted load inputs without copying runtime change callbacks."""
    from ..io.project_file import _load_case_to_dict

    return canonical_hash({
        "load_cases": [_load_case_to_dict(case) for case in sorted(project.load_cases.values(), key=lambda case: case.name)],
        "combinations": [
            {"id": item.id, "name": item.name, "factors": dict(item.factors)}
            for item in sorted(project.combinations.values(), key=lambda item: item.name)
        ],
    })


def _record_settings(value: Any) -> Any:
    """Make analysis settings deterministic and JSON-safe for provenance."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _record_settings(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_record_settings(item) for item in value]
    if hasattr(value, "tolist"):
        return _record_settings(value.tolist())
    if hasattr(value, "to_dict"):
        return _record_settings(value.to_dict())
    if is_dataclass(value):
        return _record_settings(asdict(value))
    return {"type": type(value).__name__, "repr": repr(value)}


def _submitted_input_report(
    project: Project,
    definition: AnalysisDefinition,
    options: Dict[str, Any],
    mesh: Any,
    *,
    revision: int,
    model_hash: str,
    mesh_hash: str,
    output_scopes: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Readable, deterministic solver-input record without dumping mesh arrays."""

    load_cases = []
    for case in sorted(project.load_cases.values(), key=lambda item: item.name):
        load_cases.append(
            {
                "id": case.id,
                "name": case.name,
                "follower_pressure": case.follower_pressure,
                "gravity_or_acceleration_m_per_s2": _record_settings(case.gravity),
                "gravity_coordinate_system_id": case.gravity_coordinate_system_id,
                "point_loads": _record_settings(case.point_loads),
                "pressures": _record_settings(case.pressures),
                "line_loads": _record_settings(case.line_loads),
                "surface_tractions": _record_settings(case.surface_tractions),
                **(
                    {
                        "wet_pressures": _record_settings(case.wet_pressures),
                        "time": case.time,
                        "parameters": _record_settings(case.parameters),
                        "series": case.series,
                        "step": case.step,
                    }
                    if case.wet_pressures
                    or case.time is not None
                    or case.parameters
                    or case.series is not None
                    else {}
                ),
                **(
                    {"connection_loads": _record_settings(case.connection_loads)}
                    if case.connection_loads
                    else {}
                ),
                **({"times": list(case.times)} if case.times is not None else {}),
                **({"parent": case.parent} if case.parent is not None else {}),
            }
        )
    support_inputs = []
    for support in project.supports:
        engineering = {
            dof: {
                "value": 1000.0 * float(value),
                "unit": "mm" if dof.startswith("u") else "mrad",
            }
            for dof, value in support.constraints.items()
        }
        support_inputs.append(
            {
                "id": support.id,
                "name": support.name,
                "ref": _record_settings(support.ref),
                "region": _record_settings(support.region),
                "coordinate_system_id": support.coordinate_system_id,
                "constraints_SI": dict(support.constraints),
                "constraints_engineering": engineering,
            }
        )
    mesh_summary = {
        "uuid": getattr(mesh, "id", None),
        "nodes": getattr(mesh, "num_nodes", None),
        "elements": getattr(mesh, "num_elements", None),
        "shell_elements": len(getattr(mesh, "shells", {})) if mesh is not None else 0,
        "beam_elements": len(getattr(mesh, "beams", {})) if mesh is not None else 0,
        "mesh_hash": mesh_hash,
    }
    payload = {
        "record": {
            "project": project.name,
            "revision": int(revision),
            "model_hash": model_hash,
            "note": (
                "Numerical values are SI unless a key states another unit. "
                "Mesh connectivity is identified by mesh_hash and omitted from this UI view."
            ),
        },
        "analysis": definition.to_dict(),
        "output_requests": [
            project.output_requests[identifier].to_dict()
            for identifier in definition.output_request_ids
        ],
        "output_request_scopes": list(output_scopes),
        "submitted_options": _record_settings(options),
        "mesh": mesh_summary,
        "units": project.units.to_dict(),
        "coordinate_systems": [
            item.to_dict()
            for item in sorted(project.coordinate_systems.values(), key=lambda item: item.id)
        ],
        "materials": [
            item.to_dict()
            for item in sorted(project.materials.values(), key=lambda item: item.name)
        ],
        "plate_sections": _record_settings(
            sorted(project.plate_sections.values(), key=lambda item: item.name)
        ),
        "beam_sections": _record_settings(
            sorted(project.beam_sections.values(), key=lambda item: item.name)
        ),
        "section_assignments": [
            item.to_dict()
            for item in sorted(
                project.section_assignments.values(), key=lambda item: item.id
            )
        ],
        "supports": support_inputs,
        "masses": _record_settings(project.masses),
        **(
            {"wet_surfaces": _record_settings(project.wet_surfaces)}
            if project.wet_surfaces
            else {}
        ),
        **(
            {"connections": _record_settings(project.connections)}
            if project.connections
            else {}
        ),
        "load_cases": load_cases,
        "combinations": _record_settings(
            sorted(project.combinations.values(), key=lambda item: item.name)
        ),
        "imperfections": _record_settings(project.imperfections),
        "regions": project.regions.to_list(),
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def _execute_analysis_job(
    *,
    project: Project,
    solver_function,
    analysis_name: str,
    options: Dict[str, Any],
    progress,
    cancellation_token=None,
    requested_stress=False,
    requested_global_stress=False,
    requested_patch_stress=False,
):
    """Build, preflight and solve one immutable JobManager request."""

    from ..solve.run import preflight

    resolved = dict(options)
    built = resolved.pop("built", None)
    mesh = resolved.pop("mesh", None)
    if built is None:
        build_options: Dict[str, Any] = {}
        if analysis_name == "Batch linear static":
            # The batch builds each requested case itself, so there is no
            # single "default" case to resolve here.
            build_options["load_case"] = None
        elif "combination" in resolved:
            build_options["combination"] = resolved["combination"]
        elif "load_case" in resolved:
            build_options["load_case"] = resolved["load_case"]
        if analysis_name == "Modal":
            build_options.update(require_loads=False, require_supports=False)
        elif analysis_name == "Impact" and resolved.get("load_case") is None:
            build_options.update(require_loads=False, require_supports=True)
        progress("building immutable model snapshot")
        built = build_fe_model(project, mesh, **build_options)

    kinematics = str(resolved.get("kinematics", "von_karman"))
    report = preflight(
        built,
        analysis_type=analysis_name,
        kinematics=kinematics,
        corotational_tangent=str(resolved.get("corotational_tangent", "auto")),
    )
    if report.errors:
        details = _format_preflight_errors(report.errors)
        if (
            any(issue.code == "CONSTRAINT003" for issue in report.errors)
            and int(getattr(built.mesh, "automatic_shell_connections", 0)) > 0
        ):
            details += (
                "\n[AUTOMESH001] The submitted mesh contains cyclic automatic "
                "shell-interface ties. Regenerate the mesh with the current "
                "mesher before rerunning; the supports do not need to be changed."
            )
        raise ValueError(f"preflight blocked {analysis_name}:\n{details}")
    for issue in report.warnings:
        progress(f"warning [{issue.code}]: {issue.message}")
    progress("preflight passed")

    resolved["built"] = built
    resolved["progress"] = progress
    # New wrappers advertise this argument explicitly.  Older coordinated
    # packages remain usable; cancellation then stays in the visible
    # 'cancelling' state until their library call returns.
    import inspect

    if cancellation_token is not None and "cancellation_token" in inspect.signature(solver_function).parameters:
        resolved["cancellation_token"] = cancellation_token
    solution = solver_function(**resolved)
    if requested_stress:
        from ..post.results import LinearBatchSolution, NonlinearSolution
        recovery_solutions = (solution,)
        if isinstance(solution, LinearBatchSolution):
            recovery_solutions = solution.shapes
        elif isinstance(solution, NonlinearSolution) and solution.shapes:
            recovery_solutions = solution.shapes
        for case_solution in recovery_solutions:
            if not callable(getattr(case_solution, "stresses", None)):
                continue
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled("requested stress recovery")
            progress(f"recovering requested stresses: {case_solution.label}" if hasattr(case_solution,"label")
                     else "recovering requested stresses")
            case_solution.stresses()
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled("requested stress recovery")
            if requested_global_stress:
                progress("recovering requested global stresses")
                case_solution._requested_global_stress = case_solution.stresses(return_global=True)
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled("requested global stress recovery")
            if requested_patch_stress:
                from anysolver import PatchRecoveryConfig
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled("requested patch stress recovery")
                progress("recovering requested patch stresses")
                case_solution._requested_patch_stress = case_solution.stresses(patch_config=PatchRecoveryConfig())
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled("requested patch stress recovery")
    return solution


def _format_preflight_errors(errors) -> str:
    """Format repeated entity diagnostics compactly without losing IDs."""

    grouped: dict[tuple, list[int]] = {}
    for issue in errors:
        key = (
            str(issue.code),
            str(issue.message),
            str(issue.suggestion),
            str(issue.entity_type),
            issue.measured,
            issue.limit,
        )
        identifiers = grouped.setdefault(key, [])
        if issue.entity_id is not None:
            identifiers.append(int(issue.entity_id))

    lines = []
    for key, identifiers in grouped.items():
        code, message, suggestion, entity_type, _measured, _limit = key
        line = f"[{code}] {message}"
        if identifiers:
            unique = list(dict.fromkeys(identifiers))
            shown = ", ".join(str(value) for value in unique[:16])
            if len(unique) > 16:
                shown += f", +{len(unique) - 16} more"
            label = entity_type or "entity"
            line += f" Affected {label} ID(s): {shown}."
        if suggestion:
            line += f" Suggestion: {suggestion}"
        lines.append(line)
    return "\n".join(lines)
