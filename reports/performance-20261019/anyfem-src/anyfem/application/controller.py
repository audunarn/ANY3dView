"""Headless ownership of one interactive ANYfem workbench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from ..commands import Command
from ..document import DocumentRevision, DocumentSession
from ..jobs import JobManager
from ..mesh_jobs import MeshTaskManager
from ..model.materials import steel
from ..model.project import Project
from ..model.sections import BeamSection
from ..selection import Selection

__all__ = [
    "WorkbenchController",
    "WorkbenchEvent",
    "WorkbenchSnapshot",
    "default_project",
]


def default_project() -> Project:
    """Return the application baseline without importing a GUI toolkit."""

    project = Project(name="model")
    project.add_material(steel("S355", 0.010))
    project.add_plate_section("plate", thickness=0.010, material="S355")
    project.add_beam_section(
        BeamSection(
            name="stiffener",
            profile="T-bar",
            material="S355",
            web_height=0.200,
            web_thickness=0.010,
            flange_width=0.100,
            flange_thickness=0.012,
            offset_mode="automatic",
        )
    )
    return project


@dataclass(frozen=True, slots=True)
class WorkbenchSnapshot:
    document_id: str
    revision: DocumentRevision
    view_mode: str
    selection_mode: str
    selected_count: int
    active_job_id: str | None
    active_mesh_record_id: str | None
    has_mesh: bool
    has_solution: bool
    project_path: Path | None


@dataclass(frozen=True, slots=True)
class WorkbenchEvent:
    kind: str
    field: str | None
    snapshot: WorkbenchSnapshot


WorkbenchListener = Callable[[WorkbenchEvent], Any]


class WorkbenchController:
    """State and commands shared by current and future desktop frontends.

    The controller owns no widget and imports no GUI package.  Frontends may
    observe coarse state events, while the existing detailed document and
    selection listeners remain available for efficient incremental updates.
    """

    MUTABLE_STATE_FIELDS = frozenset(
        {
            "mesh",
            "meshes",
            "mesh_record_id",
            "solution",
            "solutions",
            "result_datasets",
            "submitted_input_reports",
            "active_job_id",
            "analysis",
            "shape_index",
            "imported",
            "path",
            "seeding_overrides",
            "view_mode",
            "geometry_selection_mode",
            "active_model_hash",
        }
    )

    def __init__(
        self,
        project: Project | None = None,
        *,
        selection: Selection | None = None,
        path: Path | None = None,
        job_manager_factory: Callable[[Project], Any] = JobManager,
        mesh_task_manager_factory: Callable[[], Any] = MeshTaskManager,
    ) -> None:
        self._listeners: list[WorkbenchListener] = []
        self._job_manager_factory = job_manager_factory
        self._mesh_task_manager_factory = mesh_task_manager_factory
        self.selection = Selection(mode="vertex") if selection is None else selection
        self._install_project(project or default_project(), path=path)

    def _install_project(
        self,
        project: Project,
        *,
        path: Path | None,
        imported: Any = None,
        read_only: bool = False,
    ) -> None:
        if not isinstance(project, Project):
            raise TypeError("workbench project must be an ANYfem Project")
        self.project = project
        self.session = DocumentSession(project, selection=self.selection, path=path)
        self.session.read_only = bool(
            read_only or getattr(project, "read_only_reason", None)
        )
        self.commands = self.session.commands
        self.active_model_hash = self.session.revision.model_hash
        self.job_manager = self._job_manager_factory(project)
        self.mesh_task_manager = self._mesh_task_manager_factory()
        self.mesh = None
        self.meshes: dict[str, Any] = {}
        self.mesh_record_id: str | None = None
        self.solution = None
        self.solutions: dict[str, Any] = {}
        self.result_datasets: dict[str, Any] = {}
        self.submitted_input_reports: dict[str, str] = {}
        self.active_job_id: str | None = None
        self.analysis = "Linear static"
        self.shape_index = 0
        self.imported = imported
        self.path = path
        self.seeding_overrides: dict[int, int] = {}
        self.view_mode = "geometry"
        self.geometry_selection_mode = "vertex"

    def snapshot(self) -> WorkbenchSnapshot:
        return WorkbenchSnapshot(
            document_id=str(self.project.document_id),
            revision=self.session.revision,
            view_mode=str(self.view_mode),
            selection_mode=str(self.selection.mode),
            selected_count=len(self.selection),
            active_job_id=self.active_job_id,
            active_mesh_record_id=self.mesh_record_id,
            has_mesh=self.mesh is not None,
            has_solution=self.solution is not None,
            project_path=self.path,
        )

    def add_listener(self, listener: WorkbenchListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: WorkbenchListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _publish(self, kind: str, field: str | None = None) -> None:
        event = WorkbenchEvent(str(kind), field, self.snapshot())
        for listener in tuple(self._listeners):
            listener(event)

    def update(self, field: str, value: Any) -> None:
        """Update one frontend-visible state field and notify observers."""

        if field not in self.MUTABLE_STATE_FIELDS:
            raise AttributeError(
                f"{field!r} is not mutable workbench state; use the "
                "controller's project/session lifecycle methods"
            )
        setattr(self, field, value)
        self._publish("state_changed", field)

    def execute(self, command: Command, *, solver_affecting: bool = True) -> Any:
        result = self.session.execute(command, solver_affecting=solver_affecting)
        self._publish("command_executed")
        return result

    def execute_many(
        self,
        commands: Iterable[Command],
        *,
        label: str = "batch edit",
        solver_affecting: bool = True,
    ) -> list[Any]:
        pending = list(commands)
        if not pending:
            return []
        result = self.session.execute_many(
            pending,
            label=label,
            solver_affecting=solver_affecting,
        )
        self._publish("commands_executed")
        return result

    def replace_project(
        self,
        project: Project,
        *,
        path: Path | None = None,
        imported: Any = None,
        read_only: bool = False,
    ) -> None:
        self._release_background()
        self._install_project(
            project,
            path=path,
            imported=imported,
            read_only=read_only,
        )
        self._publish("project_replaced")

    def set_view_mode(self, mode: str) -> None:
        self.update("view_mode", str(mode))

    def presenter(self, task: str, *, status=None):
        from .presenters import TaskPresenter

        return TaskPresenter(self, task, status=status)

    def close(self) -> None:
        """Release headless background resources owned by the workbench."""

        self._release_background()
        self._publish("closed")

    def _release_background(self) -> None:
        cancel = getattr(self.job_manager, "cancel", None)
        if callable(cancel):
            active = getattr(self.job_manager, "active_job_id", None)
            if active is not None:
                cancel(active)
            for queued in tuple(getattr(self.job_manager, "queued", ())):
                cancel(queued)
        # Both replacement and close release ownership of this manager.
        # Noncooperative process work must not outlive that ownership.
        shutdown = getattr(self.job_manager, "shutdown", None)
        if callable(shutdown):
            shutdown()
        shutdown = getattr(self.mesh_task_manager, "shutdown", None)
        if callable(shutdown):
            shutdown()
