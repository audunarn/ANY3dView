"""Small toolkit-neutral task presenters."""

from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Any, Iterable

from ..commands import Command
from ..selection import mode_label
from .ports import StatusMessage, StatusPort

__all__ = ["TaskPresenter", "TaskViewModel"]


@dataclass(frozen=True, slots=True)
class TaskViewModel:
    task: str
    project_name: str
    view_mode: str
    selection_mode: str
    selected_count: int
    busy: bool
    has_mesh: bool
    has_solution: bool


class TaskPresenter:
    """Common command, selection and error behavior for one task panel."""

    def __init__(self, controller, task: str, *, status: StatusPort | None = None):
        self.controller = controller
        self.task = str(task)
        self.status = status

    @property
    def project(self):
        return self.controller.project

    @property
    def selection(self):
        return self.controller.selection

    def view_model(self) -> TaskViewModel:
        state = self.controller.snapshot()
        return TaskViewModel(
            task=self.task,
            project_name=str(self.controller.project.name),
            view_mode=state.view_mode,
            selection_mode=state.selection_mode,
            selected_count=state.selected_count,
            busy=bool(
                getattr(self.controller.job_manager, "running", False)
                or getattr(self.controller.job_manager, "queued", ())
                or getattr(self.controller.mesh_task_manager, "busy", False)
            ),
            has_mesh=state.has_mesh,
            has_solution=state.has_solution,
        )

    def execute(self, command: Command) -> Any:
        return self.controller.execute(command)

    def execute_many(self, commands: Iterable[Command]) -> list[Any]:
        return self.controller.execute_many(commands)

    def require_selection(self, kind: str, count: int | None = None) -> list[Any]:
        selection = self.selection
        if selection.mode != kind:
            raise ValueError(
                f"switch to {mode_label(kind)} mode first "
                f"(currently {mode_label(selection.mode)})"
            )
        items = list(selection.items)
        if not items:
            raise ValueError(
                f"select at least one {mode_label(kind).lower()} first"
            )
        if count is not None and len(items) != count:
            raise ValueError(
                f"select exactly {count} {mode_label(kind).lower()}s "
                f"({len(items)} selected)"
            )
        return items

    def report_error(self, error: BaseException) -> None:
        if self.status is None:
            return
        self.status.publish(
            StatusMessage(
                f"{type(error).__name__}: {error}",
                error=True,
                diagnostic={
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        )
