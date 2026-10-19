"""Toolkit-neutral desktop application state and presentation contracts."""

from .controller import (
    WorkbenchController,
    WorkbenchEvent,
    WorkbenchSnapshot,
    default_project,
)
from .ports import (
    ClipboardPort,
    DialogPort,
    SchedulerPort,
    StatusMessage,
    StatusPort,
    ViewportPort,
)
from .presenters import TaskPresenter, TaskViewModel

__all__ = [
    "ClipboardPort",
    "DialogPort",
    "SchedulerPort",
    "StatusMessage",
    "StatusPort",
    "TaskPresenter",
    "TaskViewModel",
    "ViewportPort",
    "WorkbenchController",
    "WorkbenchEvent",
    "WorkbenchSnapshot",
    "default_project",
]
