"""Tk frontend adapter.

The established :mod:`anyfem.ui.app` module remains a compatibility surface;
new launchers use this explicit toolkit namespace.
"""

from ..app import AnyFemApp, default_project, main
from ..tk_adapters import (
    CallbackStatusPort,
    TkClipboardPort,
    TkDialogPort,
    TkSchedulerPort,
)

__all__ = [
    "AnyFemApp",
    "CallbackStatusPort",
    "TkClipboardPort",
    "TkDialogPort",
    "TkSchedulerPort",
    "default_project",
    "main",
]
