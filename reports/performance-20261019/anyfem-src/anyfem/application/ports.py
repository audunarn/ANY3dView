"""Ports implemented by a desktop-toolkit adapter.

These deliberately describe behavior rather than Tk, Qt, or widget classes.
The application and presenters can therefore be qualified with small
headless fakes; only the outermost frontend imports a concrete toolkit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

__all__ = [
    "ClipboardPort",
    "DialogPort",
    "SchedulerPort",
    "StatusMessage",
    "StatusPort",
    "ViewportPort",
]


@dataclass(frozen=True, slots=True)
class StatusMessage:
    text: str
    error: bool = False
    diagnostic: Mapping[str, Any] | None = None


@runtime_checkable
class SchedulerPort(Protocol):
    def call_later(
        self, delay_ms: int, callback: Callable[[], Any]
    ) -> object: ...

    def cancel_call(self, identifier: object) -> None: ...


@runtime_checkable
class DialogPort(Protocol):
    def show_error(self, title: str, message: str) -> None: ...

    def confirm(self, title: str, message: str) -> bool: ...

    def confirm_save(self, title: str, message: str) -> bool | None: ...

    def ask_text(
        self, title: str, prompt: str, *, initial: str = ""
    ) -> str | None: ...

    def open_file(self, **options: Any) -> str: ...

    def save_file(self, **options: Any) -> str: ...


@runtime_checkable
class ClipboardPort(Protocol):
    def copy_text(self, text: str) -> None: ...


@runtime_checkable
class StatusPort(Protocol):
    def publish(self, message: StatusMessage) -> None: ...


@runtime_checkable
class ViewportPort(Protocol):
    """Minimum viewport behavior needed by an application presenter."""

    @property
    def active_backend(self) -> str: ...

    def show(self, scene: object, **options: Any) -> None: ...

    def fit(self) -> None: ...

    def switch_backend(self, backend: str) -> str: ...

    def capture_png(self, path: str | Path) -> Path: ...
