"""Tk implementations of ANYfem's desktop-toolkit ports."""

from __future__ import annotations

from typing import Any, Callable

from ..application.ports import StatusMessage

__all__ = [
    "CallbackStatusPort",
    "TkClipboardPort",
    "TkDialogPort",
    "TkSchedulerPort",
]


class TkSchedulerPort:
    def __init__(self, widget) -> None:
        self.widget = widget

    def call_later(self, delay_ms: int, callback: Callable[[], Any]) -> object:
        return self.widget.after(int(delay_ms), callback)

    def cancel_call(self, identifier: object) -> None:
        self.widget.after_cancel(identifier)


class TkDialogPort:
    """Dialog adapter with injected modules for straightforward UI tests."""

    def __init__(
        self,
        owner: Callable[[], object],
        *,
        messagebox,
        filedialog,
        simpledialog,
    ) -> None:
        self._owner = owner
        self._messagebox = messagebox
        self._filedialog = filedialog
        self._simpledialog = simpledialog

    def show_error(self, title: str, message: str) -> None:
        self._messagebox.showerror(title, message, parent=self._owner())

    def confirm(self, title: str, message: str) -> bool:
        return bool(
            self._messagebox.askyesno(title, message, parent=self._owner())
        )

    def confirm_save(self, title: str, message: str) -> bool | None:
        return self._messagebox.askyesnocancel(
            title, message, parent=self._owner()
        )

    def ask_text(
        self, title: str, prompt: str, *, initial: str = ""
    ) -> str | None:
        return self._simpledialog.askstring(
            title,
            prompt,
            initialvalue=initial,
            parent=self._owner(),
        )

    def open_file(self, **options: Any) -> str:
        return str(self._filedialog.askopenfilename(**options) or "")

    def save_file(self, **options: Any) -> str:
        return str(self._filedialog.asksaveasfilename(**options) or "")


class TkClipboardPort:
    def __init__(self, widget) -> None:
        self.widget = widget

    def copy_text(self, text: str) -> None:
        self.widget.clipboard_clear()
        self.widget.clipboard_append(str(text))
        self.widget.update_idletasks()


class CallbackStatusPort:
    def __init__(self, callback: Callable[..., Any]) -> None:
        self.callback = callback

    def publish(self, message: StatusMessage) -> None:
        self.callback(
            message.text,
            error=message.error,
            diagnostic=message.diagnostic,
        )
