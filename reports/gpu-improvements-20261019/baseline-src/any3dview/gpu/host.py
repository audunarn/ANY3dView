"""Tk OpenGL host abstraction."""

from __future__ import annotations

import os
import sys
from typing import Callable, Protocol

from ._tkgl import GLCanvas


class GLHostProtocol(Protocol):
    def make_current(self) -> None: ...
    def swap_buffers(self) -> None: ...
    def framebuffer_size(self) -> tuple[int, int]: ...
    def request_redraw(self) -> None: ...
    def cancel_redraw(self) -> None: ...
    def close(self) -> None: ...


def _gl_profile() -> str:
    """Return the TkGL profile selected for this process.

    Windows drivers have proved substantially safer with TkGL's core profile
    than with the highest-version legacy compatibility context.  In
    particular, recent NVIDIA drivers can terminate the process inside
    ``nvoglv64.dll`` while servicing the compatibility context, which cannot
    be caught by Python.  Keep the legacy profile on other platforms and as
    an explicit escape hatch for Windows hardware limited to OpenGL 3.3/4.0.
    """

    requested = os.environ.get("ANY3DVIEW_GL_PROFILE", "").strip().casefold()
    aliases = {
        "": "4_1" if sys.platform == "win32" else "legacy",
        "core": "4_1",
        "4.1": "4_1",
        "4_1": "4_1",
        "legacy": "legacy",
    }
    try:
        return aliases[requested]
    except KeyError as error:
        raise ValueError(
            "ANY3DVIEW_GL_PROFILE must be 'core', '4.1', or 'legacy'"
        ) from error


class _Surface(GLCanvas):
    profile = ""

    def __init__(self, parent, draw_callback: Callable[[], None], **options):
        self._draw_callback = draw_callback
        self.profile = _gl_profile()
        super().__init__(parent, **options)

    def draw(self) -> None:
        self._draw_callback()


class TkinterGLHost:
    """GL 3.3+ context hosted in an ordinary Tk widget hierarchy."""

    def __init__(self, parent, draw_callback: Callable[[], None], **options) -> None:
        self._draw_callback = draw_callback
        self._redraw_pending = False
        self._retry_id = None
        self._draw_idle_id = None
        self.surface = _Surface(parent, self._draw_surface, **options)
        self.profile = self.surface.profile

    def _draw_surface(self) -> None:
        """Draw only after Tk has mapped a non-trivial framebuffer."""

        if not self.surface.winfo_exists():
            self._redraw_pending = False
            return
        width, height = self.framebuffer_size()
        if not self.surface.winfo_ismapped() or width <= 1 or height <= 1:
            self._redraw_pending = True
            if self._retry_id is None:
                self._retry_id = self.surface.after(16, self._run_requested_draw)
            return
        self._redraw_pending = False
        self._draw_callback()

    def _run_requested_draw(self) -> None:
        self._retry_id = None
        if self._redraw_pending and self.surface.winfo_exists():
            self.surface.draw()

    def make_current(self) -> None:
        self.surface.make_current()

    def swap_buffers(self) -> None:
        self.surface.swap_buffers()

    def framebuffer_size(self) -> tuple[int, int]:
        return max(1, self.surface.winfo_width()), max(1, self.surface.winfo_height())

    def request_redraw(self) -> None:
        if self._redraw_pending:
            return
        self._redraw_pending = True

        def draw() -> None:
            self._draw_idle_id = None
            if self._redraw_pending and self.surface.winfo_exists():
                self.surface.draw()

        self._draw_idle_id = self.surface.after_idle(draw)

    def cancel_redraw(self) -> None:
        """Cancel pending demand work without changing the last framebuffer."""

        self._redraw_pending = False
        if self._draw_idle_id is not None:
            try:
                self.surface.after_cancel(self._draw_idle_id)
            except Exception:
                pass
            self._draw_idle_id = None
        if self._retry_id is not None:
            try:
                self.surface.after_cancel(self._retry_id)
            except Exception:
                pass
            self._retry_id = None

    def close(self) -> None:
        self.cancel_redraw()
        if self.surface.winfo_exists():
            self.surface.destroy()
