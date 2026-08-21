"""Tk OpenGL host abstraction."""

from __future__ import annotations

from typing import Callable, Protocol

try:
    from tkinter_gl import GLCanvas
except ImportError as error:  # pragma: no cover - exercised by isolated wheel tests
    raise ImportError(
        "ANY3dView GPU support requires the 'gpu' extra: pip install ANY3dView[gpu]"
    ) from error


class GLHostProtocol(Protocol):
    def make_current(self) -> None: ...
    def swap_buffers(self) -> None: ...
    def framebuffer_size(self) -> tuple[int, int]: ...
    def request_redraw(self) -> None: ...
    def close(self) -> None: ...


class _Surface(GLCanvas):
    # tkinter-gl 1.1 exposes 2.1, 3.2 and 4.1 profile requests.  The renderer
    # requires GLSL/OpenGL 3.3, so 4.1 is the first host profile that satisfies
    # the contract; ModernGL still enforces the actual 3.3 minimum below it.
    profile = "4_1"

    def __init__(self, parent, draw_callback: Callable[[], None], **options):
        self._draw_callback = draw_callback
        super().__init__(parent, **options)

    def draw(self) -> None:
        self._draw_callback()


class TkinterGLHost:
    """GL 3.3+ context hosted in an ordinary Tk widget hierarchy."""

    def __init__(self, parent, draw_callback: Callable[[], None], **options) -> None:
        self._draw_callback = draw_callback
        self._redraw_pending = False
        self._retry_id = None
        self.surface = _Surface(parent, self._draw_surface, **options)

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
            if self._redraw_pending and self.surface.winfo_exists():
                self.surface.draw()

        self.surface.after_idle(draw)

    def close(self) -> None:
        if self._retry_id is not None:
            try:
                self.surface.after_cancel(self._retry_id)
            except Exception:
                pass
            self._retry_id = None
        self._redraw_pending = False
        if self.surface.winfo_exists():
            self.surface.destroy()
