"""Permissive Python host for the native TkGL widget.

The bundled native extension is the separately licensed TkGL project.  This
module is an original wrapper around TkGL's public Tcl widget commands; it
does not contain code from the GPL-licensed ``tkinter-gl`` Python package.
"""

from __future__ import annotations

import platform
from pathlib import Path
import sys
import tkinter as tk
from typing import Any, Optional


_ASSET_ROOT = Path(__file__).with_name("tkgl")


def _platform_package_directory() -> Path:
    if sys.platform == "win32":
        platform_name = "win32"
    elif sys.platform == "darwin":
        platform_name = "darwin"
    elif sys.platform == "linux":
        platform_name = f"linux-{platform.machine().lower()}"
    else:
        raise RuntimeError(f"TkGL is not packaged for platform {sys.platform!r}")

    directory = _ASSET_ROOT / platform_name
    if not directory.is_dir():
        raise RuntimeError(
            f"TkGL native package is unavailable for {platform_name!r}: {directory}"
        )
    return directory


class GLCanvas(tk.Widget):
    """Tk widget backed by the bundled, permissively licensed TkGL extension."""

    profile = ""

    def __init__(
        self,
        parent: tk.Misc,
        cnf: Optional[dict[str, Any]] = None,
        **options: Any,
    ) -> None:
        # TkGL needs a native parent window before it creates its child surface.
        if sys.platform == "win32":
            parent.update()

        package_directory = _platform_package_directory()
        parent.tk.call("lappend", "auto_path", str(package_directory))
        parent.tk.call("package", "require", "Tkgl")

        widget_options = dict(options)
        if self.profile:
            widget_options["profile"] = self.profile
        tk.Widget.__init__(self, parent, "tkgl", cnf or {}, widget_options)
        self.bind("<Expose>", self._queue_draw, add="+")
        self.bind("<Map>", self._queue_draw, add="+")
        self.update_idletasks()

    def _queue_draw(self, _event: tk.Event) -> None:
        self.after_idle(self.draw)

    def make_current(self) -> None:
        self.tk.call(self._w, "makecurrent")

    def swap_buffers(self) -> None:
        self.tk.call(self._w, "swapbuffers")

    def gl_version(self) -> str:
        return str(self.tk.call(self._w, "glversion"))

    def gl_extensions(self) -> str:
        return str(self.tk.call(self._w, "extensions"))

    def draw(self) -> None:
        """Render callback overridden by the owning viewer surface."""

