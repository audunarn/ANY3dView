"""Toolkit host adapters for concrete viewer widgets.

The rendering and interaction contracts in :mod:`any3dview` do not prescribe
which desktop toolkit owns the native window.  A host adapter is the one small
piece that creates a toolkit widget and decides how its GPU/software fallback
works.  Importing this module deliberately imports no desktop toolkit.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from .errors import GPUUnavailableError

__all__ = ["ViewerHostAdapter", "TkViewerHostAdapter"]


@runtime_checkable
class ViewerHostAdapter(Protocol):
    """Create a viewer widget owned by one desktop toolkit.

    Qt, web, or another host can implement this protocol without changing the
    application or the renderer-neutral :class:`~any3dview.ViewerBackend`
    contract.  The returned object must satisfy that backend contract.
    """

    @property
    def toolkit_name(self) -> str: ...

    def create_viewer(
        self,
        parent: object,
        *,
        backend: str,
        **options: Any,
    ) -> object: ...


class TkViewerHostAdapter:
    """Default lazy Tk host for the bundled GPU and software viewers."""

    toolkit_name = "tk"

    def create_viewer(
        self,
        parent: object,
        *,
        backend: str,
        **options: Any,
    ) -> object:
        diagnostics: list[str] = []
        disabled = os.environ.get("ANY3DVIEW_DISABLE_GPU", "").casefold() in {
            "1",
            "true",
            "yes",
        }
        if disabled and backend == "gpu":
            raise GPUUnavailableError(
                "explicit GPU backend initialization was disabled",
                diagnostics=("ANY3DVIEW_DISABLE_GPU is enabled",),
            )
        if disabled:
            diagnostics.append("ANY3DVIEW_DISABLE_GPU is enabled")
        if backend in {"auto", "gpu"} and not disabled:
            try:
                from .gpu import Any3DView

                return Any3DView(parent, **options)
            except Exception as error:
                if isinstance(error, GPUUnavailableError):
                    diagnostics.extend(error.diagnostics)
                else:
                    diagnostics.extend((type(error).__name__, str(error)))
                if backend == "gpu":
                    raise GPUUnavailableError(
                        "explicit GPU backend initialization failed",
                        diagnostics=tuple(diagnostics),
                    ) from error
        try:
            from anytk3d import Tkinter3DCanvas
        except ImportError as error:
            diagnostics.append("ANYtk3D is not installed")
            raise GPUUnavailableError(
                "no viewer backend is available",
                diagnostics=tuple(diagnostics),
            ) from error
        viewer = Tkinter3DCanvas(parent, **options)
        viewer._backend_diagnostics = tuple(diagnostics)
        return viewer
