"""Lazy backend selection without a shared-core import cycle."""

from __future__ import annotations

import os
from typing import Any

from .errors import GPUUnavailableError


def create_viewer(parent: object, backend: str = "auto", **options: Any) -> object:
    """Create the requested Tk viewer and retain automatic-fallback diagnostics."""

    requested = str(backend).casefold()
    if requested not in {"auto", "gpu", "software"}:
        raise ValueError("backend must be 'auto', 'gpu', or 'software'")
    diagnostics: list[str] = []
    disabled = os.environ.get("ANY3DVIEW_DISABLE_GPU", "").casefold() in {
        "1",
        "true",
        "yes",
    }
    if disabled and requested == "gpu":
        raise GPUUnavailableError(
            "explicit GPU backend initialization was disabled",
            diagnostics=("ANY3DVIEW_DISABLE_GPU is enabled",),
        )
    if disabled:
        diagnostics.append("ANY3DVIEW_DISABLE_GPU is enabled")
    if requested in {"auto", "gpu"} and not disabled:
        try:
            from .gpu import Any3DView

            return Any3DView(parent, **options)
        except Exception as error:
            if isinstance(error, GPUUnavailableError):
                diagnostics.extend(error.diagnostics)
            else:
                diagnostics.extend((type(error).__name__, str(error)))
            if requested == "gpu":
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
    viewer._backend_diagnostics = tuple(diagnostics)  # compatibility backend diagnostic
    return viewer
