"""Toolkit-neutral lazy viewer creation."""

from __future__ import annotations

from typing import Any

from .hosts import TkViewerHostAdapter, ViewerHostAdapter


def create_viewer(
    parent: object,
    backend: str = "auto",
    *,
    host: ViewerHostAdapter | None = None,
    **options: Any,
) -> object:
    """Create a viewer through the selected desktop-toolkit host adapter.

    Omitting ``host`` preserves the historical Tk behavior.  Supplying an
    adapter makes widget ownership explicit and allows applications to use a
    different toolkit without changing renderer-neutral scenes or commands.
    """

    requested = str(backend).casefold()
    if requested not in {"auto", "gpu", "software"}:
        raise ValueError("backend must be 'auto', 'gpu', or 'software'")
    adapter = TkViewerHostAdapter() if host is None else host
    if not isinstance(adapter, ViewerHostAdapter):
        raise TypeError("host must implement ViewerHostAdapter")
    return adapter.create_viewer(parent, backend=requested, **options)
