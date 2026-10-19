"""On-demand Qt runtime snapshot appended to the copied support report.

The snapshot is built only when a diagnosis is actually copied: this module
installs no timers, threads, hooks or tracing, and reads nothing at startup or
during interaction.  Every field is individually guarded so a snapshot failure
degrades to an explicit bounded "unavailable" note instead of breaking the
report.  Nothing here enumerates operating-system processes, reads live
renderer state (``ctx.info``, make-current, rendering or image grabs),
environment variables, credentials or model arrays.  Window counting reads
only Qt's own lazy top-level widget list (never a ``gc`` scan of every Python
object), and the optional viewer contribution comes only from the viewer
owner's cached ``diagnostic_snapshot()`` on the canvas.
"""
from __future__ import annotations

import json
import os
import sys
import time
from itertools import islice
from typing import Any, Callable

__all__ = ["extend_report_with_qt_runtime", "qt_runtime_snapshot"]

_TEXT_LIMIT = 200
_SAFE_DEPTH = 8
_COLLECTION_LIMIT = 100


def _truncate(text: str) -> str:
    """Bound any diagnostic text so one field cannot flood the report."""
    if len(text) <= _TEXT_LIMIT:
        return text
    return text[: _TEXT_LIMIT - 3] + "..."


def _bounded_reason(error: BaseException) -> str:
    return _truncate(f"{type(error).__name__}: {error}")


def _unavailable(error: BaseException) -> str:
    return f"unavailable ({_bounded_reason(error)})"


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Keep one copied report serializable whatever the viewer returns.

    Long strings are truncated and collections are bounded with ``islice``, so
    no whole list/dict is copied before the first bounded items, and no full
    model array or huge viewer object repr can reach the report.
    """
    if depth > _SAFE_DEPTH:
        return "<maximum diagnostic depth reached>"
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth=depth + 1)
                for key, item in islice(value.items(), _COLLECTION_LIMIT)}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item, depth=depth + 1)
                for item in islice(value, _COLLECTION_LIMIT)]
    return _truncate(repr(value))


def _read_pid() -> int:
    return os.getpid()


def _read_qt_versions() -> dict[str, str]:
    """PySide6 and Qt runtime versions; Qt is imported lazily and never here."""
    versions: dict[str, str] = {}
    try:
        from importlib import metadata

        versions["pyside6"] = str(metadata.version("PySide6"))
    except Exception as error:
        versions["pyside6"] = _unavailable(error)
    try:
        from PySide6.QtCore import qVersion

        versions["qt_runtime"] = str(qVersion())
    except Exception as error:
        versions["qt_runtime"] = _unavailable(error)
    return versions


def _read_system_uptime() -> dict[str, Any]:
    """Time since boot from one read-only platform source, or an explicit reason."""
    if sys.platform == "win32":
        try:
            import ctypes

            gettickcount64 = ctypes.windll.kernel32.GetTickCount64
            # ctypes defaults to a signed 32-bit int restype, which truncates
            # and turns negative once the host has been up past ~24.8 days.
            # Configure exactly this function: unsigned 64-bit, no arguments.
            gettickcount64.restype = ctypes.c_uint64
            gettickcount64.argtypes = []
            uptime_ms = int(gettickcount64())
            return {"uptime_ms": uptime_ms, "source": "GetTickCount64"}
        except Exception as error:
            return {"uptime_ms": None, "unavailable": _bounded_reason(error)}
    boottime_clock = getattr(time, "CLOCK_BOOTTIME", None)
    if boottime_clock is not None:
        try:
            uptime_ms = int(time.clock_gettime(boottime_clock) * 1000)
            return {"uptime_ms": uptime_ms, "source": "CLOCK_BOOTTIME"}
        except Exception as error:
            return {"uptime_ms": None, "unavailable": _bounded_reason(error)}
    return {"uptime_ms": None,
            "unavailable": f"no safe uptime source on {sys.platform}"}


def _read_window_state(window: Any) -> dict[str, Any]:
    """Cached workbench geometry/DPR plus the actual canvas event_widget surface.

    ``devicePixelRatioF`` is used on both the window and the event_widget so
    fractional high-DPI ratios are preserved; the integer legacy
    ``devicePixelRatio`` would round them away.  Only cached widget geometry is
    read: no rendering, grab or make-current happens here.
    """
    state: dict[str, Any] = {}
    for key, read in (
        ("size", lambda: [int(window.width()), int(window.height())]),
        ("device_pixel_ratio", lambda: float(window.devicePixelRatioF())),
        ("visible", lambda: bool(window.isVisible())),
        ("minimized", lambda: bool(window.isMinimized())),
    ):
        try:
            state[key] = read()
        except Exception as error:
            state[key] = _unavailable(error)
    surface = getattr(getattr(getattr(window, "viewport", None), "canvas", None),
                      "event_widget", None)
    if surface is None:
        note = "unavailable (window has no canvas event_widget surface)"
        state["event_widget_size"] = note
        state["event_widget_device_pixel_ratio"] = note
        return state
    for key, read in (
        ("event_widget_size", lambda: [int(surface.width()), int(surface.height())]),
        ("event_widget_device_pixel_ratio",
         lambda: float(surface.devicePixelRatioF())),
    ):
        try:
            state[key] = read()
        except Exception as error:
            state[key] = _unavailable(error)
    return state


def _read_viewer_info(window: Any) -> dict[str, Any]:
    """Backends from the viewport; the canvas viewer's own cached snapshot.

    ``window.viewport`` is the presentation viewport: ``requested_backend`` and
    ``active_backend`` are read there, while the optional cached
    ``diagnostic_snapshot()`` belongs to the ANY3dView viewer reached as
    ``viewport.canvas``.
    """
    viewport = getattr(window, "viewport", None)
    if viewport is None:
        return {"unavailable": "window has no viewport"}
    info: dict[str, Any] = {"viewport_class": type(viewport).__name__}
    for key in ("requested_backend", "active_backend"):
        try:
            info[key] = str(getattr(viewport, key))
        except Exception as error:
            info[key] = _unavailable(error)
    canvas = getattr(viewport, "canvas", None)
    if canvas is None:
        note = "unavailable (viewport has no canvas viewer)"
        info["viewer_class"] = note
        info["diagnostic_snapshot"] = note
        return info
    info["viewer_class"] = type(canvas).__name__
    snapshot = getattr(canvas, "diagnostic_snapshot", None)
    if not callable(snapshot):
        info["diagnostic_snapshot"] = "unavailable (viewer provides no diagnostic_snapshot)"
        return info
    try:
        info["diagnostic_snapshot"] = _json_safe(snapshot())
    except Exception as error:
        info["diagnostic_snapshot"] = _unavailable(error)
    return info


def _qt_toplevel_widgets() -> list[Any] | None:
    """Qt's live top-level widget list, or None without a QApplication.

    Qt is imported lazily here so importing this module stays Qt-free, and no
    QApplication is ever constructed: this is an on-demand lightweight
    diagnosis while responsiveness may be impaired.
    """
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance()
    if application is None:
        return None
    return application.topLevelWidgets()


def _count_workbenches(window: Any) -> int:
    """Live workbench windows of the same type in this process only.

    Reads only Qt's own lazy top-level widget list filtered to the workbench
    type.  No ``gc.get_objects()`` scan of every Python object (which would
    walk the whole model/scene graph and count retained closed wrappers) and
    no operating-system process enumeration.
    """
    widgets = _qt_toplevel_widgets()
    if widgets is None:
        raise RuntimeError("no QApplication instance in this process")
    window_type = type(window)
    return sum(1 for widget in widgets if isinstance(widget, window_type))


def qt_runtime_snapshot(
    window: Any,
    *,
    pid_reader: Callable[[], int] | None = None,
    versions_reader: Callable[[], dict] | None = None,
    uptime_reader: Callable[[], dict] | None = None,
    workbench_counter: Callable[[Any], int] | None = None,
    viewer_reader: Callable[[Any], dict] | None = None,
) -> dict[str, Any]:
    """One compact, read-only Qt runtime snapshot for the support report."""
    readers = (
        ("pid", lambda: int((pid_reader or _read_pid)())),
        ("qt_versions", versions_reader or _read_qt_versions),
        ("system_uptime", uptime_reader or _read_system_uptime),
        ("window", lambda: _read_window_state(window)),
        ("viewer", lambda: (viewer_reader or _read_viewer_info)(window)),
        ("workbenches_in_process", lambda: int((workbench_counter or _count_workbenches)(window))),
    )
    snapshot: dict[str, Any] = {}
    for key, read in readers:
        try:
            snapshot[key] = read()
        except Exception as error:
            snapshot[key] = _unavailable(error)
    return snapshot


def extend_report_with_qt_runtime(report: str, window: Any) -> str:
    """Parse the inherited report and add one ``qt_runtime`` section.

    The inherited serialization is not duplicated: the report is parsed, one
    key is added and the payload is re-serialized with the same formatting.
    Any failure returns the inherited report unchanged, so a snapshot problem
    can never break Copy diagnosis or drop its error/project/context fields.
    """
    try:
        payload = json.loads(report)
    except ValueError:
        return report
    if not isinstance(payload, dict):
        return report
    try:
        payload["qt_runtime"] = _json_safe(qt_runtime_snapshot(window))
    except Exception as error:
        payload["qt_runtime"] = {"unavailable": _bounded_reason(error)}
    try:
        return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    except Exception:
        return report
