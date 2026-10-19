"""Compact, copyable diagnostics for desktop and support workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata, util
import json
import platform
import sys
from typing import Any, Callable, Iterable, Mapping

__all__ = ["ErrorDiagnostic", "build_diagnostic_report"]


_PACKAGES = (
    ("ANYfem", "anyfem"),
    ("ANYsolver", "anysolver"),
    ("ANYmesher", "anymesher"),
    ("ANYgeometry", "anygeometry"),
    ("ANYmaterial", "anymaterial"),
    ("ANYfileio", "anyfileio"),
    ("ANY3dView", "any3dview"),
    ("ANYtk3D", "anytk3d"),
    ("ANYsection", "anysection"),  # optional: standard beam sections
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return "<maximum diagnostic depth reached>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _safe(item, depth=depth + 1)
            for key, item in list(value.items())[:100]
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe(item, depth=depth + 1) for item in list(value)[:100]]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _safe(to_dict(), depth=depth + 1)
        except Exception:
            pass
    try:
        return _safe(asdict(value), depth=depth + 1)
    except (TypeError, ValueError):
        representation = repr(value)
        return representation if len(representation) <= 1000 else representation[:997] + "..."


@dataclass(frozen=True)
class ErrorDiagnostic:
    """One full error captured when the status bar receives a failure."""

    timestamp_utc: str
    message: str
    details: Any = None
    project_id: str = ""
    view: str = ""

    @classmethod
    def capture(
        cls,
        message: str,
        *,
        details: Any = None,
        project_id: str = "",
        view: str = "",
    ) -> "ErrorDiagnostic":
        return cls(
            _utc_now(),
            str(message),
            _safe(details),
            str(project_id),
            str(view),
        )


def _package_rows(
    version_reader: Callable[[str], str],
    origin_reader: Callable[[str], str | None],
) -> list[dict[str, str]]:
    rows = []
    for distribution, module in _PACKAGES:
        try:
            version = str(version_reader(distribution))
        except Exception:
            version = "metadata unavailable"
        try:
            origin = str(origin_reader(module) or "not importable")
        except Exception:
            origin = "origin unavailable"
        rows.append(
            {
                "distribution": distribution,
                "version": version,
                "module": module,
                "origin": origin,
            }
        )
    return rows


def _default_origin(module: str) -> str | None:
    specification = util.find_spec(module)
    return None if specification is None else specification.origin


def _project_summary(project: Any) -> dict[str, Any]:
    geometry = project.geometry
    load_count = sum(
        len(getattr(case, name, ()))
        for case in project.load_cases.values()
        for name in (
            "point_loads", "pressures", "line_loads", "surface_tractions",
            "connection_loads",
        )
    )
    settings = getattr(project, "native_mesh_settings", None)
    return {
        "name": str(getattr(project, "name", "")),
        "document_id": str(getattr(project, "document_id", "")),
        "read_only_reason": getattr(project, "read_only_reason", None),
        "units": _safe(getattr(project, "units", None)),
        "geometry": {
            "revision": int(getattr(geometry, "revision", 0)),
            "points": len(geometry.vertices),
            "lines": len(geometry.edges),
            "plates": len(geometry.faces),
            "sheets": len(getattr(geometry, "sheets", {})),
            "members": len(getattr(geometry, "members", {})),
            "features": len(getattr(getattr(geometry, "features", None), "records", ())),
        },
        "definitions": {
            "materials": len(project.materials),
            "plate_sections": len(project.plate_sections),
            "beam_sections": len(project.beam_sections),
            "regions": len(project.regions),
            "load_cases": len(project.load_cases),
            "loads": load_count,
            "supports": len(project.supports),
            "imperfections": len(project.imperfections),
        },
        "records": {
            "meshes": len(project.mesh_records),
            "analyses": len(project.analyses),
            "jobs": len(project.jobs),
            "artifacts": len(project.artifacts),
        },
        "mesh_settings": None if settings is None else _safe(settings),
    }


def build_diagnostic_report(
    project: Any,
    *,
    errors: Iterable[ErrorDiagnostic] = (),
    context: Mapping[str, Any] | None = None,
    recent_commands: Iterable[str] = (),
    version_reader: Callable[[str], str] = metadata.version,
    origin_reader: Callable[[str], str | None] = _default_origin,
) -> str:
    """Build one plain-text report suitable for clipboard issue reporting."""

    captured = tuple(errors)[-10:]
    payload = {
        "report": "ANYfem diagnostic",
        "generated_utc": _utc_now(),
        "errors": [asdict(item) for item in captured],
        "runtime": {
            "python": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": _package_rows(version_reader, origin_reader),
        "project": _project_summary(project),
        "application": _safe(dict(context or {})),
        "recent_gui_commands": tuple(str(item) for item in recent_commands)[-100:],
    }
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
