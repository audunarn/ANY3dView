"""Persisted submodel definitions and runs (workflow contract, section 11).

A definition names the source job and case, the child project (detached in v1),
the lineage, the transfer policy and the child case.  A run is one successful
submodel solve with an immutable dependency manifest; it is kept when its
definition or artifacts later disappear, so staleness can report them.

These records are plain data.  The run itself, the job-bound source and
staleness live in :mod:`anyfem.submodel_jobs`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Dict, Mapping, Optional, Tuple
from uuid import uuid4

SCHEMA_VERSION = 1
POLICIES = ("drive_all", "release_normal_rotation")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class SubmodelRecordError(ValueError):
    """A submodel definition or run record is malformed."""


def _uuid() -> str:
    return str(uuid4())


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise SubmodelRecordError(f"{context} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SubmodelRecordError(f"{context} must be a number") from None
    if not math.isfinite(number):
        raise SubmodelRecordError(f"{context} must be finite")
    return number


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise SubmodelRecordError(f"{context} must be a non-empty string")
    return value


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SubmodelRecordError(f"{context} must be an object")
    return value


def _schema(data: Mapping[str, Any], context: str) -> int:
    version = data.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise SubmodelRecordError(f"{context}.schema_version must be a positive integer")
    if version > SCHEMA_VERSION:
        raise SubmodelRecordError(f"{context} has schema {version}; this ANYfem reads up to {SCHEMA_VERSION}")
    return version


# ----------------------------------------------------------------------
# definition
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SourceReference:
    """The source job and the load case it solved."""

    job_id: str
    case: str

    def to_dict(self) -> Dict[str, Any]:
        return {"job_id": self.job_id, "case": self.case}

    @classmethod
    def from_dict(cls, data: Any, context: str = "source") -> "SourceReference":
        data = _mapping(data, context)
        return cls(_text(data.get("job_id"), f"{context}.job_id"), _text(data.get("case"), f"{context}.case"))


def _volume(data: Any, context: str = "volume") -> Dict[str, Any]:
    data = _mapping(data, context)
    if data.get("kind") != "box":
        raise SubmodelRecordError(f"{context}.kind must be 'box'")
    corners = []
    for name in ("lower", "upper"):
        values = data.get(name)
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise SubmodelRecordError(f"{context}.{name} must have three coordinates")
        corners.append([_finite(v, f"{context}.{name}") for v in values])
    if not all(a < b for a, b in zip(*corners)):
        raise SubmodelRecordError(f"{context}: lower must be below upper on every axis")
    transform = data.get("transform")
    if transform is not None:
        if not isinstance(transform, (list, tuple)) or len(transform) != 4 or any(
                not isinstance(row, (list, tuple)) or len(row) != 4 for row in transform):
            raise SubmodelRecordError(f"{context}.transform must be a 4x4 matrix or null")
        matrix = [[_finite(v, f"{context}.transform") for v in row] for row in transform]
        rotation = [row[:3] for row in matrix[:3]]
        for i in range(3):
            for j in range(3):
                dot = sum(rotation[k][i] * rotation[k][j] for k in range(3))
                if abs(dot - (1.0 if i == j else 0.0)) > 1.0e-9:
                    raise SubmodelRecordError(f"{context}.transform must be rigid")
        determinant = (
            rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
            - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
            + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
        )
        if abs(determinant - 1.0) > 1.0e-9 or matrix[3] != [0.0, 0.0, 0.0, 1.0]:
            raise SubmodelRecordError(f"{context}.transform must be a proper rigid transform")
        transform = matrix
    return {"kind": "box", "lower": corners[0], "upper": corners[1], "transform": transform}


def _tolerances(data: Any, context: str = "tolerances") -> Dict[str, Optional[float]]:
    data = _mapping(data, context)
    if set(data) - {"physical", "natural", "weight"}:
        raise SubmodelRecordError(f"{context} has unknown keys {sorted(set(data) - {'physical', 'natural', 'weight'})}")
    out: Dict[str, Optional[float]] = {}
    physical = data.get("physical")
    out["physical"] = None if physical is None else _finite(physical, f"{context}.physical")
    out["natural"] = _finite(data.get("natural", 1.0e-9), f"{context}.natural")
    out["weight"] = _finite(data.get("weight", 1.0e-9), f"{context}.weight")
    if any(value is not None and value <= 0.0 for value in out.values()):
        raise SubmodelRecordError(f"{context} must be positive")
    return out


def _lineage(data: Any, context: str = "lineage") -> Tuple[Dict[int, Tuple[int, ...]], Dict[str, Tuple[int, ...]]]:
    data = _mapping(data, context)
    source_faces = {}
    for key, faces in _mapping(data.get("source_faces"), f"{context}.source_faces").items():
        try:
            face = int(key)
        except (TypeError, ValueError):
            raise SubmodelRecordError(f"{context}.source_faces key {key!r} is not a face id") from None
        if not isinstance(faces, (list, tuple)) or not faces or any(isinstance(f, bool) or not isinstance(f, int) for f in faces):
            raise SubmodelRecordError(f"{context}.source_faces[{key!r}] must be a non-empty list of face ids")
        source_faces[face] = tuple(int(f) for f in faces)
    cut_faces = {}
    for key, edges in _mapping(data.get("cut_faces"), f"{context}.cut_faces").items():
        if not isinstance(edges, (list, tuple)) or not edges or any(isinstance(e, bool) or not isinstance(e, int) for e in edges):
            raise SubmodelRecordError(f"{context}.cut_faces[{key!r}] must be a non-empty list of edge ids")
        cut_faces[str(key)] = tuple(int(e) for e in edges)
    if not cut_faces:
        raise SubmodelRecordError(f"{context}.cut_faces names no cut face")
    return source_faces, cut_faces


def child_problems(child: Any) -> Tuple[str, ...]:
    """Why ``child`` is not a detached v1 child (empty when it is)."""

    problems = []
    for name in ("jobs", "artifacts", "mesh_records", "analyses", "submodels", "submodel_runs"):
        if getattr(child, name, None):
            problems.append(f"the child has {name}")
    return tuple(problems)


@dataclass
class SubmodelDefinition:
    """One submodel: where it comes from, what it is, and how it is driven."""

    name: str
    volume: Dict[str, Any]
    target_size: float
    policy: str
    source: SourceReference
    child_case: str
    child: Any  # a detached anyfem Project
    source_faces: Dict[int, Tuple[int, ...]]
    cut_faces: Dict[str, Tuple[int, ...]]
    tolerances: Dict[str, Optional[float]] = field(
        default_factory=lambda: {"physical": None, "natural": 1.0e-9, "weight": 1.0e-9})
    order: str = "linear"
    id: str = field(default_factory=_uuid)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        _text(self.id, "submodel.id")
        self.volume = _volume(self.volume)
        size = _finite(self.target_size, "submodel.target_size")
        if size <= 0.0:
            raise SubmodelRecordError("submodel.target_size must be positive")
        self.target_size = size
        if self.order != "linear":
            raise SubmodelRecordError("submodel.order must be 'linear' in v1")
        if self.policy not in POLICIES:
            raise SubmodelRecordError(f"submodel.policy must be one of {POLICIES}")
        if not isinstance(self.source, SourceReference):
            raise SubmodelRecordError("submodel.source must be a SourceReference")
        _text(self.child_case, "submodel.child_case")
        self.tolerances = _tolerances(self.tolerances)
        self.source_faces, self.cut_faces = _lineage({
            "source_faces": {str(k): list(v) for k, v in self.source_faces.items()},
            "cut_faces": {str(k): list(v) for k, v in self.cut_faces.items()},
        })
        problems = child_problems(self.child)
        if problems:
            raise SubmodelRecordError("the child project is not detached: " + "; ".join(problems))
        if self.child_case not in getattr(self.child, "load_cases", {}):
            raise SubmodelRecordError(f"the child has no load case {self.child_case!r}")

    def to_dict(self) -> Dict[str, Any]:
        from ..io.project_file import project_to_dict

        return {
            "schema_version": int(self.schema_version),
            "id": self.id,
            "name": self.name,
            "volume": self.volume,
            "target_size": float(self.target_size),
            "order": self.order,
            "policy": self.policy,
            "tolerances": dict(self.tolerances),
            "source": self.source.to_dict(),
            "child_case": self.child_case,
            "lineage": {
                "source_faces": {str(k): list(v) for k, v in sorted(self.source_faces.items())},
                "cut_faces": {str(k): list(v) for k, v in sorted(self.cut_faces.items())},
            },
            "child": project_to_dict(self.child),
        }

    @classmethod
    def from_dict(cls, data: Any, context: str = "submodel") -> "SubmodelDefinition":
        from ..io.project_file import project_from_dict

        data = _mapping(data, context)
        version = _schema(data, context)
        source_faces, cut_faces = _lineage(data.get("lineage"), f"{context}.lineage")
        name = data.get("name", "")
        if not isinstance(name, str):
            raise SubmodelRecordError(f"{context}.name must be a string")
        definition = cls(
            id=_text(data.get("id"), f"{context}.id"),
            name=name,
            volume=_volume(data.get("volume"), f"{context}.volume"),
            target_size=_finite(data.get("target_size"), f"{context}.target_size"),
            order=str(data.get("order", "")),
            policy=str(data.get("policy", "")),
            tolerances=_tolerances(data.get("tolerances", {}), f"{context}.tolerances"),
            source=SourceReference.from_dict(data.get("source"), f"{context}.source"),
            child_case=_text(data.get("child_case"), f"{context}.child_case"),
            child=project_from_dict(_mapping(data.get("child"), f"{context}.child")),
            source_faces=source_faces,
            cut_faces=cut_faces,
            schema_version=version,
        )
        definition.validate()
        return definition


def definition_hash(entry: Mapping[str, Any]) -> str:
    """Identity of a serialized definition (``SubmodelDefinition.to_dict()``).

    Everything but ``id`` and ``name``; the child contributes its model hash and
    the hash of its serialized loading, so labels and other presentation edits of
    the child do not change it.  Computed from the serialized entry, so a live
    definition and a snapshot document give the same answer.
    """

    from ..document import _model_payload, canonical_hash

    child = entry["child"]
    payload = {key: value for key, value in entry.items() if key not in ("id", "name", "child")}
    payload["child_model_hash"] = canonical_hash(_model_payload(child))
    payload["child_loading_hash"] = canonical_hash({
        "load_cases": child.get("load_cases", []),
        "combinations": child.get("combinations", []),
    })
    return canonical_hash(payload)


# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------
_DESCRIPTOR_KEYS = ("id", "kind", "uri", "sha256", "byte_size", "schema_version")


def descriptor(artifact: Any) -> Dict[str, Any]:
    """The immutable part of an ``ArtifactRef`` (no creation time)."""

    data = artifact.to_dict() if hasattr(artifact, "to_dict") else dict(artifact)
    return {key: data.get(key) for key in _DESCRIPTOR_KEYS}


_BINDING_FIELDS = ("model_hash", "problem_hash", "loading_hash", "mesh_fingerprint", "solver_fingerprint",
                   "operator_fingerprint", "load_fingerprint", "layout_fingerprint", "field_fingerprint", "load_case")
_ARTIFACT_FOLDERS = {"mesh": "meshes", "result": "results"}


def _descriptor(data: Any, context: str, kind: str) -> Dict[str, Any]:
    data = _mapping(data, context)
    out = {
        "id": _text(data.get("id"), f"{context}.id"),
        "kind": _text(data.get("kind"), f"{context}.kind"),
        "uri": _text(data.get("uri"), f"{context}.uri"),
        "sha256": data.get("sha256"),
        "byte_size": data.get("byte_size"),
        "schema_version": data.get("schema_version"),
    }
    if out["kind"] != kind:
        raise SubmodelRecordError(f"{context}.kind must be {kind!r}")
    path = PurePosixPath(out["uri"])
    if ("\\" in out["uri"] or path.is_absolute() or ".." in path.parts or len(path.parts) != 2
            or path.parts[0] != _ARTIFACT_FOLDERS[kind]):
        raise SubmodelRecordError(f"{context}.uri must be a file directly under {_ARTIFACT_FOLDERS[kind]}/")
    if not isinstance(out["sha256"], str) or not _SHA256.match(out["sha256"]):
        raise SubmodelRecordError(f"{context}.sha256 must have the form sha256:<64 lowercase hex digits>")
    for key in ("byte_size", "schema_version"):
        if isinstance(out[key], bool) or not isinstance(out[key], int) or out[key] < 0:
            raise SubmodelRecordError(f"{context}.{key} must be a non-negative integer")
    return out


def json_safe(value: Any) -> Any:
    """Diagnostics as JSON: arrays as lists, keys as strings, nonfinite as strings."""

    import numpy as np

    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return number
    if value is None or isinstance(value, str):
        return value
    return repr(value)


@dataclass
class SubmodelRun:
    """One successful submodel run and everything it depended on."""

    id: str
    definition_id: str
    definition_hash: str
    child_case: str
    source: Dict[str, Any]  # dependency manifest of the source (contract 11.2)
    child: Dict[str, Any]  # child hashes and artifact descriptors
    identity: str
    diagnostics: Dict[str, Any]
    producer_versions: Dict[str, Any]  # {"job": {...}, "run": {...}}
    created_utc: str
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "id": self.id,
            "definition_id": self.definition_id,
            "definition_hash": self.definition_hash,
            "child_case": self.child_case,
            "source": json_safe(self.source),
            "child": json_safe(self.child),
            "identity": self.identity,
            "diagnostics": json_safe(self.diagnostics),
            "producer_versions": json_safe(self.producer_versions),
            "created_utc": self.created_utc,
        }

    @classmethod
    def from_dict(cls, data: Any, context: str = "submodel_run") -> "SubmodelRun":
        data = _mapping(data, context)
        version = _schema(data, context)
        source = dict(_mapping(data.get("source"), f"{context}.source"))
        for key in ("job_id", "case", "analysis_id", "analysis_hash", "job_model_hash", "job_mesh_hash",
                    "job_analysis_hash", "job_input_hash", "job_status", "mesh_record_id",
                    "mesh_record_fingerprint", "model_hash"):
            _text(source.get(key), f"{context}.source.{key}")
        if not isinstance(source.get("job_partial"), bool):
            raise SubmodelRecordError(f"{context}.source.job_partial must be a boolean")
        if source.get("mesh_selection") not in ("explicit", "job_historical"):
            raise SubmodelRecordError(f"{context}.source.mesh_selection must be 'explicit' or 'job_historical'")
        binding = _mapping(source.get("binding"), f"{context}.source.binding")
        for key in _BINDING_FIELDS:
            _text(binding.get(key), f"{context}.source.binding.{key}")
        revalidation = _mapping(source.get("revalidation"), f"{context}.source.revalidation")
        measured = {key: _finite(revalidation.get(key), f"{context}.source.revalidation.{key}")
                    for key in ("eta_scaled", "kappa_scaled", "threshold")}
        if not 0.0 <= measured["eta_scaled"] <= measured["threshold"] or measured["kappa_scaled"] < 0.0:
            raise SubmodelRecordError(f"{context}.source.revalidation is not a certified measurement")
        source["mesh_artifact"] = _descriptor(source.get("mesh_artifact"), f"{context}.source.mesh_artifact", "mesh")
        source["result_artifact"] = _descriptor(source.get("result_artifact"), f"{context}.source.result_artifact",
                                                "result")
        child = dict(_mapping(data.get("child"), f"{context}.child"))
        for key in ("model_hash", "loading_hash", "mesh_fingerprint"):
            _text(child.get(key), f"{context}.child.{key}")
        child["mesh_artifact"] = _descriptor(child.get("mesh_artifact"), f"{context}.child.mesh_artifact", "mesh")
        child["result_artifact"] = _descriptor(child.get("result_artifact"), f"{context}.child.result_artifact",
                                               "result")
        versions = dict(_mapping(data.get("producer_versions"), f"{context}.producer_versions"))
        for key in ("job", "run"):
            _mapping(versions.get(key), f"{context}.producer_versions.{key}")
        return cls(
            id=_text(data.get("id"), f"{context}.id"),
            definition_id=_text(data.get("definition_id"), f"{context}.definition_id"),
            definition_hash=_text(data.get("definition_hash"), f"{context}.definition_hash"),
            child_case=_text(data.get("child_case"), f"{context}.child_case"),
            source=source,
            child=child,
            identity=_text(data.get("identity"), f"{context}.identity"),
            diagnostics=dict(_mapping(data.get("diagnostics", {}), f"{context}.diagnostics")),
            producer_versions=versions,
            created_utc=str(data.get("created_utc", "")),
            schema_version=version,
        )
