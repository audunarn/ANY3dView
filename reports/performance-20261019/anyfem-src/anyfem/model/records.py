"""Stable analysis, mesh, job, artifact and result metadata records."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .regions import RegionRef

__all__ = [
    "AnalysisDefinition",
    "ArtifactRef",
    "ArtifactState",
    "JobRecord",
    "JobStatus",
    "MeshRecord",
    "OutputRequest",
    "ResultQuantityDescriptor",
]


def _uuid() -> str:
    return str(uuid4())


_RESULT_LOCATIONS = frozenset(
    {"node", "element", "element_face", "integration_point", "global", "history"}
)
_FRAME_POLICIES = frozenset({"all", "first", "last", "selected", "envelope"})

# This catalogue describes quantities that the qualified ANYfem analysis
# families can actually produce.  Prefix matching deliberately accepts typed
# component keys such as ``stress.von_mises`` and ``energy.kinetic`` while an
# unknown quantity fails closed instead of appearing later as a zero field.
_QUANTITY_ANALYSIS_FAMILIES: Mapping[str, frozenset[str]] = {
    "displacement": frozenset(
        {
            "linear_static",
            "batch_linear_static",
            "modal",
            "buckling",
            "nonlinear_static",
            "arc_length",
            "transient",
            "impact",
            "capacity",
            "imported",
        }
    ),
    "buckling_mode_shape": frozenset({"capacity", "imported"}),
    "frequency": frozenset({"modal", "imported"}),
    "eigenvalue": frozenset({"modal", "buckling", "capacity", "imported"}),
    "buckling_factor": frozenset({"buckling", "capacity", "imported"}),
    "load_factor": frozenset(
        {"nonlinear_static", "arc_length", "capacity", "imported"}
    ),
    "stress": frozenset(
        {
            "linear_static",
            "batch_linear_static",
            "nonlinear_static",
            "arc_length",
            "transient",
            "impact",
            "capacity",
            "imported",
        }
    ),
    "reaction": frozenset(
        {
            "linear_static",
            "batch_linear_static",
            "nonlinear_static",
            "arc_length",
            "transient",
            "impact",
            "capacity",
            "imported",
        }
    ),
    "velocity": frozenset({"transient", "impact", "imported"}),
    "acceleration": frozenset({"transient", "impact", "imported"}),
    "impulse": frozenset({"impact", "imported"}),
    "contact": frozenset({"impact", "imported"}),
    "damage": frozenset({"impact", "capacity", "imported"}),
    "energy": frozenset(
        {"nonlinear_static", "arc_length", "transient", "impact", "capacity", "imported"}
    ),
    "convergence": frozenset(
        {"nonlinear_static", "arc_length", "capacity", "imported"}
    ),
    "history": frozenset(
        {"nonlinear_static", "arc_length", "transient", "impact", "capacity", "imported"}
    ),
}


def _analysis_family(value: object) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "linear": "linear_static",
        "static": "linear_static",
        "linear_static_many": "batch_linear_static",
        "free_vibration": "modal",
        "nonlinear": "nonlinear_static",
        "arc": "arc_length",
    }
    return aliases.get(normalized, normalized)


def _quantity_family(value: str) -> str | None:
    normalized = str(value).strip().lower().replace("-", "_")
    for family in sorted(_QUANTITY_ANALYSIS_FAMILIES, key=len, reverse=True):
        if normalized == family or normalized.startswith(
            (family + ".", family + ":", family + "_")
        ):
            return family
    return None


@dataclass(frozen=True)
class OutputRequest:
    """One persistent, region-scoped request for real solver output.

    The UUID is design identity and the label is presentation only.  Every
    numerical scope is a :class:`RegionRef`; raw topology IDs are intentionally
    not accepted here because they would silently change meaning after a
    regeneration or remesh.
    """

    quantity_keys: tuple[str, ...]
    region: RegionRef
    location: str
    recovery: str = "native"
    reduction: str = "none"
    basis: str = "global"
    frame_policy: str = "all"
    label: str = "Output request"
    id: str = field(default_factory=_uuid)
    schema_version: int = 1
    frame_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        keys = tuple(dict.fromkeys(str(value).strip() for value in self.quantity_keys))
        if not keys or any(not value for value in keys):
            raise ValueError("output request needs at least one quantity key")
        object.__setattr__(self, "quantity_keys", keys)
        if not isinstance(self.region, RegionRef):
            object.__setattr__(self, "region", RegionRef(str(self.region)))
        location = str(self.location).strip().lower()
        if location not in _RESULT_LOCATIONS:
            raise ValueError(f"unknown output-request location {self.location!r}")
        object.__setattr__(self, "location", location)
        for field_name in ("recovery", "reduction", "basis"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"output request {field_name} must not be empty")
            object.__setattr__(self, field_name, value)
        frame_policy = str(self.frame_policy).strip().lower()
        if frame_policy not in _FRAME_POLICIES:
            raise ValueError(
                f"unknown frame policy {self.frame_policy!r}; expected one of "
                f"{', '.join(sorted(_FRAME_POLICIES))}"
            )
        object.__setattr__(self, "frame_policy", frame_policy)
        try:
            indices = tuple(self.frame_indices)
        except TypeError:
            raise ValueError("frame indices must be a sequence of integers") from None
        if isinstance(self.frame_indices, (str, bytes)) or any(
            isinstance(value, bool) or not isinstance(value, Integral) for value in indices
        ):
            raise ValueError("frame indices must be a sequence of integers")
        indices = tuple(int(value) for value in indices)
        if any(value < 0 for value in indices):
            raise ValueError("frame indices must be zero based and non-negative")
        if len(set(indices)) != len(indices):
            raise ValueError("frame indices must be unique")
        if indices and frame_policy != "selected":
            raise ValueError("frame indices require the selected frame policy")
        object.__setattr__(self, "frame_indices", indices)
        if not str(self.label).strip():
            raise ValueError("output request needs a label")
        object.__setattr__(self, "label", str(self.label).strip())
        if not str(self.id).strip():
            raise ValueError("output request needs a stable ID")
        object.__setattr__(self, "id", str(self.id))

    def problems_for_analysis(self, analysis_type: object) -> tuple[str, ...]:
        """Return fail-closed availability diagnostics for one analysis."""

        analysis = _analysis_family(analysis_type)
        problems: list[str] = []
        if self.frame_policy == "selected" and not self.frame_indices:
            problems.append("selected-frame output request requires explicit frame indices")
        if self.recovery == "patch":
            if self.location != "node" or self.basis != "global":
                problems.append("patch stress recovery requires node location and global basis")
            if any(_quantity_family(key) != "stress" for key in self.quantity_keys):
                problems.append("patch recovery is available only for stress quantities")
        for key in self.quantity_keys:
            family = _quantity_family(key)
            if family is None:
                problems.append(
                    f"quantity {key!r} is not a qualified ANYfem result quantity"
                )
            elif analysis not in _QUANTITY_ANALYSIS_FAMILIES[family]:
                problems.append(
                    f"quantity {key!r} is unavailable for analysis {analysis!r}"
                )
        return tuple(problems)

    def semantic_dict(self) -> dict[str, Any]:
        """Numerical/request semantics, deliberately excluding identity/label."""

        return {
            "schema_version": int(self.schema_version),
            "quantity_keys": list(self.quantity_keys),
            "region": self.region.id,
            "location": self.location,
            "recovery": self.recovery,
            "reduction": self.reduction,
            "basis": self.basis,
            "frame_policy": self.frame_policy,
            **({"frame_indices": list(self.frame_indices)} if self.frame_indices else {}),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, **self.semantic_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OutputRequest":
        raw_keys = data.get("quantity_keys", data.get("quantities", ()))
        if isinstance(raw_keys, str):
            raw_keys = (raw_keys,)
        region_value = data.get("region", data.get("region_id"))
        if isinstance(region_value, Mapping):
            region_value = region_value.get("id")
        if region_value is None:
            raise ValueError("output request needs a canonical region")
        if data.get("location") is None:
            raise ValueError("output request needs an explicit result location")
        return cls(
            id=str(data.get("id", _uuid())),
            schema_version=int(data.get("schema_version", 1)),
            label=str(data.get("label", data.get("name", "Output request"))),
            quantity_keys=tuple(str(value) for value in raw_keys),
            region=RegionRef(str(region_value)),
            location=str(data["location"]),
            recovery=str(data.get("recovery", "native")),
            reduction=str(data.get("reduction", "none")),
            basis=str(data.get("basis", "global")),
            frame_policy=str(data.get("frame_policy", "all")),
            frame_indices=data.get("frame_indices", ()),
        )


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class ArtifactState(str, Enum):
    AVAILABLE = "available"
    MISSING = "missing"
    CORRUPT = "corrupt"
    INCOMPATIBLE = "incompatible"


@dataclass
class AnalysisDefinition:
    name: str
    type: str = "linear_static"
    target_kind: str = "load_case"
    target_id: str = "default"
    settings: dict[str, Any] = field(default_factory=dict)
    output_request_ids: tuple[str, ...] = ()
    # Reader/compatibility view for pre-v4 callers.  New code stores requests
    # in Project.output_requests and references only their UUIDs above.
    output_requests: dict[str, Any] = field(default_factory=dict)
    resource_policy: dict[str, Any] = field(default_factory=dict)
    # B3-GE is a standalone, exact-definition solver workflow.  It must never
    # replace the ordinary quadratic-beam element just because a project is
    # reopened with a newer ANYsolver.  Missing data therefore means legacy
    # B3, while the opt-in requires its own analysis type and authenticated
    # B3GENativeOptIn payload.
    beam_formulation: str = "legacy-b3"
    beam_formulation_options: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_uuid)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("analysis definition needs a name")
        if not self.type:
            raise ValueError("analysis definition needs a type")
        if self.target_kind not in (
            "load_case", "load_cases", "combination", "none"
        ):
            raise ValueError(f"unknown analysis target kind {self.target_kind!r}")
        self.output_request_ids = tuple(
            dict.fromkeys(str(value) for value in self.output_request_ids)
        )
        if any(not value for value in self.output_request_ids):
            raise ValueError("analysis output-request IDs must not be empty")
        if self.beam_formulation == "legacy-b3":
            if self.beam_formulation_options:
                raise ValueError(
                    "legacy B3 must not carry B3-GE formulation options"
                )
        elif self.beam_formulation == "b3-ge":
            if self.type != "b3_ge_native":
                raise ValueError(
                    "b3-ge requires the dedicated b3_ge_native analysis type"
                )
            from ..solve.ge_beam3 import B3GENativeOptIn

            opt_in = B3GENativeOptIn.from_dict(
                dict(self.beam_formulation_options)
            )
            # Detach and canonicalize now.  Later mutation of the caller's
            # dictionary must not change the persisted solver selection.
            self.beam_formulation_options = opt_in.to_dict()
        else:
            raise ValueError(
                f"unknown beam formulation {self.beam_formulation!r}; "
                "use 'legacy-b3' or explicit 'b3-ge'"
            )

    @classmethod
    def b3_ge(cls, name: str, opt_in: Any, **options: Any) -> "AnalysisDefinition":
        """Create the only analysis record that explicitly selects B3-GE."""

        from ..solve.ge_beam3 import B3GENativeOptIn

        if type(opt_in) is not B3GENativeOptIn:
            raise ValueError("an exact B3GENativeOptIn is required")
        return cls(
            name=name,
            type="b3_ge_native",
            target_kind="none",
            target_id="",
            beam_formulation="b3-ge",
            beam_formulation_options=opt_in.to_dict(),
            **options,
        )

    def create_b3_ge_analysis(
        self,
        boundaries: Any,
        *,
        retained_refinement: bool = False,
    ):
        """Materialize this exact opt-in through ANYsolver's public facade."""

        if self.beam_formulation != "b3-ge" or self.type != "b3_ge_native":
            raise ValueError("this analysis uses legacy B3, not explicit b3-ge")
        from ..solve.ge_beam3 import B3GENativeOptIn

        return B3GENativeOptIn.from_dict(
            self.beam_formulation_options
        ).create_analysis(
            boundaries,
            retained_refinement=retained_refinement,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "schema_version": int(self.schema_version),
            "name": self.name,
            "type": self.type,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "settings": dict(self.settings),
            "output_request_ids": list(self.output_request_ids),
            "output_requests": dict(self.output_requests),
            "resource_policy": dict(self.resource_policy),
            "beam_formulation": self.beam_formulation,
            "beam_formulation_options": dict(self.beam_formulation_options),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalysisDefinition":
        return cls(
            id=str(data.get("id", _uuid())),
            schema_version=int(data.get("schema_version", 1)),
            name=str(data.get("name", "Analysis")),
            type=str(data.get("type", "linear_static")),
            target_kind=str(data.get("target_kind", "load_case")),
            target_id=str(data.get("target_id", "default")),
            settings=dict(data.get("settings", {})),
            output_request_ids=tuple(
                str(value) for value in data.get("output_request_ids", ())
            ),
            output_requests=dict(data.get("output_requests", {})),
            resource_policy=dict(data.get("resource_policy", {})),
            beam_formulation=str(data.get("beam_formulation", "legacy-b3")),
            beam_formulation_options=dict(
                data.get("beam_formulation_options", {})
            ),
        )


@dataclass
class MeshRecord:
    name: str
    source_model_hash: str
    mesh_input_hash: str
    mesh_hash: str
    artifact_id: str | None = None
    kind: str = "generated"
    status: str = "completed"
    summary: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[Any] = field(default_factory=list)
    structural_preparation: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_uuid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "source_model_hash": self.source_model_hash,
            "mesh_input_hash": self.mesh_input_hash,
            "mesh_hash": self.mesh_hash,
            "artifact_id": self.artifact_id,
            "summary": dict(self.summary),
            "diagnostics": list(self.diagnostics),
            "structural_preparation": dict(self.structural_preparation),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeshRecord":
        return cls(
            id=str(data.get("id", _uuid())),
            name=str(data.get("name", "Mesh")),
            kind=str(data.get("kind", "generated")),
            status=str(data.get("status", "completed")),
            source_model_hash=str(data.get("source_model_hash", "")),
            mesh_input_hash=str(data.get("mesh_input_hash", "")),
            mesh_hash=str(data.get("mesh_hash", "")),
            artifact_id=None if data.get("artifact_id") is None else str(data["artifact_id"]),
            summary=dict(data.get("summary", {})),
            diagnostics=list(data.get("diagnostics", [])),
            structural_preparation=dict(data.get("structural_preparation", {})),
        )


@dataclass
class ArtifactRef:
    kind: str
    uri: str
    schema_version: int = 1
    byte_size: int = 0
    sha256: str = ""
    created_utc: str = ""
    id: str = field(default_factory=_uuid)

    def __post_init__(self) -> None:
        path = PurePosixPath(self.uri.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact URI must be a safe relative path")
        self.uri = path.as_posix()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "uri": self.uri,
            "schema_version": int(self.schema_version),
            "byte_size": int(self.byte_size),
            "sha256": self.sha256,
            "created_utc": self.created_utc,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            id=str(data.get("id", _uuid())),
            kind=str(data.get("kind", "unknown")),
            uri=str(data.get("uri", "")),
            schema_version=int(data.get("schema_version", 1)),
            byte_size=int(data.get("byte_size", 0)),
            sha256=str(data.get("sha256", "")),
            created_utc=str(data.get("created_utc", "")),
        )


@dataclass
class JobRecord:
    analysis_id: str
    name: str = "Job"
    model_hash: str = ""
    mesh_hash: str = ""
    analysis_hash: str = ""
    input_hash: str = ""
    status: JobStatus | str = JobStatus.QUEUED
    created_utc: str = ""
    started_utc: str = ""
    finished_utc: str = ""
    result_artifact_id: str | None = None
    log_artifact_id: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    outcome: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[Any] = field(default_factory=list)
    partial: bool = False
    producer_versions: dict[str, str] = field(default_factory=dict)
    id: str = field(default_factory=_uuid)

    def __post_init__(self) -> None:
        self.status = JobStatus(self.status)

    def stale_against(self, *, model_hash: str, mesh_hash: str, analysis_hash: str) -> bool:
        return (
            self.model_hash != model_hash
            or self.mesh_hash != mesh_hash
            or self.analysis_hash != analysis_hash
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "analysis_id": self.analysis_id,
            "name": self.name,
            "model_hash": self.model_hash,
            "mesh_hash": self.mesh_hash,
            "analysis_hash": self.analysis_hash,
            "input_hash": self.input_hash,
            "status": self.status.value,
            "created_utc": self.created_utc,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "result_artifact_id": self.result_artifact_id,
            "log_artifact_id": self.log_artifact_id,
            "summary": dict(self.summary),
            "outcome": dict(self.outcome),
            "diagnostics": list(self.diagnostics),
            "partial": bool(self.partial),
            "producer_versions": dict(self.producer_versions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobRecord":
        status = str(data.get("status", "queued"))
        if status in ("running", "cancelling", "queued"):
            status = "interrupted"
        return cls(
            id=str(data.get("id", _uuid())),
            analysis_id=str(data.get("analysis_id", "")),
            name=str(data.get("name", "Job")),
            model_hash=str(data.get("model_hash", "")),
            mesh_hash=str(data.get("mesh_hash", "")),
            analysis_hash=str(data.get("analysis_hash", "")),
            input_hash=str(data.get("input_hash", "")),
            status=status,
            created_utc=str(data.get("created_utc", "")),
            started_utc=str(data.get("started_utc", "")),
            finished_utc=str(data.get("finished_utc", "")),
            result_artifact_id=None if data.get("result_artifact_id") is None else str(data["result_artifact_id"]),
            log_artifact_id=None if data.get("log_artifact_id") is None else str(data["log_artifact_id"]),
            summary=dict(data.get("summary", {})),
            outcome=dict(data.get("outcome", {})),
            diagnostics=list(data.get("diagnostics", [])),
            partial=bool(data.get("partial", False)),
            producer_versions={str(k): str(v) for k, v in dict(data.get("producer_versions", {})).items()},
        )


@dataclass(frozen=True)
class ResultQuantityDescriptor:
    key: str
    label: str
    location: str
    unit: str = ""
    components: tuple[str, ...] = ()
    basis: str = "global"
    frames: tuple[float, ...] = ()
    recovery: str = "native"
    reduction: str = "none"
    deformation_required: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.location not in (
            "node", "element", "element_face", "integration_point", "global", "history"
        ):
            raise ValueError(f"unknown result location {self.location!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "location": self.location,
            "unit": self.unit,
            "components": list(self.components),
            "basis": self.basis,
            "frames": list(self.frames),
            "recovery": self.recovery,
            "reduction": self.reduction,
            "deformation_required": self.deformation_required,
            "provenance": dict(self.provenance),
        }
