"""Submodels driven from completed jobs: revalidated source, staged runs,
compare-and-publish, staleness (workflow contract, section 11).

A job-bound source is a *revalidation*: the parent is rebuilt now by one fixed
recipe and the archived displacement field is admitted only if it is certified as
the solution of that rebuilt problem.  The field is never replaced by a new solve.

Runs never edit the live project except in :func:`publish_submodel_run`.  Session
isolation relies on the host contract that every document mutation happens on
one owning thread; ``DocumentSession`` does not enforce it and neither does this
module (its own checks catch misuse of the run API only).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np

from .model.submodels import (
    SourceReference,
    SubmodelDefinition,
    SubmodelRecordError,
    SubmodelRun,
    child_problems,
    definition_hash,
    descriptor,
    json_safe,
)
from .submodel import (
    SubmodelError,
    SubmodelLineage,
    SubmodelSettings,
    _loading_hash,
    _mesh_fingerprint,
    _model_hash,
    bind_source,
    solve_submodel,
)

__all__ = [
    "JobSource",
    "PublishResult",
    "StagedRun",
    "SubmodelSnapshot",
    "add_submodel",
    "discard_staged_run",
    "job_source",
    "publish_submodel_run",
    "run_submodel",
    "stage_submodel_run",
    "submodel_orphans",
    "submodel_run_staleness",
    "submodel_snapshot",
]

_LINEAR_STATIC = "linear_static"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_cancelled(token: Any, stage: str) -> None:
    if token is None:
        return
    raiser = getattr(token, "raise_if_cancelled", None)
    if callable(raiser):
        try:
            raiser(stage)
        except Exception as error:
            raise SubmodelError("SUBMODEL_CANCELLED", f"cancelled during {stage}") from error
        return
    cancelled = getattr(token, "cancelled", False)
    if callable(cancelled):
        cancelled = cancelled()
    if cancelled:
        raise SubmodelError("SUBMODEL_CANCELLED", f"cancelled during {stage}")


def _producer_versions() -> Dict[str, str]:
    from .io.result_artifact import _producer_versions as versions

    return {k: v for k, v in versions().items() if k not in ("platform",)}


# ----------------------------------------------------------------------
# definitions
# ----------------------------------------------------------------------
def add_submodel(project: Any, definition: SubmodelDefinition) -> SubmodelDefinition:
    """Validate ``definition`` (detached, serializable child) and store it.

    Callers that want undo and a revision use a session transaction around this.
    """

    try:
        definition.validate()
        json.dumps(definition.to_dict(), allow_nan=False)
    except SubmodelRecordError as error:
        raise SubmodelError("SUBMODEL_DEFINITION", str(error)) from error
    except (TypeError, ValueError) as error:
        raise SubmodelError("SUBMODEL_DEFINITION", f"the definition cannot be serialized: {error}") from error
    if definition.id in project.submodels:
        raise SubmodelError("SUBMODEL_DEFINITION", f"submodel {definition.id!r} already exists")
    project.submodels[definition.id] = definition
    return definition


# ----------------------------------------------------------------------
# job-bound source (contract 11.3)
# ----------------------------------------------------------------------
@dataclass
class JobSource:
    solution: Any  # LinearSolution, revalidated
    manifest: Dict[str, Any]  # source part of the run manifest


def _mesh_record_fingerprint(record: Mapping[str, Any]) -> str:
    from .document import canonical_hash

    return canonical_hash({key: record.get(key) for key in (
        "id", "kind", "status", "source_model_hash", "mesh_input_hash", "mesh_hash",
        "structural_preparation", "artifact_id")})


def _status(value: Any) -> str:
    return str(getattr(value, "value", value))


def _require_recipe(analysis: Any, reference: SourceReference) -> None:
    from .model.records import AnalysisDefinition

    defaults = AnalysisDefinition(name="defaults")
    if analysis.type != _LINEAR_STATIC or analysis.target_kind != "load_case":
        raise SubmodelError("SOURCE_RECIPE_UNSUPPORTED",
                            f"the source analysis is {analysis.type}/{analysis.target_kind}, not linear static on a load case")
    if analysis.target_id != reference.case:
        raise SubmodelError("SOURCE_RECIPE_UNSUPPORTED",
                            f"the source job solved case {analysis.target_id!r}, not {reference.case!r}")
    settings = dict(analysis.settings or {})
    if settings not in ({}, {"load_case": reference.case}):
        raise SubmodelError("SOURCE_RECIPE_UNSUPPORTED",
                            f"the source job's recorded settings {sorted(settings)} are not the admitted recipe")
    if (analysis.beam_formulation != defaults.beam_formulation
            or dict(analysis.beam_formulation_options or {}) != defaults.beam_formulation_options):
        raise SubmodelError("SOURCE_RECIPE_UNSUPPORTED", "the source job uses a non-default beam formulation")


def _registered(project: Any, artifact_id: Any, kind: str, what: str) -> Any:
    artifact = project.artifacts.get(str(artifact_id or ""))
    if artifact is None:
        raise SubmodelError("SOURCE_ARTIFACT", f"the {what} artifact is not registered")
    if artifact.kind != kind:
        raise SubmodelError("SOURCE_ARTIFACT", f"the {what} artifact is a {artifact.kind!r} artifact")
    return artifact


def _verified_path(store: Any, artifact: Any, kind: str, what: str) -> Path:
    from .io.artifacts import ArtifactError

    if not artifact.sha256:
        raise SubmodelError("SOURCE_ARTIFACT", f"the {what} artifact has no checksum")
    try:
        return store._path_for_read(artifact, kind)
    except (ArtifactError, OSError) as error:
        raise SubmodelError("SOURCE_ARTIFACT", f"the {what} artifact does not verify: {error}") from error


def _select_mesh_record(project: Any, job: Any, mesh_record_id: Optional[str]) -> Any:
    if mesh_record_id is not None:
        record = project.mesh_records.get(str(mesh_record_id))
        if record is None:
            raise SubmodelError("SOURCE_MESH", f"mesh record {mesh_record_id!r} does not exist")
        if record.mesh_hash != job.mesh_hash:
            raise SubmodelError("SOURCE_MESH", "the active mesh is not the mesh the source job solved")
        return record
    matches = [r for r in project.mesh_records.values() if r.mesh_hash == job.mesh_hash]
    if len(matches) != 1:
        raise SubmodelError("SOURCE_MESH",
                            f"{len(matches)} mesh records carry the source job's mesh hash; pass mesh_record_id")
    return matches[0]


def _field_vector(dataset: Any, fe_model: Any) -> np.ndarray:
    from .io.result_artifact import _DOF_COMPONENTS, _dof_unit

    if "displacement" not in dataset.field_keys:
        raise SubmodelError("SOURCE_FIELD", "the source result has no displacement field")
    result_field = dataset.field("displacement")
    descriptor_ = result_field.descriptor
    if tuple(descriptor_.components) != _DOF_COMPONENTS:
        raise SubmodelError("SOURCE_FIELD", f"the displacement field has components {descriptor_.components}")
    if descriptor_.location != "node" or descriptor_.basis != "global":
        raise SubmodelError("SOURCE_FIELD", "the displacement field is not a global nodal field")
    if descriptor_.unit != _dof_unit("displacement", _DOF_COMPONENTS):
        raise SubmodelError("SOURCE_FIELD", f"the displacement field has unit {descriptor_.unit!r}")
    values = np.asarray(result_field.read(), dtype=float)
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != 6:
        raise SubmodelError("SOURCE_FIELD", f"the displacement field has shape {values.shape}, not one frame of six components")
    try:
        table = np.asarray(dataset.table("displacement_node_ids"))
    except Exception as error:
        raise SubmodelError("SOURCE_FIELD", f"the displacement node table is unreadable: {error}") from error
    if table.dtype.kind not in "iu":
        raise SubmodelError("SOURCE_FIELD", f"the displacement node table holds {table.dtype}, not integers")
    node_ids = [int(v) for v in table.reshape(-1)]
    stored = descriptor_.provenance.get("global_dof_indices")
    if not isinstance(stored, list) or any(isinstance(v, bool) or not isinstance(v, int) for v in stored):
        raise SubmodelError("SOURCE_FIELD", "the stored global DOF indices are not integers")
    if len(stored) != values.shape[1] * 6 or len(node_ids) != values.shape[1]:
        raise SubmodelError("SOURCE_FIELD", "the displacement field's node and DOF tables do not match its values")
    if len(set(node_ids)) != len(node_ids):
        raise SubmodelError("SOURCE_FIELD", "the displacement node table has duplicates")
    mesh = fe_model.mesh
    dm = mesh.dof_manager
    if sorted(node_ids) != sorted(int(n) for n in mesh.nodes):
        raise SubmodelError("SOURCE_LAYOUT", "the stored nodes are not the rebuilt model's nodes")
    total = int(dm.total_dofs)
    if total != 6 * len(node_ids):
        raise SubmodelError("SOURCE_LAYOUT", "the rebuilt model does not have six DOFs per node")
    for k, node in enumerate(node_ids):
        expected = [int(d) for d in dm.get_node_dofs(node)]
        if [int(d) for d in stored[6 * k: 6 * k + 6]] != expected:
            raise SubmodelError("SOURCE_LAYOUT", f"node {node}'s stored DOF indices differ from the rebuilt layout")
    indices = np.asarray(stored, dtype=int)
    if len(set(indices.tolist())) != total or indices.min() < 0 or indices.max() >= total:
        raise SubmodelError("SOURCE_LAYOUT", "the stored DOF indices do not cover the rebuilt layout exactly once")
    flat = values[0].reshape(-1)
    if not np.all(np.isfinite(flat)):
        raise SubmodelError("SOURCE_FIELD", "the displacement field is not finite")
    u = np.zeros(total)
    u[indices] = flat
    return u


def job_source(project: Any, reference: SourceReference, store: Any, *, document: Mapping[str, Any],
               model_hash: str, mesh_record_id: Optional[str] = None) -> JobSource:
    """Revalidated source from a completed job (contract 11.3).

    ``project`` is a detached parent (a thawed snapshot), ``document`` and
    ``model_hash`` the snapshot's document and revision model hash.
    """

    from anysolver.submodel_transfer import (
        SubmodelTransferError,
        _CERTIFIED_BACKWARD_ERROR,
        estimate_static_error_allowance,
    )

    from .io.artifacts import ArtifactError
    from .jobs import analysis_hash
    from .mesh_jobs import mesh_semantic_hash
    from .model.records import JobStatus, OutputRequest
    from .post.results import LinearSolution
    from .solve.build import build_fe_model
    from .solve.run import preflight

    # 1. job and analysis
    job = project.jobs.get(reference.job_id)
    if job is None:
        raise SubmodelError("SOURCE_JOB", f"job {reference.job_id!r} does not exist")
    if _status(job.status) != JobStatus.COMPLETED.value or job.partial:
        raise SubmodelError("SOURCE_JOB", f"job {reference.job_id!r} is {_status(job.status)}"
                            f"{' and partial' if job.partial else ''}, not completed")
    analysis = project.analyses.get(job.analysis_id)
    if analysis is None:
        raise SubmodelError("SOURCE_JOB", "the source job's analysis definition is missing")
    requests = [OutputRequest.from_dict(item) for item in document.get("output_requests", ())]
    current_analysis_hash = analysis_hash(analysis, {r.id: r for r in requests}, document=document)
    if current_analysis_hash != job.analysis_hash:
        raise SubmodelError("SOURCE_LOADING_CHANGED", "the source analysis or its loading changed since the job")
    # 2. recipe
    if project.mesh_only or project.imported_format:
        raise SubmodelError("SOURCE_RECIPE_UNSUPPORTED", "imported or mesh-only parents are not admitted")
    _require_recipe(analysis, reference)
    # 3. parent model
    if job.model_hash != model_hash:
        raise SubmodelError("SOURCE_MODEL_CHANGED", "the parent model changed since the source job")
    # 4. mesh
    record = _select_mesh_record(project, job, mesh_record_id)
    if record.status != "completed" or record.source_model_hash != job.model_hash:
        raise SubmodelError("SOURCE_MESH", "the source mesh record is not a completed mesh of the job's model")
    mesh_artifact = _registered(project, record.artifact_id, "mesh", "source mesh")
    _verified_path(store, mesh_artifact, "mesh", "source mesh")
    try:
        mesh = store.read_mesh(mesh_artifact)
    except (ArtifactError, OSError) as error:
        raise SubmodelError("SOURCE_ARTIFACT", f"the source mesh artifact does not decode: {error}") from error
    decoded_hash = mesh_semantic_hash(mesh, model_hash=record.source_model_hash, mesh_input_hash=record.mesh_input_hash,
                                      structural_preparation=record.structural_preparation)
    if decoded_hash != job.mesh_hash:
        raise SubmodelError("SOURCE_MESH", "the decoded source mesh does not have the job's mesh hash")
    # 5. result artifact
    result_artifact = _registered(project, job.result_artifact_id, "result", "source result")
    _verified_path(store, result_artifact, "result", "source result")
    try:
        dataset = store.open_result(result_artifact)
        dataset.validate()
        identity = dataset.identity
        summary = dataset.metadata("summary")
    except (ArtifactError, OSError) as error:
        raise SubmodelError("SOURCE_ARTIFACT", f"the source result artifact is unreadable: {error}") from error
    expected = {"artifact_id": job.id, "document_id": project.document_id, "mesh_id": record.id,
                "model_hash": job.model_hash, "mesh_hash": job.mesh_hash, "analysis_hash": job.analysis_hash,
                "frame_kind": "static", "partial": False}
    for key, value in expected.items():
        if identity.get(key) != value:
            raise SubmodelError("SOURCE_ARTIFACT", f"the source result's {key} is {identity.get(key)!r}, expected {value!r}")
    # Linear static writers store no solver outcome today (the job completes on
    # the plain branch); when one is stored, on the job or in the artifact, it
    # must be converged with results.  Certification is the numerical gate below.
    stored = [item for item in (dict(job.outcome or {}),
                                (summary or {}).get("outcome") if isinstance(summary, Mapping) else None) if item]
    for outcome in stored:
        if not isinstance(outcome, Mapping) or outcome.get("converged") is not True or outcome.get("has_results") is not True:
            raise SubmodelError("SOURCE_NOT_CERTIFIED", "the source's stored solver outcome is not converged with results")
    # 6-7. rebuild by the fixed recipe, place the field exactly, revalidate
    built = build_fe_model(project, mesh, load_case=reference.case)
    report = preflight(built, analysis_type="Linear static")
    if report.errors:
        raise SubmodelError("SOURCE_RECIPE_PREFLIGHT", "the rebuilt source does not pass preflight")
    u = _field_vector(dataset, built.fe_model)
    try:
        estimate = estimate_static_error_allowance(built.fe_model, u, built.load_case)
    except SubmodelTransferError as error:
        raise SubmodelError("SOURCE_REVALIDATION_FAILED", f"the archived field fails revalidation: {error}") from error
    eta, kappa = float(estimate.eta_scaled), float(estimate.kappa_scaled)
    if not (np.isfinite(eta) and np.isfinite(kappa)) or estimate.inconclusive or eta > _CERTIFIED_BACKWARD_ERROR:
        raise SubmodelError(
            "SOURCE_REVALIDATION_FAILED",
            f"the archived field is not a certified solution of the rebuilt parent (scaled backward error {eta:.3e}, "
            f"kappa {kappa:.3e}, inconclusive {estimate.inconclusive})",
        )
    revalidation = {"eta_scaled": eta, "kappa_scaled": kappa, "threshold": _CERTIFIED_BACKWARD_ERROR}
    solution = LinearSolution(
        displacements=u, built=built, label=f"job {job.id}",
        info={"convergence_info": {"status": "converged", "relative_backward_error_disposition": "CERTIFIED",
                                   "revalidated_from_job": job.id, **revalidation}},
    )
    manifest = {
        "job_id": job.id, "case": reference.case, "analysis_id": job.analysis_id,
        "analysis_hash": current_analysis_hash, "job_model_hash": job.model_hash, "job_mesh_hash": job.mesh_hash,
        "job_analysis_hash": job.analysis_hash, "job_input_hash": job.input_hash or "-",
        "job_status": _status(job.status), "job_partial": bool(job.partial),
        "job_result_artifact_id": job.result_artifact_id,
        "mesh_record_id": record.id, "mesh_record_fingerprint": _mesh_record_fingerprint(record.to_dict()),
        "mesh_selection": "explicit" if mesh_record_id is not None else "job_historical",
        "model_hash": model_hash,
        "mesh_artifact": descriptor(mesh_artifact), "result_artifact": descriptor(result_artifact),
        "revalidation": revalidation, "job_producer_versions": dict(job.producer_versions or {}),
    }
    return JobSource(solution, manifest)


# ----------------------------------------------------------------------
# snapshot, stage, publish (contract 11.4)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SubmodelSnapshot:
    snapshot: Any  # anyfem.document.ProjectSnapshot
    thread_id: int


def submodel_snapshot(session: Any) -> SubmodelSnapshot:
    """The committed document a run works from; refused inside a transaction."""

    if getattr(session, "_transaction_depth", 0):
        raise SubmodelError("SUBMODEL_SESSION_BUSY", "a submodel run cannot start inside a session transaction")
    return SubmodelSnapshot(session.snapshot(), threading.get_ident())


@dataclass
class StagedRun:
    run: SubmodelRun
    artifacts: Tuple[Any, Any]  # (child mesh ArtifactRef, child result ArtifactRef)
    revision_id: str
    thread_id: int
    files: Tuple[Path, ...]
    store: Any
    document_id: str = ""
    result: Any = None  # the in-memory SubmodelResult
    published: bool = False  # set once publication committed; its files are never discarded


def _definition_entry(document: Mapping[str, Any], definition_id: str) -> Mapping[str, Any]:
    for entry in document.get("submodels", ()):
        if isinstance(entry, Mapping) and entry.get("id") == definition_id:
            return entry
    raise SubmodelError("SUBMODEL_DEFINITION", f"submodel {definition_id!r} does not exist")


def _child_identity(run: SubmodelRun, document_id: str, kind: str) -> Dict[str, str]:
    """What a run's own artifact must carry inside it."""

    if kind == "mesh":
        return {"artifact_id": f"{run.id}-mesh", "document_id": document_id,
                "model_hash": run.child["model_hash"], "mesh_hash": run.child["mesh_fingerprint"]}
    return {"artifact_id": run.id, "document_id": document_id, "mesh_id": f"{run.id}-mesh",
            "model_hash": run.child["model_hash"], "mesh_hash": run.child["mesh_fingerprint"],
            "analysis_hash": run.definition_hash, "frame_kind": "static", "partial": False}


def _child_problem(store: Any, artifact: Any, kind: str, run: SubmodelRun, document_id: str) -> Optional[str]:
    """Why a run's own artifact does not match the run (None when it does)."""

    from .io.artifacts import ArtifactError

    if descriptor(artifact) != run.child[f"{kind}_artifact"]:
        return f"the {kind} artifact differs from the run's manifest"
    try:
        store._path_for_read(artifact, kind)
        if kind == "mesh":
            identity = store.read_mesh_metadata(artifact)
            store.read_mesh(artifact)
        else:
            dataset = store.open_result(artifact)
            dataset.validate()
            identity = dataset.identity
    except (ArtifactError, OSError) as error:
        return f"the {kind} artifact does not verify: {error}"
    for key, value in _child_identity(run, document_id, kind).items():
        if identity.get(key) != value:
            return f"the {kind} artifact's {key} is {identity.get(key)!r}, expected {value!r}"
    return None


def _verify_child(store: Any, artifact: Any, kind: str, run: SubmodelRun, document_id: str) -> None:
    problem = _child_problem(store, artifact, kind, run, document_id)
    if problem is not None:
        raise SubmodelError("SUBMODEL_ARTIFACT", problem)


def stage_submodel_run(snap: SubmodelSnapshot, definition_id: str, store: Any, *,
                       mesh_record_id: Optional[str] = None, cancellation_token: Any = None) -> StagedRun:
    """Solve a submodel from a snapshot and write its artifacts; touches no project."""

    from .io.result_artifact import write_solution_artifact

    snapshot = snap.snapshot
    document = snapshot.document
    entry = _definition_entry(document, definition_id)
    definition_digest = definition_hash(entry)
    parent = snapshot.thaw()
    definition = parent.submodels[definition_id]
    _check_cancelled(cancellation_token, "source admission")
    source = job_source(parent, definition.source, store, document=document,
                        model_hash=snapshot.revision.model_hash, mesh_record_id=mesh_record_id)
    binding = bind_source(source.solution)
    child = definition.child
    _check_cancelled(cancellation_token, "child meshing")
    child_mesh = child.generate_mesh(definition.target_size)
    _require_scalar_offset_sides(parent, source.solution.built.mesh, definition)
    _require_scalar_offset_sides(child, child_mesh, None)
    _check_cancelled(cancellation_token, "submodel solve")
    result = solve_submodel(
        source.solution, binding, child, child_mesh,
        SubmodelLineage(dict(definition.source_faces), dict(definition.cut_faces)),
        SubmodelSettings(policy=definition.policy, physical_tolerance=definition.tolerances["physical"],
                         natural_tolerance=definition.tolerances["natural"],
                         weight_tolerance=definition.tolerances["weight"]),
        load_case=definition.child_case,
    )
    _check_cancelled(cancellation_token, "artifact writing")
    run_id = str(uuid4())
    mesh_id = f"{run_id}-mesh"
    destinations = (store.resolve(f"meshes/{mesh_id}.anymesh.h5"), store.resolve(f"results/{run_id}.anyres.h5"))
    if any(path.exists() for path in destinations):
        raise SubmodelError("SUBMODEL_ARTIFACT", "the run's artifact destinations already exist")
    child_model = _model_hash(child)
    child_loading = _loading_hash(child)
    child_mesh_fingerprint = _mesh_fingerprint(child_mesh)
    try:
        mesh_ref = store.write_mesh(child_mesh, mesh_id=mesh_id, document_id=parent.document_id,
                                    model_hash=child_model, mesh_hash=child_mesh_fingerprint)
        result_ref = write_solution_artifact(
            store, result.solution, job_id=run_id, document_id=parent.document_id, mesh_id=mesh_id,
            model_hash=child_model, mesh_hash=child_mesh_fingerprint, analysis_hash=definition_digest,
            provenance={"submodel": {"definition_id": definition_id, "source_job_id": definition.source.job_id}},
            summary={"submodel_run": run_id},
        )
        versions = _producer_versions()
        run = SubmodelRun(
            id=run_id, definition_id=definition_id, definition_hash=definition_digest,
            child_case=definition.child_case, source={**source.manifest, "binding": vars(binding)},
            child={"model_hash": child_model, "loading_hash": child_loading, "mesh_fingerprint": child_mesh_fingerprint,
                   "mesh_artifact": descriptor(mesh_ref), "result_artifact": descriptor(result_ref)},
            identity=result.identity, diagnostics=json_safe(result.diagnostics),
            producer_versions={"job": source.manifest["job_producer_versions"], "run": versions},
            created_utc=_utc_now(),
        )
        _verify_child(store, mesh_ref, "mesh", run, parent.document_id)
        _verify_child(store, result_ref, "result", run, parent.document_id)
        json.dumps(run.to_dict(), allow_nan=False)
        json.dumps([mesh_ref.to_dict(), result_ref.to_dict()], allow_nan=False)
    except BaseException:
        # The destinations were reserved (they did not exist), so whatever is
        # there now is this attempt's, even if a writer failed after writing.
        _remove(destinations)
        raise
    return StagedRun(run, (mesh_ref, result_ref), snapshot.revision.id, snap.thread_id, destinations, store,
                     parent.document_id, result)


def _require_scalar_offset_sides(project: Any, mesh: Any, definition: Any) -> None:
    """Scalar beam offsets in a mesh stand off by ``eccentricity * plate normal``.

    The cut carries a scalar offset only where every plate at the beam is flat with
    one signed normal (``submodel_cut._require_same_offset_normals``), and the mesher
    stands it off along that normal.  An archived parent mesh made by an older
    mesher may have put it on the other side (the side followed the face loop on
    the mapped route), and the child would then model the stiffener elsewhere.
    So the actual offsets are checked in both meshes: the parent's on its beam
    edges near the box (``definition`` given), the child's on all of its beams.
    Edges whose plates are not flat with one normal are not checked here: the cut
    carries no scalar offset from them.  In the source mesh the offset is checked
    only over the beam inside the box, which is what the child models: at the ends
    of each segment's clipped part (exact, the offset being linear along a straight
    element).  Outside the box the parent's nodal normal may rightly tilt where
    another plate meets the beam (a stiffener ending at a bulkhead), which the cut's
    own check does not see either.
    """

    from anygeometry.surfaces import Plane
    from anygeometry.volume_cut import OUTSIDE, BoxVolume, _edge_extent, _planar_coons

    geometry = project.geometry
    edges = sorted(project.edge_sections)
    volume = None
    if definition is not None:
        volume = BoxVolume(**{k: definition.volume[k] for k in ("lower", "upper", "transform")})
        tolerance = geometry.tolerance.effective_length(volume.diagonal)
        edges = [e for e in edges if e in geometry.edges and volume.reaches(_edge_extent(geometry, e, volume), tolerance)]
    offsets = project.beam_offsets
    for edge in edges:
        offset = offsets.get(edge)
        if offset is None or isinstance(offset, tuple):
            continue
        faces = sorted(geometry.faces_using_edge(edge))
        if not faces or not all(isinstance(getattr(geometry.faces[f], "surface", None), Plane)
                                or _planar_coons(geometry, f, getattr(geometry.faces[f], "surface", None))
                                for f in faces):
            continue
        normals = [np.asarray(geometry.face_normal_many(f, [(0.5, 0.5)])[0], dtype=float) for f in faces]
        if any(float(np.max(np.abs(n - normals[0]))) > 1e-9 for n in normals[1:]):
            continue
        expected = float(offset) * normals[0]
        bases, stood_nodes = mesh.nodes_of_edge.get(edge, ()), mesh.offset_nodes_of_edge.get(edge, ())
        if not stood_nodes or len(stood_nodes) != len(bases):
            raise SubmodelError("SUBMODEL_OFFSET_SIDE", f"beam edge {edge} has a scalar offset but "
                                                        f"{len(stood_nodes)} offset nodes for {len(bases)} stations "
                                                        f"in the {'source' if definition is not None else 'child'} mesh")
        stations = list(zip(bases, stood_nodes, strict=True))
        points = [np.asarray(mesh.nodes[base], dtype=float) for base, _stood in stations]
        offsets_at = [np.asarray(mesh.nodes[stood], dtype=float) - points[i] for i, (_b, stood) in enumerate(stations)]
        if volume is None:
            checks = list(zip(points, offsets_at))
        else:
            # The parent beam between consecutive stations (mid-sides included) is a
            # straight element whose offset varies linearly; the child models it only
            # where it is inside the box.  So the offset is checked at the two ends of
            # each segment's clipped part, which is exact for a linear field: a box
            # between two stations still checks that segment, and a segment that only
            # touches the box is checked at the contact point, not at its outside end.
            checks = []
            for index in range(len(stations) - 1):
                span = _clip_segment(volume, points[index], points[index + 1], tolerance)
                for t in span:
                    checks.append((points[index] + t * (points[index + 1] - points[index]),
                                   (1.0 - t) * offsets_at[index] + t * offsets_at[index + 1]))
            if len(stations) == 1 and volume.classify(points[0], tolerance) != OUTSIDE:
                checks.append((points[0], offsets_at[0]))
        for point, actual in checks:
            if float(np.max(np.abs(actual - expected))) > 1e-9 * max(1.0, abs(float(offset))):
                which = "source" if definition is not None else "child"
                raise SubmodelError(
                    "SUBMODEL_OFFSET_SIDE",
                    f"the {which} mesh stands beam edge {edge} off by {np.round(actual, 9).tolist()} at "
                    f"{np.round(point, 9).tolist()}, not {np.round(expected, 9).tolist()} (eccentricity along the "
                    "plate normal); "
                    + ("the parent was meshed with an older mesher (remesh and re-solve it), or the box ends "
                       "within a parent element of a plate meeting the beam (move the box)" if definition is not None
                       else "the mesher does not follow the plate-normal convention"))


def _clip_segment(volume: Any, a: np.ndarray, b: np.ndarray, tolerance: float) -> Tuple[float, ...]:
    """Parameters ``t`` in [0, 1] where the source offset along segment ``a``-``b`` is checked.

    The part of the segment in the closed box (a side included, so a beam lying
    along a box side is kept whole) is checked at both of its ends when it is
    longer than ``tolerance``.  Along a box axis on which the segment moves by no
    more than ``tolerance`` (it runs parallel to those sides) a point within
    tolerance of a side counts as on it, as the cut classifies it, so roundoff in
    a rotated box frame cannot drop a beam lying in a side; along the axes it
    crosses, the sides are exact.  A shorter contact -- the segment touches the box,
    reaches into it by less than the tolerance, or passes within tolerance of it
    -- is checked at one point: its end station within tolerance of the box (the
    nearer one), else its point nearest the box; never further out, toward an
    outside station whose offset may rightly differ.  ``()`` when it misses.
    """

    start, end = volume.local(np.asarray(a, dtype=float))[0], volume.local(np.asarray(b, dtype=float))[0]
    length = float(np.linalg.norm(end - start))
    run = _slab(volume, a, b, 0.0, parallel_tolerance=tolerance)
    if run is not None and (run[1] - run[0]) * length > tolerance:
        return run
    touch = _slab(volume, a, b, tolerance)
    if touch is None:
        return ()
    lower, upper = np.asarray(volume.lower), np.asarray(volume.upper)

    def distance(t: float) -> float:  # to the box itself; convex along the segment
        q = start + t * (end - start)
        return float(np.linalg.norm(np.maximum(np.maximum(lower - q, q - upper), 0.0)))

    ends = [t for t in (0.0, 1.0) if distance(t) <= tolerance]
    if ends:
        return (min(ends, key=distance),)
    if run is not None:
        return (0.5 * (run[0] + run[1]),)
    low, high = touch
    for _ in range(100):
        third = (high - low) / 3.0
        if distance(low + third) <= distance(high - third):
            high -= third
        else:
            low += third
    return (0.5 * (low + high),)


def _slab(volume: Any, a: np.ndarray, b: np.ndarray, tolerance: float, *,
          parallel_tolerance: float = 0.0) -> Optional[Tuple[float, float]]:
    """The parameter interval of segment ``a``-``b`` inside the box enlarged by ``tolerance``, or None.

    On an axis along which the segment moves by no more than ``parallel_tolerance``
    the box is enlarged by ``max(tolerance, parallel_tolerance)`` instead, and the
    whole segment must lie within it.
    """

    start, end = volume.local(np.asarray(a, dtype=float))[0], volume.local(np.asarray(b, dtype=float))[0]
    lower, upper = np.asarray(volume.lower) - tolerance, np.asarray(volume.upper) + tolerance
    t0, t1 = 0.0, 1.0
    for axis in range(3):
        step = end[axis] - start[axis]
        if abs(step) <= parallel_tolerance or step == 0.0:
            band = max(tolerance, parallel_tolerance)
            if (min(start[axis], end[axis]) < volume.lower[axis] - band
                    or max(start[axis], end[axis]) > volume.upper[axis] + band):
                return None
            continue
        near, far = (lower[axis] - start[axis]) / step, (upper[axis] - start[axis]) / step
        if near > far:
            near, far = far, near
        t0, t1 = max(t0, near), min(t1, far)
        if t0 > t1:
            return None
    return t0, t1


def _remove(paths: Sequence[Path]) -> None:
    """Best-effort removal that never masks the error being handled."""

    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


@dataclass
class PublishResult:
    run: SubmodelRun
    notification_error: Optional[BaseException] = None  # a listener that raised after the commit


def _source_files_unchanged(project: Any, store: Any, manifest: Mapping[str, Any]) -> Optional[str]:
    from .io.artifacts import _sha256

    for key in ("mesh_artifact", "result_artifact"):
        expected = manifest[key]
        artifact = project.artifacts.get(expected["id"])
        if artifact is None or descriptor(artifact) != expected:
            return f"the source {key.replace('_', ' ')} record changed"
        try:
            path = store.resolve(expected["uri"])
            if not path.is_file() or _sha256(path) != expected["sha256"]:
                return f"the source {key.replace('_', ' ')} file changed"
        except Exception:
            return f"the source {key.replace('_', ' ')} file is unreadable"
    return None


def publish_submodel_run(session: Any, staged: StagedRun, *, cancellation_token: Any = None) -> PublishResult:
    """The single commit point (contract 11.4); every check happens before the transaction."""

    _check_cancelled(cancellation_token, "publication")
    if threading.get_ident() != staged.thread_id:
        raise SubmodelError("SUBMODEL_SESSION_THREAD", "publish on the thread that took the snapshot")
    if getattr(session, "_transaction_depth", 0):
        raise SubmodelError("SUBMODEL_SESSION_BUSY", "a submodel run cannot be published inside a session transaction")
    if session.revision.id != staged.revision_id:
        raise SubmodelError("SUBMODEL_INPUTS_CHANGED", "the document changed since the run's snapshot")
    project = session.project
    problem = _source_files_unchanged(project, staged.store, staged.run.source)
    if problem is not None:
        raise SubmodelError("SUBMODEL_INPUTS_CHANGED", problem)
    mesh_ref, result_ref = staged.artifacts
    _verify_child(staged.store, mesh_ref, "mesh", staged.run, project.document_id)
    _verify_child(staged.store, result_ref, "result", staged.run, project.document_id)
    for registry, key in ((project.artifacts, mesh_ref.id), (project.artifacts, result_ref.id),
                          (project.submodel_runs, staged.run.id)):
        if key in registry:
            raise SubmodelError("SUBMODEL_ARTIFACT", f"{key!r} is already registered")

    saved = dict(project.__dict__)  # the original objects, for exact restoration
    artifacts, runs = saved["artifacts"], saved["submodel_runs"]
    checked = session.revision.id
    try:
        with session.transaction("record submodel run", solver_affecting=False):
            project.artifacts[mesh_ref.id] = mesh_ref
            project.artifacts[result_ref.id] = result_ref
            project.submodel_runs[staged.run.id] = staged.run
    except BaseException as error:
        if session.revision.id != checked and staged.run.id in session.project.submodel_runs:
            staged.published = True
            return PublishResult(staged.run, notification_error=error)
        # Not committed: the session may have restored through the project reader
        # (which migrates queued jobs).  Put the original objects back exactly.
        project.__dict__.clear()
        project.__dict__.update(saved)
        artifacts.pop(mesh_ref.id, None)
        artifacts.pop(result_ref.id, None)
        runs.pop(staged.run.id, None)
        raise
    staged.published = True
    return PublishResult(staged.run)


def discard_staged_run(staged: StagedRun, project: Any = None) -> None:
    """Remove exactly the staged files; refused once the run is registered."""

    if staged.published or (project is not None and staged.run.id in project.submodel_runs):
        raise SubmodelError("SUBMODEL_ARTIFACT", "the run is published; its files are not discarded")
    _remove(staged.files)


def run_submodel(session: Any, definition_id: str, store: Any, *, mesh_record_id: Optional[str] = None,
                 cancellation_token: Any = None) -> PublishResult:
    """Snapshot, stage and publish; the attempt's files go only when nothing committed."""

    snap = submodel_snapshot(session)
    staged = stage_submodel_run(snap, definition_id, store, mesh_record_id=mesh_record_id,
                                cancellation_token=cancellation_token)
    try:
        return publish_submodel_run(session, staged, cancellation_token=cancellation_token)
    except BaseException:
        if not staged.published and staged.run.id not in session.project.submodel_runs:
            _remove(staged.files)
        raise


# ----------------------------------------------------------------------
# staleness (contract 11.6)
# ----------------------------------------------------------------------
def _artifact_ref(data: Mapping[str, Any]) -> Any:
    from .model.records import ArtifactRef

    return ArtifactRef.from_dict(data)


def _by_id(document: Mapping[str, Any], key: str) -> Dict[str, Mapping[str, Any]]:
    return {str(item.get("id")): item for item in document.get(key, ()) if isinstance(item, Mapping)}


def _file_matches(store: Any, expected: Mapping[str, Any]) -> bool:
    from .io.artifacts import _sha256

    try:
        path = store.resolve(expected["uri"])
        return path.is_file() and _sha256(path) == expected["sha256"]
    except Exception:
        return False


def submodel_run_staleness(snapshot: Any, run: SubmodelRun, store: Any, *,
                           mesh_record_id: Optional[str] = None) -> Tuple[str, ...]:
    """Typed reasons why ``run`` is not current; empty means current.

    Reads the current state from a ``ProjectSnapshot`` (or ``SubmodelSnapshot``)
    and compares it with the run's manifest.  File integrity is always checked
    fresh.  No solver work.
    """

    from .jobs import analysis_hash
    from .model.records import AnalysisDefinition, OutputRequest

    snapshot = getattr(snapshot, "snapshot", snapshot)
    document = snapshot.document
    reasons: List[str] = []
    source, child = run.source, run.child

    entry = next((item for item in document.get("submodels", ())
                  if isinstance(item, Mapping) and item.get("id") == run.definition_id), None)
    if entry is None:
        reasons.append("DEFINITION_MISSING")
    elif definition_hash(entry) != run.definition_hash:
        reasons.append("DEFINITION_CHANGED")

    jobs = _by_id(document, "jobs")
    job = jobs.get(source["job_id"])
    if job is None:
        reasons.append("SOURCE_JOB_MISSING")
    else:
        current = (str(job.get("analysis_id", "")), str(job.get("status", "")), bool(job.get("partial", False)),
                   str(job.get("model_hash", "")), str(job.get("mesh_hash", "")), str(job.get("analysis_hash", "")),
                   str(job.get("input_hash", "") or "-"), job.get("result_artifact_id"))
        recorded = (source["analysis_id"], source["job_status"], source["job_partial"], source["job_model_hash"],
                    source["job_mesh_hash"], source["job_analysis_hash"], source["job_input_hash"],
                    source.get("job_result_artifact_id"))
        if current != recorded:
            reasons.append("SOURCE_JOB_CHANGED")

    analysis_entry = _by_id(document, "analyses").get(source["analysis_id"])
    if analysis_entry is None:
        reasons.append("SOURCE_ANALYSIS_MISSING")
    else:
        requests = [OutputRequest.from_dict(item) for item in document.get("output_requests", ())]
        current_hash = analysis_hash(AnalysisDefinition.from_dict(analysis_entry), {r.id: r for r in requests},
                                     document=document)
        if current_hash != source["analysis_hash"]:
            reasons.append("SOURCE_LOADING_CHANGED")

    if snapshot.revision.model_hash != source["model_hash"]:
        reasons.append("SOURCE_MODEL_CHANGED")

    record = _by_id(document, "mesh_records").get(source["mesh_record_id"])
    if (record is None or _mesh_record_fingerprint(record) != source["mesh_record_fingerprint"]
            or (mesh_record_id is not None and str(mesh_record_id) != source["mesh_record_id"])):
        reasons.append("SOURCE_MESH_CHANGED")

    artifacts = _by_id(document, "artifacts")
    for key in ("mesh_artifact", "result_artifact"):
        expected = source[key]
        current_artifact = artifacts.get(expected["id"])
        if current_artifact is None or descriptor(current_artifact) != expected or not _file_matches(store, expected):
            reasons.append("SOURCE_ARTIFACT_CHANGED")
            break

    document_id = str((document.get("anyfem") or {}).get("document_id", ""))
    for key, kind in (("mesh_artifact", "mesh"), ("result_artifact", "result")):
        current_artifact = artifacts.get(child[key]["id"])
        if current_artifact is None or _child_problem(store, _artifact_ref(current_artifact), kind, run,
                                                      document_id) is not None:
            reasons.append("RESULT_ARTIFACT_CHANGED")
            break

    if run.producer_versions.get("run") != json_safe(_producer_versions()):
        reasons.append("REVALIDATION_REQUIRED")
    return tuple(reasons)


def submodel_orphans(project: Any, store: Any) -> Tuple[Path, ...]:
    """Files under the store that no registered artifact references; deletes nothing."""

    uris = [artifact.uri for artifact in project.artifacts.values()]
    for run in project.submodel_runs.values():
        for part in (run.source, run.child):
            uris.extend(part[key]["uri"] for key in ("mesh_artifact", "result_artifact"))
    referenced = set()
    for uri in uris:
        try:
            referenced.add(store.resolve(uri))
        except Exception:
            continue
    out = []
    for folder in ("meshes", "results"):
        directory = Path(store.root) / folder
        if directory.is_dir():
            out.extend(path.resolve() for path in sorted(directory.iterdir())
                       if path.is_file() and path.resolve() not in referenced)
    return tuple(out)
