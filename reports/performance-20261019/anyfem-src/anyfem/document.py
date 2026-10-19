"""Transactional document session and immutable solve snapshots."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping
from uuid import uuid4

from .commands import Command, CommandStack, CompositeCommand
from .model.project import Project
from .model.workplanes import Workplane

__all__ = [
    "DocumentRevision",
    "DocumentSession",
    "DocumentTransaction",
    "ProjectSnapshot",
    "canonical_hash",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DocumentRevision:
    id: str
    sequence: int
    document_hash: str
    model_hash: str
    created_utc: str
    label: str = ""


@dataclass(frozen=True)
class ProjectSnapshot:
    """A serialization-owned immutable snapshot suitable for a worker."""

    document: Mapping[str, Any]
    revision: DocumentRevision

    def thaw(self) -> Project:
        from .io.project_file import project_from_dict

        return project_from_dict(deepcopy(dict(self.document)))


@dataclass
class DocumentTransaction:
    label: str
    solver_affecting: bool = True
    committed: bool = False


class DocumentSession:
    """Non-serialized editing state around a backward-compatible Project.

    The desktop UI owns one session.  Headless callers may continue to work
    directly with :class:`Project`; scripts that want atomicity can use this
    class without importing Tk.
    """

    def __init__(
        self,
        project: Project,
        *,
        selection: Any = None,
        path: str | Path | None = None,
    ) -> None:
        self.project = project
        self.selection = selection
        self.path = None if path is None else Path(path)
        self.commands = CommandStack(project, selection=selection)
        self.dirty = False
        self.read_only = bool(getattr(project, "read_only_reason", None))
        self.mesh_cache: dict[str, Any] = {}
        self.result_cache: dict[str, Any] = {}
        # View/construction state is session-owned: changing the active plane
        # or grid must not dirty the project or stale a solver snapshot.
        self.active_workplane = Workplane()
        self._listeners: list[Callable[[DocumentRevision], None]] = []
        self._transaction_depth = 0
        self._last_saved_hash = ""
        self.revision = self._make_revision(sequence=0, label="opened")
        self._last_saved_hash = self.revision.document_hash

    # ------------------------------------------------------------------
    @contextmanager
    def transaction(
        self, label: str, *, solver_affecting: bool = True
    ) -> Iterator[DocumentTransaction]:
        if self.read_only:
            raise PermissionError("this project is open read-only")
        transaction = DocumentTransaction(label, solver_affecting)
        outermost = self._transaction_depth == 0
        if outermost:
            from .io.project_file import project_to_dict

            before = deepcopy(project_to_dict(self.project))
        else:
            before = None
        before_selection = (
            list(self.selection.items)
            if outermost
            and self.selection is not None
            and hasattr(self.selection, "items")
            else None
        )
        before_selection_filter = (
            getattr(self.selection, "_filter", None) if outermost else None
        )
        before_selection_mode = (
            getattr(self.selection, "_mode", None) if outermost else None
        )
        before_selection_rejection = (
            getattr(self.selection, "_last_rejection", None)
            if outermost
            else None
        )
        before_done = list(self.commands._done) if outermost else None
        before_undone = list(self.commands._undone) if outermost else None
        self._transaction_depth += 1
        try:
            yield transaction
            if outermost:
                # Serialization is validation: a refused revision must roll back
                # the command just like a refusal raised by its body.
                next_revision = self._make_revision(
                    sequence=self.revision.sequence + 1,
                    label=label,
                    model_hash=None if solver_affecting else self.revision.model_hash,
                )
        except BaseException:
            if outermost and before is not None:
                from .io.project_file import project_from_dict

                restored = project_from_dict(deepcopy(before))
                self.project.__dict__.clear()
                self.project.__dict__.update(restored.__dict__)
                self.commands.project = self.project
                # Trusted scripts and bulk workflows may perform several
                # low-level command-stack operations inside one transaction.
                # Roll back navigation state as well as model data so a failed
                # edit cannot leave a phantom undo item or retargeted scope.
                self.commands._done[:] = before_done or []
                self.commands._undone[:] = before_undone or []
                self.commands._notify()
                if before_selection is not None:
                    if before_selection_filter is not None and hasattr(
                        self.selection, "_filter"
                    ):
                        self.selection._filter = before_selection_filter
                        self.selection._mode = before_selection_mode
                        self.selection._items = list(before_selection)
                        self.selection._last_rejection = (
                            before_selection_rejection
                        )
                        self.selection._notify()
                    elif hasattr(self.selection, "restore"):
                        self.selection.restore(before_selection)
            raise
        else:
            if outermost:
                self._commit(next_revision, solver_affecting=solver_affecting)
                transaction.committed = True
        finally:
            self._transaction_depth -= 1

    def execute(self, command: Command, *, solver_affecting: bool = True) -> Any:
        with self.transaction(command.label, solver_affecting=solver_affecting):
            return self.commands.run(command)

    def execute_many(
        self,
        commands: Iterable[Command],
        *,
        label: str = "batch edit",
        solver_affecting: bool = True,
    ) -> list[Any]:
        composite = CompositeCommand(tuple(commands), label=label)
        if not composite.commands:
            return []
        result = self.execute(composite, solver_affecting=solver_affecting)
        return list(result)

    def undo(self) -> bool:
        if not self.commands.can_undo:
            return False
        label = self.commands.undo_label or "undo"
        with self.transaction(f"undo {label}"):
            return self.commands.undo()

    def redo(self) -> bool:
        if not self.commands.can_redo:
            return False
        label = self.commands.redo_label or "redo"
        with self.transaction(f"redo {label}"):
            return self.commands.redo()

    # ------------------------------------------------------------------
    def snapshot(self) -> ProjectSnapshot:
        from .io.project_file import project_to_dict

        document = project_to_dict(self.project)
        return ProjectSnapshot(document=deepcopy(document), revision=self.revision)

    def mark_saved(self, path: str | Path | None = None) -> None:
        if path is not None:
            self.path = Path(path)
        self._last_saved_hash = self.revision.document_hash
        self.dirty = False

    @property
    def saved_document_hash(self) -> str:
        """Hash of the last explicit save/open baseline (for recovery UI)."""

        return self._last_saved_hash

    def add_listener(self, callback: Callable[[DocumentRevision], None]) -> None:
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[DocumentRevision], None]) -> None:
        while callback in self._listeners:
            self._listeners.remove(callback)

    # ------------------------------------------------------------------
    def _commit(self, next_revision: DocumentRevision, *, solver_affecting: bool) -> None:
        previous_model_hash = self.revision.model_hash
        self.revision = next_revision
        self.dirty = next_revision.document_hash != self._last_saved_hash
        if solver_affecting and next_revision.model_hash != previous_model_hash:
            self.mesh_cache.clear()
        for callback in list(self._listeners):
            callback(next_revision)

    def _make_revision(
        self,
        *,
        sequence: int,
        label: str,
        model_hash: str | None = None,
    ) -> DocumentRevision:
        from .io.project_file import project_to_dict

        document = project_to_dict(self.project)
        return DocumentRevision(
            id=str(uuid4()),
            sequence=int(sequence),
            document_hash=canonical_hash(_document_payload(document)),
            model_hash=model_hash or canonical_hash(_model_payload(document)),
            created_utc=_utc_now(),
            label=label,
        )


def _document_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    # Hashing only removes bookkeeping fields. Copy the containers we edit,
    # rather than duplicating all coordinates and feature baseline arrays.
    # The remaining values are read synchronously by canonical_hash.
    payload = dict(document)
    header = payload.get("anyfem")
    if isinstance(header, dict):
        header = payload["anyfem"] = dict(header)
        header.pop("revision", None)
        header.pop("artifact_root", None)
    geometry = payload.get("geometry")
    if isinstance(geometry, dict):
        geometry = payload["geometry"] = dict(geometry)
        _normalize_geometry_hash_payload(geometry, include_feature_intent=True)
    return payload


def _normalize_geometry_hash_payload(
    geometry: dict[str, Any], *, include_feature_intent: bool
) -> None:
    """Remove codec/transaction bookkeeping from semantic document hashes."""

    for key in (
        "model_id",
        "revision",
        "id_state",
        "checksum",
        "replacement_history",
        "structural_replacement_history",
    ):
        geometry.pop(key, None)
    features = geometry.get("features")
    if not include_feature_intent:
        geometry.pop("features", None)
        return
    if not isinstance(features, dict):
        return
    features = geometry["features"] = dict(features)
    features.pop("next_id", None)
    baseline = features.get("baseline")
    if isinstance(baseline, dict):
        baseline = features["baseline"] = dict(baseline)
        _normalize_geometry_hash_payload(
            baseline, include_feature_intent=False
        )
    if "records" in features:
        features["records"] = _copy_hash_records(features["records"])
    for record in features.get("records", ()):
        if not isinstance(record, dict):
            continue
        for key in (
            "outputs",
            "state",
            "diagnostic",
            "materialization_checksum",
        ):
            record.pop(key, None)


def _copy_hash_records(records):
    """Detach only record mappings whose display fields will be removed."""

    return [dict(item) if isinstance(item, dict) else item for item in records]


def _model_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    # Solver-affecting intent.  Labels, display units, jobs and result indexes
    # are deliberately omitted so presentation changes do not stale results.
    keys = (
        "geometry",
        "materials",
        "plate_sections",
        "beam_sections",
        "face_sections",
        "edge_sections",
        "assignments",
        "ownership",
        "supports",
        "masses",
        "wet_surfaces",
        "connections",
        "imperfections",
        "geometry_attachment_regions",
        "meshing",
        "regions",
        "coordinate_systems",
    )
    payload = {key: document[key] for key in keys if key in document}
    for key in (
        "materials", "plate_sections", "beam_sections", "coordinate_systems",
        "regions", "supports", "masses", "wet_surfaces", "connections",
        "imperfections",
    ):
        if key in payload:
            payload[key] = _copy_hash_records(payload[key])
    for key, records_key in (
        ("assignments", "sections"), ("ownership", "sheet_joins"),
        ("meshing", "refinements"),
    ):
        values = payload.get(key)
        if isinstance(values, dict):
            values = payload[key] = dict(values)
            if records_key in values:
                values[records_key] = _copy_hash_records(values[records_key])

    # Stable UUIDs and editable labels are provenance/navigation identity, not
    # numerical input.  Keep relationship-bearing IDs (region IDs and topology
    # IDs) but remove record UUIDs and display-only names so rename/unit changes
    # do not make an otherwise identical result stale.
    canonical_sections = bool(
        isinstance(payload.get("assignments"), dict)
        and payload["assignments"].get("sections")
    )
    if canonical_sections:
        # Redundant materialized maps contain editable section labels and
        # current topology IDs.  The UUID/region records above carry the same
        # numerical intent without making a rename or regeneration look like a
        # solver edit.
        payload.pop("face_sections", None)
        payload.pop("edge_sections", None)
    for key in ("materials", "plate_sections", "beam_sections"):
        for item in payload.get(key, ()):
            if isinstance(item, dict):
                if key in ("plate_sections", "beam_sections") and canonical_sections:
                    # Canonical assignments join by immutable section UUID, so
                    # the editable section name can now be omitted safely.
                    item.pop("name", None)
                else:
                    item.pop("id", None)
    assignments = payload.get("assignments")
    if isinstance(assignments, dict):
        for item in assignments.get("sections", ()):
            if isinstance(item, dict):
                item.pop("id", None)
                item.pop("name", None)
                item.pop("legacy_singleton", None)
    ownership = payload.get("ownership")
    if isinstance(ownership, dict):
        for item in ownership.get("sheet_joins", ()):
            if isinstance(item, dict):
                item.pop("id", None)
                item.pop("name", None)
    for item in payload.get("coordinate_systems", ()):
        if isinstance(item, dict):
            item.pop("name", None)
    for item in payload.get("regions", ()):
        if isinstance(item, dict):
            item.pop("name", None)
            item.pop("hidden", None)
    for item in payload.get("supports", ()):
        if isinstance(item, dict):
            item.pop("id", None)
            item.pop("name", None)
    for item in (*payload.get("masses", ()), *payload.get("wet_surfaces", ())):
        if isinstance(item, dict):
            item.pop("id", None)
            item.pop("name", None)
    for item in payload.get("connections", ()):
        # The id stays: a connection load in a load case refers to it.
        if isinstance(item, dict):
            item.pop("name", None)
    for item in payload.get("imperfections", ()):
        if isinstance(item, dict):
            item.pop("id", None)
            item.pop("name", None)
    meshing = payload.get("meshing")
    if isinstance(meshing, dict):
        for item in meshing.get("refinements", ()):
            if isinstance(item, dict):
                item.pop("name", None)
    geometry = payload.get("geometry")
    if isinstance(geometry, dict):
        geometry = payload["geometry"] = dict(geometry)
        _normalize_geometry_hash_payload(geometry, include_feature_intent=False)
    return payload
