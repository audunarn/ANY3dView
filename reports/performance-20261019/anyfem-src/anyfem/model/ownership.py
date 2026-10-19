"""Persistent structural-ownership intent independent of feature history.

ANYgeometry owns feature materialization.  A multi-face structural Sheet is
ANYfem intent layered on that materialization, so it must be reapplied after a
full feature replay rather than encoded as a geometry feature or inferred by
proximity.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from anygeometry import GeometryError, Orientation, SheetTopologyPolicy
from anygeometry.entities import EntityRef
from anygeometry.features import FeatureOutputRef

__all__ = [
    "SheetJoinIntent",
    "infer_sheet_join_intents",
    "join_anchors",
    "reapply_sheet_join_intents",
    "sheet_join_intent_problems",
]


FaceAnchor = EntityRef | FeatureOutputRef


def _validate_uuid(value: object, what: str) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        raise ValueError(f"{what} must be a UUID") from None


@dataclass(frozen=True)
class SheetJoinIntent:
    """One exact ordered set of faces intended to remain one structural Sheet."""

    anchors: tuple[FaceAnchor, ...]
    orientations: tuple[Orientation | int, ...]
    policy: SheetTopologyPolicy = SheetTopologyPolicy()
    name: str = "Joined sheet"
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        anchors = tuple(self.anchors)
        if len(anchors) < 2:
            raise ValueError("Sheet Join intent needs at least two face anchors")
        if len(set(anchors)) != len(anchors):
            raise ValueError("Sheet Join intent contains duplicate face anchors")
        orientations = tuple(self.orientations)
        if len(orientations) != len(anchors):
            raise ValueError(
                "Sheet Join intent needs exactly one orientation per face anchor"
            )
        try:
            if any(isinstance(value, bool) for value in orientations):
                raise ValueError
            orientations = tuple(Orientation(value) for value in orientations)
        except (TypeError, ValueError):
            raise ValueError(
                "Sheet Join orientation must be FORWARD or REVERSED"
            ) from None
        for anchor in anchors:
            if not isinstance(anchor, (EntityRef, FeatureOutputRef)):
                raise TypeError(
                    "Sheet Join anchors must be EntityRef or FeatureOutputRef values"
                )
            if anchor.kind != "face":
                raise ValueError("Sheet Join anchors must identify faces")
        name = str(self.name)
        if not name or "\x00" in name:
            raise ValueError("Sheet Join name must be non-empty and contain no NUL")
        if not isinstance(self.policy, SheetTopologyPolicy):
            raise TypeError("Sheet Join policy must be a SheetTopologyPolicy")
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "orientations", orientations)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "id", _validate_uuid(self.id, "Sheet Join ID"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "anchors": [_anchor_to_dict(anchor) for anchor in self.anchors],
            "orientations": [int(value) for value in self.orientations],
            "policy": {
                "boundary": self.policy.boundary.value,
                "non_manifold": self.policy.non_manifold.value,
                "connectivity": self.policy.connectivity.value,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SheetJoinIntent":
        if not isinstance(value, Mapping):
            raise ValueError("Sheet Join intent must be a JSON object")
        anchors = value.get("anchors")
        if not isinstance(anchors, (list, tuple)):
            raise ValueError("Sheet Join anchors must be a list")
        orientations = value.get("orientations")
        if not isinstance(orientations, (list, tuple)):
            raise ValueError("Sheet Join orientations must be a list")
        policy = value.get("policy")
        if not isinstance(policy, Mapping):
            raise ValueError("Sheet Join policy must be a JSON object")
        return cls(
            id=str(value.get("id", "")),
            name=str(value.get("name", "Joined sheet")),
            anchors=tuple(_anchor_from_dict(item) for item in anchors),
            orientations=tuple(orientations),
            policy=SheetTopologyPolicy(
                boundary=policy.get("boundary", "allow"),
                non_manifold=policy.get("non_manifold", "reject"),
                connectivity=policy.get("connectivity", "require_connected"),
            ),
        )


def _anchor_to_dict(anchor: FaceAnchor) -> dict[str, Any]:
    if isinstance(anchor, FeatureOutputRef):
        return {
            "type": "feature_output",
            "feature_id": anchor.feature_id,
            "output_key": anchor.output_key,
            "kind": anchor.kind,
        }
    return {"type": "entity", "kind": anchor.kind, "id": anchor.id}


def _anchor_from_dict(value: object) -> FaceAnchor:
    if not isinstance(value, Mapping):
        raise ValueError("Sheet Join anchor must be a JSON object")
    anchor_type = value.get("type")
    if anchor_type == "feature_output":
        try:
            return FeatureOutputRef(
                int(value["feature_id"]),
                str(value["output_key"]),
                str(value["kind"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid feature-output Sheet Join anchor: {error}") from None
    if anchor_type == "entity":
        try:
            return EntityRef(str(value["kind"]), int(value["id"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid entity Sheet Join anchor: {error}") from None
    raise ValueError(f"unknown Sheet Join anchor type {anchor_type!r}")


def join_anchors(geometry, references: Iterable[EntityRef]) -> tuple[FaceAnchor, ...]:
    """Convert exact current faces to their strongest persistent identities."""

    output_owners: dict[EntityRef, FeatureOutputRef] = {}
    for record in geometry.features.records:
        for output_key, reference in record.outputs.items():
            output_owners[reference] = FeatureOutputRef(
                record.feature_id, output_key, reference.kind
            )
    made: list[FaceAnchor] = []
    for reference in references:
        if not isinstance(reference, EntityRef) or reference.kind != "face":
            raise GeometryError("Sheet Join intent accepts exact face EntityRefs")
        if reference.id not in geometry.faces:
            raise GeometryError(f"Sheet Join intent selected missing face {reference.id}")
        made.append(output_owners.get(reference, reference))
    return tuple(made)


def _resolve_anchor(geometry, anchor: FaceAnchor) -> EntityRef:
    try:
        if isinstance(anchor, FeatureOutputRef):
            resolved = geometry.features.resolve(anchor, geometry)
        elif anchor.id in geometry.faces:
            resolved = (anchor,)
        else:
            # Explicit replacement lineage is persistent identity evidence.
            # It is the only allowed fallback for an EntityRef; geometry
            # proximity is intentionally never queried.
            resolved = geometry.resolve_ref(anchor)
    except (KeyError, ValueError, GeometryError):
        resolved = ()
    faces = tuple(reference for reference in resolved if reference.kind == "face")
    if len(faces) != 1:
        raise GeometryError(
            f"Sheet Join anchor {anchor!r} resolves to {len(faces)} faces; "
            "exactly one is required"
        )
    return faces[0]


def _resolved_intent(geometry, intent: SheetJoinIntent) -> tuple[EntityRef, ...]:
    references = tuple(_resolve_anchor(geometry, anchor) for anchor in intent.anchors)
    if len(set(references)) != len(references):
        raise GeometryError(
            f"Sheet Join {intent.name!r} resolves multiple anchors to one face"
        )
    return references


def sheet_join_intent_problems(
    geometry, intents: Iterable[SheetJoinIntent]
) -> tuple[str, ...]:
    """Diagnose persistent joins that are not exactly materialized, read-only."""

    problems: list[str] = []
    claimed: dict[int, str] = {}
    for intent in sorted(intents, key=lambda item: item.id):
        try:
            references = _resolved_intent(geometry, intent)
        except GeometryError as error:
            problems.append(f"Sheet Join {intent.name!r}: {error}")
            continue
        overlap = next(
            (
                (reference.id, claimed[reference.id])
                for reference in references
                if reference.id in claimed
            ),
            None,
        )
        if overlap is not None:
            face_id, previous = overlap
            problems.append(
                f"Sheet Join intents {previous!r} and {intent.id!r} both "
                f"claim exact face {face_id}"
            )
            continue
        claimed.update((reference.id, intent.id) for reference in references)

        uses_by_face = {
            reference.id: tuple(
                use
                for use in geometry.face_uses.values()
                if use.face_id == reference.id
            )
            for reference in references
        }
        invalid_owner = next(
            (
                (face_id, len(uses))
                for face_id, uses in uses_by_face.items()
                if len(uses) != 1
            ),
            None,
        )
        if invalid_owner is not None:
            face_id, count = invalid_owner
            problems.append(
                f"Sheet Join {intent.name!r} face {face_id} has {count} "
                "structural owners; exactly one is required"
            )
            continue
        sheet_ids = {
            uses[0].sheet_id for uses in uses_by_face.values()
        }
        if len(sheet_ids) != 1:
            problems.append(
                f"Sheet Join {intent.name!r} is not materialized as one Sheet"
            )
            continue
        sheet_id = next(iter(sheet_ids))
        sheet = geometry.sheets[sheet_id]
        part = geometry.parts[sheet.part_id]
        actual_scope = {
            geometry.face_uses[face_use_id].face_id
            for face_use_id in sheet.face_use_ids
        }
        expected_scope = set(uses_by_face)
        if actual_scope != expected_scope:
            problems.append(
                f"Sheet Join {intent.name!r} expected faces "
                f"{sorted(expected_scope)} but Sheet {sheet_id} owns "
                f"{sorted(actual_scope)}"
            )
            continue
        if (
            part.name
            or part.metadata
            or set(part.sheet_ids) != {sheet.id}
            or part.member_ids
            or any(
                attachment.part_id == part.id
                for attachment in geometry.attachments.values()
            )
        ):
            problems.append(
                f"Sheet Join {intent.name!r} belongs to Part {part.id} with "
                "parent ownership intent that cannot be replayed losslessly"
            )
        if (
            sheet.metadata
            or sheet.declared_non_manifold_edges
            or any(
                owner.metadata
                for owners in uses_by_face.values()
                for owner in owners
            )
            or any(
                geometry.coedges[coedge_id].metadata
                for owners in uses_by_face.values()
                for owner in owners
                for coedge_id in owner.coedge_ids
            )
            or any(
                sheet.id in junction.sheet_ids
                for junction in geometry.junctions.values()
            )
            or any(
                attachment.sheet_id == sheet.id
                or attachment.source_key == ("sheet", sheet.id)
                or attachment.target_key == ("sheet", sheet.id)
                for attachment in geometry.attachments.values()
            )
        ):
            problems.append(
                f"Sheet Join {intent.name!r} carries owner metadata or "
                "references that cannot be replayed losslessly"
            )
        expected_orientations = {
            reference.id: orientation
            for reference, orientation in zip(references, intent.orientations)
        }
        actual_orientations = {
            geometry.face_uses[face_use_id].face_id:
            geometry.face_uses[face_use_id].orientation
            for face_use_id in sheet.face_use_ids
        }
        if actual_orientations != expected_orientations:
            problems.append(
                f"Sheet Join {intent.name!r} FaceUse orientations do not "
                "match persistent intent"
            )
        if sheet.policy != intent.policy:
            problems.append(
                f"Sheet Join {intent.name!r} topology policy does not match "
                "persistent intent"
            )
    return tuple(problems)


def _satisfied_sheet(
    geometry, intent: SheetJoinIntent, references: tuple[EntityRef, ...]
) -> int | None:
    face_ids = {reference.id for reference in references}
    sheet_ids: set[int] = set()
    for face_id in face_ids:
        uses = tuple(
            use for use in geometry.face_uses.values() if use.face_id == face_id
        )
        if len(uses) != 1:
            return None
        sheet_ids.add(uses[0].sheet_id)
    if len(sheet_ids) != 1:
        return None
    sheet_id = next(iter(sheet_ids))
    sheet = geometry.sheets[sheet_id]
    part = geometry.parts[sheet.part_id]
    owned = {
        geometry.face_uses[face_use_id].face_id
        for face_use_id in sheet.face_use_ids
    }
    if owned != face_ids:
        raise GeometryError(
            f"Sheet Join scope {sorted(face_ids)} conflicts with existing Sheet "
            f"{sheet_id} scope {sorted(owned)}"
        )
    if (
        part.name
        or part.metadata
        or set(part.sheet_ids) != {sheet.id}
        or part.member_ids
        or any(
            attachment.part_id == part.id
            for attachment in geometry.attachments.values()
        )
    ):
        raise GeometryError(
            f"Sheet Join {intent.name!r} belongs to Part {part.id} with "
            "parent ownership intent that cannot be replayed losslessly"
        )
    actual_orientations = {
        geometry.face_uses[face_use_id].face_id:
        geometry.face_uses[face_use_id].orientation
        for face_use_id in sheet.face_use_ids
    }
    expected_orientations = {
        reference.id: orientation
        for reference, orientation in zip(references, intent.orientations)
    }
    if (
        sheet.metadata
        or sheet.declared_non_manifold_edges
        or any(
            geometry.face_uses[face_use_id].metadata
            for face_use_id in sheet.face_use_ids
        )
        or any(
            geometry.coedges[coedge_id].metadata
            for face_use_id in sheet.face_use_ids
            for coedge_id in geometry.face_uses[face_use_id].coedge_ids
        )
        or any(
            sheet_id in junction.sheet_ids
            for junction in geometry.junctions.values()
        )
        or any(
            attachment.sheet_id == sheet_id
            or attachment.source_key == ("sheet", sheet_id)
            or attachment.target_key == ("sheet", sheet_id)
            for attachment in geometry.attachments.values()
        )
    ):
        raise GeometryError(
            f"Sheet Join {intent.name!r} carries owner metadata or references "
            "that cannot be replayed losslessly"
        )
    if (
        actual_orientations == expected_orientations
        and sheet.policy == intent.policy
    ):
        return sheet_id
    with geometry.transaction():
        part_id = sheet.part_id
        geometry.remove_sheet(sheet_id)
        return geometry.add_sheet(
            tuple(reference.id for reference in references),
            part_id=part_id,
            name=intent.name,
            policy=intent.policy,
            orientations=intent.orientations,
        )


def _apply_one(geometry, intent: SheetJoinIntent, references: tuple[EntityRef, ...]) -> int:
    satisfied = _satisfied_sheet(geometry, intent, references)
    if satisfied is not None:
        return satisfied

    for reference in references:
        uses = tuple(
            use
            for use in geometry.face_uses.values()
            if use.face_id == reference.id
        )
        if not uses:
            geometry.add_sheet(
                (reference.id,), name=f"join source plate {reference.id}"
            )
        elif len(uses) != 1 or len(geometry.sheets[uses[0].sheet_id].face_use_ids) != 1:
            raise GeometryError(
                f"face {reference.id} is not independently owned and cannot "
                f"reapply Sheet Join {intent.name!r}"
            )

    # Shared implementation with the user command.  This import is delayed to
    # avoid a module cycle while Project and commands initialize.
    from anyfem.commands import _apply_sheet_join, _plan_sheet_join

    plan = replace(
        _plan_sheet_join(geometry, references),
        orientations=intent.orientations,
        policy=intent.policy,
    )
    return _apply_sheet_join(geometry, plan, intent.name)


def reapply_sheet_join_intents(
    geometry, intents: Iterable[SheetJoinIntent]
) -> tuple[int, ...]:
    """Atomically restore all explicit Sheet joins after feature regeneration."""

    ordered = tuple(sorted(intents, key=lambda item: item.id))
    if not ordered:
        return ()
    resolved = tuple((intent, _resolved_intent(geometry, intent)) for intent in ordered)
    claimed: dict[int, str] = {}
    for intent, references in resolved:
        for reference in references:
            previous = claimed.get(reference.id)
            if previous is not None:
                raise GeometryError(
                    f"Sheet Join intents {previous!r} and {intent.id!r} both "
                    f"claim exact face {reference.id}"
                )
            claimed[reference.id] = intent.id

    # Prove the complete sequence on a detached closure before changing live
    # structural IDs.  All matching is exact identity or explicit lineage.
    working = geometry.clone(include_features=False)
    for intent, references in resolved:
        _apply_one(working, intent, references)

    results: list[int] = []
    with geometry.transaction():
        for intent, references in resolved:
            results.append(_apply_one(geometry, intent, references))
    return tuple(results)


def infer_sheet_join_intents(
    geometry, *, document_id: str
) -> dict[str, SheetJoinIntent]:
    """Infer only explicit, losslessly replayable current multi-face Sheets."""

    inferred: dict[str, SheetJoinIntent] = {}
    for sheet in sorted(geometry.sheets.values(), key=lambda item: item.id):
        if len(sheet.face_use_ids) < 2:
            continue
        # An empty Sheet name cannot be represented by the explicit Join
        # command contract without silently renaming the owner on replay.
        if not sheet.name:
            continue
        uses = tuple(geometry.face_uses[item] for item in sheet.face_use_ids)
        part = geometry.parts[sheet.part_id]
        if (
            sheet.metadata
            or sheet.declared_non_manifold_edges
            or any(use.metadata for use in uses)
            or any(
                geometry.coedges[coedge_id].metadata
                for use in uses
                for coedge_id in use.coedge_ids
            )
        ):
            continue
        if (
            part.name
            or part.metadata
            or set(part.sheet_ids) != {sheet.id}
            or part.member_ids
            or any(
                attachment.part_id == part.id
                for attachment in geometry.attachments.values()
            )
        ):
            continue
        if any(
            sum(
                other.face_id == use.face_id
                for other in geometry.face_uses.values()
            )
            != 1
            for use in uses
        ):
            continue
        if any(sheet.id in junction.sheet_ids for junction in geometry.junctions.values()):
            continue
        if any(
            attachment.sheet_id == sheet.id
            or attachment.source_key == ("sheet", sheet.id)
            or attachment.target_key == ("sheet", sheet.id)
            for attachment in geometry.attachments.values()
        ):
            continue
        adjacency = {use.face_id: set() for use in uses}
        edge_faces: dict[int, list[int]] = {}
        for use in uses:
            for coedge_id in use.coedge_ids:
                edge_faces.setdefault(
                    geometry.coedges[coedge_id].edge_id, []
                ).append(use.face_id)
        for owners in edge_faces.values():
            unique = tuple(dict.fromkeys(owners))
            if len(unique) == 2:
                first, second = unique
                adjacency[first].add(second)
                adjacency[second].add(first)
        visited: set[int] = set()
        pending = [uses[0].face_id]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(adjacency[current] - visited)
        if visited != set(adjacency):
            continue
        references = tuple(
            EntityRef("face", use.face_id)
            for use in sorted(uses, key=lambda item: item.face_id)
        )
        anchors = join_anchors(geometry, references)
        token = "|".join(repr(anchor) for anchor in anchors)
        identifier = str(
            uuid5(
                NAMESPACE_URL,
                f"{document_id}:sheet-join:{token}",
            )
        )
        intent = SheetJoinIntent(
            id=identifier,
            name=sheet.name or "Joined sheet",
            anchors=anchors,
            orientations=tuple(
                use.orientation
                for use in sorted(uses, key=lambda item: item.face_id)
            ),
            policy=sheet.policy,
        )
        inferred[intent.id] = intent
    return inferred
