"""Incremental renderer-neutral layer for ANYgeometry schema-4 models."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Empty, SimpleQueue
from threading import get_ident
from typing import Any, Iterable, Optional

import numpy as np

try:
    from anygeometry import ChangeSet, EntityHandle, GeometryModel, ResolutionStatus
except ImportError as error:  # pragma: no cover - isolated wheel test
    raise ImportError(
        "ANYgeometry integration requires: pip install ANY3dView[geometry]"
    ) from error

from ...arrays import MeshArrays
from ...ownership import ModelOwner, PackedOwnerTable
from ...retained import MeshHandle
from .policy import DisplayMode, DisplayPolicy
from .tessellation import UnsupportedDisplayGeometry, sampled_edge, tessellate_face


ChunkKey = tuple[str, int]


class GeometryLayer:
    """Translate immutable kernel records into local replaceable display chunks."""

    def __init__(
        self,
        model: GeometryModel,
        display_policy: DisplayPolicy = DisplayPolicy(),
    ) -> None:
        if not isinstance(model, GeometryModel):
            raise TypeError("model must be an ANYgeometry GeometryModel")
        if not isinstance(display_policy, DisplayPolicy):
            raise TypeError("display_policy must be DisplayPolicy")
        self.model = model
        self.policy = display_policy
        self.revision = model.revision
        self._queue: SimpleQueue[ChangeSet] = SimpleQueue()
        self._viewer: Any = None
        self._owner_thread = get_ident()
        self._scheduled = False
        self._poll_id: Any = None
        self._closed = False
        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending_jobs: dict[ChunkKey, tuple[int, Future[Any]]] = {}
        self._job_serial = 0
        self._handles: dict[ChunkKey, MeshHandle] = {}
        self._entity_chunks: dict[tuple[str, int], set[ChunkKey]] = defaultdict(set)
        self._entity_generations: dict[tuple[str, int], int] = defaultdict(int)
        self._face_cache: dict[tuple[object, ...], Any] = {}
        self._edge_cache: dict[tuple[object, ...], np.ndarray] = {}
        self.diagnostics: list[str] = []
        model.add_change_hook(self._on_change)

    def _cached_face(self, face_id: int):
        key = (
            self.model.model_id,
            int(face_id),
            self._entity_generations[("face", int(face_id))],
            self.policy.lod,
            self.policy.tessellation,
        )
        made = self._face_cache.get(key)
        if made is None:
            made = tessellate_face(
                self.model, face_id, self.policy.tessellation, self.policy.lod
            )
            self._face_cache[key] = made
        return made

    def _cached_edge(self, edge_id: int) -> np.ndarray:
        key = (
            self.model.model_id,
            int(edge_id),
            self._entity_generations[("edge", int(edge_id))],
            self.policy.lod,
            self.policy.tessellation,
        )
        made = self._edge_cache.get(key)
        if made is None:
            made = sampled_edge(
                self.model, edge_id, self.policy.tessellation, self.policy.lod
            )
            self._edge_cache[key] = made
        return made

    def _bump(self, kind: str, identifier: int) -> None:
        key = (str(kind), int(identifier))
        self._entity_generations[key] += 1
        if kind == "face":
            self._face_cache = {
                cache_key: value
                for cache_key, value in self._face_cache.items()
                if int(cache_key[1]) != int(identifier)
            }
        elif kind == "edge":
            self._edge_cache = {
                cache_key: value
                for cache_key, value in self._edge_cache.items()
                if int(cache_key[1]) != int(identifier)
            }

    @property
    def handles(self) -> tuple[tuple[ChunkKey, MeshHandle], ...]:
        return tuple(sorted(self._handles.items()))

    def _on_change(self, change_set: ChangeSet) -> None:
        try:
            if not isinstance(change_set, ChangeSet):
                return
            self._queue.put(change_set)
            if (
                self._viewer is not None
                and get_ident() == self._owner_thread
                and not self._scheduled
            ):
                self._scheduled = True
                self._viewer.after_idle(self.process_pending)
        except Exception:
            # Kernel observers must never make a successful transaction fail.
            return

    def attach(self, viewer: Any) -> "GeometryLayer":
        if self._closed:
            raise RuntimeError("geometry layer has been closed")
        if self._viewer is not None and self._viewer is not viewer:
            raise RuntimeError("geometry layer is already attached")
        if not callable(getattr(viewer, "add_mesh_arrays", None)):
            raise TypeError("viewer does not implement add_mesh_arrays")
        self._viewer = viewer
        self._owner_thread = get_ident()
        if self.policy.threaded_updates:
            if not callable(getattr(viewer, "submit_update", None)):
                raise TypeError("threaded geometry updates require viewer.submit_update")
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="any3dview-geometry"
            )
        self._rebuild_all()
        if self.policy.threaded_updates:
            self._schedule_poll()
        return self

    def _schedule_poll(self) -> None:
        if self._closed or self._viewer is None:
            return
        self._poll_id = self._viewer.after(16, self._poll)

    def _poll(self) -> None:
        self._poll_id = None
        self.process_pending()
        self._schedule_poll()

    def _chunk(self, kind: str, identifier: int) -> ChunkKey:
        return kind, (int(identifier) - 1) // self.policy.chunk_span

    def _face_records(self) -> list[tuple[int, tuple[ModelOwner, ...], bool]]:
        model_id = self.model.model_id
        mode = self.policy.mode
        records: list[tuple[int, tuple[ModelOwner, ...], bool]] = []
        if mode in {DisplayMode.STRUCTURAL, DisplayMode.COMBINED} and self.model.face_uses:
            for use in self.model.face_uses.values():
                sheet = self.model.sheets[use.sheet_id]
                owners = (
                    ModelOwner(model_id, "face", use.face_id),
                    ModelOwner(model_id, "face_use", use.id, 1),
                    ModelOwner(model_id, "sheet", sheet.id, 2),
                    ModelOwner(model_id, "part", sheet.part_id, 3),
                )
                records.append((use.face_id, owners, int(use.orientation) < 0))
        elif mode is not DisplayMode.RELATIONSHIPS:
            records.extend(
                (face.id, (ModelOwner(model_id, "face", face.id),), False)
                for face in self.model.faces.values()
            )
        return records

    def _member_records(self) -> list[tuple[int, tuple[ModelOwner, ...]]]:
        if self.policy.mode not in {
            DisplayMode.STRUCTURAL,
            DisplayMode.RELATIONSHIPS,
            DisplayMode.COMBINED,
        }:
            return []
        model_id = self.model.model_id
        return [
            (
                member.id,
                (
                    ModelOwner(model_id, "member", member.id, 2),
                    ModelOwner(model_id, "part", member.part_id, 3),
                ),
            )
            for member in self.model.members.values()
        ]

    def _edge_records(self) -> list[tuple[int, tuple[ModelOwner, ...]]]:
        if self.policy.mode not in {
            DisplayMode.GEOMETRY,
            DisplayMode.TOPOLOGY_DEBUG,
            DisplayMode.COMBINED,
        }:
            return []
        model_id = self.model.model_id
        return [
            (edge.id, (ModelOwner(model_id, "edge", edge.id),))
            for edge in self.model.edges.values()
        ]

    def _vertex_records(self) -> list[tuple[int, tuple[ModelOwner, ...]]]:
        if self.policy.mode not in {
            DisplayMode.TOPOLOGY_DEBUG,
            DisplayMode.COMBINED,
        }:
            return []
        model_id = self.model.model_id
        return [
            (vertex.id, (ModelOwner(model_id, "vertex", vertex.id),))
            for vertex in self.model.vertices.values()
        ]

    def _relationship_enabled(self) -> bool:
        return self.policy.mode in {DisplayMode.RELATIONSHIPS, DisplayMode.COMBINED}

    @staticmethod
    def _range_middle(value: Any) -> float:
        return 0.5 * (float(value.start) + float(value.end))

    def _member_point(self, member_id: int, parameter: float) -> np.ndarray:
        member = self.model.members[int(member_id)]
        value = min(1.0, max(0.0, float(parameter)))
        uses = [self.model.member_edge_uses[item] for item in member.edge_use_ids]
        use = next(
            (
                item
                for item in uses
                if item.parent_range.start - 1.0e-12
                <= value
                <= item.parent_range.end + 1.0e-12
            ),
            uses[-1],
        )
        span = max(1.0e-15, use.parent_range.end - use.parent_range.start)
        local = (value - use.parent_range.start) / span
        if int(use.orientation) < 0:
            local = 1.0 - local
        return np.asarray(
            self.model.evaluate_edge_many(use.edge_id, np.asarray([local]))[0],
            dtype=np.float64,
        )

    def _entity_point(
        self,
        kind: str,
        identifier: int,
        parameters: tuple[Any, ...],
    ) -> np.ndarray:
        if kind == "vertex":
            return np.asarray(self.model.vertices[identifier].position, dtype=np.float64)
        if kind == "edge":
            parameter = self._range_middle(parameters[0]) if parameters else 0.5
            return np.asarray(
                self.model.evaluate_edge_many(identifier, np.asarray([parameter]))[0],
                dtype=np.float64,
            )
        if kind == "face":
            uv = np.asarray(
                [self._range_middle(item) for item in parameters[:2]]
                if len(parameters) >= 2
                else [0.5, 0.5],
                dtype=np.float64,
            )
            return np.asarray(
                self.model.evaluate_face_many(identifier, uv[None, :])[0],
                dtype=np.float64,
            )
        if kind == "member":
            parameter = self._range_middle(parameters[0]) if parameters else 0.5
            return self._member_point(identifier, parameter)
        if kind == "sheet":
            sheet = self.model.sheets[identifier]
            use = self.model.face_uses[sheet.face_use_ids[0]]
            return self._entity_point("face", use.face_id, parameters)
        raise UnsupportedDisplayGeometry(f"cannot locate relationship entity {kind}{identifier}")

    def _face_chunk(self, bucket: int) -> tuple[MeshArrays, PackedOwnerTable, set[tuple[str, int]]]:
        positions: list[np.ndarray] = []
        triangles: list[np.ndarray] = []
        bindings: list[tuple[ModelOwner, ...]] = []
        entities: set[tuple[str, int]] = set()
        cursor = 0
        for face_id, owners, reversed_use in self._face_records():
            if self._chunk("face", face_id)[1] != bucket:
                continue
            try:
                made = self._cached_face(face_id)
            except UnsupportedDisplayGeometry as error:
                self.diagnostics.append(str(error))
                continue
            local = made.triangles[:, ::-1] if reversed_use else made.triangles
            positions.append(made.positions)
            triangles.append(local + cursor)
            bindings.extend([owners] * len(local))
            cursor += len(made.positions)
            entities.add(("face", face_id))
            face = self.model.faces[face_id]
            entities.update(("edge", item.edge) for item in face.loop)
            entities.update(("edge", item.edge) for loop in face.holes for item in loop)
        points = np.concatenate(positions) if positions else np.empty((0, 3), np.float64)
        indices = np.concatenate(triangles) if triangles else np.empty((0, 3), np.uint32)
        return (
            MeshArrays(points, indices),
            PackedOwnerTable.from_owners(triangles=bindings),
            entities,
        )

    def _member_chunk(self, bucket: int) -> tuple[MeshArrays, PackedOwnerTable, set[tuple[str, int]]]:
        positions: list[np.ndarray] = []
        lines: list[tuple[int, int]] = []
        bindings: list[tuple[ModelOwner, ...]] = []
        entities: set[tuple[str, int]] = set()
        cursor = 0
        for member_id, owners in self._member_records():
            if self._chunk("member", member_id)[1] != bucket:
                continue
            member = self.model.members[member_id]
            for use_id in member.edge_use_ids:
                use = self.model.member_edge_uses[use_id]
                edge_points = self._cached_edge(use.edge_id)
                if int(use.orientation) < 0:
                    edge_points = edge_points[::-1]
                positions.append(edge_points)
                lines.extend((cursor + index, cursor + index + 1) for index in range(len(edge_points) - 1))
                edge_owners = (
                    ModelOwner(self.model.model_id, "edge", use.edge_id),
                    ModelOwner(self.model.model_id, "member_edge_use", use.id, 1),
                    *owners,
                )
                bindings.extend([edge_owners] * (len(edge_points) - 1))
                cursor += len(edge_points)
                entities.update(
                    {
                        ("member", member_id),
                        ("member_edge_use", use.id),
                        ("edge", use.edge_id),
                    }
                )
        points = np.concatenate(positions) if positions else np.empty((0, 3), np.float64)
        line_array = np.asarray(lines, dtype=np.uint32).reshape((-1, 2))
        mesh = MeshArrays(points, np.empty((0, 3), np.uint32), lines=line_array)
        return mesh, PackedOwnerTable.from_owners(lines=bindings), entities

    def _edge_chunk(self, bucket: int) -> tuple[MeshArrays, PackedOwnerTable, set[tuple[str, int]]]:
        positions: list[np.ndarray] = []
        lines: list[tuple[int, int]] = []
        bindings: list[tuple[ModelOwner, ...]] = []
        entities: set[tuple[str, int]] = set()
        cursor = 0
        for edge_id, owners in self._edge_records():
            if self._chunk("edge", edge_id)[1] != bucket:
                continue
            edge_points = self._cached_edge(edge_id)
            positions.append(edge_points)
            count = max(0, len(edge_points) - 1)
            lines.extend((cursor + index, cursor + index + 1) for index in range(count))
            bindings.extend([owners] * count)
            cursor += len(edge_points)
            entities.add(("edge", edge_id))
            edge = self.model.edges[edge_id]
            entities.update({("vertex", edge.start), ("vertex", edge.end)})
        points = np.concatenate(positions) if positions else np.empty((0, 3), np.float64)
        line_array = np.asarray(lines, dtype=np.uint32).reshape((-1, 2))
        return (
            MeshArrays(points, np.empty((0, 3), np.uint32), lines=line_array),
            PackedOwnerTable.from_owners(lines=bindings),
            entities,
        )

    def _vertex_chunk(self, bucket: int) -> tuple[MeshArrays, PackedOwnerTable, set[tuple[str, int]]]:
        records = [
            (vertex_id, owners)
            for vertex_id, owners in self._vertex_records()
            if self._chunk("vertex", vertex_id)[1] == bucket
        ]
        positions = np.asarray(
            [self.model.vertices[vertex_id].position for vertex_id, _ in records],
            dtype=np.float64,
        ).reshape((-1, 3))
        points = np.arange(len(records), dtype=np.uint32)
        return (
            MeshArrays(
                positions,
                np.empty((0, 3), np.uint32),
                point_indices=points,
            ),
            PackedOwnerTable.from_owners(points=[owners for _vertex, owners in records]),
            {("vertex", vertex_id) for vertex_id, _owners in records},
        )

    def _attachment_chunk(self, bucket: int) -> tuple[MeshArrays, PackedOwnerTable, set[tuple[str, int]]]:
        positions: list[np.ndarray] = []
        lines: list[tuple[int, int]] = []
        points: list[int] = []
        line_owners: list[tuple[ModelOwner, ...]] = []
        point_owners: list[tuple[ModelOwner, ...]] = []
        entities: set[tuple[str, int]] = set()
        for attachment in self.model.attachments.values():
            if self._chunk("attachment", attachment.id)[1] != bucket:
                continue
            try:
                source = self._entity_point(
                    attachment.source_kind,
                    attachment.source_id,
                    (attachment.member_range,),
                )
                target = self._entity_point(
                    attachment.target_kind.value,
                    attachment.target_id,
                    attachment.target_parameters,
                )
            except (KeyError, UnsupportedDisplayGeometry) as error:
                self.diagnostics.append(f"attachment {attachment.id}: {error}")
                continue
            cursor = len(positions)
            positions.extend((source, target, 0.5 * (source + target)))
            lines.append((cursor, cursor + 1))
            points.append(cursor + 2)
            owners = (
                ModelOwner(self.model.model_id, "attachment", attachment.id),
                ModelOwner(
                    self.model.model_id,
                    attachment.source_kind,
                    attachment.source_id,
                    1,
                ),
                ModelOwner(
                    self.model.model_id,
                    attachment.target_kind.value,
                    attachment.target_id,
                    1,
                ),
            )
            line_owners.append(owners)
            point_owners.append(owners)
            entities.update(
                {
                    ("attachment", attachment.id),
                    (attachment.source_kind, attachment.source_id),
                    (attachment.target_kind.value, attachment.target_id),
                }
            )
        array = np.asarray(positions, dtype=np.float64).reshape((-1, 3))
        mesh = MeshArrays(
            array,
            np.empty((0, 3), np.uint32),
            lines=np.asarray(lines, dtype=np.uint32).reshape((-1, 2)),
            point_indices=np.asarray(points, dtype=np.uint32),
        )
        return (
            mesh,
            PackedOwnerTable.from_owners(lines=line_owners, points=point_owners),
            entities,
        )

    def _junction_chunk(self, bucket: int) -> tuple[MeshArrays, PackedOwnerTable, set[tuple[str, int]]]:
        positions: list[np.ndarray] = []
        bindings: list[tuple[ModelOwner, ...]] = []
        entities: set[tuple[str, int]] = set()
        for junction in self.model.junctions.values():
            if self._chunk("junction", junction.id)[1] != bucket:
                continue
            member_points = [
                self._member_point(
                    use.member_id, self._range_middle(use.member_range)
                )
                for use in junction.member_uses
            ]
            if not member_points:
                continue
            positions.append(np.mean(member_points, axis=0))
            owners = (
                ModelOwner(self.model.model_id, "junction", junction.id),
                *(
                    ModelOwner(self.model.model_id, "member", identifier, 1)
                    for identifier in junction.member_ids
                ),
            )
            bindings.append(owners)
            entities.add(("junction", junction.id))
            entities.update(("member", identifier) for identifier in junction.member_ids)
            entities.update(("attachment", identifier) for identifier in junction.attachment_ids)
        array = np.asarray(positions, dtype=np.float64).reshape((-1, 3))
        mesh = MeshArrays(
            array,
            np.empty((0, 3), np.uint32),
            point_indices=np.arange(len(array), dtype=np.uint32),
        )
        return mesh, PackedOwnerTable.from_owners(points=bindings), entities

    def _desired_chunks(self) -> set[ChunkKey]:
        chunks = {self._chunk("face", face_id) for face_id, _owners, _reverse in self._face_records()}
        chunks.update(self._chunk("member", member_id) for member_id, _owners in self._member_records())
        chunks.update(self._chunk("edge", edge_id) for edge_id, _owners in self._edge_records())
        chunks.update(
            self._chunk("vertex", vertex_id)
            for vertex_id, _owners in self._vertex_records()
        )
        if self._relationship_enabled():
            chunks.update(
                self._chunk("attachment", identifier)
                for identifier in self.model.attachments
            )
            chunks.update(
                self._chunk("junction", identifier)
                for identifier in self.model.junctions
            )
        return chunks

    def _build_chunk(self, key: ChunkKey):
        builders = {
            "face": self._face_chunk,
            "member": self._member_chunk,
            "edge": self._edge_chunk,
            "vertex": self._vertex_chunk,
            "attachment": self._attachment_chunk,
            "junction": self._junction_chunk,
        }
        return builders[key[0]](key[1])

    @staticmethod
    def _resolve_owner(model_id, kind, identifier):
        return EntityHandle(model_id, kind, identifier)

    def _apply_chunk(
        self,
        key: ChunkKey,
        snapshot: tuple[MeshArrays, PackedOwnerTable, set[tuple[str, int]]],
    ) -> None:
        previous = self._handles.pop(key, None)
        if previous is not None:
            previous.remove()
        for entity in tuple(self._entity_chunks):
            self._entity_chunks[entity].discard(key)
            if not self._entity_chunks[entity]:
                del self._entity_chunks[entity]
        mesh, owners, entities = snapshot
        if (
            mesh.triangle_count == 0
            and (mesh.lines is None or len(mesh.lines) == 0)
            and (mesh.point_indices is None or len(mesh.point_indices) == 0)
        ):
            return
        handle = self._viewer.add_mesh_arrays(
            mesh,
            owners=owners,
            owner_resolver=self._resolve_owner,
            color="#9aa7b4",
            line_color="#334155",
        )
        self._handles[key] = handle
        handle.set_transform(self._document_transform())
        for entity in entities:
            self._entity_chunks[entity].add(key)

    def _replace_chunk(self, key: ChunkKey) -> None:
        self._apply_chunk(key, self._build_chunk(key))

    @staticmethod
    def _finalize_snapshot(
        snapshot: tuple[MeshArrays, PackedOwnerTable, set[tuple[str, int]]],
    ) -> tuple[MeshArrays, PackedOwnerTable, set[tuple[str, int]]]:
        mesh, owners, entities = snapshot
        # Workers never receive a live model.  Owning the numeric snapshot also
        # makes stale-result disposal independent of kernel record lifetimes.
        return mesh.owned_copy(), owners, set(entities)

    def _complete_chunk(self, key: ChunkKey, serial: int, future: Future[Any]) -> None:
        current = self._pending_jobs.get(key)
        if self._closed or current is None or current[0] != serial:
            return
        del self._pending_jobs[key]
        try:
            snapshot = future.result()
        except Exception as error:  # pragma: no cover - defensive worker boundary
            self.diagnostics.append(f"chunk {key} worker failed: {error}")
            return
        self._apply_chunk(key, snapshot)

    def _replace_chunks(self, keys: Iterable[ChunkKey]) -> None:
        ordered = tuple(sorted(set(keys)))
        if self._executor is None:
            for key in ordered:
                self._replace_chunk(key)
            return
        for key in ordered:
            snapshot = self._build_chunk(key)
            self._job_serial += 1
            serial = self._job_serial
            future = self._executor.submit(self._finalize_snapshot, snapshot)
            self._pending_jobs[key] = (serial, future)

            def completed(
                made: Future[Any],
                *,
                chunk_key: ChunkKey = key,
                token: int = serial,
            ) -> None:
                viewer = self._viewer
                if viewer is None:
                    return
                try:
                    viewer.submit_update(self._complete_chunk, chunk_key, token, made)
                except RuntimeError:
                    return

            future.add_done_callback(completed)

    def _document_transform(self) -> np.ndarray:
        if not self.policy.external_coordinates or self.model.coordinate_transform is None:
            return np.eye(4, dtype=np.float64)
        return np.asarray(self.model.coordinate_transform, dtype=np.float64)

    def _apply_document_transform(self) -> None:
        transform = self._document_transform()
        for handle in self._handles.values():
            handle.set_transform(transform)

    def _rebuild_all(self) -> None:
        self._job_serial += 1
        self._pending_jobs.clear()
        for handle in tuple(self._handles.values()):
            handle.remove()
        self._handles.clear()
        self._entity_chunks.clear()
        self._face_cache.clear()
        self._edge_cache.clear()
        self.diagnostics.clear()
        for key in sorted(self._desired_chunks()):
            self._replace_chunk(key)
        self.revision = self.model.revision

    def _closure(self, changes: Iterable[tuple[str, int]]) -> set[ChunkKey]:
        chunks: set[ChunkKey] = set()
        for kind, identifier in changes:
            key = (str(kind), int(identifier))
            self._bump(kind, identifier)
            chunks.update(self._entity_chunks.get(key, ()))
            if kind == "face":
                chunks.add(self._chunk("face", identifier))
            elif kind == "member":
                chunks.add(self._chunk("member", identifier))
            elif kind == "edge" and identifier in self.model.edges:
                for face_id in self.model.faces_using_edge(identifier):
                    self._bump("face", face_id)
                    chunks.add(self._chunk("face", face_id))
                for member_id in self.model.members_using_edge(identifier):
                    self._bump("member", member_id)
                    chunks.add(self._chunk("member", member_id))
            elif kind == "vertex" and identifier in self.model.vertices:
                chunks.add(self._chunk("vertex", identifier))
                for edge_id in self.model.edges_using_vertex(identifier):
                    chunks.update(self._closure((("edge", edge_id),)))
            elif kind == "face_use" and identifier in self.model.face_uses:
                chunks.add(self._chunk("face", self.model.face_uses[identifier].face_id))
            elif kind == "member_edge_use" and identifier in self.model.member_edge_uses:
                chunks.add(
                    self._chunk("member", self.model.member_edge_uses[identifier].member_id)
                )
            elif kind == "attachment":
                chunks.add(self._chunk("attachment", identifier))
            elif kind == "junction":
                chunks.add(self._chunk("junction", identifier))
        return chunks

    def process_pending(self) -> None:
        if self._closed or self._viewer is None:
            return
        self._scheduled = False
        pending: list[ChangeSet] = []
        while True:
            try:
                pending.append(self._queue.get_nowait())
            except Empty:
                break
        if not pending:
            return
        expected = self.revision
        affected: set[ChunkKey] = set()
        document_changed = False
        for change in pending:
            if change.revision_before != expected:
                self._rebuild_all()
                return
            expected = change.revision_after
            changed = {
                *change.added,
                *change.removed,
                *change.modified,
                *change.invalidated_caches,
                *change.ownership_changes,
                *change.member_changes,
                *change.attachment_changes,
                *change.tag_changes,
            }
            affected.update(self._closure(changed))
            document_changed |= change.document_settings_changed
        desired = self._desired_chunks()
        affected.update(set(self._handles) - desired)
        affected.update(desired - set(self._handles))
        self._replace_chunks(affected)
        if document_changed:
            self._apply_document_transform()
        self.revision = expected

    def resolve_selection(self, handles: Iterable[EntityHandle]) -> tuple[EntityHandle, ...]:
        resolved: list[EntityHandle] = []
        for handle in handles:
            result = self.model.resolve_handle(handle)
            if result.status is ResolutionStatus.ACTIVE:
                resolved.append(handle)
            elif result.status is ResolutionStatus.REPLACED:
                resolved.extend(result.resolved)
        return tuple(sorted(set(resolved)))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.model.remove_change_hook(self._on_change)
        if self._poll_id is not None and self._viewer is not None:
            self._viewer.after_cancel(self._poll_id)
        for handle in tuple(self._handles.values()):
            handle.remove()
        self._handles.clear()
        self._pending_jobs.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._viewer = None
