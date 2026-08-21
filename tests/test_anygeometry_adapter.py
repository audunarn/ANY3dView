"""Schema-4 ANYgeometry adapter qualification tests."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import numpy as np
import pytest

anygeometry = pytest.importorskip("anygeometry")

from any3dview import MeshHandle
from any3dview.adapters.anygeometry import DisplayMode, DisplayPolicy, GeometryLayer


@dataclass
class _AddedMesh:
    handle: MeshHandle
    options: dict[str, Any]


class _QueuedViewer:
    def __init__(self) -> None:
        self.added: list[_AddedMesh] = []
        self.idle: list[Any] = []
        self.timers: dict[int, Any] = {}
        self._next_timer = 1

    def add_mesh_arrays(self, mesh, **options):
        handle = MeshHandle(mesh)
        self.added.append(_AddedMesh(handle, options))
        return handle

    def after_idle(self, callback):
        self.idle.append(callback)
        return len(self.idle)

    def after(self, _milliseconds, callback):
        identifier = self._next_timer
        self._next_timer += 1
        self.timers[identifier] = callback
        return identifier

    def after_cancel(self, identifier):
        self.timers.pop(identifier, None)

    def submit_update(self, callback, *args, **kwargs):
        self.idle.append(lambda: callback(*args, **kwargs))

    def flush_idle(self) -> None:
        while self.idle:
            callbacks, self.idle = self.idle, []
            for callback in callbacks:
                callback()


def _plate(model, x0: float, x1: float) -> tuple[int, tuple[int, ...]]:
    vertices = model.add_points(
        ((x0, 0.0, 0.0), (x1, 0.0, 0.0), (x1, 1.0, 0.0), (x0, 1.0, 0.0))
    )
    return model.add_plate(vertices), vertices


def test_layer_compiles_packed_model_owners_and_unsubscribes() -> None:
    model = anygeometry.GeometryModel()
    face, _vertices = _plate(model, 0.0, 1.0)
    viewer = _QueuedViewer()
    layer = GeometryLayer(
        model,
        DisplayPolicy(mode=DisplayMode.STRUCTURAL, chunk_span=1),
    ).attach(viewer)

    assert len(layer.handles) == 1
    _key, handle = layer.handles[0]
    assert handle.mesh.triangle_count == 2
    owners = viewer.added[0].options["owners"]
    selected = owners.owners_for(
        "triangle", 0, viewer.added[0].options["owner_resolver"]
    )
    assert selected[0].identity == anygeometry.EntityHandle(model.model_id, "face", face)
    assert selected[0].kind == "geometry.face"

    layer.close()

    assert handle.removed
    assert not layer.handles
    revision = layer.revision
    model.move_point(1, -0.25, 0.0, 0.0)
    assert layer.revision == revision
    assert not viewer.idle


def test_change_set_replaces_only_affected_chunk() -> None:
    model = anygeometry.GeometryModel()
    _face_a, vertices_a = _plate(model, 0.0, 1.0)
    _face_b, _vertices_b = _plate(model, 2.0, 3.0)
    viewer = _QueuedViewer()
    layer = GeometryLayer(
        model,
        DisplayPolicy(mode=DisplayMode.STRUCTURAL, chunk_span=1),
    ).attach(viewer)
    before = dict(layer.handles)

    model.move_point(vertices_a[0], -0.5, 0.0, 0.0)
    assert len(viewer.idle) == 1
    viewer.flush_idle()
    after = dict(layer.handles)

    assert layer.revision == model.revision
    changed = [key for key in before if after[key] is not before[key]]
    unchanged = [key for key in before if after[key] is before[key]]
    assert len(changed) == 1
    assert len(unchanged) == 1
    assert before[changed[0]].removed
    layer.close()


def test_document_transform_updates_existing_handles_only() -> None:
    model = anygeometry.GeometryModel()
    _plate(model, 0.0, 1.0)
    viewer = _QueuedViewer()
    layer = GeometryLayer(
        model,
        DisplayPolicy(mode=DisplayMode.STRUCTURAL, external_coordinates=True),
    ).attach(viewer)
    before = dict(layer.handles)
    transform = np.eye(4)
    transform[:3, 3] = (10.0, 20.0, 30.0)

    model.set_document_settings(coordinate_transform=transform)
    viewer.flush_idle()

    after = dict(layer.handles)
    assert after == before
    assert all(np.array_equal(handle.transform, transform) for handle in after.values())
    assert all(handle.generations.transform == 2 for handle in after.values())
    layer.close()


def test_revision_gap_forces_controlled_resynchronization() -> None:
    model = anygeometry.GeometryModel()
    _face, vertices = _plate(model, 0.0, 1.0)
    viewer = _QueuedViewer()
    layer = GeometryLayer(
        model,
        DisplayPolicy(mode=DisplayMode.STRUCTURAL),
    ).attach(viewer)
    original = layer.handles[0][1]

    # Simulate a producer that lost one queued model notification.
    model.move_point(vertices[0], -0.1, 0.0, 0.0)
    lost = layer._queue.get_nowait()  # noqa: SLF001 - deliberate gap injection
    assert lost.revision_before == layer.revision
    model.move_point(vertices[1], 1.1, 0.0, 0.0)
    viewer.flush_idle()

    assert layer.revision == model.revision
    assert layer.handles[0][1] is not original
    assert original.removed
    layer.close()


def test_threaded_policy_uses_bounded_poll_and_cancels_it() -> None:
    model = anygeometry.GeometryModel()
    _plate(model, 0.0, 1.0)
    viewer = _QueuedViewer()
    layer = GeometryLayer(
        model,
        DisplayPolicy(
            mode=DisplayMode.STRUCTURAL,
            threaded_updates=True,
        ),
    ).attach(viewer)

    assert len(viewer.timers) == 1
    timer = next(iter(viewer.timers))
    layer.close()
    assert timer not in viewer.timers


def test_threaded_adapter_discards_stale_worker_result() -> None:
    model = anygeometry.GeometryModel()
    _face, vertices = _plate(model, 0.0, 1.0)
    viewer = _QueuedViewer()
    layer = GeometryLayer(
        model,
        DisplayPolicy(
            mode=DisplayMode.STRUCTURAL,
            threaded_updates=True,
        ),
    ).attach(viewer)
    key, original = layer.handles[0]

    model.move_point(vertices[0], -0.1, 0.0, 0.0)
    viewer.idle.pop(0)()
    first_serial, first_future = layer._pending_jobs[key]  # noqa: SLF001
    model.move_point(vertices[1], 1.1, 0.0, 0.0)
    process_second = next(
        callback
        for callback in reversed(viewer.idle)
        if getattr(callback, "__self__", None) is layer
    )
    viewer.idle.remove(process_second)
    process_second()
    second_serial, second_future = layer._pending_jobs[key]  # noqa: SLF001

    first_future.result(timeout=2)
    second_future.result(timeout=2)
    layer._complete_chunk(key, first_serial, first_future)  # noqa: SLF001
    assert layer.handles[0][1] is original
    layer._complete_chunk(key, second_serial, second_future)  # noqa: SLF001
    assert layer.handles[0][1] is not original
    assert original.removed
    layer.close()


def test_topology_debug_policy_emits_faces_edges_and_vertices() -> None:
    model = anygeometry.GeometryModel()
    _plate(model, 0.0, 1.0)
    viewer = _QueuedViewer()
    layer = GeometryLayer(
        model,
        DisplayPolicy(mode=DisplayMode.TOPOLOGY_DEBUG, chunk_span=16),
    ).attach(viewer)

    by_kind = {key[0]: handle.mesh for key, handle in layer.handles}
    assert by_kind["face"].triangle_count == 2
    assert len(by_kind["edge"].lines) == 4
    assert len(by_kind["vertex"].point_indices) == 4
    layer.close()


def test_unchanged_entity_tessellation_is_reused_within_replaced_chunk(monkeypatch) -> None:
    layer_module = importlib.import_module(
        "any3dview.adapters.anygeometry.layer"
    )
    original = layer_module.tessellate_face
    calls: list[int] = []

    def counted(model, face_id, policy, lod):
        calls.append(face_id)
        return original(model, face_id, policy, lod)

    monkeypatch.setattr(layer_module, "tessellate_face", counted)
    model = anygeometry.GeometryModel()
    _face_a, vertices_a = _plate(model, 0.0, 1.0)
    face_b, _vertices_b = _plate(model, 2.0, 3.0)
    viewer = _QueuedViewer()
    layer = GeometryLayer(
        model,
        DisplayPolicy(mode=DisplayMode.STRUCTURAL, chunk_span=16),
    ).attach(viewer)
    assert calls.count(face_b) == 1

    model.move_point(vertices_a[0], -0.5, 0.0, 0.0)
    viewer.flush_idle()

    assert calls.count(face_b) == 1
    layer.close()


def test_selection_resolution_follows_replacement_and_rejects_terminal_handles() -> None:
    model = anygeometry.GeometryModel()
    first, second, deleted = model.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (9.0, 9.0, 9.0))
    )
    edge = model.add_line(first, second)
    active = anygeometry.EntityHandle(model.model_id, "vertex", first)
    old_edge = anygeometry.EntityHandle(model.model_id, "edge", edge)
    deleted_handle = anygeometry.EntityHandle(model.model_id, "vertex", deleted)
    wrong = anygeometry.EntityHandle(
        anygeometry.GeometryModel().model_id, "vertex", first
    )
    layer = GeometryLayer(model, DisplayPolicy(mode=DisplayMode.TOPOLOGY_DEBUG))

    model.begin_replacement_log()
    _point, replacements = model.split_edge(edge, 0.5)
    model.remove_vertex(deleted)
    resolved = layer.resolve_selection((active, old_edge, deleted_handle, wrong))

    assert active in resolved
    assert {
        handle for handle in resolved if handle.kind == "edge"
    } == {
        anygeometry.EntityHandle(model.model_id, "edge", identifier)
        for identifier in replacements
    }
    assert deleted_handle not in resolved
    assert wrong not in resolved
    layer.close()


def test_relationship_policy_emits_attachment_line_and_marker() -> None:
    model = anygeometry.GeometryModel()
    first, second, source = model.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 0.25, 0.0))
    )
    edge = model.add_line(first, second)
    attachment = model.add_attachment(
        None,
        anygeometry.AttachmentKind.VERTEX_ON_EDGE,
        anygeometry.AttachmentTargetKind.EDGE,
        edge,
        anygeometry.ParameterRange.point(0.0),
        (anygeometry.ParameterRange.point(0.5),),
        source_kind="vertex",
        source_id=source,
    )
    viewer = _QueuedViewer()
    layer = GeometryLayer(
        model,
        DisplayPolicy(mode=DisplayMode.RELATIONSHIPS, chunk_span=16),
    ).attach(viewer)

    attachment_mesh = dict(layer.handles)[("attachment", 0)].mesh
    assert len(attachment_mesh.lines) == 1
    assert len(attachment_mesh.point_indices) == 1
    owners = next(
        item.options["owners"]
        for item in viewer.added
        if item.handle is dict(layer.handles)[("attachment", 0)]
    )
    assert owners.owners_for("point", 0)[0].id == attachment
    layer.close()
