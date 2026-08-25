from __future__ import annotations

from dataclasses import replace

import pytest

from any3dview import (
    Camera3D,
    Point3D,
    SemanticRef,
    ViewerCommand,
    ViewerCommandController,
    ViewerCommandPriority,
    ViewerState,
    VisibilityState,
    viewer_command_manifest,
)


class MockViewer:
    backend_name = "mock"
    viewport_size = (800, 600)

    def __init__(self):
        self.camera = Camera3D()
        self._selection = ()
        self._visibility = VisibilityState()
        self._callbacks = []
        self.redraws = 0

    @property
    def semantic_selection(self):
        return self._selection

    @property
    def visibility_state(self):
        return self._visibility

    def set_semantic_selection(self, values):
        self._selection = tuple(values)

    def set_visibility_state(self, state):
        self._visibility = state

    def export_view_state(self):
        return ViewerState(
            Point3D(*self.camera.position.to_tuple()),
            Point3D(*self.camera.target.to_tuple()),
            Point3D(*self.camera.world_up.to_tuple()),
            self.camera.fov,
            self.camera.near,
            self.camera.far,
            None,
            "white",
            semantic_selection=self._selection,
            visibility=self._visibility,
        )

    def apply_view_state(self, state, *, redraw=True):
        self.camera.world_up = state.camera_world_up
        self.camera.fov = state.fov
        self.camera.near = state.near
        self.camera.far = state.far
        self.camera.set_target(state.camera_target)
        self.camera.set_position(state.camera_position)
        self._selection = state.semantic_selection
        self._visibility = state.visibility
        if redraw:
            self.redraw()

    def submit_update(self, callback, *args, **kwargs):
        self._callbacks.append((callback, args, kwargs))

    def flush(self):
        while self._callbacks:
            callback, args, kwargs = self._callbacks.pop(0)
            callback(*args, **kwargs)

    def redraw(self):
        self.redraws += 1

    def fit_to_scene(self, padding=1.25, redraw=True):
        self.camera.set_orbit(distance=5.0 * padding)

    def reset_camera(self):
        self.camera = Camera3D()

    def set_section_plane(self, *args, **kwargs):
        state = self.export_view_state()
        from any3dview import SectionPlane
        self.apply_view_state(replace(state, section_plane=SectionPlane(args[0], args[1], kwargs.get("enabled", True))))

    def clear_section_plane(self):
        self.apply_view_state(replace(self.export_view_state(), section_plane=None))


FACE = SemanticRef("model", "face", 7, "11111111-1111-1111-1111-111111111111")
EDGE = SemanticRef("model", "edge", 8, "11111111-1111-1111-1111-111111111111")


def command(operation, params=None):
    return ViewerCommand(operation, params or {})


def test_manifest_is_provider_neutral_and_anyprotocol_compatible():
    manifest = viewer_command_manifest()
    assert manifest["schema"] == "any3dview.viewer_commands"
    descriptors = {item["operation"]: item for item in manifest["commands"]}
    assert descriptors["viewer.observe"]["permission_category"] == "read_only"
    assert descriptors["viewer.visibility.hide"]["reversibility"] == "undoable"
    assert descriptors["viewer.camera.set"]["input_schema"]["additionalProperties"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"operation": "viewer.nope", "params": {}},
        {"operation": "viewer.camera.zoom", "params": {"factor": float("nan")}},
        {"operation": "viewer.camera.set", "params": {"target": [1, 2]}},
        {"operation": "viewer.observe", "params": {"code": "print(1)"}},
    ],
)
def test_commands_fail_closed(payload):
    with pytest.raises((TypeError, ValueError)):
        ViewerCommand.from_dict(payload)


def test_selection_visibility_and_undo_are_atomic():
    viewer = MockViewer()
    controller = ViewerCommandController(viewer, entity_exists=lambda value: value in {FACE, EDGE})
    selected = controller.execute(command("viewer.selection.set", {"entities": [FACE.to_dict()]}))
    assert selected.status == "ok"
    hidden = controller.execute(command("viewer.visibility.hide", {"use_selection": True}))
    assert hidden.status == "ok"
    assert viewer.semantic_selection == ()
    assert not viewer.visibility_state.accepts((FACE,))
    assert viewer.visibility_state.accepts((EDGE,))
    assert controller.execute(command("viewer.undo")).status == "ok"
    assert viewer.semantic_selection == (FACE,)
    assert viewer.visibility_state.is_default
    assert controller.execute(command("viewer.redo")).status == "ok"
    assert viewer.semantic_selection == ()


def test_unknown_target_rolls_back():
    viewer = MockViewer()
    controller = ViewerCommandController(viewer, entity_exists=lambda value: value == FACE)
    result = controller.execute(command("viewer.visibility.hide", {"entities": [EDGE.to_dict()]}))
    assert result.status == "error"
    assert result.error_code == "unknown_target"
    assert viewer.visibility_state.is_default
    assert not controller.undo_available


def test_priority_is_trusted_and_fifo_within_priority():
    viewer = MockViewer()
    controller = ViewerCommandController(viewer)
    background = controller.submit(command("viewer.camera.zoom", {"factor": 2.0}), priority=ViewerCommandPriority.BACKGROUND)
    ai = controller.submit(command("viewer.camera.zoom", {"factor": 0.5}), priority=ViewerCommandPriority.AI)
    user = controller.submit(command("viewer.camera.zoom", {"factor": 0.8}), priority=ViewerCommandPriority.USER)
    assert controller.pending_commands == 3
    viewer.flush()
    assert user.done() and ai.done() and background.done()
    assert viewer.camera.distance == pytest.approx(8.0)


def test_ai_queue_pauses_during_user_interaction_and_is_bounded():
    viewer = MockViewer()
    controller = ViewerCommandController(viewer, queue_limit=1)
    controller.set_interacting(True)
    first = controller.submit(command("viewer.camera.zoom", {"factor": 0.5}))
    second = controller.submit(command("viewer.camera.zoom", {"factor": 0.5}))
    assert not first.done()
    assert second.result().error_code == "queue_full"
    controller.set_interacting(False)
    viewer.flush()
    assert first.result().status == "ok"


def test_user_priority_executes_while_ai_is_paused():
    viewer = MockViewer()
    controller = ViewerCommandController(viewer)
    controller.set_interacting(True)
    ai = controller.submit(
        command("viewer.camera.zoom", {"factor": 0.5}),
        priority=ViewerCommandPriority.AI,
    )
    user = controller.submit(
        command("viewer.camera.zoom", {"factor": 0.8}),
        priority=ViewerCommandPriority.USER,
    )
    viewer.flush()
    assert user.done()
    assert not ai.done()
    controller.set_interacting(False)
    viewer.flush()
    assert ai.done()


def test_model_identity_and_same_model_revalidation():
    viewer = MockViewer()
    existing = {FACE, EDGE}
    controller = ViewerCommandController(viewer, entity_exists=existing.__contains__)
    controller.set_model_identity("model-a")
    controller.execute(command("viewer.selection.set", {"entities": [FACE.to_dict()]}))
    controller.execute(command("viewer.visibility.hide", {"entities": [EDGE.to_dict()]}))
    assert controller.undo_available
    existing.remove(EDGE)
    assert controller.revalidate() == (EDGE,)
    assert viewer.visibility_state.is_default
    controller.set_model_identity("model-a")
    assert controller.undo_available
    controller.set_model_identity("model-b")
    assert not controller.undo_available
