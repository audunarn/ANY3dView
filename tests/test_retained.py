from dataclasses import FrozenInstanceError
from threading import Thread

import numpy as np
import pytest

from any3dview import MeshArrays, MeshHandle, SelectionHit, PickOwner, ViewerCapabilities


def mesh():
    return MeshArrays(
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.asarray([[0, 1, 2]], dtype=np.uint32),
    )


def test_selection_hit_defaults_identity_to_legacy_owner():
    owner = PickOwner("face:1", "geometry.face")
    hit = SelectionHit(owner, 0, 1.0)

    assert hit.identity is owner


def test_retained_handle_tracks_independent_generations():
    changes = []
    handle = MeshHandle(mesh(), on_change=lambda _handle, change: changes.append(change))

    handle.update_positions(handle.mesh.positions + 1)
    handle.update_element_scalars([3.5])
    handle.set_active_elements([False])
    handle.set_selected_elements([0])
    handle.set_deformation_scale(2.0)
    handle.set_visible(False)
    handle.set_transform(np.diag([2.0, 2.0, 2.0, 1.0]))
    handle.add_chunk("local", mesh())
    handle.replace_chunk("local", mesh().owned_copy())
    handle.remove_chunk("local")

    generations = handle.generations
    assert generations.position == 1
    assert generations.scalar == 1
    assert generations.active == 1
    assert generations.selection == 1
    assert generations.displacement == 1
    assert generations.appearance == 1
    assert generations.transform == 1
    assert generations.topology == 3
    assert changes == [
        "position",
        "scalar",
        "active",
        "selection",
        "displacement",
        "appearance",
        "transform",
        "topology",
        "topology",
        "topology",
    ]


def test_remove_is_idempotent_and_other_operations_fail_afterward():
    changes = []
    handle = MeshHandle(mesh(), on_change=lambda _handle, change: changes.append(change))

    handle.remove()
    handle.remove()

    assert handle.removed
    assert changes == ["remove"]
    with pytest.raises(RuntimeError, match="removed"):
        handle.set_visible(True)


def test_updates_are_confined_to_the_owner_thread():
    handle = MeshHandle(mesh())
    errors = []

    def update():
        try:
            handle.set_visible(False)
        except Exception as error:  # noqa: BLE001 - exercise cross-thread boundary
            errors.append(error)

    worker = Thread(target=update)
    worker.start()
    worker.join()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "owning thread" in str(errors[0])


def test_capabilities_are_read_only():
    capabilities = ViewerCapabilities(dynamic_arrays=True)
    with pytest.raises(FrozenInstanceError):
        capabilities.gpu = True
