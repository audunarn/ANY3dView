from dataclasses import FrozenInstanceError
from threading import Thread

import numpy as np
import pytest

from any3dview import (
    MeshArrays,
    MeshHandle,
    PackedOwnerTable,
    PickOwner,
    SelectionHit,
    ViewerCapabilities,
)


def mesh():
    return MeshArrays(
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.asarray([[0, 1, 2]], dtype=np.uint32),
    )


def chunk_mesh():
    return MeshArrays(
        np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=np.float32,
        ),
        np.asarray([[0, 1, 2]], dtype=np.uint32),
        lines=np.asarray([[0, 3]], dtype=np.uint32),
        point_indices=np.asarray([3], dtype=np.uint32),
    )


def chunk_owners(prefix="chunk"):
    return PackedOwnerTable.from_owners(
        triangles=((PickOwner(f"{prefix}:face", "face"),),),
        lines=((PickOwner(f"{prefix}:edge", "edge"),),),
        points=((PickOwner(f"{prefix}:node", "node"),),),
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


def test_chunk_ownership_extends_records_without_changing_legacy_chunks():
    handle = MeshHandle(mesh())
    chunk = chunk_mesh()
    owners = chunk_owners()

    def resolver(*coordinates):
        return coordinates

    handle.add_chunk("local", chunk, owners=owners, owner_resolver=resolver)

    assert len(handle.chunks) == 1
    chunk_id, legacy_chunk = handle.chunks[0]
    assert chunk_id == "local"
    assert legacy_chunk is chunk

    assert len(handle.chunk_records) == 1
    record_id, record_chunk, record_owners, record_resolver = handle.chunk_records[0]
    assert record_id == "local"
    assert record_chunk is chunk
    assert record_owners is owners
    assert record_resolver is resolver
    assert handle.chunk_ownership("local") == (owners, resolver)

    # Chunk arrays follow the same explicit lifetime contract as the primary mesh:
    # compatible arrays are retained, while callers can opt into isolated storage.
    assert np.shares_memory(record_chunk.positions, chunk.positions)
    source = chunk_mesh()
    owned = source.owned_copy()
    handle.add_chunk("owned", owned)
    source.positions[0, 0] = 42.0
    assert owned.positions[0, 0] == 0.0
    assert handle.chunks[1] == ("owned", owned)
    assert handle.chunk_ownership("owned") == (None, None)


@pytest.mark.parametrize("primitive_kind", ("triangle", "line", "point"))
def test_chunk_owner_mapping_lengths_must_match_each_primitive_kind(primitive_kind):
    handle = MeshHandle(mesh())
    owner_rows = (
        (PickOwner(f"{primitive_kind}:1", primitive_kind),),
        (PickOwner(f"{primitive_kind}:2", primitive_kind),),
    )
    table_keyword = {
        "triangle": "triangles",
        "line": "lines",
        "point": "points",
    }[primitive_kind]
    owners = PackedOwnerTable.from_owners(**{table_keyword: owner_rows})

    with pytest.raises(
        ValueError,
        match=rf"{primitive_kind} owner mappings must match chunk {primitive_kind} count",
    ):
        handle.add_chunk("invalid", chunk_mesh(), owners=owners)


def test_chunk_owner_resolver_requires_a_packed_owner_table():
    handle = MeshHandle(mesh())

    with pytest.raises(ValueError, match="owner_resolver requires a chunk owner table"):
        handle.add_chunk("invalid", chunk_mesh(), owner_resolver=lambda *_: None)
    with pytest.raises(TypeError, match="owners must be a PackedOwnerTable"):
        handle.add_chunk("invalid", chunk_mesh(), owners=object())


def test_chunk_ownership_survives_replacement_and_can_be_replaced_or_cleared():
    changes = []
    handle = MeshHandle(mesh(), on_change=lambda _handle, change: changes.append(change))
    owners = chunk_owners("original")
    replacement_owners = chunk_owners("replacement")

    def resolver(*coordinates):
        return ("original", coordinates)

    def replacement_resolver(*coordinates):
        return ("replacement", coordinates)

    handle.add_chunk("local", chunk_mesh(), owners=owners, owner_resolver=resolver)
    replacement = chunk_mesh().owned_copy()
    handle.replace_chunk("local", replacement)

    assert handle.chunks[0][1] is replacement
    assert handle.chunk_ownership("local") == (owners, resolver)
    assert handle.generations.topology == 2
    assert handle.generations.selection == 0

    handle.set_chunk_ownership(
        "local", replacement_owners, owner_resolver=replacement_resolver
    )
    assert handle.chunk_ownership("local") == (
        replacement_owners,
        replacement_resolver,
    )
    assert handle.generations.selection == 1
    assert handle.generations.topology == 2

    handle.set_chunk_ownership("local", None)
    assert handle.chunk_ownership("local") == (None, None)
    assert handle.generations.selection == 2

    handle.remove_chunk("local")
    assert handle.chunks == ()
    assert handle.chunk_records == ()
    assert handle.generations.topology == 3
    with pytest.raises(KeyError):
        handle.chunk_ownership("local")
    assert changes == ["topology", "topology", "selection", "selection", "topology"]


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
