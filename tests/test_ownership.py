from uuid import uuid4

import numpy as np

from any3dview import (
    ApplicationOwner,
    ModelOwner,
    PackedOwnerTable,
    PickBinding,
    PickOwner,
)


def test_owner_table_deduplicates_rows_and_uses_numeric_primitive_maps():
    binding = PickBinding.one("element:42", "mesh.element", priority=2)
    table = PackedOwnerTable.from_owners(triangles=[binding, binding])

    assert table.owner_count == 1
    assert table.triangle_offsets.tolist() == [0, 1, 2]
    assert table.triangle_indices.dtype == np.uint32
    assert table.owners_for("triangle", 1) == (
        PickOwner("element:42", "mesh.element", 2),
    )


def test_model_owner_is_materialized_only_when_resolved():
    model_id = uuid4()
    table = PackedOwnerTable.from_owners(
        triangles=[(ModelOwner(model_id, "face", 9, 4),)]
    )
    calls = []

    def resolve(document, kind, identifier):
        calls.append((document, kind, identifier))
        return ("handle", document, kind, identifier)

    unresolved = table.owners_for("triangle", 0)
    resolved = table.owners_for("triangle", 0, resolve)

    assert unresolved == (ModelOwner(model_id, "face", 9, 4),)
    assert resolved == (
        PickOwner(str(("handle", model_id, "face", 9)), "geometry.face", 4),
    )
    assert resolved[0].identity == ("handle", model_id, "face", 9)
    assert calls == [(model_id, "face", 9)]


def test_application_owner_accepts_numeric_ids_without_per_primitive_objects():
    table = PackedOwnerTable.from_owners(
        lines=[(ApplicationOwner(1001, "mesh.element"),)] * 3
    )

    assert table.owner_count == 1
    assert table.owners_for("line", 2) == (PickOwner("1001", "mesh.element"),)
