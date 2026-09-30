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


class _CountingKey(str):
    """String key that counts equality tests and hashes."""

    equality_tests = 0
    hashes = 0

    def __eq__(self, other):
        type(self).equality_tests += 1
        return str.__eq__(self, other)

    def __hash__(self):
        type(self).hashes += 1
        return str.__hash__(self)


def _reset_counts():
    _CountingKey.equality_tests = 0
    _CountingKey.hashes = 0


def test_unique_string_keys_are_packed_without_a_linear_scan_per_lookup():
    """One unique key per element must not make packing quadratic."""

    count = 4000
    rows = [
        (ApplicationOwner(_CountingKey(f"element:{index}"), "mesh.element"),)
        for index in range(count)
    ]
    _reset_counts()
    table = PackedOwnerTable.from_owners(triangles=rows)

    assert table.owner_count == count
    assert table.string_keys[0] == "element:0"
    assert table.string_keys[-1] == f"element:{count - 1}"
    # list.index would need about count**2 / 2 comparisons (8 million here).
    assert _CountingKey.equality_tests < 20 * count


def test_slot_numbers_follow_first_seen_order_for_mixed_owner_kinds():
    model_id = uuid4()
    other_id = uuid4()
    table = PackedOwnerTable.from_owners(
        triangles=[
            (PickOwner("b", "k2"), ModelOwner(other_id, "face", 1)),
            (PickOwner("a", "k1"), ModelOwner(model_id, "face", 2)),
            (PickOwner("b", "k2"), ModelOwner(other_id, "face", 1)),
        ],
        lines=[(ApplicationOwner(7, "k1"),)],
    )

    assert table.string_keys == ("b", "a")
    assert table.kinds == ("k2", "face", "k1")
    assert table.documents == (other_id, model_id)
    assert table.owner_count == 5
    assert table.triangle_indices.tolist() == [0, 1, 2, 3, 0, 1]


def test_shared_binding_is_encoded_once_not_once_per_primitive():
    owners = tuple(
        ApplicationOwner(_CountingKey(f"face:{index}"), "geometry.face")
        for index in range(200)
    )
    # One tuple object repeated per primitive, as a batched surface supplies it.
    _reset_counts()
    table = PackedOwnerTable.from_owners(triangles=[owners] * 500)

    assert table.owner_count == 200
    assert table.triangle_offsets[-1] == 500 * 200
    assert table.owners_for("triangle", 499) == tuple(
        PickOwner(str(owner.key), owner.kind) for owner in owners
    )
    # Re-encoding 500 primitives x 200 owners would scan the key list 100 000
    # times; encoding the shared tuple once needs a few hundred comparisons.
    assert _CountingKey.equality_tests < 5 * len(owners)


def test_equal_owner_tuples_from_distinct_objects_give_identical_tables():
    def build(sharing):
        owners = tuple(PickOwner(f"face:{i}", "geometry.face") for i in range(6))
        if sharing:
            rows = [PickBinding(owners)] * 4 + [(), None, owners]
        else:
            rows = [
                PickBinding(tuple(PickOwner(f"face:{i}", "geometry.face") for i in range(6)))
                for _ in range(4)
            ] + [(), None, tuple(owners)]
        return PackedOwnerTable.from_owners(triangles=rows, points=[iter(owners[:2])])

    a, b = build(True), build(False)
    for field in a.__slots__:
        left, right = getattr(a, field), getattr(b, field)
        same = np.array_equal(left, right) if isinstance(left, np.ndarray) else left == right
        assert same, field


def test_temporary_owner_rows_never_alias_each_other_in_the_identity_memo():
    # Generators become temporary tuples whose ids may be reused; each row
    # must still encode its own owners.
    rows = [
        (PickOwner(f"key:{index}", "k") for _ in range(1))
        for index in range(50)
    ]
    # Generators are lazy: realise each one with its own index binding.
    rows = [iter((PickOwner(f"key:{index}", "k"),)) for index in range(50)]
    table = PackedOwnerTable.from_owners(triangles=rows)

    assert [table.owners_for("triangle", i)[0].key for i in range(50)] == [
        f"key:{i}" for i in range(50)
    ]
