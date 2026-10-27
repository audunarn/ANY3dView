"""Packed application misses never reconstruct unrelated application owners."""
from uuid import UUID

import numpy as np
import pytest

from any3dview import ApplicationOwner, MeshArrays, ModelOwner, PackedOwnerTable
from any3dview.retained_viewer import RetainedViewer


def mesh():
    return MeshArrays(
        np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]),
        np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32),
        triangle_to_element=np.array([1, 0], dtype=np.uint32),
        lines=np.array([[0, 1], [2, 3]], dtype=np.uint32),
        point_indices=np.array([0, 3], dtype=np.uint32),
    )


def table_for(kind, rows):
    argument = {"triangle": "triangles", "line": "lines", "point": "points"}[kind]
    return PackedOwnerTable.from_owners(**{argument: rows})


@pytest.mark.parametrize("kind", ["triangle", "line", "point"])
@pytest.mark.parametrize("key", ["face:1", 42])
def test_application_hit_and_miss_need_no_owner_reconstruction(monkeypatch, kind, key):
    table = table_for(kind, [(ApplicationOwner(key, "application"),),
                             (ApplicationOwner("other", "application"),)])
    def reject(*args, **kwargs):
        pytest.fail("application columns already determine this key")
    monkeypatch.setattr(PackedOwnerTable, "owner", reject)
    query = RetainedViewer._semantic_primitives_for_key
    expected = [1] if kind == "triangle" else [0]
    assert query(mesh(), table, None, str(key), kind).tolist() == expected
    assert query(mesh(), table, None, "unrelated-face", kind).tolist() == []


@pytest.mark.parametrize("kind", ["triangle", "line", "point"])
@pytest.mark.parametrize("resolved", [False, True])
def test_mixed_table_fallback_resolves_only_model_rows(monkeypatch, kind, resolved):
    identity = UUID("3e0d7a91-e385-4905-885e-b5a554294e91")
    model = ModelOwner(identity, "face", 9)
    table = table_for(kind, [(ApplicationOwner("application", "mesh.element"),), (model,)])
    calls = []
    original = PackedOwnerTable.owner
    def counted(self, row, resolver=None):
        calls.append(int(self.domains[row]))
        return original(self, row, resolver)
    monkeypatch.setattr(PackedOwnerTable, "owner", counted)
    resolver = (lambda document, model_kind, local_id: f"resolved:{local_id}") if resolved else None
    key = "resolved:9" if resolved else f"{identity}:face:9"
    query = RetainedViewer._semantic_primitives_for_key
    assert query(mesh(), table, resolver, "application", kind).tolist() == ([1] if kind == "triangle" else [0])
    assert calls == []
    assert query(mesh(), table, resolver, key, kind).tolist() == ([0] if kind == "triangle" else [1])
    assert calls == [0]
    calls.clear()
    assert query(mesh(), table, resolver, "absent", kind).tolist() == []
    assert calls == [0]


def test_repeated_owners_keep_distinct_element_mapping():
    owner = ApplicationOwner(42, "mesh.element")
    table = table_for("triangle", [(owner,), (owner,)])
    assert RetainedViewer._semantic_primitives_for_key(mesh(), table, None, "42", "triangle").tolist() == [0, 1]


def test_replaced_ownership_does_not_keep_negative_cached_mask():
    from any3dview import MeshHandle

    class Renderer:
        def set_semantic_masks(self, handle, **masks): pass
        def set_chunk_semantic_masks(self, handle, chunk_id, **masks):
            self.latest = masks
        def set_chunk_pickable(self, *args): pass
        def clear_chunk_semantic_masks(self, *args): pass

    # Exercise the same production cache invalidation callback as a live viewer
    # without requiring a GL context or toolkit event loop.
    viewer = RetainedViewer.__new__(RetainedViewer)
    viewer._renderer = Renderer()
    viewer._animation_frame_active = False
    viewer._highlighted_tags = frozenset({"new"})
    viewer._preselected_key = None
    viewer._preselection_from_hit = False
    viewer._selection_index = None
    viewer.redraw = lambda: None
    empty = MeshArrays(np.empty((0, 3)), np.empty((0, 3), dtype=np.uint32))
    first = table_for("triangle", [(ApplicationOwner("old"),), (ApplicationOwner("old"),)])
    second = table_for("triangle", [(ApplicationOwner("new"),), (ApplicationOwner("new"),)])
    handle = MeshHandle(empty, on_change=viewer._mesh_changed)
    viewer._entries = {id(handle): {"handle": handle, "owners": None,
        "owner_resolver": None, "tags": frozenset()}}
    handle.add_chunk("local", mesh(), owners=first)
    viewer._apply_highlight_masks()
    assert viewer._renderer.latest["selected_elements"].tolist() == []
    handle.set_chunk_ownership("local", second)
    viewer._apply_highlight_masks()
    assert viewer._renderer.latest["selected_elements"].tolist() == [0, 1]


@pytest.mark.parametrize("key", ["face:1",42])
def test_mesh_mask_queries_only_present_application_keys(monkeypatch,key):
    viewer=RetainedViewer.__new__(RetainedViewer)
    viewer._hidden_primitives_for_visibility=lambda *args: np.empty(0,dtype=np.uint32)
    table=table_for("triangle",[(ApplicationOwner(key),),(ApplicationOwner(key),)])
    query=RetainedViewer._semantic_primitives_for_key
    calls=[]
    def counted(*args):
        calls.append(args[3]);return query(*args)
    monkeypatch.setattr(viewer,"_semantic_primitives_for_key",counted)
    result=viewer._semantic_masks_for_mesh(mesh(),table,None,{},set(),{str(key),"absent"},None)
    assert "absent" not in calls
    assert result["selected_elements"].tolist()==[0,1]


def test_mesh_mask_preserves_dynamic_model_resolver_keys():
    viewer=RetainedViewer.__new__(RetainedViewer)
    viewer._hidden_primitives_for_visibility=lambda *args: np.empty(0,dtype=np.uint32)
    model=ModelOwner(UUID("3e0d7a91-e385-4905-885e-b5a554294e91"),"face",9)
    table=table_for("triangle",[(model,),(model,)])
    result=viewer._semantic_masks_for_mesh(mesh(),table,lambda *args:"dynamic",{},set(),{"dynamic"},None)
    assert result["selected_elements"].tolist()==[0,1]


@pytest.mark.parametrize("key",["042","+42"])
def test_mesh_mask_retains_accepted_numeric_spelling(key):
    viewer=RetainedViewer.__new__(RetainedViewer)
    viewer._hidden_primitives_for_visibility=lambda *args: np.empty(0,dtype=np.uint32)
    table=table_for("triangle",[(ApplicationOwner(42),),(ApplicationOwner(42),)])
    result=viewer._semantic_masks_for_mesh(mesh(),table,None,{},set(),{key},None)
    assert result["selected_elements"].tolist()==[0,1]
