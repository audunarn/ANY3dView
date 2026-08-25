"""Focused retained semantic-mask tests for the ModernGL renderer."""

from __future__ import annotations

import numpy as np
import pytest

from any3dview import (
    Camera3D,
    MeshArrays,
    MeshHandle,
    PackedOwnerTable,
    PickOwner,
    Point3D,
)


moderngl = pytest.importorskip("moderngl")


@pytest.fixture
def renderer():
    from any3dview.gpu.renderer import ModernGLRenderer

    try:
        context = moderngl.create_standalone_context(require=330)
    except Exception as error:  # pragma: no cover - driver/display dependent
        pytest.skip(f"no standalone OpenGL 3.3 context: {error}")
    value = ModernGLRenderer(context)
    yield value
    value.release()
    context.release()


def _mixed_mesh() -> MeshArrays:
    return MeshArrays(
        np.asarray(
            [
                [-2.0, -0.6, 0.0],
                [-1.0, -0.6, 0.0],
                [-1.5, 0.6, 0.0],
                [-0.4, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [1.5, 0.0, 0.0],
            ],
            np.float32,
        ),
        np.asarray([[0, 1, 2]], np.uint32),
        lines=np.asarray([[3, 4]], np.uint32),
        point_indices=np.asarray([5], np.uint32),
    )


def _triangle_owners(key: str = "chunk") -> PackedOwnerTable:
    return PackedOwnerTable.from_owners(
        triangles=((PickOwner(key, "face"),),),
    )


def _offscreen_primary() -> MeshArrays:
    return MeshArrays(
        np.asarray(
            [[-1.0, 1000.0, 0.0], [1.0, 1000.0, 0.0], [0.0, 1001.0, 0.0]],
            np.float32,
        ),
        np.asarray([[0, 1, 2]], np.uint32),
    )


def test_semantic_masks_are_compact_distinct_and_do_not_replace_app_selection(
    renderer,
):
    handle = MeshHandle(_mixed_mesh())
    handle.set_selected_elements((0,))
    renderer.add_mesh(handle)
    chunk = renderer._groups[id(handle)].chunks[0][1]

    before = renderer.semantic_buffer_updates
    renderer.set_semantic_masks(
        handle,
        selected_elements=(0,),
        preselected_elements=(0,),
        selected_lines=(0,),
        preselected_points=(0,),
        hidden_points=(0,),
    )

    assert renderer.semantic_buffer_updates == before + 3
    assert np.frombuffer(chunk.semantic_elements.read(), np.uint8)[0] == 3
    assert np.frombuffer(chunk.semantic_lines.read(), np.uint8)[0] == 1
    assert np.frombuffer(chunk.semantic_points.read(), np.uint8)[0] == 6
    assert np.frombuffer(chunk.selected.read(), np.uint32)[0] == 1
    assert handle.selected_elements.tolist() == [0]

    # Reapplying identical sets performs no GPU writes.  A partial convenience
    # setter changes only its primitive-kind buffer and preserves every other
    # semantic state.
    renderer.set_semantic_masks(
        handle,
        selected_elements=(0,),
        preselected_elements=(0,),
        selected_lines=(0,),
        preselected_points=(0,),
        hidden_points=(0,),
    )
    assert renderer.semantic_buffer_updates == before + 3
    renderer.set_preselected_lines(handle, (0,))
    assert renderer.semantic_buffer_updates == before + 4
    assert np.frombuffer(chunk.semantic_lines.read(), np.uint8)[0] == 3
    assert np.frombuffer(chunk.semantic_elements.read(), np.uint8)[0] == 3


def test_primary_element_and_primitive_masks_are_not_reused_for_local_chunks(
    renderer,
):
    handle = MeshHandle(_mixed_mesh())
    handle.set_selected_elements((0,))
    handle.add_chunk("local", _mixed_mesh())
    renderer.add_mesh(handle)
    renderer.set_semantic_masks(
        handle,
        selected_elements=(0,),
        selected_lines=(0,),
        selected_points=(0,),
    )

    chunks = dict(renderer._groups[id(handle)].chunks)
    primary = chunks[None]
    local = chunks["local"]
    assert np.frombuffer(primary.selected.read(), np.uint32)[0] == 1
    assert np.frombuffer(primary.semantic_elements.read(), np.uint8)[0] == 1
    assert np.frombuffer(primary.semantic_lines.read(), np.uint8)[0] == 1
    assert np.frombuffer(primary.semantic_points.read(), np.uint8)[0] == 1
    assert np.frombuffer(local.selected.read(), np.uint32)[0] == 0
    assert np.frombuffer(local.semantic_elements.read(), np.uint8)[0] == 0
    assert np.frombuffer(local.semantic_lines.read(), np.uint8)[0] == 0
    assert np.frombuffer(local.semantic_points.read(), np.uint8)[0] == 0


def test_triangle_line_and_point_masks_render_with_distinct_semantic_colours(
    renderer,
):
    handle = MeshHandle(_mixed_mesh())
    renderer.add_mesh(
        handle,
        color="#0000ff",
        line_color="#0000ff",
        point_color="#0000ff",
        line_width=8.0,
        point_size=15.0,
        cull_backface=False,
        lit=False,
    )
    renderer.set_semantic_masks(
        handle,
        selected_elements=(0,),
        preselected_lines=(0,),
        selected_points=(0,),
    )
    camera = Camera3D()
    framebuffer = renderer.ctx.simple_framebuffer((160, 120), components=4)
    renderer.render(
        camera,
        (160, 120),
        target=framebuffer,
        shading_enabled=False,
        selection_color="#ff0000",
        preselection_color="#00ff00",
    )

    def sample(point: Point3D) -> bytes:
        projected = camera.project_point(point, 160, 120)
        assert projected is not None
        x, y = (int(round(value)) for value in projected)
        return framebuffer.read(
            viewport=(x, 119 - y, 1, 1), components=3, alignment=1
        )

    triangle = sample(Point3D(-1.5, -0.05, 0.0))
    line = sample(Point3D(0.0, 0.0, 0.0))
    point = sample(Point3D(1.5, 0.0, 0.0))
    assert triangle[0] > triangle[2]
    assert line[1] > line[0] and line[1] > line[2]
    assert point[0] > point[2]
    framebuffer.release()


def test_pick_can_rebuild_cache_for_included_primitive_kinds(renderer):
    triangle = MeshHandle(
        MeshArrays(
            np.asarray([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32),
            np.asarray([[0, 1, 2]], np.uint32),
        )
    )
    marker = MeshHandle(
        MeshArrays(
            np.asarray([[0, 0, 0]], np.float32),
            np.empty((0, 3), np.uint32),
            point_indices=np.asarray([0], np.uint32),
        )
    )
    renderer.add_mesh(triangle, layer=1)
    renderer.add_mesh(marker, layer=20, point_size=15.0, point_overlay=True)
    camera = Camera3D()

    assert renderer.pick(32, 32, camera, (64, 64))[1] == "point"
    assert renderer.pick(
        32,
        32,
        camera,
        (64, 64),
        primitive_kinds=("triangle",),
    ) == (triangle, "triangle", 0)
    with pytest.raises(ValueError, match="primitive_kinds"):
        renderer.pick(
            32,
            32,
            camera,
            (64, 64),
            primitive_kinds=("annotation",),
        )


def test_owned_chunks_have_detailed_pick_identity_and_unowned_chunks_do_not(
    renderer,
):
    chunk = MeshArrays(
        np.asarray([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32),
        np.asarray([[0, 1, 2]], np.uint32),
    )
    unowned = MeshHandle(_offscreen_primary())
    unowned.add_chunk("unowned", chunk)
    renderer.add_mesh(unowned, cull_backface=False)
    camera = Camera3D()

    assert renderer.pick_detail(32, 32, camera, (64, 64)) is None
    assert renderer.pick(32, 32, camera, (64, 64)) is None

    renderer.remove_mesh(unowned)
    owned = MeshHandle(_offscreen_primary())
    owned.add_chunk("owned", chunk, owners=_triangle_owners())
    renderer.add_mesh(owned, cull_backface=False)

    assert renderer.pick_detail(32, 32, camera, (64, 64)) == (
        owned,
        "triangle",
        0,
        "owned",
    )
    assert renderer.pick(32, 32, camera, (64, 64)) == (
        owned,
        "triangle",
        0,
    )


def test_chunk_masks_accept_pre_sync_add_and_survive_topology_replace(renderer):
    handle = MeshHandle(_offscreen_primary())
    renderer.add_mesh(handle, cull_backface=False)
    chunk_mesh = _mixed_mesh()
    chunk_owners = PackedOwnerTable.from_owners(
        triangles=((PickOwner("face", "face"),),),
        lines=((PickOwner("edge", "edge"),),),
        points=((PickOwner("node", "node"),),),
    )

    # The renderer still has only the primary resource when the retained
    # callback supplies the new chunk's semantic state.
    handle.add_chunk("local", chunk_mesh, owners=chunk_owners)
    renderer.set_chunk_semantic_masks(
        handle,
        "local",
        selected_elements=(0,),
        preselected_lines=(0,),
        selected_points=(0,),
    )
    framebuffer = renderer.ctx.simple_framebuffer((64, 64), components=4)
    renderer.render(Camera3D(), (64, 64), target=framebuffer)
    resource = dict(renderer._groups[id(handle)].chunks)["local"]
    assert np.frombuffer(resource.semantic_elements.read(), np.uint8)[0] == 1
    assert np.frombuffer(resource.semantic_lines.read(), np.uint8)[0] == 2
    assert np.frombuffer(resource.semantic_points.read(), np.uint8)[0] == 1
    assert np.frombuffer(resource.selected.read(), np.uint32)[0] == 0

    handle.replace_chunk("local", chunk_mesh.owned_copy())
    renderer.render(Camera3D(), (64, 64), target=framebuffer)
    replacement = dict(renderer._groups[id(handle)].chunks)["local"]
    assert replacement is not resource
    assert np.frombuffer(replacement.semantic_elements.read(), np.uint8)[0] == 1
    assert np.frombuffer(replacement.semantic_lines.read(), np.uint8)[0] == 2
    assert np.frombuffer(replacement.semantic_points.read(), np.uint8)[0] == 1
    assert np.frombuffer(replacement.selected.read(), np.uint32)[0] == 0

    renderer.clear_chunk_semantic_masks(handle, "local")
    assert np.frombuffer(replacement.semantic_elements.read(), np.uint8)[0] == 0
    assert np.frombuffer(replacement.semantic_lines.read(), np.uint8)[0] == 0
    assert np.frombuffer(replacement.semantic_points.read(), np.uint8)[0] == 0
    framebuffer.release()


def test_chunk_semantic_masks_support_tag_driven_chunks_and_reject_unknown_chunks(
    renderer,
):
    handle = MeshHandle(_offscreen_primary())
    handle.add_chunk("unowned", _mixed_mesh())
    renderer.add_mesh(handle)

    renderer.set_chunk_semantic_masks(
        handle, "unowned", selected_elements=(0,)
    )
    renderer.set_chunk_pickable(handle, "unowned", True)
    assert "unowned" in renderer._groups[id(handle)].chunk_pickable
    with pytest.raises(KeyError):
        renderer.set_chunk_semantic_masks(
            handle, "missing", selected_elements=(0,)
        )
