import importlib.util

import numpy as np
import pytest

from any3dview import (
    Camera3D,
    MeshArrays,
    MeshHandle,
    PackedOwnerTable,
    PickBinding,
    SectionPlane,
    SelectionConfig,
    SelectionDepth,
    create_viewer,
)


moderngl = pytest.importorskip("moderngl")


@pytest.fixture
def renderer():
    from any3dview.gpu.renderer import ModernGLRenderer

    try:
        context = moderngl.create_standalone_context(require=330)
    except Exception as error:  # pragma: no cover - depends on CI display/driver
        pytest.skip(f"no standalone OpenGL 3.3 context: {error}")
    value = ModernGLRenderer(context)
    yield value
    value.release()
    context.release()


def mesh(**fields):
    values = {
        "positions": np.asarray(
            [[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32
        ),
        "triangles": np.asarray([[0, 1, 2]], dtype=np.uint32),
    }
    values.update(fields)
    return MeshArrays(**values)


def test_gpu_renderer_draws_and_integer_picks(renderer):
    handle = MeshHandle(mesh())
    renderer.add_mesh(handle)
    framebuffer = renderer.ctx.simple_framebuffer((64, 64), components=4)

    renderer.render(Camera3D(), (64, 64), target=framebuffer)
    picked = renderer.pick(32, 32, Camera3D(), (64, 64))

    assert renderer.draw_calls == 1
    assert picked == (handle, "triangle", 0)
    framebuffer.release()


def test_gpu_renderer_frustum_culls_offscreen_chunks(renderer):
    handle = MeshHandle(
        MeshArrays(
            np.asarray(
                [[-1, 1000, -1], [1, 1000, -1], [0, 1000, 1]],
                dtype=np.float32,
            ),
            np.asarray([[0, 1, 2]], dtype=np.uint32),
        )
    )
    renderer.add_mesh(handle)
    framebuffer = renderer.ctx.simple_framebuffer((64, 64), components=4)

    renderer.render(Camera3D(), (64, 64), target=framebuffer)

    assert renderer.draw_calls == 0
    framebuffer.release()


def test_gpu_renderer_accepts_empty_retained_chunk(renderer):
    handle = MeshHandle(
        MeshArrays(
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint32),
        )
    )
    renderer.add_mesh(handle)
    framebuffer = renderer.ctx.simple_framebuffer((32, 32), components=4)

    renderer.render(Camera3D(), (32, 32), target=framebuffer)

    assert renderer.draw_calls == 0
    framebuffer.release()


def test_section_plane_and_active_mask_affect_rendered_picking(renderer):
    handle = MeshHandle(mesh())
    renderer.add_mesh(handle)
    camera = Camera3D()

    assert renderer.pick(32, 32, camera, (64, 64)) == (handle, "triangle", 0)
    renderer.pick_dirty = True
    assert renderer.pick(
        32, 32, camera, (64, 64), SectionPlane((1, 0, 0), 2.0)
    ) is None

    handle.set_active_elements([False])
    assert renderer.pick(32, 32, camera, (64, 64)) is None


def test_scalar_and_deformation_scale_updates_do_not_upload_geometry(renderer):
    handle = MeshHandle(
        mesh(
            displacements=np.zeros((3, 3), dtype=np.float32),
            element_scalars=np.asarray([0.0], dtype=np.float32),
        )
    )
    renderer.add_mesh(handle)
    framebuffer = renderer.ctx.simple_framebuffer((32, 32), components=4)
    uploads = renderer.geometry_uploads

    handle.update_element_scalars(np.asarray([1.0], dtype=np.float32))
    handle.set_deformation_scale(5.0)
    renderer.render(Camera3D(), (32, 32), target=framebuffer)

    assert renderer.geometry_uploads == uploads
    framebuffer.release()


def test_node_scalar_updates_use_field_buffer_without_geometry_upload(renderer):
    handle = MeshHandle(
        mesh(node_scalars=np.asarray([0.0, 0.5, 1.0], dtype=np.float32))
    )
    renderer.add_mesh(handle)
    framebuffer = renderer.ctx.simple_framebuffer((32, 32), components=4)
    uploads = renderer.geometry_uploads

    handle.update_node_scalars(np.asarray([1.0, 0.5, 0.0], dtype=np.float32))
    renderer.render(Camera3D(), (32, 32), target=framebuffer)

    assert renderer.geometry_uploads == uploads
    framebuffer.release()


def test_gpu_renderer_draws_and_picks_instanced_lines(renderer):
    arrays = MeshArrays(
        np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float32),
        np.empty((0, 3), np.uint32),
        lines=np.asarray([[0, 1]], np.uint32),
    )
    handle = MeshHandle(arrays)
    renderer.add_mesh(handle, line_width=3.0)
    framebuffer = renderer.ctx.simple_framebuffer((64, 64), components=4)

    renderer.render(Camera3D(), (64, 64), target=framebuffer)

    assert renderer.draw_calls == 1
    assert renderer.pick(32, 32, Camera3D(), (64, 64)) == (handle, "line", 0)
    renderer.pick_dirty = True
    assert renderer.pick(
        32, 32, Camera3D(), (64, 64), SectionPlane((1, 0, 0), 2.0)
    ) is None
    framebuffer.release()


def test_gpu_renderer_draws_and_picks_points(renderer):
    arrays = MeshArrays(
        np.asarray([[0.0, 0.0, 0.0]], np.float32),
        np.empty((0, 3), np.uint32),
        point_indices=np.asarray([0], np.uint32),
    )
    handle = MeshHandle(arrays)
    renderer.add_mesh(handle, point_size=11.0)
    framebuffer = renderer.ctx.simple_framebuffer((64, 64), components=4)

    renderer.render(Camera3D(), (64, 64), target=framebuffer)

    assert renderer.draw_calls == 1
    assert renderer.pick(32, 32, Camera3D(), (64, 64)) == (handle, "point", 0)
    renderer.pick_dirty = True
    assert renderer.pick(
        32, 32, Camera3D(), (64, 64), SectionPlane((1, 0, 0), 1.0)
    ) is None
    framebuffer.release()


def test_backend_factory_rejects_unknown_backend_before_importing_gui():
    with pytest.raises(ValueError, match="backend"):
        create_viewer(None, backend="vulkan")


def test_explicit_gpu_disable_has_actionable_diagnostic(monkeypatch):
    from any3dview import GPUUnavailableError

    monkeypatch.setenv("ANY3DVIEW_DISABLE_GPU", "1")
    with pytest.raises(GPUUnavailableError, match="disabled") as caught:
        create_viewer(None, backend="gpu")
    assert "ANY3DVIEW_DISABLE_GPU" in " ".join(caught.value.diagnostics)


def test_gpu_extra_is_present_in_development_environment():
    assert importlib.util.find_spec("moderngl") is not None
    assert importlib.util.find_spec("tkinter_gl") is not None


class _FramebufferHost:
    @staticmethod
    def framebuffer_size():
        return 64, 64


def _bare_gpu_view(entries):
    from any3dview.gpu.widget import Any3DView

    viewer = Any3DView.__new__(Any3DView)
    viewer.camera = Camera3D()
    viewer.camera.set_orbit(azimuth=0.0, elevation=0.0, distance=10.0)
    viewer._host = _FramebufferHost()
    viewer._entries = entries
    viewer._section_plane = None
    viewer._selection_index = None
    viewer._selection_index_key = None
    viewer._selection_config = SelectionConfig()
    return viewer


def test_gpu_cpu_selection_index_supports_visible_through_box_and_lasso():
    positions = np.asarray(
        [[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
        np.float32,
    )
    entries = {}
    for index, x_offset in enumerate((1.0, 0.0), start=1):
        handle = MeshHandle(
            MeshArrays(
                positions + np.asarray([x_offset, 0.0, 0.0], np.float32),
                np.asarray([[0, 1, 2]], np.uint32),
            )
        )
        entries[id(handle)] = {
            "handle": handle,
            "owners": PackedOwnerTable.from_owners(
                triangles=[PickBinding.one(f"face:{index}", "geometry.face")]
            ),
            "owner_resolver": None,
        }
    viewer = _bare_gpu_view(entries)
    visible = SelectionConfig(depth=SelectionDepth.VISIBLE)
    through = SelectionConfig(depth=SelectionDepth.THROUGH)

    assert [hit.key for hit in viewer.query_point(32, 32, config=visible)] == [
        "face:1"
    ]
    assert {hit.key for hit in viewer.query_point(32, 32, config=through)} == {
        "face:1",
        "face:2",
    }
    assert {hit.key for hit in viewer.query_rectangle(
        (20, 20, 44, 44), crossing=True, config=through
    )} == {"face:1", "face:2"}
    assert {hit.key for hit in viewer.query_lasso(
        ((20, 20), (44, 20), (44, 44), (20, 44)), config=through
    )} == {"face:1", "face:2"}

    viewer._section_plane = SectionPlane((1, 0, 0), 0.5)
    viewer._selection_index = None
    assert [hit.key for hit in viewer.query_point(32, 32, config=through)] == [
        "face:1"
    ]
