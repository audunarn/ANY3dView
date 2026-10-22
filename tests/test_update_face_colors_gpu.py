"""GPU in-place per-face colour updates: texture rewrite, no new GL objects.

Renderer checks use the standalone-context fixture pattern of ``test_gpu.py``;
the Qt viewer check is opt-in like the other real-GPU Qt tests.
"""
import os
import time

import numpy as np
import pytest

from any3dview import Camera3D, FaceColorUpdateError, MeshHandle

from test_update_face_colors import grid, random_hex

moderngl = pytest.importorskip("moderngl")

SIZE = (160, 96)


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


def render(renderer, camera):
    framebuffer = renderer.ctx.simple_framebuffer(SIZE, components=4)
    try:
        renderer.render(camera, SIZE, target=framebuffer, shading_enabled=False)
        return framebuffer.read(components=4, alignment=1)
    finally:
        framebuffer.release()


class AllocationSpy:
    """Count GL object creation on the renderer's context."""

    NAMES = ("buffer", "texture", "vertex_array", "framebuffer", "renderbuffer")

    def __init__(self, monkeypatch, ctx):
        self.calls = {name: 0 for name in self.NAMES}
        for name in self.NAMES:
            original = getattr(ctx, name)

            def counted(*args, _name=name, _original=original, **kwargs):
                self.calls[_name] += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(ctx, name, counted)

    def total(self):
        return sum(self.calls.values())


def scaled(mesh):
    """Shrink the grid into the default camera's view."""
    from any3dview import MeshArrays

    centred = (mesh.positions - mesh.positions.mean(axis=0)) / 16.0
    return MeshArrays(centred, mesh.triangles, triangle_to_element=mesh.triangle_to_element)


@pytest.mark.parametrize("quads", [False, True])
def test_gpu_update_is_pixel_identical_and_allocates_nothing(renderer, monkeypatch, quads):
    from any3dview.gpu import renderer as module

    mesh = scaled(grid(quads=quads))
    count = mesh.element_count if quads else mesh.triangle_count
    first, second = random_hex(count, 30), random_hex(count, 31)
    handle = MeshHandle(mesh)
    camera = Camera3D()
    renderer.add_mesh(handle, cull_backface=False, lit=False, face_colors=first)
    before = render(renderer, camera)
    group = renderer._groups[id(handle)]
    chunk = group.chunks[0][1]
    texture, uploads = chunk.face_colors, renderer.geometry_uploads
    spy = AllocationSpy(monkeypatch, renderer.ctx)
    monkeypatch.setattr(module._Chunk, "create", None)  # any rebuild would fail loudly
    renderer.update_face_colors(handle, second)
    assert spy.total() == 0
    assert renderer.geometry_uploads == uploads
    assert renderer._groups[id(handle)] is group and group.chunks[0][1] is chunk
    assert chunk.face_colors is texture
    assert renderer.face_color_updates == 1
    assert group.face_colors == second
    monkeypatch.undo()
    after = render(renderer, camera)
    renderer.add_mesh(handle, cull_backface=False, lit=False, face_colors=second)
    reference = render(renderer, camera)
    assert before != after
    assert after == reference


def test_gpu_numeric_update_matches_hex_and_survives_rebuild(renderer):
    mesh = scaled(grid())
    first, second = random_hex(600, 40), random_hex(600, 41)
    values = np.asarray([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in second], np.uint8)
    handle = MeshHandle(mesh)
    camera = Camera3D()
    renderer.add_mesh(handle, cull_backface=False, lit=False, face_colors=first)
    renderer.update_face_colors(handle, values)
    numeric = render(renderer, camera)
    # A position change rebuilds the chunk from the group's retained colours.
    handle.update_positions(mesh.positions.copy())
    rebuilt = render(renderer, camera)
    renderer.add_mesh(handle, cull_backface=False, lit=False, face_colors=second)
    assert numeric == rebuilt == render(renderer, camera)


def test_gpu_update_validation(renderer):
    mesh = scaled(grid(quads=True))
    plain = MeshHandle(mesh)
    renderer.add_mesh(plain, cull_backface=False)
    with pytest.raises(FaceColorUpdateError, match="without face colours"):
        renderer.update_face_colors(plain, random_hex(mesh.triangle_count, 1))
    handle = MeshHandle(mesh)
    first = random_hex(mesh.triangle_count, 2)
    renderer.add_mesh(handle, cull_backface=False, face_colors=first)
    with pytest.raises(FaceColorUpdateError, match="count"):
        renderer.update_face_colors(handle, random_hex(5, 3))
    assert renderer._groups[id(handle)].face_colors == first
    renderer.remove_mesh(handle)
    with pytest.raises(FaceColorUpdateError, match="not registered"):
        renderer.update_face_colors(handle, first)


def test_gpu_microbench_600_faces(renderer):
    """Report replace-the-mesh versus in-place update (renderer level, GL finish)."""
    mesh = scaled(grid())
    palettes = [random_hex(600, seed) for seed in range(6)]
    handle = MeshHandle(mesh)
    renderer.add_mesh(handle, cull_backface=False, face_colors=palettes[0])
    renderer.ctx.finish()
    start = time.perf_counter()
    for colors in palettes * 5:
        renderer.add_mesh(handle, cull_backface=False, face_colors=colors)
    renderer.ctx.finish()
    replace = (time.perf_counter() - start) / 30
    numeric = [np.asarray([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in p], np.uint8)
               for p in palettes]
    timings = {}
    for label, inputs in (("hex", palettes), ("uint8", numeric)):
        start = time.perf_counter()
        for colors in inputs * 5:
            renderer.update_face_colors(handle, colors)
        renderer.ctx.finish()
        timings[label] = (time.perf_counter() - start) / 30
    print(f"\ngpu renderer 600 faces: replace {replace * 1e3:.3f} ms, update hex "
          f"{timings['hex'] * 1e3:.3f} ms, update uint8 {timings['uint8'] * 1e3:.3f} ms")
    assert max(timings.values()) < replace


@pytest.mark.skipif(
    os.environ.get("ANY3DVIEW_RUN_QT_TESTS") != "1"
    or os.environ.get("ANY3DVIEW_RUN_QT_GPU_TESTS") != "1",
    reason="real GPU acceptance is explicitly opt-in",
)
def test_real_qt_gpu_viewer_update_matches_fresh_mesh():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from any3dview import create_viewer
    from any3dview.qt import QtViewerHostAdapter

    qapp = QApplication.instance() or QApplication([])
    mesh = grid()
    first, second = random_hex(600, 50), random_hex(600, 51)
    options = dict(cull_backface=False, tags="ocean", opacity=0.7)
    viewers = [create_viewer(None, "gpu", host=QtViewerHostAdapter()) for _ in range(2)]
    try:
        images = []
        for viewer, colors in zip(viewers, (first, second)):
            viewer.resize(320, 200); viewer.show(); qapp.processEvents()
            assert viewer.backend_name == "gpu" and viewer.capabilities.face_color_update
            handle = viewer.add_mesh_arrays(mesh, face_colors=colors, **options)
            viewer.set_top_view(); viewer.fit_to_scene(); qapp.processEvents()
            images.append(np.asarray(viewer.capture_image()).copy())
        renderer = viewers[0]._renderer
        uploads = renderer.geometry_uploads
        handles = [entry["handle"] for entry in viewers[0]._entries.values()]
        handles[0].update_face_colors(second)
        qapp.processEvents()
        updated = np.asarray(viewers[0].capture_image()).copy()
        assert viewers[0].surface.error is None
        assert renderer.geometry_uploads == uploads and renderer.face_color_updates == 1
        assert not np.array_equal(images[0], updated)
        assert np.array_equal(updated, images[1])
        x, y, _ = viewers[0].project_point((15.5, 5.5, 0.0))
        assert viewers[0].pick_at(int(x), int(y)) == "ocean"
    finally:
        for viewer in viewers:
            viewer.destroy()
        qapp.processEvents()
