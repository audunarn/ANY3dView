"""Opacity mutates appearance only, including failure and backend replay paths."""
from threading import Thread
from types import SimpleNamespace

import numpy as np
import pytest

from any3dview.arrays import MeshArrays
from any3dview.retained import MeshHandle
from any3dview.retained_viewer import RetainedViewer


def mesh():
    return MeshArrays(
        np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], np.float32),
        np.array([[0, 1, 2]], np.uint32), element_scalars=[0.5],
    )


def viewer(renderer=None, **style):
    if renderer is None:
        from any3dview.qt import _SoftwareResources
        renderer = _SoftwareResources()
    h = MeshHandle(mesh())
    appearance = dict(opacity=1., cull_backface=False, pickable=True, **style)
    renderer.add_mesh(h, **appearance)
    v = SimpleNamespace(
        _closed=False, _entries={id(h): {"handle": h, "appearance": appearance}},
        _renderer=renderer, redraw=lambda: None,
        _animation_handles=[], _animation_entries={}, _selection_index=object(),
    )
    return v, h


@pytest.mark.parametrize("value", [0., .35, 1., np.float32(.4)])
def test_software_opacity_retains_fields_generations_and_replays(value):
    v, h = viewer()
    arrays, generations = h.mesh, h.generations
    index = v._selection_index
    h.set_visible(False)
    h.set_selected_elements([0])
    generations = h.generations
    uploads = v._renderer.geometry_uploads
    v._renderer.pick_dirty = False
    calls = []
    v.redraw = lambda: calls.append(True)
    RetainedViewer.set_mesh_opacity(v, h, value)
    assert v._entries[id(h)]["appearance"]["opacity"] == float(value)
    assert v._renderer.appearance[id(h)]["opacity"] == float(value)
    assert v._renderer.appearance[id(h)]["cull_backface"] is False
    assert h.mesh is arrays and h.generations == generations
    assert not h.visible and list(h.selected_elements) == [0]
    assert v._selection_index is index and not v._renderer.pick_dirty
    assert v._renderer.geometry_uploads == uploads
    assert len(calls) == (value != 1.)
    RetainedViewer._restore_live_renderer(v)
    assert v._renderer.appearance[id(h)]["opacity"] == float(value)


@pytest.mark.parametrize("value", [-.01, 1.01, float("nan"), float("inf"), True,
                                  np.bool_(False), "0.5", None, 1j])
def test_invalid_opacity_refused_before_mutation(value):
    v, h = viewer()
    revision = v._renderer.mask_revision
    with pytest.raises((TypeError, ValueError)):
        RetainedViewer.set_mesh_opacity(v, h, value)
    assert v._entries[id(h)]["appearance"]["opacity"] == 1.
    assert v._renderer.appearance[id(h)]["opacity"] == 1.
    assert v._renderer.mask_revision == revision


def test_ownership_lifetime_thread_refusals():
    v, h = viewer()
    other = MeshHandle(mesh())
    with pytest.raises(ValueError, match="belong"):
        RetainedViewer.set_mesh_opacity(v, other, .2)
    with pytest.raises(TypeError, match="MeshHandle"):
        RetainedViewer.set_mesh_opacity(v, object(), .2)
    errors = []
    def wrong_thread():
        try:
            RetainedViewer.set_mesh_opacity(v, h, .2)
        except RuntimeError as error:
            errors.append(error)
    thread = Thread(target=wrong_thread)
    thread.start()
    thread.join()
    assert len(errors) == 1 and "owning thread" in str(errors[0])
    v._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        RetainedViewer.set_mesh_opacity(v, h, .2)
    v._closed = False
    h.remove()
    with pytest.raises(RuntimeError, match="removed"):
        RetainedViewer.set_mesh_opacity(v, h, .2)
    assert v._entries[id(h)]["appearance"]["opacity"] == 1.


@pytest.mark.parametrize("failure", ["redraw", "renderer"])
def test_failure_restores_entry_renderer_and_picking(failure):
    v, h = viewer()
    v._renderer.pick_dirty = False
    if failure == "redraw":
        def fail():
            raise RuntimeError("redraw failure")
        v.redraw = fail
    else:
        original = v._renderer.set_mesh_opacity
        def fail(handle, opacity):
            original(handle, opacity)
            if opacity != 1.:
                raise RuntimeError("renderer failure")
        v._renderer.set_mesh_opacity = fail
    with pytest.raises(RuntimeError, match="failure"):
        RetainedViewer.set_mesh_opacity(v, h, .2)
    assert v._entries[id(h)]["appearance"]["opacity"] == 1.
    assert v._renderer.appearance[id(h)]["opacity"] == 1.
    assert not v._renderer.pick_dirty


@pytest.fixture
def renderer():
    import moderngl
    from any3dview.gpu.renderer import ModernGLRenderer
    context = moderngl.create_standalone_context(require=330)
    r = ModernGLRenderer(context)
    yield r
    r.release()
    context.release()


def test_gpu_opacity_changes_draw_options_only_and_rolls_back(renderer):
    v, h = viewer(renderer)
    group = renderer._groups[id(h)]
    chunk = group.chunks[0][1]
    chunks = tuple(group.chunks)
    vertex = chunk.vertex_buffer
    fields = h.mesh.element_scalars
    generations = h.generations
    group_generations = group.generations
    uploads = renderer.geometry_uploads
    renderer.pick_dirty = False
    RetainedViewer.set_mesh_opacity(v, h, .25)
    renderer._sync(group)
    assert group.opacity == .25 and not group.cull_backface and group.pickable
    assert tuple(group.chunks) == chunks and chunk.vertex_buffer is vertex
    assert h.mesh.element_scalars is fields and h.generations == generations
    assert group.generations == group_generations and not group.palette_dirty
    assert renderer.geometry_uploads == uploads and not renderer.pick_dirty
    def fail():
        raise RuntimeError("redraw failure")
    v.redraw = fail
    with pytest.raises(RuntimeError, match="redraw failure"):
        RetainedViewer.set_mesh_opacity(v, h, .8)
    assert group.opacity == .25 and not renderer.pick_dirty
    assert v._entries[id(h)]["appearance"]["opacity"] == .25


def test_stipple_opacity_invalidates_only_gpu_pick_image(renderer):
    v, h = viewer(renderer, stipple="gray50")
    generations = h.generations
    renderer.pick_dirty = False
    RetainedViewer.set_mesh_opacity(v, h, .2)
    assert renderer.pick_dirty and h.generations == generations
