"""Retained draw styles preserve geometry and fail atomically."""
from threading import Thread

import numpy as np
import pytest

from any3dview.retained import MeshHandle
from any3dview.retained_viewer import RetainedViewer
from test_retained_opacity import mesh, viewer as opacity_viewer, renderer


def viewer(renderer=None, **extra):
    return opacity_viewer(renderer, **dict(
        dict(lit=True, line_width=1., line_color="#334155", color="#9aa7b4",
             scalar_range=None, colormap=None), **extra))


def style(v, h, **values):
    RetainedViewer.set_mesh_style(v, h, **values)


def test_software_style_preserves_arrays_state_and_replay():
    v, h = viewer()
    arrays, generations = h.mesh, h.generations
    uploads = v._renderer.geometry_uploads
    index = v._selection_index
    v._renderer.pick_dirty = False
    style(v, h, opacity=.4, lit=False, outline="#ffffff", width=2., color="#112233")
    appearance = v._entries[id(h)]["appearance"]
    assert appearance["opacity"] == .4 and appearance["lit"] is False
    assert appearance["line_color"] == "#ffffff" and appearance["line_width"] == 2.
    assert appearance["color"] == "#112233" and not appearance["cull_backface"]
    assert h.mesh is arrays and h.generations == generations
    assert v._selection_index is index and v._renderer.geometry_uploads == uploads
    RetainedViewer._restore_live_renderer(v)
    assert v._renderer.appearance[id(h)] == appearance


@pytest.mark.parametrize("values", [
    {"opacity": True}, {"opacity": -1}, {"opacity": float("nan")},
    {"width": False}, {"width": .1}, {"width": float("inf")},
    {"width": "2"}, {"lit": 1}, {"lit": None},
    {"outline": ""}, {"outline": "not-a-color"}, {"outline": None},
    {"color": "#ff"}, {"color": 12},
])
def test_invalid_style_validates_entire_request_before_mutating(values):
    v, h = viewer()
    before = dict(v._entries[id(h)]["appearance"])
    revision = v._renderer.mask_revision
    with pytest.raises((TypeError, ValueError)):
        style(v, h, **dict({"opacity": .2}, **values))
    assert v._entries[id(h)]["appearance"] == before
    assert v._renderer.appearance[id(h)] == before
    assert v._renderer.mask_revision == revision


def test_style_ownership_thread_and_lifetime():
    v, h = viewer()
    with pytest.raises(TypeError): style(v, object(), lit=False)
    with pytest.raises(ValueError): style(v, MeshHandle(mesh()), lit=False)
    errors = []
    def wrong_thread():
        try: style(v, h, lit=False)
        except RuntimeError as error: errors.append(error)
    thread = Thread(target=wrong_thread)
    thread.start(); thread.join()
    assert len(errors) == 1 and "owning thread" in str(errors[0])
    v._closed = True
    with pytest.raises(RuntimeError, match="closed"): style(v, h, lit=False)
    v._closed = False
    h.remove()
    with pytest.raises(RuntimeError, match="removed"): style(v, h, lit=False)


@pytest.mark.parametrize("failure", ["redraw", "renderer"])
def test_style_failure_rolls_back_entire_update_and_revision(failure):
    v, h = viewer()
    before = dict(v._entries[id(h)]["appearance"])
    revision = v._renderer.mask_revision
    v._renderer.pick_dirty = False
    if failure == "redraw":
        def fail(): raise RuntimeError("redraw failure")
        v.redraw = fail
    else:
        original = v._renderer.set_mesh_style
        def fail(handle, **values):
            original(handle, **values)
            if values.get("opacity") == .2: raise RuntimeError("renderer failure")
        v._renderer.set_mesh_style = fail
    with pytest.raises(RuntimeError, match="failure"):
        style(v, h, opacity=.2, lit=False, width=3, outline="#123456", color="#abcdef")
    assert v._entries[id(h)]["appearance"] == before
    assert v._renderer.appearance[id(h)] == before
    assert v._renderer.mask_revision == revision and not v._renderer.pick_dirty


def test_noop_style_never_redraws():
    v, h = viewer()
    def fail(): raise AssertionError("unexpected redraw")
    v.redraw = fail
    style(v, h)
    style(v, h, opacity=1.)


def test_gpu_style_uses_uniforms_without_geometry_or_palette_updates(renderer):
    v, h = viewer(renderer)
    group = renderer._groups[id(h)]
    chunk = group.chunks[0][1]
    chunks, vertex = tuple(group.chunks), chunk.vertex_buffer
    fields, generations = h.mesh.element_scalars, h.generations
    uploads, palettes = renderer.geometry_uploads, renderer.palette_updates
    renderer.pick_dirty = False
    style(v, h, lit=False, outline="#ffffff", color="#112233", opacity=.2)
    renderer._sync(group)
    assert not group.lit and group.line_color == "#ffffff" and group.opacity == .2
    assert chunk.color == group.color == "#112233"
    assert tuple(group.chunks) == chunks and chunk.vertex_buffer is vertex
    assert h.mesh.element_scalars is fields and h.generations == generations
    assert renderer.geometry_uploads == uploads and renderer.palette_updates == palettes
    assert not renderer.pick_dirty
    style(v, h, width=3.)
    assert not renderer.pick_dirty  # this fixture has no line primitives
    renderer.pick_dirty = False
    def fail(): raise RuntimeError("redraw failure")
    v.redraw = fail
    with pytest.raises(RuntimeError, match="redraw failure"):
        style(v, h, lit=True, outline="#000000", color="#ffffff", opacity=.8, width=4.)
    assert not group.lit and group.line_color == "#ffffff" and group.opacity == .2
    assert chunk.color == group.color == "#112233" and group.line_width == 3.
    assert not renderer.pick_dirty
    assert tuple(group.chunks) == chunks and chunk.vertex_buffer is vertex


def test_gpu_style_stipple_opacity_invalidates_pick_only(renderer):
    v, h = viewer(renderer, stipple="gray50")
    renderer.pick_dirty = False
    generations = h.generations
    style(v, h, opacity=.2)
    assert renderer.pick_dirty and h.generations == generations


def test_gpu_existing_line_width_changes_picking_without_uploads(renderer):
    from types import SimpleNamespace
    from any3dview.arrays import MeshArrays
    arrays = mesh()
    h = MeshHandle(MeshArrays(arrays.positions, arrays.triangles,
                              lines=np.array([[0, 1]], np.uint32)))
    appearance = dict(opacity=1., lit=True, line_color="#334155", line_width=1.)
    renderer.add_mesh(h, **appearance)
    v = SimpleNamespace(_closed=False, _entries={id(h): {"handle": h, "appearance": appearance}},
                        _renderer=renderer, redraw=lambda: None)
    group = renderer._groups[id(h)]
    chunk = group.chunks[0][1]
    vertex, line_vertex = chunk.vertex_buffer, chunk.line_buffer
    uploads, generations = renderer.geometry_uploads, h.generations
    renderer.pick_dirty = False
    style(v, h, width=4.)
    assert renderer.pick_dirty and group.line_width == 4.
    assert chunk.vertex_buffer is vertex and chunk.line_buffer is line_vertex
    assert renderer.geometry_uploads == uploads and h.generations == generations


@pytest.mark.parametrize("failure", ["renderer", "redraw"])
def test_gpu_style_failure_after_mutation_preserves_every_draw_option(renderer, failure):
    v, h = viewer(renderer)
    group = renderer._groups[id(h)]
    chunk = group.chunks[0][1]
    if failure == "renderer":
        original = renderer.set_mesh_style
        def fail(handle, **values):
            original(handle, **values)
            if values.get("opacity") == .2: raise RuntimeError("renderer failure")
        renderer.set_mesh_style = fail
    else:
        def fail(): raise RuntimeError("redraw failure")
        v.redraw = fail
    renderer.pick_dirty = False
    with pytest.raises(RuntimeError, match="failure"):
        style(v, h, opacity=.2, lit=False, color="#ffffff", outline="#112233", width=5.)
    assert group.opacity == 1. and group.lit and group.line_width == 1.
    assert group.color == chunk.color == "#9aa7b4" and group.line_color == "#334155"
    assert not renderer.pick_dirty

