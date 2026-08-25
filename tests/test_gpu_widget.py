"""Native tkinter-gl lifecycle gates (opt in with ANY3DVIEW_RUN_GUI_TESTS=1)."""

from __future__ import annotations

import os
import statistics
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import pytest


pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(
        os.environ.get("ANY3DVIEW_RUN_GUI_TESTS") != "1",
        reason="set ANY3DVIEW_RUN_GUI_TESTS=1 on an interactive desktop",
    ),
]


def _mesh():
    from any3dview import MeshArrays

    return MeshArrays(
        np.asarray([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32),
        np.asarray([[0, 1, 2]], np.uint32),
    )


def test_gpu_widget_resize_notebook_two_viewports_and_destruction():
    from any3dview.gpu import Any3DView

    root = tk.Tk()
    root.geometry("640x360+0+0")
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)
    first_tab = ttk.Frame(notebook)
    second_tab = ttk.Frame(notebook)
    notebook.add(first_tab, text="First")
    notebook.add(second_tab, text="Second")
    first = Any3DView(first_tab, width=300, height=300)
    second = Any3DView(second_tab, width=300, height=300)
    assert first._renderer.ctx.version_code >= 330
    assert second._renderer.ctx.version_code >= 330
    first.pack(fill=tk.BOTH, expand=True)
    second.pack(fill=tk.BOTH, expand=True)
    first.add_mesh_arrays(_mesh())
    second.add_mesh_arrays(_mesh())
    root.update()

    notebook.select(second_tab)
    root.geometry("800x480+0+0")
    root.update()
    notebook.select(first_tab)
    root.update()

    assert first._host.framebuffer_size()[0] > 1
    assert second._host.framebuffer_size()[0] > 1
    assert first.renderer_diagnostics["draw_calls"] == 1
    assert second.renderer_diagnostics["draw_calls"] == 1
    second.destroy()
    first.destroy()
    root.update()
    root.destroy()


def test_gpu_widget_demand_redraw_and_repeated_create_destroy():
    from any3dview.gpu import Any3DView

    root = tk.Tk()
    root.geometry("240x180+0+0")
    for _index in range(3):
        viewer = Any3DView(root, width=240, height=180)
        viewer.pack(fill=tk.BOTH, expand=True)
        viewer.add_mesh_arrays(_mesh())
        root.update()
        frames = viewer.renderer_diagnostics["frame_count"]
        deadline = time.perf_counter() + 0.08
        while time.perf_counter() < deadline:
            root.update()
            time.sleep(0.002)
        assert viewer.renderer_diagnostics["frame_count"] == frames
        viewer.destroy()
        root.update()
    root.destroy()


def test_windows_gpu_host_prefers_core_profile(monkeypatch):
    import any3dview.gpu.host as host

    monkeypatch.delenv("ANY3DVIEW_GL_PROFILE", raising=False)
    monkeypatch.setattr(host.sys, "platform", "win32")
    assert host._gl_profile() == "4_1"

    monkeypatch.setenv("ANY3DVIEW_GL_PROFILE", "legacy")
    assert host._gl_profile() == "legacy"


def test_gpu_destroy_releases_context_before_surface(monkeypatch):
    from any3dview.gpu import Any3DView

    root = tk.Tk()
    viewer = Any3DView(root, width=240, height=180)
    viewer.pack(fill=tk.BOTH, expand=True)
    root.update()

    released = []
    context = viewer._renderer.ctx
    original_release = context.release

    def release_context():
        assert viewer._host.surface.winfo_exists()
        released.append(True)
        original_release()

    monkeypatch.setattr(context, "release", release_context)
    viewer.destroy()
    assert released == [True]
    assert viewer._renderer is None
    assert viewer._hud is None
    root.destroy()


def test_gpu_first_frame_waits_until_surface_is_mapped():
    from any3dview.gpu import Any3DView

    root = tk.Tk()
    root.geometry("320x240+0+0")
    viewer = Any3DView(root, width=320, height=240)
    viewer.add_mesh_arrays(_mesh())

    # Exercise the race where creation-time idle work runs before the GL
    # surface is packed. It must not consume the only requested frame.
    root.update_idletasks()
    assert viewer.renderer_diagnostics["frame_count"] == 0

    viewer.pack(fill=tk.BOTH, expand=True)
    root.update()
    assert viewer._host.framebuffer_size()[0] > 1
    assert viewer._host.framebuffer_size()[1] > 1
    # Tk may deliver both Map and Expose while making the widget visible.
    assert viewer.renderer_diagnostics["frame_count"] >= 1

    viewer.destroy()
    root.destroy()


def test_auto_backend_retains_software_fallback_diagnostic(monkeypatch):
    from any3dview import create_viewer

    monkeypatch.setenv("ANY3DVIEW_DISABLE_GPU", "1")
    root = tk.Tk()
    viewer = create_viewer(root, backend="auto", width=200, height=150)
    viewer.pack(fill=tk.BOTH, expand=True)
    root.update()

    assert not viewer.capabilities.gpu
    assert "ANY3DVIEW_DISABLE_GPU" in " ".join(viewer.backend_diagnostics)
    viewer.destroy()
    root.destroy()


def test_gpu_legacy_scene_state_capture_and_animation_parity():
    from any3dview import PickBinding
    from any3dview.gpu import Any3DView

    root = tk.Tk()
    root.geometry("420x300+0+0")
    viewer = Any3DView(root, width=420, height=300, bg="#f8fafc")
    viewer.pack(fill=tk.BOTH, expand=True)
    viewer.add_faces(
        (
            ((-1, -1, 0), (1, -1, 0), (0, 1, 0)),
            ((-0.5, -0.5, 0.2), (0.5, -0.5, 0.2), (0, 0.5, 0.2)),
        ),
        colors=("#60a5fa", "#f97316"),
        outline="#1e293b",
        bindings=(PickBinding.one("face:1"), PickBinding.one("face:2")),
    )
    viewer.add_line((-1, 0, 0.4), (1, 0, 0.4), draw_overlay=True, tags="edge:1")
    viewer.add_markers(((0, 0, 0.6),), tags="node:1")
    viewer.add_text((0, 0, 0.8), "GPU", color="#0f172a")
    viewer.set_thickness_legend((8.0, 10.0, 12.0), title="Thickness")
    viewer.fit_to_scene()
    root.update()

    assert viewer.backend_name == "gpu"
    assert viewer.event_widget is viewer.canvas
    assert viewer.viewport_size[0] > 1
    assert viewer.project_point((0, 0, 0)) is not None
    assert viewer.unproject_to_plane(210, 150, (0, 0, 0), (0, 0, 1)) is not None
    state = viewer.export_view_state()
    viewer.set_front_view()
    viewer.apply_view_state(state)
    viewer.set_highlight(("face:1",))
    viewer.set_preselection("face:2")
    assert viewer.highlighted_tags() == frozenset(("face:1",))

    atlas_values = {key[0] for key in viewer._hud.atlas.entries}
    assert {"GPU", "Thickness"} <= atlas_values
    assert not any(
        isinstance(child, (tk.Label, ttk.Label))
        for child in viewer.winfo_children()
    )
    viewer._render_hud(viewer.viewport_size)
    hud_colors = {
        tuple(round(value, 3) for value in vertex[5:8])
        for vertex in viewer._hud.vertices
    }
    assert tuple(round(value / 255, 3) for value in (180, 83, 9)) in hud_colors
    assert tuple(round(value / 255, 3) for value in (183, 121, 0)) in hud_colors
    base_vertices = len(viewer._hud.vertices)
    viewer._selection_dragging = True
    viewer._selection_press = (30, 30)
    viewer._selection_current = (120, 90)
    viewer._render_hud(viewer.viewport_size)
    assert len(viewer._hud.vertices) > base_vertices
    viewer._selection_dragging = False
    viewer._selection_press = None
    viewer._selection_current = None

    image = viewer.capture_image()
    assert image.size == viewer.viewport_size
    assert image.mode == "RGBA"

    viewer.begin_animation_cache()
    viewer.capture_animation_frame()
    viewer.add_text((0, 0, 0.9), "Frame two", color="#dc2626")
    viewer.set_thickness_legend((4.0, 6.0), title="Frame legend")
    viewer.capture_animation_frame()
    assert viewer.animation_frames == 2
    viewer._show_animation_frame(0)
    assert [value["text"] for value in viewer._world_text] == ["GPU"]
    assert viewer._thickness_legend["title"] == "Thickness"
    assert viewer._animation_entries
    assert any(entry["owners"] is not None for entry in viewer._animation_entries.values())
    centre = viewer.project_point((0, 0, 0))
    assert centre is not None
    assert viewer.query_point(round(centre[0]), round(centre[1]))
    viewer._show_animation_frame(1)
    assert [value["text"] for value in viewer._world_text] == ["GPU", "Frame two"]
    assert viewer._thickness_legend["title"] == "Frame legend"
    viewer.play_animation(fps=20)
    root.update()
    assert viewer.is_playing_animation
    viewer.stop_animation()
    assert not viewer.is_playing_animation

    viewer.clear(keep_canvas=True)
    viewer.destroy()
    viewer.destroy()
    root.destroy()


def test_public_query_point_large_scene_uses_cached_gpu_path(monkeypatch):
    from any3dview.benchmarks import plate_grid
    from any3dview.gpu import Any3DView

    root = tk.Tk()
    root.geometry("640x480+0+0")
    viewer = Any3DView(root, width=640, height=480)
    viewer.pack(fill=tk.BOTH, expand=True)
    scene = plate_grid(180, 180)
    assert scene.mesh.triangle_count > 50_000
    viewer.add_mesh_arrays(scene.mesh, cull_backface=False)
    viewer.fit_to_scene()
    root.update()
    x, y = (value // 2 for value in viewer.viewport_size)

    # Build the integer target once, then prove the public API stays on it.
    assert viewer.query_point(x, y)
    viewer._selection_index = None
    monkeypatch.setattr(
        viewer,
        "_projected_selection_index",
        lambda: (_ for _ in ()).throw(
            AssertionError("large cached visible pick built the CPU index")
        ),
    )
    samples = []
    for _index in range(20):
        started = time.perf_counter()
        assert viewer.query_point(x, y)
        samples.append((time.perf_counter() - started) * 1000.0)
    median_ms = statistics.median(samples)
    print(f"public query_point cached median: {median_ms:.3f} ms")
    assert median_ms < 15.0
    assert viewer._selection_index is None

    viewer.destroy()
    root.destroy()


def test_gpu_owned_chunk_pick_highlight_and_animation_identity():
    from any3dview import MeshArrays, PackedOwnerTable, PickBinding
    from any3dview.gpu import Any3DView

    root = tk.Tk()
    root.geometry("360x280+0+0")
    viewer = Any3DView(root, width=360, height=280)
    viewer.pack(fill=tk.BOTH, expand=True)
    empty = MeshArrays(
        np.empty((0, 3), np.float32), np.empty((0, 3), np.uint32)
    )
    local = MeshArrays(
        np.asarray([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32),
        np.asarray([[0, 1, 2]], np.uint32),
    )
    first = PackedOwnerTable.from_owners(
        triangles=(PickBinding.one("chunk:frame:1", "face"),)
    )
    second = PackedOwnerTable.from_owners(
        triangles=(PickBinding.one("chunk:frame:2", "face"),)
    )
    handle = viewer.add_mesh_arrays(empty, cull_backface=False)
    handle.add_chunk("local", local, owners=first)
    viewer.fit_to_scene()
    root.update()
    projected = viewer.project_point((0, 0, 0))
    assert projected is not None
    point = tuple(round(value) for value in projected[:2])
    assert viewer.query_point(*point)[0].key == "chunk:frame:1"

    viewer.set_highlight(("chunk:frame:1",))
    root.update()
    resource = dict(viewer._renderer._groups[id(handle)].chunks)["local"]
    assert np.frombuffer(resource.semantic_elements.read(), np.uint8)[0] == 1

    viewer.begin_animation_cache()
    viewer.capture_animation_frame()
    handle.set_chunk_ownership("local", second)
    viewer.set_highlight(("chunk:frame:2",))
    viewer.capture_animation_frame()

    viewer._show_animation_frame(0)
    root.update()
    assert viewer.query_point(*point)[0].key == "chunk:frame:1"
    frame_zero = next(iter(viewer._animation_entries.values()))["handle"]
    assert frame_zero.chunk_ownership("local")[0] is first

    viewer._show_animation_frame(1)
    root.update()
    assert viewer.query_point(*point)[0].key == "chunk:frame:2"
    frame_one = next(iter(viewer._animation_entries.values()))["handle"]
    assert frame_one.chunk_ownership("local")[0] is second

    viewer.destroy()
    root.destroy()
