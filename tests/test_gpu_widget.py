"""Native tkinter-gl lifecycle gates (opt in with ANY3DVIEW_RUN_GUI_TESTS=1)."""

from __future__ import annotations

import os
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
