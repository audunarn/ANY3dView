"""Frozen-application smoke entry point used by release qualification."""

from __future__ import annotations

import os
import tkinter as tk

import numpy as np

from any3dview import MeshArrays, create_viewer


def main() -> None:
    root = tk.Tk()
    root.geometry("240x180+0+0")
    backend = os.environ.get("ANY3DVIEW_BACKEND", "auto")
    viewer = create_viewer(root, backend=backend, width=240, height=180)
    viewer.pack(fill=tk.BOTH, expand=True)
    viewer.add_mesh_arrays(
        MeshArrays(
            np.asarray([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32),
            np.asarray([[0, 1, 2]], np.uint32),
        )
    )
    root.update()
    print("gpu" if viewer.capabilities.gpu else "software", flush=True)
    viewer.destroy()
    root.update()
    root.destroy()


if __name__ == "__main__":
    main()
