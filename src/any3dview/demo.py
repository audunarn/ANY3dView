"""Interactive retained-scene showcase for the available viewer backends."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence

import numpy as np

from .arrays import MeshArrays
from .errors import GPUUnavailableError


def build_demo_mesh(grid_size: int = 40) -> MeshArrays:
    """Build a deterministic plate with edges, displacement and element results."""

    size = int(grid_size)
    if size < 2:
        raise ValueError("grid_size must be at least 2")

    coordinates = np.linspace(-6.0, 6.0, size + 1, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    positions = np.column_stack(
        (
            x_grid.ravel(),
            y_grid.ravel(),
            np.zeros(x_grid.size, dtype=np.float32),
        )
    ).astype(np.float32, copy=False)

    nodes = np.arange((size + 1) ** 2, dtype=np.uint32).reshape(size + 1, size + 1)
    lower_left = nodes[:-1, :-1].ravel()
    lower_right = nodes[:-1, 1:].ravel()
    upper_left = nodes[1:, :-1].ravel()
    upper_right = nodes[1:, 1:].ravel()
    triangles = np.empty((2 * size * size, 3), dtype=np.uint32)
    triangles[0::2] = np.column_stack((lower_left, lower_right, upper_right))
    triangles[1::2] = np.column_stack((lower_left, upper_right, upper_left))
    triangle_to_element = np.repeat(
        np.arange(size * size, dtype=np.uint32), 2
    )

    horizontal = np.stack((nodes[:, :-1], nodes[:, 1:]), axis=-1).reshape(-1, 2)
    vertical = np.stack((nodes[:-1, :], nodes[1:, :]), axis=-1).reshape(-1, 2)
    lines = np.ascontiguousarray(np.concatenate((horizontal, vertical)), dtype=np.uint32)
    point_indices = np.asarray(
        (nodes[0, 0], nodes[0, -1], nodes[-1, -1], nodes[-1, 0]),
        dtype=np.uint32,
    )

    normalized_x = positions[:, 0] / 6.0
    normalized_y = positions[:, 1] / 6.0
    displacements = np.zeros_like(positions)
    displacements[:, 2] = (
        1.2
        * np.cos(0.5 * math.pi * normalized_x)
        * np.cos(0.5 * math.pi * normalized_y)
    )

    cell_x = 0.25 * (
        positions[lower_left, 0]
        + positions[lower_right, 0]
        + positions[upper_left, 0]
        + positions[upper_right, 0]
    )
    cell_y = 0.25 * (
        positions[lower_left, 1]
        + positions[lower_right, 1]
        + positions[upper_left, 1]
        + positions[upper_right, 1]
    )
    radius = np.hypot(cell_x, cell_y)
    element_scalars = np.asarray(
        55.0
        + 165.0 * np.exp(-0.055 * radius * radius)
        + 25.0 * np.sin(0.7 * cell_x) * np.cos(0.6 * cell_y),
        dtype=np.float32,
    )

    return MeshArrays(
        positions=positions,
        triangles=triangles,
        lines=lines,
        point_indices=point_indices,
        triangle_to_element=triangle_to_element,
        node_ids=np.arange(1, len(positions) + 1, dtype=np.uint64),
        element_ids=np.arange(1, size * size + 1, dtype=np.uint64),
        displacements=displacements,
        element_scalars=element_scalars,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("auto", "gpu", "software"),
        default="auto",
        help="viewer backend (default: auto)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=40,
        metavar="N",
        help="plate cells along each axis (default: 40)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Create a Tk window and run the retained-scene demonstration."""

    arguments = _parser().parse_args(argv)
    try:
        mesh = build_demo_mesh(arguments.grid_size)
    except ValueError as error:
        _parser().error(str(error))

    import tkinter as tk
    from tkinter import ttk

    from .factory import create_viewer

    root = tk.Tk()
    root.title("ANY3dView retained viewer")
    root.geometry("1100x760")
    root.minsize(760, 520)

    toolbar = ttk.Frame(root, padding=(8, 6))
    toolbar.pack(fill=tk.X)
    viewport = ttk.Frame(root)
    viewport.pack(fill=tk.BOTH, expand=True)

    try:
        viewer = create_viewer(
            viewport,
            backend=arguments.backend,
            width=1080,
            height=680,
            bg="#f8fafc",
        )
    except GPUUnavailableError as error:
        root.destroy()
        details = "; ".join(error.diagnostics)
        raise SystemExit(f"{error}{': ' + details if details else ''}") from error

    viewer.pack(fill=tk.BOTH, expand=True)
    handle = viewer.add_mesh_arrays(
        mesh,
        color="#94a3b8",
        line_color="#334155",
        line_width=1,
        point_color="#0f172a",
        point_size=7,
        scalar_range=(float(mesh.element_scalars.min()), float(mesh.element_scalars.max())),
        two_sided_shell=True,
    )

    capabilities = viewer.capabilities
    backend_name = "ModernGL GPU" if capabilities.gpu else "Tk Canvas software"
    ttk.Label(toolbar, text=f"Backend: {backend_name}").pack(side=tk.LEFT, padx=(0, 12))
    ttk.Button(toolbar, text="Fit", command=viewer.fit_to_scene).pack(side=tk.LEFT)

    section_enabled = tk.BooleanVar(value=False)

    def update_section() -> None:
        if section_enabled.get():
            viewer.set_section_plane(normal=(1.0, 0.0, 0.0), offset=0.0)
        else:
            viewer.clear_section_plane()

    ttk.Checkbutton(
        toolbar,
        text="Section x >= 0",
        variable=section_enabled,
        command=update_section,
    ).pack(side=tk.LEFT, padx=(12, 8))

    ttk.Label(toolbar, text="Deformation").pack(side=tk.LEFT, padx=(8, 4))
    deformation = tk.DoubleVar(value=0.9)

    def update_deformation(value: str | float) -> None:
        handle.set_deformation_scale(float(value))

    ttk.Scale(
        toolbar,
        from_=0.0,
        to=2.5,
        length=180,
        variable=deformation,
        command=update_deformation,
    ).pack(side=tk.LEFT)

    animate = tk.BooleanVar(value=False)
    ttk.Checkbutton(toolbar, text="Animate", variable=animate).pack(
        side=tk.LEFT, padx=(12, 4)
    )
    ttk.Label(
        toolbar,
        text="Right-drag: orbit   Middle-drag: pan   Wheel: zoom",
    ).pack(side=tk.RIGHT)

    diagnostics = "; ".join(getattr(viewer, "backend_diagnostics", ()))
    status_text = (
        f"{mesh.node_count:,} nodes   {mesh.element_count:,} elements   "
        f"{mesh.triangle_count:,} triangles"
    )
    if diagnostics:
        status_text += f"   Fallback: {diagnostics}"
    ttk.Label(root, text=status_text, padding=(8, 4), anchor=tk.W).pack(fill=tk.X)

    update_deformation(deformation.get())
    start_time = time.perf_counter()

    def animation_tick() -> None:
        if animate.get():
            scale = 1.1 + 0.9 * math.sin(2.0 * (time.perf_counter() - start_time))
            deformation.set(scale)
            handle.set_deformation_scale(scale)
        root.after(16, animation_tick)

    root.after_idle(viewer.fit_to_scene)
    root.after(16, animation_tick)
    root.bind("<Escape>", lambda _event: root.destroy())
    root.mainloop()


if __name__ == "__main__":
    main()
