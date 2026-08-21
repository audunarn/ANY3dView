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
    from tkinter import messagebox, ttk

    from .factory import create_viewer

    root = tk.Tk()
    root.title("ANY3dView retained viewer")
    root.geometry("1100x760")
    root.minsize(760, 520)

    toolbar = ttk.Frame(root, padding=(8, 6))
    toolbar.pack(fill=tk.X)
    viewport = ttk.Frame(root)
    viewport.pack(fill=tk.BOTH, expand=True)
    section_enabled = tk.BooleanVar(value=False)
    deformation = tk.DoubleVar(value=0.9)
    animate = tk.BooleanVar(value=False)
    status = tk.StringVar(value="Starting viewer...")
    state: dict[str, object] = {
        "viewer": None,
        "handle": None,
        "requested": arguments.backend,
    }

    backend_labels = {
        "Automatic": "auto",
        "GPU (ModernGL)": "gpu",
        "Tk (software)": "software",
    }
    label_for_backend = {value: label for label, value in backend_labels.items()}
    backend_choice = tk.StringVar(value=label_for_backend[arguments.backend])

    ttk.Label(toolbar, text="Renderer").pack(side=tk.LEFT, padx=(0, 4))
    backend_selector = ttk.Combobox(
        toolbar,
        textvariable=backend_choice,
        values=tuple(backend_labels),
        state="readonly",
        width=17,
    )
    backend_selector.pack(side=tk.LEFT, padx=(0, 10))

    def current_viewer():
        return state["viewer"]

    def current_handle():
        return state["handle"]

    def fit_current() -> None:
        viewer = current_viewer()
        if viewer is not None:
            viewer.fit_to_scene()

    ttk.Button(toolbar, text="Fit", command=fit_current).pack(side=tk.LEFT)

    def update_section() -> None:
        viewer = current_viewer()
        if viewer is None:
            return
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

    def update_deformation(value: str | float) -> None:
        handle = current_handle()
        if handle is not None:
            handle.set_deformation_scale(float(value))

    ttk.Scale(
        toolbar,
        from_=0.0,
        to=2.5,
        length=180,
        variable=deformation,
        command=update_deformation,
    ).pack(side=tk.LEFT)

    ttk.Checkbutton(toolbar, text="Animate", variable=animate).pack(
        side=tk.LEFT, padx=(12, 4)
    )
    ttk.Label(
        toolbar,
        text="Right-drag: orbit   Middle-drag: pan   Wheel: zoom",
    ).pack(side=tk.RIGHT)
    ttk.Label(root, textvariable=status, padding=(8, 4), anchor=tk.W).pack(fill=tk.X)

    def switch_backend(requested: str, *, initial: bool = False) -> str | None:
        """Replace the rendering widget while preserving the visible demo state."""

        try:
            new_viewer = create_viewer(
                viewport,
                backend=requested,
                width=1080,
                height=680,
                bg="#f8fafc",
            )
            new_handle = new_viewer.add_mesh_arrays(
                mesh,
                color="#94a3b8",
                line_color="#334155",
                line_width=1,
                point_color="#0f172a",
                point_size=7,
                scalar_range=(
                    float(mesh.element_scalars.min()),
                    float(mesh.element_scalars.max()),
                ),
                two_sided_shell=True,
            )
            new_handle.set_deformation_scale(deformation.get())
            if section_enabled.get():
                new_viewer.set_section_plane(normal=(1.0, 0.0, 0.0), offset=0.0)
        except GPUUnavailableError as error:
            details = "; ".join(error.diagnostics)
            rendered_error = f"{error}{': ' + details if details else ''}"
            previous = str(state["requested"])
            backend_choice.set(label_for_backend[previous])
            status.set(f"Renderer switch failed: {rendered_error}")
            if not initial:
                messagebox.showerror("Renderer unavailable", rendered_error, parent=root)
            return rendered_error

        old_viewer = current_viewer()
        if old_viewer is not None:
            old_viewer.pack_forget()
        new_viewer.pack(fill=tk.BOTH, expand=True)
        state.update(viewer=new_viewer, handle=new_handle, requested=requested)
        if old_viewer is not None:
            old_viewer.destroy()

        actual_backend = (
            "ModernGL GPU" if new_viewer.capabilities.gpu else "Tk Canvas software"
        )
        diagnostics = "; ".join(getattr(new_viewer, "backend_diagnostics", ()))
        status_text = (
            f"Renderer: {actual_backend}   {mesh.node_count:,} nodes   "
            f"{mesh.element_count:,} elements   {mesh.triangle_count:,} triangles"
        )
        if diagnostics:
            status_text += f"   Fallback: {diagnostics}"
        status.set(status_text)
        root.after_idle(new_viewer.fit_to_scene)
        return None

    def select_backend(_event: object = None) -> None:
        switch_backend(backend_labels[backend_choice.get()])

    backend_selector.bind("<<ComboboxSelected>>", select_backend)
    initial_error = switch_backend(arguments.backend, initial=True)
    if initial_error is not None:
        root.destroy()
        raise SystemExit(initial_error)

    start_time = time.perf_counter()

    def animation_tick() -> None:
        if animate.get():
            scale = 1.1 + 0.9 * math.sin(2.0 * (time.perf_counter() - start_time))
            deformation.set(scale)
            handle = current_handle()
            if handle is not None:
                handle.set_deformation_scale(scale)
        root.after(16, animation_tick)

    root.after(16, animation_tick)
    root.bind("<Escape>", lambda _event: root.destroy())
    root.mainloop()


if __name__ == "__main__":
    main()
