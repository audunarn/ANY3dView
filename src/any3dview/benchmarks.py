"""Deterministic renderer-independent benchmark scenes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arrays import MeshArrays


@dataclass(frozen=True, slots=True)
class BenchmarkScene:
    name: str
    mesh: MeshArrays


def plate_grid(columns: int = 100, rows: int = 100) -> BenchmarkScene:
    """Return a triangulated unit-cell plate grid with stable element IDs."""

    columns, rows = int(columns), int(rows)
    if columns < 1 or rows < 1:
        raise ValueError("grid dimensions must be positive")
    xs, ys = np.meshgrid(
        np.arange(columns + 1, dtype=np.float32),
        np.arange(rows + 1, dtype=np.float32),
    )
    positions = np.column_stack((xs.ravel(), ys.ravel(), np.zeros(xs.size, np.float32)))
    triangles = np.empty((columns * rows * 2, 3), dtype=np.uint32)
    mapping = np.repeat(np.arange(columns * rows, dtype=np.uint32), 2)
    stride = columns + 1
    low = (
        np.arange(rows, dtype=np.uint32)[:, None] * stride
        + np.arange(columns, dtype=np.uint32)[None, :]
    ).reshape(-1)
    triangles[0::2, 0] = low
    triangles[0::2, 1] = low + 1
    triangles[0::2, 2] = low + stride + 1
    triangles[1::2, 0] = low
    triangles[1::2, 1] = low + stride + 1
    triangles[1::2, 2] = low + stride
    return BenchmarkScene(
        f"plate_grid_{columns}x{rows}",
        MeshArrays(
            positions,
            triangles,
            triangle_to_element=mapping,
            element_ids=np.arange(1, columns * rows + 1, dtype=np.uint64),
        ),
    )


def member_lattice(columns: int = 100, rows: int = 100) -> BenchmarkScene:
    """Return a deterministic indexed line lattice."""

    grid = plate_grid(columns, rows).mesh
    horizontal = []
    vertical = []
    stride = columns + 1
    for row in range(rows + 1):
        for column in range(columns):
            start = row * stride + column
            horizontal.append((start, start + 1))
    for row in range(rows):
        for column in range(columns + 1):
            start = row * stride + column
            vertical.append((start, start + stride))
    mesh = MeshArrays(
        grid.positions,
        np.empty((0, 3), dtype=np.uint32),
        lines=np.asarray(horizontal + vertical, dtype=np.uint32),
    )
    return BenchmarkScene(f"member_lattice_{columns}x{rows}", mesh)
