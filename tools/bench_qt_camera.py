"""Bounded Qt camera-manipulation benchmark for the retained viewer.

Builds a deterministic display-only barge-like hull fixture (<= 20k triangles),
orbits the camera over a bounded number of frames and measures the cost of
each synchronous paint.  Writes JSON evidence plus an optional cProfile dump.

Display-only: no meshing, solver or application runs are involved.

Usage:
    python tools/bench_qt_camera.py --backend software --label baseline
    python tools/bench_qt_camera.py --backend gpu --label baseline --profile
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import platform
import pstats
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_TRIANGLES = 20_000
MAX_FRAMES = 60


def barge_hull_fixture(length_cells: int = 100, width_cells: int = 100):
    """Return a deterministic barge-like hull MeshArrays with FE-style elements."""
    import numpy as np
    from any3dview.arrays import MeshArrays

    length_cells = max(1, int(length_cells))
    width_cells = max(1, int(width_cells))
    triangles = length_cells * width_cells * 2
    if triangles > MAX_TRIANGLES:
        raise SystemExit(f"fixture exceeds {MAX_TRIANGLES} triangles: {triangles}")

    xs = np.linspace(-60.0, 60.0, length_cells + 1)
    ys = np.linspace(-15.0, 15.0, width_cells + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    # Bow taper: the hull narrows toward the forward end.
    taper = 1.0 - 0.55 * np.clip((grid_x - 30.0) / 30.0, 0.0, 1.0) ** 2
    grid_y = grid_y * taper
    # Shallow V-hull depth profile, deeper at the centerline.
    depth = 8.0 * (1.0 - (np.abs(grid_y) / 15.0) ** 1.5)
    positions = np.column_stack((
        grid_x.ravel(),
        grid_y.ravel(),
        (-depth).ravel(),
    )).astype(np.float64)

    stride = width_cells + 1
    cells = np.arange(length_cells * width_cells, dtype=np.int64)
    low = (cells // width_cells) * stride + (cells % width_cells)
    tri = np.empty((triangles, 3), dtype=np.uint32)
    tri[0::2, 0] = low
    tri[0::2, 1] = low + 1
    tri[0::2, 2] = low + stride + 1
    tri[1::2, 0] = low
    tri[1::2, 1] = low + stride + 1
    tri[1::2, 2] = low + stride
    # Two triangles per cell form one FE-style element.
    triangle_to_element = np.repeat(
        np.arange(length_cells * width_cells, dtype=np.uint32), 2
    )
    # Hydrostatic-pressure-like scalars with a couple of invalid samples.
    cell_x = grid_x[:-1, :-1].ravel()
    cell_z = 0.5 * (depth[:-1, :-1].ravel() + depth[1:, 1:].ravel())
    element_scalars = np.maximum(0.0, -cell_z) * (1.0 - cell_x / 120.0)
    element_scalars[len(element_scalars) // 3] = float("nan")
    element_scalars[len(element_scalars) // 7] = float("nan")
    return MeshArrays(
        positions=positions,
        triangles=tri,
        triangle_to_element=triangle_to_element,
        element_scalars=element_scalars,
    )


def camera_keyframes(frames: int):
    """Deterministic orbit/zoom keyframes; frame i is reproducible."""
    import math

    for index in range(frames):
        fraction = index / max(1, frames - 1)
        yield {
            "azimuth": math.radians(-45.0 + 90.0 * fraction),
            "elevation": math.radians(25.0 + 10.0 * math.sin(2.0 * math.pi * fraction)),
            "distance": 190.0 * (1.0 + 0.15 * math.sin(math.pi * fraction)),
        }


def paint_once(viewer, backend: str) -> None:
    if backend == "gpu":
        viewer.surface.grabFramebuffer()
    else:
        viewer.surface.grab()


def run_frames(viewer, backend: str, frames: int):
    times = []
    for keyframe in camera_keyframes(frames):
        viewer.camera.set_orbit(**keyframe)
        viewer._renderer.pick_dirty = True
        start = time.perf_counter()
        paint_once(viewer, backend)
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def stats_of(times):
    ordered = sorted(times)
    return {
        "frames": len(times),
        "mean_ms": round(statistics.fmean(times), 3),
        "median_ms": round(statistics.median(times), 3),
        "p95_ms": round(ordered[max(0, int(0.95 * len(ordered)) - 1)], 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
    }


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("software", "gpu"), required=True)
    parser.add_argument("--label", default="run", help="evidence file label")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--profile", action="store_true",
                        help="add a cProfile pass (timing frames are reduced)")
    parser.add_argument("--output-dir", default=str(ROOT / "benchmark-results" / "camera-performance"))
    args = parser.parse_args()

    if args.backend == "software":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    frames = max(1, min(args.frames, MAX_FRAMES))
    profile_frames = 0
    if args.profile:
        split = min(frames, 30)
        profile_frames = split
        frames = min(frames, MAX_FRAMES - split)

    from PySide6.QtWidgets import QApplication
    import numpy as np
    from any3dview import create_viewer
    from any3dview.qt import QtViewerHostAdapter

    app = QApplication.instance() or QApplication([])
    mesh = barge_hull_fixture()
    viewer = create_viewer(
        None, args.backend, host=QtViewerHostAdapter(), width=900, height=700,
    )
    viewer.resize(900, 700)
    viewer.show()
    app.processEvents()
    handle = viewer.add_mesh_arrays(mesh, color="#9aa7b4", tags="barge_hull")
    viewer.fit_to_scene(redraw=False)
    app.processEvents()

    # Warm-up paints (font/texture allocation) are included in the frame budget.
    warmup = 3
    for keyframe in list(camera_keyframes(warmup)):
        viewer.camera.set_orbit(**keyframe)
        paint_once(viewer, args.backend)
    budget = warmup + frames + profile_frames
    if budget > MAX_FRAMES:
        frames = max(1, MAX_FRAMES - warmup - profile_frames)

    times = run_frames(viewer, args.backend, frames)

    profile_path = None
    profile_stats = None
    if profile_frames:
        profiler = cProfile.Profile()
        profiler.enable()
        run_frames(viewer, args.backend, profile_frames)
        profiler.disable()
        profile_path = Path(args.output_dir) / f"qt-camera-{args.backend}-{args.label}.prof"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(profile_path))
        import io
        buffer = io.StringIO()
        pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(40)
        (Path(args.output_dir) / f"qt-camera-{args.backend}-{args.label}-profile.txt").write_text(
            buffer.getvalue(), encoding="utf-8"
        )
        profile_stats = {
            "frames": profile_frames,
            "profile_path": str(profile_path),
        }

    import PySide6
    payload = {
        "benchmark": "qt-camera-manipulation",
        "label": args.label,
        "backend": args.backend,
        "qt_platform": os.environ.get("QT_QPA_PLATFORM", "native"),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": git_head(),
        "python": sys.version,
        "pyside6": PySide6.__version__,
        "os": f"{platform.system()} {platform.release()}",
        "scene": {
            "triangles": int(mesh.triangle_count),
            "elements": int(mesh.element_count),
            "vertices": int(mesh.node_count),
            "viewport": [900, 700],
            "mesh_lines": bool(viewer.show_mesh_lines),
        },
        "timing": {**stats_of(times), "per_frame_ms": [round(t, 3) for t in times]},
        "profile": profile_stats,
    }
    output = Path(args.output_dir) / f"qt-camera-{args.backend}-{args.label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("backend", "label", "timing")}, indent=2))
    print(f"evidence: {output}")
    handle.remove()
    viewer.destroy()
    app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
