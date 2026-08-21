"""Deterministic standalone ModernGL benchmark with machine-readable output."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import moderngl
import numpy as np

from any3dview import Camera3D, MeshArrays, MeshHandle, Point3D
from any3dview.benchmarks import member_lattice, plate_grid
from any3dview.gpu.renderer import ModernGLRenderer


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _orbit_sample(renderer, camera, framebuffer, viewport, duration):
    cpu_ms: list[float] = []
    gpu_ms: list[float] = []
    started = time.perf_counter()
    while (elapsed := time.perf_counter() - started) < duration:
        camera.set_orbit(azimuth=-0.75 + elapsed * 0.35)
        before = time.perf_counter()
        with renderer.ctx.query(time=True) as query:
            renderer.render(camera, viewport, target=framebuffer)
        cpu_ms.append((time.perf_counter() - before) * 1000.0)
        gpu_ms.append(query.elapsed / 1_000_000.0)
    return {
        "frames": len(cpu_ms),
        "fps": len(cpu_ms) / max(duration, 1.0e-9),
        "cpu_median_ms": statistics.median(cpu_ms),
        "cpu_p95_ms": _percentile(cpu_ms, 95),
        "gpu_median_ms": statistics.median(gpu_ms),
        "gpu_p95_ms": _percentile(gpu_ms, 95),
        "draw_calls": renderer.draw_calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--columns", type=int, default=707)
    parser.add_argument("--rows", type=int, default=707)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--upload-columns", type=int, default=1000)
    parser.add_argument("--upload-rows", type=int, default=1000)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark-results/gpu.json")
    )
    options = parser.parse_args()

    scene = plate_grid(options.columns, options.rows)
    context = moderngl.create_standalone_context(require=330)
    renderer = ModernGLRenderer(context)
    framebuffer = context.simple_framebuffer((options.width, options.height), components=4)
    camera = Camera3D()
    camera.set_target(Point3D(*scene.mesh.positions.mean(axis=0)))
    camera.set_orbit(distance=max(options.columns, options.rows) * 1.35)
    handle = MeshHandle(scene.mesh)
    renderer.add_mesh(handle)

    warmup_start = time.perf_counter()
    while time.perf_counter() - warmup_start < options.warmup:
        renderer.render(camera, (options.width, options.height), target=framebuffer)

    viewport = (options.width, options.height)
    opaque_sample = _orbit_sample(
        renderer, camera, framebuffer, viewport, options.duration
    )

    lattice = member_lattice(options.columns, options.rows).mesh
    edged_mesh = MeshArrays(
        scene.mesh.positions,
        scene.mesh.triangles,
        lines=lattice.lines,
        triangle_to_element=scene.mesh.triangle_to_element,
        element_ids=scene.mesh.element_ids,
    )
    renderer.remove_mesh(handle)
    edged_handle = MeshHandle(edged_mesh)
    renderer.add_mesh(edged_handle)
    edge_sample = _orbit_sample(
        renderer, camera, framebuffer, viewport, options.duration
    )

    renderer.remove_mesh(edged_handle)
    upload_scene = plate_grid(options.upload_columns, options.upload_rows)
    upload_handle = MeshHandle(upload_scene.mesh)
    renderer.add_mesh(upload_handle)
    camera.set_target(Point3D(*upload_scene.mesh.positions.mean(axis=0)))
    camera.set_orbit(distance=max(options.upload_columns, options.upload_rows) * 1.35)
    rng = np.random.default_rng(1729)
    scalar = rng.standard_normal(upload_scene.mesh.element_count).astype(np.float32)
    before = time.perf_counter()
    upload_handle.update_element_scalars(scalar)
    renderer.render(camera, (options.width, options.height), target=framebuffer)
    scalar_upload_ms = (time.perf_counter() - before) * 1000.0

    displacement = np.zeros_like(upload_scene.mesh.positions, dtype=np.float32)
    displacement[:, 2] = np.sin(upload_scene.mesh.positions[:, 0] * 0.01)
    before = time.perf_counter()
    upload_handle.update_displacements(displacement)
    renderer.render(camera, (options.width, options.height), target=framebuffer)
    displacement_upload_ms = (time.perf_counter() - before) * 1000.0

    before = time.perf_counter()
    renderer.pick(options.width // 2, options.height // 2, camera, (options.width, options.height))
    initial_pick_ms = (time.perf_counter() - before) * 1000.0
    cached_pick_ms = []
    for _index in range(20):
        before = time.perf_counter()
        renderer.pick(options.width // 2, options.height // 2, camera, (options.width, options.height))
        cached_pick_ms.append((time.perf_counter() - before) * 1000.0)

    info = context.info
    result = {
        "schema": 1,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "moderngl": moderngl.__version__,
            "commit": _commit(),
            "gl_vendor": info.get("GL_VENDOR", "unknown"),
            "gl_renderer": info.get("GL_RENDERER", "unknown"),
            "gl_version": info.get("GL_VERSION", "unknown"),
        },
        "scene": {
            "name": scene.name,
            "viewport": [options.width, options.height],
            "nodes": scene.mesh.node_count,
            "triangles": scene.mesh.triangle_count,
            "elements": scene.mesh.element_count,
            "warmup_seconds": options.warmup,
            "sample_seconds": options.duration,
            "vsync": False,
        },
        "timings_ms": {
            "scalar_upload": scalar_upload_ms,
            "displacement_upload": displacement_upload_ms,
            "initial_pick": initial_pick_ms,
            "cached_pick_median": statistics.median(cached_pick_ms),
        },
        "render_samples": {
            "opaque": opaque_sample,
            "structural_edges": edge_sample,
        },
        "throughput": {
            "geometry_uploads": renderer.geometry_uploads,
        },
        "memory_bytes": {
            "positions": scene.mesh.positions.nbytes,
            "triangles": scene.mesh.triangles.nbytes,
            "fields": scalar.nbytes + displacement.nbytes,
        },
        "upload_scene": {
            "nodes": upload_scene.mesh.node_count,
            "triangles": upload_scene.mesh.triangle_count,
            "elements": upload_scene.mesh.element_count,
        },
        "process": {"pid": os.getpid()},
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    renderer.release()
    framebuffer.release()
    context.release()


if __name__ == "__main__":
    main()
