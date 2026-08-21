"""Deterministic standalone ModernGL benchmark with machine-readable output."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import tracemalloc

import moderngl
import numpy as np

from any3dview import Camera3D, MeshArrays, MeshHandle, Point3D
from any3dview.benchmarks import member_lattice, plate_grid
from any3dview.gpu.renderer import ModernGLRenderer


_REPOSITORY = Path(__file__).resolve().parents[1]
_GPU_SOURCE = _REPOSITORY / "src" / "any3dview" / "gpu"



def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=_REPOSITORY,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(status.strip())


def _gpu_source_sha256() -> str:
    digest = hashlib.sha256()
    files = sorted(_GPU_SOURCE.glob("*.py"))
    if not files:
        return "unknown"
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _orbit_sample(renderer, camera, framebuffer, viewport, duration):
    cpu_ms: list[float] = []
    gpu_ms: list[float] = []
    uploads_before = int(renderer.geometry_uploads)
    started = time.perf_counter()
    while (elapsed := time.perf_counter() - started) < duration:
        camera.set_orbit(azimuth=-0.75 + elapsed * 0.35)
        before = time.perf_counter()
        with renderer.ctx.query(time=True) as query:
            renderer.render(camera, viewport, target=framebuffer)
        cpu_ms.append((time.perf_counter() - before) * 1000.0)
        gpu_ms.append(query.elapsed / 1_000_000.0)
    measured = time.perf_counter() - started
    return {
        "frames": len(cpu_ms),
        "measured_seconds": measured,
        "fps": len(cpu_ms) / max(measured, 1.0e-9),
        "cpu_median_ms": statistics.median(cpu_ms),
        "cpu_p95_ms": _percentile(cpu_ms, 95),
        "gpu_median_ms": statistics.median(gpu_ms),
        "gpu_p95_ms": _percentile(gpu_ms, 95),
        "draw_calls": renderer.draw_calls,
        "geometry_uploads": int(renderer.geometry_uploads) - uploads_before,
    }


def _warmup(renderer, camera, framebuffer, viewport, duration):
    started = time.perf_counter()
    renderer.render(camera, viewport, target=framebuffer)
    while time.perf_counter() - started < max(0.0, float(duration)):
        renderer.render(camera, viewport, target=framebuffer)
    renderer.ctx.finish()


def _allocation_sample(renderer, camera, framebuffer, viewport, frames):
    """Measure live Python allocation deltas during camera-only frames."""

    count = max(1, int(frames))
    renderer.render(camera, viewport, target=framebuffer)
    gc.collect()
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for index in range(count):
        camera.set_orbit(azimuth=-0.75 + index * 0.0035)
        renderer.render(camera, viewport, target=framebuffer)
    after = tracemalloc.take_snapshot()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    differences = after.compare_to(before, "filename")
    return {
        "frames": count,
        "python_live_blocks_delta": int(
            sum(item.count_diff for item in differences)
        ),
        "python_live_bytes_delta": int(
            sum(item.size_diff for item in differences)
        ),
        "python_peak_traced_bytes": int(peak),
    }


def _idle_sample(renderer, duration):
    """Measure a renderer that receives no redraw request for a fixed period."""

    requested = max(0.0, float(duration))
    frames_before = int(renderer.frame_count)
    cpu_before = time.process_time()
    wall_before = time.perf_counter()
    time.sleep(requested)
    wall_seconds = time.perf_counter() - wall_before
    cpu_seconds = time.process_time() - cpu_before
    return {
        "scope": "standalone renderer with no redraw calls",
        "requested_seconds": requested,
        "measured_seconds": wall_seconds,
        "process_cpu_seconds": cpu_seconds,
        "process_cpu_percent": 100.0 * cpu_seconds / max(wall_seconds, 1.0e-12),
        "rendered_frames": int(renderer.frame_count) - frames_before,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--columns", type=int, default=707)
    parser.add_argument("--rows", type=int, default=707)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--allocation-frames", type=int, default=120)
    parser.add_argument("--idle-duration", type=float, default=30.0)
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
    viewport = (options.width, options.height)
    _warmup(renderer, camera, framebuffer, viewport, options.warmup)
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
    _warmup(renderer, camera, framebuffer, viewport, options.warmup)
    edge_sample = _orbit_sample(
        renderer, camera, framebuffer, viewport, options.duration
    )

    renderer.remove_mesh(edged_handle)
    upload_scene = plate_grid(options.upload_columns, options.upload_rows)
    upload_handle = MeshHandle(upload_scene.mesh)
    renderer.add_mesh(upload_handle)
    camera.set_target(Point3D(*upload_scene.mesh.positions.mean(axis=0)))
    camera.set_orbit(distance=max(options.upload_columns, options.upload_rows) * 1.35)
    renderer.render(camera, viewport, target=framebuffer)
    context.finish()
    rng = np.random.default_rng(1729)
    scalar = rng.standard_normal(upload_scene.mesh.element_count).astype(np.float32)
    before = time.perf_counter()
    upload_handle.update_element_scalars(scalar)
    renderer.render(camera, viewport, target=framebuffer)
    context.finish()
    scalar_upload_ms = (time.perf_counter() - before) * 1000.0

    displacement = np.zeros_like(upload_scene.mesh.positions, dtype=np.float32)
    displacement[:, 2] = np.sin(upload_scene.mesh.positions[:, 0] * 0.01)
    before = time.perf_counter()
    upload_handle.update_displacements(displacement)
    renderer.render(camera, viewport, target=framebuffer)
    context.finish()
    displacement_upload_ms = (time.perf_counter() - before) * 1000.0

    before = time.perf_counter()
    renderer.pick(options.width // 2, options.height // 2, camera, (options.width, options.height))
    initial_pick_ms = (time.perf_counter() - before) * 1000.0
    cached_pick_ms = []
    for _index in range(20):
        before = time.perf_counter()
        renderer.pick(options.width // 2, options.height // 2, camera, (options.width, options.height))
        cached_pick_ms.append((time.perf_counter() - before) * 1000.0)

    allocation_sample = _allocation_sample(
        renderer,
        camera,
        framebuffer,
        (options.width, options.height),
        options.allocation_frames,
    )
    idle_sample = _idle_sample(renderer, options.idle_duration)

    info = context.info
    result = {
        "schema": 1,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "moderngl": moderngl.__version__,
            "commit": _commit(),
            "git_dirty": _git_dirty(),
            "gpu_source_sha256": _gpu_source_sha256(),
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
        "timing_scope": {
            "field_uploads": (
                "CPU handle update through render and OpenGL finish; "
                "topology and positions were synchronized first"
            ),
            "pick": "wall-clock call latency",
        },
        "metric_scope": {
            "render_samples.draw_calls": (
                "per-frame count from the final sampled frame; the renderer "
                "resets the counter at the start of every frame"
            ),
        },
        "render_samples": {
            "opaque": opaque_sample,
            "structural_edges": edge_sample,
        },
        "throughput": {
            "geometry_uploads": renderer.geometry_uploads,
            "camera_motion_geometry_uploads": int(
                opaque_sample["geometry_uploads"]
                + edge_sample["geometry_uploads"]
            ),
        },
        "allocations": allocation_sample,
        "idle_sample": idle_sample,
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
