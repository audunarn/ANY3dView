"""Crease-angle smooth normals: geometry, contract, plumbing and shading."""

import math

import numpy as np
import pytest

from any3dview import Camera3D, Light, MeshArrays, MeshHandle, Point3D
from any3dview.normals import expand_with_smooth_normals, smooth_corner_normals


def facet_cylinder(segments=16, radius=1.0, height=2.0):
    """Side of a cylinder; every quad owns its four vertices, as add_faces gives."""

    positions, triangles = [], []
    for k in range(segments):
        a0, a1 = 2 * math.pi * k / segments, 2 * math.pi * (k + 1) / segments
        corners = [
            (radius * math.cos(a0), radius * math.sin(a0), 0.0),
            (radius * math.cos(a1), radius * math.sin(a1), 0.0),
            (radius * math.cos(a1), radius * math.sin(a1), height),
            (radius * math.cos(a0), radius * math.sin(a0), height),
        ]
        base = len(positions)
        positions += corners
        triangles += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return np.asarray(positions), np.asarray(triangles)


def facet_box():
    corner = [(x, y, z) for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)]
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    positions, triangles = [], []
    for quad in quads:
        base = len(positions)
        positions += [corner[index] for index in quad]
        triangles += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return np.asarray(positions), np.asarray(triangles)


def flat_normals(positions, triangles):
    corners = positions[triangles]
    raw = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def test_curved_facets_get_the_true_radial_normal():
    positions, triangles = facet_cylinder(16)

    normals = smooth_corner_normals(positions, triangles, 30.0)

    radial = positions[triangles][..., :2]
    radial = radial / np.linalg.norm(radial, axis=2, keepdims=True)
    horizontal = normals[..., :2] / np.linalg.norm(normals[..., :2], axis=2, keepdims=True)
    error = np.degrees(np.arccos(np.clip(np.einsum("tci,tci->tc", horizontal, radial), -1, 1)))
    # Flat facets are off by half the facet angle (11.25 degrees here).
    assert error.max() < 0.01
    assert np.abs(normals[..., 2]).max() < 1.0e-6
    assert np.allclose(np.linalg.norm(normals, axis=2), 1.0, atol=1.0e-6)


def test_hard_edges_beyond_the_crease_angle_stay_flat():
    positions, triangles = facet_box()

    normals = smooth_corner_normals(positions, triangles, 30.0)

    assert np.allclose(normals, flat_normals(positions, triangles)[:, None, :], atol=1.0e-6)


def test_a_wide_crease_angle_rounds_the_box_and_zero_keeps_it_flat():
    positions, triangles = facet_box()
    flat = flat_normals(positions, triangles)[:, None, :]

    rounded = smooth_corner_normals(positions, triangles, 100.0)
    sharp = smooth_corner_normals(positions, triangles, 0.0)

    assert not np.allclose(rounded, flat, atol=1.0e-3)
    assert np.allclose(sharp, flat, atol=1.0e-6)


def test_inconsistent_winding_between_neighbours_is_smoothed_not_split():
    positions, triangles = facet_cylinder(16)
    reversed_triangles = triangles.copy()
    reversed_triangles[4:8] = reversed_triangles[4:8][:, ::-1]  # flip two facets

    normals = smooth_corner_normals(positions, reversed_triangles, 30.0)

    own = flat_normals(positions, reversed_triangles)[:, None, :]
    # Every corner still lies on its own face's side, and neighbouring facets
    # of the flipped ones are blended (not left at the flat facet normal).
    assert (np.einsum("tci,tci->tc", normals, np.broadcast_to(own, normals.shape)) > 0.9).all()
    assert not np.allclose(normals[3], flat_normals(positions, reversed_triangles)[3], atol=1.0e-3)


def test_degenerate_triangles_get_zero_normals_and_do_not_poison_neighbours():
    positions, triangles = facet_cylinder(8)
    positions = np.vstack((positions, positions[0], positions[0], positions[0]))
    triangles = np.vstack((triangles, [len(positions) - 3, len(positions) - 2, len(positions) - 1]))

    normals = smooth_corner_normals(positions, triangles, 30.0)

    assert np.allclose(normals[-1], 0.0)
    assert np.isfinite(normals).all()
    assert np.allclose(np.linalg.norm(normals[:-1], axis=2), 1.0, atol=1.0e-6)


def test_welding_tolerance_joins_nearly_coincident_vertices():
    positions, triangles = facet_cylinder(16)
    jittered = positions + np.random.default_rng(3).normal(scale=1.0e-9, size=positions.shape)

    normals = smooth_corner_normals(jittered, triangles, 30.0)
    exact = smooth_corner_normals(positions, triangles, 30.0)

    assert np.allclose(normals, exact, atol=1.0e-4)


def test_expansion_gives_one_vertex_per_corner_and_remaps_lines():
    positions, triangles = facet_cylinder(8)
    positions = np.vstack((positions, [[9.0, 9.0, 9.0]]))  # used only by a line
    lines = np.asarray([[0, 1], [2, len(positions) - 1]])

    expanded, new_triangles, normals, new_lines = expand_with_smooth_normals(
        positions, triangles, 30.0, lines
    )

    assert len(expanded) == 3 * len(triangles) + 1
    assert new_triangles.tolist() == np.arange(3 * len(triangles)).reshape(-1, 3).tolist()
    assert normals.shape == expanded.shape
    # Line endpoints keep their original positions after remapping.
    assert np.allclose(expanded[new_lines[0]], positions[lines[0]])
    assert np.allclose(expanded[new_lines[1]], positions[lines[1]])


def test_mesh_arrays_validates_normals():
    positions = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    triangles = np.asarray([[0, 1, 2]])

    ok = MeshArrays(positions, triangles, normals=np.tile([0.0, 0.0, 1.0], (3, 1)))
    assert ok.normals.shape == (3, 3)
    assert ok.owned_copy().normals.shape == (3, 3)
    with pytest.raises(ValueError, match="normals"):
        MeshArrays(positions, triangles, normals=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        MeshArrays(positions, triangles, normals=np.full((3, 3), np.nan))
    assert MeshArrays(positions, triangles).normals is None


def test_add_faces_with_a_crease_angle_builds_normals_and_default_stays_flat(monkeypatch):
    from any3dview.gpu import Any3DView

    positions, triangles = facet_cylinder(12)
    polygons = [
        [tuple(positions[triangles[2 * k][0]]), tuple(positions[triangles[2 * k][1]]),
         tuple(positions[triangles[2 * k][2]]), tuple(positions[triangles[2 * k + 1][2]])]
        for k in range(12)
    ]
    recorded = []
    viewer = Any3DView.__new__(Any3DView)
    monkeypatch.setattr(
        viewer, "add_mesh_arrays", lambda arrays, **options: recorded.append(arrays)
    )

    viewer.add_faces(polygons, smooth_angle=35.0)
    viewer.add_faces(polygons)

    smooth, flat = recorded
    assert smooth.normals is not None and smooth.normals.shape == smooth.positions.shape
    assert len(smooth.positions) == 3 * smooth.triangle_count
    assert flat.normals is None


def test_capabilities_advertise_smooth_normals_only_for_the_gpu_path():
    from any3dview.gpu.widget import GPU_CAPABILITIES

    assert GPU_CAPABILITIES.smooth_normals


# ------------------------------------------------------------------ rendering
moderngl = pytest.importorskip("moderngl")


@pytest.fixture
def renderer():
    from any3dview.gpu.renderer import ModernGLRenderer

    try:
        context = moderngl.create_standalone_context(require=330)
    except Exception as error:  # pragma: no cover - depends on CI display/driver
        pytest.skip(f"no standalone OpenGL 3.3 context: {error}")
    value = ModernGLRenderer(context)
    yield value
    value.release()
    context.release()


def shaded_row(renderer, normals):
    positions = np.asarray([[-4, -4, 0], [4, -4, 0], [0, 4, 0]], dtype=np.float32)
    mesh = MeshArrays(
        positions, np.asarray([[0, 1, 2]], dtype=np.uint32), normals=normals
    )
    renderer.add_mesh(MeshHandle(mesh), "#ffffff", lit=True, cull_backface=False)
    size = (64, 64)
    context = renderer.ctx
    color = context.texture(size, 4)
    framebuffer = context.framebuffer(color, context.depth_renderbuffer(size))
    light = Light(
        direction=Point3D(1.0, 0.0, 1.0), ambient=0.1, diffuse=0.9, specular=0.0,
        follow_camera=False,
    )
    renderer.render(Camera3D(), size, target=framebuffer, clear_color=(0, 0, 0, 1), light=light)
    pixels = np.frombuffer(
        framebuffer.read(components=3, alignment=1), dtype=np.uint8
    ).reshape(size[1], size[0], 3)
    framebuffer.release()
    color.release()
    covered = np.argwhere(pixels[:, :, 0] > 0)
    row = int(np.median(covered[:, 0]))
    columns = covered[covered[:, 0] == row][:, 1]
    inner = columns[len(columns) // 5: -(len(columns) // 5) or None]
    return pixels[row, inner, 0].astype(int)


def test_supplied_normals_shade_across_a_flat_triangle(renderer):
    tilted = np.asarray([[-0.8, 0, 0.6], [0.8, 0, 0.6], [0, 0, 1]], dtype=np.float32)

    values = shaded_row(renderer, tilted)

    assert values.max() - values.min() > 20  # a visible gradient across the face


def test_without_normals_the_same_triangle_is_uniform(renderer):
    values = shaded_row(renderer, None)

    assert values.max() - values.min() <= 2


def test_zero_normals_fall_back_to_flat_shading(renderer):
    values = shaded_row(renderer, np.zeros((3, 3), dtype=np.float32))

    assert values.max() - values.min() <= 2


# ------------------------------------------------------ bounded cost at high valence
def triangle_fan(valence):
    angles = np.linspace(0.0, 2 * math.pi, valence, endpoint=False)
    rim = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(valence)))
    positions = np.vstack(([0.0, 0.0, 1.0], rim))
    triangles = np.asarray([[0, 1 + i, 1 + (i + 1) % valence] for i in range(valence)])
    return positions, triangles


def test_high_valence_hub_costs_bounded_memory_and_stays_flat():
    import tracemalloc

    from any3dview.normals import MAX_SMOOTHING_VALENCE

    valence = 4096
    assert valence > MAX_SMOOTHING_VALENCE
    positions, triangles = triangle_fan(valence)

    tracemalloc.start()
    normals = smooth_corner_normals(positions, triangles, 35.0)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # All-pairs expansion needed about 2 GB here (valence squared); batching
    # and the valence limit keep it to a few megabytes.
    assert peak < 100e6
    assert np.isfinite(normals).all()
    hub = normals[:, 0]  # corner 0 of every triangle is the shared apex
    assert np.allclose(hub, flat_normals(positions, triangles), atol=1.0e-5)


def test_batching_does_not_change_the_result(monkeypatch):
    import any3dview.normals as module

    positions, triangles = facet_cylinder(24)
    reference = smooth_corner_normals(positions, triangles, 30.0)

    monkeypatch.setattr(module, "PAIR_BATCH", 40)  # force many small batches
    batched = smooth_corner_normals(positions, triangles, 30.0)

    assert np.allclose(batched, reference, atol=1.0e-6)


def test_groups_above_the_valence_limit_keep_their_face_normals(monkeypatch):
    import any3dview.normals as module

    positions, triangles = facet_cylinder(16)
    monkeypatch.setattr(module, "MAX_SMOOTHING_VALENCE", 1)  # every shared corner is over the limit

    limited = smooth_corner_normals(positions, triangles, 30.0)
    monkeypatch.setattr(module, "MAX_SMOOTHING_VALENCE", 256)
    smoothed = smooth_corner_normals(positions, triangles, 30.0)

    assert np.allclose(limited, flat_normals(positions, triangles)[:, None, :], atol=1.0e-5)
    assert not np.allclose(smoothed, limited, atol=1.0e-3)
