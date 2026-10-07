"""Pixel-level checks of transparency and antialiasing in the GPU renderer."""

import numpy as np
import pytest

from any3dview import Camera3D, MeshArrays, MeshHandle

moderngl = pytest.importorskip("moderngl")

SIZE = (64, 64)
WHITE = (1.0, 1.0, 1.0, 1.0)


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


SCALE = 4.0


def triangle(z=0.0, shift=(0.0, 0.0), scale=SCALE):
    return MeshArrays(
        np.asarray(
            [[-scale + shift[0], -scale + shift[1], z],
             [scale + shift[0], -scale + shift[1], z],
             [shift[0], scale + shift[1], z]],
            dtype=np.float32,
        ),
        np.asarray([[0, 1, 2]], dtype=np.uint32),
    )


def render(renderer, *, samples=0, depth_texture=False):
    context = renderer.ctx
    color = context.texture(SIZE, 4)
    depth = context.depth_texture(SIZE) if depth_texture else context.depth_renderbuffer(SIZE)
    framebuffer = context.framebuffer(color, depth)
    renderer.render(
        Camera3D(), SIZE, target=framebuffer, clear_color=WHITE, samples=samples
    )
    pixels = np.frombuffer(
        framebuffer.read(components=3, alignment=1), dtype=np.uint8
    ).reshape(SIZE[1], SIZE[0], 3).astype(int)
    depth_values = (
        np.frombuffer(depth.read(), dtype=np.float32).reshape(SIZE[1], SIZE[0])
        if depth_texture else None
    )
    framebuffer.release()
    color.release()
    depth.release()
    return pixels, depth_values


_INTERIOR: list[tuple[int, int]] = []


def interior_points(renderer):
    """Pixels well inside the triangle, found from an opaque reference render."""

    if not _INTERIOR:
        from any3dview.gpu.renderer import ModernGLRenderer

        reference = ModernGLRenderer(renderer.ctx)
        reference.add_mesh(MeshHandle(triangle()), "#000000", lit=False)
        pixels, _ = render(reference)
        reference.release()
        mask = pixels[:, :, 0] < 128
        eroded = mask.copy()
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                eroded &= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
        points = np.argwhere(eroded)
        assert len(points) >= 20, "reference triangle too small for the test"
        _INTERIOR.extend(map(tuple, points[:: max(1, len(points) // 24)]))
    return _INTERIOR


def interior(pixels, renderer):
    return np.asarray([pixels[y, x] for y, x in interior_points(renderer)])


def test_opacity_blends_the_surface_colour_without_dithering(renderer):
    renderer.add_mesh(MeshHandle(triangle()), "#ff0000", opacity=0.4, lit=False)

    pixels, _ = render(renderer)
    values = interior(pixels, renderer)

    # 0.4 * red over white; every interior pixel identical, no Bayer dots.
    expected = np.asarray([255, round(255 * 0.6), round(255 * 0.6)])
    assert (np.abs(values - expected) <= 3).all(), values
    assert len({tuple(value) for value in values}) == 1


def test_translucent_result_does_not_depend_on_draw_order(renderer):
    from any3dview.gpu.renderer import ModernGLRenderer

    def result(order):
        local = ModernGLRenderer(renderer.ctx)
        for color, z in order:
            local.add_mesh(
                MeshHandle(triangle(z)), color, opacity=0.5, lit=False,
                cull_backface=False,
            )
        pixels, _ = render(local)
        local.release()
        return interior(pixels, renderer)

    first = result([("#ff0000", 0.0), ("#0000ff", 0.3)])
    second = result([("#0000ff", 0.3), ("#ff0000", 0.0)])

    assert np.abs(first - second).max() <= 2
    # Both layers contribute: neither pure red-over-white nor pure blue.
    assert first[:, 0].min() < 255 and first[:, 2].min() < 255
    # The depth weight varies a little across a tilted face; no dither.
    assert np.abs(first - first.mean(axis=0)).max() <= 12


def test_opaque_surface_occludes_translucent_one_behind_it(renderer):
    renderer.add_mesh(MeshHandle(triangle(0.0)), "#00ff00", lit=False)
    renderer.add_mesh(
        MeshHandle(triangle(-1.0)), "#ff0000", opacity=0.5, lit=False
    )

    pixels, _ = render(renderer)

    assert (np.abs(interior(pixels, renderer) - np.asarray([0, 255, 0])).max() <= 2)


def test_translucent_surface_in_front_tints_the_opaque_one(renderer):
    renderer.add_mesh(MeshHandle(triangle(0.0)), "#00ff00", lit=False)
    # Larger, so parallax from its different depth still covers the footprint.
    renderer.add_mesh(
        MeshHandle(triangle(1.0, scale=2 * SCALE)), "#ff0000", opacity=0.5, lit=False
    )

    values = interior(render(renderer)[0], renderer)

    # Half red over green: both channels present, and no dither pattern.
    assert (np.abs(values - np.asarray([127, 127, 0])).max() <= 4)


def test_fallback_blending_without_float_targets_is_not_dithered(renderer):
    renderer._oit_failed = True
    renderer.add_mesh(MeshHandle(triangle()), "#ff0000", opacity=0.4, lit=False)

    values = interior(render(renderer)[0], renderer)

    assert len({tuple(value) for value in values}) == 1
    assert 0 < values[0][1] < 255


def test_explicit_stipple_pattern_still_dithers(renderer):
    renderer.add_mesh(
        MeshHandle(triangle()), "#ff0000", stipple="gray50", lit=False
    )

    pixels, _ = render(renderer)
    patch = np.asarray([pixels[y, x] for y, x in interior_points(renderer)])

    colours = {tuple(value) for value in patch}
    assert colours == {(255, 0, 0), (255, 255, 255)}


def test_translucent_surface_is_pickable_over_its_whole_area(renderer):
    handle = MeshHandle(triangle())
    renderer.add_mesh(handle, "#ff0000", opacity=0.3, lit=False)
    camera = Camera3D()

    hits = [
        # pick() takes top-origin rows; readback rows are bottom-up.
        renderer.pick(x, SIZE[1] - 1 - y, camera, SIZE) is not None
        for y, x in interior_points(renderer)
    ]

    assert all(hits)


def test_multisampling_antialiases_silhouette_edges(renderer):
    renderer.add_mesh(MeshHandle(triangle()), "#000000", lit=False)

    plain, _ = render(renderer, samples=0)
    smooth, _ = render(renderer, samples=4)

    def edge_levels(pixels):
        grey = pixels[:, :, 0]
        return int(((grey > 10) & (grey < 245)).sum())

    assert edge_levels(plain) == 0
    assert edge_levels(smooth) > 20
    # The filled interior is unchanged.
    assert np.abs(interior(plain, renderer) - interior(smooth, renderer)).max() <= 1


def test_multisample_resolve_restores_scene_depth(renderer):
    renderer.add_mesh(MeshHandle(triangle()), "#000000", lit=False)

    _pixels, depth = render(renderer, samples=4, depth_texture=True)

    y, x = interior_points(renderer)[0]
    assert depth[y, x] < 0.999  # triangle covers this pixel
    assert depth[2, 2] == pytest.approx(1.0)  # background is cleared depth


def test_multisampling_request_is_remembered_when_unavailable(renderer, monkeypatch):
    attempts = []

    def refuse(*_args, **_kwargs):
        attempts.append(True)
        raise RuntimeError("no multisampled renderbuffers")

    monkeypatch.setattr(renderer.ctx, "renderbuffer", refuse)
    renderer.add_mesh(MeshHandle(triangle()), "#000000", lit=False)

    first, _ = render(renderer, samples=4)
    second, _ = render(renderer, samples=4)

    assert len(attempts) == 1  # not retried every frame
    assert (first == second).all()
    assert (np.abs(interior(first, renderer)) <= 1).all()  # still rendered, unsmoothed


@pytest.mark.parametrize(
    "value, expected",
    [("", 4), ("0", 0), ("off", 0), ("1", 4), ("8", 8), ("99", 16), ("junk", 4)],
)
def test_msaa_samples_setting(monkeypatch, value, expected):
    from any3dview.gpu.quality import msaa_samples

    monkeypatch.setenv("ANY3DVIEW_MSAA", value)

    assert msaa_samples() == expected


# ---------------------------------------------------------------- write masks
def test_depth_only_group_occludes_without_drawing_any_colour(renderer):
    renderer.add_mesh(MeshHandle(triangle(0.0)), "#ff0000", depth_only=True, lit=False)
    renderer.add_mesh(MeshHandle(triangle(-1.0, scale=2 * SCALE)), "#0000ff", lit=False)

    pixels, _ = render(renderer)

    # The occluder itself paints nothing and hides the blue triangle behind it.
    assert (interior(pixels, renderer) == 255).all()


@pytest.mark.parametrize("fallback", [False, True])
def test_translucent_surfaces_do_not_write_depth(renderer, fallback):
    renderer._oit_failed = fallback
    renderer.add_mesh(MeshHandle(triangle()), "#ff0000", opacity=0.5, lit=False)

    _pixels, depth = render(renderer, depth_texture=True)

    y, x = interior_points(renderer)[0]
    assert depth[y, x] == pytest.approx(1.0)  # still the cleared depth


def test_a_stale_depth_mask_does_not_block_the_next_frames_depth_clear(renderer):
    renderer.add_mesh(MeshHandle(triangle()), "#000000", lit=False)
    context = renderer.ctx
    color = context.texture(SIZE, 4)
    depth = context.depth_texture(SIZE)
    framebuffer = context.framebuffer(color, depth)
    framebuffer.depth_mask = False  # left behind by some earlier pass
    framebuffer.color_mask = (False, False, False, False)

    renderer.render(Camera3D(), SIZE, target=framebuffer, clear_color=WHITE)

    y, x = interior_points(renderer)[0]
    values = np.frombuffer(depth.read(), dtype=np.float32).reshape(SIZE[1], SIZE[0])
    assert values[y, x] < 0.999 and values[2, 2] == pytest.approx(1.0)
    pixels = np.frombuffer(framebuffer.read(components=3, alignment=1), np.uint8)
    assert pixels.reshape(SIZE[1], SIZE[0], 3)[y, x].max() < 10  # black triangle drawn
    framebuffer.release()
    color.release()
    depth.release()


def closed_box(half=3.0):
    corners = np.asarray(
        [[x, y, z] for z in (-half, half) for y in (-half, half) for x in (-half, half)],
        dtype=np.float32,
    )
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    triangles = []
    for a, b, c, d in quads:
        triangles += [(a, b, c), (a, c, d)]
    return MeshArrays(corners, np.asarray(triangles, dtype=np.uint32))


@pytest.mark.parametrize("opacity", [0.35, 0.5])
def test_both_walls_of_a_translucent_shell_contribute(renderer, opacity):
    """Looking through a closed shell crosses two layers, so alpha compounds."""

    renderer.add_mesh(
        MeshHandle(closed_box()), "#000000", opacity=opacity, lit=False,
        cull_backface=False,
    )

    pixels, _ = render(renderer)

    centre = pixels[SIZE[1] // 2, SIZE[0] // 2, 0] / 255.0
    expected = 1.0 - (1.0 - opacity) ** 2  # front wall over back wall
    assert 1.0 - centre == pytest.approx(expected, abs=0.03)
