"""Round polyline joints, and order-independent translucent lines and points."""

import numpy as np
import pytest

from any3dview import Camera3D, MeshArrays, MeshHandle

moderngl = pytest.importorskip("moderngl")

SIZE = (64, 64)


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


def polyline(points):
    positions = np.asarray(points, dtype=np.float32)
    count = len(positions)
    return MeshArrays(
        positions,
        np.empty((0, 3), dtype=np.uint32),
        lines=np.asarray([[i, i + 1] for i in range(count - 1)], dtype=np.uint32),
    )


def points_mesh(points):
    positions = np.asarray(points, dtype=np.float32)
    return MeshArrays(
        positions,
        np.empty((0, 3), dtype=np.uint32),
        point_indices=np.arange(len(positions), dtype=np.uint32),
    )


def triangle(z, scale=4.0):
    return MeshArrays(
        np.asarray([[-scale, -scale, z], [scale, -scale, z], [0, scale, z]], dtype=np.float32),
        np.asarray([[0, 1, 2]], dtype=np.uint32),
    )


def render(renderer, *, samples=0):
    context = renderer.ctx
    color = context.texture(SIZE, 4)
    framebuffer = context.framebuffer(color, context.depth_renderbuffer(SIZE))
    renderer.render(
        Camera3D(), SIZE, target=framebuffer, clear_color=(1, 1, 1, 1),
        samples=samples,
    )
    pixels = np.frombuffer(
        framebuffer.read(components=3, alignment=1), dtype=np.uint8
    ).reshape(SIZE[1], SIZE[0], 3).astype(int)
    framebuffer.release()
    color.release()
    return pixels


def dark(pixels):
    return pixels[:, :, 0] < 128


V_SHAPE = ((-3, -2, 0), (0, 1, 0), (3, -2, 0))


def covered_extent(renderer, monkeypatch, *, round_joins, points, width=12):
    from any3dview.gpu import renderer as module

    monkeypatch.setattr(module, "_ROUND_JOINS", round_joins)
    renderer.add_mesh(
        MeshHandle(polyline(points)), "#000000", line_color="#000000", line_width=width
    )
    mask = dark(render(renderer))
    rows = np.where(mask.any(axis=1))[0]
    columns = np.where(mask.any(axis=0))[0]
    return rows.min(), rows.max(), columns.min(), columns.max(), int(mask.sum())


def test_sharp_polyline_joint_is_filled_by_a_round_join(renderer, monkeypatch):
    from any3dview.gpu.renderer import ModernGLRenderer

    butt = covered_extent(renderer, monkeypatch, round_joins=False, points=V_SHAPE)
    other = ModernGLRenderer(renderer.ctx)
    try:
        rounded = covered_extent(other, monkeypatch, round_joins=True, points=V_SHAPE)
    finally:
        other.release()

    # The outer corner of the apex reaches further up with a round join.
    assert rounded[0] <= butt[0] - 3
    assert rounded[4] > butt[4]


def test_wide_line_ends_are_round_caps_extending_half_a_width(renderer, monkeypatch):
    from any3dview.gpu.renderer import ModernGLRenderer

    segment = ((-2, 0, 0), (2, 0, 0))
    butt = covered_extent(renderer, monkeypatch, round_joins=False, points=segment)
    other = ModernGLRenderer(renderer.ctx)
    try:
        rounded = covered_extent(other, monkeypatch, round_joins=True, points=segment)
    finally:
        other.release()

    assert rounded[2] <= butt[2] - 3  # left end
    assert rounded[3] >= butt[3] + 3  # right end


def test_thin_lines_keep_the_cheap_butt_quad(renderer):
    renderer.add_mesh(
        MeshHandle(polyline(V_SHAPE)), "#000000", line_color="#000000", line_width=1
    )
    wide = MeshHandle(polyline(V_SHAPE))
    renderer.add_mesh(wide, "#000000", line_color="#000000", line_width=6)
    groups = list(renderer._groups.values())

    renderer.pixel_ratio = 1.0
    assert renderer._line_cap_segments(groups[0]) == 0
    assert renderer._line_cap_segments(groups[1]) >= 3
    renderer.pixel_ratio = 2.0  # a 1 px line is 2 device pixels wide
    assert renderer._line_cap_segments(groups[0]) == 0
    renderer.add_mesh(
        MeshHandle(polyline(V_SHAPE)), "#000000", line_color="#000000", line_width=1.5
    )
    assert renderer._line_cap_segments(list(renderer._groups.values())[-1]) >= 3


def test_round_join_edge_is_antialiased_by_multisampling(renderer):
    renderer.add_mesh(
        MeshHandle(polyline(V_SHAPE)), "#000000", line_color="#000000", line_width=12
    )

    smooth = render(renderer, samples=4)[:, :, 0]

    partial = int(((smooth > 10) & (smooth < 245)).sum())
    assert partial > 40  # a real polygon rim, not a discarded hard edge


# ------------------------------------------------------- translucent lines/points
CENTRE_ROW = 33


def line_row(pixels):
    mask = dark_or_tinted(pixels)
    rows = np.where(mask.any(axis=1))[0]
    return int(rows.mean())


def dark_or_tinted(pixels):
    return pixels.sum(axis=2) < 3 * 250


def test_translucent_line_blends_once_over_the_background(renderer):
    renderer.add_mesh(
        MeshHandle(polyline(((-2, 0, 0), (2, 0, 0)))), "#000000",
        line_color="#000000", line_width=8, opacity=0.5,
    )

    pixels = render(renderer)
    row = line_row(pixels)

    centre = pixels[row, 32]
    assert np.abs(centre - 128).max() <= 4  # 50% black over white


def test_translucent_lines_do_not_depend_on_draw_order(renderer):
    from any3dview.gpu.renderer import ModernGLRenderer

    def result(order):
        local = ModernGLRenderer(renderer.ctx)
        for color, a, b in order:
            local.add_mesh(
                MeshHandle(polyline((a, b))), "#000000", line_color=color,
                line_width=8, opacity=0.5,
            )
        pixels = render(local)
        local.release()
        return pixels

    red = ("#ff0000", (-2, -1.5, 0.0), (2, 1.5, 0.3))
    blue = ("#0000ff", (-2, 1.5, 0.0), (2, -1.5, 0.3))

    first, second = result([red, blue]), result([blue, red])

    assert np.abs(first - second).max() <= 2
    # Where they cross both contribute.
    crossing = first[dark_or_tinted(first)]
    assert (crossing[:, 0] < 255).any() and (crossing[:, 2] < 255).any()


def test_translucent_points_blend_without_sorting(renderer):
    renderer.add_mesh(
        MeshHandle(points_mesh(((0, 0, 0),))), "#000000",
        point_color="#ff0000", point_size=16, opacity=0.5,
    )

    pixels = render(renderer)

    row = line_row(pixels)
    columns = np.where(dark_or_tinted(pixels)[row])[0]
    centre = pixels[row, int(columns.mean())]
    assert np.abs(centre - np.asarray([255, 128, 128])).max() <= 4


def test_opaque_line_still_occludes_a_translucent_surface_behind_it(renderer):
    renderer.add_mesh(MeshHandle(triangle(0.0, 6.0)), "#ff0000", opacity=0.5, lit=False)
    renderer.add_mesh(
        MeshHandle(polyline(((-2, 0, 1.0), (2, 0, 1.0)))), "#000000",
        line_color="#00ff00", line_width=6,
    )

    pixels = render(renderer)

    green = (pixels[:, :, 1] > 200) & (pixels[:, :, 0] < 40) & (pixels[:, :, 2] < 40)
    assert green.sum() > 30  # the line is pure green where it is in front


def test_translucent_surface_in_front_of_an_opaque_line_tints_it(renderer):
    renderer.add_mesh(MeshHandle(triangle(1.0, 6.0)), "#ff0000", opacity=0.5, lit=False)
    renderer.add_mesh(
        MeshHandle(polyline(((-2, 0, 0.0), (2, 0, 0.0)))), "#000000",
        line_color="#00ff00", line_width=6,
    )

    pixels = render(renderer)

    pure_green = (pixels[:, :, 1] > 200) & (pixels[:, :, 0] < 40) & (pixels[:, :, 2] < 40)
    tinted = (pixels[:, :, 0] > 100) & (pixels[:, :, 1] > 100) & (pixels[:, :, 2] < 40)
    assert pure_green.sum() == 0
    assert tinted.sum() > 30


def test_fallback_blending_still_handles_translucent_lines_and_points(renderer):
    renderer._oit_failed = True
    renderer.add_mesh(
        MeshHandle(polyline(((-2, 0, 0), (2, 0, 0)))), "#000000",
        line_color="#000000", line_width=8, opacity=0.5,
    )

    pixels = render(renderer)
    row = line_row(pixels)

    assert np.abs(pixels[row, 32] - 128).max() <= 4


def test_overlay_translucent_lines_draw_on_top_of_opaque_surfaces(renderer):
    renderer.add_mesh(MeshHandle(triangle(1.0, 6.0)), "#00ff00", lit=False)
    renderer.add_mesh(
        MeshHandle(polyline(((-2, 0, 0.0), (2, 0, 0.0)))), "#000000",
        line_color="#000000", line_width=8, opacity=0.5, line_overlay=True,
    )

    pixels = render(renderer)

    # Green surface is in front of the line, yet the overlay line shows through.
    assert (np.abs(pixels - np.asarray([0, 128, 0])).max(axis=2) <= 6).sum() > 30


# ----------------------------------------------- union within a translucent group
CHAIN = ((-3, 0, 0), (-1.5, 0, 0), (0, 0, 0), (1.5, 0, 0), (3, 0, 0))


def tinted(pixels):
    return pixels.sum(axis=2) < 3 * 250


def test_translucent_polyline_has_uniform_alpha_at_its_joints(renderer):
    renderer.add_mesh(
        MeshHandle(polyline(CHAIN)), "#000000", line_color="#000000",
        line_width=12, opacity=0.5,
    )

    pixels = render(renderer)

    values = pixels[:, :, 0][tinted(pixels)]
    # 50% black over white everywhere: no darker bead where capsules overlap.
    assert values.min() >= 128 - 4
    assert np.abs(values - 128).max() <= 4 or (values > 128).any()  # rim may be lighter


def test_translucent_sharp_corner_is_not_darkened_by_the_overlap(renderer):
    renderer.add_mesh(
        MeshHandle(polyline(V_SHAPE)), "#000000", line_color="#000000",
        line_width=12, opacity=0.5,
    )

    pixels = render(renderer)

    assert pixels[:, :, 0][tinted(pixels)].min() >= 128 - 4


def test_separate_translucent_groups_still_accumulate_where_they_cross(renderer):
    for points in (((-2, -1.5, 0), (2, 1.5, 0)), ((-2, 1.5, 0), (2, -1.5, 0))):
        renderer.add_mesh(
            MeshHandle(polyline(points)), "#000000", line_color="#000000",
            line_width=8, opacity=0.5,
        )

    pixels = render(renderer)

    # Two layers of 50% black: 25% of white remains where they cross.
    assert pixels[:, :, 0][tinted(pixels)].min() <= 64 + 6


def test_opaque_surface_in_front_hides_a_translucent_line_behind_it(renderer):
    renderer.add_mesh(MeshHandle(triangle(1.0, 6.0)), "#00ff00", lit=False)
    renderer.add_mesh(
        MeshHandle(polyline(((-2, 0, 0.0), (2, 0, 0.0)))), "#000000",
        line_color="#000000", line_width=8, opacity=0.5,
    )

    pixels = render(renderer)

    # Where the surface covers the line it stays pure green (depth is shared
    # with the scratch layer, and clearing the layer does not erase it).
    inside = (np.abs(pixels - np.asarray([0, 255, 0])).max(axis=2) <= 2)
    assert inside.sum() > 200
    assert not ((np.abs(pixels - np.asarray([0, 128, 0])).max(axis=2) <= 6)).any()


# ------------------------------------------------------------ antialiased points
def point_pixels(renderer, size, *, outline="", samples=0):
    renderer.add_mesh(
        MeshHandle(points_mesh(((0, 0, 0),))), "#000000", point_color="#000000",
        point_outline=outline, point_size=size,
    )
    return render(renderer, samples=samples)[:, :, 0]


def test_point_rim_is_antialiased_without_multisampling(renderer):
    values = point_pixels(renderer, 14)

    intermediate = int(((values > 10) & (values < 245)).sum())
    assert intermediate >= 12  # a hard discard would give none


def test_point_keeps_its_nominal_size(renderer):
    values = point_pixels(renderer, 14)

    coverage = float((255 - values).sum() / 255.0)
    assert coverage == pytest.approx(np.pi * 7.0**2, rel=0.08)


def test_point_pick_sprite_is_not_enlarged_by_the_render_margin(renderer):
    renderer.add_mesh(MeshHandle(points_mesh(((0, 0, 0),))), "#000000", point_size=14)
    render(renderer)  # compiles and uses the render program

    assert renderer.point_pick_program["u_point_margin"].value == 0.0
    assert renderer.point_program["u_point_margin"].value == 2.0


def test_outlined_point_has_a_smooth_outline_ring(renderer):
    values = point_pixels(renderer, 18, outline="#ffffff")

    # Black disc with a white 1 px rim: the ring blends rather than switching.
    ring = values[(values > 10) & (values < 245)]
    assert ring.size >= 20


def test_translucent_point_rim_blends_through_oit(renderer):
    renderer.add_mesh(
        MeshHandle(points_mesh(((0, 0, 0),))), "#000000", point_color="#000000",
        point_size=16, opacity=0.5,
    )

    values = render(renderer)[:, :, 0]

    rim = values[(values > 140) & (values < 245)]
    assert rim.size >= 8  # partial coverage x 50% alpha, not a hard edge


def test_only_chunks_with_shared_vertices_need_the_union_layer(renderer):
    chain = MeshHandle(polyline(CHAIN))
    disjoint = MeshHandle(
        MeshArrays(
            np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32),
            np.empty((0, 3), dtype=np.uint32),
            lines=np.asarray([[0, 1], [2, 3]], dtype=np.uint32),
        )
    )
    single = MeshHandle(polyline(((0, 0, 0), (1, 0, 0))))
    for handle in (chain, disjoint, single):
        renderer.add_mesh(handle, "#000000", line_color="#000000", opacity=0.5)

    flags = [
        renderer._groups[id(handle)].chunks[0][1].lines_joined
        for handle in (chain, disjoint, single)
    ]

    assert flags == [True, False, False]
