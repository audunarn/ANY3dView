"""Logical-pixel sizing: lines, points, the HUD and selection tolerances scale."""

import numpy as np
import pytest

from any3dview import Camera3D, MeshArrays, MeshHandle, SelectionConfig

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


def line_mesh():
    return MeshArrays(
        np.asarray([[-2, 0, 0], [2, 0, 0]], dtype=np.float32),
        np.empty((0, 3), dtype=np.uint32),
        lines=np.asarray([[0, 1]], dtype=np.uint32),
    )


def drawn_thickness(renderer, ratio):
    renderer.pixel_ratio = ratio
    size = (96, 96)
    context = renderer.ctx
    color = context.texture(size, 4)
    framebuffer = context.framebuffer(color, context.depth_renderbuffer(size))
    renderer.render(
        Camera3D(), size, target=framebuffer, clear_color=(1, 1, 1, 1),
        occlude_lines=False,
    )
    pixels = np.frombuffer(
        framebuffer.read(components=3, alignment=1), dtype=np.uint8
    ).reshape(size[1], size[0], 3)
    framebuffer.release()
    color.release()
    column = pixels[:, size[0] // 2, 0]
    return int((column < 128).sum())


def test_line_width_is_in_logical_pixels(renderer):
    renderer.add_mesh(MeshHandle(line_mesh()), "#000000", line_color="#000000", line_width=4)

    standard = drawn_thickness(renderer, 1.0)
    doubled = drawn_thickness(renderer, 2.0)

    assert 3 <= standard <= 5
    assert doubled == pytest.approx(2 * standard, abs=1)


def test_hud_scales_positions_widths_and_text_to_device_pixels(renderer):
    from any3dview.gpu.hud import GPUHudRenderer

    hud = GPUHudRenderer(renderer.ctx)
    try:
        hud.begin((200, 100), 2.0)
        hud.quad((1, 2, 11, 12), "#ff0000")
        xs = [vertex[0] for vertex in hud.vertices]
        ys = [vertex[1] for vertex in hud.vertices]
        assert (min(xs), max(xs), min(ys), max(ys)) == (2.0, 22.0, 4.0, 24.0)

        hud.begin((200, 100), 2.0)
        hud.line((0, 0), (10, 0), "#000000", width=3)
        ys = [vertex[1] for vertex in hud.vertices]
        assert max(ys) - min(ys) == pytest.approx(6.0)  # 3 logical -> 6 device px

        def text_width(scale):
            hud.begin((200, 100), scale)
            hud.text((0, 0), "Label", "#000000", font=("Segoe UI", 10, ""), anchor="nw")
            xs = [vertex[0] for vertex in hud.vertices]
            return max(xs) - min(xs)

        assert text_width(2.0) > 1.6 * text_width(1.0)
        # Text is positioned in logical pixels too: its origin doubles.
        hud.begin((200, 100), 2.0)
        hud.text((10, 5), "x", "#000000", anchor="nw")
        assert min(vertex[0] for vertex in hud.vertices) == pytest.approx(20.0)
    finally:
        hud.release()


class RecordingHud:
    def __init__(self):
        self.begin_calls = []
        self.lines = []

    def begin(self, *args):
        self.begin_calls.append(args)

    def render(self, **_options):
        pass

    def line(self, start, end, *_args, **_options):
        self.lines.append((tuple(start), tuple(end)))

    def text(self, *_args, **_options):
        pass


def hud_viewer(monkeypatch):
    from any3dview.gpu import Any3DView

    viewer = Any3DView.__new__(Any3DView)
    viewer._hud = RecordingHud()
    viewer.camera = Camera3D()
    viewer._world_text = []
    viewer._section_plane = None
    viewer._show_axis_indicator = True
    viewer.show_axis_ruler = False
    viewer._thickness_legend = None
    viewer._highlighted_tags = frozenset()
    viewer._preselected_key = None
    viewer._selection_dragging = False
    viewer._selection_press = None
    viewer._selection_current = None
    return viewer


def test_hud_lays_out_in_logical_pixels_on_a_hidpi_display(monkeypatch):
    pytest.importorskip("moderngl")
    monkeypatch.setenv("ANY3DVIEW_PIXEL_RATIO", "2")
    viewer = hud_viewer(monkeypatch)

    viewer._render_hud((800, 600))

    assert viewer._hud.begin_calls == [((800, 600), 2.0)]
    # The axis triad's origin keeps its 58 px offset from the *logical* corner.
    assert viewer._hud.lines[0][0] == (58.0, 600 / 2 - 58.0)


def test_hud_keeps_the_one_argument_contract_at_ratio_one(monkeypatch):
    pytest.importorskip("moderngl")
    monkeypatch.delenv("ANY3DVIEW_PIXEL_RATIO", raising=False)
    viewer = hud_viewer(monkeypatch)

    viewer._render_hud((800, 600))

    assert viewer._hud.begin_calls == [((800, 600),)]
    assert viewer._hud.lines[0][0] == (58.0, 600 - 58.0)


@pytest.mark.parametrize(
    "override, native, expected",
    [("", 1.0005, 1.0), ("", 1.5, 1.5), ("", 1.74, 1.75), ("2", 1.0, 2.0),
     ("junk", 1.5, 1.0), ("99", 1.0, 8.0), ("0.1", 1.0, 0.5)],
)
def test_pixel_ratio_snaps_native_values_and_honours_the_override(
    monkeypatch, override, native, expected
):
    pytest.importorskip("moderngl")
    from any3dview.gpu import Any3DView

    monkeypatch.setenv("ANY3DVIEW_PIXEL_RATIO", override)
    viewer = Any3DView.__new__(Any3DView)
    monkeypatch.setattr(viewer, "_native_pixel_ratio", lambda: native)

    assert viewer.pixel_ratio == expected


def test_selection_click_radius_is_in_logical_pixels(monkeypatch):
    pytest.importorskip("moderngl")
    from any3dview.gpu import Any3DView

    monkeypatch.setenv("ANY3DVIEW_PIXEL_RATIO", "2")
    viewer = Any3DView.__new__(Any3DView)
    viewer._selection_config = SelectionConfig(click_radius_px=4)
    radii = []
    monkeypatch.setattr(
        viewer,
        "_gpu_point_hits",
        lambda x, y, selection_filter, radius=0, **kwargs: radii.append(radius) or (),
    )
    monkeypatch.setattr(viewer, "_display_primitive_count", lambda _limit: 1_000_000)

    viewer.query_point(5, 6)

    assert radii == [8]
