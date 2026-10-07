"""Failure containment, reported degradation and portability guards."""

import re
from types import SimpleNamespace

import numpy as np
import pytest

from any3dview import Camera3D, MeshArrays, MeshHandle, SelectionConfig

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


def triangle(z=0.0, scale=4.0):
    return MeshArrays(
        np.asarray([[-scale, -scale, z], [scale, -scale, z], [0, scale, z]], dtype=np.float32),
        np.asarray([[0, 1, 2]], dtype=np.uint32),
    )


def points_mesh(points):
    positions = np.asarray(points, dtype=np.float32)
    return MeshArrays(
        positions, np.empty((0, 3), dtype=np.uint32),
        point_indices=np.arange(len(positions), dtype=np.uint32),
    )


def render(renderer, *, samples=0):
    context = renderer.ctx
    color = context.texture(SIZE, 4)
    framebuffer = context.framebuffer(color, context.depth_renderbuffer(SIZE))
    renderer.render(
        Camera3D(), SIZE, target=framebuffer, clear_color=(1, 1, 1, 1), samples=samples
    )
    pixels = np.frombuffer(
        framebuffer.read(components=3, alignment=1), dtype=np.uint8
    ).reshape(SIZE[1], SIZE[0], 3).astype(int)
    framebuffer.release()
    color.release()
    return pixels


def centre(pixels):
    return pixels[SIZE[1] // 2, SIZE[0] // 2]


# --------------------------------------------------------- point rim and depth
def test_antialiased_point_rim_does_not_hide_geometry_drawn_after_it(renderer):
    """The fringe has fractional coverage; it must not occlude a farther surface."""

    renderer.add_mesh(
        MeshHandle(points_mesh(((0, 0, 0.5),))), "#000000", point_color="#000000",
        point_size=12, layer=1,
    )
    renderer.add_mesh(MeshHandle(triangle(0.0, 30.0)), "#00ff00", lit=False, layer=5)

    pixels = render(renderer)

    dark = pixels.sum(axis=2) < 200
    rows, columns = np.where(dark)
    row, column = int(rows.mean()), int(columns.mean())
    window = pixels[row - 9: row + 10, column - 9: column + 10]
    background = int((window.sum(axis=2) > 740).sum())
    assert background == 0  # no ring of cleared background around the disc


def test_opaque_points_are_still_depth_tested_against_nearer_surfaces(renderer):
    renderer.add_mesh(
        MeshHandle(points_mesh(((0, 0, -0.5),))), "#000000", point_color="#000000",
        point_size=14, layer=9,
    )
    renderer.add_mesh(MeshHandle(triangle(0.0, 30.0)), "#00ff00", lit=False, layer=1)

    pixels = render(renderer)

    assert (np.abs(centre(pixels) - np.asarray([0, 255, 0])).max() <= 2)  # hidden


# ------------------------------------------------------- OIT failure containment
class _Dirty:
    """Leave the state an aborted OIT pass would, then fail."""

    def __init__(self, renderer, monkeypatch):
        original = type(renderer)._render_oit

        def aborted(self_, *args, **kwargs):
            self_._set_uniform_value(self_.render_program, "u_oit_depth_only", True)
            self_._set_uniform_value(self_.render_program, "u_oit", True)
            self_._set_uniform_value(self_.render_program, "u_depth_bias", 0.01)
            self_.ctx.blend_equation = moderngl.MAX
            raise RuntimeError("simulated driver failure mid-pass")

        monkeypatch.setattr(type(renderer), "_render_oit", aborted)
        self.original = original


def test_a_failure_inside_the_oit_pass_is_contained_reported_and_recovered(
    renderer, monkeypatch
):
    renderer.add_mesh(MeshHandle(triangle(-1.0, 30.0)), "#00ff00", lit=False)
    renderer.add_mesh(MeshHandle(triangle(0.0, 30.0)), "#ff0000", opacity=0.5, lit=False)
    _Dirty(renderer, monkeypatch)

    pixels = render(renderer)  # must not raise

    assert renderer._oit_failed
    assert any("order-independent transparency disabled" in note for note in renderer.diagnostics)
    # The same frame still shows the translucent surface, blended over green.
    assert 20 < centre(pixels)[0] < 235 and centre(pixels)[1] > 100
    # Cached flags and GL state were restored, so ordinary drawing is intact.
    assert not renderer.render_program["u_oit_depth_only"].value
    assert not renderer.render_program["u_oit"].value
    monkeypatch.undo()
    again = render(renderer)
    assert np.abs(again - pixels).max() <= 2


def test_stale_depth_only_flag_cannot_blank_surfaces_in_later_frames(renderer, monkeypatch):
    renderer.add_mesh(MeshHandle(triangle()), "#00ff00", lit=False)
    # Simulate a pass that died after switching the program to depth-only.
    renderer._set_uniform_value(renderer.render_program, "u_oit_depth_only", True)

    pixels = render(renderer)

    assert np.abs(centre(pixels) - np.asarray([0, 255, 0])).max() <= 2


def test_oit_resource_failure_is_reported_not_silent(renderer, monkeypatch):
    renderer.add_mesh(MeshHandle(triangle()), "#ff0000", opacity=0.5, lit=False)
    real_texture = renderer.ctx.texture

    def refuse_float(size, components, *args, **kwargs):
        if kwargs.get("dtype") == "f2":
            raise RuntimeError("float render targets unavailable")
        return real_texture(size, components, *args, **kwargs)

    monkeypatch.setattr(renderer.ctx, "texture", refuse_float)

    pixels = render(renderer)

    assert 100 < centre(pixels)[1] < 200  # blended, not dithered or missing
    notes = renderer.diagnostics
    assert any("order-independent transparency unavailable" in n and "float render" in n for n in notes)


def test_multisampling_failure_is_reported(renderer, monkeypatch):
    renderer.add_mesh(MeshHandle(triangle()), "#000000", lit=False)

    def refuse(*_args, **_kwargs):
        raise RuntimeError("no multisampled renderbuffers")

    monkeypatch.setattr(renderer.ctx, "renderbuffer", refuse)

    render(renderer, samples=4)
    render(renderer, samples=4)

    notes = [n for n in renderer.diagnostics if "multisampling unavailable" in n]
    assert len(notes) == 1 and "4x" in notes[0]  # reported once, not per frame


def test_a_failed_msaa_resolve_does_not_break_depth_ordering_afterwards(
    renderer, monkeypatch
):
    renderer.add_mesh(MeshHandle(triangle(0.5, 6.0)), "#00ff00", lit=False)   # near
    renderer.add_mesh(MeshHandle(triangle(-0.5, 6.0)), "#0000ff", lit=False)  # far, drawn later
    render(renderer, samples=4)  # allocate the multisample resources
    monkeypatch.setattr(
        renderer, "_msaa_vao", SimpleNamespace(render=lambda **_k: (_ for _ in ()).throw(RuntimeError("resolve failed")))
    )
    with pytest.raises(RuntimeError, match="resolve failed"):
        render(renderer, samples=4)
    monkeypatch.undo()

    pixels = render(renderer)

    # A stale ALWAYS depth function would let the later, farther blue triangle win.
    assert np.abs(centre(pixels) - np.asarray([0, 255, 0])).max() <= 2


def test_viewer_backend_diagnostics_include_renderer_degradations():
    from any3dview.gpu import Any3DView

    viewer = Any3DView.__new__(Any3DView)
    viewer._backend_diagnostics = ("GPUUnavailableError",)
    viewer._renderer = SimpleNamespace(diagnostics=("multisampling unavailable at 4x", "GPUUnavailableError"))

    assert viewer.backend_diagnostics == (
        "GPUUnavailableError", "multisampling unavailable at 4x",
    )


def test_backend_diagnostics_tolerate_renderers_without_the_attribute():
    from any3dview.gpu import Any3DView

    viewer = Any3DView.__new__(Any3DView)
    viewer._backend_diagnostics = ("note",)
    viewer._renderer = object()

    assert viewer.backend_diagnostics == ("note",)


# ------------------------------------------------------------ query_point units
def test_explicit_query_radius_stays_in_device_pixels_but_the_default_scales(monkeypatch):
    from any3dview.gpu import Any3DView

    monkeypatch.setenv("ANY3DVIEW_PIXEL_RATIO", "2")
    viewer = Any3DView.__new__(Any3DView)
    viewer._selection_config = SelectionConfig(click_radius_px=4)
    radii = []
    monkeypatch.setattr(
        viewer, "_gpu_point_hits",
        lambda x, y, selection_filter, radius=0, **kw: radii.append(radius) or (),
    )
    monkeypatch.setattr(viewer, "_display_primitive_count", lambda _limit: 1_000_000)

    viewer.query_point(5, 6)             # policy default: logical -> 8 device px
    viewer.query_point(5, 6, radius=3)   # explicit: device px, unchanged

    assert radii == [8, 3]


# ------------------------------------------------------------- shader portability
BUILTINS = (
    "step smoothstep mix min max clamp length normalize dot cross floor ceil abs sign pow exp log "
    "sin cos tan mod fract texture reflect distance sqrt round trunc"
).split()


def test_no_glsl_declaration_shadows_a_builtin_function():
    from any3dview.gpu import hud, renderer

    sources = {
        f"{module.__name__}.{name}": value
        for module in (renderer, hud)
        for name, value in vars(module).items()
        if isinstance(value, str) and value.lstrip().startswith("#version")
    }
    assert len(sources) >= 12
    pattern = re.compile(
        r"\b(?:bool|int|uint|float|vec[234]|ivec[234]|mat[234])\s+(%s)\b" % "|".join(BUILTINS)
    )
    offenders = {
        name: sorted(set(pattern.findall(text)))
        for name, text in sources.items() if pattern.search(text)
    }
    assert offenders == {}
