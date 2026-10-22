"""In-place per-face colour updates on retained mesh handles.

Normalisation and handle contracts run everywhere; the software-backend
pixel-equivalence checks follow the Qt opt-in of ``test_qt_host.py``.
"""
import os
import time

import numpy as np
import pytest

from any3dview import FaceColorUpdateError, MeshArrays, MeshHandle, ViewerCapabilities
from any3dview.core import parse_color
from any3dview.face_colors import normalize_face_colors


def grid(columns=30, rows=10, quads=False):
    """A flat (columns x rows) quad grid: 2*columns*rows triangles."""
    xs, ys = np.meshgrid(np.arange(columns + 1, dtype=float), np.arange(rows + 1, dtype=float))
    positions = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    triangles = []
    for row in range(rows):
        for column in range(columns):
            a = row * (columns + 1) + column
            b, c, d = a + 1, a + columns + 2, a + columns + 1
            triangles += [(a, b, c), (a, c, d)]
    triangles = np.asarray(triangles, dtype=np.uint32)
    mapping = np.repeat(np.arange(len(triangles) // 2, dtype=np.uint32), 2) if quads else None
    return MeshArrays(positions, triangles, triangle_to_element=mapping)


def random_hex(count, seed):
    values = np.random.default_rng(seed).integers(0, 256, size=(count, 3))
    return tuple("#%02x%02x%02x" % tuple(int(v) for v in row) for row in values)


def reference_rgb(colors):
    """The GPU renderer's historical per-face parse (renderer._rgb)."""
    return np.asarray(
        [tuple(channel / 255.0 for channel in (parse_color(c) or (154, 167, 180))) for c in colors],
        np.float32,
    )


# --- normalisation ---------------------------------------------------------

def test_hex_strings_are_bit_identical_to_the_per_face_parse():
    colors = random_hex(600, 1) + ("#ABCDEF", "#abc", "red", "not-a-colour", " #102030")
    strings, rgb = normalize_face_colors(colors)
    assert strings == tuple(colors)
    assert rgb.dtype == np.float32 and rgb.shape == (len(colors), 3)
    assert rgb.tobytes() == reference_rgb(colors).tobytes()


@pytest.mark.parametrize("dtype", [np.uint8, np.int64])
def test_integer_arrays_match_their_hex_strings(dtype):
    values = np.random.default_rng(2).integers(0, 256, size=(50, 3)).astype(dtype)
    strings, rgb = normalize_face_colors(values)
    expected = tuple("#%02x%02x%02x" % tuple(int(v) for v in row) for row in values)
    assert strings == expected
    assert rgb.tobytes() == reference_rgb(expected).tobytes()


def test_float_rgba_arrays_quantise_like_hex_and_require_opaque_alpha():
    values = np.random.default_rng(3).random((40, 3))
    strings, rgb = normalize_face_colors(np.column_stack([values, np.ones(40)]))
    quantised = np.rint(values * 255.0).astype(int)
    assert strings == tuple("#%02x%02x%02x" % tuple(row) for row in quantised)
    assert rgb.tobytes() == reference_rgb(strings).tobytes()
    with pytest.raises(FaceColorUpdateError, match="alpha"):
        normalize_face_colors(np.column_stack([values, np.full(40, 0.5)]))
    # Python RGB tuples are accepted through the same numeric route.
    assert normalize_face_colors([(255, 0, 0), (0, 0, 255)])[0] == ("#ff0000", "#0000ff")


@pytest.mark.parametrize("bad", [
    np.zeros((4, 2)), np.zeros((4, 3, 1)), np.full((4, 3), 256), np.full((4, 3), -1),
    np.full((4, 3), 1.5), np.full((4, 3), np.nan), np.zeros((4, 3), dtype=bool),
])
def test_invalid_numeric_colours_are_refused(bad):
    with pytest.raises((FaceColorUpdateError, TypeError)):
        normalize_face_colors(bad)


# --- handle contracts (no backend) ------------------------------------------

def test_capability_flag_defaults_off_and_handle_method_exists():
    assert ViewerCapabilities().face_color_update is False
    assert callable(getattr(MeshHandle, "update_face_colors", None))


def test_unattached_handle_refuses_with_typed_error():
    handle = MeshHandle(grid(2, 1))
    with pytest.raises(NotImplementedError):
        handle.update_face_colors(random_hex(4, 0))


# --- software backend (Qt offscreen, opt-in like test_qt_host.py) -----------

qt_only = pytest.mark.skipif(
    os.environ.get("ANY3DVIEW_RUN_QT_TESTS") != "1", reason="set ANY3DVIEW_RUN_QT_TESTS=1"
)


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def software_viewer(qapp):
    from any3dview import create_viewer
    from any3dview.qt import QtViewerHostAdapter

    viewer = create_viewer(None, "software", host=QtViewerHostAdapter())
    viewer.resize(320, 200)
    viewer.show()
    qapp.processEvents()
    return viewer


def capture(viewer, qapp):
    viewer.set_top_view()
    viewer.fit_to_scene()
    qapp.processEvents()
    return np.asarray(viewer.capture_image()).copy()


OPTIONS = dict(cull_backface=False, tags="ocean", line_color="#000000", opacity=1.0)


@qt_only
@pytest.mark.parametrize("quads", [False, True])
@pytest.mark.parametrize("form", ["hex", "uint8", "float"])
def test_software_update_is_pixel_identical_to_fresh_mesh(qapp, quads, form):
    mesh = grid(quads=quads)
    count = mesh.element_count if quads else mesh.triangle_count
    first, second = random_hex(count, 10), random_hex(count, 11)
    values = np.asarray([parse_color(c) for c in second], dtype=np.uint8)
    update = {"hex": list(second), "uint8": values, "float": values / 255.0}[form]
    updated, fresh = software_viewer(qapp), software_viewer(qapp)
    try:
        handle = updated.add_mesh_arrays(mesh, face_colors=first, **OPTIONS)
        before = capture(updated, qapp)
        prep = updated._software_prep[(id(handle), None)]
        world, uploads = prep.world, updated._renderer.geometry_uploads
        handle.update_face_colors(update)
        after = capture(updated, qapp)
        fresh.add_mesh_arrays(mesh, face_colors=second, **OPTIONS)
        reference = capture(fresh, qapp)
        assert not np.array_equal(before, after)
        assert np.array_equal(after, reference)
        # Only the colour list was rebuilt: same retained world arrays, no upload.
        assert updated._software_prep[(id(handle), None)].world is world
        assert updated._renderer.geometry_uploads == uploads
        assert updated._entries[id(handle)]["appearance"]["face_colors"] == tuple(second)
    finally:
        updated.destroy(); fresh.destroy(); qapp.processEvents()


@qt_only
def test_software_update_keeps_transform_highlight_picking_and_removal(qapp):
    mesh = grid()
    first, second = random_hex(mesh.triangle_count, 20), random_hex(mesh.triangle_count, 21)
    transform = np.eye(4); transform[:3, 3] = (0.5, 0.25, 0.0)
    updated, fresh = software_viewer(qapp), software_viewer(qapp)
    try:
        handle = updated.add_mesh_arrays(mesh, face_colors=first, **OPTIONS)
        handle.set_transform(transform)
        updated.set_highlight(["ocean"])
        capture(updated, qapp)
        handle.update_face_colors(second)
        result = capture(updated, qapp)
        reference_handle = fresh.add_mesh_arrays(mesh, face_colors=second, **OPTIONS)
        reference_handle.set_transform(transform)
        fresh.set_highlight(["ocean"])
        assert np.array_equal(result, capture(fresh, qapp))
        np.testing.assert_array_equal(handle.transform, transform)
        assert "ocean" in updated.highlighted_tags()
        # The whole tag is highlighted, so clear it to see the updated colours.
        updated.set_highlight([]); fresh.set_highlight([])
        unhighlighted = capture(updated, qapp)
        assert not np.array_equal(unhighlighted, result)
        assert np.array_equal(unhighlighted, capture(fresh, qapp))
        x, y, _ = updated.project_point((15.5, 5.5, 0.0))
        assert updated.pick_at(int(x), int(y)) == "ocean"
        generations = handle.generations
        handle.update_face_colors(first)
        assert handle.generations == generations
        handle.remove()
        with pytest.raises(RuntimeError, match="removed"):
            handle.update_face_colors(second)
    finally:
        updated.destroy(); fresh.destroy(); qapp.processEvents()


@qt_only
def test_software_update_validation_leaves_state_unchanged(qapp):
    viewer = software_viewer(qapp)
    try:
        assert viewer.capabilities.face_color_update
        mesh = grid(quads=True)
        plain = viewer.add_mesh_arrays(mesh, color="#336699", **OPTIONS)
        with pytest.raises(FaceColorUpdateError, match="without face colours"):
            plain.update_face_colors(random_hex(mesh.triangle_count, 1))
        first = random_hex(mesh.triangle_count, 2)
        handle = viewer.add_mesh_arrays(mesh, face_colors=first, **OPTIONS)
        image = capture(viewer, qapp)
        for wrong in (random_hex(7, 3), (), np.zeros((7, 3), np.uint8)):
            with pytest.raises(FaceColorUpdateError, match="count"):
                handle.update_face_colors(wrong)
        with pytest.raises(TypeError):
            handle.update_face_colors(["#000000"] * (mesh.triangle_count - 1) + [3.0])
        assert viewer._entries[id(handle)]["appearance"]["face_colors"] == first
        assert np.array_equal(capture(viewer, qapp), image)
    finally:
        viewer.destroy(); qapp.processEvents()


@qt_only
def test_software_microbench_600_faces(qapp):
    """Report replace-the-mesh versus in-place update for a 600-face grid."""
    mesh = grid()
    assert mesh.triangle_count == 600
    palettes = [random_hex(600, seed) for seed in range(6)]
    viewer = software_viewer(qapp)
    try:
        handle = viewer.add_mesh_arrays(mesh, face_colors=palettes[0], **OPTIONS)
        capture(viewer, qapp)
        # The software backend prepares at paint time, so each step includes
        # one painted frame (capture_image) to measure the work it causes.
        start = time.perf_counter()
        for colors in palettes * 5:
            handle.remove()
            handle = viewer.add_mesh_arrays(mesh, face_colors=colors, **OPTIONS)
            viewer.capture_image()
        replace = (time.perf_counter() - start) / 30
        start = time.perf_counter()
        for colors in palettes * 5:
            handle.update_face_colors(colors)
            viewer.capture_image()
        update = (time.perf_counter() - start) / 30
        print(f"\nsoftware 600 faces incl. one painted frame: replace {replace * 1e3:.3f} ms, "
              f"update {update * 1e3:.3f} ms")
        # Report only: the per-frame painter (projection, sorting, drawing)
        # dominates both and makes a timing assertion noisy; the retained-
        # preparation reuse itself is asserted in the equivalence test.
        assert update > 0 and replace > 0
    finally:
        viewer.destroy(); qapp.processEvents()
