"""Numeric public face batches must match the established legacy submission."""
from dataclasses import fields
from uuid import UUID

import numpy as np
import pytest

from any3dview import ModelOwner, PickBinding, PickOwner
from any3dview.retained_viewer import RetainedViewer
from any3dview.ownership import PackedOwnerTable, _repeat_triangle_rows


class Capture(RetainedViewer):
    def __init__(self):
        self.outputs = []

    def add_mesh_arrays(self, mesh, **options):
        self.outputs.append((mesh, options))


class LegacyCapture(Capture):
    def _numeric_face_batch(self, *args, **kwargs):
        return False


def same_array(first, second):
    assert first.dtype == second.dtype and first.shape == second.shape
    assert first.tobytes() == second.tobytes()


def same_outputs(first, second):
    assert len(first) == len(second)
    for (mesh_a, options_a), (mesh_b, options_b) in zip(first, second):
        for descriptor in fields(mesh_a):
            a, b = getattr(mesh_a, descriptor.name), getattr(mesh_b, descriptor.name)
            if a is None or b is None:
                assert a is b
            else:
                same_array(a, b)
        assert options_a.keys() == options_b.keys()
        for key, a in options_a.items():
            b = options_b[key]
            if isinstance(a, PackedOwnerTable):
                for descriptor in fields(a):
                    av, bv = getattr(a, descriptor.name), getattr(b, descriptor.name)
                    if isinstance(av, np.ndarray):
                        same_array(av, bv)
                    else:
                        assert av == bv
            elif isinstance(a, np.ndarray):
                same_array(a, b)
            else:
                assert a == b


@pytest.mark.parametrize("outline", ["", "black"])
@pytest.mark.parametrize("colors, backs", [
    ("red", None),
    (["red", "blue", "red", "green", "red", "blue"], None),
    (["red", "blue", "red", "green", "red", "blue"],
     ["gray", "gray", "black", "black", "gray", "gray"]),
])
@pytest.mark.parametrize("smooth", [None, 35.])
def test_fans_edges_colors_and_owner_order_match_legacy(outline, colors, backs, smooth):
    rng = np.random.default_rng(7)
    polygons = [rng.normal(size=(count, 3)) for count in (0, 2, 3, 4, 5, 7)]
    a, b = PickOwner("a", "face", 2), PickOwner("b", "face", 1)
    bindings = [None, PickBinding((b,)), PickBinding((a,)),
                PickBinding((b, a)), None, PickBinding((a,))]
    options = dict(colors=colors, back_colors=backs, outline=outline,
                   bindings=bindings, smooth_angle=smooth, opacity=.7,
                   two_sided_shell=True, tags="frozen")
    fast, old = Capture(), LegacyCapture()
    fast.add_faces(polygons, **options)
    old.add_faces(polygons, **options)
    same_outputs(fast.outputs, old.outputs)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
def test_noncontiguous_arrays_signed_zero_and_source_mutation(dtype):
    source = np.array([[0, 9, 0, 9, 0, 9], [1, 9, 0, 9, 0, 9],
                       [0, 9, 1, 9, 0, 9]], dtype=dtype)
    polygon = source[:, ::2]
    if dtype in (np.float32, np.float64):
        polygon[0, 0] = -0.
    fast, old = Capture(), LegacyCapture()
    fast.add_faces([polygon], outline="black")
    old.add_faces([polygon], outline="black")
    same_outputs(fast.outputs, old.outputs)
    before = fast.outputs[0][0].positions.copy()
    source[:] = 7
    same_array(before, fast.outputs[0][0].positions)


@pytest.mark.parametrize("failure", ["nonfinite", "color_count", "binding_count"])
def test_failure_does_not_publish_the_numeric_batch(failure):
    polygon = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    options = {}
    if failure == "nonfinite":
        polygon[0, 0] = np.inf
    elif failure == "color_count":
        options["colors"] = ["red", "blue"]
    else:
        options["bindings"] = [None, None]
    fast, old = Capture(), LegacyCapture()
    for view in (fast, old):
        with pytest.raises(ValueError):
            view.add_faces([polygon], **options)
        assert view.outputs == []


def test_group_failure_preserves_same_preceding_publication():
    good = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    bad = good.copy()
    bad[0, 0] = np.nan
    fast, old = Capture(), LegacyCapture()
    for view in (fast, old):
        with pytest.raises(ValueError):
            view.add_faces([good, bad], colors=["red", "blue"],
                           back_colors=["gray", "black"])
        assert len(view.outputs) == 1
    same_outputs(fast.outputs, old.outputs)


def test_repeated_owner_csr_matches_direct_encoding_with_empty_and_shared_rows():
    model = ModelOwner(UUID(int=1), "face", 9, 5)
    a, b = PickOwner("a", "face", 1), PickOwner("b", "edge", 2)
    rows = [(a, a), None, PickBinding((b, a)), (model,), ()]
    repeats = np.array([2, 3, 1, 4, 2])
    actual = _repeat_triangle_rows(PackedOwnerTable.from_owners(triangles=rows), repeats, 11)
    expected = PackedOwnerTable.from_owners(
        triangles=[row for row, count in zip(rows, repeats) for _ in range(count)],
        lines=[None] * 11)
    for descriptor in fields(actual):
        a, b = getattr(actual, descriptor.name), getattr(expected, descriptor.name)
        if isinstance(a, np.ndarray):
            same_array(a, b)
        else:
            assert a == b


def test_empty_and_short_numeric_polygons_publish_nothing():
    fast, old = Capture(), LegacyCapture()
    for view in (fast, old):
        view.add_faces([np.empty((0, 3)), np.zeros((2, 3))], outline="black")
        assert view.outputs == []


@pytest.mark.parametrize("change", ["shape", "dtype", "coordinates"])
def test_argument_callbacks_preserve_eager_row_metadata_and_live_coordinates(change):
    fast, old = Capture(), LegacyCapture()
    for view in (fast, old):
        polygon = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        # Explicit row materialization reproduces the previous add_faces entry.
        submitted = [polygon] if view is fast else [list(polygon)]

        def colors():
            if change == "shape":
                polygon.shape = (1, 9)
            elif change == "dtype":
                polygon.dtype = np.int64
            else:
                polygon[1, 0] = 2.
            yield "red"

        view.add_faces(submitted, colors=colors(), outline="black")
    same_outputs(fast.outputs, old.outputs)


def test_short_polygons_keep_extended_precision_conversion_refusal():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform has no wider floating dtype")
    polygon = np.full((2, 3), np.finfo(np.longdouble).max, dtype=np.longdouble)
    for view in (Capture(), LegacyCapture()):
        with np.errstate(over="raise"), pytest.raises(FloatingPointError):
            view.add_faces([polygon])
        assert not view.outputs
