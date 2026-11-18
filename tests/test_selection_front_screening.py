"""Batched, screened front queries must equal a direct front-to-back scan.

The visibility part of window/crossing/lasso selection asks for the front
primitive at many sample pixels.  The index answers those in batches over
small cells with a vectorised bounding-box test and a conservative triangle
half-plane test before the exact scalar test.  These checks compare every
answer with a brute-force scan over all primitives, on scenes built to hit
the screening edge cases: triangles of both windings, slivers, degenerate
triangles, samples exactly on vertices and edges, dense depth stacks, layers,
filters, off-screen samples and primitives larger than the tile limit.
"""
import random
import time

import numpy as np
import pytest

import any3dview.selection as selection
from any3dview import (
    PickBinding,
    ProjectedPrimitive,
    ProjectedSelectionIndex,
    SelectionDepth,
    SelectionFilter,
)
from any3dview.selection import _primitive_distance


class ReferenceIndex(ProjectedSelectionIndex):
    """The front query of any3dview 0.x before screening (origin/main c8033ae).

    Candidates are the tile members around the pixel, fully sorted front to
    back, then tested one by one; this is the behaviour the batched and
    screened query must reproduce exactly.
    """

    def _front_primitive_at(self, point, radius=.75, selection_filter=None):
        rect = (point[0]-radius, point[1]-radius, point[0]+radius, point[1]+radius)
        candidates = sorted(self._indices_for_rect(rect), key=lambda index: (
            self.primitives[index].depth, -self.primitives[index].layer, -index))
        for index in candidates:
            primitive = self.primitives[index]
            x0, y0, x1, y1 = primitive.bbox
            if point[0] < x0-radius or point[0] > x1+radius or point[1] < y0-radius or point[1] > y1+radius:
                continue
            if selection_filter is not None and primitive.shape != "polygon":
                binding = primitive.binding
                if binding is None or not any(selection_filter.accepts(owner) for owner in binding.owners):
                    continue
            if _primitive_distance(primitive, point) <= radius:
                return index
        return None

    def _visible_owners(self, owners, samples_of, selection_filter):
        # The previous per-owner loop: every sample is queried on its own.
        visible = set()
        for key, indices in owners.items():
            for index in indices:
                for sample in samples_of(self.primitives[index]):
                    front = self._front_primitive_at(sample, selection_filter=selection_filter)
                    if front is None:
                        continue
                    binding = self.primitives[front].binding
                    if front == index or (binding is not None and any(o.key == key for o in binding.owners)):
                        visible.add(key)
                        break
                if key in visible:
                    break
        return visible


def _scene(seed, count=400, size=256):
    rng = random.Random(seed)
    primitives = []
    for i in range(count):
        x, y = rng.uniform(-30, size + 30), rng.uniform(-30, size + 30)
        roll = rng.random()
        owner = f"o{rng.randrange(count // 6)}"
        if roll < .55:
            # Triangles: ordinary, slivers, both windings, some degenerate.
            span = rng.choice((2.0, 12.0, 60.0, 400.0))
            points = [(x, y), (x + rng.uniform(-span, span), y + rng.uniform(-span, span))]
            if rng.random() < .2:
                # Sliver: third corner almost on the first edge.
                t = rng.random()
                points.append((points[0][0] + t * (points[1][0] - points[0][0]) + rng.uniform(-.3, .3),
                               points[0][1] + t * (points[1][1] - points[0][1]) + rng.uniform(-.3, .3)))
            elif rng.random() < .05:
                points.append(points[1])  # degenerate
            else:
                points.append((x + rng.uniform(-span, span), y + rng.uniform(-span, span)))
            shape, kind = "polygon", "geometry.face"
        elif roll < .7:
            w, h = rng.uniform(1, 90), rng.uniform(1, 90)
            points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            shape, kind = "polygon", "geometry.face"
        elif roll < .9:
            points = [(x, y), (x + rng.uniform(-80, 80), y + rng.uniform(-80, 80))]
            shape, kind = "segment", "geometry.edge"
        else:
            points = [(x, y)]
            shape, kind = "point", "geometry.vertex"
        depth = float(rng.choice((1, 2, 2, 3, 4.5, rng.uniform(0, 5))))
        binding = None if rng.random() < .1 else PickBinding.one(owner + kind[-4:], kind)
        primitives.append(ProjectedPrimitive(
            i, shape, tuple(points), (depth,) * len(points), binding,
            layer=float(rng.choice((0, 0, 1))),
            radius=2.5 if shape == "segment" else 3.5 if shape == "point" else 0.0,
        ))
    return primitives


def _samples(primitives, seed, size=256):
    rng = random.Random(seed)
    points = [(rng.uniform(-5, size + 5), rng.uniform(-5, size + 5)) for _ in range(300)]
    for p in primitives[::3]:
        points.extend(p.points)
        for a, b in zip(p.points, p.points[1:] + p.points[:1]):
            points.append(((a[0] + b[0]) * .5, (a[1] + b[1]) * .5))
    # Pixel centres and integer pixels, as gestures produce.
    points.extend((float(x), float(y)) for x in range(0, size, 37) for y in range(0, size, 41))
    return points


FILTERS = (None, SelectionFilter(), SelectionFilter(kinds=frozenset({"geometry.face"})),
           SelectionFilter(kinds=frozenset({"geometry.edge"})))


@pytest.mark.parametrize("seed", (3, 17, 2026))
def test_batched_front_matches_full_scan(seed):
    primitives = _scene(seed)
    fast = ProjectedSelectionIndex(primitives, 256, 200)
    slow = ReferenceIndex(primitives, 256, 200)
    points = _samples(primitives, seed + 1)
    for filt in FILTERS:
        expected = [slow._front_primitive_at(p, selection_filter=filt) for p in points]
        assert fast._front_many(points, selection_filter=filt) == expected
        # One-at-a-time queries answer the same.
        assert [fast._front_primitive_at(p, selection_filter=filt) for p in points[:200]] == expected[:200]
        for radius in (0.0, 2.0):
            assert fast._front_many(points[:200], radius, filt) == [
                slow._front_primitive_at(p, radius, filt) for p in points[:200]]


@pytest.mark.parametrize("seed", (5, 99))
def test_screened_window_crossing_and_lasso_match_reference(seed):
    primitives = _scene(seed, count=300)
    fast = ProjectedSelectionIndex(primitives, 256, 200)
    slow = ReferenceIndex(primitives, 256, 200)
    rng = random.Random(seed)
    rects = [(0, 0, 256, 200), (-20, -20, 300, 260), (40, 30, 180, 150)] + [
        tuple(rng.uniform(-10, 260) for _ in range(4)) for _ in range(6)]
    for filt in FILTERS[1:]:
        for rect in rects:
            for crossing in (False, True):
                assert fast.rectangle_hits(rect, filt, crossing=crossing, depth=SelectionDepth.VISIBLE) == \
                    slow.rectangle_hits(rect, filt, crossing=crossing, depth=SelectionDepth.VISIBLE)
        lasso = ((10, 10), (240, 30), (200, 190), (30, 160))
        assert fast.polygon_hits(lasso, filt, depth=SelectionDepth.VISIBLE) == \
            slow.polygon_hits(lasso, filt, depth=SelectionDepth.VISIBLE)


def _stacked_plates(rows=40, columns=40, layers=6):
    """A tessellated plate stack seen face-on: only the front layer is visible."""
    primitives = []
    for layer in range(layers):
        for row in range(rows):
            for column in range(columns):
                x0, y0 = 10 + column * 5.0, 10 + row * 5.0
                quad = ((x0, y0), (x0 + 5, y0), (x0 + 5, y0 + 5), (x0, y0 + 5))
                for points in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
                    primitives.append(ProjectedPrimitive(
                        len(primitives), "polygon", points, (1.0 + layer,) * 3,
                        PickBinding.one(f"plate{layer}", "geometry.face")))
    return primitives


def test_hidden_owner_window_query_does_bounded_exact_work(monkeypatch):
    """Regression: hidden plates used to rescan whole depth stacks per sample.

    Window-selecting a stack of tessellated plates checks every sample of
    every hidden plate.  The exact distance test used to run on every
    primitive whose box reached a sample (millions of calls here); screened
    batches keep it to a few calls per sample.
    """
    primitives = _stacked_plates()
    index = ProjectedSelectionIndex(primitives, 220, 220)
    calls = [0]
    original = selection._primitive_distance

    def counted(primitive, point):
        calls[0] += 1
        return original(primitive, point)

    monkeypatch.setattr(selection, "_primitive_distance", counted)
    start = time.perf_counter()
    hits = index.rectangle_hits((0, 0, 220, 220), SelectionFilter(), crossing=False,
                                depth=SelectionDepth.VISIBLE)
    elapsed = time.perf_counter() - start
    assert [hit.key for hit in hits] == ["plate0"]
    samples = len({sample for p in primitives for sample in ProjectedSelectionIndex._samples(p)})
    assert calls[0] <= 4 * samples, (calls[0], samples)
    assert elapsed < 20.0  # generous wall-clock guard; the count above is the contract


# Review findings (Codex round 1), each reproduced against the previous query.

def _same(primitives, width, height, rect, *, filt=None, crossing=False, tile_size=16):
    filt = filt or SelectionFilter()
    fast = ProjectedSelectionIndex(primitives, width, height, tile_size=tile_size)
    slow = ReferenceIndex(primitives, width, height, tile_size=tile_size)
    for depth in (SelectionDepth.VISIBLE, SelectionDepth.THROUGH):
        # repr: NaN depths compare unequal to themselves.
        assert repr(fast.rectangle_hits(rect, filt, crossing=crossing, depth=depth)) == \
            repr(slow.rectangle_hits(rect, filt, crossing=crossing, depth=depth))
    return fast


@pytest.mark.parametrize("tile_size", (8, 9, 10, 11, 13, 16, 17))
@pytest.mark.parametrize("radius", (0.0, 3.5))
def test_fine_cells_spanning_two_tiles_keep_every_candidate(tile_size, radius):
    point = ProjectedPrimitive(0, "point", ((9.2, 5.0),), (1.0,), PickBinding.one("p", "geometry.vertex"),
                               radius=radius)
    _same([point], 32, 32, (0, 0, 32, 32), tile_size=tile_size)
    primitives = _scene(tile_size, count=200, size=96)
    for crossing in (False, True):
        _same(primitives, 96, 80, (-5, -5, 101, 85), crossing=crossing, tile_size=tile_size)


@pytest.mark.parametrize("radius", (0.0, 3.5))
def test_nan_depths_keep_the_scalar_tile_order(radius):
    nan = float("nan")
    primitives = [
        ProjectedPrimitive(index, "point", (position,), (depth,), PickBinding.one(str(index), "geometry.vertex"),
                           radius=radius)
        for index, (position, depth) in enumerate((((5, 5), 0.0), ((25, 5), 0.0), ((5, 5), nan)))
    ]
    for crossing in (False, True):
        _same(primitives, 32, 32, (4, 4, 6, 6), crossing=crossing)


@pytest.mark.parametrize("radius", (0.0, 3.5))
def test_huge_finite_triangle_is_never_screened_out(radius):
    occluder = ProjectedPrimitive(0, "polygon", ((3e17, 2.4e17), (-3e17 + 448, -2.4e17 + 128), (5.4e17, -7.9e17)),
                                  (0.0,) * 3, None)
    point = ProjectedPrimitive(1, "point", ((349.0, 144.0),), (1.0,), PickBinding.one("p", "geometry.vertex"),
                               radius=radius)
    _same([occluder, point], 512, 512, (348, 143, 350, 145))


def test_integer_coordinates_beyond_float_precision_use_the_scalar_query():
    big = 2 ** 60
    primitives = [
        ProjectedPrimitive(0, "polygon", ((big, 0), (big + 1, 0), (big, 1)), (0.0,) * 3, None),
        ProjectedPrimitive(1, "point", ((3, 3),), (1.0,), PickBinding.one("p", "geometry.vertex"), radius=3.5),
    ]
    _same(primitives, 16, 16, (0, 0, 16, 16))


def test_retained_front_candidates_are_bounded(monkeypatch):
    rng = random.Random(8)
    primitives = [
        ProjectedPrimitive(i, "polygon", ((rng.uniform(-900, -800), rng.uniform(-900, -800)),
                                          (rng.uniform(1200, 1300), rng.uniform(-900, -800)),
                                          (rng.uniform(0, 300), rng.uniform(1200, 1300))),
                           (0.5,) * 3, None)
        for i in range(100)
    ]
    primitives += [
        ProjectedPrimitive(100 + i, "point", ((float(i % 32) * 8 + 2, float(i // 32) * 8 + 2),), (1.0,),
                           PickBinding.one(f"p{i}", "geometry.vertex"), radius=3.5)
        for i in range(1024)
    ]
    monkeypatch.setattr(ProjectedSelectionIndex, "_FRONT_CACHE_IDS", 20_000)
    fast = _same(primitives, 256, 256, (0, 0, 256, 256))
    assert fast._front_retained <= 20_000
    assert sum(len(ids) + fast._FRONT_ENTRY_COST for ids in fast._front_fine.values()) == fast._front_retained


# Review findings (Codex round 2).

@pytest.mark.parametrize("number", (np.float16, np.float32, np.longdouble, np.int32, np.int64))
@pytest.mark.parametrize("radius", (0.75, 3.5))
def test_non_float64_coordinates_keep_scalar_arithmetic(number, radius):
    triangle = ProjectedPrimitive(0, "polygon", tuple(tuple(number(v) for v in p)
                                                      for p in ((0, 0), (0, 1000), (1000, 0))), (0.0,) * 3, None)
    point = ProjectedPrimitive(1, "point", ((800.0, 800.0),), (1.0,), PickBinding.one("p"), radius=radius)
    _same([triangle, point], 1024, 1024, (795, 795, 805, 805))
    rng = random.Random(4)
    cloud = [ProjectedPrimitive(i, "polygon", tuple((number(rng.uniform(0, 200)), number(rng.uniform(0, 200)))
                                                    for _ in range(3)), (float(i % 5),) * 3,
                                PickBinding.one(f"f{i}", "geometry.face")) for i in range(150)]
    _same(cloud, 220, 220, (0, 0, 220, 220))


def test_filter_masks_are_bounded():
    primitives = [ProjectedPrimitive(i, "point", ((5.0, 5.0),), (0.0,), PickBinding.one(f"p{i}:"), radius=3.5)
                  for i in range(300)]
    index = ProjectedSelectionIndex(primitives, 16, 16)
    reference = ReferenceIndex(primitives, 16, 16)
    for i in range(40):
        filt = SelectionFilter(key_prefixes=(f"p{i}:",))
        assert index.rectangle_hits((4, 4, 6, 6), filt, crossing=False, depth=SelectionDepth.VISIBLE) == \
            reference.rectangle_hits((4, 4, 6, 6), filt, crossing=False, depth=SelectionDepth.VISIBLE)
        assert len(index._front_opaque) <= ProjectedSelectionIndex._FRONT_FILTER_MASKS


# Review findings (Codex round 3): only plain Python numbers are screened.

@pytest.mark.parametrize("radius", (0.0, 3.5))
def test_equal_samples_of_different_types_are_not_shared(radius):
    occluder = ProjectedPrimitive(0, "polygon", ((0.0, 0.0), (0.0, 1000.0), (1000.0, 0.0)), (0.0,) * 3, None)
    a = ProjectedPrimitive(1, "point", ((800.0, 800.0),), (2.0,), PickBinding.one("a"), radius=radius)
    b = ProjectedPrimitive(2, "point", ((np.float16(800), np.float16(800)),), (1.0,), PickBinding.one("b"),
                           radius=radius)
    _same([occluder, a, b], 1024, 1024, (795, 795, 805, 805))


@pytest.mark.parametrize("radius", (0.0, 3.5))
def test_mixed_numeric_depth_types_keep_the_scalar_order(radius):
    primitives = [
        ProjectedPrimitive(i, "point", (position,), (depth,), PickBinding.one(str(i)), radius=radius)
        for i, (position, depth) in enumerate((((5.0, 5.0), 2 ** 53), ((100.0, 100.0), np.float64(2 ** 53)),
                                               ((5.0, 5.0), 2 ** 53 + 1)))
    ]
    _same(primitives, 128, 128, (4, 4, 6, 6))


@pytest.mark.parametrize("radius", (np.float32(0.75), np.float16(0.75), 1, np.float64(0.75)))
def test_non_python_float_radius_uses_the_scalar_query(radius):
    primitives = [ProjectedPrimitive(0, "point", ((5.0, 5.0),), (1.0,), PickBinding.one("p"))]
    fast = ProjectedSelectionIndex(primitives, 16, 16)
    slow = ReferenceIndex(primitives, 16, 16)
    for x in (5.75000001, 5.75, 5.7499999, 6.0):
        assert fast._front_primitive_at((x, 5.0), radius) == slow._front_primitive_at((x, 5.0), radius)


def test_empty_candidate_entries_count_against_the_cache_bound():
    primitives = [ProjectedPrimitive(0, "point", ((5.0, 5.0),), (1.0,), PickBinding.one("p"))]
    index = ProjectedSelectionIndex(primitives, 16, 16)
    for i in range(4000):
        index._front_primitive_at((5.0, 5.0), selection_filter=SelectionFilter(key_prefixes=(f"absent{i}:",)))
    bound = index._FRONT_CACHE_IDS // index._FRONT_ENTRY_COST + 1
    assert len(index._front_fine) <= bound
    assert len(index._front_opaque) <= index._FRONT_FILTER_MASKS
