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
