"""Independent geometric checks of display approximation and failure policy."""
from dataclasses import replace
from types import SimpleNamespace
import sys

import numpy as np
import pytest

pytest.importorskip("anygeometry")
from any3dview.adapters.anygeometry.policy import TessellationPolicy
from any3dview.adapters.anygeometry.tessellation import (
    UnsupportedDisplayGeometry, _refine_surface, _triangulate, sampled_edge,
)


def test_missing_triangulator_is_actionable_and_never_uses_a_fan(monkeypatch):
    monkeypatch.setitem(sys.modules, "mapbox_earcut", None)
    with pytest.raises(UnsupportedDisplayGeometry, match=r"ANY3dView\[geometry\]"):
        _triangulate([np.array([(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)])])


@pytest.mark.parametrize("loops, expected", [
    ([[(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)]], 7),
    ([[(0, 0), (4, 0), (4, 4), (0, 4)], [(1, 1), (1, 3), (3, 3), (3, 1)]], 12),
])
def test_concavity_and_holes_preserve_area(loops, expected):
    pytest.importorskip("mapbox_earcut")
    points, triangles = _triangulate([np.asarray(loop, float) for loop in loops])
    p = points[triangles]
    a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    assert np.abs(a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]).sum()/2 == pytest.approx(expected)


class Circle:
    edges = {1: SimpleNamespace(id=1, curve=object())}

    def evaluate_edge_many(self, _, t):
        theta = np.asarray(t) * np.pi / 2
        return np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))


def test_chord_tolerance_controls_circle_sagitta_and_relative_scale():
    policy = TessellationPolicy(chord_tolerance=.01, relative_chord_tolerance=1e-8, angular_tolerance=2.)
    loose = sampled_edge(Circle(), 1, policy, 2)
    strict = sampled_edge(Circle(), 1, replace(policy, chord_tolerance=.0001), 2)
    assert len(strict) > len(loose)
    # Unit circle midpoint sagitta is an independent analytic error reference.
    assert np.max(1 - np.linalg.norm((strict[:-1] + strict[1:])/2, axis=1)) <= .0001
    relative_loose = replace(policy, chord_tolerance=1e-8, relative_chord_tolerance=.01)
    assert len(sampled_edge(Circle(), 1, relative_loose, 2)) < len(strict)


def test_angular_tolerance_and_segment_budget_are_enforced():
    policy = TessellationPolicy(chord_tolerance=1., relative_chord_tolerance=1., angular_tolerance=.05)
    points = sampled_edge(Circle(), 1, policy, 2)
    assert len(points) > 5
    with pytest.raises(UnsupportedDisplayGeometry, match="max_curve_segments"):
        sampled_edge(Circle(), 1, replace(policy, max_curve_segments=4), 2)


class Paraboloid:
    def evaluate_face_many(self, _, uv):
        u, v = np.asarray(uv).T
        return np.column_stack((u, v, u*u + v*v))

    def face_normal_many(self, _, uv):
        u, v = np.asarray(uv).T
        return np.column_stack((-2*u, -2*v, np.ones_like(u)))


def test_surface_interior_refines_and_preserves_trim_domain():
    uv = np.array([(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
    triangles = np.array([(0, 1, 2), (0, 2, 3)], np.uint32)
    policy = TessellationPolicy(chord_tolerance=.01, relative_chord_tolerance=1e-8, angular_tolerance=.5)
    result = _refine_surface(Paraboloid(), 1, uv, triangles, policy, 2)
    assert len(result.triangles) > 2
    p = result.positions[result.triangles]
    centroid = p.mean(axis=1)
    exact_z = centroid[:, 0]**2 + centroid[:, 1]**2
    assert np.max(abs(centroid[:, 2] - exact_z)) <= .01
    assert np.all((result.positions[:, :2] >= 0) & (result.positions[:, :2] <= 1))
    a, b = p[:, 1, :2] - p[:, 0, :2], p[:, 2, :2] - p[:, 0, :2]
    assert np.abs(a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]).sum()/2 == pytest.approx(1.)
    with pytest.raises(UnsupportedDisplayGeometry, match="max_surface_triangles"):
        _refine_surface(Paraboloid(), 1, uv, triangles, replace(policy, max_surface_triangles=4), 2)


@pytest.mark.parametrize("field", ["chord_tolerance", "relative_chord_tolerance", "angular_tolerance"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0., -1.])
def test_invalid_tolerance_rejected(field, value):
    with pytest.raises(ValueError):
        TessellationPolicy(**{field: value})


@pytest.mark.parametrize("field", ["max_curve_segments", "lod_levels", "max_surface_triangles"])
def test_fractional_work_limits_are_rejected(field):
    with pytest.raises(ValueError, match="integer"):
        TessellationPolicy(**{field: 4.5})


def test_translated_polygon_retains_area_without_cancellation():
    pytest.importorskip("mapbox_earcut")
    square = np.asarray([(0., 0.), (2., 0.), (2., 2.), (0., 2.)]) + 1e12
    vertices, triangles = _triangulate([square])
    assert len(triangles) == 2
    assert np.array_equal(vertices, square)
