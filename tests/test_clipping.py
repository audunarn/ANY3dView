import math

import pytest

from any3dview import Point3D, SectionPlane


def tuples(points):
    return [point.to_tuple() for point in points]


def test_plane_normalizes_direction_and_keeps_offset_as_distance():
    plane = SectionPlane((2, 0, 0), 1.5)
    assert plane.normal.to_tuple() == (1.0, 0.0, 0.0)
    assert plane.offset == 1.5
    assert plane.contains((1.5, 0, 0))
    assert plane.contains((3, 0, 0))
    assert not plane.contains((1, 0, 0))


@pytest.mark.parametrize("normal", [(0, 0, 0), (math.inf, 0, 0), (math.nan, 0, 0)])
def test_plane_rejects_invalid_normals(normal):
    with pytest.raises(ValueError, match="normal"):
        SectionPlane(normal)


def test_plane_rejects_nonfinite_offset():
    with pytest.raises(ValueError, match="offset"):
        SectionPlane((1, 0, 0), math.inf)


def test_segment_is_kept_discarded_or_intersected():
    plane = SectionPlane((1, 0, 0), 0)
    assert plane.clip_segment((1, 0, 0), (2, 0, 0)) is not None
    assert plane.clip_segment((-2, 0, 0), (-1, 0, 0)) is None
    clipped = plane.clip_segment((-2, 0, 0), (2, 0, 0))
    assert clipped is not None
    assert tuples(clipped) == [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]


def test_polygon_is_clipped_without_a_generated_cap():
    plane = SectionPlane((1, 0, 0), 0)
    polygon = plane.clip_polygon(
        [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    )
    assert tuples(polygon) == [
        (0.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ]


def test_disabled_plane_is_transparent_to_queries():
    plane = SectionPlane((1, 0, 0), 5, enabled=False)
    assert plane.contains((-100, 0, 0))
    assert tuples(plane.clip_polygon([(-1, 0, 0), (1, 0, 0)])) == [
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ]
