"""
Backend-neutral tessellation helpers for common 3D shapes.

Every builder returns a :class:`Mesh` - a vertex list plus index faces -
expressed in a local frame where the shape's own axis is +Z and the origin
sits at the shape's reference point (usually its centre).  ``Mesh.placed``
moves that local frame anywhere in the world, so the same tessellation code
serves ``add_box``, ``add_cone``, ``add_beam`` and friends.

Faces are wound counter-clockwise seen from outside, which makes the face
normal point away from the solid.  The renderer relies on that convention
for back-face culling and for two-sided lighting.

The module is pure geometry: it imports nothing from Tkinter and can be
used and tested headlessly.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

from .core import _EPS, Point3D, as_point


Vector = Tuple[float, float, float]
Face = Tuple[int, ...]
Profile2D = Sequence[Tuple[float, float]]


# ----------------------------------------------------------------------
# Mesh container and frame placement
# ----------------------------------------------------------------------


class Mesh:
    """A vertex list with index faces."""

    __slots__ = ("vertices", "faces")

    def __init__(self, vertices: Sequence[Vector], faces: Sequence[Face]) -> None:
        self.vertices: List[Vector] = [
            (float(v[0]), float(v[1]), float(v[2])) for v in vertices
        ]
        self.faces: List[Face] = [tuple(int(index) for index in face) for face in faces]

    def __repr__(self) -> str:
        return f"Mesh({len(self.vertices)} vertices, {len(self.faces)} faces)"

    def merged(self, other: "Mesh") -> "Mesh":
        offset = len(self.vertices)
        return Mesh(
            self.vertices + other.vertices,
            self.faces + [tuple(index + offset for index in face) for face in other.faces],
        )

    def scaled(self, sx: float, sy: float, sz: float) -> "Mesh":
        return Mesh(
            [(v[0] * sx, v[1] * sy, v[2] * sz) for v in self.vertices],
            self.faces,
        )

    def placed(
        self,
        origin: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        up_hint: Optional[Point3D] = None,
    ) -> "Mesh":
        """Map the local frame so that local +Z follows ``axis`` at ``origin``."""
        if origin is None and axis is None:
            return self
        ox, oy, oz = (0.0, 0.0, 0.0) if origin is None else as_point(origin).to_tuple()
        if axis is None:
            return Mesh(
                [(v[0] + ox, v[1] + oy, v[2] + oz) for v in self.vertices],
                self.faces,
            )
        ex, ey, ez = frame_from_axis(axis, up_hint)
        return Mesh(
            [
                (
                    ox + v[0] * ex.x + v[1] * ey.x + v[2] * ez.x,
                    oy + v[0] * ex.y + v[1] * ey.y + v[2] * ez.y,
                    oz + v[0] * ex.z + v[1] * ey.z + v[2] * ez.z,
                )
                for v in self.vertices
            ],
            self.faces,
        )

    def points(self) -> List[Point3D]:
        return [Point3D(*vertex) for vertex in self.vertices]

    def face_points(self) -> List[List[Point3D]]:
        points = self.points()
        return [[points[index] for index in face] for face in self.faces]

    def bounds(self) -> Tuple[Point3D, Point3D]:
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        if not xs:
            zero = Point3D(0.0, 0.0, 0.0)
            return zero, zero
        return (
            Point3D(min(xs), min(ys), min(zs)),
            Point3D(max(xs), max(ys), max(zs)),
        )


def merge_meshes(meshes: Iterable[Mesh]) -> Mesh:
    vertices: List[Vector] = []
    faces: List[Face] = []
    for mesh in meshes:
        offset = len(vertices)
        vertices.extend(mesh.vertices)
        faces.extend(tuple(index + offset for index in face) for face in mesh.faces)
    return Mesh(vertices, faces)


def frame_from_axis(
    axis: Point3D,
    up_hint: Optional[Point3D] = None,
) -> Tuple[Point3D, Point3D, Point3D]:
    """Return an orthonormal right-handed frame whose third vector is ``axis``."""
    ez = as_point(axis).normalized()
    if ez.length() <= _EPS:
        ez = Point3D(0.0, 0.0, 1.0)

    hint = as_point(up_hint) if up_hint is not None else None
    if hint is None or abs(hint.normalized().dot(ez)) > 1.0 - 1.0e-6:
        hint = Point3D(0.0, 0.0, 1.0)
        if abs(ez.z) > 0.9:
            hint = Point3D(1.0, 0.0, 0.0)

    ex = hint.cross(ez)
    if ex.length() <= _EPS:
        ex = Point3D(1.0, 0.0, 0.0).cross(ez)
    ex = ex.normalized()
    ey = ez.cross(ex).normalized()
    return ex, ey, ez


# ----------------------------------------------------------------------
# Polygon helpers
# ----------------------------------------------------------------------


def polygon_area_2d(profile: Profile2D) -> float:
    """Signed area; positive when the profile winds counter-clockwise."""
    total = 0.0
    count = len(profile)
    for index in range(count):
        x0, y0 = profile[index]
        x1, y1 = profile[(index + 1) % count]
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def ensure_ccw(profile: Profile2D) -> List[Tuple[float, float]]:
    points = [(float(x), float(y)) for x, y in profile]
    if polygon_area_2d(points) < 0.0:
        points.reverse()
    return points


def triangulate_polygon(profile: Profile2D) -> List[Tuple[int, int, int]]:
    """
    Ear-clip a simple, possibly concave polygon.

    Returns index triples wound the same way as the input outline.  Only
    reflex vertices can block an ear, and a vertex lying exactly on a
    candidate ear's edge blocks it too - structural sections such as I and
    channel profiles put vertices in precisely those positions, and clipping
    through one would fold the triangulation over itself.
    """
    points = [(float(x), float(y)) for x, y in profile]
    count = len(points)
    if count < 3:
        return []
    if count == 3:
        return [(0, 1, 2)]

    indices = list(range(count))
    if polygon_area_2d(points) < 0.0:
        indices.reverse()

    scale = max(
        max(abs(x) for x, _ in points),
        max(abs(y) for _, y in points),
        1.0e-9,
    )
    tolerance = scale * scale * 1.0e-12

    def cross(ax: float, ay: float, bx: float, by: float) -> float:
        return ax * by - ay * bx

    def turn(previous: int, current: int, following: int) -> float:
        px, py = points[previous]
        cx, cy = points[current]
        nx, ny = points[following]
        return cross(cx - px, cy - py, nx - cx, ny - cy)

    def contains(previous: int, current: int, following: int, probe: int) -> bool:
        ax, ay = points[previous]
        bx, by = points[current]
        cx, cy = points[following]
        px, py = points[probe]
        return (
            cross(bx - ax, by - ay, px - ax, py - ay) >= 0.0
            and cross(cx - bx, cy - by, px - bx, py - by) >= 0.0
            and cross(ax - cx, ay - cy, px - cx, py - cy) >= 0.0
        )

    def neighbours(position: int) -> Tuple[int, int, int]:
        return (
            indices[position - 1],
            indices[position],
            indices[(position + 1) % len(indices)],
        )

    triangles: List[Tuple[int, int, int]] = []
    guard = 0
    while len(indices) > 3 and guard < 4 * count * count:
        guard += 1

        # Collinear vertices are not ears and never will be; drop them so the
        # clip keeps making progress.
        degenerate = next(
            (
                position
                for position in range(len(indices))
                if abs(turn(*neighbours(position))) <= tolerance
            ),
            None,
        )
        if degenerate is not None:
            indices.pop(degenerate)
            continue

        reflex = {
            indices[position]
            for position in range(len(indices))
            if turn(*neighbours(position)) < 0.0
        }

        clipped = False
        for position in range(len(indices)):
            previous, current, following = neighbours(position)
            if current in reflex:
                continue
            if any(
                contains(previous, current, following, probe)
                for probe in reflex
                if probe not in (previous, current, following)
            ):
                continue
            triangles.append((previous, current, following))
            indices.pop(position)
            clipped = True
            break

        if not clipped:
            # Self-intersecting or otherwise invalid input: stop cleanly and
            # let the caller keep whatever has been produced so far.
            break

    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
    elif len(indices) > 3:
        triangles.extend(
            (indices[0], indices[position], indices[position + 1])
            for position in range(1, len(indices) - 1)
        )

    return [
        triangle
        for triangle in triangles
        if abs(
            polygon_area_2d(
                (points[triangle[0]], points[triangle[1]], points[triangle[2]])
            )
        )
        > tolerance
    ]


# ----------------------------------------------------------------------
# Solids
# ----------------------------------------------------------------------


def box(
    size_x: float = 1.0,
    size_y: Optional[float] = None,
    size_z: Optional[float] = None,
) -> Mesh:
    """Axis-aligned box centred on the local origin."""
    hx = 0.5 * abs(float(size_x))
    hy = 0.5 * abs(float(size_y if size_y is not None else size_x))
    hz = 0.5 * abs(float(size_z if size_z is not None else size_x))
    vertices = [
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
    ]
    faces = [
        (0, 3, 2, 1),  # -Z
        (4, 5, 6, 7),  # +Z
        (0, 1, 5, 4),  # -Y
        (1, 2, 6, 5),  # +X
        (2, 3, 7, 6),  # +Y
        (3, 0, 4, 7),  # -X
    ]
    return Mesh(vertices, faces)


def box_from_bounds(minimum: Point3D, maximum: Point3D) -> Mesh:
    low = as_point(minimum)
    high = as_point(maximum)
    centre = Point3D(
        0.5 * (low.x + high.x),
        0.5 * (low.y + high.y),
        0.5 * (low.z + high.z),
    )
    return box(high.x - low.x, high.y - low.y, high.z - low.z).placed(origin=centre)


def sphere(radius: float = 1.0, segments: int = 24, rings: int = 16) -> Mesh:
    """UV sphere centred on the local origin."""
    radius = abs(float(radius))
    segments = max(3, int(segments))
    rings = max(2, int(rings))

    vertices: List[Vector] = []
    faces: List[Face] = []

    south = len(vertices)
    vertices.append((0.0, 0.0, -radius))
    ring_start: List[int] = []
    for ring in range(1, rings):
        latitude = -0.5 * math.pi + math.pi * ring / rings
        cos_lat = math.cos(latitude)
        z = radius * math.sin(latitude)
        ring_start.append(len(vertices))
        for segment in range(segments):
            longitude = 2.0 * math.pi * segment / segments
            vertices.append(
                (
                    radius * cos_lat * math.cos(longitude),
                    radius * cos_lat * math.sin(longitude),
                    z,
                )
            )
    north = len(vertices)
    vertices.append((0.0, 0.0, radius))

    first = ring_start[0]
    for segment in range(segments):
        following = (segment + 1) % segments
        faces.append((south, first + following, first + segment))

    for ring in range(len(ring_start) - 1):
        lower = ring_start[ring]
        upper = ring_start[ring + 1]
        for segment in range(segments):
            following = (segment + 1) % segments
            faces.append(
                (
                    lower + segment,
                    lower + following,
                    upper + following,
                    upper + segment,
                )
            )

    last = ring_start[-1]
    for segment in range(segments):
        following = (segment + 1) % segments
        faces.append((last + segment, last + following, north))

    return Mesh(vertices, faces)


def frustum(
    radius_bottom: float = 1.0,
    radius_top: Optional[float] = None,
    height: float = 1.0,
    segments: int = 32,
    height_segments: int = 1,
    capped: bool = True,
) -> Mesh:
    """Cylinder / cone / truncated cone centred on the local origin."""
    radius_bottom = max(0.0, float(radius_bottom))
    radius_top = radius_bottom if radius_top is None else max(0.0, float(radius_top))
    height = abs(float(height))
    segments = max(3, int(segments))
    height_segments = max(1, int(height_segments))

    angles = [2.0 * math.pi * index / segments for index in range(segments)]
    cosines = [math.cos(angle) for angle in angles]
    sines = [math.sin(angle) for angle in angles]

    vertices: List[Vector] = []
    faces: List[Face] = []
    rings: List[Optional[int]] = []
    apexes: List[Optional[int]] = []

    for level in range(height_segments + 1):
        fraction = level / height_segments
        z = -0.5 * height + fraction * height
        radius = radius_bottom + fraction * (radius_top - radius_bottom)
        if radius <= _EPS:
            rings.append(None)
            apexes.append(len(vertices))
            vertices.append((0.0, 0.0, z))
            continue
        apexes.append(None)
        rings.append(len(vertices))
        for index in range(segments):
            vertices.append((radius * cosines[index], radius * sines[index], z))

    for level in range(height_segments):
        lower = rings[level]
        upper = rings[level + 1]
        for index in range(segments):
            following = (index + 1) % segments
            if lower is None and upper is None:
                continue
            if lower is None:
                faces.append((apexes[level], upper + following, upper + index))
            elif upper is None:
                faces.append((lower + index, lower + following, apexes[level + 1]))
            else:
                faces.append(
                    (
                        lower + index,
                        lower + following,
                        upper + following,
                        upper + index,
                    )
                )

    if capped:
        if rings[0] is not None:
            base = rings[0]
            faces.append(tuple(base + index for index in range(segments - 1, -1, -1)))
        if rings[-1] is not None:
            top = rings[-1]
            faces.append(tuple(top + index for index in range(segments)))

    return Mesh(vertices, faces)


def cylinder(
    radius: float = 1.0,
    height: float = 1.0,
    segments: int = 32,
    height_segments: int = 1,
    capped: bool = True,
) -> Mesh:
    return frustum(radius, radius, height, segments, height_segments, capped)


def cone(
    radius: float = 1.0,
    height: float = 1.0,
    segments: int = 32,
    capped: bool = True,
) -> Mesh:
    return frustum(radius, 0.0, height, segments, 1, capped)


def tube(
    outer_radius: float = 1.0,
    inner_radius: float = 0.7,
    height: float = 1.0,
    segments: int = 32,
    height_segments: int = 1,
    capped: bool = True,
) -> Mesh:
    """Hollow pipe centred on the local origin."""
    outer_radius = max(0.0, float(outer_radius))
    inner_radius = max(0.0, min(float(inner_radius), outer_radius - _EPS))
    if inner_radius <= _EPS:
        return frustum(outer_radius, outer_radius, height, segments, height_segments, capped)

    height = abs(float(height))
    segments = max(3, int(segments))

    outer = frustum(outer_radius, outer_radius, height, segments, height_segments, False)
    inner = frustum(inner_radius, inner_radius, height, segments, height_segments, False)
    inner = Mesh(inner.vertices, [tuple(reversed(face)) for face in inner.faces])
    mesh = outer.merged(inner)

    if not capped:
        return mesh

    caps: List[Face] = []
    vertices = list(mesh.vertices)
    for z, upward in ((0.5 * height, True), (-0.5 * height, False)):
        base = len(vertices)
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            vertices.append((inner_radius * math.cos(angle), inner_radius * math.sin(angle), z))
            vertices.append((outer_radius * math.cos(angle), outer_radius * math.sin(angle), z))
        for index in range(segments):
            following = (index + 1) % segments
            quad = (
                base + 2 * index,
                base + 2 * index + 1,
                base + 2 * following + 1,
                base + 2 * following,
            )
            caps.append(quad if upward else tuple(reversed(quad)))
    return Mesh(vertices, mesh.faces + caps)


def disk(
    outer_radius: float = 1.0,
    inner_radius: float = 0.0,
    segments: int = 32,
) -> Mesh:
    """Flat disk or annulus in the local XY plane, normal along +Z."""
    outer_radius = max(0.0, float(outer_radius))
    inner_radius = max(0.0, min(float(inner_radius), outer_radius))
    segments = max(3, int(segments))

    vertices: List[Vector] = []
    faces: List[Face] = []
    if inner_radius <= _EPS:
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            vertices.append((outer_radius * math.cos(angle), outer_radius * math.sin(angle), 0.0))
        faces.append(tuple(range(segments)))
        return Mesh(vertices, faces)

    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        vertices.append((inner_radius * math.cos(angle), inner_radius * math.sin(angle), 0.0))
        vertices.append((outer_radius * math.cos(angle), outer_radius * math.sin(angle), 0.0))
    for index in range(segments):
        following = (index + 1) % segments
        faces.append(
            (
                2 * index,
                2 * index + 1,
                2 * following + 1,
                2 * following,
            )
        )
    return Mesh(vertices, faces)


def torus(
    major_radius: float = 1.0,
    minor_radius: float = 0.25,
    segments: int = 36,
    rings: int = 18,
) -> Mesh:
    """Torus around the local Z axis."""
    major_radius = max(0.0, float(major_radius))
    minor_radius = max(0.0, float(minor_radius))
    segments = max(3, int(segments))
    rings = max(3, int(rings))

    vertices: List[Vector] = []
    for u_index in range(segments):
        u = 2.0 * math.pi * u_index / segments
        cos_u, sin_u = math.cos(u), math.sin(u)
        for v_index in range(rings):
            v = 2.0 * math.pi * v_index / rings
            radius = major_radius + minor_radius * math.cos(v)
            vertices.append((radius * cos_u, radius * sin_u, minor_radius * math.sin(v)))

    faces: List[Face] = []
    for u_index in range(segments):
        u_next = (u_index + 1) % segments
        for v_index in range(rings):
            v_next = (v_index + 1) % rings
            faces.append(
                (
                    u_index * rings + v_index,
                    u_next * rings + v_index,
                    u_next * rings + v_next,
                    u_index * rings + v_next,
                )
            )
    return Mesh(vertices, faces)


def pyramid(
    base_radius: float = 1.0,
    height: float = 1.0,
    sides: int = 4,
    capped: bool = True,
) -> Mesh:
    """Regular pyramid; the base sits at local z = -height/2."""
    base_radius = max(0.0, float(base_radius))
    height = abs(float(height))
    sides = max(3, int(sides))

    z_base = -0.5 * height
    # A four-sided pyramid reads best as an axis-aligned square base.
    offset = math.pi / 4.0 if sides == 4 else 0.0
    vertices: List[Vector] = [
        (
            base_radius * math.cos(2.0 * math.pi * index / sides + offset),
            base_radius * math.sin(2.0 * math.pi * index / sides + offset),
            z_base,
        )
        for index in range(sides)
    ]
    apex = len(vertices)
    vertices.append((0.0, 0.0, 0.5 * height))

    faces: List[Face] = [
        (index, (index + 1) % sides, apex) for index in range(sides)
    ]
    if capped:
        faces.append(tuple(range(sides - 1, -1, -1)))
    return Mesh(vertices, faces)


def wedge(
    size_x: float = 1.0,
    size_y: float = 1.0,
    size_z: float = 1.0,
) -> Mesh:
    """Triangular prism: a box cut along its X/Z diagonal."""
    hx = 0.5 * abs(float(size_x))
    hy = 0.5 * abs(float(size_y))
    hz = 0.5 * abs(float(size_z))
    vertices = [
        (-hx, -hy, -hz), (hx, -hy, -hz), (-hx, -hy, hz),
        (-hx, hy, -hz), (hx, hy, -hz), (-hx, hy, hz),
    ]
    faces = [
        (0, 1, 2),           # -Y triangular end
        (3, 5, 4),           # +Y triangular end
        (0, 3, 4, 1),        # bottom
        (0, 2, 5, 3),        # -X back
        (1, 4, 5, 2),        # slope
    ]
    return Mesh(vertices, faces)


def prism(profile: Profile2D, height: float = 1.0, capped: bool = True) -> Mesh:
    """Extrude a 2D profile along local Z, centred on the local origin."""
    points = ensure_ccw(profile)
    count = len(points)
    if count < 3:
        return Mesh([], [])
    height = float(height)
    z_low = -0.5 * height
    z_high = 0.5 * height

    vertices: List[Vector] = [(x, y, z_low) for x, y in points]
    vertices.extend((x, y, z_high) for x, y in points)

    faces: List[Face] = []
    for index in range(count):
        following = (index + 1) % count
        faces.append((index, following, count + following, count + index))

    if capped:
        triangles = triangulate_polygon(points)
        faces.extend((a + count, b + count, c + count) for a, b, c in triangles)
        faces.extend((c, b, a) for a, b, c in triangles)
    return Mesh(vertices, faces)


def hollow_prism(
    outer_profile: Profile2D,
    inner_profile: Profile2D,
    height: float = 1.0,
    capped: bool = True,
) -> Mesh:
    """
    Extrude a profile with a matching hole along local Z.

    Both outlines must have the same number of points, given in
    corresponding order, so the end caps can be closed with quads.  This is
    how rectangular hollow sections and any other constant-wall tube are
    built.
    """
    outer = ensure_ccw(outer_profile)
    inner = ensure_ccw(inner_profile)
    count = len(outer)
    if count < 3 or len(inner) != count:
        return prism(outer_profile, height, capped)

    height = float(height)
    z_low = -0.5 * height
    z_high = 0.5 * height

    vertices: List[Vector] = [(x, y, z_low) for x, y in outer]
    vertices.extend((x, y, z_high) for x, y in outer)
    vertices.extend((x, y, z_low) for x, y in inner)
    vertices.extend((x, y, z_high) for x, y in inner)
    inner_low = 2 * count
    inner_high = 3 * count

    faces: List[Face] = []
    for index in range(count):
        following = (index + 1) % count
        faces.append((index, following, count + following, count + index))
        faces.append(
            (
                inner_low + following,
                inner_low + index,
                inner_high + index,
                inner_high + following,
            )
        )

    if capped:
        for index in range(count):
            following = (index + 1) % count
            faces.append(
                (
                    inner_high + index,
                    count + index,
                    count + following,
                    inner_high + following,
                )
            )
            faces.append(
                (
                    inner_low + following,
                    following,
                    index,
                    inner_low + index,
                )
            )
    return Mesh(vertices, faces)


def rectangular_tube(
    width: float = 0.3,
    height: float = 0.3,
    thickness: float = 0.02,
    length: float = 1.0,
    capped: bool = True,
) -> Mesh:
    """Rectangular hollow section extruded along local Z."""
    half_w = 0.5 * abs(float(width))
    half_h = 0.5 * abs(float(height))
    wall = max(_EPS, min(abs(float(thickness)), min(half_w, half_h) - _EPS))
    outer = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    inner = [
        (-half_w + wall, -half_h + wall),
        (half_w - wall, -half_h + wall),
        (half_w - wall, half_h - wall),
        (-half_w + wall, half_h - wall),
    ]
    return hollow_prism(outer, inner, length, capped)


def extrusion(
    profile: Profile2D,
    path: Sequence[Point3D],
    capped: bool = True,
    up_hint: Optional[Point3D] = None,
) -> Mesh:
    """Sweep a 2D profile along a polyline, keeping the profile perpendicular."""
    points = ensure_ccw(profile)
    stations = [as_point(point) for point in path]
    if len(points) < 3 or len(stations) < 2:
        return Mesh([], [])

    count = len(points)
    vertices: List[Vector] = []
    for index, station in enumerate(stations):
        if index == 0:
            axis = stations[1] - stations[0]
        elif index == len(stations) - 1:
            axis = stations[-1] - stations[-2]
        else:
            axis = stations[index + 1] - stations[index - 1]
        ex, ey, _ez = frame_from_axis(axis, up_hint)
        for x, y in points:
            vertices.append(
                (
                    station.x + x * ex.x + y * ey.x,
                    station.y + x * ex.y + y * ey.y,
                    station.z + x * ex.z + y * ey.z,
                )
            )

    faces: List[Face] = []
    for station_index in range(len(stations) - 1):
        base = station_index * count
        following_base = base + count
        for index in range(count):
            following = (index + 1) % count
            faces.append(
                (
                    base + index,
                    base + following,
                    following_base + following,
                    following_base + index,
                )
            )

    if capped:
        triangles = triangulate_polygon(points)
        last = (len(stations) - 1) * count
        faces.extend((c, b, a) for a, b, c in triangles)
        faces.extend((a + last, b + last, c + last) for a, b, c in triangles)
    return Mesh(vertices, faces)


def plane(
    size_x: float = 1.0,
    size_y: float = 1.0,
    nx: int = 1,
    ny: int = 1,
) -> Mesh:
    """Subdivided rectangle in the local XY plane, normal along +Z."""
    size_x = float(size_x)
    size_y = float(size_y)
    nx = max(1, int(nx))
    ny = max(1, int(ny))

    vertices: List[Vector] = []
    for j in range(ny + 1):
        y = -0.5 * size_y + size_y * j / ny
        for i in range(nx + 1):
            x = -0.5 * size_x + size_x * i / nx
            vertices.append((x, y, 0.0))

    faces: List[Face] = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            base = j * stride + i
            faces.append((base, base + 1, base + stride + 1, base + stride))
    return Mesh(vertices, faces)


def arrow(
    start: Point3D,
    end: Point3D,
    shaft_radius: Optional[float] = None,
    head_radius: Optional[float] = None,
    head_length: Optional[float] = None,
    segments: int = 16,
) -> Mesh:
    """Cylindrical shaft with a conical head pointing from ``start`` to ``end``."""
    tail = as_point(start)
    tip = as_point(end)
    direction = tip - tail
    length = direction.length()
    if length <= _EPS:
        return Mesh([], [])

    shaft_radius = 0.02 * length if shaft_radius is None else abs(float(shaft_radius))
    head_radius = 2.6 * shaft_radius if head_radius is None else abs(float(head_radius))
    head_length = min(
        0.9 * length,
        3.0 * head_radius if head_length is None else abs(float(head_length)),
    )
    shaft_length = max(_EPS, length - head_length)
    segments = max(3, int(segments))

    shaft = frustum(shaft_radius, shaft_radius, shaft_length, segments, 1, True)
    shaft = shaft.placed(origin=Point3D(0.0, 0.0, 0.5 * shaft_length))
    head = frustum(head_radius, 0.0, head_length, segments, 1, True)
    head = head.placed(origin=Point3D(0.0, 0.0, shaft_length + 0.5 * head_length))
    return shaft.merged(head).placed(origin=tail, axis=direction)


def grid_lines(
    size_x: float = 10.0,
    size_y: float = 10.0,
    step: float = 1.0,
    z: float = 0.0,
    centre: Optional[Point3D] = None,
) -> List[Tuple[Point3D, Point3D]]:
    """Ground-grid segments in a plane of constant Z."""
    size_x = abs(float(size_x))
    size_y = abs(float(size_y))
    step = max(_EPS, abs(float(step)))
    origin = as_point(centre) if centre is not None else Point3D(0.0, 0.0, 0.0)
    z = float(z) + origin.z

    nx = max(1, int(round(size_x / step)))
    ny = max(1, int(round(size_y / step)))
    x0 = origin.x - 0.5 * size_x
    y0 = origin.y - 0.5 * size_y

    segments: List[Tuple[Point3D, Point3D]] = []
    for index in range(nx + 1):
        x = x0 + index * size_x / nx
        segments.append((Point3D(x, y0, z), Point3D(x, y0 + size_y, z)))
    for index in range(ny + 1):
        y = y0 + index * size_y / ny
        segments.append((Point3D(x0, y, z), Point3D(x0 + size_x, y, z)))
    return segments


# ----------------------------------------------------------------------
# Structural profiles
# ----------------------------------------------------------------------


#: Cross-section codes accepted by :func:`profile_section`.
PROFILE_KINDS = ("FB", "T", "I", "L", "C", "BOX")


def profile_section(
    kind: str = "T",
    web_height: float = 0.2,
    web_thickness: float = 0.01,
    flange_width: float = 0.1,
    flange_thickness: float = 0.015,
) -> List[Tuple[float, float]]:
    """
    Return a closed 2D cross-section outline for a structural profile.

    The section lies in the XY plane with the web along +Y and the web's
    mid-plane on X = 0, so extruding it along Z builds a beam whose local
    "up" direction is +Y.  Supported kinds: ``FB`` (flat bar), ``T``, ``I``,
    ``L`` (angle), ``C`` (channel) and ``BOX``.
    """
    code = str(kind).strip().upper().replace("-", "").replace("_", "")
    if code in ("FLATBAR", "FLAT", "BAR"):
        code = "FB"
    if code in ("LBULB", "BULB", "HP"):
        # A bulb flat is drawn as an angle: close enough at viewport scale.
        code = "L"
    if code in ("CHANNEL", "U"):
        code = "C"
    if code in ("TUBE", "RHS", "HOLLOW"):
        code = "BOX"

    hw = max(_EPS, abs(float(web_height)))
    tw = max(_EPS, abs(float(web_thickness)))
    bf = max(0.0, abs(float(flange_width)))
    tf = max(0.0, abs(float(flange_thickness)))

    if code == "FB" or bf <= _EPS or tf <= _EPS:
        half = 0.5 * tw
        return [(-half, 0.0), (half, 0.0), (half, hw), (-half, hw)]

    half_web = 0.5 * tw
    half_flange = 0.5 * bf

    if code == "T":
        top = hw
        return [
            (-half_web, 0.0), (half_web, 0.0),
            (half_web, top - tf), (half_flange, top - tf),
            (half_flange, top), (-half_flange, top),
            (-half_flange, top - tf), (-half_web, top - tf),
        ]

    if code == "I":
        return [
            (-half_flange, 0.0), (half_flange, 0.0),
            (half_flange, tf), (half_web, tf),
            (half_web, hw - tf), (half_flange, hw - tf),
            (half_flange, hw), (-half_flange, hw),
            (-half_flange, hw - tf), (-half_web, hw - tf),
            (-half_web, tf), (-half_flange, tf),
        ]

    if code == "L":
        return [
            (-half_web, 0.0), (half_web, 0.0),
            (half_web, hw - tf), (bf - half_web, hw - tf),
            (bf - half_web, hw), (-half_web, hw),
        ]

    if code == "C":
        return [
            (-half_web, 0.0), (bf - half_web, 0.0),
            (bf - half_web, tf), (half_web, tf),
            (half_web, hw - tf), (bf - half_web, hw - tf),
            (bf - half_web, hw), (-half_web, hw),
        ]

    # BOX outlines the rectangular hollow section; the hole itself is added by
    # box_section_profiles(), which beam() uses to build the closed shell.
    return [
        (-half_flange, 0.0), (half_flange, 0.0),
        (half_flange, hw), (-half_flange, hw),
    ]


def box_section_profiles(
    web_height: float = 0.2,
    flange_width: float = 0.1,
    thickness: float = 0.015,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Outer and inner outlines of a rectangular hollow section."""
    hw = max(_EPS, abs(float(web_height)))
    bf = max(_EPS, abs(float(flange_width)))
    half = 0.5 * bf
    wall = max(_EPS, min(abs(float(thickness)), min(half, 0.5 * hw) - _EPS))
    outer = [(-half, 0.0), (half, 0.0), (half, hw), (-half, hw)]
    inner = [
        (-half + wall, wall),
        (half - wall, wall),
        (half - wall, hw - wall),
        (-half + wall, hw - wall),
    ]
    return outer, inner


def beam(
    start: Point3D,
    end: Point3D,
    kind: str = "T",
    web_height: float = 0.2,
    web_thickness: float = 0.01,
    flange_width: float = 0.1,
    flange_thickness: float = 0.015,
    up: Optional[Point3D] = None,
    capped: bool = True,
) -> Mesh:
    """Extrude a structural profile between two points, web along ``up``."""
    tail = as_point(start)
    tip = as_point(end)
    axis = tip - tail
    if axis.length() <= _EPS:
        return Mesh([], [])

    code = str(kind).strip().upper().replace("-", "").replace("_", "")
    if code in ("BOX", "RHS", "HOLLOW", "TUBE"):
        outer, inner = box_section_profiles(
            web_height=web_height,
            flange_width=flange_width,
            thickness=flange_thickness,
        )
        # hollow_prism builds around the local origin, so shift the section to
        # the start station before orienting it along the beam axis.
        mesh = hollow_prism(outer, inner, axis.length(), capped)
        mesh = mesh.placed(origin=Point3D(0.0, 0.0, 0.5 * axis.length()))
        return mesh.placed(origin=tail, axis=axis, up_hint=up)

    section = profile_section(
        code,
        web_height=web_height,
        web_thickness=web_thickness,
        flange_width=flange_width,
        flange_thickness=flange_thickness,
    )
    return extrusion(section, (tail, tip), capped=capped, up_hint=up)
