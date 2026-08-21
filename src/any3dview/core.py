"""
Vector math, orbit camera and colour helpers shared by 3D view backends.

The module deliberately contains no GUI-toolkit imports.  Backend packages
such as :mod:`anytk3d` re-export these values for compatibility.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple


_EPS = 1.0e-12

# Blue -> cyan -> green -> yellow -> orange -> red.  The interpolation is
# implemented locally so the canvas keeps its zero-dependency design.
DEFAULT_COLOR_STOPS: Tuple[Tuple[float, str], ...] = (
    (0.00, "#313695"),
    (0.18, "#4575b4"),
    (0.36, "#74add1"),
    (0.52, "#abd9e9"),
    (0.66, "#e0f3f8"),
    (0.78, "#fee090"),
    (0.88, "#fdae61"),
    (0.95, "#f46d43"),
    (1.00, "#a50026"),
)

#: Historic name for the default scale.  Assigning to it does *not* change
#: the colours any more - use :func:`set_color_stops`.
_THICKNESS_COLOR_STOPS: Tuple[Tuple[float, str], ...] = DEFAULT_COLOR_STOPS

_active_color_stops: Tuple[Tuple[float, str], ...] = DEFAULT_COLOR_STOPS

#: Bumped whenever the scale changes, so a canvas can notice that colours it
#: resolved earlier are stale and rebuild its scene.
_color_stop_generation = 0


def set_color_stops(stops: Any) -> Tuple[Tuple[float, str], ...]:
    """
    Replace the colour scale used by :func:`_interpolate_thickness_color`.

    ``stops`` is an iterable of ``(position, colour)`` pairs with positions in
    ``[0, 1]``; they are sorted and validated here.  This is how an
    application swaps in a different colour map - a viridis or turbo ramp
    sampled at a handful of positions, say - without needing to know how the
    interpolation is implemented.

    Colours a canvas has already resolved into its compiled scene are
    refreshed on the next redraw; colours the caller resolved itself need
    rebuilding by the caller.
    """
    global _active_color_stops, _color_stop_generation

    cleaned: List[Tuple[float, str]] = []
    for entry in stops:
        position, color = entry
        position = float(position)
        if not math.isfinite(position):
            raise ValueError(f"colour stop position must be finite, got {position!r}")
        color = str(color)
        if parse_color(color) is None:
            raise ValueError(f"colour stop {color!r} is not a colour this module can read")
        cleaned.append((min(1.0, max(0.0, position)), color))

    if len(cleaned) < 2:
        raise ValueError("a colour scale needs at least two stops")

    cleaned.sort(key=lambda item: item[0])
    _active_color_stops = tuple(cleaned)
    _color_stop_generation += 1
    return _active_color_stops


def get_color_stops() -> Tuple[Tuple[float, str], ...]:
    """The colour scale currently in use."""
    return _active_color_stops


def reset_color_stops() -> Tuple[Tuple[float, str], ...]:
    """Restore the built-in blue-to-red scale."""
    return set_color_stops(DEFAULT_COLOR_STOPS)


def color_stop_generation() -> int:
    """Counter that changes whenever the colour scale is replaced."""
    return _color_stop_generation

# Tk colour names that appear in the package defaults and in ANYstructure.
# Resolving them locally avoids a round trip through the Tk interpreter when
# the shading model needs the RGB components of a fill colour.
_NAMED_COLORS = {
    "aliceblue": "#f0f8ff",
    "black": "#000000",
    "blue": "#0000ff",
    "brown": "#a52a2a",
    "cyan": "#00ffff",
    "darkgray": "#a9a9a9",
    "darkgrey": "#a9a9a9",
    "dimgray": "#696969",
    "dimgrey": "#696969",
    "gainsboro": "#dcdcdc",
    "gold": "#ffd700",
    "gray": "#bebebe",
    "green": "#00ff00",
    "grey": "#bebebe",
    "khaki": "#f0e68c",
    "lightblue": "#add8e6",
    "lightgray": "#d3d3d3",
    "lightgrey": "#d3d3d3",
    "lightsteelblue": "#b0c4de",
    "magenta": "#ff00ff",
    "maroon": "#b03060",
    "navy": "#000080",
    "olive": "#808000",
    "orange": "#ffa500",
    "peru": "#cd853f",
    "pink": "#ffc0cb",
    "purple": "#a020f0",
    "red": "#ff0000",
    "salmon": "#fa8072",
    "silver": "#c0c0c0",
    "skyblue": "#87ceeb",
    "slategray": "#708090",
    "slategrey": "#708090",
    "steelblue": "#4682b4",
    "tan": "#d2b48c",
    "teal": "#008080",
    "tomato": "#ff6347",
    "violet": "#ee82ee",
    "wheat": "#f5deb3",
    "white": "#ffffff",
    "yellow": "#ffff00",
}


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB colour, got {color!r}")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(red: float, green: float, blue: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, round(red))),
        max(0, min(255, round(green))),
        max(0, min(255, round(blue))),
    )


def parse_color(color: str) -> Optional[Tuple[int, int, int]]:
    """Return the RGB components of a colour, or ``None`` when unknown.

    Accepts ``#rgb``, ``#rrggbb``, ``#rrrrggggbbbb`` and the Tk colour names
    used across this package.  Unknown names return ``None`` so callers can
    fall back to using the colour string verbatim.
    """
    if not color:
        return None
    text = color.strip()
    if text.startswith("#"):
        digits = text[1:]
        length = len(digits)
        try:
            if length == 3:
                return tuple(int(digit * 2, 16) for digit in digits)  # type: ignore[return-value]
            if length == 6:
                return _hex_to_rgb(text)
            if length == 12:
                return tuple(  # type: ignore[return-value]
                    int(digits[index:index + 4], 16) >> 8
                    for index in (0, 4, 8)
                )
        except ValueError:
            return None
        return None
    return _hex_to_rgb(_NAMED_COLORS[text.lower()]) if text.lower() in _NAMED_COLORS else None


def _interpolate_thickness_color(
    value: float,
    minimum: float,
    maximum: float,
) -> str:
    """Map a value onto the active colour scale (see :func:`set_color_stops`)."""
    if maximum <= minimum + _EPS:
        position = 0.5
    else:
        position = (float(value) - minimum) / (maximum - minimum)
    position = max(0.0, min(1.0, position))

    stops = _active_color_stops
    for index in range(len(stops) - 1):
        start_position, start_color = stops[index]
        end_position, end_color = stops[index + 1]
        if position <= end_position:
            span = max(_EPS, end_position - start_position)
            fraction = (position - start_position) / span
            start_rgb = parse_color(start_color) or (0, 0, 0)
            end_rgb = parse_color(end_color) or (0, 0, 0)
            return _rgb_to_hex(
                start_rgb[0] + fraction * (end_rgb[0] - start_rgb[0]),
                start_rgb[1] + fraction * (end_rgb[1] - start_rgb[1]),
                start_rgb[2] + fraction * (end_rgb[2] - start_rgb[2]),
            )

    return stops[-1][1]


def _flatten_numeric_values(value: Any) -> List[float]:
    """Extract finite numeric values from scalar or nested list/tuple input."""
    if value is None or callable(value) or isinstance(value, (str, bytes)):
        return []
    if isinstance(value, (int, float)):
        number = float(value)
        return [number] if math.isfinite(number) else []
    try:
        items = list(value)
    except TypeError:
        return []

    result: List[float] = []
    for item in items:
        result.extend(_flatten_numeric_values(item))
    return result


class Point3D:
    """A lightweight three-dimensional vector/point."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self) -> str:
        return f"Point3D({self.x:g}, {self.y:g}, {self.z:g})"

    def to_tuple(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __add__(self, other: "Point3D") -> "Point3D":
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Point3D") -> "Point3D":
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> "Point3D":
        return Point3D(-self.x, -self.y, -self.z)

    def __mul__(self, scalar: float) -> "Point3D":
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Point3D":
        return self * scalar

    def __truediv__(self, scalar: float) -> "Point3D":
        if abs(scalar) <= _EPS:
            raise ZeroDivisionError("Cannot divide Point3D by zero")
        return Point3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Point3D":
        magnitude = self.length()
        if magnitude <= _EPS:
            return Point3D(0.0, 0.0, 0.0)
        return self / magnitude

    def dot(self, other: "Point3D") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Point3D") -> "Point3D":
        return Point3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def rotate_x(self, angle: float) -> "Point3D":
        cosine = math.cos(angle)
        sine = math.sin(angle)
        return Point3D(
            self.x,
            self.y * cosine - self.z * sine,
            self.y * sine + self.z * cosine,
        )

    def rotate_y(self, angle: float) -> "Point3D":
        cosine = math.cos(angle)
        sine = math.sin(angle)
        return Point3D(
            self.x * cosine + self.z * sine,
            self.y,
            -self.x * sine + self.z * cosine,
        )

    def rotate_z(self, angle: float) -> "Point3D":
        cosine = math.cos(angle)
        sine = math.sin(angle)
        return Point3D(
            self.x * cosine - self.y * sine,
            self.x * sine + self.y * cosine,
            self.z,
        )


def as_point(value: Any) -> Point3D:
    """Coerce ``Point3D``, sequences and numpy rows into a ``Point3D``."""
    if isinstance(value, Point3D):
        return value
    if value is None:
        return Point3D(0.0, 0.0, 0.0)
    x, y, z = value
    return Point3D(float(x), float(y), float(z))


class Camera3D:
    """Orbit camera looking at a target point."""

    def __init__(self) -> None:
        self.target = Point3D(0.0, 0.0, 0.0)
        self.world_up = Point3D(0.0, 0.0, 1.0)

        self.fov = math.radians(45.0)
        self.near = 0.01
        self.far = 10000.0

        self.azimuth = math.radians(-45.0)
        self.elevation = math.radians(25.0)
        self.distance = 10.0

        self.position = Point3D(0.0, 0.0, 0.0)
        self._update_position()

    def _update_position(self) -> None:
        cosine_elevation = math.cos(self.elevation)
        offset = Point3D(
            self.distance * cosine_elevation * math.cos(self.azimuth),
            self.distance * cosine_elevation * math.sin(self.azimuth),
            self.distance * math.sin(self.elevation),
        )
        self.position = self.target + offset

    def set_orbit(
        self,
        azimuth: Optional[float] = None,
        elevation: Optional[float] = None,
        distance: Optional[float] = None,
    ) -> None:
        if azimuth is not None:
            self.azimuth = float(azimuth)
        if elevation is not None:
            limit = math.radians(89.5)
            self.elevation = max(-limit, min(limit, float(elevation)))
        if distance is not None:
            self.distance = max(float(distance), self.near * 2.0)
        self._update_position()

    def orbit(
        self,
        delta_azimuth: float = 0.0,
        delta_elevation: float = 0.0,
        delta_distance: float = 0.0,
    ) -> None:
        self.set_orbit(
            azimuth=self.azimuth + delta_azimuth,
            elevation=self.elevation + delta_elevation,
            distance=self.distance + delta_distance,
        )

    def zoom(self, factor: float) -> None:
        if factor > 0.0:
            self.set_orbit(distance=max(self.near * 2.0, self.distance * factor))

    def pan_view_pixels(self, delta_x: float, delta_y: float, width: int, height: int) -> None:
        width = max(1, int(width))
        height = max(1, int(height))
        right, camera_up, _forward = self.basis()
        visible_height = 2.0 * self.distance * math.tan(self.fov / 2.0)
        visible_width = visible_height * float(width) / float(height)
        world_dx = -float(delta_x) * visible_width / float(width)
        world_dy = float(delta_y) * visible_height / float(height)
        offset = right * world_dx + camera_up * world_dy
        self.target = self.target + offset
        self._update_position()

    def set_target(self, target: Point3D) -> None:
        self.target = Point3D(target.x, target.y, target.z)
        self._update_position()

    def set_position(self, position: Point3D) -> None:
        offset = position - self.target
        distance = max(offset.length(), self.near * 2.0)
        self.distance = distance
        self.azimuth = math.atan2(offset.y, offset.x)
        self.elevation = math.asin(max(-1.0, min(1.0, offset.z / distance)))
        self._update_position()

    def basis(self) -> Tuple[Point3D, Point3D, Point3D]:
        """Return camera right, camera up and camera forward vectors."""
        forward = (self.target - self.position).normalized()
        right = forward.cross(self.world_up)
        if right.length() <= _EPS:
            right = forward.cross(Point3D(0.0, 1.0, 0.0))
        right = right.normalized()
        camera_up = right.cross(forward).normalized()
        return right, camera_up, forward

    def world_to_camera(self, point: Point3D) -> Tuple[float, float, float]:
        right, camera_up, forward = self.basis()
        relative = point - self.position
        return relative.dot(right), relative.dot(camera_up), -relative.dot(forward)

    def project_point(
        self,
        point: Point3D,
        width: int,
        height: int,
    ) -> Optional[Tuple[float, float]]:
        """Compatibility projection method; frame rendering uses a faster path."""
        width = max(1, int(width))
        height = max(1, int(height))
        camera_x, camera_y, camera_z = self.world_to_camera(point)
        depth = -camera_z
        if depth <= self.near or depth >= self.far:
            return None
        scale = 1.0 / math.tan(self.fov / 2.0)
        aspect = width / height
        return (
            (camera_x * scale / aspect / depth + 1.0) * 0.5 * width,
            (1.0 - camera_y * scale / depth) * 0.5 * height,
        )

    def screen_ray(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
    ) -> Tuple[Point3D, Point3D]:
        """World-space ray from the camera through one viewport pixel."""

        width = max(1, int(width))
        height = max(1, int(height))
        ndc_x = 2.0 * float(x) / float(width) - 1.0
        ndc_y = 1.0 - 2.0 * float(y) / float(height)
        tangent = math.tan(self.fov / 2.0)
        aspect = float(width) / float(height)
        right, camera_up, forward = self.basis()
        direction = (
            forward
            + right * (ndc_x * aspect * tangent)
            + camera_up * (ndc_y * tangent)
        ).normalized()
        return Point3D(self.position.x, self.position.y, self.position.z), direction

    def unproject_to_plane(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        plane_point: Point3D,
        plane_normal: Point3D,
    ) -> Optional[Point3D]:
        """Intersect a screen ray with a world-space plane.

        ``None`` means the ray is parallel to the plane or the intersection is
        behind the camera.
        """

        origin, direction = self.screen_ray(x, y, width, height)
        normal = as_point(plane_normal).normalized()
        if normal.length() <= _EPS:
            raise ValueError("plane normal must be non-zero")
        point = as_point(plane_point)
        denominator = direction.dot(normal)
        if abs(denominator) <= _EPS:
            return None
        distance = (point - origin).dot(normal) / denominator
        if distance < 0.0:
            return None
        return origin + direction * distance
