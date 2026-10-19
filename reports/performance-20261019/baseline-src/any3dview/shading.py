"""
Backend-neutral flat-shaded directional lighting.

A Tk canvas can only fill a polygon with one solid colour, so lighting is
evaluated once per face: ambient + Lambert diffuse + a Blinn-Phong specular
highlight.  The resulting intensity is quantised to a small number of steps
and the shaded colour strings are cached per base colour, which turns
per-frame shading into a list index instead of string formatting.

With the default world-fixed light the shaded colours are computed once when
the scene is compiled and reused for every frame; only ``follow_camera``
lights pay a per-frame cost.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import Point3D, _rgb_to_hex, as_point, parse_color


#: Number of quantisation steps in a shade table.  The count is chosen so
#: that a shade of exactly 1.0 lands on a level boundary, which keeps a fully
#: lit face at its literal base colour - important when the colour encodes a
#: value such as plate thickness or a utilisation factor.
SHADE_LEVELS = 61

#: Shade values above 1.0 blend the base colour toward white, which is what
#: gives specular highlights somewhere to go.
MAX_SHADE = 1.5

_TABLE_CACHE: Dict[str, Optional[Tuple[str, ...]]] = {}


def shade_table(base_color: str) -> Optional[Tuple[str, ...]]:
    """
    Return ``SHADE_LEVELS`` colour strings ramping ``base_color`` from black
    through its own value and on to white.

    Returns ``None`` when the colour cannot be parsed locally, in which case
    the caller should draw with the unshaded colour.
    """
    if base_color in _TABLE_CACHE:
        return _TABLE_CACHE[base_color]

    rgb = parse_color(base_color)
    if rgb is None:
        _TABLE_CACHE[base_color] = None
        return None

    red, green, blue = rgb
    table: List[str] = []
    for level in range(SHADE_LEVELS):
        shade = MAX_SHADE * level / (SHADE_LEVELS - 1)
        if shade <= 1.0:
            table.append(_rgb_to_hex(red * shade, green * shade, blue * shade))
        else:
            highlight = (shade - 1.0) / (MAX_SHADE - 1.0)
            table.append(
                _rgb_to_hex(
                    red + (255.0 - red) * highlight,
                    green + (255.0 - green) * highlight,
                    blue + (255.0 - blue) * highlight,
                )
            )
    result = tuple(table)
    _TABLE_CACHE[base_color] = result
    return result


def shade_color(base_color: str, shade: float) -> str:
    """Shade a single colour; used for outlines and one-off decorations."""
    table = shade_table(base_color)
    if table is None:
        return base_color
    return table[shade_level(shade)]


def shade_level(shade: float) -> int:
    index = int(shade / MAX_SHADE * (SHADE_LEVELS - 1) + 0.5)
    return 0 if index < 0 else (SHADE_LEVELS - 1 if index >= SHADE_LEVELS else index)


def shade_levels(shade: np.ndarray) -> np.ndarray:
    """Vectorised :func:`shade_level`."""
    scaled = shade * ((SHADE_LEVELS - 1) / MAX_SHADE) + 0.5
    return np.clip(scaled, 0, SHADE_LEVELS - 1).astype(np.int32)


class Light:
    """A single directional light plus the ambient term."""

    __slots__ = (
        "direction",
        "ambient",
        "diffuse",
        "specular",
        "shininess",
        "follow_camera",
        "enabled",
    )

    def __init__(
        self,
        direction: Optional[Point3D] = None,
        ambient: float = 0.45,
        diffuse: float = 0.55,
        specular: float = 0.12,
        shininess: float = 24.0,
        follow_camera: bool = False,
        enabled: bool = True,
    ) -> None:
        # ambient + diffuse deliberately sums to 1.0: a face turned toward the
        # light renders in its exact base colour and every other face only
        # darkens, so colour-coded scenes stay readable against their legend.
        # Default: a sun above, in front of and to the right of the default
        # iso view, so surfaces facing the viewer stay bright.
        self.direction = (
            as_point(direction) if direction is not None else Point3D(0.35, -0.55, 0.76)
        ).normalized()
        if self.direction.length() <= 0.0:
            self.direction = Point3D(0.0, 0.0, 1.0)
        self.ambient = float(ambient)
        self.diffuse = float(diffuse)
        self.specular = float(specular)
        self.shininess = max(1.0, float(shininess))
        self.follow_camera = bool(follow_camera)
        self.enabled = bool(enabled)

    def __repr__(self) -> str:
        return (
            f"Light(direction={self.direction!r}, ambient={self.ambient:g}, "
            f"diffuse={self.diffuse:g}, specular={self.specular:g}, "
            f"shininess={self.shininess:g}, follow_camera={self.follow_camera}, "
            f"enabled={self.enabled})"
        )

    def key(self) -> Tuple[float, ...]:
        """Identity of the light, used to invalidate cached shaded colours."""
        return (
            self.direction.x,
            self.direction.y,
            self.direction.z,
            self.ambient,
            self.diffuse,
            self.specular,
            self.shininess,
            float(self.follow_camera),
            float(self.enabled),
        )

    def world_direction(
        self,
        camera_basis: Optional[Sequence[Point3D]] = None,
    ) -> np.ndarray:
        """
        Direction pointing from the surface toward the light.

        With ``follow_camera`` the stored direction is interpreted in camera
        space (x right, y up, z toward the viewer), so the light rides along
        with the orbit like a head lamp.
        """
        vector = self.direction
        if self.follow_camera and camera_basis is not None:
            right, up, forward = camera_basis
            vector = (
                right * vector.x + up * vector.y + forward * (-vector.z)
            ).normalized()
        return np.array([vector.x, vector.y, vector.z], dtype=np.float32)


def face_shade(
    normals: np.ndarray,
    light: Light,
    light_direction: np.ndarray,
    view_direction: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Flat-shade intensities for unit face ``normals``.

    ``view_direction`` points from the surface toward the camera; passing the
    camera's own axis (rather than a per-face vector) is accurate enough for
    flat shading and keeps this a single matrix-vector product.
    """
    if not light.enabled or len(normals) == 0:
        return np.ones(len(normals), dtype=np.float32)

    lambert = normals @ light_direction
    np.clip(lambert, 0.0, 1.0, out=lambert)
    shade = light.ambient + light.diffuse * lambert

    if light.specular > 0.0 and view_direction is not None:
        half = light_direction + view_direction
        norm = float(np.linalg.norm(half))
        if norm > 1.0e-9:
            half = half / norm
            highlight = normals @ half
            np.clip(highlight, 0.0, 1.0, out=highlight)
            # Only lit faces may glint; otherwise back faces pick up highlights.
            shade += light.specular * np.power(highlight, light.shininess) * (lambert > 0.0)

    return shade.astype(np.float32, copy=False)


def shaded_color_list(
    base_colors: Sequence[str],
    shade: np.ndarray,
) -> List[str]:
    """Map per-face base colours and shade values to Tk colour strings."""
    levels = shade_levels(shade).tolist()
    result: List[str] = []
    for color, level in zip(base_colors, levels):
        table = shade_table(color)
        result.append(color if table is None else table[level])
    return result


def sun_direction(azimuth_degrees: float, elevation_degrees: float) -> Point3D:
    """Convenience helper: a light direction from compass-style angles."""
    azimuth = math.radians(float(azimuth_degrees))
    elevation = math.radians(float(elevation_degrees))
    cosine = math.cos(elevation)
    return Point3D(
        cosine * math.cos(azimuth),
        cosine * math.sin(azimuth),
        math.sin(elevation),
    ).normalized()
