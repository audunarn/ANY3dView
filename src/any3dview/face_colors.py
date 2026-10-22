"""Batch normalisation of per-face colours for in-place colour updates.

``normalize_face_colors`` turns the per-face colour specifications accepted
by :meth:`MeshHandle.update_face_colors` into the two forms the backends use:
the canonical string tuple stored in appearance state (software painter,
rebuilds, snapshots) and an ``(N, 3)`` float32 RGB array for the GPU colour
texture.  The RGB array is bit-identical to the historical per-face parse
(``parse_color(c) or (154, 167, 180)`` divided by 255), and ``#rrggbb``
strings and numeric arrays avoid per-face Python parsing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import parse_color
from .errors import FaceColorUpdateError

# Fallback used by the GPU renderer for colours that parse_color cannot read.
_UNKNOWN_RGB = (154, 167, 180)

_NIBBLE = np.full(128, -1, dtype=np.int16)
for _index, _digit in enumerate("0123456789abcdef"):
    _NIBBLE[ord(_digit)] = _index
    _NIBBLE[ord(_digit.upper())] = _index
_HEX_CODES = np.frombuffer(b"0123456789abcdef", dtype=np.uint8).astype(np.uint32)


def _rgb_float(rgb8: np.ndarray) -> np.ndarray:
    # channel / 255.0 in double precision, then float32, as the per-face parse.
    return (rgb8.astype(np.float64) / 255.0).astype(np.float32)


def _hex_strings(rgb8: np.ndarray) -> tuple[str, ...]:
    codes = np.empty((len(rgb8), 7), dtype=np.uint32)
    codes[:, 0] = ord("#")
    codes[:, 1::2] = _HEX_CODES[rgb8 >> 4]
    codes[:, 2::2] = _HEX_CODES[rgb8 & 15]
    return tuple(codes.view("<U7").ravel().tolist())


def _parse_strings(strings: tuple[str, ...]) -> np.ndarray:
    """Return uint8 RGB for colour strings, vectorised for ``#rrggbb``."""

    count = len(strings)
    array = np.asarray(strings)
    if array.dtype.kind == "U" and array.dtype.itemsize == 28 and count:
        codes = array.view(np.uint32).reshape(count, 7)
        if bool((codes[:, 0] == ord("#")).all()) and bool((codes[:, 1:] < 128).all()):
            nibbles = _NIBBLE[codes[:, 1:]]
            if bool((nibbles >= 0).all()):
                return (nibbles[:, 0::2] * 16 + nibbles[:, 1::2]).astype(np.uint8)
    return np.asarray(
        [parse_color(color) or _UNKNOWN_RGB for color in strings], dtype=np.uint8
    ).reshape(count, 3)


def _numeric_rgb8(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype == np.bool_ or array.dtype.kind not in "iuf":
        raise TypeError("numeric face colours must be an integer or float array")
    if array.ndim != 2 or array.shape[1] not in (3, 4):
        raise FaceColorUpdateError("numeric face colours must have shape (N, 3) or (N, 4)")
    if array.dtype.kind == "f":
        if not np.isfinite(array).all() or array.min(initial=0.0) < 0.0 or array.max(initial=0.0) > 1.0:
            raise FaceColorUpdateError("float face colours must be finite and within [0, 1]")
        array = np.rint(array * 255.0)
    elif array.min(initial=0) < 0 or array.max(initial=0) > 255:
        raise FaceColorUpdateError("integer face colours must be within [0, 255]")
    if array.shape[1] == 4:
        if not bool((array[:, 3] == 255).all()):
            raise FaceColorUpdateError(
                "per-face alpha is not supported; use the mesh opacity instead"
            )
    return np.ascontiguousarray(array[:, :3], dtype=np.uint8)


def normalize_face_colors(values: Any) -> tuple[tuple[str, ...], np.ndarray]:
    """Return ``(strings, rgb_float32)`` for a per-face colour specification.

    Accepts a sequence of colour strings (as ``add_mesh_arrays`` does), an
    ``(N, 3|4)`` integer array in ``[0, 255]``, an ``(N, 3|4)`` float array
    in ``[0, 1]`` (quantised to 8 bits like a hex string), or a sequence of
    RGB(A) tuples.  Alpha must be opaque.  Length checks belong to the caller.
    """

    if isinstance(values, np.ndarray) and values.dtype.kind in "biuf":
        rgb8 = _numeric_rgb8(values)
        return _hex_strings(rgb8), _rgb_float(rgb8)
    if isinstance(values, (str, bytes)):
        raise TypeError("face colours must be a sequence with one colour per face")
    items = tuple(values)
    if items and not isinstance(items[0], str):
        if any(isinstance(item, str) for item in items):
            raise TypeError("face colours must be all strings or all numeric")
        rgb8 = _numeric_rgb8(items)
        return _hex_strings(rgb8), _rgb_float(rgb8)
    if not all(isinstance(item, str) for item in items):
        raise TypeError("face colours must be all strings or all numeric")
    strings = tuple(str(item) for item in items)
    return strings, _rgb_float(_parse_strings(strings))
