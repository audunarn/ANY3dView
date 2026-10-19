"""Conservative reuse of unchanged scene submissions through public viewer APIs.

Coordinates remain mutable at the scene boundary, so identity alone is unsafe.
Hash all displayed payloads rather than retaining another copy of their arrays.
An invisible, empty mesh handle witnesses the lifetime of the canvas contents:
the viewer removes it on clear, including clears performed outside this adapter.
"""
from dataclasses import fields, is_dataclass
import hashlib
import struct

import numpy as np


def _value(digest, value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays cannot identify retained scene contents")
        digest.update(b"array")
        _value(digest, (value.dtype.str, value.shape))
        digest.update(np.ascontiguousarray(value))
    elif is_dataclass(value):
        _value(digest, (type(value).__module__, type(value).__qualname__))
        for field in fields(value):
            _value(digest, (field.name, getattr(value, field.name)))
    elif isinstance(value, dict):
        digest.update(b"dict[")
        for key, item in value.items():
            _value(digest, (key, item))
        digest.update(b"]")
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence[" + str(len(value)).encode() + b":")
        for item in value:
            _value(digest, item)
        digest.update(b"]")
    elif isinstance(value, (float, np.floating)):
        digest.update(b"float:" + struct.pack("!d", float(value)))
    elif isinstance(value, (str, bytes, int, np.integer, bool, type(None))):
        encoded = repr(value).encode()
        digest.update(type(value).__name__.encode() + b":" + str(len(encoded)).encode() + b":" + encoded)
    else:
        raise TypeError(f"unsupported scene identity: {type(value).__name__}")


def scene_key(viewport, scene):
    # Construction callbacks and formatters may change independently of scene
    # data. Rebuild those overlays normally. GPU result fields have their own
    # retained scalar/appearance update contract and must not enter this cache.
    if (scene.result_field is not None or viewport.construction_active
            or getattr(viewport, "plate_draft", None) is not None
            or not hasattr(viewport.canvas, "preselected_key")
            or not callable(getattr(viewport.canvas, "add_mesh_arrays", None))):
        return None
    digest = hashlib.sha256()
    try:
        _value(digest, (scene.faces, scene.lines, scene.points, scene.arrows,
                       scene.spheres, scene.legend, viewport.visualization,
                       viewport._commercial_selection))
    except TypeError:
        # Third-party payloads remain drawable through the uncached route.
        return None
    return digest.digest()


def lifetime_guard(canvas):
    from any3dview import MeshArrays

    handle = canvas.add_mesh_arrays(
        MeshArrays(np.empty((0, 3)), np.empty((0, 3), dtype=np.uint32)),
        pickable=False,
    )
    if handle is None or not callable(getattr(handle, "set_visible", None)):
        return None
    handle.set_visible(False)
    return handle
