"""Host-independent rendering quality settings (no Tk or Qt imports)."""

from __future__ import annotations

import os


DEFAULT_MSAA_SAMPLES = 4


def msaa_samples() -> int:
    """Return the requested multisample count; ``0`` disables antialiasing.

    ``ANY3DVIEW_MSAA`` accepts ``0``/``off`` to disable, ``1``/``on`` for the
    default of four samples, or an explicit count (``2``, ``4``, ``8``,
    ``16``).  Anything unrecognised keeps the default rather than silently
    degrading image quality.
    """

    requested = os.environ.get("ANY3DVIEW_MSAA", "").strip().casefold()
    if requested in {"0", "off", "false", "no", "none"}:
        return 0
    if requested.isdigit():
        count = int(requested)
        if count <= 1:
            return DEFAULT_MSAA_SAMPLES if count == 1 else 0
        return min(16, count)
    return DEFAULT_MSAA_SAMPLES
