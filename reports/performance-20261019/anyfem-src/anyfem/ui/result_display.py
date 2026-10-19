"""Pure result-display unit conversions used by the Tk results workspace."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from ..post.history import Series

SI_DISPLAY = "SI (m / Pa)"
ENGINEERING_DISPLAY = "Engineering (mm / MPa)"
DISPLAY_UNIT_SYSTEMS = (SI_DISPLAY, ENGINEERING_DISPLAY)


def unit_transform(unit: str, profile: str) -> tuple[float, str]:
    """Return multiplier and label without changing stored SI result data."""

    if profile != ENGINEERING_DISPLAY:
        return 1.0, str(unit)
    normalized = str(unit).strip()
    if normalized == "m":
        return 1000.0, "mm"
    if normalized == "Pa":
        return 1.0e-6, "MPa"
    return 1.0, normalized


def converted_series(series: Iterable[Series], profile: str) -> list[Series]:
    converted: list[Series] = []
    for item in series:
        x_scale, x_unit = unit_transform(item.x_unit, profile)
        y_scale, y_unit = unit_transform(item.y_unit, profile)
        converted.append(
            replace(
                item,
                x=np.asarray(item.x, dtype=float) * x_scale,
                y=np.asarray(item.y, dtype=float) * y_scale,
                x_unit=x_unit,
                y_unit=y_unit,
            )
        )
    return converted
