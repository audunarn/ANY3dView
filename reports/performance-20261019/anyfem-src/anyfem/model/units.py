"""Project display units with SI as the immutable storage convention.

The solver and the public modelling API deliberately continue to use plain SI
floats.  :class:`UnitProfile` is an input/output boundary used by the desktop
application, reports and scripts that explicitly opt into formatted values.
Keeping conversion here prevents display preferences from changing model
hashes or numerical results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import ClassVar, Mapping

import numpy as np

__all__ = ["UNIT_PROFILES", "UnitError", "UnitProfile", "unit_profile"]


class UnitError(ValueError):
    """Raised when a unit or quantity cannot be interpreted safely."""


# Scale is the number of SI units represented by one display unit.
_UNITS: dict[str, tuple[str, float]] = {
    "": ("dimensionless", 1.0),
    "1": ("dimensionless", 1.0),
    "m": ("length", 1.0),
    "cm": ("length", 1.0e-2),
    "mm": ("length", 1.0e-3),
    "N": ("force", 1.0),
    "kN": ("force", 1.0e3),
    "MN": ("force", 1.0e6),
    "Pa": ("pressure", 1.0),
    "kPa": ("pressure", 1.0e3),
    "MPa": ("pressure", 1.0e6),
    "GPa": ("pressure", 1.0e9),
    "kg": ("mass", 1.0),
    "t": ("mass", 1.0e3),
    "s": ("time", 1.0),
    "ms": ("time", 1.0e-3),
    "rad": ("angle", 1.0),
    "deg": ("angle", np.pi / 180.0),
    "N*m": ("moment", 1.0),
    "N*mm": ("moment", 1.0e-3),
    "kN*m": ("moment", 1.0e3),
    "N/mm": ("line_load", 1.0e3),
    "N/m": ("line_load", 1.0),
    "kN/m": ("line_load", 1.0e3),
    "kg/m3": ("density", 1.0),
    "t/m3": ("density", 1.0e3),
    "m/s2": ("acceleration", 1.0),
    "mm/s2": ("acceleration", 1.0e-3),
}


_NUMBER_UNIT = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*$"
)


@dataclass(frozen=True)
class UnitProfile:
    """Named display units for structural quantities.

    ``parse`` accepts either a bare number (interpreted in the profile unit)
    or an explicit supported suffix.  ``to_display`` and ``to_si`` also work
    on NumPy arrays, which is useful for coordinate/result tables.
    """

    name: str
    units: Mapping[str, str] = field(default_factory=dict)

    REQUIRED: ClassVar[tuple[str, ...]] = (
        "length",
        "force",
        "pressure",
        "mass",
        "time",
        "angle",
        "moment",
        "line_load",
        "density",
        "acceleration",
    )

    def __post_init__(self) -> None:
        resolved = dict(self.units)
        missing = [dimension for dimension in self.REQUIRED if dimension not in resolved]
        if missing:
            raise UnitError(
                f"unit profile {self.name!r} is missing {', '.join(missing)}"
            )
        for dimension, symbol in resolved.items():
            actual = _UNITS.get(symbol)
            if actual is None:
                raise UnitError(f"unsupported unit {symbol!r} for {dimension}")
            if actual[0] != dimension:
                raise UnitError(
                    f"unit {symbol!r} measures {actual[0]}, not {dimension}"
                )
        object.__setattr__(self, "units", resolved)

    def symbol(self, dimension: str) -> str:
        try:
            return self.units[dimension]
        except KeyError:
            raise UnitError(f"profile {self.name!r} has no {dimension} unit") from None

    def factor(self, dimension: str, symbol: str | None = None) -> float:
        chosen = self.symbol(dimension) if symbol is None else symbol
        try:
            actual, scale = _UNITS[chosen]
        except KeyError:
            raise UnitError(f"unsupported unit {chosen!r}") from None
        if actual != dimension:
            raise UnitError(f"unit {chosen!r} measures {actual}, not {dimension}")
        return float(scale)

    def to_si(self, value, dimension: str, *, unit: str | None = None):
        return np.asarray(value) * self.factor(dimension, unit) if np.ndim(value) else float(value) * self.factor(dimension, unit)

    def to_display(self, value, dimension: str):
        factor = self.factor(dimension)
        return np.asarray(value) / factor if np.ndim(value) else float(value) / factor

    def parse(self, text: str | float | int, dimension: str) -> float:
        if isinstance(text, (int, float, np.integer, np.floating)) and not isinstance(text, (bool, np.bool_)):
            value = float(text)
            unit = None
        else:
            match = _NUMBER_UNIT.match(str(text))
            if match is None:
                raise UnitError(f"{text!r} is not a number with an optional unit")
            value = float(match.group(1))
            unit = match.group(2) or None
        if not np.isfinite(value):
            raise UnitError("quantity must be finite")
        return float(self.to_si(value, dimension, unit=unit))

    def format(self, value: float, dimension: str, *, precision: int = 6) -> str:
        shown = float(self.to_display(value, dimension))
        return f"{shown:.{precision}g} {self.symbol(dimension)}".rstrip()

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "units": dict(self.units)}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "UnitProfile":
        name = str(data.get("name", "custom"))
        raw = data.get("units", {})
        if not isinstance(raw, Mapping):
            raise UnitError("unit profile units must be a mapping")
        return cls(name=name, units={str(k): str(v) for k, v in raw.items()})


def _profile(name: str, **overrides: str) -> UnitProfile:
    units = {
        "length": "m",
        "force": "N",
        "pressure": "Pa",
        "mass": "kg",
        "time": "s",
        "angle": "deg",
        "moment": "N*m",
        "line_load": "N/m",
        "density": "kg/m3",
        "acceleration": "m/s2",
    }
    units.update(overrides)
    return UnitProfile(name=name, units=units)


UNIT_PROFILES: dict[str, UnitProfile] = {
    "SI-m-N-Pa": _profile("SI-m-N-Pa"),
    "SI-mm-N-MPa": _profile(
        "SI-mm-N-MPa",
        length="mm",
        pressure="MPa",
        moment="N*mm",
        line_load="N/mm",
        acceleration="mm/s2",
    ),
    "SI-m-kN-kPa": _profile(
        "SI-m-kN-kPa",
        force="kN",
        pressure="kPa",
        moment="kN*m",
        line_load="kN/m",
    ),
}


def unit_profile(name: str = "SI-m-N-Pa") -> UnitProfile:
    try:
        return UNIT_PROFILES[name]
    except KeyError:
        raise UnitError(
            f"unknown unit profile {name!r}; expected one of "
            f"{', '.join(UNIT_PROFILES)}"
        ) from None
