"""Adapters between ANYfem's wet loads and ANYloads' evaluation.

ANYloads owns how a pressure is defined and evaluated (equation, trusted code,
table) and which way fluid pressure acts; this module only translates ANYfem's
records into its vocabulary and its errors into ANYfem's.
"""

from __future__ import annotations

import numpy as np

from anyloads import LoadError, UntrustedCodeError
from anyloads import fluid_sign as _fluid_sign

from .attributes import LoadCase, WetPressure, WetSurface

__all__ = [
    "UntrustedWetCodeError",
    "WetFieldError",
    "field_pressure",
    "fluid_sign",
]


class WetFieldError(ValueError):
    """A wet pressure cannot be evaluated."""


class UntrustedWetCodeError(WetFieldError):
    """Pressure code from a project has not been approved on this machine."""


def fluid_sign(
    surface: WetSurface, centroid: np.ndarray, normal: np.ndarray
) -> float:
    """Sign turning a pressure magnitude into a normal-signed pressure."""

    try:
        return _fluid_sign(
            surface.fluid_side,
            surface.interior_point,
            centroid,
            normal,
            label=f"wet surface {surface.name!r}",
        )
    except LoadError as error:
        raise WetFieldError(str(error)) from None


def field_pressure(
    case: LoadCase, load: WetPressure, centroids: np.ndarray
) -> np.ndarray:
    """A pressure source's magnitude at each centroid for this case (scaled).

    Covers every kind except a mapping by plate or element id, which needs the
    mesh and is resolved by the solver build.
    """

    where = f"load case {case.name!r} wet pressure {load.name!r}"
    try:
        return load.field.evaluate(case.context(), centroids)
    except UntrustedCodeError as error:
        raise UntrustedWetCodeError(f"{where}: {error}") from None
    except LoadError as error:
        hint = (
            "; set one, or generate the case with add_load_case_series"
            if case.time is None and "no time" in str(error)
            else ""
        )
        raise WetFieldError(f"{where}: {error}{hint}") from None
