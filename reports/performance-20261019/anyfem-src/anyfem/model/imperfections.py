"""Geometric imperfections, bound to geometry like every other attribute.

An imperfection alters the *stress-free reference geometry*.  It is not an
initial displacement and it is not a residual stress: the imperfect shape is
the shape the structure would have with no load on it at all.  The solver keeps
that distinction, and so does this layer -- imperfections are declared on the
project and applied when the FEModel is built, never mixed into a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np

from anygeometry.entities import EntityRef

__all__ = ["Imperfection", "member_bow", "plate_mode"]

IMPERFECTION_KINDS: tuple[str, ...] = ("auto", "plate_mode", "member_bow")


def _integral_pair(values: Sequence[int], message: str) -> Tuple[int, int]:
    """Normalize exactly two integral values without truncating fractions."""

    try:
        pair = tuple(values)
    except TypeError:
        raise ValueError(message) from None
    if len(pair) != 2:
        raise ValueError(message)
    normalized = []
    for value in pair:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(message)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(message) from None
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(message)
        normalized.append(int(numeric))
    return normalized[0], normalized[1]


@dataclass(frozen=True)
class Imperfection:
    """A stress-free shape deviation on one plate or line.

    ``amplitude`` of None takes the solver's standard default: span/200 for a
    plate mode, length/300 for a member bow.
    """

    ref: EntityRef
    kind: str = "auto"
    amplitude: Optional[float] = None
    direction: Sequence[float] = (0.0, 0.0, 1.0)
    waves: Tuple[int, int] = (1, 1)
    axes: Tuple[int, int] = (0, 1)
    name: str = "imperfection"
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if not str(self.id):
            raise ValueError("imperfection ID cannot be empty")
        if self.kind not in IMPERFECTION_KINDS:
            raise ValueError(
                f"unknown imperfection kind {self.kind!r}; expected one of "
                f"{', '.join(IMPERFECTION_KINDS)}"
            )
        if self.amplitude is not None:
            if isinstance(self.amplitude, (bool, np.bool_)):
                raise ValueError(
                    "an imperfection amplitude must be a finite, non-negative number"
                )
            try:
                amplitude = float(self.amplitude)
            except (TypeError, ValueError):
                raise ValueError(
                    "an imperfection amplitude must be a finite, non-negative number"
                ) from None
            if not np.isfinite(amplitude) or amplitude < 0.0:
                raise ValueError(
                    "an imperfection amplitude must be a finite, non-negative number"
                )
            object.__setattr__(self, "amplitude", amplitude)

        try:
            direction = np.asarray(self.direction, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                "an imperfection direction needs three finite, non-zero components"
            ) from None
        if (
            direction.shape != (3,)
            or not np.all(np.isfinite(direction))
            or float(np.linalg.norm(direction)) == 0.0
        ):
            raise ValueError(
                "an imperfection direction needs three finite, non-zero components"
            )
        object.__setattr__(
            self, "direction", tuple(float(value) for value in direction)
        )

        waves = _integral_pair(
            self.waves, "wave counts must be exactly two positive integers"
        )
        if any(wave < 1 for wave in waves):
            raise ValueError("wave counts must be exactly two positive integers")
        object.__setattr__(self, "waves", waves)

        # standard_plate_mode indexes the three Cartesian coordinate columns;
        # any distinct pair is therefore supported, including vertical plates.
        axes = _integral_pair(
            self.axes,
            "imperfection axes must be exactly two distinct coordinate axes "
            "chosen from 0, 1, 2",
        )
        if len(set(axes)) != 2 or any(axis not in (0, 1, 2) for axis in axes):
            raise ValueError(
                "imperfection axes must be exactly two distinct coordinate axes "
                "chosen from 0, 1, 2"
            )
        object.__setattr__(self, "axes", axes)

        if self.resolved_kind == "plate_mode" and self.ref.kind != "face":
            raise ValueError("a plate mode applies to a plate")
        if self.resolved_kind == "member_bow" and self.ref.kind != "edge":
            raise ValueError("a member bow applies to a line")

    @property
    def resolved_kind(self) -> str:
        """The kind to build, with ``auto`` decided by what it is attached to."""

        if self.kind != "auto":
            return self.kind
        return "plate_mode" if self.ref.kind == "face" else "member_bow"


def plate_mode(
    ref: EntityRef,
    amplitude: Optional[float] = None,
    waves: Tuple[int, int] = (1, 1),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    name: str = "plate_mode",
) -> Imperfection:
    """A sinusoidal buckling-shaped deviation across a plate."""

    return Imperfection(
        ref=ref,
        kind="plate_mode",
        amplitude=amplitude,
        waves=waves,
        direction=direction,
        name=name,
    )


def member_bow(
    ref: EntityRef,
    amplitude: Optional[float] = None,
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    name: str = "member_bow",
) -> Imperfection:
    """A half-sine bow along a line, as a member out-of-straightness."""

    return Imperfection(
        ref=ref,
        kind="member_bow",
        amplitude=amplitude,
        direction=direction,
        name=name,
    )
