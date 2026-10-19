"""Application-facing material specifications backed by ANYmaterial.

ANYfem stores descriptions rather than live constitutive objects so project
files remain JSON serializable. ``Material`` keeps the historical isotropic
constructor while also accepting the full ``MaterialSpec`` schema.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Optional

from anymaterial import MaterialSpec, available_grades, dnv_c208_steel_properties

__all__ = [
    "Material",
    "MaterialSpec",
    "canonical_dnv_grade",
    "dnv_steel_material",
    "dnv_steel_material_name",
    "steel",
]


def canonical_dnv_grade(value: str) -> str:
    """Resolve a DNV grade without confusing it with a material label.

    Older UI versions exposed only one text box, so project labels such as
    ``S355_NL`` were sometimes passed to the RP-C208 lookup as grades.  Accept
    that legacy form only when the delimiter-separated leading grade is
    unambiguous.  Invalid values are left for ANYmaterial's authoritative
    validation and diagnostic.
    """

    token = str(value).strip().upper()
    grades = tuple(str(grade).upper() for grade in available_grades())
    if token in grades:
        return token
    matches = tuple(
        grade
        for grade in grades
        if token.startswith((f"{grade}_", f"{grade}-", f"{grade} "))
    )
    return matches[0] if len(matches) == 1 else token


def _hardening_descriptor(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 3:
            raise ValueError("a legacy hardening descriptor needs kind, grade and thickness")
        kind, grade, thickness = value
        if str(kind) != "dnv_c208":
            raise ValueError(
                f"unknown hardening source {kind!r}; only 'dnv_c208' is available"
            )
        return {
            "kind": "dnv_c208",
            "grade": str(grade),
            "thickness": float(thickness),
        }
    raise TypeError("hardening must be a descriptor mapping or legacy 3-item sequence")


class Material(MaterialSpec):
    """A ``MaterialSpec`` with ANYfem's former isotropic constructor."""

    def __init__(
        self,
        name: str,
        elastic_modulus: Optional[float] = None,
        poisson_ratio: Optional[float] = None,
        density: float = 7850.0,
        yield_stress: float = 0.0,
        hardening: Any = None,
        *,
        symmetry: str = "isotropic",
        constants: Optional[Mapping[str, float]] = None,
        hill: Optional[Mapping[str, float]] = None,
    ) -> None:
        if constants is None:
            if elastic_modulus is None or poisson_ratio is None:
                raise ValueError(
                    "an isotropic material needs elastic_modulus and poisson_ratio"
                )
            constants = {
                "elastic_modulus": float(elastic_modulus),
                "poisson_ratio": float(poisson_ratio),
            }
        super().__init__(
            name=str(name),
            symmetry=str(symmetry),
            constants=dict(constants),
            density=float(density),
            yield_stress=float(yield_stress),
            hardening=_hardening_descriptor(hardening),
            hill=None if hill is None else dict(hill),
        )
        # MaterialSpec validates the schema. Building also validates the actual
        # constants immediately, preserving ANYfem's fail-fast constructor.
        self.build()

    @property
    def elastic_modulus(self) -> float:
        if self.symmetry != "isotropic":
            raise AttributeError("an orthotropic material has directional elastic moduli")
        return float(self.constants["elastic_modulus"])

    @property
    def poisson_ratio(self) -> float:
        if self.symmetry != "isotropic":
            raise AttributeError("an orthotropic material has directional Poisson ratios")
        return float(self.constants["poisson_ratio"])


def steel(
    grade: str = "S355",
    thickness: float = 0.010,
    *,
    name: Optional[str] = None,
    density: float = 7850.0,
    nonlinear: bool = False,
) -> Material:
    """Build a serializable steel specification from ANYmaterial's table."""

    grade = canonical_dnv_grade(grade)
    properties = dnv_c208_steel_properties(grade, thickness)
    return Material(
        name=name or str(properties["grade"]),
        elastic_modulus=float(properties["E_pa"]),
        poisson_ratio=0.3,
        density=density,
        yield_stress=float(properties["sigma_yield"]),
        hardening=("dnv_c208", grade, float(thickness)) if nonlinear else None,
    )


def dnv_steel_material_name(
    grade: str,
    thickness: float,
    *,
    nonlinear: bool = True,
) -> str:
    """Return a deterministic, thickness-qualified DNV steel name.

    A grade alone is not a unique material specification: DNV C208 yield and
    hardening data depend on product thickness.  The generated name therefore
    prevents an S355 plate of one thickness from replacing an S355 plate of
    another thickness in a project's material registry.
    """

    value = float(thickness)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("DNV steel thickness must be finite and positive")
    grade_token = canonical_dnv_grade(grade)
    if not grade_token:
        raise ValueError("DNV steel grade must not be empty")
    thickness_mm = format(value * 1000.0, ".12g")
    behavior = "NL" if nonlinear else "EL"
    return f"{grade_token}-DNV-C208-t{thickness_mm}mm-{behavior}"


def dnv_steel_material(
    grade: str = "S355",
    thickness: float = 0.010,
    *,
    nonlinear: bool = True,
    density: float = 7850.0,
    name: Optional[str] = None,
) -> Material:
    """Build a reusable thickness-qualified DNV steel material.

    Nonlinear DNV hardening is enabled by default because this factory is used
    by the automatic plate-section workflow.  Use :func:`steel` when the
    historical grade-only name or a custom name is required.
    """

    grade = canonical_dnv_grade(grade)
    material_name = name or dnv_steel_material_name(
        grade, thickness, nonlinear=nonlinear
    )
    return steel(
        grade,
        thickness,
        name=material_name,
        density=density,
        nonlinear=nonlinear,
    )
