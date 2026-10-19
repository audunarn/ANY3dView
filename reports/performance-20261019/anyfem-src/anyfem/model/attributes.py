"""Boundary conditions and loads, bound to geometry entities.

Nothing here names a node or an element.  Attributes reference geometry by
``EntityRef``, and the mesh association map resolves them at build time, so a
load survives a re-mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Sequence
from uuid import uuid4

import numpy as np

from anygeometry.entities import EntityRef
from anyloads import (
    FLUID_SIDES,
    CaseContext,
    Envelope,
    LoadError,
    PressureField,
    PressureTable,
    check_fluid_side,
    check_parameter,
    expression_names,
    trust_code,
)
from .connections import ConnectionLoad
from .regions import RegionRef

__all__ = [
    "DOF_NAMES",
    "Combination",
    "LineLoad",
    "LoadCase",
    "Mass",
    "PointLoad",
    "Pressure",
    "Support",
    "SurfaceTraction",
    "PressureTable",
    "WetPressure",
    "WetSurface",
    "antisymmetry",
    "fixed",
    "pinned",
    "prescribed",
    "simply_supported",
    "support",
    "symmetry",
]

DOF_NAMES: tuple[str, ...] = ("ux", "uy", "uz", "rx", "ry", "rz")


def _uuid() -> str:
    return str(uuid4())

# The three planes a symmetry condition can be expressed in exactly, keyed by
# the axis their normal points along.
_AXES: dict[str, int] = {"x": 0, "y": 1, "z": 2}


def _finite_scalar(value: float, what: str) -> float:
    """Return one modelling scalar, refusing flags and non-finite values."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{what} is a numeric value, not a boolean flag")
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a finite number") from None
    if not np.isfinite(resolved):
        raise ValueError(f"{what} must be a finite number")
    return resolved


def _vector3(values: Sequence[float], what: str) -> np.ndarray:
    """Return an owned, finite xyz vector with an actionable shape error."""

    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{what} needs three finite components") from None
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{what} needs three finite components")
    return vector.copy()


@dataclass(frozen=True)
class Support:
    """Prescribed degrees of freedom on one geometry entity."""

    name: str
    ref: EntityRef
    constraints: Mapping[str, float]
    region: RegionRef | None = None
    coordinate_system_id: str = "global"
    id: str = field(default_factory=_uuid)

    def __post_init__(self) -> None:
        unknown = set(self.constraints) - set(DOF_NAMES)
        if unknown:
            raise ValueError(
                f"support {self.name!r}: unknown degrees of freedom "
                f"{sorted(unknown)}; valid names are {', '.join(DOF_NAMES)}"
            )
        if not self.constraints:
            raise ValueError(f"support {self.name!r} constrains nothing")
        constraints = {}
        for dof, value in self.constraints.items():
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(
                    f"support {self.name!r} constraint {dof} is a prescribed "
                    "displacement, not a flag; use a numeric value, not a "
                    "boolean flag"
                )
            constraints[dof] = _finite_scalar(
                value, f"support {self.name!r} constraint {dof}"
            )
        object.__setattr__(self, "constraints", constraints)
        if not str(self.coordinate_system_id):
            raise ValueError("support coordinate system ID cannot be empty")


def support(ref: EntityRef, name: str | None = None, **dofs: float) -> Support:
    """Constrain named degrees of freedom, e.g. ``support(ref, uz=0.0)``.

    The value is the prescribed displacement, not a flag.  ``uz=True`` reads
    like "restrain uz" but would mean "move this by one metre", so booleans are
    rejected rather than quietly coerced -- the resulting model solves happily
    and gives a wrong answer, which is the worst kind of mistake to allow.
    """

    return Support(
        name=name or f"support_{ref}",
        ref=ref,
        constraints=dict(dofs),
    )


def _symmetry_axis(normal: Sequence[float] | str) -> str:
    """Which global axis a symmetry normal points along.

    Refuses anything else.  The solver's boundary conditions are expressed in
    global axes with no nodal transformation, so a tilted symmetry plane cannot
    be written down exactly -- only approximated by restraining the wrong
    directions.  Half a model with the wrong symmetry condition still solves and
    still looks plausible, so this fails closed instead.
    """

    if isinstance(normal, str):
        axis = normal.strip().lower()
        if axis not in _AXES:
            raise ValueError(
                f"unknown symmetry plane normal {normal!r}; expected 'x', 'y' "
                "or 'z'"
            )
        return axis

    vector = np.asarray(normal, dtype=float)
    if vector.shape != (3,):
        raise ValueError("a symmetry plane normal needs three components")
    if not np.all(np.isfinite(vector)):
        raise ValueError("a symmetry plane normal needs three finite components")
    length = float(np.linalg.norm(vector))
    if length <= 0.0:
        raise ValueError("a symmetry plane normal must be non-zero")
    vector = vector / length

    for axis, index in _AXES.items():
        if abs(abs(vector[index]) - 1.0) <= 1.0e-9:
            return axis
    raise ValueError(
        f"symmetry plane normal {tuple(np.round(vector, 6))} is not along a "
        "global axis. The solver applies boundary conditions in global axes "
        "with no nodal transformation, so a tilted symmetry plane cannot be "
        "expressed exactly -- only approximated, which would quietly give the "
        "wrong stiffness. Model the region with its symmetry planes on the "
        "global axes, or model it in full."
    )


def symmetry(
    ref: EntityRef, normal: Sequence[float] | str, name: str | None = None
) -> Support:
    """A symmetry plane: nothing crosses it, and nothing rotates out of it.

    On a plane with normal ``n`` the normal translation is restrained and so
    are the two rotations about in-plane axes; the rotation about ``n`` stays
    free.  For a plane normal to x that is ``ux = ry = rz = 0``, which is the
    usual XSYMM condition.

    Halving a model this way is not merely a convenience -- it is often the
    difference between a mesh that fits in memory and one that does not -- but
    it is only valid when the *loading* is symmetric too, which is the modeller's
    judgement and not something this function can check.
    """

    axis = _symmetry_axis(normal)
    index = _AXES[axis]
    constraints = {f"u{axis}": 0.0}
    for other, position in _AXES.items():
        if position != index:
            constraints[f"r{other}"] = 0.0
    return Support(
        name=name or f"symmetry_{axis}_{ref}",
        ref=ref,
        constraints=constraints,
    )


def antisymmetry(
    ref: EntityRef, normal: Sequence[float] | str, name: str | None = None
) -> Support:
    """The complement of a symmetry plane, for antisymmetric loading.

    The two in-plane translations and the rotation about the normal are
    restrained -- exactly the degrees of freedom :func:`symmetry` leaves free.
    Used where the load is antisymmetric about the plane, such as a torsion or
    an in-plane shear case on a half model.
    """

    axis = _symmetry_axis(normal)
    index = _AXES[axis]
    constraints = {f"r{axis}": 0.0}
    for other, position in _AXES.items():
        if position != index:
            constraints[f"u{other}"] = 0.0
    return Support(
        name=name or f"antisymmetry_{axis}_{ref}",
        ref=ref,
        constraints=constraints,
    )


def fixed(ref: EntityRef, name: str | None = None) -> Support:
    """Clamp all six degrees of freedom."""

    return Support(
        name=name or f"fixed_{ref}",
        ref=ref,
        constraints={dof: 0.0 for dof in DOF_NAMES},
    )


def pinned(ref: EntityRef, name: str | None = None) -> Support:
    """Restrain the three translations, leave rotations free."""

    return Support(
        name=name or f"pinned_{ref}",
        ref=ref,
        constraints={"ux": 0.0, "uy": 0.0, "uz": 0.0},
    )


def simply_supported(
    ref: EntityRef, name: str | None = None, normal: str = "uz"
) -> Support:
    """Restrain only the out-of-plane translation."""

    if normal not in ("ux", "uy", "uz"):
        raise ValueError("the simply supported direction must be ux, uy or uz")
    return Support(
        name=name or f"ss_{ref}", ref=ref, constraints={normal: 0.0}
    )


def prescribed(ref: EntityRef, name: str | None = None, **dofs: float) -> Support:
    """Force degrees of freedom to stated non-zero values.

    ``prescribed(ref, uz=0.005)`` pushes the entity 5 mm rather than holding
    it.  This is the same mechanism as a support -- the solver constrains a
    DOF to a value -- so a prescribed displacement and a restraint are the
    same object with different numbers.
    """

    if not dofs:
        raise ValueError("a prescribed displacement needs at least one value")
    return support(ref, name=name or f"prescribed_{ref}", **dofs)


@dataclass(frozen=True)
class Mass:
    """A lumped mass attached to geometry.

    On a point it is one mass; on a line or plate the total is shared equally
    over the nodes there.  It enters the mass matrix, so it shifts natural
    frequencies as well as producing inertial load under acceleration.
    """

    ref: EntityRef
    value: float
    name: str = "mass"
    region: RegionRef | None = None
    distribution_policy: str = "total_distributed"
    id: str = field(default_factory=_uuid)

    def __post_init__(self) -> None:
        value = _finite_scalar(self.value, f"mass {self.name!r}")
        if value < 0.0:
            raise ValueError(f"mass {self.name!r} must not be negative")
        object.__setattr__(self, "value", value)
        if self.distribution_policy not in ("per_target", "total_distributed"):
            raise ValueError("mass distribution policy must be per_target or total_distributed")


@dataclass(frozen=True)
class Combination:
    """A factored sum of load cases."""

    name: str
    factors: Mapping[str, float]
    id: str = field(default_factory=_uuid)

    def __post_init__(self) -> None:
        if not self.factors:
            raise ValueError(f"combination {self.name!r} combines nothing")
        factors = {
            case: _finite_scalar(
                factor, f"combination {self.name!r} factor for {case!r}"
            )
            for case, factor in self.factors.items()
        }
        object.__setattr__(self, "factors", factors)


@dataclass(frozen=True)
class PointLoad:
    """A force and moment applied at a point."""

    ref: EntityRef
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    moment: np.ndarray = field(default_factory=lambda: np.zeros(3))
    region: RegionRef | None = None
    coordinate_system_id: str = "global"
    distribution_policy: str = "per_target"
    id: str = field(default_factory=_uuid)
    # When the load acts and how strongly over a time history; None = always.
    envelope: Envelope | None = None

    def __post_init__(self) -> None:
        if self.ref.kind != "vertex":
            raise ValueError("a point load applies to a point")
        object.__setattr__(self, "envelope", Envelope.coerce(self.envelope))
        object.__setattr__(self, "force", _vector3(self.force, "point-load force"))
        object.__setattr__(self, "moment", _vector3(self.moment, "point-load moment"))
        if self.distribution_policy not in ("per_target", "total_distributed"):
            raise ValueError("point-load distribution policy must be per_target or total_distributed")


@dataclass(frozen=True)
class Pressure:
    """Uniform pressure on a plate, positive along the element normal."""

    ref: EntityRef
    value: float
    region: RegionRef | None = None
    id: str = field(default_factory=_uuid)
    # When the load acts and how strongly over a time history; None = always.
    envelope: Envelope | None = None

    def __post_init__(self) -> None:
        if self.ref.kind != "face":
            raise ValueError("a pressure applies to a plate")
        object.__setattr__(self, "envelope", Envelope.coerce(self.envelope))
        object.__setattr__(self, "value", _finite_scalar(self.value, "pressure"))


@dataclass(frozen=True)
class LineLoad:
    """A distributed force per unit length along a line, in N/m."""

    ref: EntityRef
    force_per_length: np.ndarray
    region: RegionRef | None = None
    coordinate_system_id: str = "global"
    id: str = field(default_factory=_uuid)
    # When the load acts and how strongly over a time history; None = always.
    envelope: Envelope | None = None

    def __post_init__(self) -> None:
        if self.ref.kind != "edge":
            raise ValueError("a line load applies to a line")
        object.__setattr__(self, "envelope", Envelope.coerce(self.envelope))
        object.__setattr__(
            self,
            "force_per_length",
            _vector3(self.force_per_length, "line-load force per length"),
        )


@dataclass(frozen=True)
class SurfaceTraction:
    """A distributed force per unit area on a plate, in N/m^2.

    Unlike a pressure, the direction is fixed in space rather than following
    the plate normal, so it describes things like a horizontal wind load on a
    sloping surface.
    """

    ref: EntityRef
    traction: np.ndarray
    region: RegionRef | None = None
    coordinate_system_id: str = "global"
    id: str = field(default_factory=_uuid)
    # When the load acts and how strongly over a time history; None = always.
    envelope: Envelope | None = None

    def __post_init__(self) -> None:
        if self.ref.kind != "face":
            raise ValueError("a surface traction applies to a plate")
        object.__setattr__(self, "envelope", Envelope.coerce(self.envelope))
        object.__setattr__(
            self, "traction", _vector3(self.traction, "surface traction")
        )


WET_FLUID_SIDES: tuple[str, ...] = FLUID_SIDES
WET_MAPPING_KEYS: tuple[str, ...] = ("face", "element")


@dataclass(frozen=True)
class WetSurface:
    """A plate wetted by fluid: the plates a wet pressure is applied to.

    A pressure on a wet surface is the fluid pressing *into* the plate. Which
    way that is depends on which side of the plate the fluid is on, so the
    surface records it:

    ``positive_normal``
        the fluid touches the side the element normal points to, so the load
        acts against the normal.
    ``negative_normal``
        the fluid touches the opposite side, so the load acts along the normal.
    ``away_from_point``
        the fluid is outside a closed volume containing ``interior_point``
        (a hull, a tank). The side is decided per element from where its
        normal points, so inconsistent element orientation cannot flip a load.
    """

    ref: EntityRef
    name: str = "wet surface"
    region: RegionRef | None = None
    fluid_side: str = "positive_normal"
    interior_point: np.ndarray | None = None
    id: str = field(default_factory=_uuid)

    def __post_init__(self) -> None:
        if self.ref.kind != "face":
            raise ValueError("a wet surface is made of plates")
        # ANYloads owns what a fluid side means; this record only carries it.
        object.__setattr__(
            self,
            "interior_point",
            check_fluid_side(self.fluid_side, self.interior_point),
        )


@dataclass(frozen=True)
class WetPressure:
    """Fluid pressure on wet surfaces: a value, equation, mapping, code or table.

    Exactly one of ``value``, ``expression``, ``mapping``, ``code`` and
    ``table`` is given. The pressure is a positive magnitude pressing into the
    plate; the wet surface's fluid side turns it into the solver's
    normal-signed pressure.

    ``expression``
        A formula of the element centroid ``x, y, z`` (metres), the case time
        ``t`` and the case's named parameters, e.g.
        ``rho*g*max(0, draft - z)`` for hydrostatic pressure. Elements above
        the waterline evaluate to zero through ``max``.
    ``mapping``
        Pressures given directly: ``{face_id: value}`` for a pressure per
        geometric plate (``mapping_key="face"``) or ``{element_id: value}`` for
        a pressure per mesh element (``mapping_key="element"``). Wet plates or
        elements the mapping does not list are dry (zero). Naming one that is
        not on a wet surface is refused.

    ``code``
        Python source defining ``def pressure(case, x, y, z): ...`` that
        returns the pressure for a load case number and an element centroid.
        It may also declare ``t``, ``name`` and ``parameters``. It runs only
        once trusted (see :mod:`anyfem.model.usercode`); with
        ``code_vectorized`` it is called once with numpy arrays instead of per
        element.
    ``table``
        A :class:`PressureTable`: pressures at points per load case, from a
        CSV/Excel file. ``table_case`` pins which case identifier to use
        instead of deriving it from the load case.

    ``surfaces`` restricts the pressure to some wet surfaces by id; empty
    means every wet surface. ``scale`` multiplies the result (e.g. 1000 for a
    table in kPa).
    """

    value: float | None = None
    expression: str | None = None
    mapping: Mapping[int, float] | None = None
    mapping_key: str = "face"
    surfaces: tuple[str, ...] = ()
    scale: float = 1.0
    name: str = "wet pressure"
    id: str = field(default_factory=_uuid)
    code: str | None = None
    code_vectorized: bool = False
    table: PressureTable | None = None
    table_case: float | int | str | None = None
    # When the load acts and how strongly over a time history; None = always.
    envelope: Envelope | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "envelope", Envelope.coerce(self.envelope))
        given = [
            label
            for label, item in (
                ("value", self.value),
                ("expression", self.expression),
                ("mapping", self.mapping),
                ("code", self.code),
                ("table", self.table),
            )
            if item is not None
        ]
        if len(given) != 1:
            raise ValueError(
                "a wet pressure needs exactly one of a value, an expression, "
                f"a mapping, code or a table (got {', '.join(given) or 'none'})"
            )
        if self.mapping is None:
            # ANYloads validates and normalises every source it evaluates.
            normalised = PressureField(
                value=self.value,
                expression=self.expression,
                code=self.code,
                code_vectorized=self.code_vectorized,
                table=self.table,
                table_case=self.table_case,
                scale=self.scale,
            )
            object.__setattr__(self, "value", normalised.value)
            object.__setattr__(self, "code_vectorized", normalised.code_vectorized)
            object.__setattr__(self, "scale", normalised.scale)
        elif self.code is not None or self.table_case is not None:
            raise ValueError("a mapping cannot also carry code or a table case")
        if self.mapping is not None:
            if self.mapping_key not in WET_MAPPING_KEYS:
                raise ValueError(
                    "mapping key must be one of " + ", ".join(WET_MAPPING_KEYS)
                )
            if not self.mapping:
                raise ValueError("a wet pressure mapping needs at least one entry")
            mapping: dict[int, float] = {}
            for key, value in dict(self.mapping).items():
                try:
                    if isinstance(key, (bool, np.bool_)):
                        raise ValueError
                    identifier = int(key)
                    if isinstance(key, float) and identifier != key:
                        raise ValueError
                except (TypeError, ValueError):
                    raise ValueError(
                        "a wet pressure mapping is keyed by integer ids"
                    ) from None
                mapping[identifier] = _finite_scalar(
                    value, f"pressure for {key!r}"
                )
            object.__setattr__(self, "mapping", mapping)
        object.__setattr__(
            self, "scale", _finite_scalar(self.scale, "pressure scale")
        )
        object.__setattr__(self, "surfaces", tuple(str(i) for i in self.surfaces))

    @property
    def kind(self) -> str:
        if self.value is not None:
            return "uniform"
        if self.expression is not None:
            return "expression"
        if self.code is not None:
            return "code"
        return "table" if self.table is not None else "mapping"

    @property
    def field(self) -> PressureField:
        """The ANYloads definition that evaluates this pressure (not a mapping)."""

        if self.mapping is not None:
            raise ValueError("a mapping is evaluated per plate or element id")
        return PressureField(
            value=self.value,
            expression=self.expression,
            code=self.code,
            code_vectorized=self.code_vectorized,
            table=self.table,
            table_case=self.table_case,
            scale=self.scale,
        )


@dataclass
class LoadCase:
    """A named collection of loads.

    ``follower_pressure`` is a property of the whole case, not of individual
    pressures, because that is how the solver models it: the load vector is
    assembled either in the reference configuration or the current one.
    Mixing dead and follower pressure inside one case is therefore not
    representable, and asking for it is refused rather than silently resolved.
    """

    name: str
    point_loads: List[PointLoad] = field(default_factory=list)
    pressures: List[Pressure] = field(default_factory=list)
    line_loads: List[LineLoad] = field(default_factory=list)
    surface_tractions: List[SurfaceTraction] = field(default_factory=list)
    gravity: np.ndarray | None = None
    follower_pressure: bool = False
    id: str = field(default_factory=_uuid)
    gravity_coordinate_system_id: str = "global"
    _region_factory: Callable[[EntityRef], RegionRef] | None = field(
        default=None, repr=False, compare=False
    )
    wet_pressures: List[WetPressure] = field(default_factory=list)
    # Force and moment at a connection's reference point.
    connection_loads: List[ConnectionLoad] = field(default_factory=list)
    # Quasi-static stepping: a case may carry the time its equations are
    # evaluated at and named numbers those equations read. ``series`` and
    # ``step`` record which generated sequence a case belongs to.
    time: float | None = None
    parameters: dict[str, float] = field(default_factory=dict)
    series: str | None = None
    step: int | None = None
    # The load case number handed to pressure code and matched by a
    # number-keyed pressure table. Assigned when the project creates the case;
    # a series step is numbered by its step instead (see ``case_number``).
    number: int | None = None
    # A load series definition: its loads (each with an optional envelope) are
    # stored once, and one derived step case per time takes them (see
    # ``Project.add_load_series``).  A definition is never solved itself.
    times: tuple[float, ...] | None = None
    # A derived step: the series definition it takes its loads from.  It holds
    # no loads of its own.
    parent: str | None = None
    # The definition's times, kept on the resolved view of a step so an
    # envelope window by number can tell when it opened.
    _window_times: tuple[float, ...] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def is_series(self) -> bool:
        """True for a load series definition (its steps are the solvable cases)."""

        return self.times is not None

    def envelope_factor(self, envelope: Envelope | None) -> float:
        """The multiplier an envelope gives a load in this case (1 if none)."""

        if envelope is None:
            return 1.0
        start_time = None
        times = self._window_times
        if envelope.by == "number" and envelope.start is not None and times:
            index = int(round(envelope.start)) - 1
            if 0 <= index < len(times):
                start_time = float(times[index])
        return envelope.factor(self.context(), start_time=start_time)

    @property
    def case_number(self) -> int:
        """1-based number of this case: step + 1 within a series, else ``number``."""

        if self.step is not None:
            return int(self.step) + 1
        return 0 if self.number is None else int(self.number)

    def add_point_load(
        self,
        ref: EntityRef,
        force: Sequence[float] = (0.0, 0.0, 0.0),
        moment: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        region: RegionRef | None = None,
        coordinate_system_id: str = "global",
        distribution_policy: str = "per_target",
        envelope: Envelope | Mapping | None = None,
    ) -> PointLoad:
        region = region or (
            self._region_factory(ref) if self._region_factory is not None else None
        )
        load = PointLoad(
            ref=ref,
            force=force,  # type: ignore[arg-type]
            moment=moment,  # type: ignore[arg-type]
            region=region,
            coordinate_system_id=coordinate_system_id,
            distribution_policy=distribution_policy,
            envelope=envelope,  # type: ignore[arg-type]
        )
        self.point_loads.append(load)
        return load

    def add_pressure(
        self,
        ref: EntityRef,
        value: float,
        *,
        region: RegionRef | None = None,
        envelope: Envelope | Mapping | None = None,
    ) -> Pressure:
        region = region or (
            self._region_factory(ref) if self._region_factory is not None else None
        )
        load = Pressure(ref=ref, value=value, region=region, envelope=envelope)  # type: ignore[arg-type]
        self.pressures.append(load)
        return load

    def add_wet_pressure(
        self,
        *,
        value: float | None = None,
        expression: str | None = None,
        mapping: Mapping[int, float] | None = None,
        mapping_key: str = "face",
        surfaces: Sequence[str] = (),
        scale: float = 1.0,
        name: str = "wet pressure",
        code: str | None = None,
        code_vectorized: bool = False,
        table: PressureTable | None = None,
        table_case: float | int | str | None = None,
        envelope: Envelope | Mapping | None = None,
    ) -> WetPressure:
        """Apply fluid pressure to the project's wet surfaces.

        Code passed here is the caller's own, so it is trusted on this
        machine; code that arrives inside a project file is not.
        """

        load = WetPressure(
            value=value,
            expression=expression,
            mapping=mapping,
            mapping_key=mapping_key,
            surfaces=tuple(surfaces),
            scale=scale,
            name=name,
            code=code,
            code_vectorized=code_vectorized,
            table=table,
            table_case=table_case,
            envelope=envelope,  # type: ignore[arg-type]
        )
        if code is not None:
            trust_code(code)
        self.wet_pressures.append(load)
        return load

    def add_connection_load(
        self,
        connection: str,
        force: Sequence[float] = (0.0, 0.0, 0.0),
        moment: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        name: str = "connection load",
        envelope: Envelope | Mapping | None = None,
    ) -> ConnectionLoad:
        """Apply a force and moment at a connection's reference point.

        ``connection`` is the id of a connection added to the project; a moment
        here is a moment on every entity the connection ties together.
        """

        load = ConnectionLoad(
            connection=connection,
            force=force,  # type: ignore[arg-type]
            moment=moment,  # type: ignore[arg-type]
            name=name,
            envelope=envelope,  # type: ignore[arg-type]
        )
        self.connection_loads.append(load)
        return load

    def set_time(self, time: float | None) -> None:
        """The time this case's equations are evaluated at (``t``)."""

        self.time = None if time is None else _finite_scalar(time, "case time")

    def set_parameter(self, name: str, value: float) -> None:
        """A named number the case's pressure equations may read."""

        key, number = check_parameter(name, value)
        self.parameters[key] = number

    def context(self) -> CaseContext:
        """This case as ANYloads sees it: number, name, time, parameters."""

        return CaseContext(
            number=self.case_number,
            name=self.name,
            time=self.time,
            parameters=self.parameters,
        )

    def add_line_load(
        self, ref: EntityRef, force_per_length: Sequence[float], *,
        region: RegionRef | None = None,
        coordinate_system_id: str = "global",
        envelope: Envelope | Mapping | None = None,
    ) -> LineLoad:
        region = region or (
            self._region_factory(ref) if self._region_factory is not None else None
        )
        load = LineLoad(
            ref=ref, force_per_length=force_per_length,  # type: ignore[arg-type]
            region=region, coordinate_system_id=coordinate_system_id,
            envelope=envelope,  # type: ignore[arg-type]
        )
        self.line_loads.append(load)
        return load

    def add_surface_traction(
        self, ref: EntityRef, traction: Sequence[float], *,
        region: RegionRef | None = None,
        coordinate_system_id: str = "global",
        envelope: Envelope | Mapping | None = None,
    ) -> SurfaceTraction:
        region = region or (
            self._region_factory(ref) if self._region_factory is not None else None
        )
        load = SurfaceTraction(
            ref=ref, traction=traction,  # type: ignore[arg-type]
            region=region, coordinate_system_id=coordinate_system_id,
            envelope=envelope,  # type: ignore[arg-type]
        )
        self.surface_tractions.append(load)
        return load

    def set_gravity(
        self,
        gx: float = 0.0,
        gy: float = 0.0,
        gz: float = -9.81,
        *,
        coordinate_system_id: str = "global",
    ) -> None:
        self.gravity = _vector3((gx, gy, gz), "gravity")
        self.gravity_coordinate_system_id = str(coordinate_system_id)

    def set_acceleration(
        self,
        ax: float = 0.0,
        ay: float = 0.0,
        az: float = 0.0,
        *,
        coordinate_system_id: str = "global",
    ) -> None:
        """A design acceleration field, e.g. a vessel motion.

        This is the same mechanism as gravity -- a consistent inertial load
        over the structural mass -- so setting one replaces the other.
        """

        self.gravity = _vector3((ax, ay, az), "acceleration")
        self.gravity_coordinate_system_id = str(coordinate_system_id)

    def set_follower_pressure(self, follower: bool = True) -> None:
        """Make this case's pressures act on the deformed configuration."""

        self.follower_pressure = bool(follower)

    @property
    def is_template(self) -> bool:
        """True when an equation reads the time but the case has none.

        Such a case is a pattern to generate a series from, not something to
        solve; batch solves skip it unless it is named explicitly.
        """

        if self.times is not None:
            return True
        if self.time is not None:
            return False
        return any(
            load.expression is not None and "t" in expression_names(load.expression)
            for load in self.wet_pressures
        )

    def is_empty(self) -> bool:
        if self.parent is not None:
            return False  # a step carries its definition's loads
        return not (
            self.point_loads
            or self.pressures
            or self.line_loads
            or self.surface_tractions
            or self.wet_pressures
            or self.connection_loads
            or self.gravity is not None
        )
