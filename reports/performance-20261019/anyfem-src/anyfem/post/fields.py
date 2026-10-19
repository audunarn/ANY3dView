"""Scalar fields over a result: one way to ask for any quantity.

Displacement lives at nodes; stress lives at elements, several values deep
(one per Gauss point, and membrane and bending separately).  Rather than
scatter that difference through the display code, everything is answered as a
:class:`Field` -- a name, a unit, and values keyed either by node or by
element.  The scene builder, the probes, the path plots, the envelopes and the
report all consume the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


__all__ = [
    "DISPLACEMENT_FIELDS",
    "Field",
    "PLASTICITY_FIELDS",
    "STRESS_FIELDS",
    "available_fields",
    "element_centroids",
    "evaluate_field",
    "field_unit",
]

DISPLACEMENT_FIELDS: Tuple[str, ...] = (
    "magnitude",
    "ux",
    "uy",
    "uz",
    "rx",
    "ry",
    "rz",
)

# Recovered shell components, plus the surface stresses derived from them.
_SHELL_COMPONENTS: Tuple[str, ...] = (
    "von_mises",
    "membrane_xx",
    "membrane_yy",
    "membrane_xy",
    "bending_xx",
    "bending_yy",
    "bending_xy",
    "shear_xz",
    "shear_yz",
)

_SURFACE_COMPONENTS: Tuple[str, ...] = (
    "top_xx",
    "top_yy",
    "top_xy",
    "bottom_xx",
    "bottom_yy",
    "bottom_xy",
)

_BEAM_COMPONENTS: Tuple[str, ...] = (
    "axial_stress",
    "bending_stress_y",
    "bending_stress_z",
    "shear_stress_y",
    "shear_stress_z",
    "torsional_stress",
)

STRESS_FIELDS: Tuple[str, ...] = (
    _SHELL_COMPONENTS + _SURFACE_COMPONENTS + _BEAM_COMPONENTS
)

PLASTICITY_FIELDS: Tuple[str, ...] = ("equivalent_plastic_strain",)

_ROTATION_FIELDS = frozenset({"rx", "ry", "rz"})

# How several Gauss-point values become one number for an element.
REDUCTIONS = ("mean", "max_abs", "min", "max")


def available_fields() -> List[str]:
    """Every field name that can be asked for."""

    return list(DISPLACEMENT_FIELDS) + list(STRESS_FIELDS)


def field_unit(name: str) -> str:
    if name in DISPLACEMENT_FIELDS:
        return "" if name in _ROTATION_FIELDS else "m"
    if name in STRESS_FIELDS:
        return "Pa"
    if name in PLASTICITY_FIELDS:
        return "1"
    return ""


@dataclass
class Field:
    """A scalar field over one shape.

    Exactly one of ``node_values`` and ``element_values`` is populated, because
    that is a real difference: a displacement is known at nodes and a stress is
    recovered on elements, and pretending otherwise would invent data.
    """

    name: str
    unit: str = ""
    node_values: Dict[int, float] = dataclass_field(default_factory=dict)
    element_values: Dict[int, float] = dataclass_field(default_factory=dict)
    reduction: str = "mean"
    missing: Tuple[int, ...] = ()

    @property
    def per_element(self) -> bool:
        return bool(self.element_values)

    @property
    def values(self) -> Dict[int, float]:
        return self.element_values if self.per_element else self.node_values

    def array(self) -> np.ndarray:
        return np.array([self.values[key] for key in sorted(self.values)])

    def range(self) -> Tuple[float, float]:
        data = self.array()
        if data.size == 0:
            return 0.0, 1.0
        low, high = float(data.min()), float(data.max())
        return (low, high) if high > low else (low, low + 1.0)

    def extreme(self) -> Tuple[Optional[int], float]:
        """The key with the largest magnitude, and its value."""

        if not self.values:
            return None, 0.0
        key = max(self.values, key=lambda item: abs(self.values[item]))
        return key, float(self.values[key])

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:  # pragma: no cover - display only
        low, high = self.range()
        where = "elements" if self.per_element else "nodes"
        return (
            f"Field({self.name!r}, {len(self)} {where}, "
            f"{low:.4g} to {high:.4g} {self.unit})"
        )


def _reduce(values, reduction):
    from anypost import reduce_samples
    array=np.asarray(values,dtype=float).reshape(1,-1)
    return float(reduce_samples(array,reduction,nan_policy="propagate",empty_policy="zero")[0])


def _surface_value(stress,name):
    side,_,component=name.partition("_")
    if stress.get("membrane_"+component) is None or stress.get("bending_"+component) is None:return None
    from anysolver_results import shell_surface_component
    return shell_surface_component(stress,side,component)


def evaluate_field(
    shape,
    name: str = "magnitude",
    *,
    reduction: str = "mean",
    stresses: Any = None,
) -> Field:
    """Evaluate one named field over a shape.

    ``stresses`` lets a caller reuse a recovery it already has, which matters
    when sweeping every step of a transient.

    A name this module does not label is still evaluated if the solver's
    recovery reports it.  The named lists here exist to give fields a unit and
    a menu entry, not to gate what the solver is allowed to produce: recovery
    grows components over time, and a probe that asks for everything the solver
    knows must not fail merely because this list is a version behind it.
    """

    # A result read from another solver's file carries its stresses already
    # computed. They are looked up before anything else and never recovered:
    # there is no displacement field here complete enough to recover from, and
    # recomputing would silently replace the imported answer with this layer's
    # opinion of it.
    from .access import evaluate_shape_field, _RESULT_ACCESS
    access=_RESULT_ACCESS.get()
    if access is not None and stresses is None:
        return access.field(shape,name,reduction=reduction)
    return evaluate_shape_field(shape, name, reduction=reduction, stresses=stresses)


def reported_fields(stresses: Any, shape: Any) -> List[str]:
    """Every element-stress component this particular result actually carries.

    Taken from the recovery rather than from a list here, so a component the
    solver has recently started reporting shows up without ANYfem being
    changed to know its name.

    Entries whose value is not a number are skipped.  Recovery carries its own
    provenance in the same dictionary -- which equivalent-stress measure was
    used, for instance -- and that is a label about the result, not a field
    over it.
    """

    mesh = shape.built.mesh
    names: List[str] = []
    for element_id in list(mesh.shells) + list(mesh.beams):
        stress = stresses.element_stresses.get(element_id)
        if not stress:
            continue
        for name, value in stress.items():
            if name not in names and _is_numeric(value):
                names.append(name)
    return names


def _is_numeric(value: Any) -> bool:
    """Whether a recovered entry is a number, or an array of them."""

    if isinstance(value, str):
        return False
    try:
        np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return True


def _displacement_field(shape,name):
    from .access import evaluate_shape_field
    return evaluate_shape_field(shape,name)


def _stress_field(shape,name,*,reduction,stresses=None):
    from .access import evaluate_shape_field
    return evaluate_shape_field(shape,name,reduction=reduction,stresses=stresses)


def recover(shape, **kwargs: Any):
    """Recover element stresses for any shape, caching on the shape itself."""

    caller_options = bool(kwargs)
    cached = getattr(shape, "_stress", None)
    if cached is not None and not caller_options:
        return cached

    from anysolver import recover_stress_result

    nonlinear_result = getattr(shape, "raw_result", None)
    if (
        nonlinear_result is not None
        and "nonlinear_result" not in kwargs
        and "element_states" not in kwargs
    ):
        kwargs.setdefault("nonlinear_result", nonlinear_result)
    result = recover_stress_result(
        shape.built.fe_model, shape.displacements, **kwargs
    )
    if not caller_options:
        try:
            shape._stress = result
        except AttributeError:  # pragma: no cover - frozen shapes
            pass
    return result


def _plasticity_field(shape,name):
    from .access import evaluate_shape_field
    return evaluate_shape_field(shape,name)


def element_centroids(mesh) -> Dict[int, np.ndarray]:
    """Centroid of every element, for locating element-based values in space."""

    centroids: Dict[int, np.ndarray] = {}
    for element_id in mesh.shells:
        # Corners only: averaging in the mid-side nodes of a quadratic element
        # would pull the centroid towards whichever sides are longer.
        centroids[element_id] = np.mean(
            [mesh.nodes[node] for node in mesh.corners_of(element_id)], axis=0
        )
    for element_id, nodes in mesh.beams.items():
        centroids[element_id] = np.mean(
            [mesh.nodes[node] for node in nodes], axis=0
        )
    return centroids
