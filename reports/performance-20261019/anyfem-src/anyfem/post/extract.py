"""Pulling numbers out of a result: probes, paths and envelopes.

Everything here is expressed in terms of :class:`~anyfem.post.fields.Field`, so
a probe, a path plot and an envelope all mean the same thing by "von Mises" as
the contour does.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from anygeometry.entities import EntityRef
from .fields import Field, _is_numeric, evaluate_field, field_unit, recover

__all__ = [
    "Envelope",
    "PathResult",
    "Probe",
    "along_line",
    "envelope",
    "nodes_to_elements",
    "probe",
]

_DOF_NAMES = ("ux", "uy", "uz", "rx", "ry", "rz")


def nodes_to_elements(mesh) -> Dict[int, List[int]]:
    """Which elements touch each node."""

    attached: Dict[int, List[int]] = {}
    for element_id, nodes in list(mesh.shells.items()) + list(mesh.beams.items()):
        for node_id in nodes:
            attached.setdefault(node_id, []).append(element_id)
    return attached


# ----------------------------------------------------------------------
# probes
# ----------------------------------------------------------------------
@dataclass
class Probe:
    """Every component at one place, for a readout."""

    ref: EntityRef
    node_id: Optional[int] = None
    displacement: Dict[str, float] = dataclass_field(default_factory=dict)
    stresses: Dict[str, float] = dataclass_field(default_factory=dict)
    elements: Tuple[int, ...] = ()

    def text(
        self,
        *,
        length_scale: float = 1.0,
        length_unit: str = "m",
        stress_scale: float = 1.0e-6,
        stress_unit: str = "MPa",
    ) -> str:
        lines = [f"{self.ref}"]
        if self.node_id is not None:
            lines.append(f"  node {self.node_id}")
        if self.displacement:
            lines.append("  displacement")
            for name in _DOF_NAMES:
                if name in self.displacement:
                    unit = length_unit if name.startswith("u") else "rad"
                    scale = length_scale if name.startswith("u") else 1.0
                    lines.append(
                        f"    {name:<4} {self.displacement[name] * scale: .6g} {unit}"
                    )
        if self.stresses:
            lines.append(
                f"  stress over {len(self.elements)} element(s) [{stress_unit}]"
            )
            for name, value in self.stresses.items():
                lines.append(f"    {name:<16} {value * stress_scale: .5g}")
        return "\n".join(lines)


def probe(
    shape,
    ref: EntityRef,
    *,
    reduction: str = "max_abs",
    components: Optional[Sequence[str]] = None,
) -> Probe:
    """Read every component at one geometry entity.

    For a point that is the node's displacement plus the stress in the
    elements around it.  For a line or a plate it is the extreme over
    everything on that entity, because a single number for a whole plate can
    only honestly be an extreme.
    """

    mesh = shape.built.mesh
    node_ids = mesh.nodes_on(ref)
    if not node_ids:
        raise KeyError(f"{ref} has no nodes in the mesh")

    result = Probe(ref=ref, node_id=node_ids[0] if ref.kind == "vertex" else None)

    if ref.kind == "vertex":
        values = shape.node_displacement(node_ids[0])
        result.displacement = {
            name: float(value) for name, value in zip(_DOF_NAMES, values)
        }
    else:
        stacked = np.array(
            [shape.node_displacement(node_id) for node_id in node_ids]
        )
        peak = stacked[np.argmax(np.abs(stacked), axis=0), range(6)]
        result.displacement = {
            name: float(value) for name, value in zip(_DOF_NAMES, peak)
        }

    if ref.kind == "face":
        elements = mesh.elements_on(ref)
    elif ref.kind == "edge":
        elements = mesh.elements_on(ref) or _elements_touching(mesh, node_ids)
    else:
        elements = _elements_touching(mesh, node_ids)
    result.elements = tuple(elements)

    if elements:
        recovered = recover(shape)
        wanted = components or _probe_components(recovered, elements)
        for name in wanted:
            field = evaluate_field(
                shape, name, reduction=reduction, stresses=recovered
            )
            local = [
                field.element_values[element_id]
                for element_id in elements
                if element_id in field.element_values
            ]
            if local:
                result.stresses[name] = float(
                    max(local, key=abs) if reduction == "max_abs" else np.mean(local)
                )
    return result


def _elements_touching(mesh, node_ids: Sequence[int]) -> List[int]:
    attached = nodes_to_elements(mesh)
    found: List[int] = []
    for node_id in node_ids:
        for element_id in attached.get(node_id, ()):
            if element_id not in found:
                found.append(element_id)
    return found


def _probe_components(recovered, elements: Sequence[int]) -> List[str]:
    """Whichever components the elements at this place actually carry.

    Recovery mixes numeric components with its own provenance labels in one
    dictionary, so non-numeric entries are skipped: a probe reports fields, and
    the name of the equivalent-stress measure is not one.
    """

    names: List[str] = []
    for element_id in elements:
        stress = recovered.element_stresses.get(element_id)
        if not stress:
            continue
        for name, value in stress.items():
            if name not in names and _is_numeric(value):
                names.append(name)
    return names


# ----------------------------------------------------------------------
# along a line
# ----------------------------------------------------------------------
@dataclass
class PathResult:
    """A field sampled along a line, for a path plot."""

    ref: EntityRef
    name: str
    unit: str
    distances: np.ndarray
    values: np.ndarray
    positions: np.ndarray

    def __len__(self) -> int:
        return len(self.distances)

    @property
    def length(self) -> float:
        return 0.0 if len(self.distances) == 0 else float(self.distances[-1])

    def to_csv(self) -> str:
        header = f"distance_m,{self.name}" + (f"_{self.unit}" if self.unit else "")
        rows = [
            f"{distance:.9g},{value:.9g}"
            for distance, value in zip(self.distances, self.values)
        ]
        return "\n".join([header, *rows])


def along_line(
    shape,
    ref: EntityRef,
    name: str = "magnitude",
    *,
    reduction: str = "mean",
) -> PathResult:
    """Sample a field along a line, node by node.

    A per-element field has no value *at* a node, so each node takes the
    average of the elements meeting it there.  That is a stated choice, not a
    recovered nodal stress.
    """

    if ref.kind != "edge":
        raise ValueError("along_line expects a line reference")

    mesh = shape.built.mesh
    node_ids = mesh.nodes_on(ref)
    if not node_ids:
        raise KeyError(f"{ref} has no nodes in the mesh")

    positions = np.array([mesh.nodes[node_id] for node_id in node_ids])
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    distances = np.concatenate(([0.0], np.cumsum(steps)))

    field = evaluate_field(shape, name, reduction=reduction)
    if field.per_element:
        attached = nodes_to_elements(mesh)
        on_line = set(mesh.elements_on(ref))
        values = []
        for node_id in node_ids:
            candidates = [
                element_id
                for element_id in attached.get(node_id, ())
                if element_id in field.element_values
                and (not on_line or element_id in on_line)
            ]
            if not candidates:
                candidates = [
                    element_id
                    for element_id in attached.get(node_id, ())
                    if element_id in field.element_values
                ]
            values.append(
                float(np.mean([field.element_values[e] for e in candidates]))
                if candidates
                else 0.0
            )
        values = np.array(values)
    else:
        values = np.array([field.node_values[node_id] for node_id in node_ids])

    return PathResult(
        ref=ref,
        name=name,
        unit=field.unit,
        distances=distances,
        values=values,
        positions=positions,
    )


# ----------------------------------------------------------------------
# envelopes
# ----------------------------------------------------------------------
@dataclass
class Envelope:
    """The extreme of a field over every shape of a result."""

    field: Field
    mode: str
    shape_of: Dict[int, int] = dataclass_field(default_factory=dict)
    shape_count: int = 0
    admitted_case_ids: tuple[str,...] = ()
    failed_case_ids: tuple[str,...] = ()

    @property
    def admitted_count(self):return len(self.admitted_case_ids)

    @property
    def failed_count(self):return len(self.failed_case_ids)

    @property
    def name(self) -> str:
        return self.field.name

    def worst_shape(self) -> Optional[int]:
        """Which shape index produced the overall extreme."""

        key, _value = self.field.extreme()
        return None if key is None else self.shape_of.get(key)


def envelope(
    solution,
    name: str = "magnitude",
    *,
    mode: str = "max_abs",
    reduction: str = "mean",
) -> Envelope:
    """Combine a field across every shape of a modal or transient result.

    ``mode`` is how the shapes combine: ``max_abs`` for a worst-case
    magnitude, ``max`` or ``min`` for a signed bound.
    """

    if mode not in ("max_abs", "max", "min"):
        raise ValueError(
            f"unknown envelope mode {mode!r}; expected max_abs, max or min"
        )

    from .access import evaluate_envelope
    return evaluate_envelope(solution, name, mode=mode, reduction=reduction)


def _is_worse(candidate: float, current: float, mode: str) -> bool:
    if mode == "max_abs":
        return abs(candidate) > abs(current)
    if mode == "max":
        return candidate > current
    if mode == "min":
        return candidate < current
    raise ValueError(
        f"unknown envelope mode {mode!r}; expected max_abs, max or min"
    )
