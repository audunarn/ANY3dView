"""Connections: one reference point that stands for a set of lines, points or plates.

A load on a structure is rarely a pair of line loads.  A bending moment is a
single action at the end of a section, and it reaches the plating through
whatever carries it -- a stiff diaphragm, a flange, a crane base.  A
*connection* models that carrier.  It ties a selected set of geometry to one
reference point, normally the geometric centre of the selection, and a load
(force and moment) is then applied at that point instead of on each entity.

Two behaviours are typical in finite-element practice, and both are offered:

``rigid``
    The tied nodes move as one rigid body with the reference point
    (Nastran RBE2, Abaqus kinematic coupling, Ansys rigid remote point).  The
    section is held plane, so it adds stiffness; the reference point's
    displacement and rotation are the displacement and rotation of the
    section.  ``dofs`` chooses which motions are tied: all six is a fully rigid
    body, translations only is a pin-like ball joint, and a single translation
    such as ``ux`` is the classic *plane section stays plane* condition
    (the section may still warp and contract in its own plane).

``distributing``
    The reference point moves with the weighted average motion of the tied
    nodes (Nastran RBE3, Abaqus distributing coupling, Ansys deformable remote
    point).  Nothing is stiffened: the load applied at the reference point is
    spread over the nodes in proportion to their weight and distance from the
    reference point, and the structure is free to deform.  The weight is the
    tributary length or area of each node, so the result does not change with
    the mesh, or one per node.

``spring``
    A flexible link.  The selection is tied to an inner *section* point as
    for a rigid or distributing connection (``tie``), and the loaded outer
    reference point is joined to it by six springs (``stiffness``: three
    translational in N/m, three rotational in N*m/rad, in global axes).  Use it
    for a compliant support or pedestal, a bearing, or a connection whose
    stiffness is known; a very large stiffness approaches the rigid case, and
    the section's own motion is reported next to the reference point's.

Nothing here names a node.  Connections reference geometry like every other
attribute and are resolved against the mesh when the solver model is built.
The equations in this module are linear in the small-rotation sense, which is
what the linear solvers use; they are exact for translations and first-order
for rotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
from uuid import uuid4

import numpy as np

from anygeometry.entities import EntityRef
from anyloads import Envelope

from .regions import RegionRef

__all__ = [
    "CONNECTION_DOF_NAMES",
    "CONNECTION_KINDS",
    "CONNECTION_PRESETS",
    "CONNECTION_REFERENCES",
    "CONNECTION_TIES",
    "CONNECTION_WEIGHTINGS",
    "Connection",
    "ConnectionEquation",
    "ConnectionLoad",
    "distributing_equations",
    "null_moment_directions",
    "preset_dofs",
    "resolve_dofs",
    "resolve_stiffness",
    "rigid_equations",
    "weighted_centroid",
]

CONNECTION_DOF_NAMES: tuple[str, ...] = ("ux", "uy", "uz", "rx", "ry", "rz")
CONNECTION_KINDS: tuple[str, ...] = ("rigid", "distributing", "spring")
CONNECTION_TIES: tuple[str, ...] = ("rigid", "distributing")
CONNECTION_REFERENCES: tuple[str, ...] = ("centroid", "bounding_box", "point")
CONNECTION_WEIGHTINGS: tuple[str, ...] = ("tributary", "uniform")

# Named sets of tied motions.  A preset is only a shorthand for ``dofs``; the
# record keeps the explicit tuple so a file never depends on a preset's meaning.
CONNECTION_PRESETS: dict[str, tuple[str, ...]] = {
    "full": CONNECTION_DOF_NAMES,
    "translations": ("ux", "uy", "uz"),
    "plane_section_x": ("ux",),
    "plane_section_y": ("uy",),
    "plane_section_z": ("uz",),
}

# Inertia eigenvalues below this fraction of the largest are a direction the
# selected geometry cannot transmit a moment about (a moment about the axis of
# a straight line, for one).
_NULL_DIRECTION_TOLERANCE = 1.0e-9


def _uuid() -> str:
    return str(uuid4())


def _finite_vector(values: Sequence[float], what: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{what} needs three finite components") from None
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{what} needs three finite components")
    return vector.copy()


def preset_dofs(name: str) -> tuple[str, ...]:
    """The tied motions a preset stands for."""

    try:
        return CONNECTION_PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown connection preset {name!r}; use one of "
            f"{', '.join(CONNECTION_PRESETS)}"
        ) from None


def resolve_dofs(dofs: str | Sequence[str]) -> tuple[str, ...]:
    """Tied motions from a preset name, a comma list or a sequence of names."""

    if isinstance(dofs, str):
        dofs = [dofs]
    names: list[str] = []
    for item in dofs:
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            names.extend(CONNECTION_PRESETS[part] if part in CONNECTION_PRESETS else (part,))
    return tuple(names)


def resolve_stiffness(stiffness: Sequence[float]) -> tuple[float, ...]:
    """Six spring stiffnesses (ux, uy, uz, rx, ry, rz) from 2 or 6 numbers.

    Two numbers are ``(translational, rotational)`` applied to all three
    translations and all three rotations.  Zero means "no spring": that motion
    of the loaded point is not carried.
    """

    if isinstance(stiffness, str):
        stiffness = [part for part in stiffness.replace(",", " ").split()]
    try:
        values = [float(value) for value in stiffness]
    except (TypeError, ValueError):
        raise ValueError("spring stiffness must be numbers") from None
    if len(values) == 2:
        values = [values[0]] * 3 + [values[1]] * 3
    if len(values) != 6:
        raise ValueError(
            "spring stiffness is six numbers (ux, uy, uz, rx, ry, rz) or two "
            "(translational, rotational)"
        )
    if not all(np.isfinite(values)) or any(value < 0.0 for value in values):
        raise ValueError("spring stiffness must be finite and not negative")
    if not any(value > 0.0 for value in values):
        raise ValueError("a spring connection needs at least one positive stiffness")
    return tuple(values)


@dataclass(frozen=True)
class Connection:
    """A reference point tied to the selected geometry.

    ``ref`` is the first selected entity and ``region`` holds the whole
    selection, exactly as for a support or a load.  ``dofs`` are the motions
    that are tied: for a rigid connection the listed degrees of freedom of
    every selected node follow the reference point; for a distributing
    connection they are the degrees of freedom of the reference point that
    follow the selected nodes.  All components are in global axes.

    ``reference`` places the point: ``centroid`` is the length-weighted centre
    of the selected lines (area-weighted for plates, plain average for
    points), ``bounding_box`` the centre of their bounding box, and ``point`` a
    given position in metres.
    """

    ref: EntityRef
    name: str = "connection"
    region: RegionRef | None = None
    kind: str = "rigid"
    dofs: tuple[str, ...] = CONNECTION_DOF_NAMES
    reference: str = "centroid"
    reference_point: tuple[float, float, float] | None = None
    weighting: str = "tributary"
    id: str = field(default_factory=_uuid)
    # Spring connections only: stiffness of the six springs between the section
    # point and the loaded point, and how the selection is tied to the former.
    stiffness: tuple[float, ...] | None = None
    tie: str = "rigid"

    def __post_init__(self) -> None:
        if self.ref.kind not in ("vertex", "edge", "face"):
            raise ValueError("a connection ties points, lines or plates")
        if self.kind not in CONNECTION_KINDS:
            raise ValueError(
                f"connection kind must be one of {', '.join(CONNECTION_KINDS)}"
            )
        dofs = resolve_dofs(self.dofs)
        unknown = [item for item in dofs if item not in CONNECTION_DOF_NAMES]
        if unknown or not dofs:
            raise ValueError(
                "connection dofs must be a non-empty subset of "
                f"{', '.join(CONNECTION_DOF_NAMES)}"
            )
        # Canonical order, no repeats: equal connections compare and hash equal.
        object.__setattr__(
            self, "dofs", tuple(name for name in CONNECTION_DOF_NAMES if name in dofs)
        )
        if self.reference not in CONNECTION_REFERENCES:
            raise ValueError(
                "connection reference must be one of "
                f"{', '.join(CONNECTION_REFERENCES)}"
            )
        if self.reference == "point":
            if self.reference_point is None:
                raise ValueError("a connection with reference 'point' needs reference_point")
            # A tuple, not an array: the record is frozen and must compare and hash.
            object.__setattr__(
                self,
                "reference_point",
                tuple(
                    float(value)
                    for value in _finite_vector(
                        self.reference_point, "connection reference point"
                    )
                ),
            )
        elif self.reference_point is not None:
            raise ValueError(
                "reference_point is only used when the reference is 'point'"
            )
        if self.weighting not in CONNECTION_WEIGHTINGS:
            raise ValueError(
                "connection weighting must be one of "
                f"{', '.join(CONNECTION_WEIGHTINGS)}"
            )
        if self.kind == "spring":
            if self.stiffness is None:
                raise ValueError("a spring connection needs stiffness")
            object.__setattr__(self, "stiffness", resolve_stiffness(self.stiffness))
            if self.tie not in CONNECTION_TIES:
                raise ValueError(
                    f"spring tie must be one of {', '.join(CONNECTION_TIES)}"
                )
        else:
            if self.stiffness is not None:
                raise ValueError("stiffness is only used by a spring connection")
            if self.tie != "rigid":
                raise ValueError("tie is only used by a spring connection")
        if self.tied_by == "distributing" and not any(
            name.startswith("u") for name in self.dofs
        ):
            # A reference point that follows only rotations has no position the
            # fit can place it at, and a force there would have nothing to push.
            raise ValueError(
                "a distributing connection must follow at least one translation"
            )
        if not str(self.name).strip():
            raise ValueError("a connection needs a name")

    @property
    def tied_by(self) -> str:
        """How the selection is tied to its (section) point."""

        return self.tie if self.kind == "spring" else self.kind

    @property
    def translations(self) -> tuple[int, ...]:
        return tuple(index for index, name in enumerate(CONNECTION_DOF_NAMES[:3]) if name in self.dofs)

    @property
    def rotations(self) -> tuple[int, ...]:
        return tuple(index for index, name in enumerate(CONNECTION_DOF_NAMES[3:]) if name in self.dofs)


@dataclass(frozen=True)
class ConnectionLoad:
    """A force and moment applied at a connection's reference point.

    ``connection`` is the connection's id.  The moment is about the reference
    point, in newton metres, in global axes; the force acts at the reference
    point, so a force away from the centre of the selection also loads it with
    the moment of that offset.
    """

    connection: str
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    moment: np.ndarray = field(default_factory=lambda: np.zeros(3))
    name: str = "connection load"
    id: str = field(default_factory=_uuid)
    # When the load acts and how strongly over a time history; None = always.
    envelope: Envelope | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "envelope", Envelope.coerce(self.envelope))
        if not str(self.connection).strip():
            raise ValueError("a connection load names its connection")
        object.__setattr__(
            self, "force", _finite_vector(self.force, "connection-load force")
        )
        object.__setattr__(
            self, "moment", _finite_vector(self.moment, "connection-load moment")
        )


# ----------------------------------------------------------------------
# the equations
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ConnectionEquation:
    """One affine equation ``sum(coefficient * dof) = 0`` with a dependent DOF.

    DOFs are named ``(owner, index)`` where ``owner`` is a node id or the
    string ``"ref"`` for the reference point, and ``index`` is 0-5 in the order
    ux, uy, uz, rx, ry, rz.  The dependent term is first in ``terms``.
    """

    terms: tuple[tuple[object, int, float], ...]

    @property
    def dependent(self) -> tuple[object, int]:
        owner, index, _ = self.terms[0]
        return owner, index


REF = "ref"


def _cross_coefficients(offset: np.ndarray) -> np.ndarray:
    """Matrix ``C`` with ``C @ theta == theta x offset`` (small rotation)."""

    x, y, z = offset
    return np.array([[0.0, z, -y], [-z, 0.0, x], [y, -x, 0.0]])


def _trim(value: float, scale: float) -> bool:
    return abs(value) > 1.0e-14 * max(scale, 1.0e-300)


def weighted_centroid(
    positions: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    if weights is None:
        return positions.mean(axis=0)
    weights = np.asarray(weights, dtype=float)
    return (weights[:, None] * positions).sum(axis=0) / weights.sum()


def rigid_equations(
    node_ids: Sequence[int],
    positions: np.ndarray,
    reference: np.ndarray,
    dofs: Sequence[str],
) -> list[ConnectionEquation]:
    """Tie each node to the reference point as a rigid body.

    ``u_node = u_ref + theta_ref x r`` for each translation in ``dofs`` and
    ``theta_node = theta_ref`` for each rotation, with ``r = x_node - x_ref``.
    The node DOF is the dependent one, so a node DOF may appear in one rigid
    connection only.
    """

    positions = np.asarray(positions, dtype=float)
    reference = np.asarray(reference, dtype=float)
    scale = float(np.max(np.abs(positions - reference))) if len(positions) else 1.0
    equations: list[ConnectionEquation] = []
    for node, position in zip(node_ids, positions):
        coupling = _cross_coefficients(position - reference)
        for index, name in enumerate(CONNECTION_DOF_NAMES):
            if name not in dofs:
                continue
            terms: list[tuple[object, int, float]] = [
                (int(node), index, 1.0),
                (REF, index, -1.0),
            ]
            if index < 3:
                for rotation in range(3):
                    value = -float(coupling[index, rotation])
                    if _trim(value, scale):
                        terms.append((REF, 3 + rotation, value))
            equations.append(ConnectionEquation(tuple(terms)))
    return equations


def distributing_equations(
    node_ids: Sequence[int],
    positions: np.ndarray,
    weights: np.ndarray,
    reference: np.ndarray,
    dofs: Sequence[str],
) -> tuple[list[ConnectionEquation], np.ndarray]:
    """Make the reference point follow the weighted fit of the node motion.

    The fit is the least-squares rigid motion of the nodes,
    ``min sum w_i |u_i - u_ref - theta x r_i|^2``, restricted to the rotation
    components in ``dofs``.  Its virtual work gives the load distribution: a
    force ``F`` and moment ``M`` at the reference point become nodal forces
    that sum to ``F`` and carry the moment ``M`` about the point, each
    proportional to the node's weight (a linear distribution across the
    selection, like a beam section).

    Returns the equations and the matrix ``I`` of the rotation fit (rows and
    columns in ``dofs`` order) so a caller can tell which moments the selection
    can carry: a moment about the axis of a straight line, say, cannot be
    carried by forces and shows as a null direction of ``I``.
    """

    positions = np.asarray(positions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(node_ids) == 0 or weights.sum() <= 0.0 or np.any(weights < 0.0):
        raise ValueError("a distributing connection needs nodes with positive weight")
    total = float(weights.sum())
    centre = (weights[:, None] * positions).sum(axis=0) / total
    relative = positions - centre
    rotations = [index for index, name in enumerate(CONNECTION_DOF_NAMES[3:]) if name in dofs]
    translations = [index for index, name in enumerate(CONNECTION_DOF_NAMES[:3]) if name in dofs]
    equations: list[ConnectionEquation] = []

    # theta = I^+ b with b = sum w (r' x u): the coefficient of u_i,b in
    # b_k is w_i * eps(k, a, b) * r'_ia.
    inertia = np.zeros((len(rotations), len(rotations)))
    gain = np.zeros((len(rotations), len(node_ids), 3))
    if rotations:
        full = np.zeros((3, 3))
        for weight, r in zip(weights, relative):
            full += weight * (float(r @ r) * np.eye(3) - np.outer(r, r))
        inertia = full[np.ix_(rotations, rotations)]
        pseudo = _pseudo_inverse(inertia)
        for node_index, (weight, r) in enumerate(zip(weights, relative)):
            # b = sum w (r' x u) and r' x u = -(u x r') = -C(r') u, so row kk of
            # b has the coefficient -w * C(r')[kk] on u_i.
            cross = _cross_coefficients(r)
            for local in range(len(rotations)):
                gain[local, node_index, :] = sum(
                    pseudo[local, other] * (-weight) * cross[kk, :]
                    for other, kk in enumerate(rotations)
                )
        gain_scale = float(np.max(np.abs(gain))) if gain.size else 0.0
        for local, k in enumerate(rotations):
            terms: list[tuple[object, int, float]] = [(REF, 3 + k, 1.0)]
            for node_index, node in enumerate(node_ids):
                for component in range(3):
                    value = -float(gain[local, node_index, component])
                    if gain_scale > 0.0 and _trim(value, gain_scale):
                        terms.append((int(node), component, value))
            equations.append(ConnectionEquation(tuple(terms)))

    # u_ref = mean(u) - theta x (centre - reference): the reference point need
    # not be the weighted centroid, and the rotation moves it.
    offset = centre - reference
    coupling = _cross_coefficients(offset)
    off_scale = float(np.max(np.abs(offset))) if offset.size else 0.0
    for k in translations:
        terms = [(REF, k, 1.0)]
        for node, weight in zip(node_ids, weights):
            terms.append((int(node), k, -float(weight) / total))
        if rotations:
            for rotation in rotations:
                value = float(coupling[k, rotation])
                if off_scale > 0.0 and _trim(value, off_scale):
                    terms.append((REF, 3 + rotation, value))
        equations.append(ConnectionEquation(tuple(terms)))
    return equations, inertia


def _pseudo_inverse(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    values, vectors = np.linalg.eigh(matrix)
    limit = _NULL_DIRECTION_TOLERANCE * max(float(np.max(np.abs(values))), 1.0e-300)
    inverse = np.where(values > limit, 1.0 / np.where(values > limit, values, 1.0), 0.0)
    return (vectors * inverse) @ vectors.T


def null_moment_directions(inertia: np.ndarray) -> np.ndarray:
    """Unit vectors (columns) of moments the fit cannot carry."""

    if inertia.size == 0:
        return np.zeros((0, 0))
    values, vectors = np.linalg.eigh(inertia)
    limit = _NULL_DIRECTION_TOLERANCE * max(float(np.max(np.abs(values))), 1.0e-300)
    return vectors[:, values <= limit]
