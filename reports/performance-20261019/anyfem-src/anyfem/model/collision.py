"""A rigid sphere thrown at the structure.

Impact is the one analysis where the two things users most often get wrong are
not the model but the *settings*: the contact penalty stiffness and the time
step.  Both have a defensible automatic value, so ANYfem computes them by
default and says what it used, rather than leaving a field at zero and
producing a plausible-looking answer from a badly conditioned contact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Collision",
    "fracture",
    "CollisionTiming",
    "auto_timing",
    "impact_damage",
    "impact_point",
    "impact_refinement",
]


@dataclass(frozen=True)
class Collision:
    """A rigid sphere with a mass, a starting point and a velocity."""

    name: str = "impact"
    mass: float = 1000.0
    radius: float = 0.25
    start: Sequence[float] = (0.0, 0.0, 1.0)
    direction: Sequence[float] = (0.0, 0.0, -1.0)
    speed: float = 5.0
    t_start: float = 0.0

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError(f"collision {self.name!r}: mass must be positive")
        if self.radius <= 0.0:
            raise ValueError(f"collision {self.name!r}: radius must be positive")
        if self.speed <= 0.0:
            raise ValueError(f"collision {self.name!r}: speed must be positive")
        if float(np.linalg.norm(np.asarray(self.direction, dtype=float))) <= 0.0:
            raise ValueError(
                f"collision {self.name!r}: travel direction must be non-zero"
            )

    @property
    def unit_direction(self) -> np.ndarray:
        direction = np.asarray(self.direction, dtype=float)
        return direction / float(np.linalg.norm(direction))

    @property
    def kinetic_energy(self) -> float:
        """The energy the sphere arrives with, in joules."""

        return 0.5 * float(self.mass) * float(self.speed) ** 2

    @property
    def momentum(self) -> np.ndarray:
        return float(self.mass) * float(self.speed) * self.unit_direction

    def to_solver(self):
        """The solver's own sphere description."""

        from anysolver import RigidSphereImpact

        return RigidSphereImpact(
            name=self.name,
            radius=float(self.radius),
            mass=float(self.mass),
            start_point=np.asarray(self.start, dtype=float),
            travel_direction=self.unit_direction,
            speed=float(self.speed),
            t_start=float(self.t_start),
        )

    def summary(self) -> str:
        return (
            f"{self.name}: {self.mass:g} kg sphere, r = {self.radius:g} m, "
            f"{self.speed:g} m/s, {self.kinetic_energy / 1000.0:.4g} kJ"
        )


@dataclass(frozen=True)
class CollisionTiming:
    """A time step and duration chosen for one sphere and one structure."""

    dt: float
    t_end: float
    time_to_contact: float
    gap: float
    steps: int
    contact_period: Optional[float] = None
    start: Optional[Tuple[float, float, float]] = None
    notes: Tuple[str, ...] = ()

    def summary(self) -> str:
        period = (
            ""
            if self.contact_period is None
            else f", contact period {self.contact_period:.3g} s"
        )
        extra = "" if not self.notes else " | " + "; ".join(self.notes)
        return (
            f"dt = {self.dt:.4g} s, duration {self.t_end:.4g} s "
            f"({self.steps} steps), contact at about "
            f"{self.time_to_contact:.4g} s{period}{extra}"
        )


def _reachable_nodes(mesh, collision: Collision):
    """Nodes the sphere's swept cylinder passes over, and how far it travels.

    ``travel`` is how far the sphere's *centre* moves before its surface
    touches each node: ``along - sqrt(r^2 - lateral^2)``.  Using the axial
    distance alone would tie every node on a flat plate -- they are all the same
    distance downrange -- and the tie would then be broken by node numbering.
    The sphere's surface is curved, so the node nearest the axis is struck
    first, and this expression says so.

    Shared by the timing and the refinement, so "does it hit, and where" has
    one definition.  Two answers that could disagree about whether a sphere
    strikes the structure would be worse than either alone.
    """

    positions = mesh.node_positions()
    if positions.size == 0:
        raise ValueError("the model has no nodes for the sphere to strike")

    start = np.asarray(collision.start, dtype=float)
    direction = collision.unit_direction
    offsets = positions - start
    along = offsets @ direction
    lateral = np.linalg.norm(offsets - np.outer(along, direction), axis=1)

    reachable = (lateral <= collision.radius) & (along > 0.0)
    if not np.any(reachable):
        nearest = float(lateral.min())
        raise ValueError(
            f"collision {collision.name!r} misses the structure: travelling "
            f"from {tuple(np.round(start, 4))} the sphere passes no closer "
            f"than {nearest:.4g} m to any node, and its radius is "
            f"{collision.radius:g} m. Check the start point and direction."
        )

    reach = np.sqrt(
        np.maximum(collision.radius**2 - np.minimum(lateral, collision.radius) ** 2, 0.0)
    )
    travel = along - reach
    return positions, along, travel, reachable


def impact_point(mesh, collision: Collision) -> np.ndarray:
    """Where the sphere first touches, from a mesh of the structure.

    Reported as a position rather than a node, so the caller can refine around
    it without depending on a node number that the re-mesh will change.
    """

    positions, _along, travel, reachable = _reachable_nodes(mesh, collision)
    indices = np.flatnonzero(reachable)
    return positions[indices[int(np.argmin(travel[indices]))]].copy()


def impact_refinement(
    mesh,
    collision: Collision,
    *,
    elements_per_radius: float = 4.0,
    zone_radii: float = 1.5,
    growth: float = 1.15,
):
    """A refinement zone covering the contact patch.

    A rigid sphere loads a patch about its own radius across, so a mesh whose
    elements are that size cannot resolve the contact at all -- the peak force
    and the local damage both come out of how the patch is discretised.  This
    asks for ``elements_per_radius`` elements across the sphere radius, over a
    zone ``zone_radii`` sphere radii wide, and lets the size grow back to the
    global target outside it.  The 1.15 default stays below the structured
    mesher's 1.25 adjacent-element quality limit after integer edge seeding.

    Needs a mesh only to find where the sphere lands; that mesh can be as
    coarse as the model's own target size, since the contact point comes from
    node positions rather than from any element.
    """

    from ..mesh.refinement import refine_at

    return refine_at(
        impact_point(mesh, collision),
        size=float(collision.radius) / float(elements_per_radius),
        radius=float(zone_radii) * float(collision.radius),
        growth=float(growth),
        name=f"{collision.name}_zone",
    )


def auto_timing(
    mesh,
    collision: Collision,
    *,
    penalty_stiffness: Optional[float] = None,
    steps_per_contact: float = 20.0,
    steps_per_radius: float = 20.0,
    post_contact_periods: float = 20.0,
    skip_approach: bool = True,
    standoff_radii: float = 0.05,
    max_steps: int = 20_000,
    wave_speed: Optional[float] = None,
    min_element_size: Optional[float] = None,
) -> CollisionTiming:
    """Choose a time step and duration for an impact.

    Three things set the step, and the smallest wins:

    * the **contact period** ``2 pi sqrt(m/k)``, resolved into
      ``steps_per_contact`` increments.  This is the one that matters most: a
      step near the contact period does not merely lose accuracy, the contact
      iteration fails to converge and the run reports a peak force that is
      nonsense;
    * the **travel per step**, so the sphere cannot pass through the structure
      between two steps;
    * the **wave transit time of the smallest element at the contact**,
      ``h / c``, when the caller supplies both.  This one binds only on a
      locally refined mesh, and it is why it exists: refining under the sphere
      raises the local frequencies without changing the contact period, so a
      step chosen from the contact period alone becomes too coarse for the mesh
      and the contact iteration diverges.  Bounding by transit time lets a
      refined mesh run at the penalty the solver recommended, instead of
      needing a softer contact -- which would change the answer rather than
      resolve it.

    The duration covers the approach plus ``post_contact_periods`` contact
    periods, which is the impact event and its immediate rebound.  Pass an
    explicit ``t_end`` to watch more of the structural response afterwards.

    ``skip_approach`` moves the sphere to just clear of the structure.  Free
    flight is exact -- constant velocity, no forces -- so integrating it costs
    steps and yields nothing.  The move is reported in ``notes`` rather than
    made silently.

    Refuses when the sphere would miss.  A sphere aimed past the structure
    otherwise runs to completion and reports a perfectly clean nothing.
    """

    start = np.asarray(collision.start, dtype=float)
    direction = collision.unit_direction
    _positions, _along, travel, reachable = _reachable_nodes(mesh, collision)

    # How far the centre moves before the surface touches anything.
    gap = max(float(travel[reachable].min()), 0.0)
    notes: list = []

    moved_start = None
    if skip_approach and gap > standoff_radii * collision.radius:
        standoff = standoff_radii * collision.radius
        moved_start = start + (gap - standoff) * direction
        notes.append(
            f"moved the sphere {gap - standoff:.4g} m forward to a "
            f"{standoff:.4g} m standoff (free flight is exact)"
        )
        gap = standoff

    time_to_contact = gap / collision.speed

    contact_period = None
    dt = collision.radius / (collision.speed * float(steps_per_radius))
    if penalty_stiffness:
        contact_period = 2.0 * np.pi * np.sqrt(
            float(collision.mass) / float(penalty_stiffness)
        )
        dt = min(dt, contact_period / float(steps_per_contact))

    if wave_speed and min_element_size:
        transit = float(min_element_size) / float(wave_speed)
        if transit < dt:
            notes.append(
                f"time step cut to {transit:.4g} s so a stress wave takes at "
                f"least one step to cross the smallest element at the contact "
                f"({min_element_size:.4g} m); the contact period alone would "
                f"have allowed {dt:.4g} s"
            )
            dt = transit

    window = (
        post_contact_periods * contact_period
        if contact_period is not None
        else post_contact_periods * dt * steps_per_contact
    )
    t_end = time_to_contact + window
    steps = int(np.ceil(t_end / dt))

    if steps > max_steps:
        # Shorten the window rather than coarsen the step: a coarser step
        # breaks the contact, a shorter window merely shows less of the
        # aftermath, and the difference is worth being explicit about.
        t_end = max_steps * dt
        steps = max_steps
        notes.append(
            f"duration shortened to {t_end:.4g} s to stay within "
            f"{max_steps} steps; the time step was kept so the contact stays "
            "resolved"
        )

    return CollisionTiming(
        dt=float(dt),
        t_end=float(t_end),
        time_to_contact=float(time_to_contact),
        gap=float(gap),
        steps=int(steps),
        contact_period=None if contact_period is None else float(contact_period),
        start=None if moved_start is None else tuple(float(v) for v in moved_start),
        notes=tuple(notes),
    )


def impact_damage(
    mode: str = "accumulated_damage",
    *,
    capacity_basis: str = "yield",
    delete_at: float = 1.0,
    softening_start: float = 0.6,
    max_deleted_fraction: float = 0.25,
    **options: Any,
):
    """The solver's impact damage configuration, with its own defaults.

    Passed straight through rather than re-wrapped: damage measures are the
    solver's to define, and a second vocabulary here would be one more place
    for the two to drift apart.
    """

    from anysolver import ImpactDamageConfig

    return ImpactDamageConfig(
        mode=mode,
        capacity_basis=capacity_basis,
        delete_at=delete_at,
        softening_start=softening_start,
        max_deleted_fraction=max_deleted_fraction,
        **options,
    )


def fracture(threshold: float, **options: Any):
    """The solver's fracture configuration for a nonlinear static solve.

    Erosion means residual stiffness scaling after a converged increment.  It
    is an engineering screen, not crack mechanics, and the solver says so;
    this wrapper does not upgrade the claim.
    """

    from anysolver import FractureConfig

    return FractureConfig(threshold=float(threshold), **options)
