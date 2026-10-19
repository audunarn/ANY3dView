"""Analytical verification, as dated evidence.

Every case states what it is checked against and how close it has to be, and
the runner records what it actually got.  That turns "verified" into something
a reader can audit instead of something the authors assert.

What this is *not*: a claim that ANYfem is correct for every geometry and load.
These are closed-form comparisons on simple cases.  The solver's own
qualification -- element validity, plasticity, recovery policy, external
CalculiX comparison -- lives in ANYsolver and is not restated here.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "VerificationCase",
    "VerificationReport",
    "VerificationResult",
    "cases",
    "run_verification",
    "write_verification_report",
]

MODULUS = 210.0e9
POISSON = 0.3
DENSITY = 7850.0
GRAVITY = 9.81


@dataclass
class VerificationResult:
    """One case's outcome."""

    case_id: str
    title: str
    reference: str
    computed: float
    expected: float
    tolerance: float
    unit: str = ""
    detail: str = ""
    error: Optional[str] = None

    @property
    def relative_error(self) -> float:
        if self.expected == 0.0:
            return abs(self.computed)
        return abs(self.computed - self.expected) / abs(self.expected)

    @property
    def passed(self) -> bool:
        return self.error is None and self.relative_error <= self.tolerance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "reference": self.reference,
            "computed": self.computed,
            "expected": self.expected,
            "relative_error": self.relative_error,
            "tolerance": self.tolerance,
            "unit": self.unit,
            "status": "passed" if self.passed else "failed",
            "detail": self.detail,
            "error": self.error,
        }


@dataclass
class VerificationCase:
    """A closed-form comparison."""

    case_id: str
    title: str
    reference: str
    tolerance: float
    unit: str
    run: Callable[[], tuple]

    def evaluate(self) -> VerificationResult:
        try:
            computed, expected, detail = self.run()
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            return VerificationResult(
                case_id=self.case_id,
                title=self.title,
                reference=self.reference,
                computed=float("nan"),
                expected=float("nan"),
                tolerance=self.tolerance,
                unit=self.unit,
                error=f"{type(error).__name__}: {error}",
            )
        return VerificationResult(
            case_id=self.case_id,
            title=self.title,
            reference=self.reference,
            computed=float(computed),
            expected=float(expected),
            tolerance=self.tolerance,
            unit=self.unit,
            detail=detail,
        )


@dataclass
class VerificationReport:
    """Everything a run produced."""

    results: List[VerificationResult] = field(default_factory=list)
    generated: str = ""
    environment: Dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def counts(self) -> Dict[str, int]:
        passed = sum(1 for result in self.results if result.passed)
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": len(self.results) - passed,
        }

    def summary(self) -> str:
        counts = self.counts
        state = "PASSED" if self.passed else "FAILED"
        return (
            f"ANYfem verification {state}: "
            f"{counts['passed']}/{counts['total']} cases"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated": self.generated,
            "environment": self.environment,
            "counts": self.counts,
            "status": "passed" if self.passed else "failed",
            "results": [result.to_dict() for result in self.results],
        }

    def to_markdown(self) -> str:
        lines = [
            "# ANYfem verification",
            "",
            f"- generated: {self.generated}",
            f"- status: **{'passed' if self.passed else 'failed'}**",
            f"- cases: {self.counts['passed']} passed, {self.counts['failed']} failed",
        ]
        for key, value in sorted(self.environment.items()):
            lines.append(f"- {key}: {value}")
        lines += [
            "",
            "| case | checked against | computed | expected | error | tol | status |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for result in self.results:
            if result.error:
                lines.append(
                    f"| {result.case_id} | {result.reference} | - | - | - | "
                    f"{result.tolerance:.1%} | error: {result.error} |"
                )
                continue
            lines.append(
                f"| {result.case_id} | {result.reference} | "
                f"{result.computed:.6g} | {result.expected:.6g} | "
                f"{result.relative_error:.2%} | {result.tolerance:.1%} | "
                f"{'pass' if result.passed else 'FAIL'} |"
            )
        lines += [
            "",
            "Most of these are closed-form comparisons on simple cases. A few "
            "are not, and say so in their reference column: SYMM-01 and "
            "INTR-01 check a model against the same model solved without a "
            "simplification, and ELEM-01 against a converged finite element "
            "answer rather than a series solution. They do not claim ANYfem "
            "is correct for every geometry or load regime, and they do not "
            "restate ANYsolver's own qualification evidence.",
            "",
        ]
        return "\n".join(lines)


# ----------------------------------------------------------------------
# model builders shared by the cases
# ----------------------------------------------------------------------
def _plate(side: float, thickness: float, pressure: float = 0.0):
    from anyfem import Project, pinned, steel

    project = Project(name="verification_plate")
    project.add_material(steel("S355", thickness))
    project.add_plate_section("plate", thickness=thickness, material="S355")
    geometry = project.geometry
    points = geometry.add_points(
        [(0, 0, 0), (side, 0, 0), (side, side, 0), (0, side, 0)]
    )
    edges = geometry.add_polyline(points, close=True)
    face = geometry.add_face(edges)
    project.assign_plate(face, "plate")
    for edge in edges:
        project.add_support(pinned(project.edge(edge)))
    if pressure:
        project.load_case().add_pressure(project.face(face), pressure)
    return project, face, edges


def _bar(length: float, load: float, direction: str = "z"):
    from anyfem import Project, fixed, steel
    from anyfem.model import BeamSection

    project = Project(name="verification_bar")
    project.add_material(steel("S355", 0.020))
    section = BeamSection(
        name="bar", profile="Flatbar", material="S355",
        flange_width=0.10, flange_thickness=0.02, web_direction=(0.0, 0.0, 1.0),
    )
    project.add_beam_section(section)
    geometry = project.geometry
    root = geometry.add_point(0.0, 0.0, 0.0)
    tip = geometry.add_point(length, 0.0, 0.0)
    edge = geometry.add_line(root, tip)
    project.assign_beam(edge, "bar")
    project.add_support(fixed(project.point(root)))
    force = (0.0, 0.0, -load) if direction == "z" else (load, 0.0, 0.0)
    project.load_case().add_point_load(project.point(tip), force=force)
    return project, section, root, tip


def _stiffened_strip(eccentricity: float, pressure: float = 1000.0):
    """A plate strip with one flat bar running the full span down its centre.

    Supported at the two short ends only, so the strip and its stiffener act as
    one composite beam.  With ``eccentricity`` zero the bar shares the plate
    nodes and contributes only its own inertia; with the bar offset by
    ``t/2 + h/2`` it stands proud of the plating and the section gains the
    transfer terms.  The difference between the two is the whole point of the
    coupling, so both are built from the same function.
    """

    from anyfem import Project, pinned, steel, support
    from anyfem.geometry.operations import strip_face
    from anyfem.model import BeamSection

    length, width, thickness = 4.0, 0.2, 0.008
    bar_height, bar_thickness = 0.20, 0.010

    project = Project(name="verification_stiffened_strip")
    project.add_material(steel("S355", thickness))
    project.add_plate_section("plate", thickness=thickness, material="S355")
    project.add_beam_section(
        BeamSection(
            name="bar", profile="Flatbar", material="S355",
            flange_width=bar_thickness, flange_thickness=bar_height,
            web_direction=(0.0, 0.0, 1.0), eccentricity=eccentricity,
        )
    )
    geometry = project.geometry
    points = geometry.add_points(
        [(0, 0, 0), (length, 0, 0), (length, width, 0), (0, width, 0)]
    )
    face = geometry.add_face(geometry.add_polyline(points, close=True))
    project.assign_plate(face, "plate")
    strips, dividers = strip_face(geometry, face, axis=1, count=2)
    project.assign_plates(strips, "plate")
    project.assign_beam(dividers[0], "bar")

    def mid_x(edge_id: int) -> float:
        edge = geometry.edges[edge_id]
        ends = [geometry.vertices[v].position for v in (edge.start, edge.end)]
        return 0.5 * (ends[0][0] + ends[1][0])

    for edge_id in list(geometry.edges):
        if abs(mid_x(edge_id)) < 1.0e-9:
            project.add_support(pinned(project.edge(edge_id)))
        elif abs(mid_x(edge_id) - length) < 1.0e-9:
            # A roller: held down, free to slide, so no membrane stiffening.
            project.add_support(
                support(project.edge(edge_id), uy=0.0, uz=0.0)
            )
    for strip in strips:
        project.load_case().add_pressure(project.face(strip), pressure)

    geometry_data = {
        "length": length, "width": width, "thickness": thickness,
        "bar_height": bar_height, "bar_thickness": bar_thickness,
        "divider": dividers[0],
    }
    return project, geometry_data


def _transformed_section(data: Dict[str, float]) -> Dict[str, float]:
    """Neutral axis and inertia of the strip-plus-bar section, by hand."""

    plate_area = data["width"] * data["thickness"]
    bar_area = data["bar_thickness"] * data["bar_height"]
    plate_inertia = data["width"] * data["thickness"] ** 3 / 12.0
    bar_inertia = data["bar_thickness"] * data["bar_height"] ** 3 / 12.0
    offset = 0.5 * data["thickness"] + 0.5 * data["bar_height"]
    neutral_axis = bar_area * offset / (plate_area + bar_area)
    return {
        "offset": offset,
        "neutral_axis": neutral_axis,
        "shared": plate_inertia + bar_inertia,
        "eccentric": (
            plate_inertia
            + plate_area * neutral_axis**2
            + bar_inertia
            + bar_area * (offset - neutral_axis) ** 2
        ),
    }


def _flexural_rigidity(thickness: float) -> float:
    return MODULUS * thickness**3 / (12.0 * (1.0 - POISSON**2))


# ----------------------------------------------------------------------
# the cases
# ----------------------------------------------------------------------
def _cantilever_deflection():
    from anyfem import solve_linear_static

    length, load = 2.0, 100.0
    project, section, _root, tip = _bar(length, load)
    solution = solve_linear_static(project, target_size=length / 10)
    properties = section.properties()
    shear_modulus = MODULUS / (2.0 * (1.0 + POISSON))
    expected = load * length**3 / (3.0 * MODULUS * properties["Iy"]) + (
        load * length
        / (properties["shear_factor_z"] * shear_modulus * properties["area"])
    )
    computed = abs(solution.point_displacement(project.point(tip))[2])
    return computed, expected, "10 beam elements, Timoshenko shear included"


def _plate_deflection():
    from anyfem import solve_linear_static

    side, thickness, pressure = 1.0, 0.010, 10_000.0
    project, _face, _edges = _plate(side, thickness, pressure)
    solution = solve_linear_static(project, target_size=side / 16)
    expected = 0.00406 * pressure * side**4 / _flexural_rigidity(thickness)
    return solution.max_translation()[1], expected, "16 x 16 Q4 mesh"


def _plate_bending_stress():
    from anyfem import solve_linear_static
    from anyfem.post import evaluate_field

    side, thickness, pressure = 1.0, 0.010, 10_000.0
    project, _face, _edges = _plate(side, thickness, pressure)
    solution = solve_linear_static(project, target_size=side / 16)
    field = evaluate_field(solution, "bending_xx", reduction="max_abs")
    _key, peak = field.extreme()
    expected = 6.0 * (0.0479 * pressure * side**2) / thickness**2
    return abs(peak), expected, "peak element value, 16 x 16 mesh"


def _beam_axial_stress():
    from anyfem import solve_linear_static
    from anyfem.post import evaluate_field

    length, load = 2.0, 50_000.0
    project, section, _root, _tip = _bar(length, load, direction="x")
    solution = solve_linear_static(project, target_size=length / 10)
    field = evaluate_field(solution, "axial_stress", reduction="max_abs")
    _key, peak = field.extreme()
    return abs(peak), load / section.properties()["area"], "uniform axial bar"


def _cantilever_frequency():
    from anyfem import solve_modal

    length = 2.0
    project, section, _root, _tip = _bar(length, 0.0)
    project.load_case().add_point_load(project.point(2), force=(0, 0, -1.0))
    solution = solve_modal(project, target_size=length / 20, num_modes=3)
    properties = section.properties()
    expected = (
        1.875104**2
        / (2.0 * np.pi * length**2)
        * np.sqrt(MODULUS * properties["Iy"] / (DENSITY * properties["area"]))
    )
    return solution.frequencies[0], expected, "first bending mode, 20 elements"


def _euler_buckling():
    from anyfem import Project, solve_buckling, steel
    from anyfem.model import BeamSection, support

    length, load = 2.0, 1000.0
    project = Project(name="verification_strut")
    project.add_material(steel("S355", 0.020))
    section = BeamSection(
        name="bar", profile="Flatbar", material="S355",
        flange_width=0.10, flange_thickness=0.02, web_direction=(0.0, 0.0, 1.0),
    )
    project.add_beam_section(section)
    geometry = project.geometry
    start = geometry.add_point(0.0, 0.0, 0.0)
    end = geometry.add_point(length, 0.0, 0.0)
    edge = geometry.add_line(start, end)
    project.assign_beam(edge, "bar")
    project.add_support(support(project.point(start), ux=0.0, uy=0.0, uz=0.0, rx=0.0))
    project.add_support(support(project.point(end), uy=0.0, uz=0.0, rx=0.0))
    project.load_case().add_point_load(project.point(end), force=(-load, 0, 0))

    solution = solve_buckling(project, target_size=length / 20, num_modes=2)
    expected = np.pi**2 * MODULUS * section.properties()["Iy"] / length**2
    return (
        solution.critical_factor * load,
        expected,
        "pinned-pinned, 20 elements, weak axis",
    )


def _step_response():
    from anyfem import solve_linear_static, solve_transient

    side, thickness, pressure = 1.0, 0.008, 20_000.0
    project, _face, _edges = _plate(side, thickness, pressure)
    mesh = project.generate_mesh(side / 8)
    static = solve_linear_static(project, mesh=mesh).max_translation()[1]
    transient = solve_transient(
        project, mesh=mesh, dt=2.0e-4, t_end=0.02, save_every=2
    )
    return (
        transient.peak_displacement / static,
        2.0,
        "undamped Newmark, suddenly applied pressure",
    )


def _self_weight():
    from anysolver import assemble_load_vector

    from anyfem import solve_linear_static

    side, thickness = 2.0, 0.010
    project, _face, _edges = _plate(side, thickness)
    project.load_case().set_gravity()
    solution = solve_linear_static(project, target_size=side / 8)
    vector = assemble_load_vector(
        solution.built.fe_model, solution.built.load_case
    )
    if isinstance(vector, tuple):
        vector = vector[0]
    computed = float(vector[2::6].sum())
    expected = -DENSITY * GRAVITY * side * side * thickness
    return computed, expected, "consistent inertial load over the plate"


def _combination_linearity():
    from anyfem import solve_linear_static

    side, thickness = 1.0, 0.010
    project, face, _edges = _plate(side, thickness)
    project.load_case("dead").add_pressure(project.face(face), 10_000.0)
    project.load_case("live").add_pressure(project.face(face), 4_000.0)
    project.add_combination("ULS", {"dead": 1.2, "live": 1.5})

    mesh = project.generate_mesh(side / 8)
    dead = solve_linear_static(project, mesh=mesh, load_case="dead")
    live = solve_linear_static(project, mesh=mesh, load_case="live")
    combined = solve_linear_static(project, mesh=mesh, combination="ULS")
    expected = 1.2 * dead.max_translation()[1] + 1.5 * live.max_translation()[1]
    return combined.max_translation()[1], expected, "1.2 dead + 1.5 live"


def _cylinder_exactness():
    from anyfem.geometry import GeometryModel
    from anyfem.mesh import generate_mesh

    radius = 2.0
    model = GeometryModel()
    start = model.add_point(radius, 0.0, 0.0)
    end = model.add_point(radius, 0.0, 3.0)
    edge = model.add_line(start, end)
    model.revolve([edge], (0, 0, 0), (0, 0, 1), 2.0 * np.pi)
    mesh = generate_mesh(model, target_size=0.4)
    positions = mesh.node_positions()
    worst = float(np.abs(np.linalg.norm(positions[:, :2], axis=1) - radius).max())
    return radius + worst, radius, f"{mesh.num_nodes} nodes, full revolve"


def _mesh_conformity():
    from anyfem.geometry import GeometryModel
    from anyfem.mesh import generate_mesh

    model = GeometryModel()
    points = model.add_points(
        [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)]
    )
    left_bottom = model.add_line(points[0], points[1])
    right_bottom = model.add_line(points[1], points[2])
    right = model.add_line(points[2], points[3])
    right_top = model.add_line(points[3], points[4])
    left_top = model.add_line(points[4], points[5])
    left = model.add_line(points[5], points[0])
    shared = model.add_line(points[1], points[4])
    first = model.add_face([left_bottom, shared, left_top, left])
    second = model.add_face([right_bottom, right, right_top, shared])

    mesh = generate_mesh(model, target_size=0.25)
    left_nodes = set(mesh.nodes_on(model.entity_ref("face", first)))
    right_nodes = set(mesh.nodes_on(model.entity_ref("face", second)))
    shared_nodes = set(mesh.nodes_on(model.entity_ref("edge", shared)))
    matched = 1.0 if (left_nodes & right_nodes) == shared_nodes else 0.0
    return matched, 1.0, "shared edge nodes are the same node objects"


def _impact_momentum():
    from anyfem.model.collision import Collision
    from anyfem.solve import solve_impact

    side, thickness = 1.0, 0.008
    project, _face, _edges = _plate(side, thickness)
    collision = Collision(
        mass=200.0, radius=0.15, start=(0.5, 0.5, 0.6),
        direction=(0.0, 0.0, -1.0), speed=4.0,
    )
    solution = solve_impact(
        project, collision=collision, target_size=side / 8
    )
    # The solver's own balance check, normalised by the sphere's momentum.
    reference = collision.mass * collision.speed
    return (
        1.0 + abs(solution.momentum_balance_error) / reference,
        1.0,
        f"{solution.status}, peak force {solution.peak_contact_force:.4g} N",
    )


def _impact_energy():
    from anyfem.model.collision import Collision
    from anyfem.solve import solve_impact

    side, thickness = 1.0, 0.008
    project, _face, _edges = _plate(side, thickness)
    collision = Collision(
        mass=200.0, radius=0.15, start=(0.5, 0.5, 0.6),
        direction=(0.0, 0.0, -1.0), speed=4.0,
    )
    solution = solve_impact(project, collision=collision, target_size=side / 8)
    energy = solution.energy()
    return (
        energy["initial"],
        collision.kinetic_energy,
        f"{energy['absorbed'] / 1000.0:.4g} kJ transferred to the structure",
    )


def _eccentric_neutral_axis():
    from anyfem import solve_linear_static

    section = _transformed_section(
        _stiffened_strip(0.0)[1]  # geometry only; eccentricity is irrelevant
    )
    project, data = _stiffened_strip(section["offset"])
    mesh = project.generate_mesh(0.05)
    solution = solve_linear_static(project, mesh=mesh)

    plate_nodes = mesh.nodes_of_edge[data["divider"]]
    bar_nodes = mesh.offset_nodes_of_edge[data["divider"]]
    stations = np.array([mesh.nodes[node][0] for node in plate_nodes])
    # Two stations either side of midspan: the axial strain there is largest,
    # and the neutral axis is the height at which it vanishes.
    first = int(np.argmin(np.abs(stations - 1.9)))
    last = int(np.argmin(np.abs(stations - 2.1)))
    span = stations[last] - stations[first]
    plate_strain = (
        solution.node_displacement(plate_nodes[last])[0]
        - solution.node_displacement(plate_nodes[first])[0]
    ) / span
    bar_strain = (
        solution.node_displacement(bar_nodes[last])[0]
        - solution.node_displacement(bar_nodes[first])[0]
    ) / span
    computed = section["offset"] * plate_strain / (plate_strain - bar_strain)
    return (
        computed,
        section["neutral_axis"],
        f"{len(mesh.couplings)} couplings, strain zero-crossing at midspan",
    )


def _eccentric_stiffness():
    from anyfem import solve_linear_static

    section = _transformed_section(_stiffened_strip(0.0)[1])
    deflection = {}
    for label, eccentricity in (
        ("shared", 0.0), ("eccentric", section["offset"])
    ):
        project, _data = _stiffened_strip(eccentricity)
        solution = solve_linear_static(project, target_size=0.05)
        deflection[label] = solution.max_translation()[1]
    return (
        deflection["shared"] / deflection["eccentric"],
        section["eccentric"] / section["shared"],
        "deflection ratio against transformed-section inertia ratio",
    )


def _hardening_curve_transfer():
    """The curve a nonlinear solve uses must be the material's own table row.

    ``Material`` stores the hardening as a ``(source, grade, thickness)``
    recipe and rebuilds the curve on demand.  If that recipe ever drifted from
    the grade and thickness the elastic properties came from, the model would
    solve with one steel's stiffness and another's yield -- quietly, and only
    in the plastic range.  Comparing the two yield stresses catches it.
    """

    from anyfem import steel

    material = steel("S355", 0.020, nonlinear=True)
    curve = material.hardening_curve()
    return (
        float(curve.sigma_yield),
        material.yield_stress,
        f"proportional limit {curve.sigma_prop / 1e6:.4g} MPa, "
        f"flow stress at 5% plastic strain "
        f"{curve.flow_stress(0.05) / 1e6:.4g} MPa",
    )


def _graded_size_honoured():
    """A refinement zone must actually produce the size it asks for."""

    from anyfem.geometry import GeometryModel
    from anyfem.mesh import generate_mesh, refine_at

    side, target, refined = 4.0, 0.5, 0.1
    model = GeometryModel()
    points = model.add_points(
        [(0, 0, 0), (side, 0, 0), (side, side, 0), (0, side, 0)]
    )
    face = model.add_face(model.add_polyline(points, close=True))
    zone = refine_at((0.0, 0.0, 0.0), size=refined, radius=0.3)
    mesh = generate_mesh(model, target_size=target, refinements=[zone])

    edge = model.faces[face].loop[0].edge
    stations = np.sort(
        [mesh.nodes[node][0] for node in mesh.nodes_of_edge[edge]]
    )
    spacings = np.diff(stations)
    return (
        float(spacings.min()),
        refined,
        f"{len(spacings)} divisions, coarsest {spacings.max():.4g} m against a "
        f"{target:g} m target",
    )


def _quadratic_convergence():
    """Q8 against Q4 at the same element count, both against their own limit.

    Compared to the converged finite element answer rather than the thin-plate
    series, because the two are different numbers: these elements are
    Mindlin-Reissner and carry transverse shear, so they converge about 1%
    above the Kirchhoff coefficient. Using the series here would credit Q4 for
    passing through it on the way up.
    """

    from anyfem import solve_linear_static

    side, thickness, pressure = 1.0, 0.010, 10_000.0

    def deflection(order: str, divisions: int) -> float:
        project, _face, _edges = _plate(side, thickness, pressure)
        project.set_element_order(order)
        return solve_linear_static(
            project, target_size=side / divisions
        ).max_translation()[1]

    converged = deflection("quadratic", 24)
    coarse = deflection("quadratic", 4)
    linear = abs(deflection("linear", 4) / converged - 1.0)
    return (
        coarse,
        converged,
        f"16 elements; Q4 on the same 16 is {linear:.2%} out, and STAT-02 needs "
        f"256 Q4 elements for the same tolerance. Converged "
        f"{converged * 1000:.5g} mm",
    )


def _impact_contact_resolution():
    """Refining for impact must resolve the contact patch it was asked to."""

    from anyfem.commands import CommandStack, RefineForImpact
    from anyfem.model.collision import Collision
    from anyfem.solve import solve_impact

    side, thickness = 1.0, 0.008
    collision = Collision(
        mass=200.0, radius=0.15, start=(0.5, 0.5, 0.6),
        direction=(0.0, 0.0, -1.0), speed=4.0,
    )

    coarse_project, _face, _edges = _plate(side, thickness)
    coarse = solve_impact(
        coarse_project, collision=collision, target_size=side / 8
    ).info["contact_resolution"]

    project, _face, _edges = _plate(side, thickness)
    CommandStack(project).run(
        RefineForImpact(
            collision=collision, target_size=side / 8, elements_per_radius=4.0
        )
    )
    mesh = project.generate_mesh(side / 8)
    from anyfem.solve.run import _contact_resolution

    refined = _contact_resolution(mesh, collision)
    return (
        refined["elements_per_radius"],
        4.0,
        f"asked for 4 elements per sphere radius; the uniform mesh gave "
        f"{coarse['elements_per_radius']:.2f}. Measured as a mean over the "
        "patch, which grading makes slightly coarser than the target at its "
        "edge",
    )


def _symmetry_matches_the_full_model():
    """A quarter model with symmetry planes against the whole plate.

    Not a closed form, and a stronger check than one: the reference is the same
    structure solved without the simplification, so any error in the symmetry
    conditions shows up directly rather than inside a series coefficient's
    tolerance.
    """

    from anyfem import Project, pinned, solve_linear_static, steel

    side, thickness, pressure = 2.0, 0.010, 10_000.0

    def plate(x_end: float, y_end: float):
        project = Project(name="symmetry")
        project.add_material(steel("S355", thickness))
        project.add_plate_section("plate", thickness=thickness, material="S355")
        geometry = project.geometry
        points = geometry.add_points(
            [(0, 0, 0), (x_end, 0, 0), (x_end, y_end, 0), (0, y_end, 0)]
        )
        edges = geometry.add_polyline(points, close=True)
        face = geometry.add_face(edges)
        project.assign_plate(face, "plate")
        project.load_case().add_pressure(project.face(face), pressure)
        return project, edges

    full, edges = plate(side, side)
    for edge_id in edges:
        full.add_support(pinned(full.edge(edge_id)))
    reference = solve_linear_static(
        full, target_size=side / 16
    ).max_translation()[1]

    quarter, quarter_edges = plate(side / 2, side / 2)
    geometry = quarter.geometry
    for edge_id in quarter_edges:
        edge = geometry.edges[edge_id]
        ends = [geometry.vertices[v].position for v in (edge.start, edge.end)]
        middle = 0.5 * (ends[0] + ends[1])
        if abs(middle[0]) < 1.0e-9:
            quarter.add_symmetry(quarter.edge(edge_id), "x")
        elif abs(middle[1]) < 1.0e-9:
            quarter.add_symmetry(quarter.edge(edge_id), "y")
        else:
            quarter.add_support(pinned(quarter.edge(edge_id)))
    computed = solve_linear_static(
        quarter, target_size=side / 16
    ).max_translation()[1]

    return (
        computed,
        reference,
        "quarter plate with x and y symmetry planes against the full plate at "
        "the same element size",
    )


def _result_round_trip():
    """A solution written as an FRD and read back must be the same numbers.

    Checks the interop path end to end -- write, parse, match by node ID,
    rebuild a displayable shape -- against the solve it came from.  Any loss in
    the format's precision, the node matching or the DOF mapping shows up here
    as a difference from a reference that is exact by construction.
    """

    import tempfile

    from anyfem import solve_linear_static
    from anyfem.io import import_calculix_results
    from anyfem.solve.build import build_fe_model

    side, thickness, pressure = 1.0, 0.008, 10_000.0
    project, _face, _edges = _plate(side, thickness, pressure)
    mesh = project.generate_mesh(side / 4)
    built = build_fe_model(project, mesh)
    solution = solve_linear_static(project, mesh=mesh)

    nodes = sorted(mesh.nodes)
    manager = built.fe_model.mesh.dof_manager
    lines = ["    1C", f"    2C{len(nodes):30d}"]
    for node in nodes:
        x, y, z = mesh.nodes[node]
        lines.append(f" -1{node:10d}{x:12.5E}{y:12.5E}{z:12.5E}")
    lines.append(" -3")
    lines.append("    1PSTEP                         1")
    lines.append(f"  100CL  101{len(nodes):12d}".ljust(60) + "1")
    lines.append(" -4  DISP        4    1")
    for name in ("D1", "D2", "D3", "ALL"):
        lines.append(f" -5  {name:8s} 1    2    1    0")
    for node in nodes:
        values = solution.displacements[manager.get_node_dofs(node)[:3]]
        lines.append(
            f" -1{node:10d}{values[0]:12.5E}{values[1]:12.5E}{values[2]:12.5E}"
        )
    lines.append(" -3")

    with tempfile.TemporaryDirectory() as directory:
        written = Path(directory) / "verification.frd"
        written.write_text("\n".join(lines) + "\n", encoding="utf-8")
        attached = import_calculix_results(written).attach(built)
        computed = attached.max_translation()[1]

    return (
        computed,
        solution.max_translation()[1],
        f"{len(nodes)} nodes written and matched back; the format stores five "
        "significant figures, which is the tolerance here",
    )


def cases() -> List[VerificationCase]:
    """Every verification case, in a stable order."""

    return [
        VerificationCase(
            "GEOM-01", "Revolved cylinder radius", "exact circle", 1.0e-9, "m",
            _cylinder_exactness,
        ),
        VerificationCase(
            "MESH-01", "Conformity across a shared edge", "set equality", 0.0,
            "", _mesh_conformity,
        ),
        VerificationCase(
            "STAT-01", "Cantilever tip deflection", "PL^3/3EI + shear", 0.01,
            "m", _cantilever_deflection,
        ),
        VerificationCase(
            "STAT-02", "Simply supported plate deflection",
            "Timoshenko 0.00406 q a^4 / D", 0.02, "m", _plate_deflection,
        ),
        VerificationCase(
            "STAT-03", "Plate bending stress", "6M/t^2, M = 0.0479 q a^2",
            0.02, "Pa", _plate_bending_stress,
        ),
        VerificationCase(
            "STAT-04", "Beam axial stress", "P/A", 0.01, "Pa",
            _beam_axial_stress,
        ),
        VerificationCase(
            "LOAD-01", "Self weight resultant", "rho g V", 1.0e-6, "N",
            _self_weight,
        ),
        VerificationCase(
            "LOAD-02", "Combination linearity", "1.2 dead + 1.5 live", 1.0e-9,
            "m", _combination_linearity,
        ),
        VerificationCase(
            "MODE-01", "Cantilever first frequency",
            "1.875^2 / 2 pi L^2 sqrt(EI/rho A)", 0.02, "Hz",
            _cantilever_frequency,
        ),
        VerificationCase(
            "BUCK-01", "Euler strut buckling", "pi^2 EI / L^2", 0.01, "N",
            _euler_buckling,
        ),
        VerificationCase(
            "DYN-01", "Suddenly applied load", "undamped peak = 2 x static",
            0.05, "-", _step_response,
        ),
        VerificationCase(
            "IMPA-01", "Sphere momentum balance", "closed momentum balance",
            1.0e-3, "-", _impact_momentum,
        ),
        VerificationCase(
            "IMPA-02", "Sphere kinetic energy", "1/2 m v^2", 1.0e-9, "J",
            _impact_energy,
        ),
        VerificationCase(
            "ECC-01", "Eccentric stiffener neutral axis",
            "transformed section A e / sum A", 0.01, "m",
            _eccentric_neutral_axis,
        ),
        VerificationCase(
            "ECC-02", "Eccentric stiffener stiffness",
            "transformed section inertia ratio", 0.03, "-",
            _eccentric_stiffness,
        ),
        VerificationCase(
            "MATL-01", "Hardening curve proportional limit",
            "DNV-RP-C208 yield stress", 1.0e-9, "Pa",
            _hardening_curve_transfer,
        ),
        VerificationCase(
            "MESH-02", "Graded element size at a refinement zone",
            "the size the zone asks for", 0.10, "m", _graded_size_honoured,
        ),
        VerificationCase(
            "ELEM-01", "Q8 plate deflection on 16 elements",
            "the converged finite element answer", 0.02, "m",
            _quadratic_convergence,
        ),
        VerificationCase(
            "IMPA-03", "Contact resolution after refining for impact",
            "4 elements per sphere radius", 0.20, "-",
            _impact_contact_resolution,
        ),
        VerificationCase(
            "SYMM-01", "Quarter model with symmetry planes",
            "the same plate solved in full", 1.0e-9, "m",
            _symmetry_matches_the_full_model,
        ),
        VerificationCase(
            "INTR-01", "Result written as FRD and read back",
            "the solution it was written from", 1.0e-4, "m",
            _result_round_trip,
        ),
    ]


def run_verification(
    selected: Optional[Sequence[str]] = None,
) -> VerificationReport:
    """Run the cases and collect the evidence."""

    from anyfem import __version__

    wanted = None if selected is None else set(selected)
    results = [
        case.evaluate()
        for case in cases()
        if wanted is None or case.case_id in wanted
    ]
    return VerificationReport(
        results=results,
        generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        environment={
            "anyfem": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    )


def write_verification_report(
    report: VerificationReport, directory: str | Path = "reports/verification"
) -> Dict[str, Path]:
    """Write the evidence as JSON and Markdown."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "verification.json"
    markdown_path = target / "verification.md"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    markdown_path.write_text(report.to_markdown(), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    """Run the verification suite and write its evidence."""

    import argparse

    parser = argparse.ArgumentParser(description="Run ANYfem verification.")
    parser.add_argument(
        "--out", default="reports/verification", help="where to write evidence"
    )
    parser.add_argument("--case", action="append", help="run only these case IDs")
    parser.add_argument(
        "--no-save", action="store_true", help="print only, write nothing"
    )
    args = parser.parse_args(argv)

    report = run_verification(args.case)
    print(report.to_markdown())
    if not args.no_save:
        written = write_verification_report(report, args.out)
        print(f"written: {written['markdown']}")
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
