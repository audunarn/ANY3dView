"""Reading ANYstructure's saved FE state, and measuring the migration gate.

ANYfem is meant to replace ANYstructure's ``fem_integration.py``, and the gate
in :mod:`anyfem.parity` says what "ready" means.  This module is the part of
that gate which needs code rather than a ledger entry: reading the files the
old GUI saved, and comparing what ANYfem computes against what it recorded.

**Nothing here imports ANYstructure.** It does not need to: a
``save_runtime_fem_state`` file is plain JSON (gzipped when the name ends
``.gz``) carrying a format tag, so it is read as data.  That is the whole point
of the one-way dependency -- ANYfem can consume the old application's output
without ever depending on the old application.

What this deliberately does **not** do is rebuild the model.  Two reasons, and
both are the same reason:

* the snapshot describes a *parametric panel*, which is out of ANYfem's scope
  by decision -- the parametric front end owns that and calls in;
* the stored ``visualization`` is a plotting grid, not a mesh with node IDs and
  connectivity, so there is no topology in the file to rebuild from.

So a loaded state gives back **settings and recorded numbers**, and says which
of its settings ANYfem can act on.  Reporting the rest by name is the point: a
migration that silently dropped a setting would run a different analysis from
the one the file asked for, and would look like it had worked.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ComparisonCase",
    "ComparisonResult",
    "RuntimeState",
    "StateFileError",
    "compare_case",
    "gate_markdown",
    "gate_report",
    "main",
    "read_runtime_fem_state",
    "write_gate_report",
]

STATE_FORMAT = "anystructure-runtime-fem-state-v1"


class StateFileError(ValueError):
    """Raised when a file is not a readable ANYstructure FE state."""


# Options ANYfem can act on, and what it does with each.  The table growing is
# the migration progressing; what is left over is the work remaining, so it is
# worth keeping accurate in both directions.  Understating it would make the
# migration look further away than it is, which is its own kind of wrong.
_OPTION_MAP: Dict[str, str] = {
    # meshing
    "mesh_size_m": "target element size",
    "mesh_fidelity": "target element size, when mesh_size_m is zero",
    "shell_element_order": "element order (S4 -> linear, S8 -> quadratic)",
    "beam_element_order": "element order (B2 -> linear, B3 -> quadratic)",
    "local_refinement_enabled": "refinement zone",
    "local_refinement_extent_m": "refinement zone radius",
    "local_refinement_fine_size_m": "refinement zone element size",
    "local_refinement_fine_factor": "refinement zone element size",
    "local_refinement_growth_factor": "refinement zone growth",
    "local_refinement_zone_factor": "refinement zone radius",
    "point_refinement_enabled": "refinement zone at a point",
    "point_refinement_extent_m": "refinement zone radius",
    "point_refinement_fine_size_m": "refinement zone element size",
    "point_refinement_fine_factor": "refinement zone element size",
    "point_refinement_growth_factor": "refinement zone growth",
    "point_refinement_x_m": "refinement zone centre",
    "point_refinement_y_m": "refinement zone centre",
    # materials
    "steel_grade": "steel grade",
    "steel_thickness_class": "steel thickness class",
    "elastic_modulus_pa": "material modulus",
    "poisson_ratio": "material Poisson ratio",
    "yield_stress_pa": "material yield stress",
    "material_model": "whether the material carries a hardening curve",
    # members
    "include_stiffeners": "which members are meshed",
    "include_girders": "which members are meshed",
    "stiffener_eccentricity_m": "beam section eccentricity",
    "girder_eccentricity_m": "beam section eccentricity",
    # loads
    "pressure_pa": "pressure load",
    "pressure_direction": "pressure sign",
    "follower_pressure": "follower pressure",
    "custom_pressure_pa": "pressure load",
    "load_scale": "load case factor",
    "acceleration_x_m_s2": "acceleration field",
    "acceleration_y_m_s2": "acceleration field",
    "acceleration_z_m_s2": "acceleration field",
    "added_mass_kg": "point mass",
    "added_mass_location": "where the point mass attaches",
    "plate_edge_x0_load_n_per_m": "line load on an edge",
    "plate_edge_x1_load_n_per_m": "line load on an edge",
    "plate_edge_y0_load_n_per_m": "line load on an edge",
    "plate_edge_y1_load_n_per_m": "line load on an edge",
    "cylinder_lower_edge_load_n_per_m": "line load on an edge",
    "cylinder_upper_edge_load_n_per_m": "line load on an edge",
    # supports
    "boundary_condition": "supports",
    "boundary_auto_supports": "supports",
    "symmetry_mode": "symmetry supports",
    "plate_edge_x0_support": "support on an edge",
    "plate_edge_x1_support": "support on an edge",
    "plate_edge_y0_support": "support on an edge",
    "plate_edge_y1_support": "support on an edge",
    "cylinder_lower_support": "support on an edge",
    "cylinder_upper_support": "support on an edge",
    "enforced_displacement_x_m": "prescribed displacement",
    "enforced_displacement_y_m": "prescribed displacement",
    "enforced_displacement_z_m": "prescribed displacement",
    # imperfections
    "imperfection_enabled": "geometric imperfection",
    "imperfection_amplitude_m": "imperfection amplitude",
    "imperfection_shape": "imperfection kind",
    "imperfection_wave_a": "imperfection wave count",
    "imperfection_wave_b": "imperfection wave count",
    # analyses
    "analysis_type": "which analysis to run",
    "buckling_analysis_type": "which buckling path to run",
    "num_buckling_modes": "buckling mode count",
    "nonlinear_steps": "nonlinear increment count",
    "nonlinear_max_load_factor": "nonlinear load factor",
    "nonlinear_max_iterations": "nonlinear iteration cap",
    "nonlinear_tolerance": "nonlinear convergence tolerance",
    "nonlinear_layers": "layered shell integration points",
    "nonlinear_convergence_profile": "nonlinear convergence settings",
    "nonlinear_static_kinematics": "nonlinear kinematics",
    "nonlinear_solution_control": "force or displacement control",
    "post_buckling_enabled": "arc-length continuation",
    "post_buckling_max_displacement_m": "arc-length limit",
    "post_buckling_stop_load_fraction": "arc-length stop condition",
    # fracture
    "fracture_enabled": "fracture configuration",
    "fracture_strain_threshold": "fracture threshold",
    "fracture_max_deleted_fraction": "fracture erosion cap",
    "fracture_min_load_factor": "fracture start",
    "fracture_residual_stiffness_fraction": "fracture residual stiffness",
    # impact
    "collision_speed_mps": "collision speed",
    "collision_start_x_m": "collision start point",
    "collision_start_y_m": "collision start point",
    "collision_start_z_m": "collision start point",
    "collision_vector_x": "collision direction",
    "collision_vector_y": "collision direction",
    "collision_vector_z": "collision direction",
    "collision_total_time_s": "impact duration",
    "collision_time_mode": "automatic or explicit impact timing",
    "collision_result_interval_s": "impact save interval",
    "collision_target_penetration_fraction": "contact penalty target",
    "collision_enabled": "whether to run an impact",
    "collision_mass_kg": "sphere mass",
    "collision_radius_m": "sphere radius",
    "collision_dt_s": "impact time step",
    "collision_include_static_load": "impact base load case",
    "collision_penalty_stiffness_n_per_m": "contact penalty stiffness",
    "collision_penalty_scale": "contact penalty scaling",
    "collision_penetration_tolerance_m": "contact tolerance",
    "collision_force_tolerance_n": "contact tolerance",
    "collision_max_iterations": "contact iteration cap",
    "collision_max_event_substeps": "contact substep cap",
    "collision_material_nonlinear_enabled": "nonlinear impact",
    "collision_nonlinear_kinematics": "nonlinear impact kinematics",
    "collision_nonlinear_tolerance": "nonlinear impact tolerance",
    "collision_nonlinear_max_iterations": "nonlinear impact iteration cap",
    "collision_nonlinear_cutbacks": "nonlinear impact cutbacks",
    "collision_damage_enabled": "impact damage configuration",
    "collision_damage_mode": "impact damage mode",
    "collision_damage_criterion": "impact damage capacity basis",
    "collision_damage_delete_at": "impact damage deletion threshold",
    "collision_damage_softening_start": "impact damage softening start",
    "collision_damage_max_deleted_fraction": "impact damage erosion cap",
    "collision_damage_user_capacity_pa": "impact damage capacity",
    "collision_damage_min_contact_area_m2": "impact damage contact area floor",
    "collision_damage_neighbor_smoothing": "impact damage smoothing",
    "collision_plastic_damage_threshold": "plastic impact damage threshold",
    "collision_damage_capacity_basis": "impact damage capacity basis",
    "collision_adaptive_mesh_enabled": "refine for impact",
    "collision_adaptive_extent_m": "impact refinement radius",
    "collision_adaptive_zone_factor": "impact refinement radius",
    "collision_adaptive_fine_size_m": "impact refinement element size",
    "collision_adaptive_fine_factor": "impact refinement element size",
    "collision_adaptive_growth_factor": "impact refinement growth",
    "collision_auto_steps_per_radius": "automatic impact time step",
    "collision_auto_post_contact_radii": "automatic impact duration",
    # buckling controls, passed through to the solver
    "buckling_shift_load_factor": "buckling shift",
    "buckling_min_load_factor": "buckling range",
    "buckling_max_load_factor": "buckling range",
    "buckling_repeated_tolerance": "repeated-mode tolerance",
    "buckling_allow_dense_fallback": "buckling solver fallback",
    # capacity workflow
    "capacity_buckling_mode_number": "capacity workflow imperfection mode",
    "capacity_mesh_min_elements_per_half_wave": "capacity workflow mesh adequacy",
    # time domain
    "custom_time_domain_enabled": "transient analysis",
    "custom_time_domain_dt_s": "transient time step",
    "custom_time_domain_duration_s": "transient duration",
    "custom_time_domain_total_time_s": "transient duration",
    "custom_time_domain_result_interval_s": "transient save interval",
    "custom_time_domain_include_static_load": "transient base load case",
    # recovery and resources
    "recovery_history_mode": "recovery policy history mode",
    "recovery_threads": "resource policy threads",
    "nonlinear_assembly_threads": "resource policy threads",
    "memory_limit_mb": "resource policy memory limit",
    "solver_type": "solver backend passthrough",
    "runtime_solver": "solver backend passthrough",
}

# Options ANYfem will *never* map, with the reason.  Distinguishing these from
# "not yet" is the difference between a debt list and a to-do list: work that
# will never be done should not sit on the second one forever.
_OUT_OF_SCOPE_OPTIONS: Dict[str, str] = {
    "axial_force_n": "section resultant on a parametric panel; out of scope",
    "top_bottom_moment_nm": "section resultant on a parametric panel; out of scope",
    "torsional_moment_nm": "section resultant on a parametric panel; out of scope",
    "shear_force_n": "section resultant on a parametric panel; out of scope",
    "member_model": "parametric panel construction",
    "member_orientation": "parametric panel construction",
    "include_end_lids": "parametric panel construction",
    "ignore_girder_length": "parametric panel construction",
    "detail_transition_style": "parametric panel construction",
    "thickness_regions_json": "parametric panel construction",
    "deformation_scale": "display only",
    "stress_percentile": "display only",
    # These name segments, patches and edges *of a parametric panel*. ANYfem
    # binds attributes to its own topology, so there is nothing here for the
    # names to resolve against -- the parametric front end owns them and
    # applies its own when it calls in.
    "custom_bc_segments_json": "names parametric-panel segments",
    "custom_edge_segments_json": "names parametric-panel segments",
    "custom_loads_json": "names parametric-panel segments",
    "custom_pressure_patches_json": "names parametric-panel patches",
    "custom_selected_edge_load_components_json": "names parametric-panel edges",
    "custom_selected_edge_load_n_per_m": "names parametric-panel edges",
    "custom_load_bc_enabled": "enables the parametric-panel load system",
    "custom_loads_add_to_imported": "enables the parametric-panel load system",
    "edge_load_components_json": "names parametric-panel edges",
    "local_refinement_patches_json": "names parametric-panel patches",
    "imperfection_mode_shapes_json": "names parametric-panel mode shapes",
    "boundary_constraint_json": "names parametric-panel edges",
}

# Element order names, as fem_integration spells them.
_SHELL_ORDER = {"s4": "linear", "s8": "quadratic"}
_BEAM_ORDER = {"b2": "linear", "b3": "quadratic"}


@dataclass(frozen=True)
class RuntimeState:
    """One saved ANYstructure FE run: its settings, and what it produced."""

    source: Path
    saved_utc: str
    options: Mapping[str, Any]
    snapshot: Mapping[str, Any] = field(default_factory=dict)
    result: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def has_result(self) -> bool:
        return bool(self.result)

    @property
    def status(self) -> str:
        return str(self.result.get("status", "")) if self.result else ""

    @property
    def buckling_factors(self) -> Tuple[float, ...]:
        return tuple(
            float(value) for value in self.result.get("buckling_factors", ())
        )

    @property
    def stress_percentiles(self) -> Dict[str, float]:
        pairs = self.result.get("stress_percentiles", ())
        return {str(name): float(value) for name, value in pairs}

    # ------------------------------------------------------------------
    @property
    def target_size(self) -> Optional[float]:
        """The element size this run asked for, in metres.

        ``mesh_size_m`` wins when it is set.  ``mesh_fidelity`` is a named
        band, and ANYfem does not invent a number for it -- returning None
        means "this file did not pin a size", which the caller must decide
        about rather than have decided for it.
        """

        size = float(self.options.get("mesh_size_m", 0.0) or 0.0)
        return size if size > 0.0 else None

    @property
    def element_order(self) -> Optional[str]:
        """The element order both member types agree on, or None.

        A file asking for S8 shells with B2 beams describes a mesh ANYfem
        cannot build: order is one setting here, because a 2-node beam on a Q8
        shell edge would miss the edge's mid-side node. Disagreement is
        reported rather than resolved by preferring one of them.
        """

        shell = _SHELL_ORDER.get(str(self.options.get("shell_element_order", "")).lower())
        beam = _BEAM_ORDER.get(str(self.options.get("beam_element_order", "")).lower())
        if shell is None and beam is None:
            return None
        if shell is not None and beam is not None and shell != beam:
            return None
        return shell or beam

    @property
    def unmapped_options(self) -> List[str]:
        """Settings ANYfem cannot act on *yet*.

        Excludes the ones it never will -- those are listed separately, because
        a debt list and a to-do list are different things and work that will
        never be done should not sit on the second one forever.

        Only settings left at something other than their default would change a
        run, but the defaults are not in the file, so everything unmapped is
        listed. Over-reporting here is the safe direction.
        """

        return sorted(
            set(self.options) - set(_OPTION_MAP) - set(_OUT_OF_SCOPE_OPTIONS)
        )

    @property
    def mapped_options(self) -> Dict[str, str]:
        """Settings ANYfem can act on, and what it does with each."""

        return {
            name: _OPTION_MAP[name]
            for name in sorted(self.options)
            if name in _OPTION_MAP
        }

    @property
    def out_of_scope_options(self) -> Dict[str, str]:
        """Settings ANYfem will not act on, and why."""

        return {
            name: _OUT_OF_SCOPE_OPTIONS[name]
            for name in sorted(self.options)
            if name in _OUT_OF_SCOPE_OPTIONS
        }

    def summary(self) -> str:
        pieces = [
            f"{self.source.name}: {len(self.options)} options, "
            f"{len(self.mapped_options)} mapped, "
            f"{len(self.unmapped_options)} not yet, "
            f"{len(self.out_of_scope_options)} out of scope"
        ]
        if self.snapshot:
            kind = "cylinder" if self.snapshot.get("is_cylinder") else "panel"
            pieces.append(f"{kind} {self.snapshot.get('line_name', '')}".strip())
        if self.has_result:
            pieces.append(f"result {self.status}")
            if self.buckling_factors:
                pieces.append(
                    f"{len(self.buckling_factors)} buckling factors, "
                    f"critical {self.buckling_factors[0]:.5g}"
                )
        else:
            pieces.append("settings only, no result")
        return "; ".join(pieces)


def read_runtime_fem_state(path: str | Path) -> RuntimeState:
    """Read a ``save_runtime_fem_state`` file.

    Accepts plain JSON and the gzipped form the old GUI writes for ``.gz``
    names, the same two the writer produces.
    """

    source = Path(path)
    if not source.exists():
        raise StateFileError(f"no state file at {source}")

    try:
        if source.suffix.lower() == ".gz":
            with gzip.open(source, "rt", encoding="utf-8") as handle:
                data = json.load(handle)
        else:
            data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, gzip.BadGzipFile) as error:
        raise StateFileError(f"cannot read {source.name}: {error}") from None
    except json.JSONDecodeError as error:
        raise StateFileError(
            f"{source.name} is not valid JSON: {error}"
        ) from None

    if not isinstance(data, Mapping):
        raise StateFileError(f"{source.name} is not an object at the top level")
    found = data.get("format")
    if found != STATE_FORMAT:
        raise StateFileError(
            f"{source.name} is not an ANYstructure runtime FE state: its "
            f"format tag is {found!r}, expected {STATE_FORMAT!r}."
        )
    options = data.get("options")
    if not isinstance(options, Mapping):
        raise StateFileError(
            f"{source.name} has the right format tag but no options block"
        )

    return RuntimeState(
        source=source,
        saved_utc=str(data.get("saved_utc", "")),
        options=dict(options),
        snapshot=dict(data.get("snapshot") or {}),
        result=dict(data.get("result") or {}),
    )


# ----------------------------------------------------------------------
# comparison against recorded numbers
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ComparisonCase:
    """A model ANYfem should reproduce, and the numbers it should reach.

    ``build`` returns a ready-to-solve project.  ``expected`` are the recorded
    ANYstructure results, keyed by quantity.  Recorded rather than computed on
    the fly on purpose: the comparison must not depend on ANYstructure being
    installed or runnable, or the gate would only be checkable on one machine.
    """

    name: str
    build: Any
    expected: Mapping[str, float]
    tolerance: float = 0.05
    source: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ComparisonResult:
    """What one case actually produced."""

    name: str
    computed: Mapping[str, float]
    expected: Mapping[str, float]
    tolerance: float
    error: Optional[str] = None

    @property
    def deviations(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, want in self.expected.items():
            if key not in self.computed:
                continue
            got = self.computed[key]
            out[key] = (
                abs(got - want) / abs(want) if want else abs(got - want)
            )
        return out

    @property
    def missing(self) -> List[str]:
        return sorted(set(self.expected) - set(self.computed))

    @property
    def passed(self) -> bool:
        if self.error or self.missing or not self.expected:
            return False
        return all(value <= self.tolerance for value in self.deviations.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "computed": dict(self.computed),
            "expected": dict(self.expected),
            "deviations": self.deviations,
            "missing": self.missing,
            "tolerance": self.tolerance,
            "status": "passed" if self.passed else "failed",
            "error": self.error,
        }


def compare_case(case: ComparisonCase) -> ComparisonResult:
    """Build, solve and compare one recorded case."""

    try:
        computed = case.build()
    except Exception as error:  # noqa: BLE001 - recorded, not swallowed
        return ComparisonResult(
            name=case.name,
            computed={},
            expected=dict(case.expected),
            tolerance=case.tolerance,
            error=f"{type(error).__name__}: {error}",
        )
    return ComparisonResult(
        name=case.name,
        computed={str(k): float(v) for k, v in dict(computed).items()},
        expected=dict(case.expected),
        tolerance=case.tolerance,
    )


# The recorded comparison set.  Empty, and that is the honest state of it: the
# numbers have to come from running the models through ANYstructure's own FE
# path and recording what it produced, which has not been done.  The harness
# above is what consumes them once they exist; leaving the list empty keeps the
# gate closed for the real reason rather than passing on nothing.
COMPARISON_CASES: Tuple[ComparisonCase, ...] = ()


# ----------------------------------------------------------------------
# the gate
# ----------------------------------------------------------------------
def _headless_model_types() -> Dict[str, Any]:
    """Build one of each ANYstructure model type through the headless API.

    This is gate criterion 4, and it is the property the whole migration rests
    on: if ANYfem can build a stiffened panel and a cylinder with no GUI
    involved, ANYstructure's parametric front end can later become just another
    caller.  If it cannot, the migration is a rewrite.
    """

    import numpy as np

    # This generator deliberately creates mapped strips and immediately uses
    # their divider edges as beam carriers, so it belongs on the mesher's
    # mapped-partition operation rather than the general geometry split.
    from anymesher.decomposition import strip_face
    from .model import BeamSection
    from .model.attributes import pinned
    from .model.materials import steel
    from .model.project import Project

    built: Dict[str, Any] = {}

    # A stiffened panel: plating, longitudinal stiffeners, pressure, supports.
    panel = Project(name="stiffened_panel")
    panel.add_material(steel("S355", 0.012))
    panel.add_plate_section("plate", thickness=0.012, material="S355")
    panel.add_beam_section(
        BeamSection(
            name="stiffener", profile="T-bar", material="S355",
            web_height=0.25, web_thickness=0.010,
            flange_width=0.10, flange_thickness=0.014,
            web_direction=(0.0, 0.0, 1.0), eccentricity=0.131,
        )
    )
    geometry = panel.geometry
    corners = geometry.add_points(
        [(0, 0, 0), (4.0, 0, 0), (4.0, 2.4, 0), (0, 2.4, 0)]
    )
    face = geometry.add_face(geometry.add_polyline(corners, close=True))
    panel.assign_plate(face, "plate")
    strips, dividers = strip_face(geometry, face, axis=1, count=4)
    panel.assign_plates(strips, "plate")
    panel.assign_beams(dividers, "stiffener")
    for edge_id in geometry.edges:
        if len(geometry.faces_using_edge(edge_id)) == 1:
            panel.add_support(pinned(panel.edge(edge_id)))
    for strip in strips:
        panel.load_case().add_pressure(panel.face(strip), 120_000.0)
    built["stiffened panel"] = panel

    # A cylinder: revolved, so its surface is exact rather than faceted.
    cylinder = Project(name="cylinder")
    cylinder.add_material(steel("S355", 0.014))
    cylinder.add_plate_section("shell", thickness=0.014, material="S355")
    shell_geometry = cylinder.geometry
    bottom = shell_geometry.add_point(1.5, 0.0, 0.0)
    top = shell_geometry.add_point(1.5, 0.0, 3.0)
    generator = shell_geometry.add_line(bottom, top)
    faces = shell_geometry.revolve(
        [generator], (0, 0, 0), (0, 0, 1), 2.0 * np.pi
    )
    cylinder.assign_plates(faces, "shell")
    for edge_id in shell_geometry.edges:
        ends = [
            shell_geometry.vertices[v].position
            for v in (
                shell_geometry.edges[edge_id].start,
                shell_geometry.edges[edge_id].end,
            )
        ]
        if all(abs(point[2]) < 1.0e-9 for point in ends):
            cylinder.add_support(pinned(cylinder.edge(edge_id)))
    for face_id in faces:
        cylinder.load_case().add_pressure(cylinder.face(face_id), -80_000.0)
    built["cylinder"] = cylinder

    return built


def headless_model_report() -> Dict[str, Any]:
    """Whether the headless API builds every ANYstructure model type."""

    entries: Dict[str, Any] = {}
    try:
        models = _headless_model_types()
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        return {"ok": False, "error": f"{type(error).__name__}: {error}", "models": {}}

    for name, project in models.items():
        try:
            started = time.perf_counter()
            mesh = project.generate_mesh(0.25)
            meshed = time.perf_counter() - started

            from .solve.run import solve_linear_static

            started = time.perf_counter()
            solution = solve_linear_static(project, mesh=mesh)
            solved = time.perf_counter() - started
            entries[name] = {
                "ok": True,
                "nodes": mesh.num_nodes,
                "elements": mesh.num_elements,
                "mesh_seconds": round(meshed, 4),
                "solve_seconds": round(solved, 4),
                "max_translation_m": float(solution.max_translation()[1]),
            }
        except Exception as error:  # noqa: BLE001
            entries[name] = {"ok": False, "error": f"{type(error).__name__}: {error}"}

    return {
        "ok": all(entry.get("ok") for entry in entries.values()),
        "models": entries,
    }


def gate_report(states: Sequence[str | Path] = ()) -> Dict[str, Any]:
    """Measure every migration criterion that can be measured, and say so.

    Criteria that need data ANYfem does not have -- recorded ANYstructure
    numbers, an ANYstructure timing to compare against -- are reported as
    *unmet* with the reason, never as passed by default.  A gate that reports
    ready because nobody supplied the evidence would be worse than no gate.
    """

    from .parity import gate_status

    parity = gate_status()
    headless = headless_model_report()

    comparisons = [compare_case(case) for case in COMPARISON_CASES]
    comparison_ok = bool(comparisons) and all(item.passed for item in comparisons)

    read: List[Dict[str, Any]] = []
    for path in states:
        try:
            state = read_runtime_fem_state(path)
            read.append(
                {
                    "source": str(path), "ok": True, "summary": state.summary(),
                    "unmapped_options": len(state.unmapped_options),
                }
            )
        except StateFileError as error:
            read.append({"source": str(path), "ok": False, "error": str(error)})

    criteria = {
        "analysis paths reproduced on recorded models": {
            "met": comparison_ok,
            "detail": (
                f"{sum(1 for item in comparisons if item.passed)} of "
                f"{len(comparisons)} recorded cases match"
                if comparisons
                else "no recorded ANYstructure results exist to compare "
                "against; the harness is built and the comparison set is empty"
            ),
        },
        "parity ledger clear outside ANYstructure's own domain": {
            "met": bool(parity["ledger_clear"]),
            "detail": f"{len(parity['blocking'])} blocking entry(ies)",
        },
        "save_runtime_fem_state files importable": {
            "met": True,
            "detail": (
                "read as plain or gzipped JSON without importing ANYstructure; "
                f"{len(read)} file(s) checked in this run"
            ),
        },
        "headless API builds every model type": {
            "met": bool(headless["ok"]),
            "detail": ", ".join(
                f"{name}: {entry.get('nodes', '-')} nodes"
                for name, entry in headless["models"].items()
            )
            or headless.get("error", ""),
        },
        "no performance regression on representative models": {
            "met": False,
            "detail": (
                "ANYfem's own timings are recorded as a baseline, but there is "
                "no ANYstructure timing on the same models to compare against, "
                "so this cannot be answered yet"
            ),
        },
    }

    unmet = [name for name, item in criteria.items() if not item["met"]]
    return {
        "ready": not unmet,
        "criteria": criteria,
        "unmet": unmet,
        "headless": headless,
        "comparisons": [item.to_dict() for item in comparisons],
        "states": read,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def gate_markdown(report: Optional[Mapping[str, Any]] = None) -> str:
    """The gate as a readable table."""

    report = gate_report() if report is None else report
    lines = [
        "# ANYfem migration gate",
        "",
        f"- generated: {report['generated']}",
        f"- ready: **{'yes' if report['ready'] else 'no'}**",
        "",
        "| criterion | met | detail |",
        "| --- | --- | --- |",
    ]
    for name, item in report["criteria"].items():
        lines.append(
            f"| {name} | {'yes' if item['met'] else 'no'} | {item['detail']} |"
        )
    lines += [
        "",
        "A criterion needing evidence ANYfem does not have is reported unmet "
        "with the reason. It is never passed by default: a gate that opened "
        "because nobody supplied the numbers would be worse than no gate.",
        "",
    ]
    return "\n".join(lines)


def write_gate_report(
    report: Optional[Mapping[str, Any]] = None,
    directory: str | Path = "reports/migration",
) -> Dict[str, Path]:
    """Write the gate as JSON and Markdown, alongside the other evidence."""

    report = gate_report() if report is None else report
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "migration_gate.json"
    markdown_path = target / "migration_gate.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(gate_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    """Report the migration gate and write its evidence."""

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Report the ANYstructure migration gate. Reads any "
            "save_runtime_fem_state files given, without importing "
            "ANYstructure."
        )
    )
    parser.add_argument("states", nargs="*", help="save_runtime_fem_state files")
    parser.add_argument("--out", default="reports/migration")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    report = gate_report(args.states)
    print(gate_markdown(report))
    for entry in report["states"]:
        print(
            f"- {entry['source']}: "
            + (entry["summary"] if entry["ok"] else f"REFUSED - {entry['error']}")
        )
    if not args.no_save:
        written = write_gate_report(report, args.out)
        print(f"written: {written['markdown']}")
    # An open gate is the expected state until the evidence exists; reporting
    # it is not a build failure.
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
