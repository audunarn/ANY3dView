"""Resolve a meshed project into an ``anysolver`` FEModel.

This is the only place ANYfem speaks the solver's vocabulary.  Everything
upstream talks about plates, lines and points; everything downstream talks
about nodes, elements and DOFs.  Attribute resolution -- geometry entity to
node set -- happens here through the mesh association map.
"""

from __future__ import annotations

import builtins as _builtins
import os
import sys
import types
from dataclasses import dataclass, field
from itertools import permutations
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from anygeometry.entities import EntityRef
from anyloads import LoadError
from anysolver import (
    BeamElement,
    BoundaryCondition,
    CoupledBeamShellElement,
    FEModel,
    LegacyQ4DeprecationWarning,
    LegacyShellElement,
    QuadraticBeamElement,
    create_shell_element,
)
from anysolver import LoadCase as SolverLoadCase
from anymesher import (
    QUALIFIED_S3_FORMULATION_ID,
    QUALIFIED_S3_PRODUCTION_CONTRACT_ID,
    S3_QUALITY_CONTRACT_ID,
    S3_REPAIR_CONTRACT_ID,
    S3QualityError,
    assert_s3_admissible,
)

from ..mesh.mapped import Mesh
from ..model.attributes import LoadCase, DOF_NAMES, WetPressure, WetSurface
from ..model.connections import (
    Connection,
    ConnectionEquation,
    distributing_equations,
    null_moment_directions,
    rigid_equations,
)
from ..model.wetfield import WetFieldError, field_pressure, fluid_sign
from ..model import wetfield as _wetfield
from ..model.formulations import ShellFormulationPolicy
from ..model.project import Project, ProjectError
from ..model.regions import ElementFaceRef, MeshEntityRef, RegionRef

__all__ = ["BuiltModel", "ResolvedConnection", "build_fe_model"]


@dataclass
class BuiltModel:
    """A solver-ready model, with the mesh that produced it."""

    fe_model: FEModel
    load_case: SolverLoadCase | None
    mesh: Mesh
    project: Project
    combination: str | None = None
    shell_formulation_policy: ShellFormulationPolicy | None = field(
        default=None,
        init=False,
    )
    # Reference point of each connection, by connection id.
    connections: Dict[str, "ResolvedConnection"] = field(default_factory=dict)

    def __post_init__(self) -> None:
        policy = getattr(self.project, "shell_formulation_policy", None)
        if policy is not None and not isinstance(policy, ShellFormulationPolicy):
            raise TypeError(
                "shell_formulation_policy must be a ShellFormulationPolicy"
            )
        self.shell_formulation_policy = policy

    def node_dofs(self, node_id: int) -> List[int]:
        return self.fe_model.mesh.dof_manager.get_node_dofs(node_id)


def build_fe_model(
    project: Project,
    mesh: Mesh,
    load_case: str | LoadCase | None = "default",
    *,
    combination: str | None = None,
    apply_imperfections: bool = True,
    require_loads: bool = True,
    require_supports: bool = True,
) -> BuiltModel:
    """Build the solver model for one load case or combination."""

    automation = mesh.hybrid_diagnostics.get("automation", {})
    if isinstance(automation, Mapping) and automation.get("status") == "inspection_only":
        raise ProjectError(
            "inspection-only mesh is not admitted for solver assembly"
        )
    project.validate(
        require_loads=require_loads, require_supports=require_supports
    )

    fe_model = FEModel(name=project.name)
    for specification in project.materials.values():
        # ANYmaterial owns validation and construction. Registering the live
        # object supports both isotropic and orthotropic specifications.
        fe_model.register_material(specification.build())

    for node_id in sorted(mesh.nodes):
        x, y, z = mesh.nodes[node_id]
        fe_model.add_node(node_id, float(x), float(y), float(z))

    _add_shells(project, mesh, fe_model)
    _add_beams(project, mesh, fe_model)
    _add_couplings(project, mesh, fe_model)
    _add_supports(project, mesh, fe_model)
    connections = _add_connections(project, mesh, fe_model)
    _add_masses(project, mesh, fe_model)

    # Imperfections move the stress-free reference geometry, so they are
    # applied to the finished model before anything is assembled from it.
    if apply_imperfections and project.imperfections:
        fe_model = _apply_imperfections(project, mesh, fe_model)

    if combination is not None:
        cases = _build_combination(project, mesh, fe_model, combination)
        return BuiltModel(
            fe_model=fe_model,
            load_case=cases,
            mesh=mesh,
            project=project,
            combination=combination,
            connections=connections,
        )

    # A load-free build is used for deck handoff, neutral-mesh inspection, and
    # displacement-controlled analysis.  In the latter case the applied
    # action lives in the affine constraint RHS rather than in a LoadCase.
    has_prescribed_motion = any(
        value != 0.0
        for support in project.supports
        for value in support.constraints.values()
    )
    if (
        (not require_loads or has_prescribed_motion)
        and load_case == "default"
        and "default" not in project.load_cases
    ):
        resolved = None
    else:
        resolved = _resolve_load_case(project, load_case)
    # The UI may retain an empty named/default case after its last load was
    # deleted.  With a nonzero prescribed motion that case is not the driving
    # action and must not disguise the affine displacement path as a load-case
    # workflow in buckling/capacity metadata.
    if (
        has_prescribed_motion
        and resolved is not None
        and resolved.is_empty()
    ):
        resolved = None
    solver_case = (
        None if resolved is None else _build_load_case(project, mesh, resolved)
    )
    if solver_case is not None:
        fe_model.add_load_case(solver_case)

    return BuiltModel(
        fe_model=fe_model,
        load_case=solver_case,
        mesh=mesh,
        project=project,
        connections=connections,
    )


def _build_combination(
    project: Project, mesh: Mesh, fe_model: FEModel, name: str,
    *, _load_context: Any = None, _register: bool = True,
    _time_override: float | None = None,
) -> SolverLoadCase:
    """Fold a factored set of cases into one equivalent load case.

    The solver can assemble a combination directly, but folding it here keeps
    every analysis path taking a single load case, and keeps the factored
    result addressable in exactly the same way.
    """

    try:
        combination = project.combinations[name]
    except KeyError:
        raise ProjectError(f"no load combination named {name!r}") from None

    folded = SolverLoadCase(name=name)
    follower = False
    for case_name, factor in combination.factors.items():
        try:
            case = project.load_cases[case_name]
        except KeyError:
            raise ProjectError(
                f"combination {name!r} references undefined load case "
                f"{case_name!r}"
            ) from None
        follower = follower or project.resolve_case(case).follower_pressure
        if _time_override is not None:
            from copy import copy

            _refuse_definition(project, case)
            case = copy(project.resolve_case(case))
            case.time = _time_override
        _accumulate_case(
            project, mesh, case, float(factor), folded, _load_context=_load_context
        )

    if follower and any(
        not project.resolve_case(project.load_cases[case_name]).follower_pressure
        for case_name in combination.factors
    ):
        raise ProjectError(
            f"combination {name!r} mixes follower and dead pressure. The "
            "solver assembles a case in either the reference or the current "
            "configuration, not both; split the combination."
        )
    folded.follower_pressure = follower

    if _register:
        fe_model.add_load_case(folded)
    return folded


def _add_shells(project: Project, mesh: Mesh, fe_model: FEModel) -> None:
    qualified_ids: list[int] = []
    owner_normals: dict[int, np.ndarray] = {}
    if project.shell_formulation_policy.s3 == "e4-pl-s3-v2d":
        qualified_ids, owner_normals = _qualified_s3_owner_normals(project, mesh)
        if qualified_ids:
            try:
                assert_s3_admissible(
                    mesh,
                    element_ids=qualified_ids,
                    element_owner_normals=owner_normals,
                )
            except S3QualityError as error:
                raise ProjectError(
                    f"qualified S3 mesh admission failed: {error}"
                ) from error

    for face_id, element_ids in mesh.elements_of_face.items():
        section = project.plate_section_of(face_id)
        for element_id in element_ids:
            if element_id in mesh.quads:
                node_ids = mesh.quads[element_id]
            elif element_id in mesh.tris:
                node_ids = mesh.tris[element_id]
            else:
                raise ProjectError(
                    f"face {face_id} references missing shell element {element_id}"
                )
            formulation = project.shell_formulation_policy.for_node_count(
                len(node_ids)
            )
            extra: dict[str, Any] = {}
            if formulation == "e4-pl-s3-v2d":
                extra["reference_normal"] = owner_normals[int(element_id)]
            # The application policy is explicit and persisted, so the solver's
            # user-facing deprecation warning would otherwise be repeated once
            # per legacy Q4 element.  Keep the warning for direct solver callers.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", LegacyQ4DeprecationWarning)
                element = create_shell_element(
                    element_id,
                    list(node_ids),
                    material_name=section.material,
                    thickness=section.thickness,
                    formulation=formulation,
                    **extra,
                )
            expected_id = project.shell_formulation_policy.formulation_id_for_node_count(
                len(node_ids)
            )
            if formulation in {"e4-pl", "e4-pl-s3-v2d"}:
                if getattr(element, "formulation_id", None) != expected_id:
                    raise ProjectError(
                        f"shell element {element_id} formulation identity mismatch: "
                        f"expected {expected_id}"
                    )
            elif type(element) is not LegacyShellElement:
                raise ProjectError(
                    f"shell element {element_id} did not resolve to the "
                    "persisted legacy identity"
                )
            fe_model.add_element(element_id, element)


def _qualified_s3_owner_normals(
    project: Project,
    mesh: Mesh,
) -> tuple[list[int], dict[int, np.ndarray]]:
    """Validate and return the mesher's immutable physical-normal authority."""

    triangle_ids = tuple(
        sorted(
            int(element_id)
            for element_id, node_ids in mesh.tris.items()
            if len(node_ids) == 3
        )
    )
    if not triangle_ids:
        return [], {}

    preparation = getattr(mesh, "structural_preparation", None)
    if not isinstance(preparation, Mapping):
        raise ProjectError(
            "qualified S3 mesh authority is missing structural preparation"
        )
    record = preparation.get("qualified_s3")
    if not isinstance(record, Mapping):
        raise ProjectError(
            "qualified S3 mesh authority has a malformed preparation record"
        )
    # v2 (ANYmesher >= 0.5) adds the admission-floor and repair-target evidence
    # to the v1 record; v1 records persisted in older projects stay readable.
    v1_keys = {
        "admission",
        "authority_model",
        "contract_id",
        "element_ids",
        "element_owner_normals",
        "element_owner_sources",
        "formulation_id",
        "legacy_fallback",
        "nodal_normals",
        "quality_contract_id",
        "repair",
        "repair_contract_id",
        "schema",
        "status",
    }
    expected_keys_by_schema = {
        "anymesher.qualified-s3-production-preparation-v1": v1_keys,
        "anymesher.qualified-s3-production-preparation-v2": v1_keys
        | {"quality_policy", "quality_target"},
    }
    expected_keys = expected_keys_by_schema.get(record.get("schema"))
    if expected_keys is None or set(record) != expected_keys:
        raise ProjectError(
            "qualified S3 mesh authority has a malformed preparation record"
        )
    expected_identity = {
        "contract_id": QUALIFIED_S3_PRODUCTION_CONTRACT_ID,
        "formulation_id": QUALIFIED_S3_FORMULATION_ID,
        "legacy_fallback": "FORBIDDEN",
        "quality_contract_id": S3_QUALITY_CONTRACT_ID,
        "repair_contract_id": S3_REPAIR_CONTRACT_ID,
        "status": "ADMITTED",
    }
    if any(record[key] != value for key, value in expected_identity.items()):
        raise ProjectError("qualified S3 mesh authority identity or terminal differs")
    if record["schema"].endswith("-v2") and (
        not isinstance(record["quality_policy"], Mapping)
        or not isinstance(record["quality_target"], Mapping)
    ):
        raise ProjectError(
            "qualified S3 mesh authority has a malformed preparation record"
        )
    if not isinstance(record["admission"], Mapping):
        raise ProjectError("qualified S3 mesh admission evidence is malformed")
    if not isinstance(record["repair"], Mapping):
        raise ProjectError("qualified S3 mesh repair evidence is malformed")

    raw_ids = record["element_ids"]
    if (
        type(raw_ids) is not list
        or any(type(value) is not int for value in raw_ids)
        or raw_ids != list(triangle_ids)
    ):
        raise ProjectError(
            "qualified S3 mesh authority element scope differs from the mesh"
        )

    authority = record["authority_model"]
    authority_keys = {
        "prepared_revision",
        "scope",
        "source_model_id",
        "source_revision",
    }
    if not isinstance(authority, Mapping) or set(authority) != authority_keys:
        raise ProjectError("qualified S3 source-model authority is malformed")
    if (
        type(authority["prepared_revision"]) is not int
        or authority["scope"] != "PREPARED_GEOMETRY_ORIENTED_SHEET_FACE_USE"
        or type(authority["source_model_id"]) is not str
        or type(authority["source_revision"]) is not int
    ):
        raise ProjectError("qualified S3 source-model authority is malformed")
    project_model_id = str(getattr(project.geometry, "model_id", ""))
    project_revision = getattr(project.geometry, "revision", None)
    mesh_model_id = str(getattr(mesh, "geometry_model_id", ""))
    mesh_revision = getattr(mesh, "geometry_revision", None)
    if (
        not project_model_id
        or type(project_revision) is not int
        or not mesh_model_id
        or type(mesh_revision) is not int
        or authority["source_model_id"] != project_model_id
        or authority["source_model_id"] != mesh_model_id
        or authority["source_revision"] != project_revision
        or authority["source_revision"] != mesh_revision
    ):
        raise ProjectError("qualified S3 mesh authority is stale for the source geometry")

    shell_ids = tuple(sorted(int(value) for value in mesh.shells))
    expected_normal_keys = {str(value) for value in shell_ids}
    raw_normals = record["element_owner_normals"]
    if not isinstance(raw_normals, Mapping) or set(raw_normals) != expected_normal_keys:
        raise ProjectError(
            "qualified S3 physical owner-normal scope differs from the mesh"
        )
    raw_sources = record["element_owner_sources"]
    if not isinstance(raw_sources, Mapping) or set(raw_sources) != expected_normal_keys:
        raise ProjectError(
            "qualified S3 physical owner-source scope differs from the mesh"
        )
    for key, source in raw_sources.items():
        if (
            not isinstance(source, Mapping)
            or set(source) != {"face_id", "face_use_ids", "sheet_ids"}
            or type(source["face_id"]) is not int
            or type(source["face_use_ids"]) is not list
            or type(source["sheet_ids"]) is not list
            or not source["face_use_ids"]
            or len(source["face_use_ids"]) != len(source["sheet_ids"])
            or any(type(value) is not int for value in source["face_use_ids"])
            or any(type(value) is not int for value in source["sheet_ids"])
        ):
            raise ProjectError(
                f"qualified S3 physical owner source {key!r} is malformed"
            )

    nodal_normals = record["nodal_normals"]
    shell_node_ids = {
        int(node_id)
        for connectivity in mesh.shells.values()
        for node_id in connectivity
    }
    if (
        not isinstance(nodal_normals, Mapping)
        or set(nodal_normals) != {str(value) for value in shell_node_ids}
    ):
        raise ProjectError("qualified S3 nodal-normal scope differs from the mesh")
    for key, value in nodal_normals.items():
        _validated_unit_normal(value, label=f"qualified S3 nodal normal {key}")

    owner_normals = {
        element_id: _validated_unit_normal(
            raw_normals[str(element_id)],
            label=f"qualified S3 element {element_id} owner normal",
        )
        for element_id in triangle_ids
    }
    return list(triangle_ids), owner_normals


def _validated_unit_normal(value: object, *, label: str) -> np.ndarray:
    try:
        normal = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ProjectError(f"{label} must be a finite unit three-vector") from error
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise ProjectError(f"{label} must be a finite unit three-vector")
    length = float(np.linalg.norm(normal))
    if not np.isfinite(length) or abs(length - 1.0) > 1.0e-12:
        raise ProjectError(f"{label} must be a finite unit three-vector")
    return normal.copy()


def _add_beams(project: Project, mesh: Mesh, fe_model: FEModel) -> None:
    """Beam elements, linear or quadratic according to the mesh's order.

    Shell topology is resolved above through the project's persisted policy.
    The beam classes are separate, so their order choice is made here.
    """

    element_class = QuadraticBeamElement if mesh.is_quadratic else BeamElement
    # Native mesh edge maps also retain source-boundary aliases for loads and
    # supports. Only section-bearing edges define physical beam ownership.
    owners = {}
    for edge_id, element_ids in mesh.elements_of_edge.items():
        if edge_id not in project.edge_sections:
            continue
        for element_id in dict.fromkeys(element_ids):
            if element_id not in mesh.beams:
                continue
            if element_id in owners and owners[element_id] != edge_id:
                raise ProjectError(f"beam element {element_id} has multiple section-bearing edge owners")
            owners[element_id] = edge_id
    unassigned = set(mesh.beams) - owners.keys()
    if unassigned:
        raise ProjectError(f"beam elements have no section-bearing edge owner: {sorted(unassigned)[:10]}")
    for edge_id, element_ids in mesh.elements_of_edge.items():
        if edge_id not in project.edge_sections:
            continue
        section = project.beam_section_of(edge_id)
        _axis, _local_y, local_z = project.beam_section_frame(edge_id)
        properties = section.properties(orientation=local_z)
        for element_id in dict.fromkeys(element_ids):
            if element_id not in mesh.beams:
                continue
            fe_model.add_element(
                element_id,
                element_class(
                    element_id,
                    list(mesh.beams[element_id]),
                    material_name=section.material,
                    cross_section=dict(properties),
                ),
            )


def _add_couplings(project: Project, mesh: Mesh, fe_model: FEModel) -> None:
    """Tie eccentric stiffener nodes back to the plating.

    A kinematic constraint, not a stiff spring: the solver derives the offset
    from the node positions and enforces ``u_beam = u_shell + theta_shell x r``
    exactly, so there is no penalty parameter to get wrong.
    """

    material = next(iter(project.materials), "default")
    # Adjacent geometry edges may represent consecutive spans of one physical
    # stiffener.  ANYmesher deliberately reuses their shared offset node, and
    # older/generated meshes can consequently carry the same end-station
    # coupling once for each span.  One dependent beam-node DOF may have only
    # one MPC owner, so collapse semantically identical records here at the
    # solver boundary.  Do not silently choose between genuinely different
    # master definitions: that would change structural connectivity.
    coupling_by_beam_node: dict[
        int,
        tuple[int, tuple[tuple[int, float], ...], tuple[float, float, float]],
    ] = {}
    for element_id, coupling in mesh.couplings.items():
        if hasattr(coupling, "beam_node"):
            beam_node = int(coupling.beam_node)
            plate_nodes = tuple(int(node) for node in coupling.plate_nodes)
            weights = tuple(float(value) for value in coupling.weights)
            eccentricity = tuple(float(value) for value in coupling.eccentricity)
        else:
            # Wheels from the pre-extraction ANYfem line stored a pair. Keep
            # accepting one so an application-provided legacy Mesh still builds.
            beam_node, shell_node = coupling
            beam_node = int(beam_node)
            plate_nodes = (int(shell_node),)
            weights = (1.0,)
            eccentricity = tuple(
                float(value)
                for value in (mesh.nodes[beam_node] - mesh.nodes[int(shell_node)])
            )

        masters = tuple(
            sorted(
                (int(node), float(weight))
                for node, weight in zip(plate_nodes, weights)
            )
        )
        previous = coupling_by_beam_node.get(beam_node)
        if previous is not None:
            previous_id, previous_masters, previous_eccentricity = previous
            same_masters = (
                tuple(node for node, _weight in previous_masters)
                == tuple(node for node, _weight in masters)
                and np.allclose(
                    tuple(weight for _node, weight in previous_masters),
                    tuple(weight for _node, weight in masters),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
            same_eccentricity = np.allclose(
                previous_eccentricity,
                eccentricity,
                rtol=0.0,
                atol=1.0e-12,
            )
            if same_masters and same_eccentricity:
                continue
            raise ProjectError(
                f"offset beam node {beam_node} has conflicting shell coupling "
                f"records {previous_id} and {element_id}; the same dependent "
                "node cannot be tied to two different plate interpolations"
            )
        coupling_by_beam_node[beam_node] = (
            int(element_id),
            masters,
            tuple(float(value) for value in eccentricity),
        )

        if len(plate_nodes) == 1:
            element = CoupledBeamShellElement(
                element_id,
                beam_node_id=beam_node,
                shell_node_id=plate_nodes[0],
                material_name=material,
            )
        else:
            # This remains solver functionality: ANYmesher describes the
            # interpolation, while ANYsolver turns it into exact MPC equations.
            from anysolver import InterpolatedBeamShellMPCElement

            element = InterpolatedBeamShellMPCElement(
                element_id,
                beam_node_id=beam_node,
                shell_node_ids=list(plate_nodes),
                shape_weights=np.asarray(weights, dtype=float),
                eccentricity=np.asarray(eccentricity, dtype=float),
                material_name=material,
            )
        fe_model.add_element(element_id, element)


def _implied_zero_clamp_nodes(project: Project, mesh: Mesh) -> set[int]:
    """Zero-slave clamps already imposed by fully fixed MPC masters.

    The implication is established only from exact global homogeneous full
    clamps. Zero-valued slave conditions implied by those full clamps are
    redundant. Other partial, nonzero and local-frame conditions stay audited.
    """
    clamped = set()
    for support in project.supports:
        if (support.coordinate_system_id == "global"
                and set(support.constraints) == set(DOF_NAMES)
                and all(value == 0. for value in support.constraints.values())):
            for target in _attribute_targets(project, support.region, support.ref):
                clamped.update(_nodes_on_target(mesh, target, constrained=True))
    implied = set()
    for coupling in mesh.couplings.values():
        if hasattr(coupling, "beam_node"):
            beam_node = int(coupling.beam_node)
            masters = [int(node) for node,weight in zip(coupling.plate_nodes,coupling.weights) if weight != 0.]
        else:
            beam_node, shell_node = coupling
            masters = [int(shell_node)]
        if beam_node in clamped and masters and all(node in clamped for node in masters):
            implied.add(beam_node)
    return implied


def _add_supports(project: Project, mesh: Mesh, fe_model: FEModel) -> None:
    implied_zero = _implied_zero_clamp_nodes(project, mesh)
    for support in project.supports:
        targets = _attribute_targets(project, support.region, support.ref)
        # A restraint holds a physical location, so it reaches the offset
        # stiffener nodes as well as the plating.
        node_ids = sorted(
            {
                node_id
                for target in targets
                for node_id in _nodes_on_target(mesh, target, constrained=True)
            }
        )
        if not node_ids:
            raise ProjectError(
                f"support {support.name!r} has an empty or unresolved scope in "
                "the active mesh"
            )
        system = _coordinate_system(project, support.coordinate_system_id)
        if system.id == "global":
            if all(value == 0. for value in support.constraints.values()):
                node_ids = [node for node in node_ids if node not in implied_zero]
                if not node_ids:
                    continue
            fe_model.add_boundary_condition(
                BoundaryCondition(
                    name=support.name,
                    node_ids=list(node_ids),
                    dof_constraints=dict(support.constraints),
                )
            )
            continue

        # Local prescribed values are affine equations in the global nodal
        # degrees of freedom.  Pick distinct, well-conditioned pivots for all
        # local components in one translation/rotation block; a naive
        # "largest coefficient" pivot can choose the same global DOF twice for
        # a rotated basis and make an otherwise valid support look dependent.
        for node_id in node_ids:
            basis = system.basis_at(mesh.nodes[node_id])
            dofs = fe_model.mesh.dof_manager.get_node_dofs(node_id)
            for prefix, offset in (("u", 0), ("r", 3)):
                rows = [
                    ("xyz".index(name[1]), name, float(value))
                    for name, value in support.constraints.items()
                    if name.startswith(prefix)
                ]
                if not rows:
                    continue
                pivots = _constraint_pivots(basis, [axis for axis, _name, _value in rows])
                for (axis, name, value), pivot in zip(rows, pivots):
                    terms = tuple(
                        (dofs[offset + component], float(basis[component, axis]))
                        for component in range(3)
                        if abs(float(basis[component, axis])) > 1.0e-13
                    )
                    fe_model.add_constraint_equation(
                        terms=terms,
                        rhs=value,
                        source_id=f"{support.id}:{node_id}:{name}",
                        dependent_dof=dofs[offset + pivot],
                    )



# ----------------------------------------------------------------------
# connections
# ----------------------------------------------------------------------
@dataclass
class ResolvedConnection:
    """A connection resolved against one mesh.

    A pure function of the project and the mesh, so the solver build and every
    load case built later agree on the reference node without sharing state.
    """

    connection: Connection
    reference_node: int
    reference: np.ndarray
    node_ids: tuple[int, ...]
    positions: np.ndarray
    weights: np.ndarray
    equations: list[ConnectionEquation]
    # Indices (0-5) of the loaded node's DOFs that can carry a load: the ones
    # some equation involves (a spring connection: the ones with a spring).
    # The others are held at zero, and a load on one is refused.
    active: frozenset[int]
    # Unit moments (3 x k) the selection cannot transmit; empty unless the
    # connection distributes forces over collinear geometry.
    null_moments: np.ndarray
    # Spring connections only: the inner node the selection is tied to, and the
    # DOFs of it that its equations involve.  The loaded node is joined to it
    # by springs.  Other connections tie the selection to the loaded node.
    section_node: int | None = None
    tie_active: frozenset[int] = frozenset()

    @property
    def tie_node(self) -> int:
        """The node the tie equations are written for."""

        return self.reference_node if self.section_node is None else self.section_node

    @property
    def inactive(self) -> tuple[int, ...]:
        return tuple(index for index in range(6) if index not in self.active)

    @property
    def tie_inactive(self) -> tuple[int, ...]:
        return tuple(index for index in range(6) if index not in self.tie_active)


def _connection_nodes(
    project: Project, mesh: Mesh, connection: Connection
) -> tuple[dict[int, float], bool]:
    """Nodes a connection ties, with their tributary length/area/count.

    The weight of a node is its consistent share of the selected lines or
    plates, so it does not change when the mesh is refined.  Beam nodes that
    are already dependent on the plating (offset stiffener nodes) are left out:
    they follow the node they are tied to.  The flag says whether every weight
    is a real tributary measure (not a plain count).
    """

    targets = _attribute_targets(
        project,
        connection.region,
        connection.ref,
        geometry_kinds=("vertex", "edge", "face"),
        mesh_kinds=("node",),
    )
    dependent = _beam_coupling_nodes(mesh)
    weights: dict[int, float] = {}
    tributary = True
    for target in targets:
        if isinstance(target, EntityRef) and target.kind == "edge":
            shares = [
                (node, float(force[0]))
                for node, force in _line_load_to_nodes(
                    mesh, target, np.array([1.0, 0.0, 0.0])
                )
            ]
        elif isinstance(target, EntityRef) and target.kind == "face":
            shares = [
                (node, float(force[0]))
                for node, force in _traction_to_nodes(
                    mesh, target, np.array([1.0, 0.0, 0.0])
                )
            ]
        else:
            shares = [(int(node), 1.0) for node in _nodes_on_target(mesh, target)]
            # A bare node or point has no length or area to share.
            tributary = tributary and isinstance(target, EntityRef)
        for node, share in shares:
            if node in dependent:
                continue
            weights[node] = weights.get(node, 0.0) + share
    return weights, tributary


def _beam_coupling_nodes(mesh: Mesh) -> set[int]:
    nodes: set[int] = set()
    for coupling in mesh.couplings.values():
        nodes.add(
            int(coupling.beam_node if hasattr(coupling, "beam_node") else coupling[0])
        )
    return nodes


def _support_dofs(project: Project, mesh: Mesh) -> dict[tuple[int, int], str]:
    """Node DOFs a support holds, with the support's name.

    A support in a local axis system constrains a mix of global DOFs, so every
    DOF of the same kind (translation or rotation) counts as held.
    """

    held: dict[tuple[int, int], str] = {}
    for support in project.supports:
        targets = _attribute_targets(project, support.region, support.ref)
        nodes = {
            node
            for target in targets
            for node in _nodes_on_target(mesh, target, constrained=True)
        }
        local = support.coordinate_system_id != "global"
        indices: set[int] = set()
        for name in support.constraints:
            index = DOF_NAMES.index(name)
            if not local:
                indices.add(index)
            else:
                indices.update(range(0, 3) if index < 3 else range(3, 6))
        for node in nodes:
            for index in indices:
                held.setdefault((node, index), support.name)
    return held


def _resolve_connections(project: Project, mesh: Mesh) -> Dict[str, ResolvedConnection]:
    if not project.connections:
        return {}
    next_node = (max(mesh.nodes) if mesh.nodes else 0) + 1
    held = _support_dofs(project, mesh)
    owners: dict[tuple[int, int], str] = {}
    resolved: Dict[str, ResolvedConnection] = {}
    for connection in project.connections:
        weights, tributary = _connection_nodes(project, mesh, connection)
        if not weights:
            raise ProjectError(
                f"connection {connection.name!r} has an empty or unresolved "
                "scope in the active mesh"
            )
        node_ids = tuple(sorted(weights))
        positions = np.array([mesh.nodes[node] for node in node_ids], dtype=float)
        uniform = connection.weighting == "uniform" or not tributary
        values = (
            np.ones(len(node_ids))
            if uniform
            else np.array([weights[node] for node in node_ids], dtype=float)
        )
        if np.any(values <= 0.0):
            raise ProjectError(
                f"connection {connection.name!r}: the tributary share of a node "
                "is not positive on this element order; set the connection's "
                "weighting to 'uniform'"
            )
        reference = _connection_reference(project, connection, positions, values)
        null = np.zeros((3, 0))
        if connection.tied_by == "rigid":
            equations = rigid_equations(node_ids, positions, reference, connection.dofs)
            for equation in equations:
                owner, index = equation.dependent
                key = (int(owner), index)
                if key in held:
                    raise ProjectError(
                        f"connection {connection.name!r} ties {DOF_NAMES[index]} of "
                        f"node {owner}, which support {held[key]!r} also "
                        "constrains. Leave the supported entities out of a rigid "
                        "connection, tie fewer DOFs, or use a distributing "
                        "connection."
                    )
                if key in owners:
                    raise ProjectError(
                        f"connections {owners[key]!r} and {connection.name!r} both "
                        f"tie {DOF_NAMES[index]} of node {owner}; a node can "
                        "belong to one rigid connection"
                    )
                owners[key] = connection.name
        else:
            try:
                equations, inertia = distributing_equations(
                    node_ids, positions, values, reference, connection.dofs
                )
            except ValueError as error:
                raise ProjectError(f"connection {connection.name!r}: {error}") from None
            if connection.rotations:
                directions = null_moment_directions(inertia)
                null = np.zeros((3, directions.shape[1]))
                for column in range(directions.shape[1]):
                    for local, axis in enumerate(connection.rotations):
                        null[axis, column] = directions[local, column]
        tie_active = frozenset(
            index
            for equation in equations
            for owner, index, _ in equation.terms
            if owner == "ref"
        )
        if connection.kind == "spring":
            section_node, reference_node = next_node, next_node + 1
            next_node += 2
            active = frozenset(
                index for index, value in enumerate(connection.stiffness) if value > 0.0
            )
        else:
            section_node, reference_node = None, next_node
            next_node += 1
            active = tie_active
        resolved[connection.id] = ResolvedConnection(
            connection=connection,
            reference_node=reference_node,
            reference=reference,
            node_ids=node_ids,
            positions=positions,
            weights=values,
            equations=equations,
            active=active,
            null_moments=null,
            section_node=section_node,
            tie_active=tie_active,
        )
    return resolved


def _connection_reference(
    project: Project,
    connection: Connection,
    positions: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """The reference point: from the geometry, or from the nodes without one."""

    if connection.reference == "point" or not project.mesh_only:
        return project.connection_reference_point(connection)
    if connection.reference == "bounding_box":
        return 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    return (weights[:, None] * positions).sum(axis=0) / weights.sum()


def _add_connections(
    project: Project, mesh: Mesh, fe_model: FEModel
) -> Dict[str, ResolvedConnection]:
    resolved = _resolve_connections(project, mesh)
    manager = fe_model.mesh.dof_manager
    for item in resolved.values():
        connection = item.connection
        for node in filter(None, (item.section_node, item.reference_node)):
            fe_model.add_node(node, *(float(v) for v in item.reference))
        reference_dofs = manager.get_node_dofs(item.tie_node)

        def dof(owner: object, index: int) -> int:
            if owner == "ref":
                return reference_dofs[index]
            return manager.get_node_dofs(int(owner))[index]

        for number, equation in enumerate(item.equations):
            owner, index = equation.dependent
            fe_model.add_constraint_equation(
                terms=tuple(
                    (dof(term_owner, term_index), coefficient)
                    for term_owner, term_index, coefficient in equation.terms
                ),
                rhs=0.0,
                source_id=f"connection:{connection.id}:{number}",
                dependent_dof=dof(owner, index),
            )
        # DOFs no equation (or spring) reaches carry no stiffness; hold them so
        # the system stays solvable, and refuse a load on them.
        for node, unused in (
            (item.tie_node, item.tie_inactive),
            (item.reference_node, item.inactive if item.section_node else ()),
        ):
            if unused:
                fe_model.add_boundary_condition(
                    BoundaryCondition(
                        name=f"{connection.name} unused reference DOFs",
                        node_ids=[node],
                        dof_constraints={DOF_NAMES[index]: 0.0 for index in unused},
                    )
                )
        if item.section_node is not None:
            from .spring import SpringConnectorElement

            element_id = max(fe_model.mesh.elements, default=0) + 1
            fe_model.add_element(
                element_id,
                SpringConnectorElement(
                    element_id,
                    item.section_node,
                    item.reference_node,
                    connection.stiffness,
                ),
            )
    return resolved


def _accumulate_connection_loads(
    project: Project,
    mesh: Mesh,
    case: LoadCase,
    factor: float,
    target: SolverLoadCase,
    *,
    _resolved: Any = None,
) -> None:
    resolved = _resolve_connections(project, mesh) if _resolved is None else _resolved
    case_factor = factor
    for load in case.connection_loads:
        factor = case_factor * _envelope_factor(case, load)
        if factor == 0.0:
            continue
        item = resolved.get(load.connection)
        if item is None:
            try:
                item = resolved.get(project.connection_for(load.connection).id)
            except ProjectError:
                item = None
        if item is None:
            raise ProjectError(
                f"connection load {load.name!r} in case {case.name!r} names a "
                f"connection that does not exist ({load.connection!r})"
            )
        name = item.connection.name
        components = np.concatenate([load.force, load.moment])
        scale = float(np.max(np.abs(components)))
        for index, value in enumerate(components):
            if value != 0.0 and index not in item.active:
                raise ProjectError(
                    f"connection {name!r} does not transmit a {DOF_NAMES[index]} "
                    f"load: that component is not among its tied DOFs "
                    f"({', '.join(item.connection.dofs)})"
                )
        if item.null_moments.shape[1]:
            carried = item.null_moments.T @ load.moment
            if scale > 0.0 and float(np.max(np.abs(carried))) > 1.0e-9 * scale:
                raise ProjectError(
                    f"connection {name!r} distributes forces over geometry that "
                    "cannot carry a moment about its own axis, so part of this "
                    "moment would be lost. Select geometry that is not "
                    "collinear, or use a rigid connection."
                )
        _add_nodal(target, item.reference_node, factor * load.force, factor * load.moment)


def _constraint_pivots(basis: np.ndarray, local_axes: Sequence[int]) -> tuple[int, ...]:
    """Choose distinct global pivot components for independent local rows."""

    best: tuple[float, tuple[int, ...]] | None = None
    for candidates in permutations(range(3), len(local_axes)):
        magnitudes = [abs(float(basis[pivot, axis])) for pivot, axis in zip(candidates, local_axes)]
        if any(value <= 1.0e-12 for value in magnitudes):
            continue
        score = float(np.prod(magnitudes))
        if best is None or score > best[0]:
            best = (score, tuple(candidates))
    if best is None:
        raise ProjectError("local support directions are linearly dependent")
    return best[1]


def _resolve_load_case(
    project: Project, load_case: str | LoadCase | None
) -> LoadCase | None:
    if load_case is None:
        return None
    if isinstance(load_case, LoadCase):
        return load_case
    try:
        case = project.load_cases[load_case]
    except KeyError:
        raise ProjectError(f"no load case named {load_case!r}") from None
    _refuse_definition(project, case)
    return case


def _refuse_definition(project: Project, case: LoadCase) -> None:
    """A series definition is not a case to solve.

    A time-less *template* (an equation reading ``t`` on a case with no time) is
    left to the evaluator, whose error names the unknown variable.
    """

    if case.times is not None:
        steps = project.series_children(case.name)
        examples = ", ".join(repr(step.name) for step in steps[:2])
        raise ProjectError(
            f"{case.name!r} is a load series definition, not a load case: its "
            f"loads are applied at each of its {len(steps)} time steps. Solve a "
            f"step ({examples}{', ...' if len(steps) > 2 else ''}) or run "
            "'Batch linear static' to solve them all."
        )


def _envelope_factor(case: LoadCase, load: Any) -> float:
    """What a load's envelope makes of it in ``case``: 0 when it is not acting."""

    try:
        return case.envelope_factor(getattr(load, "envelope", None))
    except LoadError as error:
        raise ProjectError(
            f"load {getattr(load, 'name', type(load).__name__)!r} in load case "
            f"{case.name!r}: {error}"
        ) from None


def _build_load_case(
    project: Project, mesh: Mesh, case: LoadCase, *, _load_context: Any = None
) -> SolverLoadCase:
    solver_case = SolverLoadCase(name=case.name)
    _accumulate_case(project, mesh, case, 1.0, solver_case, _load_context=_load_context)
    solver_case.follower_pressure = bool(project.resolve_case(case).follower_pressure)
    return solver_case


def _accumulate_case(
    project: Project,
    mesh: Mesh,
    case: LoadCase,
    factor: float,
    target: SolverLoadCase,
    *,
    _load_context: Any = None,
) -> None:
    """Add one case's loads into a solver load case, scaled by ``factor``.

    Accumulating rather than assigning matters: the solver's
    ``add_pressure_load`` overwrites an element's pressure, so a combination
    that puts two pressures on the same plate would otherwise keep only the
    last one.

    A step of a load series is resolved to its definition's loads at its own
    time, and each load is scaled by its envelope: zero outside its window.
    """

    _refuse_definition(project, case)
    case = project.resolve_case(case)
    case_factor = factor
    attribute_targets = (
        _attribute_targets if _load_context is None else _load_context.attribute_targets
    )
    nodes_on_target = (
        _nodes_on_target if _load_context is None else _load_context.nodes_on_target
    )
    elements_on_target = (
        _elements_on_target if _load_context is None else _load_context.elements_on_target
    )

    for load in case.point_loads:
        factor = case_factor * _envelope_factor(case, load)
        if factor == 0.0:
            continue
        targets = attribute_targets(
            project,
            load.region,
            load.ref,
            geometry_kinds=("vertex",),
            mesh_kinds=("node",),
        )
        node_ids = sorted(
            {
                node_id
                for scoped in targets
                for node_id in nodes_on_target(mesh, scoped)
            }
        )
        if not node_ids:
            raise ProjectError(
                f"point load {load.id!r} has an empty or unresolved scope in "
                "the active mesh"
            )
        share = 1.0 / len(node_ids) if load.distribution_policy == "total_distributed" else 1.0
        system = _coordinate_system(project, load.coordinate_system_id)
        for node_id in node_ids:
            position = mesh.nodes[node_id]
            force = system.to_global(load.force, position)
            moment = system.to_global(load.moment, position)
            _add_nodal(
                target,
                node_id,
                factor * share * force,
                factor * share * moment,
            )

    for load in case.pressures:
        factor = case_factor * _envelope_factor(case, load)
        if factor == 0.0:
            continue
        targets = attribute_targets(
            project,
            load.region,
            load.ref,
            geometry_kinds=("face",),
            mesh_kinds=("element", "element_face"),
        )
        element_ids = sorted(
            {
                element_id
                for scoped in targets
                for element_id in elements_on_target(mesh, scoped)
            }
        )
        if not element_ids:
            raise ProjectError(
                f"pressure {load.id!r} has an empty or unresolved scope in the "
                "active mesh"
            )
        for element_id in element_ids:
            target.pressure_loads[element_id] = (
                target.pressure_loads.get(element_id, 0.0) + factor * load.value
            )

    if case.wet_pressures:
        _accumulate_wet_pressures(
            project, mesh, case, case_factor, target, _load_context=_load_context
        )

    if case.connection_loads:
        _accumulate_connection_loads(
            project, mesh, case, case_factor, target,
            _resolved=None if _load_context is None else _load_context.connections,
        )

    for load in case.line_loads:
        factor = case_factor * _envelope_factor(case, load)
        if factor == 0.0:
            continue
        system = _coordinate_system(project, load.coordinate_system_id)
        targets = attribute_targets(
            project,
            load.region,
            load.ref,
            geometry_kinds=("edge",),
            mesh_kinds=("element",),
        )
        contributions: Dict[int, np.ndarray] = {}
        for scoped in targets:
            for node_id, local_force in _line_target_to_nodes(
                mesh, scoped, load.force_per_length
            ):
                contributions.setdefault(node_id, np.zeros(3))
                contributions[node_id] += system.to_global(
                    local_force, mesh.nodes[node_id]
                )
        if not contributions:
            raise ProjectError(
                f"line load {load.id!r} has an empty or unresolved scope in "
                "the active mesh"
            )
        for node_id, force in contributions.items():
            _add_nodal(target, node_id, factor * force, np.zeros(3))

    for load in case.surface_tractions:
        factor = case_factor * _envelope_factor(case, load)
        if factor == 0.0:
            continue
        system = _coordinate_system(project, load.coordinate_system_id)
        targets = attribute_targets(
            project,
            load.region,
            load.ref,
            geometry_kinds=("face",),
            mesh_kinds=("element", "element_face"),
        )
        contributions: Dict[int, np.ndarray] = {}
        for scoped in targets:
            for node_id, local_force in _traction_target_to_nodes(
                mesh, scoped, load.traction
            ):
                contributions.setdefault(node_id, np.zeros(3))
                contributions[node_id] += system.to_global(
                    local_force, mesh.nodes[node_id]
                )
        if not contributions:
            raise ProjectError(
                f"surface traction {load.id!r} has an empty or unresolved "
                "scope in the active mesh"
            )
        for node_id, force in contributions.items():
            _add_nodal(target, node_id, factor * force, np.zeros(3))

    if case.gravity is not None:
        # Gravity is an acceleration field, so combining cases sums the fields
        # rather than the resulting forces.
        system = _coordinate_system(project, case.gravity_coordinate_system_id)
        if system.kind == "cylindrical":
            raise ProjectError(
                "a cylindrical gravity/acceleration field varies by position "
                "and cannot be represented by the solver's uniform body-load "
                "field; use scoped nodal or distributed loads"
            )
        current = (
            np.zeros(3) if target.gravity is None else np.asarray(target.gravity)
        )
        target.gravity = current + case_factor * system.to_global(case.gravity)


def _wet_elements(
    project: Project, mesh: Mesh, surfaces: Sequence[WetSurface]
) -> tuple[Dict[int, WetSurface], Dict[int, list[int]]]:
    """Shell elements on the given wet surfaces.

    Returns ``(owner, by_face)``: each element's wet surface, and the elements
    of each geometric plate that is reached as a whole plate (needed to map a
    pressure to a plate). A plate or element on two wet surfaces is refused:
    its fluid side would be ambiguous.
    """

    owner: Dict[int, WetSurface] = {}
    by_face: Dict[int, list[int]] = {}
    for surface in surfaces:
        targets = _attribute_targets(
            project,
            surface.region,
            surface.ref,
            geometry_kinds=("face",),
            mesh_kinds=("element", "element_face"),
        )
        found = 0
        for scoped in targets:
            element_ids = _elements_on_target(mesh, scoped)
            if isinstance(scoped, EntityRef):
                by_face.setdefault(int(scoped.id), []).extend(element_ids)
            for element_id in element_ids:
                element_id = int(element_id)
                if element_id not in mesh.quads and element_id not in mesh.tris:
                    raise ProjectError(
                        f"wet surface {surface.name!r} reaches element "
                        f"{element_id}, which is not a plate element"
                    )
                previous = owner.get(element_id)
                if previous is not None and previous.id != surface.id:
                    raise ProjectError(
                        f"element {element_id} is on two wet surfaces "
                        f"({previous.name!r} and {surface.name!r}); its fluid "
                        "side would be ambiguous"
                    )
                owner[element_id] = surface
                found += 1
        if not found:
            raise ProjectError(
                f"wet surface {surface.name!r} ({surface.id!r}) has an empty "
                "or unresolved scope in the active mesh"
            )
    return owner, by_face


def _element_centroid_normal(mesh: Mesh, element_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Corner-mean centroid and unit normal, in the solver's node order.

    The normal is the one the solver's positive pressure acts along.
    """

    # A time history asks for the same elements once per step; the geometry is
    # fixed for a mesh, so keep it with the mesh.
    try:
        cache = mesh.__dict__.setdefault("_anyfem_centroid_normal", {})
    except AttributeError:
        cache = {}
    found = cache.get(element_id)
    if found is not None:
        return found
    corners = np.array([mesh.nodes[node] for node in mesh.corners_of(element_id)])
    if len(corners) == 3:
        normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    else:
        normal = np.cross(corners[2] - corners[0], corners[3] - corners[1])
    length = float(np.linalg.norm(normal))
    if length == 0.0:
        raise ProjectError(f"element {element_id} has zero area")
    cache[element_id] = (corners.mean(axis=0), normal / length)
    return cache[element_id]


@dataclass(frozen=True)
class WetElementPressure:
    """One wet shell element's normal-signed pressure and where it acts."""

    element_id: int
    pressure: float
    centroid: np.ndarray
    normal: np.ndarray


def wet_element_pressures(
    project: Project,
    mesh: Mesh,
    case: LoadCase,
    *,
    limit: int | None = None,
    _load_context: Any = None,
) -> List[WetElementPressure]:
    """Evaluate a case's wet pressures per shell element.

    The pressure is sampled once per element at its corner-mean centroid and
    applied uniformly over the element, the same representation the solver
    uses for a uniform pressure. A pressure that varies strongly across an
    element (a waterline through it) is therefore resolved only to the mesh
    size. Several wet pressures on one element are summed.

    ``limit`` keeps an evenly spread subset of that many elements per wet
    pressure, for drawing; the solver always takes every element.
    """
    accumulated = _wet_pressure_accumulation(
        project, mesh, case, limit=limit, _load_context=_load_context
    )
    made = []
    for key in sorted(accumulated):
        pressure, centroid, normal, shared = accumulated[key]
        if shared:
            # M31b: records own fresh writable arrays, as on the uncached path.
            centroid, normal = centroid.copy(), normal.copy()
        made.append(WetElementPressure(key, pressure, centroid, normal))
    return made


# M31d: NumPy's default error state as an immutable value.
_DEFAULT_ERROR_ITEMS = (("divide", "warn"), ("invalid", "warn"), ("over", "warn"), ("under", "ignore"))


def _sign_producers() -> tuple:
    """M31b/M31c: everything that computes an element's fluid side on the uncached
    path, for comparison by identity: the two function bindings on the call path,
    their code, defaults and closures, and for every name their code uses the
    module or builtin binding and, for a module, its attributes of those names."""
    made = []
    builtin_names = vars(_builtins)
    for function in (fluid_sign, getattr(_wetfield, "_fluid_sign", None)):
        code = getattr(function, "__code__", None)
        names = getattr(function, "__globals__", None) or {}
        kwdefaults = getattr(function, "__kwdefaults__", None) or {}
        made.extend((function, code, getattr(function, "__defaults__", None),
                     getattr(function, "__closure__", None)))
        made.extend(sorted(kwdefaults))
        made.extend(kwdefaults[key] for key in sorted(kwdefaults))
        if code is None:
            continue
        for name in code.co_names:
            value = names[name] if name in names else builtin_names.get(name)
            made.append(value)
            if isinstance(value, types.ModuleType):
                made.extend(vars(value).get(attribute) for attribute in code.co_names)
    return tuple(made)


def _same_producers(left, right) -> bool:
    return left is not None and len(left) == len(right) and all(a is b for a, b in zip(left, right))


def _library_own(value, package: str) -> bool:
    module = getattr(value, "__module__", None)
    return type(value) is np.ufunc or (type(module) is str and module.split(".")[0] == package)


def _capture_sign_chain():
    """M31c/M31d: the fluid-side chain as loaded, if it is the libraries' own: each
    function is the one its module defines (no closure, compiled from that module's
    file) and the module attributes its code names belong to that module's package;
    otherwise None, and no side is ever reused."""
    expected = ((fluid_sign, _wetfield, "fluid_sign"),
                (getattr(_wetfield, "_fluid_sign", None), sys.modules.get("anyloads.fluid"), "fluid_sign"))
    for function, module, name in expected:
        if type(function) is not types.FunctionType or module is None or function.__closure__ is not None:
            return None
        if function.__module__ != module.__name__ or vars(module).get(name) is not function:
            return None
        source = getattr(module, "__file__", None)
        if type(source) is not str or os.path.normcase(os.path.abspath(function.__code__.co_filename)) \
                != os.path.normcase(os.path.abspath(source)):
            return None
        for global_name in function.__code__.co_names:
            value = function.__globals__.get(global_name)
            if isinstance(value, types.ModuleType):
                package = value.__name__.split(".")[0]
                for attribute in function.__code__.co_names:
                    found = vars(value).get(attribute)
                    if found is not None and not _library_own(found, package):
                        return None
    return _sign_producers()


# M31c/M31d: the fluid-side functions as loaded (trust root: import time, the model of
# the solver's response authority). A side is reused only while everything that
# computes it is this original; a replaced function is called every time. M31e (owner
# decision): replacing library or module functions after import is outside the
# supported contract; these checks are defence in depth. M31f: hardware
# floating-point exception traps stay masked (the probe below rounds).
_SIGN_CHAIN = _capture_sign_chain()
# M31e: the two bindings themselves, compared before anything of them is read.
_SIGN_FUNCTIONS = (fluid_sign, vars(_wetfield).get("_fluid_sign"))
# M31d: the NumPy functions the reuse guard itself calls, as loaded.
_GUARD_NAMES = ("geterr", "errstate", "asarray", "isfinite", "all", "abs")
_GUARD_ORIGINALS = tuple((name, vars(np).get(name)) for name in _GUARD_NAMES
                         if _library_own(vars(np).get(name), "numpy"))

_ONE, _THREE_QUARTER_ULP = np.float64(1.0), np.float64(3.0 * 2.0 ** -54)
_SMALLEST_NORMAL, _SMALLEST_SUBNORMAL, _HALF = np.float64(2.0 ** -1022), np.float64(2.0 ** -1074), np.float64(0.5)


def _standard_float_environment() -> bool:
    """M31c: round to nearest, no flush-to-zero and no denormals-are-zero in this
    thread, checked at run time (each probe is exact or differs between the modes)."""
    with np.errstate(all="ignore"):
        up = _ONE + _THREE_QUARTER_ULP
        down = -_ONE - _THREE_QUARTER_ULP
        halved = _SMALLEST_NORMAL * _HALF
        kept = _SMALLEST_SUBNORMAL * _ONE
    return bool(up != _ONE and down != -_ONE and halved != 0.0 and kept != 0.0)


_SURFACE_FIELDS = frozenset(("ref", "name", "region", "fluid_side", "interior_point", "id"))
_FLOAT64_DTYPE = np.dtype(np.float64)
_REF_FIELDS, _REGION_FIELDS = frozenset(("kind", "id")), frozenset(("id",))


def _plain_fields(record, names):
    """M31g: the instance dictionary of a record when it holds exactly these fields
    under plain str keys; read without attribute lookup, so nothing of the record
    runs and nothing raises (None for anything else)."""
    fields = getattr(record, "__dict__", None)
    if type(fields) is not dict or len(fields) != len(names):
        return None
    for key in fields:
        if type(key) is not str or key not in names:
            return None
    return fields


def _wet_surface_state(surface):
    """M31b/M31d/M31g: every field of a plain wet-surface record as plain values (the
    interior point as bytes) and the sizes of the record dictionaries, or None for
    any other record. Only exact types and exactly the fields are accepted and read
    from the instance dictionaries, so no code of the record runs and nothing
    raises; a prepared context neither caches nor reuses anything for a selection
    with such a record, and a kept record that grew (a larger value, an attribute
    added, or a dictionary enlarged and emptied again) no longer matches the
    fingerprint it was charged with."""
    if type(surface) is not WetSurface:
        return None
    fields = _plain_fields(surface, _SURFACE_FIELDS)
    if fields is None:
        return None
    ref, region, point = fields["ref"], fields["region"], fields["interior_point"]
    name, side, identity = fields["name"], fields["fluid_side"], fields["id"]
    if type(name) is not str or type(side) is not str or type(identity) is not str:
        return None
    if type(ref) is not EntityRef:
        return None
    ref_fields = _plain_fields(ref, _REF_FIELDS)
    if ref_fields is None or type(ref_fields["kind"]) is not str or type(ref_fields["id"]) is not int:
        return None
    region_fields = None
    if region is not None:
        if type(region) is not RegionRef:
            return None
        region_fields = _plain_fields(region, _REGION_FIELDS)
        if region_fields is None or type(region_fields["id"]) is not str:
            return None
    # M31h: NumPy's own float64 dtype object (a dtype with metadata compares equal
    # but can hold any payload).
    if point is not None and (type(point) is not np.ndarray or point.dtype is not _FLOAT64_DTYPE
                              or point.shape != (3,) or point.base is not None):
        return None
    return (identity, name, ref_fields["kind"], ref_fields["id"],
            None if region_fields is None else region_fields["id"], side,
            None if point is None else point.tobytes(),
            sys.getsizeof(fields), sys.getsizeof(ref_fields),
            None if region_fields is None else sys.getsizeof(region_fields))


def _bounded(values) -> bool:
    """M31b: all values finite and below 1e150 in size (no comparison with a
    non-finite value is made)."""
    made = np.asarray(values, dtype=float)
    return bool(np.all(np.isfinite(made))) and bool(np.all(np.abs(made) < 1e150))


def _signs_reusable(selected, cached) -> bool:
    """M31b/M31c/M31d: a fluid side may be reused only where computing it again would
    run no other code and could not give another value, warn or raise: the guard's
    own NumPy functions and the fluid-side chain as loaded (_GUARD_ORIGINALS,
    _SIGN_CHAIN), NumPy's default error state in the standard floating-point
    environment, plain surface records whose state is still the one the kept entry
    was built (and charged) for, and centroids, normals and interior points finite
    and below 1e150 in size (the difference and the dot product of the rule cannot
    overflow; underflow is not reported in that state)."""
    names = vars(np)
    if len(_GUARD_ORIGINALS) != len(_GUARD_NAMES) or any(names.get(name) is not value
                                                           for name, value in _GUARD_ORIGINALS):
        return False
    if not cached["bounded"] or tuple(sorted(np.geterr().items())) != _DEFAULT_ERROR_ITEMS:
        return False
    if _SIGN_CHAIN is None or fluid_sign is not _SIGN_FUNCTIONS[0] \
            or vars(_wetfield).get("_fluid_sign") is not _SIGN_FUNCTIONS[1]:
        return False                 # M31e: a replacement is rejected before it is inspected
    if not _standard_float_environment() or not _same_producers(_SIGN_CHAIN, _sign_producers()):
        return False
    states = tuple(_wet_surface_state(surface) for surface in selected)
    if any(state is None for state in states) or states != cached["state"]:
        return False
    for surface in selected:
        if surface.interior_point is not None and not _bounded(surface.interior_point):
            return False
    return True


def _wet_pressure_accumulation(
    project: Project,
    mesh: Mesh,
    case: LoadCase,
    *,
    limit: int | None,
    _load_context: Any,
) -> Dict[int, tuple]:
    """Element id -> (pressure, centroid, normal, shared) for a case's wet
    pressures; ``shared`` marks geometry held by a prepared context's cache
    (read-only). The arithmetic and accumulation order of wet_element_pressures."""

    if not project.wet_surfaces:
        raise ProjectError(
            f"load case {case.name!r} has wet pressure but the model has no "
            "wet surfaces"
        )
    by_surface_id = {surface.id: surface for surface in project.wet_surfaces}
    accumulated: Dict[int, tuple] = {}  # element id -> (pressure, centroid, normal, shared)
    for load in case.wet_pressures:
        scale = _envelope_factor(case, load)
        if scale == 0.0:
            continue
        if load.surfaces:
            missing = [item for item in load.surfaces if item not in by_surface_id]
            if missing:
                raise ProjectError(
                    f"wet pressure {load.name!r} names unknown wet surface(s) "
                    f"{missing}"
                )
            selected = [by_surface_id[item] for item in load.surfaces]
        else:
            selected = list(project.wet_surfaces)
        owner, by_face = (
            _wet_elements(project, mesh, selected)
            if _load_context is None
            else _load_context.wet_elements(selected)
        )
        cached = (
            None
            if _load_context is None or limit is not None
            or not hasattr(_load_context, "wet_geometry")
            else _load_context.wet_geometry(selected, owner)
        )
        if cached is not None:
            # M31 (prepared contexts): the element ids, centroids and per-element
            # geometry of this wet set are fixed; the fluid side of an element is
            # computed on its first non-zero pressure and then reused. Values and
            # accumulation order are those of the path below.
            element_ids, centroids = cached["element_ids"], cached["centroids"]
            geometry, signs = cached["geometry"], cached["signs"]
            pick = np.arange(len(element_ids))
            try:
                # M31e: the pressure source may run caller code; it gets what the path
                # below gives it, an (n, 3) view of a new array.
                sampled = (centroids.copy().reshape(-1, 3) if len(element_ids)
                           else np.array([]).reshape(-1, 3))
                magnitude = scale * _wet_pressure_values(
                    case, load, element_ids, by_face, pick, sampled
                )
                _load_context.require_unchanged("wet pressure evaluation")
                # M31c: decided after the pressure source ran (it may run caller code);
                # from here to the end of the loop only this module's code and, when
                # sides are reused, the original fluid-side functions run. (M31h:
                # changing surface records during a request is outside the supported
                # contract; this check is defence in depth.)
                # M31g: and only for plain float64 magnitudes, whose iteration below
                # runs no code (the ANYloads sources return such arrays).
                reuse = (type(magnitude) is np.ndarray and magnitude.dtype == np.float64
                         and _signs_reusable(selected, cached))
                for element_id, value in zip(element_ids, magnitude):
                    if value == 0.0:
                        continue
                    key = int(element_id)
                    centroid, normal = geometry[key]
                    if reuse:
                        sign = signs.get(key)
                        if sign is None:
                            sign = fluid_sign(owner[key], centroid, normal)
                            signs[key] = sign
                    else:
                        # As on the path below: the function gets fresh writable
                        # arrays, which the record then holds.
                        centroid, normal = centroid.copy(), normal.copy()
                        sign = fluid_sign(owner[key], centroid, normal)
                    signed = sign * float(value)
                    previous = accumulated.get(key)
                    accumulated[key] = (
                        signed + (0.0 if previous is None else previous[0]),
                        centroid,
                        normal,
                        reuse,
                    )
            except WetFieldError as error:
                raise ProjectError(str(error)) from None
            continue
        element_ids = np.array(sorted(owner), dtype=int)
        if limit is None or len(element_ids) <= limit:
            pick = np.arange(len(element_ids))
        else:
            pick = np.unique(
                np.linspace(0, len(element_ids) - 1, max(1, limit)).astype(int)
            )

        geometry = {
            int(e): (
                _element_centroid_normal(mesh, int(e))
                if _load_context is None
                else _load_context.centroid_normal(int(e))
            )
            for e in element_ids[pick]
        }
        centroids = np.array(
            [geometry[int(e)][0] for e in element_ids[pick]]
        ).reshape(-1, 3)
        try:
            magnitude = scale * _wet_pressure_values(
                case, load, element_ids, by_face, pick, centroids
            )
            if _load_context is not None:
                _load_context.require_unchanged("wet pressure evaluation")
            for element_id, value in zip(element_ids[pick], magnitude):
                if value == 0.0:
                    continue
                centroid, normal = geometry[int(element_id)]
                # ``value`` already carries the load's scale.
                signed = fluid_sign(owner[int(element_id)], centroid, normal) * float(value)
                key = int(element_id)
                previous = accumulated.get(key)
                accumulated[key] = (
                    signed + (0.0 if previous is None else previous[0]),
                    centroid,
                    normal,
                    False,
                )
        except WetFieldError as error:
            raise ProjectError(str(error)) from None
    return accumulated


def _accumulate_wet_pressures(
    project: Project,
    mesh: Mesh,
    case: LoadCase,
    factor: float,
    target: SolverLoadCase,
    *,
    _load_context: Any = None,
) -> None:
    """Add a case's wet pressures to a solver case as element pressures."""

    # M31b: pressures only (no geometry arrays leave the accumulation).
    accumulated = _wet_pressure_accumulation(
        project, mesh, case, limit=None, _load_context=_load_context
    )
    for key in sorted(accumulated):
        target.pressure_loads[key] = (
            target.pressure_loads.get(key, 0.0)
            + factor * accumulated[key][0]
        )


def _wet_pressure_values(
    case: LoadCase,
    load: WetPressure,
    element_ids: np.ndarray,
    by_face: Mapping[int, list[int]],
    pick: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Scaled pressure magnitude of the picked elements of ``element_ids``.

    ``centroids`` belong to the picked elements only. Every source except a
    mapping by id is evaluated by ANYloads.
    """

    if load.kind != "mapping":
        return field_pressure(case, load, centroids)

    values = np.zeros(len(element_ids))
    position = {int(e): index for index, e in enumerate(element_ids)}
    if load.mapping_key == "element":
        for element_id, value in load.mapping.items():
            if element_id not in position:
                raise ProjectError(
                    f"wet pressure {load.name!r} maps a pressure to element "
                    f"{element_id}, which is not on a selected wet surface"
                )
            values[position[element_id]] = value
        return load.scale * values[pick]
    for face_id, value in load.mapping.items():
        elements = by_face.get(face_id)
        if not elements:
            raise ProjectError(
                f"wet pressure {load.name!r} maps a pressure to plate "
                f"{face_id}, which is not on a selected wet surface"
            )
        for element_id in elements:
            values[position[int(element_id)]] = value
    return load.scale * values[pick]


def _add_nodal(
    target: SolverLoadCase,
    node_id: int,
    force: np.ndarray,
    moment: np.ndarray,
) -> None:
    existing = target.nodal_loads.get(node_id)
    if existing is None:
        existing = np.zeros(6)
        target.nodal_loads[node_id] = existing
    existing[:3] += np.asarray(force, dtype=float)
    existing[3:] += np.asarray(moment, dtype=float)


def _coordinate_system(project: Project, coordinate_system_id: str):
    try:
        return project.coordinate_systems[str(coordinate_system_id)]
    except KeyError:
        raise ProjectError(
            f"coordinate system {coordinate_system_id!r} does not exist"
        ) from None


def _attribute_targets(
    project: Project,
    region_ref: RegionRef | None,
    fallback: EntityRef,
    *,
    geometry_kinds: Sequence[str] | None = None,
    mesh_kinds: Sequence[str] | None = None,
) -> tuple[Any, ...]:
    """Resolve a canonical region, retaining the legacy direct-ref fallback.

    Feature-output anchors resolve only through history output keys.  An edit
    that removes an output therefore produces an empty scope and a blocking
    diagnostic here; it is never silently retargeted to nearby geometry.
    """

    if region_ref is None:
        targets: tuple[Any, ...] = (fallback,) if project.mesh_only else tuple(project.geometry.resolve_ref(fallback))
    else:
        try:
            geometry = None if project.mesh_only else project.geometry
            feature_resolver = None
            if geometry is not None:
                feature_resolver = lambda anchor: project.geometry.features.resolve(
                    anchor, project.geometry
                )
            targets = project.regions.resolve(
                region_ref.id,
                geometry=geometry,
                feature_resolver=feature_resolver,
            )
        except Exception as exc:
            raise ProjectError(
                f"region {region_ref.id!r} cannot be resolved: {exc}"
            ) from exc
    if not targets:
        raise ProjectError(f"region {getattr(region_ref, 'id', '')!r} resolves to nothing")

    geometry_allowed = None if geometry_kinds is None else set(geometry_kinds)
    mesh_allowed = None if mesh_kinds is None else set(mesh_kinds)
    for target in targets:
        if isinstance(target, EntityRef):
            if geometry_allowed is not None and target.kind not in geometry_allowed:
                raise ProjectError(
                    f"scope contains geometry {target.kind!r}; expected "
                    f"{', '.join(sorted(geometry_allowed))}"
                )
        elif isinstance(target, MeshEntityRef):
            if mesh_allowed is not None and target.kind not in mesh_allowed:
                raise ProjectError(
                    f"scope contains mesh {target.kind!r}; expected "
                    f"{', '.join(sorted(mesh_allowed))}"
                )
        elif isinstance(target, ElementFaceRef):
            if mesh_allowed is not None and "element_face" not in mesh_allowed:
                raise ProjectError("scope contains an element face, which is invalid here")
        else:
            raise ProjectError(f"scope resolved to unsupported target {target!r}")
    return tuple(targets)


def _element_nodes(mesh: Mesh, element_id: int) -> tuple[int, ...]:
    for collection in (mesh.quads, mesh.tris, mesh.beams):
        if element_id in collection:
            return tuple(int(node) for node in collection[element_id])
    if element_id in mesh.couplings:
        coupling = mesh.couplings[element_id]
        if hasattr(coupling, "beam_node"):
            return (int(coupling.beam_node),) + tuple(
                int(node) for node in coupling.plate_nodes
            )
    raise ProjectError(f"mesh scope references missing element {element_id}")


def _nodes_on_target(
    mesh: Mesh, target: Any, *, constrained: bool = False
) -> list[int]:
    if isinstance(target, EntityRef):
        resolver = mesh.constrained_nodes_on if constrained else mesh.nodes_on
        return list(resolver(target))
    if isinstance(target, MeshEntityRef):
        if target.kind == "node":
            if target.id not in mesh.nodes:
                raise ProjectError(f"mesh scope references missing node {target.id}")
            return [int(target.id)]
        return list(_element_nodes(mesh, int(target.id)))
    if isinstance(target, ElementFaceRef):
        # The qualified structural element families are beams and shells.  A
        # selected shell face therefore maps to the shell's interpolation
        # nodes; no solid-face topology is invented.
        return list(_element_nodes(mesh, int(target.element_id)))
    raise ProjectError(f"unsupported scoped target {target!r}")


def _elements_on_target(mesh: Mesh, target: Any) -> list[int]:
    if isinstance(target, EntityRef):
        return list(mesh.elements_on(target))
    if isinstance(target, MeshEntityRef):
        if target.kind != "element":
            return []
        _element_nodes(mesh, int(target.id))
        return [int(target.id)]
    if isinstance(target, ElementFaceRef):
        _element_nodes(mesh, int(target.element_id))
        return [int(target.element_id)]
    raise ProjectError(f"unsupported scoped target {target!r}")


def _line_target_to_nodes(
    mesh: Mesh, target: Any, force_per_length: np.ndarray
) -> List[tuple[int, np.ndarray]]:
    if isinstance(target, EntityRef):
        return _line_load_to_nodes(mesh, target, force_per_length)
    if isinstance(target, MeshEntityRef) and target.kind == "element":
        if target.id not in mesh.beams:
            raise ProjectError(
                f"line load mesh scope element {target.id} is not a beam"
            )
        nodes = tuple(int(node) for node in mesh.beams[target.id])
        shares = (
            (0.5, 0.5)
            if len(nodes) == 2
            else (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)
        )
        positions = [mesh.nodes[node] for node in nodes]
        length = float(
            sum(
                np.linalg.norm(second - first)
                for first, second in zip(positions, positions[1:])
            )
        )
        intensity = np.asarray(force_per_length, dtype=float)
        return [
            (node, float(weight) * length * intensity)
            for node, weight in zip(nodes, shares)
        ]
    raise ProjectError(f"unsupported line-load scope target {target!r}")


def _traction_target_to_nodes(
    mesh: Mesh, target: Any, traction: np.ndarray
) -> List[tuple[int, np.ndarray]]:
    if isinstance(target, EntityRef):
        return _traction_to_nodes(mesh, target, traction)
    element_ids = _elements_on_target(mesh, target)
    accumulated: Dict[int, np.ndarray] = {}
    for element_id in element_ids:
        for node_id, force in _traction_elements_to_nodes(
            mesh, (element_id,), traction
        ):
            accumulated.setdefault(node_id, np.zeros(3))
            accumulated[node_id] += force
    return list(accumulated.items())


def _traction_to_nodes(
    mesh: Mesh, ref: EntityRef, traction: np.ndarray
) -> List[tuple[int, np.ndarray]]:
    """Lump a force per unit area onto the nodes of a plate.

    The shares are the element's *consistent* load vector for a uniform
    traction, so the resultant is exact and a non-uniform mesh still
    distributes it correctly. For Q4 and T3 the shares are equal. For Q8 it is
    one third to each mid-side node and **minus** one twelfth to each corner;
    for T6 the corner integrals are zero and each mid-side node takes one
    third. These are the exact integrals of the corresponding shape functions.
    """

    element_ids = mesh.elements_of_face.get(ref.id)
    if not element_ids:
        raise ProjectError(
            f"surface traction references {ref}, which has no elements in the mesh"
        )

    return _traction_elements_to_nodes(mesh, element_ids, traction)


def _traction_elements_to_nodes(
    mesh: Mesh, element_ids: Iterable[int], traction: np.ndarray
) -> List[tuple[int, np.ndarray]]:
    """Consistently lump a uniform traction over explicit shell elements."""

    accumulated: Dict[int, np.ndarray] = {}
    intensity = np.asarray(traction, dtype=float)
    for element_id in element_ids:
        if element_id in mesh.quads:
            nodes = mesh.quads[element_id]
            shares_by_count = {
                4: (0.25,) * 4,
                8: (-1.0 / 12.0,) * 4 + (1.0 / 3.0,) * 4,
            }
        elif element_id in mesh.tris:
            nodes = mesh.tris[element_id]
            shares_by_count = {
                3: (1.0 / 3.0,) * 3,
                6: (0.0,) * 3 + (1.0 / 3.0,) * 3,
            }
        else:
            raise ProjectError(
                f"surface traction references missing shell element {element_id}"
            )
        try:
            shares = shares_by_count[len(nodes)]
        except KeyError:
            raise ProjectError(
                f"surface traction does not support a {len(nodes)}-node shell"
            ) from None
        corners = np.array(
            [mesh.nodes[node] for node in mesh.corners_of(element_id)]
        )
        if len(corners) == 3:
            area = 0.5 * float(
                np.linalg.norm(
                    np.cross(corners[1] - corners[0], corners[2] - corners[0])
                )
            )
        else:
            area = 0.5 * (
                float(
                    np.linalg.norm(
                        np.cross(corners[1] - corners[0], corners[2] - corners[0])
                    )
                )
                + float(
                    np.linalg.norm(
                        np.cross(corners[2] - corners[0], corners[3] - corners[0])
                    )
                )
            )
        for node_id, weight in zip(nodes, shares):
            if node_id not in accumulated:
                accumulated[node_id] = np.zeros(3)
            accumulated[node_id] += weight * area * intensity
    return list(accumulated.items())


def _add_masses(project: Project, mesh: Mesh, fe_model: FEModel) -> None:
    """Attach lumped masses, sharing a total equally over an entity's nodes."""

    for mass in project.masses:
        targets = _attribute_targets(project, mass.region, mass.ref)
        node_ids = sorted(
            {
                node_id
                for target in targets
                for node_id in _nodes_on_target(mesh, target)
            }
        )
        if not node_ids:
            raise ProjectError(
                f"mass {mass.name!r} has an empty or unresolved scope in the mesh"
            )
        share = (
            float(mass.value) / len(node_ids)
            if mass.distribution_policy == "total_distributed"
            else float(mass.value)
        )
        for node_id in node_ids:
            fe_model.add_point_mass(node_id, share)


def _apply_imperfections(
    project: Project, mesh: Mesh, fe_model: FEModel
) -> FEModel:
    """Move the stress-free geometry, through the solver's own builders."""

    from anysolver import (
        CompositeImperfection,
        apply_imperfection,
        standard_member_bow,
        standard_plate_mode,
    )

    fields = []
    for imperfection in project.imperfections:
        references = project.resolve_geometry_attachment(imperfection.ref)
        node_ids = list(dict.fromkeys(node for reference in references for node in mesh.nodes_on(reference)))
        if not node_ids:
            raise ProjectError(
                f"imperfection {imperfection.name!r} references "
                f"{imperfection.ref}, which has no nodes in the mesh"
            )
        if imperfection.resolved_kind == "plate_mode":
            fields.append(
                standard_plate_mode(
                    fe_model,
                    node_ids,
                    amplitude=imperfection.amplitude,
                    direction=imperfection.direction,
                    axes=imperfection.axes,
                    waves=imperfection.waves,
                    name=imperfection.name,
                )
            )
        else:
            fields.append(
                standard_member_bow(
                    fe_model,
                    node_ids,
                    amplitude=imperfection.amplitude,
                    direction=imperfection.direction,
                    name=imperfection.name,
                )
            )

    combined = fields[0] if len(fields) == 1 else CompositeImperfection(fields)
    return apply_imperfection(fe_model, combined, copy_model=False)


def _line_load_to_nodes(
    mesh: Mesh, ref: EntityRef, force_per_length: np.ndarray
) -> List[tuple[int, np.ndarray]]:
    """Lump a distributed line load onto the nodes along that line.

    The shares are the element's *consistent* load vector for a uniform load:
    half and half for a 2-node element, and one sixth, two thirds, one sixth
    for a 3-node one.  Splitting a quadratic element's load evenly between its
    stations instead would put a quarter at each end where a sixth belongs,
    and an inconsistent load vector drags the quadratic element back down to
    the convergence rate of a linear one -- so the extra nodes would cost
    something and buy nothing.
    """

    sequence = mesh.nodes_of_edge.get(ref.id)
    if not sequence:
        raise ProjectError(
            f"line load references {ref}, which has no nodes in the mesh"
        )

    step = 2 if mesh.is_quadratic else 1
    shares = (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0) if step == 2 else (0.5, 0.5)

    accumulated: Dict[int, np.ndarray] = {}
    intensity = np.asarray(force_per_length, dtype=float)
    for start in range(0, len(sequence) - step, step):
        span = sequence[start : start + step + 1]
        positions = [mesh.nodes[node] for node in span]
        length = float(
            sum(
                np.linalg.norm(second - first)
                for first, second in zip(positions, positions[1:])
            )
        )
        for node_id, weight in zip(span, shares):
            if node_id not in accumulated:
                accumulated[node_id] = np.zeros(3)
            accumulated[node_id] += weight * length * intensity
    return [(node_id, force) for node_id, force in accumulated.items()]
