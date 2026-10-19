"""Sequential displacement-driven submodelling: the workflow part (slice S4a-1).

Implements the cut-independent part of ``docs/SUBMODEL_WORKFLOW_CONTRACT.md``:
source admission and binding, an explicit face lineage with named cut faces, the
transfer of a source solution onto the cut of a separately meshed submodel, the
driven solve, and the transfer/equilibrium diagnostic.  The volume cut
(ANYgeometry, slice S2a) will later produce the submodel geometry and the lineage;
until then the caller supplies both.

Source route in this slice: an in-memory :class:`LinearSolution` admitted through
:func:`bind_source`, which records fingerprints of the solved model, mesh, DOF
layout and field.  :func:`solve_submodel` refuses when the source no longer matches
its binding.  Job and artifact provenance (job id, artifact hash) belongs to the
persisted route (slice S4b) and is not claimed here.

Mechanics live in their owners: point correspondence in
``anymesher.submodel_correspondence``; interpolation, the driven trace, interface
forces and error allowances in ``anysolver.submodel_transfer``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .model.project import ProjectError

__all__ = [
    "SubmodelError",
    "SourceBinding",
    "SubmodelLineage",
    "SubmodelSettings",
    "SubmodelResult",
    "admit_source",
    "bind_source",
    "solve_submodel",
    "free_body_ratios",
    "refinement_change",
]


class SubmodelError(ProjectError):
    """A typed, fail-closed refusal of the submodel workflow."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


# ----------------------------------------------------------------------
# source admission and binding
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SourceBinding:
    """Fingerprints of an admitted in-memory source solution."""

    model_hash: str  # ANYfem's solver-affecting project hash (excludes loads)
    problem_hash: str  # the model hash without meshing controls: the physical problem
    loading_hash: str  # load cases and combinations, as jobs hash them
    mesh_fingerprint: str  # nodes, connectivity and face/edge associations of the solved mesh
    solver_fingerprint: str  # complete public state of solved nodes, elements, materials, supports, equations
    operator_fingerprint: str  # assembled stiffness and audited constraint rows (incl. element couplings)
    load_fingerprint: str  # assembled load vector of the solved load case
    layout_fingerprint: str  # node -> six global DOFs of the solved model
    field_fingerprint: str  # SHA-256 of the displacement vector
    load_case: str


def _sha(*arrays: Any) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(np.asarray(array)).tobytes())
    return digest.hexdigest()


def _model_hash(project: Any) -> str:
    from .document import _model_payload, canonical_hash
    from .io.project_file import project_to_dict

    return canonical_hash(_model_payload(project_to_dict(project)))


def _problem_hash(project: Any) -> str:
    """The model hash without meshing controls (element size, order, seeding,
    refinement regions): what two refinements of one source must share."""

    from .document import _model_payload, canonical_hash
    from .io.project_file import project_to_dict

    payload = _model_payload(project_to_dict(project))
    payload.pop("meshing", None)
    return canonical_hash(payload)


def _loading_hash(project: Any) -> str:
    """Same recipe as ``anyfem.application.workflow._loading_hash`` (jobs); kept here
    so the headless workflow does not import the application layer."""

    from .document import canonical_hash
    from .io.project_file import _load_case_to_dict

    return canonical_hash({
        "load_cases": [_load_case_to_dict(case) for case in sorted(project.load_cases.values(), key=lambda case: case.name)],
        "combinations": [
            {"id": item.id, "name": item.name, "factors": dict(item.factors)}
            for item in sorted(project.combinations.values(), key=lambda item: item.name)
        ],
    })


def _text(value: Any) -> np.ndarray:
    return np.frombuffer(repr(value).encode(), dtype=np.uint8)


def _mesh_fingerprint(mesh: Any) -> str:
    ids = np.array(sorted(mesh.nodes), dtype=np.int64)
    coordinates = np.array([mesh.nodes[n] for n in ids], dtype=float)
    connectivity = [(int(e), tuple(int(n) for n in nodes)) for e, nodes in sorted(mesh.shells.items())]
    connectivity += [(int(e), tuple(int(n) for n in nodes)) for e, nodes in sorted(mesh.beams.items())]
    associations = [
        sorted((int(f), tuple(sorted(int(e) for e in els))) for f, els in mesh.elements_of_face.items()),
        sorted((int(k), tuple(int(n) for n in nodes)) for k, nodes in mesh.nodes_of_edge.items()),
    ]
    return _sha(ids, coordinates, _text(connectivity), _text(associations))


_DEPTH = 12


def _canonical(value: Any, path: str, depth: int = 0, active: Optional[set] = None) -> Any:
    """Complete canonical form of solved-model state; unsupported values refuse.

    Private attributes (leading underscore) are runtime caches and are not part of
    the physical input; everything public is serialized, nothing is skipped.
    """

    import enum

    from scipy import sparse

    if depth > _DEPTH:
        raise SubmodelError("SOURCE_STATE_UNSUPPORTED", f"{path} is nested too deeply to fingerprint")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return ("i", int(value))
    if isinstance(value, (float, np.floating)):
        return ("f", float(value).hex())
    if isinstance(value, enum.Enum):
        return ("enum", type(value).__qualname__, _canonical(value.value, path, depth + 1, active))
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return ("objarray", value.shape, tuple(_canonical(v, path, depth + 1, active) for v in value.reshape(-1)))
        array = np.ascontiguousarray(value)
        return ("array", array.dtype.str, array.shape, hashlib.sha256(array.tobytes()).hexdigest())
    if sparse.issparse(value):
        csr = sparse.csr_matrix(value, copy=True)
        csr.sum_duplicates()
        csr.sort_indices()
        return ("sparse", csr.shape, _canonical(csr.indptr, path, depth + 1, active),
                _canonical(csr.indices, path, depth + 1, active), _canonical(csr.data, path, depth + 1, active))
    if isinstance(value, Mapping):
        items = [(_canonical(k, path, depth + 1, active), _canonical(v, f"{path}[{k!r}]", depth + 1, active))
                 for k, v in value.items()]
        return ("map", tuple(sorted(items, key=repr)))
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(_canonical(v, f"{path}[{i}]", depth + 1, active) for i, v in enumerate(value)))
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted((_canonical(v, path, depth + 1, active) for v in value), key=repr)))
    if callable(value) or isinstance(value, type) or not hasattr(value, "__dict__"):
        raise SubmodelError("SOURCE_STATE_UNSUPPORTED",
                            f"{path} holds a {type(value).__name__}, which cannot be fingerprinted")
    active = set() if active is None else active
    if id(value) in active:
        raise SubmodelError("SOURCE_STATE_UNSUPPORTED", f"{path} refers back to itself")
    active.add(id(value))
    try:
        state = tuple(
            (key, _canonical(item, f"{path}.{key}", depth + 1, active))
            for key, item in sorted(vars(value).items()) if not key.startswith("_")
        )
    finally:
        active.discard(id(value))
    return ("obj", type(value).__module__, type(value).__qualname__, state)


def _solver_fingerprint(fe_model: Any) -> str:
    """Complete public state of what was solved: nodes, elements, materials, supports, equations."""

    mesh = fe_model.mesh
    nodes = [(int(n), tuple(float(v) for v in mesh.get_node(n).coords())) for n in sorted(mesh.nodes)]
    elements = [(int(e), _canonical(el, f"element {e}")) for e, el in sorted(mesh.elements.items())]
    materials = [(str(name), _canonical(material, f"material {name}"))
                 for name, material in sorted(fe_model.materials.items())]
    supports = [_canonical(bc, "support") for bc in fe_model.boundary_conditions]
    equations = [_canonical(eq, "equation") for eq in fe_model.constraint_equations]
    return _sha(_text((nodes, elements, materials, supports, equations)))


def _operator_fingerprint(fe_model: Any) -> str:
    """The solved operator itself: the assembled stiffness and every audited
    constraint row (supports, model equations and element-provided couplings)."""

    from anysolver.constraint_audit import require_valid_constraints
    from anysolver.matrix_assembly import assemble_stiffness_matrix

    stiffness, _info = assemble_stiffness_matrix(fe_model)
    rows = [equation.to_dict() for equation in require_valid_constraints(fe_model).equations]
    return _sha(_text(_canonical(stiffness, "stiffness")), _text(_canonical(rows, "constraint rows")))


def _load_fingerprint(fe_model: Any, load_case: Any) -> str:
    from anysolver.matrix_assembly import assemble_load_vector

    load, _info = assemble_load_vector(fe_model, load_case)
    return _sha(np.asarray(load, dtype=float))


def _require_consistent(built: Any) -> None:
    """Every element the locator may use is solved exactly as the mesher mesh says.

    Correspondence uses only the shell elements of the source faces
    (``elements_of_face``); each must be a mesher shell and a solved element with the
    same node ids, and each of its nodes must have the same coordinates in both.
    Beams, couplings and solver-generated entities are not used by the transfer and
    are bound through the solver and operator fingerprints instead.
    """

    fe_mesh = built.fe_model.mesh
    mesh = built.mesh
    for face_id, element_ids in mesh.elements_of_face.items():
        for element_id in element_ids:
            nodes = mesh.shells.get(int(element_id))
            solved = fe_mesh.elements.get(int(element_id))
            if nodes is None or solved is None:
                raise SubmodelError("SOURCE_INCONSISTENT",
                                    f"element {element_id} of source face {face_id} is not a solved shell")
            if tuple(int(n) for n in solved.node_ids) != tuple(int(n) for n in nodes):
                raise SubmodelError("SOURCE_INCONSISTENT",
                                    f"source element {element_id} differs between the mesh and the solved model")
            for node in nodes:
                point = fe_mesh.get_node(int(node))
                if point is None or not np.array_equal(np.asarray(point.coords(), dtype=float),
                                                       np.asarray(mesh.nodes[node], dtype=float)):
                    raise SubmodelError("SOURCE_INCONSISTENT",
                                        f"source node {node} differs between the mesh and the solved model")


def _layout(fe_model: Any) -> str:
    dm = fe_model.mesh.dof_manager
    rows = []
    for node in sorted(fe_model.mesh.nodes):
        dofs = dm.get_node_dofs(node)
        if len(dofs) != 6:
            raise SubmodelError("SOURCE_LAYOUT", f"source node {node} does not carry six DOFs")
        rows.append([node, *dofs])
    return _sha(np.array(rows, dtype=np.int64), np.array([dm.total_dofs], dtype=np.int64))


def admit_source(solution: Any) -> None:
    """Admit only a converged, certified linear static solution with a full field."""

    from .post.results import LinearSolution

    if type(solution) is not LinearSolution:
        raise SubmodelError(
            "SOURCE_NOT_ADMITTED",
            f"a submodel is driven from a linear static solution, not {type(solution).__name__}",
        )
    convergence = (getattr(solution, "info", None) or {}).get("convergence_info", {})
    if convergence.get("status") != "converged" or convergence.get("relative_backward_error_disposition") != "CERTIFIED":
        raise SubmodelError("SOURCE_NOT_CERTIFIED", "the source solve is not converged and certified")
    built = solution.built
    if getattr(built.fe_model.mesh, "element_activity", None) is not None:
        raise SubmodelError("SOURCE_ACTIVITY", "sources with an element activity state are not admitted in v1")
    u = np.asarray(solution.displacements, dtype=float).reshape(-1)
    if u.shape != (int(built.fe_model.mesh.dof_manager.total_dofs),) or not np.all(np.isfinite(u)):
        raise SubmodelError("SOURCE_FIELD", "the source displacement field is incomplete or not finite")
    _layout(built.fe_model)


def _binding_of(solution: Any) -> SourceBinding:
    built = solution.built
    _require_consistent(built)
    return SourceBinding(
        model_hash=_model_hash(built.project),
        problem_hash=_problem_hash(built.project),
        loading_hash=_loading_hash(built.project),
        mesh_fingerprint=_mesh_fingerprint(built.mesh),
        solver_fingerprint=_solver_fingerprint(built.fe_model),
        operator_fingerprint=_operator_fingerprint(built.fe_model),
        load_fingerprint=_load_fingerprint(built.fe_model, built.load_case),
        layout_fingerprint=_layout(built.fe_model),
        field_fingerprint=_sha(np.asarray(solution.displacements, dtype=float)),
        load_case=str(getattr(built.load_case, "name", "")),
    )


def bind_source(solution: Any) -> SourceBinding:
    """Admit ``solution`` and record its fingerprints.  Bind immediately after the
    solve: the binding asserts that the project and mesh are the solved ones."""

    admit_source(solution)
    return _binding_of(solution)


def _require_binding(solution: Any, binding: SourceBinding) -> None:
    admit_source(solution)
    current = _binding_of(solution)
    for name in ("model_hash", "problem_hash", "loading_hash", "mesh_fingerprint", "solver_fingerprint",
                 "operator_fingerprint", "load_fingerprint", "layout_fingerprint", "field_fingerprint", "load_case"):
        if getattr(current, name) != getattr(binding, name):
            raise SubmodelError("SOURCE_STALE", f"the source no longer matches its binding ({name} changed)")


# ----------------------------------------------------------------------
# lineage and settings
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SubmodelLineage:
    """Which source faces each submodel face came from, and the artificial cut.

    ``cut_faces`` names each physical cut face (a group of submodel edges, for
    example several collinear edge segments of one box side).  Corner records are
    nodes shared by different cut faces.
    """

    source_faces: Mapping[int, Tuple[int, ...]]
    cut_faces: Mapping[str, Tuple[int, ...]]

    @property
    def cut_edges(self) -> Tuple[int, ...]:
        return tuple(sorted({int(e) for edges in self.cut_faces.values() for e in edges}))


@dataclass(frozen=True)
class SubmodelSettings:
    """``policy`` is ``"drive_all"`` or ``"release_normal_rotation"``; there is no
    default, because the drilling decision (PLAN.md D3) belongs to the caller."""

    policy: str
    physical_tolerance: Optional[float] = None
    natural_tolerance: float = 1.0e-9
    weight_tolerance: float = 1.0e-9


@dataclass
class SubmodelResult:
    solution: Any  # LinearSolution of the submodel
    trace: Any  # anysolver DrivenTrace
    interface: Any  # anysolver InterfaceForces
    transfer: Dict[int, Any]  # target node -> anysolver TransferValue
    transfer_allowance: Dict[int, np.ndarray]
    binding: SourceBinding
    identity: str = ""  # child model, loading, mesh, lineage cut, policy and transfer settings
    hits: Dict[int, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


_PLANARITY = 1.0e-9


def _face_normal(project: Any, mesh: Any, face_id: int) -> np.ndarray:
    """Normal of a planar source face from its mesh nodes; refuses non-planar or degenerate faces.

    The planar slice admits faces whose mesh nodes lie within ``1e-9`` of the face
    size from their best-fit plane.
    """

    nodes = np.array([np.asarray(mesh.nodes[n], dtype=float) for n in mesh.nodes_on(project.face(int(face_id)))])
    if len(nodes) < 3:
        raise SubmodelError("SUBMODEL_FACE_DEGENERATE", f"source face {face_id} has fewer than three nodes")
    centred = nodes - nodes.mean(axis=0)
    _u, sigma, vt = np.linalg.svd(centred, full_matrices=False)
    size = float(np.max(np.ptp(nodes, axis=0)))
    if sigma.size < 2 or sigma[1] <= 1.0e-12 * max(sigma[0], 1.0e-300):
        raise SubmodelError("SUBMODEL_FACE_DEGENERATE", f"source face {face_id} does not span a plane")
    normal = vt[-1] / np.linalg.norm(vt[-1])
    out_of_plane = float(np.max(np.abs(centred @ normal)))
    if out_of_plane > _PLANARITY * size:
        raise SubmodelError(
            "SUBMODEL_FACE_NOT_PLANAR",
            f"source face {face_id} deviates {out_of_plane / size:.3e} of its size from a plane; the planar slice "
            "cannot release a normal rotation there",
        )
    return normal


@dataclass(frozen=True)
class _Surface:
    """A face's surface class (workflow contract 12.1)."""

    kind: str  # "planar" or "cylinder"
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    radius: float = 0.0


def _planar_nodes(project: Any, mesh: Any, face_id: int) -> bool:
    try:
        _face_normal(project, mesh, face_id)
    except SubmodelError as error:
        if error.args and "FACE_NOT_PLANAR" in str(error):
            return False
        raise
    return True


def _classify(project: Any, mesh: Any, face_id: int, which: str) -> _Surface:
    from anygeometry.surfaces import Cylinder, Plane

    surface = getattr(project.geometry.faces[int(face_id)], "surface", None)
    if isinstance(surface, Cylinder):
        return _Surface("cylinder", tuple(float(v) for v in surface.origin), tuple(float(v) for v in surface.axis),
                        float(surface.radius))
    if isinstance(surface, Plane) or _planar_nodes(project, mesh, face_id):
        return _Surface("planar")
    raise SubmodelError("SUBMODEL_SURFACE_UNSUPPORTED",
                        f"{which} face {face_id} is neither planar nor an analytic cylinder")


def _same_cylinder(a: _Surface, b: _Surface, tolerance: float) -> bool:
    """Same cylinder up to axis sign, origin shift along the axis and parameterization.

    Symmetric: the axes are parallel within 1e-9 (cross product), and each origin
    lies within ``tolerance`` of the other's axis line.
    """

    if a.kind != "cylinder" or b.kind != "cylinder" or abs(a.radius - b.radius) > tolerance:
        return False
    axis_a, axis_b = np.asarray(a.axis), np.asarray(b.axis)
    if float(np.linalg.norm(np.cross(axis_a, axis_b))) > 1.0e-9:
        return False
    offset = np.asarray(b.origin) - np.asarray(a.origin)
    return (float(np.linalg.norm(offset - (offset @ axis_a) * axis_a)) <= tolerance
            and float(np.linalg.norm(offset - (offset @ axis_b) * axis_b)) <= tolerance)


def _radial(surface: _Surface, point: np.ndarray) -> Tuple[float, np.ndarray]:
    """Distance from the axis and the unit radial direction at ``point``."""

    axis = np.asarray(surface.axis)
    offset = np.asarray(point, dtype=float) - np.asarray(surface.origin)
    radial = offset - (offset @ axis) * axis
    length = float(np.linalg.norm(radial))
    return length, radial / length if length > 0.0 else radial


def _node_faces(mesh: Any) -> Dict[int, set]:
    shells = mesh.shells
    out: Dict[int, set] = {}
    for face_id, element_ids in mesh.elements_of_face.items():
        for element_id in element_ids:
            for node in shells.get(int(element_id), ()):
                out.setdefault(int(node), set()).add(int(face_id))
    return out


# ----------------------------------------------------------------------
# workflow
# ----------------------------------------------------------------------
def solve_submodel(
    source: Any,
    binding: SourceBinding,
    sub_project: Any,
    sub_mesh: Any,
    lineage: SubmodelLineage,
    settings: SubmodelSettings,
    *,
    load_case: str = "default",
) -> SubmodelResult:
    """Drive ``sub_project`` on its cut from the bound ``source`` and solve it."""

    from anymesher.submodel_correspondence import CorrespondenceError, locate_for_transfer
    from anysolver import solve_linear
    from anysolver.submodel_transfer import (
        SubmodelTransferError,
        TransferPoint,
        apply_driven_trace,
        build_driven_trace,
        estimate_static_error_allowance,
        interface_forces,
        interpolate_nodal_field,
        interpolation_allowance,
    )

    from .post.results import LinearSolution
    from .solve.build import build_fe_model

    _require_binding(source, binding)
    if sub_project.imperfections:
        raise SubmodelError("SUBMODEL_UNSUPPORTED", "imperfections in a submodel are not supported in v1")
    source_built = source.built
    source_mesh = source_built.mesh
    source_project = source_built.project
    u_source = np.asarray(source.displacements, dtype=float)
    solved_elements = {int(e) for e in source_built.fe_model.mesh.elements}

    for face_id in sub_mesh.elements_of_face:
        sources = lineage.source_faces.get(int(face_id))
        if not sources:
            raise SubmodelError("LINEAGE_MISSING", f"submodel face {face_id} has no source faces")
        for source_face in sources:
            if int(source_face) not in source_mesh.elements_of_face:
                raise SubmodelError("LINEAGE_UNKNOWN", f"source face {source_face} is not in the source mesh")
    if not lineage.cut_faces or not lineage.cut_edges:
        raise SubmodelError("LINEAGE_NO_CUT", "the lineage names no cut faces")
    for group, edges in lineage.cut_faces.items():
        if not edges:
            raise SubmodelError("LINEAGE_EMPTY_CUT_FACE", f"cut face {group!r} has no edges")

    # Surface classes (contract 12.1): every child face and the source faces its
    # lineage names; a cylindrical child must sit on the same cylinder as its sources.
    source_positions = np.array([np.asarray(x, dtype=float) for x in source_mesh.nodes.values()])
    source_extent = float(np.linalg.norm(np.ptp(source_positions, axis=0))) if len(source_positions) else 1.0
    source_classes: Dict[int, _Surface] = {}
    child_classes: Dict[int, _Surface] = {}
    for face_id in sub_mesh.elements_of_face:
        child_class = child_classes[int(face_id)] = _classify(sub_project, sub_mesh, int(face_id), "child")
        for source_face in lineage.source_faces[int(face_id)]:
            if int(source_face) not in source_classes:
                source_classes[int(source_face)] = _classify(source_project, source_mesh, int(source_face), "source")
            source_class = source_classes[int(source_face)]
            tolerance = 1.0e-9 * max(child_class.radius, source_extent)
            if child_class.kind != source_class.kind or (
                    child_class.kind == "cylinder" and not _same_cylinder(child_class, source_class, tolerance)):
                raise SubmodelError(
                    "SUBMODEL_SURFACE_MISMATCH",
                    f"child face {face_id} ({child_class.kind}) and source face {source_face} ({source_class.kind}) "
                    "are not the same surface",
                )
        if child_class.kind == "cylinder":
            tolerance = 1.0e-9 * max(child_class.radius, source_extent)
            for node in sub_mesh.nodes_on(sub_project.face(int(face_id))):
                distance, _direction = _radial(child_class, sub_mesh.nodes[node])
                if abs(distance - child_class.radius) > tolerance:
                    raise SubmodelError(
                        "SUBMODEL_OFF_SURFACE",
                        f"child node {node} is {abs(distance - child_class.radius):.3e} m off its cylinder",
                    )

    # Driven nodes; candidates from every submodel face meeting the node (target
    # connectivity), through their lineage, restricted to elements of the solved operator.
    node_faces = _node_faces(sub_mesh)
    groups_of_node: Dict[int, set] = {}
    for group, edges in lineage.cut_faces.items():
        for edge_id in edges:
            nodes = sub_mesh.nodes_on(sub_project.edge(int(edge_id)))
            if not nodes:
                raise SubmodelError("LINEAGE_NO_CUT", f"cut edge {edge_id} has no nodes in the submodel mesh")
            for node in nodes:
                groups_of_node.setdefault(int(node), set()).add(str(group))
    candidates: Dict[int, set] = {}
    for node in groups_of_node:
        allowed: set = set()
        for face_id in node_faces.get(node, ()):
            for source_face in lineage.source_faces.get(face_id, ()):
                allowed.update(int(e) for e in source_mesh.elements_of_face[int(source_face)])
        candidates[node] = allowed & solved_elements
    points = {node: np.asarray(sub_mesh.nodes[node], dtype=float) for node in groups_of_node}

    # One surface per driven node (contract 12.4): its incident child faces must be
    # all planar, or all on one cylinder.
    node_surface: Dict[int, _Surface] = {}
    for node in groups_of_node:
        classes = [child_classes[int(face)] for face in sorted(node_faces.get(node, ()))]
        if not classes:
            raise SubmodelError("SUBMODEL_NO_SHELL", f"driven node {node} belongs to no child shell face")
        first = classes[0]
        tolerance = 1.0e-9 * max(first.radius, source_extent)
        if any(c.kind != first.kind or (c.kind == "cylinder" and not _same_cylinder(first, c, tolerance))
               for c in classes[1:]):
            raise SubmodelError("SUBMODEL_SURFACE_JUNCTION",
                                f"driven node {node} joins faces of different surfaces; not supported in this slice")
        node_surface[node] = first
    groups: Dict[Any, List[int]] = {}
    for node, surface in node_surface.items():
        key = None if surface.kind == "planar" else surface
        matched = next((k for k in groups if k is not None and key is not None
                        and _same_cylinder(k, key, 1.0e-9 * max(key.radius, source_extent))), key)
        groups.setdefault(matched, []).append(node)

    try:
        from anymesher.submodel_correspondence import CylinderFrame

        hits = {}
        for surface, members in groups.items():
            subset = {node: points[node] for node in members}
            if surface is None:
                hits.update(locate_for_transfer(
                    source_mesh, subset, candidate_elements={n: candidates[n] for n in members},
                    physical_tolerance=settings.physical_tolerance, natural_tolerance=settings.natural_tolerance,
                    weight_tolerance=settings.weight_tolerance,
                ))
            else:
                frame = CylinderFrame(surface.origin, surface.axis, surface.radius,
                                      1.0e-9 * max(surface.radius, source_extent))
                hits.update(locate_for_transfer(
                    source_mesh, subset, candidate_elements={n: candidates[n] for n in members},
                    natural_tolerance=settings.natural_tolerance, weight_tolerance=settings.weight_tolerance,
                    cylinder=frame,
                ))
        hits = dict(sorted(hits.items()))
        source_allowance = estimate_static_error_allowance(source_built.fe_model, u_source, source_built.load_case)
        if source_allowance.inconclusive:
            raise SubmodelError("SOURCE_INCONCLUSIVE", "the source solve's error allowance is inconclusive")
        transfer = interpolate_nodal_field(source_built.fe_model, u_source, [
            TransferPoint(node, hit.source_element_id, hit.natural, tuple(points[node]), hit.weights,
                          hit.inverse_residual, hit.alternatives)
            for node, hit in hits.items()
        ])
        allowance = {
            node: interpolation_allowance(value, source_built.fe_model, u_source, source_allowance.displacement)
            for node, value in transfer.items()
        }
        built = build_fe_model(sub_project, sub_mesh, load_case, require_loads=False, require_supports=False,
                               apply_imperfections=False)
        normals = None
        if settings.policy == "release_normal_rotation":
            cache: Dict[int, np.ndarray] = {}
            normals = {}
            for node, hit in hits.items():
                surface = node_surface[node]
                if surface.kind == "cylinder":
                    # The analytic normal at x_f, not a facet normal (contract 12.5).
                    normals[node] = [_radial(surface, points[node])[1]]
                    continue
                for face in hit.incident_source_faces:
                    if face not in cache:
                        cache[face] = _face_normal(source_project, source_mesh, face)
                normals[node] = [cache[face] for face in hit.incident_source_faces]
        trace = build_driven_trace(
            built.fe_model, {node: value.values for node, value in transfer.items()},
            policy=settings.policy, normals=normals, allowance=allowance,
        )
        apply_driven_trace(built.fe_model, trace)
        u, info = solve_linear(built.fe_model, built.load_case)
        solution = LinearSolution(displacements=u, built=built, info=info, label="submodel")
        forces = interface_forces(built.fe_model, u, built.load_case, trace)
    except (CorrespondenceError, SubmodelTransferError) as error:
        raise SubmodelError(getattr(error, "code", "SUBMODEL_REFUSED"), str(error)) from error

    identity = _sha(_text((
        _model_hash(sub_project), _loading_hash(sub_project), _mesh_fingerprint(sub_mesh),
        sorted((str(k), tuple(sorted(int(e) for e in v))) for k, v in lineage.cut_faces.items()),
        sorted((int(k), tuple(sorted(int(f) for f in v))) for k, v in lineage.source_faces.items()),
        settings.policy, settings.physical_tolerance, settings.natural_tolerance, settings.weight_tolerance,
        str(load_case),
    )))
    result = SubmodelResult(solution, trace, forces, transfer, allowance, binding, identity, hits)
    result.diagnostics = _diagnostics(sub_mesh, built, lineage, groups_of_node, forces, source_allowance, allowance)
    curved = [hit for node, hit in hits.items() if node_surface[node].kind == "cylinder"]
    result.diagnostics["cylindrical"] = {
        "nodes": len(curved),
        "max_facet_offset": float(max((h.distance for h in curved), default=0.0)),
        "max_facet_bound": float(max((h.bound for h in curved), default=0.0)),
        "max_float_allowance": float(max((h.float_allowance for h in curved), default=0.0)),
        "note": "the source's facet-position error is a modelling error the submodel cannot remove",
    }
    return result


# ----------------------------------------------------------------------
# diagnostic
# ----------------------------------------------------------------------
def _resultant_terms(model: Any, forces: Mapping[int, Sequence[float]], about: np.ndarray):
    """Resultant (6) and its force/moment scales about ``about``."""

    total = np.zeros(6)
    force_scale = 0.0
    moment_scale = 0.0
    for node, six in forces.items():
        f = np.asarray(six, dtype=float).reshape(6)
        arm = np.asarray(model.mesh.get_node(int(node)).coords(), dtype=float) - about
        total[:3] += f[:3]
        total[3:] += f[3:] + np.cross(arm, f[:3])
        force_scale += float(np.linalg.norm(f[:3]))
        moment_scale += float(np.linalg.norm(f[3:])) + float(np.linalg.norm(arm)) * float(np.linalg.norm(f[:3]))
    return total, force_scale, moment_scale


def free_body_ratios(model: Any, force_sets: Iterable[Mapping[int, Sequence[float]]], about: Sequence[float]) -> Dict[str, Any]:
    """Free-body residual of several nodal force sets, with separate force and
    moment ratios (each normalised by the total absolute force or moment involved)."""

    point = np.asarray(about, dtype=float).reshape(3)
    total = np.zeros(6)
    force_scale = moment_scale = 0.0
    for forces in force_sets:
        part, fs, ms = _resultant_terms(model, forces, point)
        total += part
        force_scale += fs
        moment_scale += ms

    def ratio(residual: float, scale: float) -> float:
        if scale > 0.0:
            return residual / scale
        return 0.0 if residual == 0.0 else float("inf")

    return {
        "residual": total,
        "force_ratio": ratio(float(np.linalg.norm(total[:3])), force_scale),
        "moment_ratio": ratio(float(np.linalg.norm(total[3:])), moment_scale),
        "force_scale": force_scale,
        "moment_scale": moment_scale,
    }


def _diagnostics(sub_mesh, built, lineage, groups_of_node, forces, source_allowance, allowance):
    """Transfer/equilibrium diagnostic (workflow contract section 8).

    Evidence, not adequacy: matching prescribed displacements does not prove the
    global model was adequate at the cut.
    """

    from anysolver.matrix_assembly import assemble_load_vector

    model = built.fe_model
    corners = sorted(node for node, groups in groups_of_node.items() if len(groups) > 1)
    centre = np.mean([sub_mesh.nodes[n] for n in groups_of_node], axis=0)
    per_face = {}
    for group in lineage.cut_faces:
        nodes = [n for n, groups in groups_of_node.items() if group in groups and n not in corners]
        about = np.mean([sub_mesh.nodes[n] for n, g in groups_of_node.items() if group in g], axis=0)
        own, _fs, _ms = _resultant_terms(model, {n: forces.node_forces[n] for n in nodes}, about)
        per_face[str(group)] = {"about": about, "resultant": own, "nodes": len(nodes)}
    corner_records = {int(n): forces.node_forces[int(n)] for n in corners}
    whole, _fs, _ms = _resultant_terms(model, forces.node_forces, centre)
    # Rebuild the whole-cut resultant from the reported records:
    # M_centre = M_about + (about - centre) x F, corners added once.
    reconstructed = np.zeros(6)
    for record in per_face.values():
        force, moment = record["resultant"][:3], record["resultant"][3:]
        reconstructed[:3] += force
        reconstructed[3:] += moment + np.cross(record["about"] - centre, force)
    corner_total, _fs, _ms = _resultant_terms(model, corner_records, centre)
    reconstructed += corner_total

    load, _info = assemble_load_vector(model, built.load_case)
    load = np.asarray(load, dtype=float)
    dm = model.mesh.dof_manager
    applied = {}
    for node in model.mesh.nodes:
        six = load[dm.get_node_dofs(node)]
        if np.any(six != 0.0):
            applied[int(node)] = six
    body = free_body_ratios(model, (forces.node_forces, forces.retained_support_reactions, applied), centre)
    return {
        "driven_nodes": len(groups_of_node),
        "corner_records": corner_records,
        "per_cut_face": per_face,
        "whole_cut_resultant": whole,
        "about": centre,
        "grouping_reconstruction_error": float(np.max(np.abs(reconstructed - whole))),
        "free_body": body,
        "equilibrium_ratio": forces.equilibrium_ratio,
        "trace_residual_ratio": forces.trace_residual_ratio,
        "kept_trace_rows": int(np.size(forces.multipliers)),
        "source_eta_scaled": source_allowance.eta_scaled,
        "source_kappa_scaled": source_allowance.kappa_scaled,
        "max_transfer_allowance": float(max((float(np.max(a)) for a in allowance.values()), default=0.0)),
        "near_far_error": {
            "available": False,
            "reason": "needs a reference solution (acceptance slice S6); refinement_change() reports the "
                      "change between two source refinements instead",
        },
        "note": "transfer/equilibrium diagnostic; it does not prove the global model was adequate at the cut",
    }


def refinement_change(first: SubmodelResult, second: SubmodelResult, *, near_distance: float) -> Dict[str, Any]:
    """Change of one submodel driven from two refinements of the source.

    Both results must have the same comparison identity (child model and loading,
    mesh, cut, lineage, policy and transfer settings), the same source problem
    (source model without meshing controls, source loading, selected load case) and
    different source meshes; anything else refuses.  The change is reported near the cut (within
    ``near_distance`` of a driven node) and away from it.  It is a change between two
    results, not an error against a reference.
    """

    if not np.isfinite(near_distance) or near_distance <= 0.0:
        raise SubmodelError("CONTROL_INVALID", "near_distance must be positive")
    if not first.identity or first.identity != second.identity:
        raise SubmodelError("REFINEMENT_INCOMPATIBLE", "the two results differ in more than the source")
    a, b = first.binding, second.binding
    differs = [label for label, x, y in (("model", a.problem_hash, b.problem_hash),
                                         ("loading", a.loading_hash, b.loading_hash),
                                         ("load case", a.load_case, b.load_case)) if x != y]
    if differs:
        raise SubmodelError("REFINEMENT_SOURCE_PROBLEM",
                            f"the two sources differ in {', '.join(differs)}, not only in discretisation")
    if first.binding.mesh_fingerprint == second.binding.mesh_fingerprint:
        raise SubmodelError("REFINEMENT_SAME_SOURCE", "both results come from the same source mesh")
    if set(first.transfer) != set(second.transfer):
        raise SubmodelError("REFINEMENT_INCOMPATIBLE", "the driven node sets differ")
    mesh_a = first.solution.built.mesh
    driven = np.array([mesh_a.nodes[n] for n in first.transfer])
    near = np.zeros(6)
    far = np.zeros(6)
    far_count = 0
    for node, position in mesh_a.nodes.items():
        if node in first.transfer:
            continue
        change = np.abs(first.solution.node_displacement(node) - second.solution.node_displacement(node))
        if float(np.min(np.linalg.norm(driven - np.asarray(position), axis=1))) <= near_distance:
            near = np.maximum(near, change)
        else:
            far = np.maximum(far, change)
            far_count += 1
    return {"near_cut_max_change": near, "away_from_cut_max_change": far, "away_nodes": far_count,
            "near_nodes": len(mesh_a.nodes) - len(first.transfer) - far_count,
            "near_distance": float(near_distance), "kind": "change between source refinements, not an error"}
