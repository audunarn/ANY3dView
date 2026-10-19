"""Results, addressed the way the model was built.

The solver returns flat DOF vectors.  Postprocessing here re-attaches them to
the things the user actually placed -- points, lines and plates -- so a query
never has to know a node number.

Every analysis produces one or more *shapes*: a static deflection, a vibration
mode, a buckling mode, a step of a nonlinear path, an instant of a transient.
They all share :class:`ShapeView`, so anything that can display a static result
-- the scene builder, the results panel, the animation -- displays a mode or a
time step without knowing the difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from anygeometry.entities import EntityRef
from ..solve.build import BuiltModel

__all__ = [
    "BucklingSolution",
    "CapacitySolution",
    "ImpactSolution",
    "ImportedSolution",
    "LinearBatchSolution",
    "LinearSolution",
    "LinearStaticBatchResult",
    "ModalSolution",
    "MultiShapeSolution",
    "NonlinearSolution",
    "NonlinearIncrementView",
    "ShapeView",
    "TransientSolution",
]

_DOF_INDEX = {"ux": 0, "uy": 1, "uz": 2, "rx": 3, "ry": 4, "rz": 5}


@dataclass
class ShapeView:
    """One displacement field over the model.

    ``value`` carries whatever number the shape is indexed by -- a frequency, a
    buckling load factor, a load factor, a time -- so a caller can label it
    without knowing which analysis produced it.
    """

    displacements: np.ndarray
    built: BuiltModel
    label: str = "result"
    value: float = 0.0

    # ------------------------------------------------------------------
    # displacement access
    # ------------------------------------------------------------------
    def node_displacement(self, node_id: int) -> np.ndarray:
        """The six displacement components at one node."""

        dofs = self.built.fe_model.mesh.dof_manager.get_node_dofs(node_id)
        if not dofs:
            raise KeyError(f"node {node_id} is not in the solved model")
        return self.displacements[dofs]

    def displacement_at(self, ref: EntityRef) -> Dict[int, np.ndarray]:
        """Displacements of every node on one geometry entity."""

        return {
            node_id: self.node_displacement(node_id)
            for node_id in self.built.mesh.nodes_on(ref)
        }

    def connection_motion(self, connection: str) -> Dict[str, np.ndarray]:
        """Where a connection's reference point is and how it moved.

        ``connection`` is its name or id.  For a rigid connection this is the
        motion of the whole tied section; for a distributing one it is the
        weighted average motion of the tied nodes.  Rotations are in radians.
        """

        item = self.built.connections.get(connection)
        if item is None:
            item = next(
                (
                    candidate
                    for candidate in self.built.connections.values()
                    if candidate.connection.name == connection
                ),
                None,
            )
        if item is None:
            raise KeyError(f"no connection {connection!r} in the solved model")
        values = self.node_displacement(item.reference_node)
        motion = {
            "reference_point": np.asarray(item.reference, dtype=float).copy(),
            "displacement": np.asarray(values[:3], dtype=float),
            "rotation": np.asarray(values[3:], dtype=float),
        }
        if item.section_node is not None:
            # A spring connection: the loaded point and the section it is
            # sprung from move differently; the difference is the spring.
            section = self.node_displacement(item.section_node)
            motion["section_displacement"] = np.asarray(section[:3], dtype=float)
            motion["section_rotation"] = np.asarray(section[3:], dtype=float)
        return motion

    def point_displacement(self, ref: EntityRef) -> np.ndarray:
        """The six components at a modelled point."""

        if ref.kind != "vertex":
            raise ValueError("point_displacement expects a point reference")
        node_id = self.built.mesh.node_of_vertex.get(ref.id)
        if node_id is None:
            raise KeyError(f"{ref} has no node in the mesh")
        return self.node_displacement(node_id)

    def component(self, name: str) -> np.ndarray:
        """One displacement component for every node, ordered by node ID."""

        try:
            offset = _DOF_INDEX[name]
        except KeyError:
            raise ValueError(
                f"unknown component {name!r}; expected one of "
                f"{', '.join(_DOF_INDEX)}"
            ) from None
        manager = self.built.fe_model.mesh.dof_manager
        return np.array(
            [
                self.displacements[manager.get_node_dofs(node_id)[offset]]
                for node_id in sorted(self.built.mesh.nodes)
            ]
        )

    def translations(self) -> np.ndarray:
        """Translation vectors for every node, ordered by node ID."""

        manager = self.built.fe_model.mesh.dof_manager
        return np.array(
            [
                self.displacements[manager.get_node_dofs(node_id)[:3]]
                for node_id in sorted(self.built.mesh.nodes)
            ]
        )

    def max_translation(self) -> Tuple[int, float]:
        """The node with the largest translation magnitude, and that magnitude."""

        node_ids = sorted(self.built.mesh.nodes)
        magnitudes = np.linalg.norm(self.translations(), axis=1)
        index = int(np.argmax(magnitudes))
        return node_ids[index], float(magnitudes[index])

    def deformed_positions(self, scale: float = 1.0) -> np.ndarray:
        """Node positions displaced by ``scale`` times the translation."""

        base = self.built.mesh.node_positions()
        return base + scale * self.translations()


@dataclass
class ImportedSolution(ShapeView):
    """A result another solver produced, displayed through the same interface.

    Two things make it different from a solved one, and both are enforced
    rather than smoothed over.

    It knows **which components it has**.  A CalculiX FRD carries three
    translations and no rotations, and a rotation reported as zero is a
    plausible number and a wrong one, so asking for an absent component raises.
    The underlying array holds NaN there, so code that indexes it directly
    still cannot mistake the gap for an answer.

    Its **stresses came from the file**, not from a recovery here.  They are
    kept as ready-made fields; nothing recomputes them, and a node-valued
    stress stays node-valued so it is never confused with something this layer
    derived.
    """

    results: Any = None
    components: frozenset = field(default_factory=frozenset)
    fields: Dict[str, Any] = field(default_factory=dict)
    covered: int = 0
    label: str = "imported"

    def _require_translations(self) -> None:
        if not {"ux", "uy", "uz"}.issubset(self.components):
            raise KeyError(
                f"{self.label} carries no displacement field; this imported "
                "result contains stresses only"
            )

    def node_displacement(self, node_id: int) -> np.ndarray:
        self._require_translations()
        return super().node_displacement(node_id)

    def translations(self) -> np.ndarray:
        self._require_translations()
        return super().translations()

    def component(self, name: str) -> np.ndarray:
        if name in _DOF_INDEX and name not in self.components:
            available = ", ".join(sorted(self.components)) or "none"
            raise KeyError(
                f"{self.label} carries no {name!r}: this file format does not "
                f"store it. Available components are {available}. It is not "
                "zero -- it is absent."
            )
        return super().component(name)

    def available_fields(self) -> List[str]:
        """Displacement components plus whatever stresses the file named."""

        order = [name for name in _DOF_INDEX if name in self.components]
        displacement = ["magnitude"] + order if order else []
        return displacement + list(self.fields)

    def summary(self) -> str:
        pieces = [self.label, f"{self.covered} nodes matched"]
        if self.results is not None:
            pieces.append(self.results.summary())
        if {"ux", "uy", "uz"}.issubset(self.components):
            node_id, magnitude = self.max_translation()
            pieces.append(
                f"max translation {magnitude:.6g} m at node {node_id}"
            )
        return "; ".join(pieces)


@dataclass
class LinearSolution(ShapeView):
    """A linear static result."""

    info: Dict[str, Any] = field(default_factory=dict)
    label: str = "static"
    _stress: Any = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def stresses(self, **kwargs: Any):
        """Recover element stresses, caching the result.

        Delegates to the solver's unified recovery so provenance and fallback
        labelling are the solver's, not a reimplementation here.
        """

        if self._stress is None or kwargs:
            from anysolver import recover_stress_result

            result = recover_stress_result(
                self.built.fe_model, self.displacements, **kwargs
            )
            if not kwargs:
                self._stress = result
            return result
        return self._stress

    def face_von_mises(self, ref: EntityRef) -> Dict[int, float]:
        """Peak von Mises stress per element on one plate."""

        if ref.kind != "face":
            raise ValueError("face_von_mises expects a plate reference")
        recovered = self.stresses()
        values: Dict[int, float] = {}
        for element_id in self.built.mesh.elements_on(ref):
            stress = recovered.element_stresses.get(element_id)
            if stress is None:
                continue
            von_mises = stress.get("von_mises")
            if von_mises is None:
                continue
            values[element_id] = float(np.max(np.abs(von_mises)))
        return values

    def summary(self) -> str:
        node_id, magnitude = self.max_translation()
        mesh = self.built.mesh
        return (
            f"{self.built.project.name}: {mesh.num_nodes} nodes, "
            f"{mesh.num_elements} elements, "
            f"max translation {magnitude:.6g} m at node {node_id}"
        )

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"LinearSolution({self.summary()})"


# ----------------------------------------------------------------------
# multi-shape results
# ----------------------------------------------------------------------
@dataclass
class MultiShapeSolution:
    """A result made of several shapes, browsed by index."""

    built: BuiltModel
    shapes: List[ShapeView] = field(default_factory=list)
    status: str = "ok"
    info: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.shapes)

    def __getitem__(self, index: int) -> ShapeView:
        return self.shapes[index]

    def __iter__(self):
        return iter(self.shapes)

    @property
    def values(self) -> List[float]:
        return [shape.value for shape in self.shapes]

    @property
    def labels(self) -> List[str]:
        return [shape.label for shape in self.shapes]

    def shape(self, index: int = 0) -> ShapeView:
        if not self.shapes:
            raise IndexError(f"{type(self).__name__} has no shapes")
        return self.shapes[index]

    # A multi-shape result stands in for its first shape wherever a single
    # displacement field is expected, so existing displays keep working.
    @property
    def displacements(self) -> np.ndarray:
        return self.shape(0).displacements

    def max_translation(self) -> Tuple[int, float]:
        return self.shape(0).max_translation()


@dataclass
class LinearBatchSolution(MultiShapeSolution):
    """Several linear load cases solved through one stiffness factorization."""

    case_names: tuple[str, ...] = ()

    def case(self, name: str) -> LinearSolution:
        try:
            index = self.case_names.index(str(name))
        except ValueError:
            raise KeyError(f"no solved load case {name!r}") from None
        shape = self.shapes[index]
        if not isinstance(shape, LinearSolution):  # pragma: no cover - invariant
            raise TypeError("linear batch contains a non-linear shape")
        return shape

    def connection_history(self, connection: str) -> Dict[str, np.ndarray]:
        """A connection's reference point through every solved case.

        Returns ``case`` (names), ``time`` (the cases' times, ``nan`` where a case
        has none), ``displacement`` and ``rotation`` (one row per case), in the
        order the cases were solved.
        """

        project = self.built.project
        motions = [shape.connection_motion(connection) for shape in self.shapes]
        times = [
            np.nan if project.load_cases[name].time is None else project.load_cases[name].time
            for name in self.case_names
        ]
        return {
            "case": np.array(self.case_names),
            "time": np.array(times, dtype=float),
            "displacement": np.array([m["displacement"] for m in motions]),
            "rotation": np.array([m["rotation"] for m in motions]),
        }

    def summary(self) -> str:
        return (
            f"{self.built.project.name}: {len(self.shapes)} linear load cases "
            "solved with one factorization"
        )


@dataclass(frozen=True)
class LinearStaticBatchResult:
    """One streamed block; only explicitly selected DOFs are retained."""

    built: BuiltModel
    block: Any
    connections: Mapping[str, Any]
    labels: tuple[str, ...] = ()
    node_dofs: Mapping[int, tuple[int, ...]] = field(default_factory=dict, repr=False)
    _rows: Mapping[int, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_rows", {
            int(dof): index for index, dof in enumerate(self.output_dofs)
        })

    @property
    def input_indices(self):
        return self.block.input_indices

    @property
    def case_ids(self):
        return self.block.case_ids

    @property
    def times(self):
        return self.block.times

    @property
    def output_dofs(self):
        return self.block.output_dofs

    @property
    def displacements(self):
        return self.block.displacements

    @property
    def case_results(self):
        return self.block.case_results

    @property
    def diagnostics(self):
        return self.block.diagnostics

    def node_displacement(self, node_id: int, case_index: int = 0) -> np.ndarray:
        dofs = self.node_dofs.get(int(node_id), ())
        if not dofs:
            raise KeyError(f"node {node_id} is not in the structural model")
        rows = self._rows
        if any(int(dof) not in rows for dof in dofs):
            raise KeyError(f"node {node_id} displacement was not selected")
        return self.displacements[[rows[int(dof)] for dof in dofs], case_index].copy()

    def connection_motion(self, connection: str, case_index: int = 0) -> Dict[str, np.ndarray]:
        item = self.connections.get(connection)
        if item is None:
            item = next((value for value in self.connections.values()
                         if value.connection.name == connection), None)
        if item is None:
            raise KeyError(f"connection {connection!r} motion was not selected")
        values = self.node_displacement(item.reference_node, case_index)
        motion = {
            "reference_point": np.asarray(item.reference, dtype=float).copy(),
            "displacement": values[:3], "rotation": values[3:],
        }
        if item.section_node is not None:
            section = self.node_displacement(item.section_node, case_index)
            motion.update(section_displacement=section[:3], section_rotation=section[3:])
        return motion

    def stresses(self, case_index: int = 0):
        """Return the solver's selected element/component/station stress maps."""
        return self.case_results[case_index]["element_stresses"]

    def reactions(self, case_index: int = 0):
        return self.case_results[case_index]["reactions"]


@dataclass
class ModalSolution(MultiShapeSolution):
    """Natural frequencies and their mode shapes."""

    rigid_body_modes: int = 0

    @property
    def frequencies(self) -> List[float]:
        """Natural frequencies in Hz, one per mode."""

        return [shape.value for shape in self.shapes]

    def periods(self) -> List[float]:
        return [
            float("inf") if frequency <= 0.0 else 1.0 / frequency
            for frequency in self.frequencies
        ]

    def summary(self) -> str:
        if not self.shapes:
            return f"{self.built.project.name}: no modes found"
        listed = ", ".join(f"{value:.4g}" for value in self.frequencies[:5])
        extra = "" if len(self.shapes) <= 5 else ", ..."
        note = (
            ""
            if not self.rigid_body_modes
            else f" ({self.rigid_body_modes} rigid-body)"
        )
        return (
            f"{self.built.project.name}: {len(self.shapes)} modes{note}, "
            f"f = {listed}{extra} Hz"
        )


@dataclass
class BucklingSolution(MultiShapeSolution):
    """Elastic buckling load factors and their mode shapes.

    A load factor multiplies the *reference load case* that produced the
    prestress, so it is only meaningful together with that case.
    """

    reference_case: str = "default"

    @property
    def load_factors(self) -> List[float]:
        return [shape.value for shape in self.shapes]

    @property
    def critical_factor(self) -> float:
        if not self.shapes:
            raise IndexError("no buckling modes were found")
        return float(min(self.load_factors))

    def summary(self) -> str:
        if not self.shapes:
            return f"{self.built.project.name}: no buckling modes found"
        listed = ", ".join(f"{value:.4g}" for value in self.load_factors[:5])
        return (
            f"{self.built.project.name}: critical factor "
            f"{self.critical_factor:.5g} on case {self.reference_case!r} "
            f"(factors {listed})"
        )


def _recover_committed_stresses(view, kwargs):
    """Consume owner recovery with the retained state and its kinematics."""
    if view._stress is not None and not kwargs:
        return view._stress
    from anysolver import recover_stress_result
    caller_options = bool(kwargs)
    if view.raw_result is not None and "nonlinear_result" not in kwargs and "element_states" not in kwargs:
        kwargs["nonlinear_result"] = view.raw_result
    kinematics = getattr(view, "recovery_kinematics", None)
    if kinematics is not None and "kinematics" not in kwargs:
        kwargs["kinematics"] = kinematics
    result = recover_stress_result(view.built.fe_model, view.displacements, **kwargs)
    if not caller_options:
        view._stress = result
    return result


@dataclass
class NonlinearSolution(ShapeView):
    """The converged end state of an incremental solve, plus its path."""

    steps: List[Any] = field(default_factory=list)
    status: str = "ok"
    info: Dict[str, Any] = field(default_factory=dict)
    label: str = "nonlinear"
    peak_load_factor: Optional[float] = None
    deleted_elements: Tuple[int, ...] = ()
    # The committed solver result is required for history-aware recovery.
    # Keeping only ``displacements`` would reconstruct yielded stresses as if
    # the material were elastic.
    raw_result: Any = field(default=None, repr=False)
    _stress: Any = field(default=None, repr=False)
    _increment_views: Optional[List[Any]] = field(default=None, init=False, repr=False)

    @property
    def shapes(self) -> List[Any]:
        """True saved nonlinear increments, or no frames when none were saved.

        A converged step record contains useful scalar diagnostics but not a
        displacement/state field.  Only the solver's opt-in committed
        snapshots can therefore be displayed or animated.  Keeping the empty
        case explicit prevents the UI from fabricating intermediate shapes by
        scaling the final displacement.
        """

        if self._increment_views is not None:
            return self._increment_views
        snapshots = tuple(getattr(self.raw_result, "snapshots", ()) or ())
        steps = {
            int(getattr(step, "step_index", index)): step
            for index, step in enumerate(self.steps, start=1)
        }
        count = len(snapshots)
        raw_info = getattr(self.raw_result, "info", None)
        kinematics = raw_info.get("kinematics") if isinstance(raw_info, Mapping) else None
        self._increment_views = [
            NonlinearIncrementView(
                displacements=np.asarray(snapshot.displacements, dtype=float),
                built=self.built,
                label=(
                    f"increment {index}/{count}"
                    f"  (load factor {float(snapshot.load_factor):.5g})"
                ),
                value=float(snapshot.load_factor),
                step=steps.get(int(getattr(snapshot, "step_index", index))),
                raw_result=snapshot,
                recovery_kinematics=kinematics,
            )
            for index, snapshot in enumerate(snapshots, start=1)
        ]
        return self._increment_views

    @property
    def element_states(self) -> Mapping[int, Any]:
        return getattr(self.raw_result, "element_states", {}) or {}

    @property
    def load_factor(self) -> float:
        """How much of the load case was actually carried."""

        return float(self.value)

    def stresses(self, **kwargs: Any):
        """Recover stresses from the committed material history.

        A nonlinear end displacement is not, by itself, enough to reconstruct
        a yielded stress state.  The retained raw solver result supplies the
        committed shell-layer/beam-fibre states to ANYsolver's unified
        recovery.  Callers can still explicitly supply ``element_states`` or a
        different ``nonlinear_result`` for restart/diagnostic comparisons.
        """

        return _recover_committed_stresses(self, kwargs)

    def available_fields(self) -> List[str]:
        """Display fields actually carried by this committed nonlinear state."""

        from .fields import available_fields

        fields = available_fields()
        if any(
            isinstance(state, Mapping)
            and np.asarray(state.get("alpha", ())).size > 0
            for state in self.element_states.values()
        ):
            fields.append("equivalent_plastic_strain")
        return fields

    def history(self) -> Dict[str, np.ndarray]:
        """Load factor and displacement norm at each converged step.

        This is the load-displacement path: the thing worth plotting.
        """

        history = {
            "step": np.array([step.step_index for step in self.steps]),
            "load_factor": np.array([step.load_factor for step in self.steps]),
            "displacement_norm": np.array(
                [step.displacement_norm for step in self.steps]
            ),
            "iterations": np.array([step.iterations for step in self.steps]),
        }
        support_names = sorted(
            {
                str(name)
                for step in self.steps
                for name in (getattr(step, "support_reactions", {}) or {})
            }
        )
        components = ("Fx", "Fy", "Fz")
        for name in support_names:
            rows = []
            for step in self.steps:
                values = (getattr(step, "support_reactions", {}) or {}).get(name)
                row = np.asarray(
                    values if values is not None else (np.nan,) * 6,
                    dtype=float,
                ).reshape(-1)
                rows.append(row if row.size >= 6 else np.full(6, np.nan))
            matrix = np.vstack(rows) if rows else np.empty((0, 6), dtype=float)
            history[f"support_reaction::{name}::force_magnitude"] = np.linalg.norm(
                matrix[:, :3], axis=1
            )
            for index, component in enumerate(components):
                history[f"support_reaction::{name}::{component}"] = matrix[:, index]
        return history

    def summary(self) -> str:
        node_id, magnitude = self.max_translation()
        peak = (
            ""
            if self.peak_load_factor is None
            else f", peak factor {self.peak_load_factor:.5g}"
        )
        eroded = (
            ""
            if not self.deleted_elements
            else f", {len(self.deleted_elements)} element(s) eroded"
        )
        return (
            f"{self.built.project.name}: {self.status}, "
            f"load factor {self.load_factor:.5g}{peak}, "
            f"{len(self.steps)} steps, "
            f"max translation {magnitude:.6g} m at node {node_id}{eroded}"
        )


@dataclass
class NonlinearIncrementView(ShapeView):
    """One real committed nonlinear increment used for contours/playback."""

    step: Any = None
    raw_result: Any = field(default=None, repr=False)
    _stress: Any = field(default=None, repr=False)
    recovery_kinematics: Optional[str] = None

    def stresses(self, **kwargs):
        """Recover this committed snapshot through the shared owner adapter."""
        return _recover_committed_stresses(self, kwargs)

    @property
    def element_states(self) -> Mapping[int, Any]:
        return getattr(self.raw_result, "element_states", {}) or {}


@dataclass
class CapacitySolution(NonlinearSolution):
    """A whole capacity assessment: static, buckling, imperfection, collapse.

    A ``NonlinearSolution`` at heart -- the shape it carries is the imperfect
    nonlinear end state, so everything that displays a nonlinear result displays
    this unchanged -- with the earlier stages kept alongside rather than thrown
    away.  The elastic critical factor and the actual capacity are the two
    numbers the workflow exists to compare, and seeing them apart is the point:
    the ratio is how much the imperfection and the plasticity cost.
    """

    label: str = "capacity"
    critical_factor: Optional[float] = None
    buckling: Optional["BucklingSolution"] = None
    mesh_adequacy: Dict[str, Any] = field(default_factory=dict)

    @property
    def capacity_factor(self) -> float:
        """The load factor the structure actually reached."""

        return float(self.value)

    @property
    def capacity_ratio(self) -> Optional[float]:
        """Capacity over elastic critical load, when both are known.

        Either side of one is a real result, and which one says what kind of
        structure this is.  Below one is imperfection sensitivity: a shell or a
        column that yields, or that its imperfection knocks down, never reaches
        the elastic critical load.  Above one is post-buckling reserve: a plate
        in compression keeps taking load after it buckles, by shedding it into
        membrane action near the supported edges.  Neither is a warning sign on
        its own -- reading it as one would be the easy mistake.
        """

        if not self.critical_factor:
            return None
        return self.capacity_factor / float(self.critical_factor)

    def summary(self) -> str:
        pieces = [f"capacity {self.capacity_factor:.4g} x the reference load"]
        if self.critical_factor:
            pieces.append(f"elastic critical {self.critical_factor:.4g}")
            ratio = self.capacity_ratio
            if ratio is not None:
                pieces.append(f"capacity/critical {ratio:.3f}")
        status = (self.mesh_adequacy or {}).get("status")
        if status and status != "ok":
            pieces.append(f"mesh adequacy: {status}")
        return "; ".join(pieces)


@dataclass
class TransientSolution(MultiShapeSolution):
    """A response history, one shape per stored time step."""

    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    peak_displacement: float = 0.0
    peak_node: Optional[int] = None

    def at_time(self, time: float) -> ShapeView:
        """The stored step nearest a given time."""

        if not self.shapes:
            raise IndexError("the transient result has no stored steps")
        index = int(np.argmin(np.abs(self.times - float(time))))
        return self.shapes[index]

    def node_history(self, node_id: int, component: str = "uz") -> np.ndarray:
        """One component at one node, over every stored step."""

        try:
            offset = _DOF_INDEX[component]
        except KeyError:
            raise ValueError(f"unknown component {component!r}") from None
        dofs = self.built.fe_model.mesh.dof_manager.get_node_dofs(node_id)
        if not dofs:
            raise KeyError(f"node {node_id} is not in the solved model")
        return np.array(
            [shape.displacements[dofs[offset]] for shape in self.shapes]
        )

    def summary(self) -> str:
        span = 0.0 if self.times.size == 0 else float(self.times[-1])
        node = "" if self.peak_node is None else f" at node {self.peak_node}"
        return (
            f"{self.built.project.name}: {self.status}, {len(self.shapes)} steps "
            f"to t = {span:g} s, peak displacement "
            f"{self.peak_displacement:.6g} m{node}"
        )


@dataclass
class ImpactSolution(MultiShapeSolution):
    """A rigid-sphere impact: the response, and what the contact did.

    Browsable by time like a transient, plus the things only an impact has --
    where the sphere was, how hard it pushed, how long it touched, and whether
    the model gave up any elements.
    """

    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sphere_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    sphere_velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # One force vector per saved step, not a scalar: the direction matters
    # for an oblique impact, so the magnitude is derived rather than stored.
    contact_force_history: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3))
    )
    peak_contact_force: float = 0.0
    contact_duration: float = 0.0
    max_penetration: float = 0.0
    max_penetration_ratio: float = 0.0
    momentum_balance_error: float = 0.0
    peak_displacement: float = 0.0
    peak_node: Optional[int] = None
    deleted_elements: Tuple[int, ...] = ()
    collision: Any = None
    timing: Any = None

    # ------------------------------------------------------------------
    def at_time(self, time: float) -> ShapeView:
        """The stored step nearest a given time."""

        if not self.shapes:
            raise IndexError("the impact result has no stored steps")
        index = int(np.argmin(np.abs(self.times - float(time))))
        return self.shapes[index]

    def sphere_speed(self) -> np.ndarray:
        """Sphere speed at each stored step."""

        if self.sphere_velocities.size == 0:
            return np.zeros(0)
        return np.linalg.norm(self.sphere_velocities, axis=1)

    def energy(self) -> Dict[str, float]:
        """Kinetic energy before and after, and what the structure took.

        The difference is energy the sphere no longer has: it went into
        deformation, plastic work and damage.  It is not a strain-energy
        measurement, and it is not split into those parts here.
        """

        if self.collision is None or self.sphere_velocities.size == 0:
            return {"initial": 0.0, "final": 0.0, "absorbed": 0.0}
        mass = float(self.collision.mass)
        speeds = self.sphere_speed()
        initial = 0.5 * mass * float(speeds[0]) ** 2
        final = 0.5 * mass * float(speeds[-1]) ** 2
        return {"initial": initial, "final": final, "absorbed": initial - final}

    def contact_force_magnitude(self) -> np.ndarray:
        """Contact force magnitude at each stored step."""

        history = np.asarray(self.contact_force_history, dtype=float)
        if history.size == 0:
            return np.zeros(0)
        return np.linalg.norm(history, axis=1) if history.ndim == 2 else np.abs(history)

    def contact_history(self) -> Dict[str, np.ndarray]:
        """Time against contact force and sphere speed, for plotting."""

        return {
            "time": np.asarray(self.times, dtype=float),
            "contact_force": self.contact_force_magnitude(),
            "contact_force_vector": np.asarray(
                self.contact_force_history, dtype=float
            ),
            "sphere_speed": self.sphere_speed(),
        }

    def time_of_peak_force(self) -> float:
        """When the contact force peaked."""

        magnitude = self.contact_force_magnitude()
        if magnitude.size == 0:
            return 0.0
        return float(self.times[int(np.argmax(magnitude))])

    def touched(self) -> bool:
        """Whether the sphere actually made contact."""

        return self.peak_contact_force > 0.0

    def summary(self) -> str:
        if not self.touched():
            return (
                f"{self.built.project.name}: {self.status}, the sphere did not "
                f"make contact over {len(self.shapes)} steps"
            )
        energy = self.energy()
        deleted = (
            ""
            if not self.deleted_elements
            else f", {len(self.deleted_elements)} element(s) deleted"
        )
        return (
            f"{self.built.project.name}: {self.status}, peak contact force "
            f"{self.peak_contact_force:.6g} N over {self.contact_duration:.4g} s, "
            f"{energy['absorbed'] / 1000.0:.4g} kJ absorbed, peak displacement "
            f"{self.peak_displacement:.6g} m{deleted}"
        )
