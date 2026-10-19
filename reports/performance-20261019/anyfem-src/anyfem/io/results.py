"""Reading results back from an external solver.

A model can be exported, solved somewhere else, and the answers brought back
here to be looked at through the same contours, probes, paths and reports that
an ANYfem solve uses. ANYfileio owns both supported format parsers; this module
only adapts their neutral records to ANYfem fields and solutions.

The awkward part is not the parsing.  It is that an external result does not
carry the same things an internal one does, and the difference must not be
papered over:

* A CalculiX FRD carries **three** displacement components per node.  There are
  no rotations in it.  A shell rotation of zero is a perfectly plausible number
  and completely wrong, so absent components are *refused* when asked for, not
  returned as zero.
* Stresses arrive **per node**, already averaged by the writing solver, rather
  than per element from a recovery here.  ``Field`` distinguishes the two, and
  an imported stress stays a node field so nothing downstream mistakes it for
  something this layer computed.
* Component names come from the file.  They are carried through as they are
  found rather than mapped onto ANYfem's own list, for the same reason recovery
  is not filtered: a list frozen here silently drops whatever it does not know.

Matching is by node ID, which means the result file and the model must be the
same mesh.  When they are not, that is reported with the overlap rather than
attached partially -- a field covering a third of the model still draws a
picture, and the picture is a lie.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..post.fields import Field

__all__ = [
    "ImportedResults",
    "ResultImportError",
    "import_calculix_results",
    "import_sesam_results",
]

# The order the solver reports a symmetric stress tensor in.  Only used when a
# file gives components positionally and does not name them.
_TENSOR_COMPONENTS: Tuple[str, ...] = (
    "sxx", "syy", "szz", "sxy", "syz", "szx",
)


class ResultImportError(ValueError):
    """Raised when a result file cannot be attached to a model."""


@dataclass
class ImportedResults:
    """Results read from another solver's output file.

    Holds them in the file's own terms -- node IDs, whatever components it
    named -- until they are attached to a model, because until then there is
    nothing to check them against.
    """

    source: Path
    format: str
    displacements: Dict[int, Tuple[float, ...]] = field(default_factory=dict)
    node_stresses: Dict[int, Dict[str, float]] = field(default_factory=dict)
    element_stresses: Dict[int, Dict[str, float]] = field(default_factory=dict)
    reactions: Dict[int, Tuple[float, ...]] = field(default_factory=dict)
    buckling_factors: Tuple[float, ...] = ()
    frequencies: Tuple[float, ...] = ()
    warnings: Tuple[str, ...] = ()
    source_hash: str = ""

    @property
    def components(self) -> List[str]:
        """Every stress component name the file actually carried."""

        names: List[str] = []
        for source in (self.node_stresses, self.element_stresses):
            for values in source.values():
                for name in values:
                    if name not in names:
                        names.append(name)
        return names

    @property
    def has_rotations(self) -> bool:
        """Whether every displacement record includes the three rotations."""

        return bool(self.displacements) and all(
            len(value) >= 6 for value in self.displacements.values()
        )

    @property
    def displacement_components(self) -> Tuple[str, ...]:
        """Components present for every displacement record in the file."""

        if not self.displacements:
            return ()
        return (
            ("ux", "uy", "uz", "rx", "ry", "rz")
            if self.has_rotations
            else ("ux", "uy", "uz")
        )

    def summary(self) -> str:
        pieces = [f"{self.format} results from {self.source.name}"]
        if self.displacements:
            width = 6 if self.has_rotations else 3
            pieces.append(
                f"{len(self.displacements)} node displacements "
                f"({width} components)"
            )
        if self.node_stresses:
            pieces.append(f"{len(self.node_stresses)} node stresses")
        if self.element_stresses:
            pieces.append(f"{len(self.element_stresses)} element stresses")
        if self.buckling_factors:
            pieces.append(f"{len(self.buckling_factors)} buckling factors")
        if self.frequencies:
            pieces.append(f"{len(self.frequencies)} frequencies")
        return "; ".join(pieces)

    # ------------------------------------------------------------------
    def attach(self, built: Any, *, require_all_nodes: bool = True):
        """Bind these results to a model, giving a displayable shape.

        ``built`` is a ``BuiltModel`` from an ANYfem solve, or an
        ``ImportedModel`` from a file import.  Matching is by node ID.
        """

        from ..post.results import ImportedSolution

        mesh = getattr(built, "mesh", None)
        if mesh is None:
            raise ResultImportError(
                "attach needs a built or imported model with a mesh"
            )

        model_nodes = set(mesh.nodes)
        displacement_nodes = set(self.displacements)
        stress_nodes = set(self.node_stresses)
        result_nodes = displacement_nodes | stress_nodes
        model_elements = set(mesh.shells) | set(mesh.beams)
        stress_elements = set(self.element_stresses)
        if not result_nodes and not stress_elements:
            raise ResultImportError(
                f"{self.source.name} carries no nodal results or element "
                "stresses to attach"
            )

        malformed = sorted(
            node_id
            for node_id, values in self.displacements.items()
            if len(values) not in (3, 6)
        )
        if malformed:
            raise ResultImportError(
                f"{self.source.name} has displacement records that are not "
                f"three translations or six DOFs at node(s) {malformed[:10]}"
            )

        mismatches = []
        for label, result_ids, model_ids in (
            ("displacement nodes", displacement_nodes, model_nodes),
            ("stress nodes", stress_nodes, model_nodes),
            ("stress elements", stress_elements, model_elements),
        ):
            if not result_ids:
                continue
            missing = len(model_ids - result_ids)
            extra = len(result_ids - model_ids)
            if missing or extra:
                mismatches.append(f"{label}: {missing} missing, {extra} extra")
        if require_all_nodes and mismatches:
            raise ResultImportError(
                f"{self.source.name} does not match this model: "
                f"{'; '.join(mismatches)}. Results are matched by node and "
                "element ID, so each field has to come from the same mesh. "
                "Pass require_all_nodes=False to attach the overlap anyway, "
                "knowing the picture will be partial."
            )

        return ImportedSolution(
            displacements=self._dof_vector(built, mesh),
            built=built,
            label=f"{self.format}: {self.source.name}",
            results=self,
            components=frozenset(self.displacement_components),
            fields=self._fields(mesh),
            covered=len(model_nodes & result_nodes),
        )

    # ------------------------------------------------------------------
    def _dof_vector(self, built: Any, mesh: Any) -> np.ndarray:
        """A solver-shaped DOF vector, with absent components left as NaN.

        NaN rather than zero on purpose.  Anything that reaches around
        :meth:`ImportedSolution.component` and indexes the array directly gets
        a number that cannot be mistaken for an answer.
        """

        manager = built.fe_model.mesh.dof_manager
        total = int(getattr(built.fe_model.mesh, "total_dofs", 0)) or (
            6 * len(mesh.nodes)
        )
        vector = np.full(total, np.nan, dtype=float)
        for node_id, values in self.displacements.items():
            dofs = manager.get_node_dofs(node_id)
            if not dofs:
                continue
            for index, value in enumerate(values[: len(dofs)]):
                vector[dofs[index]] = float(value)
        return vector

    def _fields(self, mesh: Any) -> Dict[str, Field]:
        """One Field per component the file carried."""

        built: Dict[str, Field] = {}
        for name in self.components:
            node_values = {
                node_id: float(values[name])
                for node_id, values in self.node_stresses.items()
                if name in values and node_id in mesh.nodes
            }
            element_values = {
                element_id: float(values[name])
                for element_id, values in self.element_stresses.items()
                if name in values
                and (element_id in mesh.shells or element_id in mesh.beams)
            }
            if not node_values and not element_values:
                continue
            built[name] = Field(
                name=name,
                unit="Pa",
                # Element values win when a file gives both, because they are
                # what the writing solver computed; the nodal ones are its
                # average of them.
                node_values={} if element_values else node_values,
                element_values=element_values,
            )
        return built


# ----------------------------------------------------------------------
# CalculiX
# ----------------------------------------------------------------------
def _parse_snapshot(source,reader):
    """Parse exactly the bytes authenticated by the imported provenance."""
    import hashlib,tempfile
    digest=hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="anyfem-result-") as directory:
        snapshot=Path(directory)/Path(source).name
        with Path(source).open("rb") as incoming,snapshot.open("wb") as outgoing:
            while chunk:=incoming.read(1024*1024):
                digest.update(chunk);outgoing.write(chunk)
        parsed=reader(snapshot)
    return parsed,digest.hexdigest()


def import_calculix_results(
    path: str | Path, *, extra: Sequence[str | Path] = ()
) -> ImportedResults:
    """Read a CalculiX ``.frd`` result file, and any ``.dat`` beside it.

    The FRD carries displacements, reactions and stresses; the DAT carries
    buckling factors, frequencies and printed reaction totals.  Pass further
    files through ``extra`` to merge them, which is how a run split across
    steps comes back as one result.
    """

    from anyfileio import merge_results, parse_dat, parse_frd

    source = Path(path)
    if not source.exists():
        raise ResultImportError(f"no result file at {source}")

    def read(item: Path):
        if item.suffix.lower() == ".dat":
            return parse_dat(item)
        return parse_frd(item)

    parsed,source_hash = _parse_snapshot(source,read)
    snapshots = [_parse_snapshot(Path(item),read) for item in extra]
    others = [item[0] for item in snapshots]
    if others:
        parsed = merge_results(parsed, *others)
        import hashlib
        source_hash=hashlib.sha256("\n".join([source_hash]+[item[1] for item in snapshots]).encode("ascii")).hexdigest()

    if not parsed.has_results:
        raise ResultImportError(
            f"{source.name} parsed cleanly but carries no results. CalculiX "
            "writes an empty FRD when the step produced nothing to store; "
            "check the deck asked for output."
        )

    return ImportedResults(
        source=source,
        format="CalculiX",
        source_hash=source_hash,
        displacements={
            int(node): tuple(float(v) for v in value)
            for node, value in parsed.displacements.items()
        },
        node_stresses={
            int(node): _name_components(value)
            for node, value in parsed.stresses.items()
        },
        reactions={
            int(node): tuple(float(v) for v in value)
            for node, value in parsed.reaction_forces.items()
        },
        buckling_factors=tuple(float(v) for v in parsed.buckling_factors),
        frequencies=tuple(float(v) for v in parsed.frequencies_hz),
        warnings=tuple(str(item) for item in parsed.warnings),
    )


def _name_components(values: Sequence[float]) -> Dict[str, float]:
    """Give a positional stress tuple its component names.

    An FRD stress block is a symmetric tensor in a fixed order.  Anything
    longer than the names available keeps a positional name rather than being
    dropped, so a file with more per node still arrives intact.
    """

    named: Dict[str, float] = {}
    for index, value in enumerate(values):
        if index < len(_TENSOR_COMPONENTS):
            named[_TENSOR_COMPONENTS[index]] = float(value)
        else:
            named[f"s{index}"] = float(value)
    return named


# ----------------------------------------------------------------------
# SESAM
# ----------------------------------------------------------------------
def import_sesam_results(
    path: str | Path, *, load_case: Optional[int] = None
) -> ImportedResults:
    """Read RVSTRESS shell results from a SESAM SIF file.

    ``load_case`` of None takes the first case in the file, which is the
    solver's own choice and keeps element and nodal stresses from one
    consistent case rather than blending every case present.
    """

    from anyfileio import read_sesam_sif_stress

    source = Path(path)
    if not source.exists():
        raise ResultImportError(f"no result file at {source}")

    parsed,source_hash = _parse_snapshot(source,lambda snapshot:read_sesam_sif_stress(snapshot,load_case=load_case))
    names = tuple(str(name) for name in parsed.components)
    if not names:
        raise ResultImportError(
            f"{source.name} names no stress components. A SIF file without "
            "RVSTRESS records is a model, not a result -- import it with "
            "import_sesam() instead."
        )

    def named(values: Sequence[float]) -> Dict[str, float]:
        return {
            name: float(value) for name, value in zip(names, values)
        }

    results = ImportedResults(
        source=source,
        format="SESAM",
        source_hash=source_hash,
        node_stresses={
            int(node): named(values)
            for node, values in parsed.nodal_stress.items()
        },
        element_stresses={
            int(element): named(values)
            for element, values in parsed.element_stress.items()
        },
    )
    if not results.node_stresses and not results.element_stresses:
        raise ResultImportError(
            f"{source.name} yielded no shell stresses. Either it carries no "
            "RVSTRESS records -- in which case it is a model, not a result, "
            "and import_sesam() is what reads it -- or its stresses are for "
            "elements ANYfem does not read. Shell stresses only."
        )
    return results
