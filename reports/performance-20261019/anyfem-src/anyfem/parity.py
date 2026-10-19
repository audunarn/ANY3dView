"""The ANYstructure parity ledger, and the migration gate.

ANYfem is meant to replace ANYstructure's ``fem_integration.py`` eventually,
but only once it is complete and verified.  Deciding when that is would be a
matter of opinion unless someone writes down what "complete" means, so this
module writes it down.

The ledger is deliberately unflattering.  ``fem_integration`` exposes 176
options; the entries below are grouped by capability and marked against what
ANYfem actually does today.  A capability is ``covered`` only when ANYfem can
do the same job, ``partial`` when it can do some of it, and ``missing``
otherwise.  Marking something covered because it is nearly covered would
defeat the point of keeping a ledger at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

__all__ = [
    "GATE_CRITERIA",
    "LEDGER",
    "ParityEntry",
    "gate_status",
    "parity_markdown",
    "parity_summary",
    "write_parity_report",
]

STATUSES = ("covered", "partial", "missing")

# The reference the ledger is written against.
SOURCE = "ANYstructure anystruct/fem_integration.py (RuntimeFEMOptions, 176 options)"


@dataclass(frozen=True)
class ParityEntry:
    """One capability of the GUI ANYfem intends to replace."""

    area: str
    capability: str
    status: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(
                f"{self.capability!r}: status must be one of {', '.join(STATUSES)}"
            )


LEDGER: Tuple[ParityEntry, ...] = (
    # -- modelling -----------------------------------------------------
    ParityEntry(
        "Modelling", "Parametric stiffened panel and cylinder geometry",
        "missing",
        "ANYfem models general geometry instead. The parametric front end is "
        "what ANYstructure would keep and call into, so this is a seam to "
        "build, not a gap to close.",
    ),
    ParityEntry(
        "Modelling", "General geometry (points, lines, arcs, plates, sweeps)",
        "covered", "Beyond what fem_integration can express at all.",
    ),
    ParityEntry(
        "Modelling", "Plate thickness regions", "covered",
        "A plate section per face; split a plate to vary it.",
    ),
    ParityEntry(
        "Modelling", "Stiffener and girder eccentricity", "covered",
        "A beam section carries an eccentricity; the mesher offsets its nodes "
        "along the plate normal and couples every station back to the plating "
        "with the solver's own MPC. Verified against a transformed section: "
        "ECC-01 puts the neutral axis within 0.2% and ECC-02 the stiffness "
        "within 1.5%.",
    ),
    ParityEntry(
        "Modelling", "Symmetry modelling", "covered",
        "project.add_symmetry() restrains the normal translation and the two "
        "in-plane rotations, with antisymmetry as the complement, and checks "
        "the entity actually lies in the plane before adding anything. A "
        "quarter plate matches the full model to nine figures. Planes must be "
        "normal to a global axis: the solver applies boundary conditions in "
        "global axes with no nodal transformation, so a tilted plane can only "
        "be approximated, and is refused instead.",
    ),
    # -- meshing -------------------------------------------------------
    ParityEntry(
        "Meshing", "Target element size and mesh fidelity", "covered",
        "Target size plus per-line division pinning.",
    ),
    ParityEntry(
        "Meshing", "Conformal mesh across shared boundaries", "covered",
        "Structural rather than tolerance based.",
    ),
    ParityEntry(
        "Meshing", "Local and point mesh refinement patches", "covered",
        "A refinement zone binds to a point, line or plate; seeding integrates "
        "the resulting size field along each edge and node placement follows "
        "the same field. MESH-02 checks the size asked for is the size "
        "produced. One limit is inherent to mapped meshing rather than to this "
        "implementation: a Coons interior is the blend of its boundary, so a "
        "zone in the middle of a plate refines nothing until the plate is "
        "decomposed. RefineForImpact does that decomposition.",
    ),
    ParityEntry(
        "Meshing", "8-node shells and 3-node beams (S8, B3)", "covered",
        "Element order is a project setting. Q8 is the accuracy win: ELEM-01 "
        "reaches 1% on 16 elements where Q4 needs 256 for the same tolerance. "
        "B3 is there for compatibility, not accuracy -- ANYsolver's 2-node "
        "Timoshenko beam is already exact for a tip-loaded cantilever, which "
        "B3 is not, so 3-node beams exist so a stiffener can share the "
        "mid-side nodes of a Q8 shell edge. Q8R stays out, being experimental "
        "in the solver, and a quadratic beam on a curved line is refused "
        "because the solver's B3 is straight-sided.",
    ),
    # -- loads and boundary conditions ---------------------------------
    ParityEntry(
        "Loads and BC", "Pressure, dead and follower", "covered", "",
    ),
    ParityEntry(
        "Loads and BC", "Edge and line loads with components", "covered", "",
    ),
    ParityEntry(
        "Loads and BC", "Point loads and moments", "covered", "",
    ),
    ParityEntry(
        "Loads and BC", "Surface traction", "covered",
        "Not in fem_integration; ANYfem adds it.",
    ),
    ParityEntry(
        "Loads and BC", "Acceleration fields and added mass", "covered", "",
    ),
    ParityEntry(
        "Loads and BC", "Enforced displacement", "covered", "",
    ),
    ParityEntry(
        "Loads and BC", "Supports, including per-DOF and per-edge", "covered", "",
    ),
    ParityEntry(
        "Loads and BC", "Load cases and factored combinations", "covered",
        "Not in fem_integration, which solves one case at a time.",
    ),
    ParityEntry(
        "Loads and BC", "Axial force, moment and shear resultants on a section",
        "missing",
        "Deliberately out of scope rather than outstanding. It is a shorthand "
        "that only means something on a parametric panel, where the section to "
        "resolve it onto is implied by the geometry; on general geometry the "
        "same request needs a cut plane and a distribution rule that would be "
        "invented here. The parametric front end applies its own resultants "
        "when it calls in. Recorded as missing, and excluded from the gate by "
        "name in OUT_OF_SCOPE.",
    ),
    # -- materials -----------------------------------------------------
    ParityEntry(
        "Materials", "DNV-RP-C208 steel grades and thickness classes",
        "covered", "Through ANYmaterial's canonical table.",
    ),
    ParityEntry(
        "Materials", "Nonlinear material with a hardening curve", "covered",
        "steel(..., nonlinear=True) attaches ANYmaterial's RP-C208 curve, and "
        "ANYsolver integrates it in the nonlinear solve. Plasticity is the layered-shell path: "
        "beams stay elastic, which is the solver's scope, not a wrapper "
        "limitation.",
    ),
    # -- analyses ------------------------------------------------------
    ParityEntry("Analyses", "Linear static", "covered", ""),
    ParityEntry("Analyses", "Modal", "covered", ""),
    ParityEntry(
        "Analyses", "Linear eigenvalue buckling", "covered",
        "Including shift, range and repeated-mode controls by passthrough.",
    ),
    ParityEntry("Analyses", "Nonlinear static", "covered", ""),
    ParityEntry(
        "Analyses", "Arc length and post-buckling", "covered", "",
    ),
    ParityEntry("Analyses", "Transient dynamics", "covered", ""),
    ParityEntry(
        "Analyses", "Rigid-sphere collision and impact", "covered",
        "Contact with an automatic penalty and time step, contact history, "
        "momentum balance, energy, damage and erosion, plus adaptive "
        "refinement at the contact point through RefineForImpact. Every result "
        "reports how many elements lie across the sphere radius, because a "
        "contact patch spread over one element gives a peak force that belongs "
        "to the mesh rather than the structure. The time step is bounded by the "
        "wave transit time of the smallest element at the contact as well as by "
        "the contact period, so a refined mesh converges at the penalty the "
        "solver recommends.",
    ),
    ParityEntry(
        "Analyses", "Fracture and element erosion", "covered",
        "Available in an impact and in a nonlinear static solve, through the "
        "solver's own damage and fracture configurations. Eroded element IDs "
        "come back on the result. Erosion is residual stiffness scaling after "
        "a converged increment -- an engineering screen, not crack mechanics, "
        "which is how the solver describes it too.",
    ),
    ParityEntry(
        "Analyses", "Capacity workflow (static to buckling to imperfect nonlinear)",
        "covered",
        "solve_capacity() drives the solver's own packaged workflow rather "
        "than re-chaining the stages here, so the prestress recovery between "
        "the static and buckling solves and the mesh-adequacy check on the "
        "chosen mode stay the solver's. The result is a nonlinear solution "
        "carrying the buckling stage alongside it, so it displays unchanged, "
        "and reports the elastic critical factor and the achieved capacity "
        "separately -- their ratio is the point of running it.",
    ),
    ParityEntry(
        "Analyses", "Geometric imperfections, standard and eigenmode",
        "covered", "",
    ),
    # -- results -------------------------------------------------------
    ParityEntry(
        "Results", "Displacement and stress contouring", "covered",
        "28 fields including derived surface stress.",
    ),
    ParityEntry("Results", "Probes and node/element readout", "covered", ""),
    ParityEntry(
        "Results", "Result along a path", "covered",
        "Not in fem_integration; ANYfem adds it.",
    ),
    ParityEntry(
        "Results", "Envelopes over steps or modes", "covered",
        "Not in fem_integration.",
    ),
    ParityEntry("Results", "Animation of steps and modes", "covered", ""),
    ParityEntry(
        "Results", "Time-history plots at a probe", "covered",
        "history_series() reduces a transient, an impact and an incremental "
        "solve to the same Series type -- two arrays with labels and units -- "
        "and a Tk canvas widget draws it with round axis ticks, a peak marker "
        "and a hover readout. Hand-written rather than matplotlib, so the "
        "GUI's dependency set stays Tk, matching how ANYtk3D draws its own "
        "viewport. The axis arithmetic is separate from the drawing and is "
        "tested without a display.",
    ),
    ParityEntry(
        "Results", "Stress reduction to panel buckling input", "missing",
        "fe_plate_fields reduces FE stresses into buckling-code input. That "
        "is ANYstructure domain logic and stays there.",
    ),
    ParityEntry(
        "Results", "Recovery policy controls (history mode, threads, memory)",
        "covered",
        "recovery_policy() and resource_policy() build the solver's own "
        "configuration objects. ANYfem deliberately supplies no default "
        "component list: recovery's filter drops everything it does not name, "
        "and a whitelist frozen here would silently discard components the "
        "solver later adds -- the orthotropic Hill utilisation being the "
        "example that already happened. History modes are read from the "
        "solver rather than copied. ResourceConfig reaches every solver family "
        "ANYfem drives: direct solver_options for linear, modal, buckling and "
        "arc-length, transient configuration for transient and impact, and the "
        "resources argument for nonlinear and capacity workflows. Each thread "
        "or memory field remains scoped to the solver phase that owns it.",
    ),
    # -- application ---------------------------------------------------
    ParityEntry("Application", "3D viewport with picking", "covered", ""),
    ParityEntry(
        "Application", "Loads and supports drawn on the model", "covered", "",
    ),
    ParityEntry(
        "Application", "Threaded solve with progress and cancel", "covered", "",
    ),
    ParityEntry(
        "Application", "Undo and redo", "covered",
        "Not in fem_integration.",
    ),
    ParityEntry(
        "Application", "Save and restore the analysis state", "covered",
        "ANYfem has its own project format, and anyfem.migration reads "
        "fem_integration's save_runtime_fem_state files as well -- plain or "
        "gzipped JSON, without importing ANYstructure. Of the 176 options, 144 "
        "map onto ANYfem settings, 24 are out of scope by decision (section "
        "resultants, parametric-panel construction, and the JSON blobs that "
        "name panel segments and patches) and 8 are solver internals ANYfem "
        "does not surface; the file reports which is which rather than "
        "silently dropping any. It restores settings and recorded results, not "
        "the model: the snapshot describes a parametric panel, and the stored "
        "visualisation is a plotting grid rather than a mesh, so there is no "
        "topology in the file to rebuild from.",
    ),
    ParityEntry(
        "Application", "Report export", "covered",
        "Markdown and CSV.",
    ),
    # -- interop -------------------------------------------------------
    ParityEntry(
        "Interop", "SESAM FEM import", "covered",
        "ANYfileio resolves the file to neutral mesh and material records; "
        "ANYfem maps those public records into its project model.",
    ),
    ParityEntry(
        "Interop", "SESAM SIF result import", "covered",
        "import_sesam_results() reads RVSTRESS shell stresses through "
        "ANYfileio, keeping the component names the file gives "
        "rather than mapping them onto a list here. A SIF with no shell "
        "stresses is refused with the reason, not attached empty.",
    ),
    ParityEntry(
        "Interop", "CalculiX deck export", "covered",
        "ANYfem flattens the solved model and ANYfileio owns deck serialization.",
    ),
    ParityEntry(
        "Interop", "CalculiX FRD/INP result import", "covered",
        "import_calculix_results() reads FRD and DAT through ANYfileio and "
        "attaches them by node ID, so a model exported as a deck "
        "can be solved elsewhere and read back into the same contours, probes "
        "and reports. An FRD carries three translation components and no "
        "rotations: asking for one raises rather than returning zero, and the "
        "raw array holds NaN there so nothing that indexes it directly can "
        "mistake the gap for an answer. A file for a different mesh is "
        "refused with the overlap rather than attached partially.",
    ),
    ParityEntry(
        "Interop", "Handoff to the ANYstructure buckling session", "missing",
        "The migration itself, and the last thing to build.",
    ),
)


# ----------------------------------------------------------------------
# the gate
# ----------------------------------------------------------------------
GATE_CRITERIA: Tuple[str, ...] = (
    "every fem_integration analysis path reproduced within tolerance on a "
    "fixed set of ANYstructure models",
    "parity ledger has no missing or partial entries outside ANYstructure's "
    "own domain logic",
    "existing save_runtime_fem_state files importable",
    "headless API builds every ANYstructure model type without a GUI",
    "no performance regression on representative models",
)

# Entries that belong to ANYstructure rather than to ANYfem, and so do not
# block the migration.  Naming them explicitly keeps the gate honest: they are
# excluded on purpose, not by quietly leaving them out of the ledger.
OUT_OF_SCOPE: Tuple[str, ...] = (
    "Stress reduction to panel buckling input",
    "Handoff to the ANYstructure buckling session",
    "Parametric stiffened panel and cylinder geometry",
    # A parametric-panel shorthand, not a capability ANYfem is short of.
    # fem_integration applies section resultants because its geometry *is* a
    # panel with an obvious cut, so "the axial force on this section" has one
    # meaning. ANYfem applies loads to geometry directly, where the same phrase
    # would need a cut plane and a distribution rule invented to go with it.
    # It belongs with the parametric front end, which can apply resultants to
    # its own panels when it calls in.
    "Axial force, moment and shear resultants on a section",
)


def parity_summary() -> Dict[str, Any]:
    """Counts by status, and by area."""

    counts = {status: 0 for status in STATUSES}
    by_area: Dict[str, Dict[str, int]] = {}
    for entry in LEDGER:
        counts[entry.status] += 1
        area = by_area.setdefault(entry.area, {status: 0 for status in STATUSES})
        area[entry.status] += 1
    return {
        "source": SOURCE,
        "total": len(LEDGER),
        "counts": counts,
        "by_area": by_area,
        "coverage": counts["covered"] / len(LEDGER) if LEDGER else 0.0,
    }


def gate_status() -> Dict[str, Any]:
    """Whether the *ledger* says the migration can proceed, and what blocks it.

    This answers one of the five gate criteria and nothing else.  ``ready`` is
    therefore always False here: a clear ledger is necessary and not
    sufficient, and reporting otherwise from this module would be the kind of
    partial answer that gets quoted as a whole one.
    :func:`anyfem.migration.gate_report` measures all five.
    """

    blocking = [
        entry
        for entry in LEDGER
        if entry.status in ("missing", "partial")
        and entry.capability not in OUT_OF_SCOPE
    ]
    reason = (
        "The ledger is clear. That is one of five gate criteria; run "
        "anyfem.migration.gate_report() for the others, which need recorded "
        "ANYstructure results to compare against."
        if not blocking
        else (
            f"The ledger has {len(blocking)} open entry(ies): "
            + ", ".join(entry.capability for entry in blocking)
            + "."
        )
    )
    return {
        "ledger_clear": not blocking,
        "blocking": [
            {
                "area": entry.area,
                "capability": entry.capability,
                "status": entry.status,
                "note": entry.note,
            }
            for entry in blocking
        ],
        "out_of_scope": list(OUT_OF_SCOPE),
        "criteria": list(GATE_CRITERIA),
        "ready": False,
        "reason": reason,
    }


def parity_markdown() -> str:
    """The ledger as a report."""

    summary = parity_summary()
    gate = gate_status()
    counts = summary["counts"]

    lines = [
        "# ANYstructure parity ledger",
        "",
        f"Measured against {SOURCE}.",
        "",
        f"- capabilities tracked: {summary['total']}",
        f"- covered: {counts['covered']}",
        f"- partial: {counts['partial']}",
        f"- missing: {counts['missing']}",
        f"- coverage: {summary['coverage']:.0%}",
        "",
        "## Migration gate",
        "",
        f"**Ready: {'yes' if gate['ready'] else 'no'}.** {gate['reason']}",
        "",
        "Criteria:",
        "",
    ]
    lines += [f"{index}. {text}" for index, text in enumerate(GATE_CRITERIA, 1)]
    lines += [
        "",
        f"{len(gate['blocking'])} ledger entries are open (excluding "
        f"{len(OUT_OF_SCOPE)} that belong to ANYstructure rather than ANYfem).",
        "",
        "## Ledger",
        "",
    ]

    current_area = None
    for entry in LEDGER:
        if entry.area != current_area:
            current_area = entry.area
            lines += ["", f"### {current_area}", "", "| capability | status | note |", "| --- | --- | --- |"]
        mark = {"covered": "covered", "partial": "partial", "missing": "missing"}[
            entry.status
        ]
        lines.append(f"| {entry.capability} | {mark} | {entry.note} |")

    lines += [
        "",
        "---",
        "",
        "A capability is marked covered only when ANYfem can do the same job. "
        "Marking something covered because it is nearly covered would defeat "
        "the purpose of keeping this ledger.",
        "",
    ]
    return "\n".join(lines)


def write_parity_report(directory: str | Path = "reports/parity") -> Dict[str, Path]:
    """Write the ledger as JSON and Markdown."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "parity.json"
    markdown_path = target / "parity.md"
    json_path.write_text(
        json.dumps(
            {
                "summary": parity_summary(),
                "gate": gate_status(),
                "entries": [
                    {
                        "area": entry.area,
                        "capability": entry.capability,
                        "status": entry.status,
                        "note": entry.note,
                    }
                    for entry in LEDGER
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(parity_markdown(), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    """Print the parity ledger and write its report."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Report ANYstructure parity and the migration gate."
    )
    parser.add_argument("--out", default="reports/parity")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    print(parity_markdown())
    if not args.no_save:
        written = write_parity_report(args.out)
        print(f"written: {written['markdown']}")
    # The gate is information, not a failure: an open ledger is the expected
    # state until the migration is actually ready.
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
