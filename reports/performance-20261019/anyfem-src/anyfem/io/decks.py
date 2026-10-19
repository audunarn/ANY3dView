"""Writing models out for other solvers.

CalculiX deck generation uses ANYsolver's FEModel-to-neutral-deck adapter and
ANYfileio's canonical writer. The result is therefore what the solver would
hand to CalculiX for its verification runs, not a second divergent translation.

SESAM export is deliberately *not* offered. Semantic export from an arbitrary
FEModel is outside the supported gate; writing one
anyway would produce a file that looks authoritative and is not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

__all__ = ["DeckExportError", "export_calculix_deck", "export_sesam"]


class DeckExportError(ValueError):
    """Raised when a model cannot be written in the requested format."""


def export_calculix_deck(
    built,
    path: str | Path,
    *,
    analysis: str = "static",
    metadata: Optional[Mapping[str, Any]] = None,
):
    """Write a CalculiX input deck for a built model.

    A generated deck is a reproducibility handoff, not evidence: until it has
    actually been run and its results compared, it says nothing about
    agreement.  ANYsolver labels unexecuted decks ``not_executed`` for exactly
    that reason, and this wrapper does not dress it up as more.
    """

    from anysolver import write_calculix_input_deck

    destination = Path(path)
    if not destination.suffix:
        destination = destination.with_suffix(".inp")
    destination.parent.mkdir(parents=True, exist_ok=True)

    details = {
        "source": "ANYfem",
        "model": built.project.name,
    }
    if built.combination:
        details["load_combination"] = built.combination
    elif built.load_case is not None:
        details["load_case"] = built.load_case.name
    if metadata:
        details.update(dict(metadata))

    try:
        return write_calculix_input_deck(
            built.fe_model,
            built.load_case,
            destination,
            analysis=analysis,
            metadata=details,
        )
    except Exception as error:  # noqa: BLE001 - reported verbatim
        raise DeckExportError(f"could not write {destination.name}: {error}") from None


def export_sesam(*_args: Any, **_kwargs: Any):
    """Refuse SESAM export, and say why.

    ANYfileio supports guarded round-tripping of a SESAM *document* it parsed,
    but not semantic export from an arbitrary solver model. A file written outside
    that gate would look like an interchange file and would not be one, so
    ANYfem does not write it at all rather than writing something plausible.
    """

    raise DeckExportError(
        "ANYfem does not export SESAM. ANYfileio supports guarded round-trip "
        "of a SESAM document it parsed, but semantic export from an arbitrary "
        "model is outside its supported gate, and a file written anyway would "
        "look authoritative without being so. Export a CalculiX deck, or save "
        "an ANYfem project file."
    )
