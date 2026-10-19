"""Recovery and resource policy: what to keep, and what to spend keeping it.

Two knobs that do not change the answer, only what it costs to get and to hold:

* **recovery policy** -- which nodes, elements and stress components come back,
  and whether a transient keeps every step's history or only summaries.  On a
  large transient the histories dominate memory, and most of them are never
  looked at.
* **resource policy** -- thread counts, determinism and a memory ceiling.

Both are the solver's own configuration objects, wrapped only to give them
defaults and a docstring here rather than reproduced.  The wrapping is
deliberately thin: a policy this layer paraphrased would drift from the one the
solver actually honours.

One thing is worth stating because getting it wrong is silent.  Recovery's
``components`` filter drops every component it does not name.  ANYfem does not
supply a default list, and should not: the solver's recovery grows components
over time -- the orthotropic Hill utilisation is a recent example -- and a
whitelist frozen here would quietly discard them while everything still ran.
Name components only when you want exactly those.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

__all__ = ["history_modes", "recovery_policy", "resource_policy"]


def history_modes() -> tuple[str, ...]:
    """The history modes the installed solver accepts.

    Read from the solver rather than listed here.  A copy of this list in
    ANYfem was written first and was wrong within the hour -- it is exactly the
    kind of thing that goes stale silently, and a menu offering a mode the
    solver rejects is worse than no menu.
    """

    from anysolver import recovery

    modes = getattr(recovery, "_HISTORY_MODES", None)
    if not modes:
        return ("full",)
    return tuple(sorted(str(mode) for mode in modes))


def recovery_policy(
    *,
    node_ids: Optional[Sequence[int]] = None,
    element_ids: Optional[Sequence[int]] = None,
    components: Optional[Sequence[str]] = None,
    include_displacements: bool = True,
    include_stresses: bool = True,
    include_reactions: bool = True,
    history_mode: str = "full",
    store_full_histories: bool = True,
    **options: Any,
):
    """What a run should recover and keep.

    ``None`` means "everything available", which is the behaviour without a
    policy at all.  Narrowing to the nodes or elements actually being reported
    on is what makes a large model's postprocessing affordable.

    ``components`` filters stress dictionaries by key and **drops everything
    else**, including components this layer does not itself name.  Leave it
    ``None`` unless you want exactly the ones you list.
    """

    from anysolver import RecoveryConfig

    return RecoveryConfig(
        node_ids=node_ids,
        element_ids=element_ids,
        components=components,
        include_displacements=include_displacements,
        include_stresses=include_stresses,
        include_reactions=include_reactions,
        history_mode=history_mode,
        store_full_histories=store_full_histories,
        **options,
    )


def resource_policy(
    *,
    solver_threads: Optional[int] = None,
    assembly_threads: Optional[int] = None,
    recovery_threads: Optional[int] = None,
    process_workers: Optional[int] = None,
    deterministic: bool = True,
    memory_limit_bytes: Optional[int] = None,
    **options: Any,
):
    """Thread counts, determinism and a memory ceiling.

    ``deterministic`` defaults to True, matching the solver: a run that gives a
    different answer depending on how the work happened to be scheduled is not
    something an engineering report should have to caveat.

    The solver entry points used by ANYfem accept this policy. Pass it as
    ``resource_config`` to linear, modal, buckling and arc-length solves, or to
    the transient/impact configuration; nonlinear and capacity workflows use
    ANYfem's ``resources`` argument. Buckling applies one requested policy to
    both its prerequisite static solve and its eigensolve.

    Each field governs only its named work: ``solver_threads`` scopes supported
    BLAS/MKL pools, ``assembly_threads`` applies where parallel element assembly
    exists, and recovery workers and memory limits apply to the corresponding
    phases. Omitting the policy preserves the solver backend defaults.
    """

    from anysolver import ResourceConfig

    return ResourceConfig(
        solver_threads=solver_threads,
        assembly_threads=assembly_threads,
        recovery_threads=recovery_threads,
        process_workers=process_workers,
        deterministic=deterministic,
        memory_limit_bytes=memory_limit_bytes,
        **options,
    )
