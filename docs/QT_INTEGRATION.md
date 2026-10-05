# Qt integration — 2026-10-02

## Objective and authority

Integrate the outstanding playback and performance work, add automatic Qt CI,
verify the combined viewer in ANYfem, and reconcile the superseded viewer PR.
The user authorized carrying this sequence through delivery. Existing release
authority is not reused; this is compatibility integration, not a package release.
Policy transition: apply ecosystem policy revision 2026-09-24.2 prospectively;
retain the September gate as historical evidence with its original identities.

## Candidate and ownership

Base: ANY3dView `9d27c1a6ca1b52691bd36791de1131cc8b1d9c0b`.
Integrated inputs: performance `d05093a` and playback `d65f47b`.
ANYfem baseline: `784cf2c8ba294d55ff8984fec0fb08869eb4e966`.
Work is isolated in `codex/qt-integration-20261002` and the corresponding ANYfem
`codex/viewer-qt-integration-20261002` worktree. Original dirty work and the
separate Tk capture PR remain owned by their existing tasks.

## Decision and verification

Question: does combining the two changes preserve Qt scene/selection semantics
and fix playback without losing the tested ANYfem integration?
Competing explanations are a straightforward compatible combination versus
software/GPU routing or host-lifecycle regressions. Check the focused contracts,
installed Qt lifecycle on Windows and Linux, then the existing ANYfem Qt profile.
Resolve concrete failures before integration; retain failed evidence. Reuse
historical scientific evidence only for unchanged owner implementations.

The assessment found 3 playback regressions on the pre-integration local viewer;
the pinned playback branch passes all 3. The local performance delta passes 11
focused checks. Those are input evidence, not acceptance of the combined artifact.

Automatic viewer CI retains Python 3.10–3.14 and geometry version/platform
coverage, and adds installed Qt checks. ANYfem owns its application-level Qt
profile and will test the immutable combined viewer commit. Runtime and packaging
checks do not authorize new scientific scope or a published package version.

## Current state

The two input commits apply cleanly. Independent Mistral review found unsafe
identity memoization of normalized tuple subclasses in the performance input.
The counterexample returned the wrong owner in 48/50 rows; the memo now retains
its normalized tuple and a regression covers this case. Through-depth selection
deliberately keeps the full stack; the front-only optimization is documented as
visible-depth only. The optional CPU outline remains bounded at 2,500 primitives;
retained selection masks continue to provide highlighting.

The combined native Windows Qt, ownership and contract checks pass 56 tests,
including actual GPU reparenting, resize/capture, state transfer and playback.
The new camera round-trip assertion accounts for the existing normalized orbit
representation (1e-12), not object identity. The first independent pass reached
its turn limit; its focused continuation produced the findings above. The initial
headless suite passed 179 tests but rejected a new artifact-action revision;
the workflow now uses the repository's already approved immutable action pin.

Logs are in `%TEMP%/any3dview-qt-integration-20261002-*.log`, including retained
counterexample, intermediate test failures and independent review tool outcomes.
The focused independent follow-up confirmed the memo correction. Its suspected
Windows wheel-glob blocker is not present: the wheel-install block already uses
`shell: bash`. Twine also explicitly uses Bash for consistent glob expansion.
The approved-action pin check and licensing check pass after the CI correction.
## Integrated evidence and delivery

The runtime candidate is `c69aa8311e61b16f54895802fe8935cc2fe980ad`.
All 32 jobs in [viewer compatibility CI](https://github.com/audunarn/ANY3dView/actions/runs/36987880953)
passed. The Linux desktop suite passed 205 tests (one Qt opt-in module skipped),
the two Linux installed Qt jobs passed all 15 tests each, and Windows software
jobs passed 6 each with 9 intentionally GPU-only cases skipped. Those same 15
Qt tests also passed locally against an installed wheel on native Windows GPU;
the import origin was the disposable environment's `site-packages`.

ANYfem `e2748d423a5b57606aab54877579ac6084a99f71` pins this exact viewer in its
GUI extra, license inventory, README and both CI workflows. All four jobs in
[ANYfem Qt CI](https://github.com/audunarn/ANYfem/actions/runs/36988077070) passed.
Each of the six application profile runs (Windows/Linux software, Linux GPU,
Python 3.11/3.14) passed 170 tests with zero failures, errors or skips. Every job
also passed 22 launcher contracts and 42 feature/persistence contracts; both
Linux jobs passed the 15 viewer GPU tests. Solver/material/geometry/mesh pins,
scientific predicates and tolerances remain unchanged.

The [viewer integration PR](https://github.com/audunarn/ANY3dView/pull/4) and
[ANYfem alignment PR](https://github.com/audunarn/ANYfem/pull/11) carry final merge
status and the full ANYfem regression result. Merge requires all existing checks
to pass. A merge commit preserves the immutable viewer pin's reachability.
This document's later evidence-only update does not change the tested runtime.

The old PR's runtime changes are already on main; its missing workflow and
historical implementation note are reconciled here. PR closure follows verified
delivery, with links to the replacement and preserved September evidence.
ANYtk3D #3 and ANYmesh #7 remain separate. Dirty geometry-adapter work and ANYfem
startup/selection work in the original checkouts are preserved for their owners.
This integration establishes the stated software compatibility, not a package
release or broader physical-GPU/scientific qualification.

## False diagonal display correction — 2026-10-05

ANYfem reported a selected quad showing its fill-triangulation diagonal. Scope is
rendering only: preserve geometry/FE topology, scientific behavior, picking and
per-triangle results. The fix is isolated from unrelated dirty adapter work on
base `2fba4dc`. Software fills have no triangle pens or per-triangle antialias
seams. Source polygon grouping, or explicit MeshArrays triangle-to-element
grouping, removes internal indexed edges; ungrouped real triangles retain theirs.
Selection/preselection HUD boundaries use the same indexed topology, with no
coordinate or shared-tag topology inference. Retain holes, clipping, masks,
chunks/deformation, approximate software painter depth ordering and animation.
Private grouping is invalidated when its indexed topology changes.

Focused viewer/retained/selection/array checks plus six ANYfem rendered regressions:
45 pass, 9 explicitly GPU-opt-in skipped, 3.72 seconds. Failed seam and depth
checks remain under `reports/qt/false-diagonal/`. ANYfem independently reproduces
four old false-diagonal cases and checks real FE triangles as positive controls.
OpenAI implementation and parent review are used after the earlier Mistral Edit
Git-boundary violation; external Mistral READ review requires separate source
permission, currently pending. This fix does not claim physical GPU acceptance
or authorize package release.
