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
Automatic CI and ANYfem verification remain pending.
The old PR's runtime changes are already on main; its missing workflow and
historical implementation note are reconciled here. PR closure follows verified
delivery, with links to the replacement and preserved September evidence.
