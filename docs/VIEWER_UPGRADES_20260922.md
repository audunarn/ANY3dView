# Viewer upgrades — implementation and verification

Implemented in the working trees on 2026-09-22. This is development evidence,
not a release qualification or publication authorization. Package versions and
historical release ledgers were not changed.

## Changes

1. Geometry display requires the triangulation extra, validates trimmed area,
   and raises actionable errors instead of using an unsafe triangle fan.
   Updates build replacements before removing displayed chunks; failed updates
   preserve the old revision and can be explicitly retried. Initial attachment
   failures close their callbacks and worker resources.
2. ANYmesher's `gui3d` extra now requires `ANYtk3D>=0.5.5,<0.6`, matching the
   ANYfem/ANYstructure viewers. The ANYmesh change is limited to that constraint.
3. Chord/relative/angular controls drive bounded edge and surface refinement.
   Shared midpoint refinement preserves a conforming UV triangulation and holes.
   Work limits fail explicitly. These are sampled display checks, not certified
   kernel error bounds. The kernel remains the owner of geometric truth.
4. Both viewers have compatibility workflows: core Python 3.10–3.14 on three
   operating systems, geometry 0.4.0/current 0.4.x, installed wheels, a Linux
   desktop job, and consumer viewport tests. Action references are commit-pinned.
   Manual runs accept a sibling viewer `peer_ref`. Python 3.10 test tooling uses
   tomli where necessary. These workflows have not been run on GitHub yet.
5. View-state exchange preserves copied lighting and immutable selection policy.
   The Tk adapter tolerates the older core state shape. Pillow image export now
   rasterizes canvas items (including HUD, arrows and stipple), independently of
   desktop overlap. A laid-out viewport is required; fonts may differ from Tk.

The earcut range is now `>=1.0.2,<3`: upstream added NumPy 2 wheel compatibility
in 1.0.2 and Python 3.14 support in 2.0.0. Both 1.0.3 and 2.1.0 were tested.
Reference: https://github.com/skogler/mapbox_earcut_python/blob/main/CHANGELOG.md

## Verification

Source bases: ANY3dView `721b3be`, ANYtk3D `2caa923`, with working-tree changes.
Pre-existing GPU host/widget and contract-test edits were preserved. Verification
therefore applies to the combined working tree, not solely to this task's diff.

All recorded test commands exited successfully. Full suites were followed by
focused reruns covering subsequent edits; counts below are separate runs and
must not be added together as distinct tests.

| Check | Result |
|---|---|
| ANY3dView full Windows/Python 3.14 suite, GUI and GPU enabled | 188 passed |
| ANYtk3D full Windows/Python 3.14 suite, GUI enabled, native input excluded | 177 passed, 3 deselected |
| Final affected core/geometry/packaging contracts, Python 3.14 | 65 passed |
| Final Tk shared-state/export/workflow contracts | 31 passed |
| Geometry with installed ANYgeometry 0.4.0 and earcut 1.0.3, Python 3.13 | 33 passed |
| ANYfem and ANYstructure focused viewer contracts | 34 passed |
| ANYmesh affected packaging metadata checks | 3 passed |
| Workflow checks (including existing release authority unit tests) | 32 passed |
| Both licensing checks, workflow YAML parsing, diff whitespace checks | Passed |
| Wheel/sdist builds, twine checks, separately installed wheel pair | Passed |
| Hidden-window export from installed Tk wheel | Passed; PNG visually inspected |

The new geometry regressions use polygon areas (7 for the concave example, 12
for a square with a hole), analytic circle sagitta, and an analytic paraboloid.
They also cover missing dependencies, failed update preservation/retry, invalid
limits, work-budget exhaustion and large coordinate translations. Export tests
cover painter order, hidden items, stipple, arrows, desktop independence and
both directions of renderer-state exchange.

## Local artifacts

Logs are under `%TEMP%`:

- `viewer-implementation-core-full.log`
- `viewer-implementation-tk-full.log`
- `viewer-implementation-final-core-focus.log`
- `viewer-implementation-final-tk-focus.log`
- `viewer-implementation-cp313-geometry.log`
- `viewer-implementation-consumers.log`
- `viewer-implementation-mesh-metadata.log`

Built distributions, isolated installed environment, build log and inspected PNG:
`%TEMP%/viewer-wheel-review-kfdst6xo/`.

Some earlier default pytest runs emitted a permission warning while cleaning an
unrelated shared temporary symlink. Full suite runs used dedicated temporary
directories. No shared directory or another task's files were removed.

## Next gate

Run the new GitHub matrix with the coordinated viewer and ANYmesh changes, then
obtain review and the normal release qualification. Native desktop-input tests,
non-Windows local execution, publication and independent acceptance were not
performed by this task.
