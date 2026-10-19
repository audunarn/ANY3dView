# Viewer compatibility gate — 2026-09-22

Status: both viewer compatibility matrices passed (30/30 jobs each).
Coordinated merge readiness is blocked by the broader ANYmesh CI failure
described below. No release acceptance is claimed.

## Review candidates

| Repository | Candidate | Draft review |
| --- | --- | --- |
| ANY3dView | `82b6aeed9a1ca2dd2c8260ad40284916e4e67d5d` | https://github.com/audunarn/ANY3dView/pull/3 |
| ANYtk3D | `7e2456ec8bad38c5168b1812dd399d5319953a98` | https://github.com/audunarn/ANYtk3D/pull/3 |
| ANYmesh | `4d00022262912c4512ceb5447fff31bb32d470dc` | https://github.com/audunarn/ANYmesh/pull/7 |

Candidates were prepared in isolated clones. Original workspaces remain on their
original branches with existing dirty work preserved. The core candidate includes
host-adapter prerequisite commit `721b3be5c2adb2b42b9df19450337c756240455e`.
Unrelated local GPU batching and Windows parent-mapping changes are excluded.

## Gate corrections

- Enforced the triangle budget for legacy unparameterized planar faces.
- Added the Linux OpenGL development library needed by the desktop runner.
- Resolved immutable peer source identities once before each matrix; consumer
  lookup follows each repository's default branch, including ANYstructure/master.
- Realized the Linux TkGL native drawable before current-context detection,
  without mapping the viewer or entering a nested event loop.
- Read the explicit current viewport during GPU capture; regression coverage
  now exercises capture after both growing and shrinking the window.

The TkGL diagnosis was checked against upstream native source:
https://github.com/3-manifolds/TkGL/blob/main/unix/tkglGLX.c

## Verification

- Both final viewer matrices: **60 successful jobs**, including Python
  3.10–3.14 on Windows/Linux/macOS, wheel builds/imports, minimum/current geometry,
  paired Linux desktop and consumer integration.

- Final paired Linux desktop job: **190 core tests passed**, **178 software-viewer
  tests passed**, 3 native pointer-input tests deselected.
- Windows GPU lifecycle/capture regression after the last runtime change:
  **9 passed**.
- Isolated Windows software-viewer suite: **178 passed**, 3 native-input tests
  deselected. The final Tk candidate changes after this result affect CI only.
- Focused geometry tests after the planar budget correction: **34 passed**.
- Existing licensing scripts: exit 0 for both viewer candidates.
- Original isolated core run: 182 passed and 7 native loader failures caused by
  sandbox file visibility. All 9 GPU tests passed with normal desktop access;
  the later complete Linux run passed all 190 tests. The failed run is retained.

CI runs:

- Core: https://github.com/audunarn/ANY3dView/actions/runs/35729422349
- Software viewer (attempt 2): https://github.com/audunarn/ANYtk3D/actions/runs/35728333917/attempts/2
- Mesh: https://github.com/audunarn/ANYmesh/actions/runs/35727031691

Both viewer matrices resolve the same peer and mesh candidate identities, plus:

- ANYfem `18b4d3014f4f515d96ac5b027f51bbe92b57ba2b`
- ANYstructure `4666c462c781af6e0c10d56cc3b7ac8027dce999`

## ANYmesh CI blocker

Ubuntu/Python 3.11 job `106743081817` reports **21 failed, 1479 passed,
54 skipped**. All 21 failures are in the existing quadrilateral public integration
and qualification tests with the native `quad_mcf_worker` executable absent.
The CI install uses `[dev,planar]`, not the changed `[gui3d]` extra; this candidate
changes only the GUI dependency bound. The paired viewer consumer job installs
`[gui3d]` successfully and passes its viewport contracts.

This blocker was recorded without weakening the mesh tests or changing native
worker qualification. The final mesh snapshot has 9 successful, 3 failed and 10 running jobs; only
the detailed Ubuntu/Python 3.11 failure above was diagnosed in this scope.
Full failure log: `%TEMP%/viewer-gate-mesh-106743081817.log`.

## Evidence and limits

Full logs, immutable-source records, candidate clones, previous failed attempts
and `gate-manifest.json` are in:
`C:/Users/AudunArnesenNyhus/AppData/Local/Temp/viewer-gate-20260922-su3by8i5/`.

Additional failure logs under the same user's Temp directory:
`viewer-gate-job-106743060005.log`, `viewer-gate-resolve-106745833110.log`,
`viewer-gate-desktop-106747479172.log`, and
`viewer-gate-desktop-106749610250.log`.

This is self-review and ordinary compatibility regression evidence, not
independent acceptance or a release qualification. Native pointer-input
qualification was not run. No version bump, merge or publication was performed.
The earlier implementation report describes its own combined-working-tree
checks and must not be substituted for these isolated candidate results.

Next gate: independent review of the coordinated draft PRs and resolution of
the ANYmesh native-worker CI prerequisite in its own workstream. Follow with the
registered release qualification for an approved release candidate.
