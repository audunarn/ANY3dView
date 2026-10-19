# Early checkpoint — five performance improvements, 2026-10-19

Owner: Mistral worker. Scope: src/any3dview/qt.py, retained_viewer.py, scheduler.py + focused tests.
Baseline: reports/performance-20261019/baseline-src + preexisting.patch (not touched).
No renderer.py changes; no other repos; no git mutations; no native windows/GPU runs.

## Inspection findings (current dirty source, HEAD cba89a7)

1. GPU hover: `_InputSurface.mouseMoveEvent` calls `owner._hover_select(e)` synchronously
   for every unpressed GPU move. Software already coalesces via `_queue_hover` (120 ms
   single-shot QTimer, latest-event-wins, `_cancel_hover` on press/wheel/drag/leave/hide/focus).
2. Software paint: `QtViewer._paint_software` recomputes per camera frame, per (handle, chunk):
   world = (positions + scale*displacements) @ transform, mask sets, scalar limits,
   per-triangle admission + color, per-line section clipping, point admission, and
   `_surface_outline_loops` boundary extraction. Only projection/depth-sort/submission is
   camera-dependent. `_PaintSurface` already caches the finished QImage (exposure only).
3. Animation: `RetainedViewer._show_animation_frame` removes all renderer groups and builds
   fresh MeshHandles + `renderer.add_mesh` per frame. GPU renderer `_sync` already supports
   in-place generation-based updates (displacement/scalar/active/selection/appearance/transform)
   for an existing group; topology/position generation changes trigger its own rebuild.
4. Selection: `_selection_key` includes full `handle.generations` (scalar+appearance included),
   so scalar-only changes rebuild `ProjectedSelectionIndex` even though `_mesh_changed` already
   avoids clearing it for scalar-only changes. Index consumes geometry/visibility/ownership only.
5. Scheduling: Qt host polls `ViewerScheduler.drain()` from a persistent 2 ms QTimer
   (`_start_update_pump`). Scheduler is a SimpleQueue with owner-thread drain, 1024 bound,
   close() clearing; no notification hook. Tk path uses `after(2)` rescheduling (unchanged).

## Concrete choices

1. GPU hover: route unpressed GPU moves through the existing `_queue_hover` machinery with a
   0 ms interval (one latest hover per event-loop pass); software keeps 120 ms. Exact clicks
   unchanged (press path already cancels hover first); stale cancel paths unchanged.
2. Software retained preparation: per-(handle, chunk) prep cache in qt.py keyed by a signature
   (handle.generations, per-handle/chunk mask revision, section plane key, show_mesh_lines,
   highlight fill, scalar style, face_colors, outline groups). Caches camera-invariant data:
   world positions, admitted triangles + colors, section-clipped triangle polygons, clipped
   world line segments, admitted points, outline loops. Projection, depth ordering and painter
   submission unchanged -> identical pixels. `_surface_outline_segments` gains an optional
   precomputed-loops argument (backward compatible).
3. Animation reuse: per-item compatibility signature (positions/triangles/lines/points/
   element mapping arrays equal, chunk ids + chunk topology equal, static appearance kwargs
   equal). Compatible -> reuse existing MeshHandle, update via its mutators (positions,
   displacements, scalar fields, active, transform, deformation scale, selected, visible,
   appearance); renderer picks up generation deltas. Incompatible -> existing full rebuild.
   Live-entry removal each frame, entry semantics/owners rebuild, restore/cancel paths unchanged.
4. Selection key: narrow generations dependency to (topology, position, displacement, active,
   selection, transform); keep visible/removed/deformation_scale and all other key members.
   Scalar/appearance-only changes no longer rebuild the projected index.
5. Scheduler: optional thread-safe notifier hook (`set_notifier`), invoked after enqueue in
   `submit` (failure-safe: item stays queued), `pending()` helper, drain reposts notification
   when the 1024 bound leaves items, close() detaches notifier first. Qt: replace the 2 ms
   QTimer with a QObject receiving a registered custom event posted via
   `QCoreApplication.postEvent` (thread-safe, no lost wakeup, discarded after close/deletion).
   Tk default (no notifier) unchanged.

## Test plan (focused, no windows/GPU)

- tests/test_qt_update_pump.py: rewrite for queued-notification contract (idempotent start,
  cross-thread FIFO via posted event, 1024 bound + continuation, close/stale safety,
  initial-failure safety, wrong-thread refusal).
- tests/test_scheduler.py: add notifier/pending/close tests; keep existing two tests.
- New tests/test_performance_improvements.py: GPU hover coalescing + exact clicks; software
  prep reuse counts + pixel parity across invalidating mutations; animation handle reuse,
  incompatible rebuild, restore/cancel; selection-key scalar exclusion vs geometry
  invalidation; Tk compatibility (no notifier).
- Run with C:/Python/Python314/python.exe -m pytest, each <=120s; logs in this directory.

## Status

Inspection and design complete. Implementation not yet started. No source files modified yet.
