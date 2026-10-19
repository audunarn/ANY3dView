# Independent integration review (in progress)

Baseline benchmark complete on actual Examples barge with display-only mesh. Authoritative baseline-final-{gpu,software}; older runs retained as harness diagnostics. No approximation/LOD. Full immutable input snapshot under anyfem-src and baseline-src.

Review checks to resolve before acceptance:
- Scheduler: a callback exception with more queued callbacks must still schedule remaining work; an empty queue must not poll. Submit/close race must not retain accepted work silently after close. Notifier errors must not strand accepted work without observable failure.
- Software cache: deformation, transform, active/hidden/selection masks, section and highlight/color changes must invalidate; camera-only changes reuse. Outline loops must depend on admitted visible triangles where near/far clipping changes; do not draw hidden boundaries.
- Animation: chunk scalar/displacement changes, None transitions, dynamic owner mappings, failed mutation, stop and incompatible frame must retain valid live scene/ownership. Same static appearance must reuse; different appearance must not accidentally preserve prior one.
- Picking key: scalar edits may reuse geometry index; camera basis/visibility/transform/deformation/owner changes cannot.
- Hover: latest event only, but exact click and all cancel paths unaffected. Native benchmark should not confuse intentional cancellation with reduced work.

No author code modifications by parent while worker owns the slice.

Concrete first-pass review findings (candidate still being authored):
- qt.py imports _UpdateNotifier(QObject) without QObject imported; import smoke must catch.
- _software_prep entries retain handles and world arrays without pruning when entries/chunks removed or animation clears. Add bounded pruning and clear on destroy/clear so cache cannot accumulate obsolete meshes.
- Current postEvent per submission produces an event per callback; coalesce pending wakeups with thread-safe state and bounded continuation. Finally ensure callback exception cannot strand remaining work. Atomic close/submit and notifier failure semantics need tests.
- GPU hover design 0ms per event-loop pass only collapses synthetic bursts; steady mouse events each dispatched in separate loop iterations can still issue a read every time. Adopt a bounded ~16ms non-restarting single-shot interval for GPU hover (latest payload, start only if inactive); software remains120ms debounce. Cancel paths remain exact. Verify sustained high-rate moves produce at most one read per interval and latest event is delivered.

Second-pass review delta:
- Preserve original scheduler fail-fast semantics. drain() previously propagated a callback error before executing subsequent callbacks. Current continue-on-error loop changes that contract. Use finally to notify remaining work and re-raise immediately; next owner event can drain remaining callbacks. Focused test must assert subsequent callback not executed before first error reaches caller, then executes on next drain.
- _UpdateNotifier.wake sets _pending True before postEvent; if postEvent throws, flag must be reset so future successful submission can retry. Current stuck flag suppresses every subsequent wakeup.
- Blocking animation correctness: _animation_item_compatible accepts same-shaped changed chunk positions/scalars/displacements/active data; _apply_animation_item never updates chunk arrays (only ownership). Must either replace changed chunks or declare any changed chunk data incompatible and rebuild. Add actual animation frame test with changed chunk scalars AND positions, verify new values/picks not stale. Also normals must be compared as static mesh data if no update API.

Independent executable evidence now confirms all three blockers: tests/test_performance_review_oracle.py, report review-oracle-before-fix2/stdout.log (3 failures): changed chunk positions stay old, changed normals stay old, scheduler callback executes later work before raising. First run's two animation failures were fake-host missing stop_animation; repaired by binding actual QtViewer.stop_animation, then rerun confirms actual product failures. Do NOT weaken these root-owned oracles. Fix source so all three pass.

Software acceptance remains unmet: same barge software orbit median232.70->230.57ms (~1%, noise-sized), so not accepted as a measured gain. New profile reports/performance-20261019/software-profile/profile.txt: three paints1.236s; outline boundary recovery .298s, _surface_outline_segments .386s, per-primitive pixel_ratio .112s. Camera-invariant outline recovery was not cached in delivered prep implementation. Next narrow slice: retain semantic outline loops in _SoftwareChunkPrep keyed with geometry/admission/outline groups; use cached loops only when visible/admitted triangles match, otherwise original clipping/outline route. Reuse already computed screen_all for fully unclipped outline segments. Hoist effective pixel_ratio once perpaint (not necessarily surfaceDPR; preserve override) and cache per-frame identical QPen/QColor styles without changing draworder or antialiasing. Parent oracle already checks section,active,selection,transform,deformation,hide,position, plus final images/picks; exact parity required. No LOD or dropped edges. Parent handles all native benchmarks; worker fake checks only.
