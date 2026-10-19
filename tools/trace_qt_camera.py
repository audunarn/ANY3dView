"""Passive, bounded camera/render trace for an EXISTING Qt GPU viewer.

No window, timer, GPU synchronization or file I/O is started by this module.
Use CameraPresentationTrace(viewer), then trace.snapshot() when saving the
user session, and trace.close() to restore methods. Attach after GPU startup.
Qt swap notifications are not GPU-completion or physical-display evidence.
"""
from collections import deque
from time import perf_counter_ns


def camera_pose(camera):
    return dict(azimuth=camera.azimuth, elevation=camera.elevation,
                distance=camera.distance, position=camera.position.to_tuple(),
                target=camera.target.to_tuple(), near=camera.near, far=camera.far)


class CameraPresentationTrace:
    def __init__(self, viewer, *, capacity=10000):
        self.viewer = viewer
        self.renderer = viewer._renderer
        self.surface = viewer.event_widget
        if viewer.backend_name != 'gpu' or not hasattr(self.surface, 'frameSwapped'):
            raise ValueError('Attach to an initialized Qt GPU viewer')
        self.events = deque(maxlen=max(1, int(capacity)))
        self.total = 0
        self.started_ns = perf_counter_ns()
        self.motion_id = 0
        self.render_id = 0
        self.last_render = None
        self._patches = []
        self._closed = False
        self._patch(viewer, '_motion', self._motion)
        for name in ('_press', '_release', '_cancel_interaction'):
            self._patch(viewer, name, self._interaction(name))
        self._patch(self.renderer, 'render', self._render)
        self.surface.frameSwapped.connect(self._swapped)

    def _record(self, kind, **data):
        self.total += 1
        self.events.append(dict(kind=kind, at_ns=perf_counter_ns(), **data))

    def _patch(self, obj, name, wrapper):
        previous = getattr(obj, name)
        had_local = name in vars(obj)
        def invoke(*args, **kwargs):
            return wrapper(previous, *args, **kwargs)
        self._patches.append((obj, name, previous, had_local, invoke))
        setattr(obj, name, invoke)

    def _motion(self, original, event):
        self.motion_id += 1
        identity = self.motion_id
        before = camera_pose(self.viewer.camera)
        mode = self.viewer._drag
        mouse = self.viewer._mouse
        start = perf_counter_ns()
        try:
            return original(event)
        finally:
            self._record('motion', motion_id=identity, start_ns=start,
                input_xy=(event.x,event.y), previous_xy=mouse, mode_before=mode,
                mode_after=self.viewer._drag, before=before,
                after=camera_pose(self.viewer.camera),
                original_renderer_active=self.viewer._renderer is self.renderer,
                original_surface_active=self.viewer.event_widget is self.surface)

    def _interaction(self, name):
        def traced(original, *args, **kwargs):
            before = self.viewer._drag
            try:
                return original(*args, **kwargs)
            finally:
                self._record(name, mode_before=before, mode_after=self.viewer._drag,
                             pose=camera_pose(self.viewer.camera))
        return traced

    def _render(self, original, camera, viewport, **kwargs):
        self.render_id += 1
        identity = self.render_id
        pose = camera_pose(camera)
        start = perf_counter_ns()
        try:
            result = original(camera, viewport, **kwargs)
        except BaseException as error:
            self.last_render = None
            self._record('render_failed', render_id=identity, error=type(error).__name__)
            raise
        self.last_render = dict(render_id=identity, motion_id=self.motion_id, pose=pose)
        self._record('render_submitted', start_ns=start, viewport=viewport, **self.last_render)
        return result

    def _swapped(self):
        self._record('qt_swap', last_submission=self.last_render)

    def snapshot(self):
        return dict(started_ns=self.started_ns, total_events=self.total,
                    dropped_events=max(0,self.total-len(self.events)),
                    original_renderer_active=self.viewer._renderer is self.renderer,
                    original_surface_active=self.viewer.event_widget is self.surface,
                    events=list(self.events),
                    limitation='Qt swap is not GPU completion or physical presentation; reattach after backend/context replacement.')

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.surface.frameSwapped.disconnect(self._swapped)
        except (RuntimeError, TypeError):
            pass  # Surface may already have been destroyed.
        for obj, name, previous, had_local, installed in reversed(self._patches):
            if getattr(obj,name,None) is installed:
                if had_local:
                    setattr(obj,name,previous)
                else:
                    delattr(obj,name)
        self._patches.clear()
