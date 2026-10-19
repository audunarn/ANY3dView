"""Optional Qt hosts for the shared retained viewer, imported only on request."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QOpenGLContext, QPen, QPolygonF, QSurfaceFormat
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .core import Point3D, parse_color, _interpolate_thickness_color
from .appearance import scalar_color
from .errors import GPUUnavailableError
from .retained_viewer import RetainedViewer, GPU_CAPABILITIES, _FrameProjection, _surface_outline_loops


_UPDATE_EVENT_TYPE = QEvent.Type(QEvent.registerEventType())

# Upper bound on retained software preparations; the oldest entries are
# evicted first so the cache cannot grow with transient scenes.
_SOFTWARE_PREP_LIMIT = 1024


class _UpdateEvent(QEvent):
    """Owner-thread wakeup posted by the shared scheduler's notifier."""

    def __init__(self):
        super().__init__(_UPDATE_EVENT_TYPE)


class _UpdateNotifier(QObject):
    """Drain queued viewer updates when the scheduler posts a wakeup.

    The scheduler invokes its notifier hook from any thread; postEvent is
    Qt's thread-safe handoff and delivers customEvent on the owner thread.
    """

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self._owner = owner
        # Coalescing flag: many submissions before the owner thread runs
        # its event loop produce one wakeup, and the flag is only cleared
        # when processing starts, so work submitted during a drain still
        # posts a fresh wakeup.  The lock makes the flag safe from any
        # submitting thread.
        self._pending = False
        self._wake_lock = Lock()

    def wake(self):
        with self._wake_lock:
            if self._pending:
                return
            self._pending = True
        try:
            QCoreApplication.postEvent(self, _UpdateEvent())
        except Exception:
            # A failed post must not leave the coalescing flag stuck:
            # the next successful submission has to be able to retry.
            with self._wake_lock:
                self._pending = False
            raise

    def customEvent(self, event):
        if event.type() is not _UPDATE_EVENT_TYPE:
            return
        with self._wake_lock:
            self._pending = False
        owner = self._owner
        if owner._closed:
            return
        owner._poll_updates()


class _SoftwareChunkPrep:
    """Camera-invariant retained preparation for one software mesh chunk.

    Holds world-space geometry, per-triangle admission/colors, clipped
    line segments and point admission produced by the same loops the
    uncached painter ran; projection, depth ordering and submission stay
    per-frame so cached and direct paints produce identical pixels.
    """

    __slots__ = ("handle", "signature", "world", "triangles", "colors",
                 "section_polygons", "line_segments", "line_selected",
                 "points", "point_selected", "outline_loops")

    def __init__(self, handle, signature, world, triangles, colors,
                 section_polygons, line_segments, line_selected,
                 points, point_selected):
        self.handle = handle
        self.signature = signature
        self.world = world
        self.triangles = triangles
        self.colors = colors
        self.section_polygons = section_polygons
        self.line_segments = line_segments
        self.line_selected = line_selected
        self.points = points
        self.point_selected = point_selected
        self.outline_loops = ()


class _QtGLLoader:
    """Resolve functions through the widget's current Qt context, including EGL."""
    def __init__(self, context):
        self.context = context

    def load_opengl_function(self, name):
        pointer = self.context.getProcAddress(name.encode("ascii"))
        return int(pointer) if pointer else 0

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

    def release(self):
        pass  # Qt owns the native context.


class _SoftwareResources:
    """Appearance and semantic masks for the Qt painter backend."""
    pick_dirty = True
    draw_calls = frame_count = geometry_uploads = 0

    def __init__(self):
        self.appearance = {}
        self.masks = {}
        self.chunk_masks = {}
        self.chunk_pickable = {}
        # Bumped on every mask/appearance mutation so retained software
        # preparation can invalidate without introspecting contents.
        self.mask_revision = 0

    def add_mesh(self, handle, **appearance):
        self.appearance[id(handle)] = appearance
        self.geometry_uploads += 1
        self.mask_revision += 1

    def remove_mesh(self, handle):
        self.appearance.pop(id(handle), None)
        self.masks.pop(id(handle), None)
        for key in tuple(self.chunk_masks):
            if key[0] == id(handle):
                self.chunk_masks.pop(key, None)
                self.chunk_pickable.pop(key, None)
        self.mask_revision += 1

    def set_semantic_masks(self, handle, **masks):
        self.masks[id(handle)] = masks
        self.mask_revision += 1

    def set_highlighted_elements(self, handle, indices):
        self.masks.setdefault(id(handle), {})["highlighted_elements"] = indices
        self.mask_revision += 1

    def set_chunk_semantic_masks(self, handle, chunk_id, **masks):
        self.chunk_masks[id(handle), chunk_id] = masks
        self.mask_revision += 1

    def clear_chunk_semantic_masks(self, handle, chunk_id):
        self.chunk_masks.pop((id(handle), chunk_id), None)
        self.mask_revision += 1

    def set_chunk_pickable(self, handle, chunk_id, enabled):
        self.chunk_pickable[id(handle), chunk_id] = enabled
        self.mask_revision += 1

    def release(self):
        self.appearance.clear()
        self.masks.clear()
        self.chunk_masks.clear()
        self.chunk_pickable.clear()


class _PainterHud:
    """Painter implementation of the same overlay operations as the GPU HUD."""
    uploads = 0
    def __init__(self):self.painter=None;self.scale=1.0
    def begin(self,viewport,scale=1.0):self.uploads+=1;self.scale=float(scale)
    def render(self,target=None):pass
    def release(self):self.painter=None
    def quad(self,rect,color,alpha=1,**options):
        fill=QColor(color);fill.setAlphaF(alpha)
        self.painter.setPen(Qt.NoPen);self.painter.setBrush(fill)
        s=self.scale
        x0,y0,x1,y1=(value*s for value in rect)
        from PySide6.QtCore import QRectF
        self.painter.drawRect(QRectF(x0,y0,x1-x0,y1-y0))
    def line(self,start,end,color,width=1,**options):
        s=self.scale
        self.painter.setPen(QPen(QColor(color),width*s));self.painter.setBrush(Qt.NoBrush)
        self.painter.drawLine(QPointF(start[0]*s,start[1]*s),QPointF(end[0]*s,end[1]*s))
    def polyline(self,points,color,width=1,closed=False,**options):
        for a,b in zip(points,points[1:]):self.line(a,b,color,width)
        if closed and points:self.line(points[-1],points[0],color,width)
    def rectangle(self,rect,color,width=1,fill=None,fill_alpha=1,**options):
        if fill:self.quad(rect,fill,fill_alpha)
        x0,y0,x1,y1=rect
        self.polyline(((x0,y0),(x1,y0),(x1,y1),(x0,y1)),color,width,True)
    def circle(self,point,radius,color,width=1,**options):
        s=self.scale
        self.painter.setPen(QPen(QColor(color),width*s));self.painter.setBrush(Qt.NoBrush)
        self.painter.drawEllipse(QPointF(point[0]*s,point[1]*s),radius*s,radius*s)
    def text(self,point,text,color,font=("Segoe UI",10,""),anchor="nw",**options):
        s=self.scale
        qfont=QFont(font[0],max(1,round(font[1]*s)));qfont.setBold("bold" in font[2])
        self.painter.setFont(qfont);self.painter.setPen(QColor(color))
        metrics=self.painter.fontMetrics();x,y=point[0]*s,point[1]*s
        if "e" in anchor:x-=metrics.horizontalAdvance(str(text))
        elif "w" not in anchor:x-=metrics.horizontalAdvance(str(text))/2
        if "n" in anchor:y+=metrics.ascent()
        elif "s" not in anchor:y+=(metrics.ascent()-metrics.descent())/2
        self.painter.drawText(QPointF(x,y),str(text))


class _InputSurface:
    def focus_set(self):
        self.setFocus()

    def _event(self, event):
        p = event.position()
        ratio = self.devicePixelRatioF()
        modifiers = event.modifiers()
        state = (1 if modifiers & Qt.ShiftModifier else 0)
        state |= 4 if modifiers & Qt.ControlModifier else 0
        state |= 8 if modifiers & Qt.AltModifier else 0
        return SimpleNamespace(x=round(p.x()*ratio), y=round(p.y()*ratio), state=state)

    def mousePressEvent(self, event):
        self.owner._cancel_hover()
        self.setFocus()
        e = self._event(event)
        if event.button() == Qt.LeftButton:
            self.owner._press_select(e)
        else:
            self.owner._press(e, "orbit" if event.button() == Qt.RightButton else "pan")
        event.accept()

    def mouseMoveEvent(self, event):
        e = self._event(event)
        if event.buttons() & Qt.LeftButton:
            self.owner._cancel_hover()
            self.owner._drag_select(e)
        elif event.buttons():
            self.owner._cancel_hover()
            self.owner._motion(e)
        elif self.owner.backend_name == "software":
            self.owner._queue_hover(e)
        else:
            # Hover is optional feedback: keep the latest event and
            # evaluate it at most once per bounded interval instead of
            # running a synchronous GPU pick per move. The timer is not
            # restarted by later moves, so sustained move bursts issue
            # at most one read per interval. Exact clicks bypass this
            # path and stale requests are canceled below.
            self.owner._queue_hover(e, 16, restart=False)
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.owner._release_select(self._event(event))
        else:
            self.owner._release(self._event(event))
        event.accept()

    def wheelEvent(self, event):
        self.owner._cancel_hover()
        self.owner._zoom(0.9 ** (event.angleDelta().y()/120))
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.owner._cancel_interaction()
        else:
            super().keyPressEvent(event)

    def leaveEvent(self, event):
        self.owner._cancel_hover()
        super().leaveEvent(event)

    def hideEvent(self, event):
        self.owner._cancel_hover()
        super().hideEvent(event)

    def focusOutEvent(self, event):
        self.owner._cancel_hover()
        self.owner._cancel_interaction()
        super().focusOutEvent(event)


class _PaintSurface(_InputSurface, QWidget):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self._cached_image = None
        self._cached_key = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def paintEvent(self, event):
        owner = self.owner
        if owner._closed:
            return
        camera = owner.camera
        # External camera edits also invalidate the cache, even without an
        # explicit redraw. Logical size and DPR determine the raster target.
        key = (owner._software_revision, self.width(), self.height(),
               self.devicePixelRatioF(), camera.position.to_tuple(),
               camera.target.to_tuple(), camera.world_up.to_tuple(),
               camera.fov, camera.near, camera.far, owner.bg)
        if key != self._cached_key:
            ratio = self.devicePixelRatioF()
            image = QImage(max(1,round(self.width()*ratio)),
                           max(1,round(self.height()*ratio)), QImage.Format_RGBA8888)
            image.setDevicePixelRatio(ratio)
            painter = QPainter(image)
            try:
                owner._paint_software(painter)
            finally:
                if painter.isActive():
                    painter.end()
            self._cached_image, self._cached_key = image, key
        painter = QPainter(self)
        try:
            painter.drawImage(QPointF(0,0), self._cached_image)
        finally:
            painter.end()


def _gl_surface_format():
    """Keep renderer requirements while honoring the application's swap policy."""
    fmt = QSurfaceFormat()
    fmt.setSwapInterval(QSurfaceFormat.defaultFormat().swapInterval())
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    from .gpu.quality import msaa_samples
    samples = msaa_samples()
    if samples > 1:
        fmt.setSamples(samples)
    fmt.setDepthBufferSize(24)
    return fmt


def _capture_gl_info(context):
    """Cache identifying GL strings once, while the context is current."""
    try:
        info = context.info
        return {"vendor": str(info.get("GL_VENDOR", "")),
                "renderer": str(info.get("GL_RENDERER", "")),
                "version": str(info.get("GL_VERSION", ""))}
    except Exception:
        return {}


class _GLSurface(_InputSurface, QOpenGLWidget):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.error = None
        fmt = _gl_surface_format()
        self.setFormat(fmt)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def initializeGL(self):
        self.error = None
        owner = self.owner
        owner._note_lifecycle("initialize_gl_attempts")
        try:
            context = self.context()
            prior = getattr(owner, "_gl_context", None)
            if prior is context and getattr(owner, "_renderer", None) is not None:
                if getattr(owner, "_gl_ready", False):
                    # Qt re-ran initializeGL on the same live context: the
                    # installed renderer/HUD are still owned and unreleased, so
                    # re-running the constructor path would only install
                    # unreleased replacements over them.
                    owner._note_lifecycle("initialize_gl_repeats")
                    return
                # A previous attempt on this same context failed after ownership
                # was assigned (replay or highlight initialization): partially
                # initialized resources must not be treated as a successful
                # repeat. Release them and rebind the context before rebuilding.
                owner._note_lifecycle("initialize_gl_incomplete_repeats")
                owner._release_gl(context)
                self.makeCurrent()
            if prior is not None and prior is not context:
                owner._note_lifecycle("context_recreations")
                # Reconcile prior ownership before installing replacements:
                # the old token's callback is disconnected and the abandoned
                # resources are dropped. The old context is no longer this
                # surface's context, so _release_gl skips unsafe GL calls
                # and records the abandonment instead of touching the
                # replacement context.
                owner._release_gl(prior)
            import inspect
            import moderngl
            from .gpu.renderer import ModernGLRenderer
            from .gpu.hud import GPUHudRenderer
            renderer = ModernGLRenderer(moderngl.create_context(
                require=330, context=_QtGLLoader(context)))
            try:
                hud = GPUHudRenderer(renderer.ctx)
            except BaseException as error:
                # The context is current here; release the renderer's
                # derived resources so a failing HUD constructor cannot
                # leak them past the ownership token below. The original
                # constructor error is preserved if this cleanup fails.
                try:
                    renderer.release()
                except BaseException as cleanup_error:
                    owner._note_lifecycle("gl_release_failures",
                                           error=cleanup_error)
                raise
            owner._renderer = renderer
            owner._hud = hud
            owner._gl_context = context
            owner._gl_info = _capture_gl_info(renderer.ctx)
            # Store the exact slot and its context so cleanup can
            # disconnect it (Qt's aboutToBeDestroyed lifecycle contract).
            # The closure carries the context token: a stale callback for
            # a replaced context must not release the replacement's
            # resources.
            slot = lambda: owner._release_gl(context)
            owner._gl_destroy_slot = (context, slot)
            context.aboutToBeDestroyed.connect(slot)
            # Qt can recreate the context when the widget is reparented.
            # Re-upload retained snapshots rather than losing the visible scene.
            accepted=inspect.signature(renderer.add_mesh).parameters
            for entry in owner._display_entries().values():
                renderer.add_mesh(entry["handle"],**{k:v for k,v in entry["appearance"].items() if k in accepted})
            owner._apply_highlight_masks()
            owner._gl_ready = True
        except Exception as error:
            self.error = error
            if not getattr(self.owner,"_initializing",True):
                self._defer_render_failure()

    def _defer_render_failure(self):
        # Context recreation can recover this surface (or replace it) before
        # the queued callback runs. Keep the identity of the failed attempt.
        owner, error = self.owner, self.error
        QTimer.singleShot(0, lambda: owner._render_failed(self, error))

    def paintGL(self):
        if self.error is not None or self.owner._closed:
            return
        try:
            owner = self.owner
            renderer = owner._renderer
            owner._sync_pixel_ratio()
            target = renderer.ctx.detect_framebuffer(self.defaultFramebufferObject())
            renderer.render(owner.camera, owner.viewport_size,
                            target=target, section_plane=owner._section_plane,
                            clear_color=tuple(c/255 for c in (parse_color(owner.bg) or (255,255,255)))+(1,),
                            light=owner._light, shading_enabled=owner._shading_enabled,
                            occlude_lines=owner._occlude_lines,
                            show_mesh_lines=owner.show_mesh_lines,
                            selection_color=owner._highlight_fill)
            owner._render_hud(owner.viewport_size, target=target)
        except Exception as error:
            self.error = error
            self._defer_render_failure()


class QtViewer(RetainedViewer, QWidget):
    """QWidget satisfying ViewerBackend through shared retained behavior."""
    render_error = Signal(str)
    backend_changed = Signal(str)
    def __init__(self, parent=None, *, backend="software", **options):
        QWidget.__init__(self, parent)
        allowed = {k: options[k] for k in ("width", "height", "bg", "interactive_fps", "shading", "interaction_profile") if k in options}
        self.initialize_state(**allowed)
        self._backend = backend
        # Explicit GL ownership: the token is the Qt context that the
        # current renderer/HUD resources were allocated under; it is set
        # only once both are constructed. Cached GL strings and counters
        # are passive diagnostics and own no GL state.
        self._gl_context = None
        self._gl_ready = False
        self._gl_destroy_slot = None
        self._gl_info = {}
        self._lifecycle_counters = {"initialize_gl_attempts": 0,
            "context_recreations": 0, "gl_release_attempts": 0,
            "gl_release_failures": 0, "destroy_calls": 0}
        self._lifecycle_errors = []
        self._initializing = True
        self._timers = set()
        self._software_revision = 0
        self._hover_timer = None
        self._pending_hover = None
        self._software_prep = {}
        self._update_notifier = None
        self._host = self
        self._renderer = _SoftwareResources()
        self._hud = _PainterHud() if backend == "software" else None
        self.surface = _GLSurface(self) if backend == "gpu" else _PaintSurface(self)
        self.canvas = self.surface
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.surface)
        self.resize(self.width, self.height)
        if backend == "gpu":
            # Probe a real widget context before returning a usable backend.
            self.surface.resize(self.width, self.height)
            self.surface.show()
            try:
                self.surface.grabFramebuffer()
            except Exception as error:
                # Roll back the half-constructed viewer before the failure
                # escapes, mirroring the invalid-context branch below.
                self.destroy()
                raise GPUUnavailableError(str(error), diagnostics=(type(error).__name__, str(error)))
            if not self.surface.isValid() or self.surface.error is not None:
                error = self.surface.error or RuntimeError("Qt OpenGL context is unavailable")
                self.destroy()
                raise GPUUnavailableError(str(error), diagnostics=(type(error).__name__, str(error)))
        self._start_update_pump()
        self._initializing = False

    @property
    def backend_name(self):
        return self._backend

    @property
    def capabilities(self):
        if self._backend == "gpu":
            return GPU_CAPABILITIES
        return replace(GPU_CAPABILITIES, gpu=False, integer_picking=False,
                       shader_deformation=False, line_occlusion=False,
                       stippled_transparency=False, smooth_normals=False)

    def _note_lifecycle(self, event, error=None):
        """Count a lifecycle event; retain bounded evidence of failures."""
        counters = getattr(self, "_lifecycle_counters", None)
        if counters is not None:
            counters[event] = counters.get(event, 0) + 1
        if error is not None:
            errors = getattr(self, "_lifecycle_errors", None)
            if errors is not None:
                errors.append(f"{type(error).__name__}: {error}")
                del errors[:-8]

    def diagnostic_snapshot(self):
        """Passive, JSON-compatible lifecycle diagnostics for this viewer.

        Reads only cached state and Qt's isValid(); it never renders,
        makes the context current, switches backends, queries live GL,
        enumerates processes, touches project data or mutates ownership.
        """
        context = getattr(self, "_gl_context", None)
        valid = False
        if context is not None:
            try:
                valid = bool(context.isValid())
            except RuntimeError:
                valid = False  # Deleted Qt object.
        info = getattr(self, "_gl_info", None) or {}
        return {
            "schema": "any3dview.qt.lifecycle",
            "version": 1,
            "active_backend": self._backend,
            "closed": bool(self._closed),
            "context_valid": valid,
            "context_ready": bool(getattr(self, "_gl_ready", False)),
            "gl_vendor": str(info.get("vendor", "")),
            "gl_renderer": str(info.get("renderer", "")),
            "gl_version": str(info.get("version", "")),
            "lifecycle_counters": dict(getattr(self, "_lifecycle_counters", None) or {}),
            "lifecycle_errors": list(getattr(self, "_lifecycle_errors", None) or ()),
        }

    def _queue_hover(self, event, interval=120, restart=True):
        if self._closed:
            return
        # Hover is optional feedback. Coalesce pointer transit so leaving for
        # a control does not enqueue an obsolete CPU selection-index build.
        # The latest event always wins. A restarting interval debounces the
        # software path; the GPU path starts a bounded non-restarting
        # interval only when inactive, so steady move bursts produce at
        # most one read per interval.
        self._pending_hover = (self.surface, SimpleNamespace(x=event.x, y=event.y))
        if self._hover_timer is None:
            self._hover_timer = QTimer(self)
            self._hover_timer.setSingleShot(True)
            self._hover_timer.timeout.connect(self._deliver_hover)
            self._timers.add(self._hover_timer)
        if restart or not self._hover_timer.isActive():
            self._hover_timer.start(interval)

    def _cancel_hover(self):
        self._pending_hover = None
        if self._hover_timer is not None:
            self._hover_timer.stop()

    def _deliver_hover(self):
        pending, self._pending_hover = self._pending_hover, None
        if pending is None or self._closed:
            return
        surface, event = pending
        if surface is self.surface and surface.underMouse() and not self._drag and not self._selection_dragging:
            self._hover_select(event)

    def _start_update_pump(self):
        if self._closed or self._update_poll_id is not None:
            return
        # Drain work that arrived before a wakeup existed; a callback
        # failure here escapes the constructor with no notifier armed.
        self._poll_updates()
        if self._closed:
            return
        notifier = _UpdateNotifier(self, self)
        self._update_notifier = notifier
        self._update_poll_id = notifier
        # The scheduler invokes this hook from any submitting thread; the
        # notifier coalesces pending wakeups into one owner-thread event,
        # so no idle polling timer is needed.
        self._update_scheduler.set_notifier(notifier.wake)
        # A submission racing setup gets no wakeup; drain once more so
        # it cannot wait for the next submit.
        self._poll_updates()

    def _poll_updates(self):
        if self._closed:
            return
        completed = self._update_scheduler.drain()
        if completed:
            self.redraw()

    def after(self, delay, callback):
        timer = QTimer(self)
        timer.setSingleShot(True)
        self._timers.add(timer)
        def run():
            self._timers.discard(timer)
            timer.deleteLater()
            if not self._closed:
                callback()
        timer.timeout.connect(run)
        timer.start(delay)
        return timer

    def after_cancel(self, timer):
        timer.stop()
        self._timers.discard(timer)
        timer.deleteLater()

    def _native_pixel_ratio(self):
        return float(self.surface.devicePixelRatioF())

    def framebuffer_size(self):
        ratio = self.surface.devicePixelRatioF()
        return max(1, round(self.surface.width()*ratio)), max(1, round(self.surface.height()*ratio))

    def request_redraw(self):
        self._software_revision += 1
        self.surface.update()

    def cancel_redraw(self):
        pass  # Qt coalesces update requests; closed state prevents drawing.

    def make_current(self):
        if self._closed:
            raise RuntimeError("Cannot update a closed Qt viewer")
        if self._backend == "gpu":
            self.surface.makeCurrent()

    def swap_buffers(self):
        pass  # QOpenGLWidget owns composition and presentation.

    def _mesh_changed(self, handle, change):
        self.make_current()
        result = super()._mesh_changed(handle, change)
        if change in ("remove", "topology"):
            # Removed meshes and vanished chunks must not retain world
            # arrays in the software preparation cache.
            self._prune_software_prep()
        return result

    def _apply_highlight_masks(self):
        self.make_current()
        return super()._apply_highlight_masks()

    def _restore_live_renderer(self):
        self.make_current()
        result = super()._restore_live_renderer()
        self._prune_software_prep()
        return result

    def _show_animation_frame(self, index):
        self.make_current()
        result = super()._show_animation_frame(index)
        # Superseded animation frames must not leave obsolete handles
        # and world arrays cached for the software painter.
        self._prune_software_prep()
        return result

    def _press_select(self, event):
        self.surface.setFocus()
        self._selection_modifiers = self._event_modifiers(event)
        self._selection_operation = self._operation_from_modifiers(*self._selection_modifiers)
        point = (int(event.x), int(event.y))
        self._selection_press = self._selection_current = point
        self._selection_points = [point]
        self._selection_dragging = False
        if self._interaction_profile == "legacy":
            self._press(event, "pan")
        elif self._selection_config.click_on_press:
            hits = self._emit_click(point, self._selection_operation)
            self._selection_press_hit_keys = frozenset(hit.key for hit in hits)

    def stop_animation(self):
        was_playing = self._is_playing_animation or bool(self._animation_handles)
        self._is_playing_animation = False
        if self._animation_after_id is not None:
            self.after_cancel(self._animation_after_id)
            self._animation_after_id = None
        if was_playing:
            self._restore_live_renderer()
            if self._animation_live_hud is not None:
                live=self._animation_live_hud
                self._world_text=[dict(value) for value in live["world_text"]]
                self._thickness_legend=None if live["legend"] is None else dict(live["legend"])
                for key in ("highlighted_tags","preselected_key","preselection_from_hit","highlight_fill","highlight_outline"):
                    setattr(self,"_"+key,live[key])
                self._animation_live_hud=None
            self._apply_highlight_masks()
            self.redraw()

    def capture_image(self):
        from PIL import Image
        image = self.surface.grabFramebuffer() if self._backend == "gpu" else self.surface.grab().toImage()
        image = image.convertToFormat(QImage.Format.Format_RGBA8888)
        return Image.frombytes("RGBA", (image.width(), image.height()), bytes(image.constBits()), "raw", "RGBA", image.bytesPerLine())

    def _release_gl(self, context=None):
        """Release GL resources owned by this viewer's tracked context.

        The optional token is the Qt context whose aboutToBeDestroyed
        fired; a stale callback for a replaced context must not release
        the replacement's resources. Qt owns the native context: only
        derived resources are released, and only after the surface's
        context and Qt's thread-current context are verified to be the
        owned token; a valid replacement context is never released into.
        """
        if self._backend != "gpu":
            return
        owner_context = getattr(self, "_gl_context", None)
        if owner_context is None:
            return  # No GL resources were allocated, or already released.
        if context is not None and context is not owner_context:
            return
        # Qt's lifecycle contract: disconnect the exact stored slot (and
        # its context) during cleanup so a later aboutToBeDestroyed
        # cannot re-enter release for resources that are already gone;
        # a stale entry for another context is left untouched.
        slot_entry = getattr(self, "_gl_destroy_slot", None)
        if slot_entry is not None and slot_entry[0] is owner_context:
            try:
                owner_context.aboutToBeDestroyed.disconnect(slot_entry[1])
            except Exception:
                pass  # Already disconnected, or the Qt object is gone.
            self._gl_destroy_slot = None
        self._gl_context = None
        self._gl_ready = False
        self._note_lifecycle("gl_release_attempts")
        hud, renderer, surface = self._hud, self._renderer, self.surface
        failures = []
        try:
            valid = bool(surface.isValid()) and bool(owner_context.isValid())
        except Exception:
            valid = False  # Deleted/broken Qt objects; native resources are gone.
        if not valid:
            failures.append("skipped GL release: Qt context unavailable")
        bound = False
        if valid:
            # The surface's context identity is verified BEFORE binding:
            # during initializeGL on a replacement context, makeCurrent
            # would bind the new context and doneCurrent would strip it
            # before moderngl.create_context runs. A replaced surface
            # context is recorded and skipped without any binding.
            try:
                surface_context = surface.context()
            except Exception:
                surface_context = None
            if surface_context is not owner_context:
                failures.append("skipped GL release: surface context is not the owned context")
                valid = False
        if valid:
            try:
                surface.makeCurrent()
                bound = True
            except Exception as error:
                failures.append(f"makeCurrent: {type(error).__name__}: {error}")
                valid = False
        if valid:
            # Binding succeeded; Qt's thread-current context must still be
            # the owned token before any release call is issued.
            try:
                current = QOpenGLContext.currentContext()
            except Exception:
                current = None
            if current is not owner_context:
                failures.append("skipped GL release: current context is not the owned context")
                valid = False
        if valid:
            # Each owner releases independently so one failure cannot skip
            # the others or the doneCurrent pairing.
            for name, resource in (("hud", hud), ("renderer", renderer)):
                if resource is None:
                    continue
                try:
                    resource.release()
                except BaseException as error:
                    failures.append(f"{name}.release: {type(error).__name__}: {error}")
        if bound:
            # The binding succeeded, so unbind it even when release was
            # skipped; a doneCurrent failure must not mask earlier ones.
            try:
                surface.doneCurrent()
            except Exception as error:
                failures.append(f"doneCurrent: {type(error).__name__}: {error}")
        for failure in failures:
            self._note_lifecycle("gl_release_failures", error=RuntimeError(failure))
        self._hud = None
        self._renderer = None

    def _render_failed(self, source=None, failure=None):
        if self._closed or self._backend != "gpu":
            return
        if source is not None and source is not self.surface:
            return
        error = self.surface.error
        if error is None or (source is not None and error is not failure):
            return
        self._backend_diagnostics += (f"Qt rendering failed: {error}",)
        self.setToolTip(self._backend_diagnostics[-1])
        self.render_error.emit(self._backend_diagnostics[-1])
        if getattr(self,"_allow_fallback",False):
            try:
                for entry in self._display_entries().values():
                    handle=entry["handle"]
                    for mesh in (handle.mesh,*(mesh for _key,mesh in handle.chunks)):
                        self._validate_mesh_support(mesh,backend="software")
            except NotImplementedError as refused:
                self._backend_diagnostics += (f"Qt software fallback refused: {refused}",)
                self.setToolTip(self._backend_diagnostics[-1])
                self.render_error.emit(self._backend_diagnostics[-1])
                return
        old=self.surface
        self._release_gl()
        old_available = True
        try:
            old.hide()
        except RuntimeError as error:
            old_available = False
            self._note_lifecycle("deleted_surface_cleanup", error=error)
        if getattr(self,"_allow_fallback",False):
            self._backend="software"
            self._renderer=_SoftwareResources();self._hud=_PainterHud()
            self.surface=_PaintSurface(self);self.canvas=self.surface
            if old_available:
                self.layout().replaceWidget(old,self.surface)
            else:
                self.layout().addWidget(self.surface)
            for entry in self._display_entries().values():
                self._renderer.add_mesh(entry["handle"],**entry["appearance"])
            self._apply_highlight_masks()
            self.surface.show();self.redraw()
            self.backend_changed.emit("software")
            if old_available:
                old.deleteLater()

    def destroy(self):
        if self._closed:
            return
        counters = getattr(self, "_lifecycle_counters", None)
        if counters is not None:
            counters["destroy_calls"] = counters.get("destroy_calls", 0) + 1
        failures = []
        def attempt(name, callback):
            try:
                callback()
            except Exception as error:
                failures.append(f"{name}: {type(error).__name__}: {error}")
        attempt("stop_animation", self.stop_animation)
        self._closed = True
        attempt("cancel_hover", self._cancel_hover)
        attempt("clear_prep_cache", lambda: getattr(self, "_software_prep", {}).clear())
        attempt("close_scheduler", self._update_scheduler.close)
        for timer in tuple(self._timers):
            attempt("cancel_timer", lambda timer=timer: self.after_cancel(timer))
            self._timers.discard(timer)
        attempt("_release_gl", self._release_gl)
        renderer = self._renderer
        if renderer is not None:
            # Never bypass the GL ownership/context guard after failed cleanup.
            # Before initialization the GPU host owns only this software placeholder.
            if getattr(self, "_backend", "software") == "software" or isinstance(renderer, _SoftwareResources):
                attempt("renderer.release", renderer.release)
            else:
                failures.append("skipped leftover GL renderer release: context ownership unverified")
        attempt("hide", self.hide)
        attempt("deleteLater", self.deleteLater)
        note = getattr(self, "_note_lifecycle", None)
        if note is not None:
            for failure in failures:
                note("destroy_failures", error=RuntimeError(failure))

    def closeEvent(self, event):
        self.destroy()
        event.accept()

    def _prune_software_prep(self):
        """Drop cached preparations that no longer match displayed chunks.

        Removed handles, vanished chunks and superseded animation
        frames must not retain world arrays; only chunks referenced by
        the current display entries stay cached.
        """
        cache = self._software_prep
        if not cache:
            return
        keep = set()
        for entry in self._display_entries().values():
            handle = entry["handle"]
            if handle.removed or not handle.visible:
                continue
            if entry.get("appearance", {}).get("depth_only"):
                continue
            keep.add((id(handle), None))
            for chunk_id, _mesh in handle.chunks:
                keep.add((id(handle), chunk_id))
        for key in tuple(cache):
            if key not in keep:
                del cache[key]

    def _software_prep_signature(self, entry, handle):
        appearance = entry["appearance"]
        return (
            handle.generations,
            getattr(self._renderer, "mask_revision", 0),
            None if self._section_plane is None else self._section_plane.key,
            self._highlight_fill,
            handle.appearance.scalar_range,
            id(handle.appearance.colormap),
            appearance.get("color"),
            appearance.get("face_colors"),
            appearance.get("invalid_color"),
        )

    def _software_chunk_prep(self, entry, handle, chunk_id, mesh, section):
        """Build or reuse the camera-invariant preparation for one chunk.

        The retained record is invalidated by its signature (handle
        generations, semantic-mask revision, section plane, highlight
        fill and static color style); projection and submission stay
        per frame, so cached and direct paints give identical pixels.
        """
        signature = self._software_prep_signature(entry, handle)
        key = (id(handle), chunk_id)
        prep = self._software_prep.get(key)
        if prep is not None and prep.handle is handle and prep.signature == signature:
            return prep
        appearance = dict(entry["appearance"],scalar_range=handle.appearance.scalar_range,colormap=handle.appearance.colormap)
        positions = mesh.positions
        if mesh.displacements is not None:
            positions = positions + handle.deformation_scale*mesh.displacements
        world = positions @ handle.transform[:3,:3].T + handle.transform[:3,3]
        mask = self._renderer.masks.get(id(handle), {}) if chunk_id is None else self._renderer.chunk_masks.get((id(handle),chunk_id), {})
        hidden = set(mask.get("hidden_elements", ()))
        highlighted = set(mask.get("selected_elements", ())) | set(handle.selected_elements)
        preselected = set(mask.get("preselected_elements", ()))
        colors = appearance.get("face_colors")
        scalars = mesh.element_scalars if mesh.element_scalars is not None else mesh.node_scalars
        scalar_limits = None
        if scalars is not None and not colors:
            finite = scalars[np.isfinite(scalars)]
            scalar_limits = (appearance.get("scalar_range") or (float(finite.min()),float(finite.max()))) if len(finite) else (0.,1.)
        triangles, triangle_colors = [], []
        section_polygons = [] if section is not None else None
        for index, triangle in enumerate(mesh.triangles):
            element = int(mesh.triangle_to_element[index]) if mesh.triangle_to_element is not None else index
            if mesh.active_elements is not None and not mesh.active_elements[element]:
                continue
            if element in hidden:
                continue
            if section is not None:
                section_polygons.append(np.array([p.to_tuple() for p in section.clip_polygon(world[triangle])]))
            color = self._highlight_fill if element in highlighted else (colors[index if len(colors)==mesh.triangle_count else element] if colors else appearance["color"])
            if scalars is not None and not colors and element not in highlighted:
                value=float(scalars[element]) if mesh.element_scalars is not None else float(np.mean(scalars[triangle]))
                color=scalar_color(value,*scalar_limits,appearance.get('colormap')) if np.isfinite(value) else appearance["invalid_color"]
            if element in preselected and element not in highlighted:color="#ffd166"
            triangles.append(index)
            triangle_colors.append(color)
        line_segments, line_selected = [], []
        if mesh.lines is not None and not appearance.get("mesh_lines"):
            hidden_lines=set(mask.get("hidden_lines",()))
            selected_lines=set(mask.get("selected_lines",()))
            for index,(a,b) in enumerate(mesh.lines):
                if index in hidden_lines:continue
                segment=(Point3D(*world[a]),Point3D(*world[b]))
                if self._section_plane is not None:segment=self._section_plane.clip_segment(*segment)
                if segment is None:continue
                line_segments.append((segment[0].to_tuple(), segment[1].to_tuple()))
                line_selected.append(index in selected_lines)
        points, point_selected = [], []
        if mesh.point_indices is not None:
            hidden_points=set(mask.get("hidden_points",()))
            selected_points=set(mask.get("selected_points",()))
            for index,i in enumerate(mesh.point_indices):
                if index in hidden_points:continue
                if self._section_plane is not None and not self._section_plane.contains(world[i]):continue
                points.append(int(i))
                point_selected.append(index in selected_points)
        prep = _SoftwareChunkPrep(handle, signature, world, triangles, triangle_colors,
                                   section_polygons, line_segments, line_selected,
                                   points, point_selected)
        prep.outline_loops = tuple(_surface_outline_loops(
            mesh, triangles, self._outline_groups(entry, chunk_id, mesh)))
        cache = self._software_prep
        cache.pop(key, None)
        cache[key] = prep
        while len(cache) > _SOFTWARE_PREP_LIMIT:
            cache.pop(next(iter(cache)))
        return prep

    def _paint_software(self, painter):
        self._prune_software_prep()
        pixel_ratio = self.pixel_ratio
        painter.setRenderHint(QPainter.Antialiasing)
        ratio = self.surface.devicePixelRatioF()
        painter.scale(1/ratio, 1/ratio)
        width, height = self.viewport_size
        painter.fillRect(0, 0, width, height, QColor(self.bg))
        # One camera basis/viewport/scale evaluation serves every projection
        # in the frame; the retained preparation below removes the
        # camera-invariant per-primitive work from repeated frames.
        frame = _FrameProjection(self)
        section = self._section_plane if self._section_plane is not None and self._section_plane.enabled else None
        faces, lines, points, outlines = [], [], [], []
        for entry in self._display_entries().values():
            handle = entry["handle"]
            if handle.removed or not handle.visible:
                continue
            appearance = dict(entry["appearance"],scalar_range=handle.appearance.scalar_range,colormap=handle.appearance.colormap)
            if appearance.get("depth_only"):
                continue
            for chunk_id, mesh in [(None, handle.mesh), *handle.chunks]:
                prep = self._software_chunk_prep(entry, handle, chunk_id, mesh, section)
                world = prep.world
                screen_all, visible_all = self._project_world_points(world, frame)
                triangle_visible = None
                if section is None and len(mesh.triangles):
                    triangle_visible = visible_all[np.asarray(mesh.triangles)].all(axis=1)
                visible_triangles = []
                for position, index in enumerate(prep.triangles):
                    if section is not None:
                        verts = prep.section_polygons[position]
                        screen, visible = self._project_world_points(verts, frame)
                        if len(screen)<3 or not visible.all():
                            continue
                    else:
                        if not triangle_visible[index]:
                            continue
                        screen = screen_all[mesh.triangles[index]]
                    visible_triangles.append(index)
                    faces.append((sum(p[2] for p in screen)/len(screen), screen, prep.colors[position], appearance))
                if self.show_mesh_lines and (appearance.get("mesh_lines") or entry.get("outline_groups") is None):
                    if section is None and visible_all.all():
                        # All admitted triangles are inside near/far: reuse their
                        # exact semantic boundaries and this frame's projection.
                        segments = ((tuple(screen_all[a]), tuple(screen_all[b]))
                                    for loop in prep.outline_loops
                                    for a, b in ([loop] if len(loop) == 2 else zip(loop, (*loop[1:], loop[0]))))
                    else:
                        segments = self._surface_outline_segments(entry, chunk_id, mesh, visible_triangles, world, frame)
                    for screen in segments:
                        outlines.append((sum(point[2] for point in screen) / len(screen), screen, appearance))
                for segment, selected in zip(prep.line_segments, prep.line_selected):
                    a,b=(self._project_point_with_frame(Point3D(*p), frame) for p in segment)
                    if a is not None and b is not None:
                        style=dict(appearance)
                        if selected:style["line_color"]=self._highlight_fill
                        lines.append((a,b,style))
                for node, selected in zip(prep.points, prep.point_selected):
                    if not visible_all[node]:continue
                    style=dict(appearance)
                    if selected:style["point_color"]=self._highlight_fill
                    points.append((tuple(screen_all[node]),style))
        surfaces = [(depth, 0, screen, color, appearance) for depth, screen, color, appearance in faces]
        surfaces.extend((depth, 1, screen, None, appearance) for depth, screen, appearance in outlines)
        # Keep the software painter's depth ordering: nearer fills must cover
        # more distant boundaries. At equal depth, fill before its own outline.
        for _, kind, screen, color, appearance in sorted(surfaces, key=lambda surface: (-float(f"{surface[0]:.12g}"), surface[1])):
            polygon = QPolygonF([QPointF(*point[:2]) for point in screen])
            if kind == 0:
                # Antialiasing each tessellation triangle independently leaves
                # pale seams where their partially covered edge pixels meet.
                painter.setRenderHint(QPainter.Antialiasing, False)
                fill=QColor(color); fill.setAlphaF(appearance["opacity"])
                painter.setBrush(fill)
                painter.setPen(Qt.NoPen)
                painter.drawPolygon(polygon)
            else:
                painter.setRenderHint(QPainter.Antialiasing, True)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(appearance["line_color"]), appearance["line_width"]*pixel_ratio))
                painter.drawPolyline(polygon)
        painter.setRenderHint(QPainter.Antialiasing, True)
        for a,b,appearance in lines:
            painter.setPen(QPen(QColor(appearance["line_color"]),appearance["line_width"]*pixel_ratio))
            painter.drawLine(QPointF(*a[:2]),QPointF(*b[:2]))
        for p,appearance in points:
            painter.setBrush(QColor(appearance["point_color"]))
            painter.setPen(Qt.NoPen)
            r=appearance["point_size"]*pixel_ratio/2
            painter.drawEllipse(QPointF(*p[:2]),r,r)
        self._hud.painter=painter
        self._render_hud(self.viewport_size)
        self._hud.painter=None
        self._renderer.frame_count+=1
        self._renderer.draw_calls=len(faces)+len(lines)+len(points)
        painter.end()

class QtViewerHostAdapter:
    toolkit_name = "qt"

    def create_viewer(self, parent, *, backend, **options):
        diagnostics = []
        disabled = os.environ.get("ANY3DVIEW_DISABLE_GPU", "").casefold() in {"1","true","yes"}
        if disabled:
            diagnostics.append("ANY3DVIEW_DISABLE_GPU is enabled")
            if backend == "gpu":
                raise GPUUnavailableError("explicit GPU initialization disabled", diagnostics=tuple(diagnostics))
        if backend in {"auto","gpu"} and not disabled:
            try:
                viewer=QtViewer(parent, backend="gpu", **options)
                viewer._allow_fallback=backend=="auto"
                return viewer
            except Exception as error:
                if backend == "gpu":
                    raise
                diagnostics.extend((type(error).__name__,str(error)))
        viewer = QtViewer(parent, backend="software", **options)
        viewer._backend_diagnostics = tuple(diagnostics)
        return viewer
