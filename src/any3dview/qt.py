"""Optional Qt hosts for the shared retained viewer, imported only on request."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF, QSurfaceFormat
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .core import Point3D, parse_color, _interpolate_thickness_color
from .errors import GPUUnavailableError
from .retained_viewer import RetainedViewer, GPU_CAPABILITIES


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

    def add_mesh(self, handle, **appearance):
        self.appearance[id(handle)] = appearance
        self.geometry_uploads += 1

    def remove_mesh(self, handle):
        self.appearance.pop(id(handle), None)
        self.masks.pop(id(handle), None)
        for key in tuple(self.chunk_masks):
            if key[0] == id(handle):
                self.chunk_masks.pop(key, None)
                self.chunk_pickable.pop(key, None)

    def set_semantic_masks(self, handle, **masks):
        self.masks[id(handle)] = masks

    def set_highlighted_elements(self, handle, indices):
        self.masks.setdefault(id(handle), {})["highlighted_elements"] = indices

    def set_chunk_semantic_masks(self, handle, chunk_id, **masks):
        self.chunk_masks[id(handle), chunk_id] = masks

    def clear_chunk_semantic_masks(self, handle, chunk_id):
        self.chunk_masks.pop((id(handle), chunk_id), None)

    def set_chunk_pickable(self, handle, chunk_id, enabled):
        self.chunk_pickable[id(handle), chunk_id] = enabled

    def release(self):
        self.appearance.clear()
        self.masks.clear()
        self.chunk_masks.clear()
        self.chunk_pickable.clear()


class _PainterHud:
    """Painter implementation of the same overlay operations as the GPU HUD."""
    uploads = 0
    def __init__(self):self.painter=None
    def begin(self,viewport):self.uploads+=1
    def render(self,target=None):pass
    def release(self):self.painter=None
    def quad(self,rect,color,alpha=1,**options):
        fill=QColor(color);fill.setAlphaF(alpha)
        self.painter.setPen(Qt.NoPen);self.painter.setBrush(fill)
        x0,y0,x1,y1=rect
        from PySide6.QtCore import QRectF
        self.painter.drawRect(QRectF(x0,y0,x1-x0,y1-y0))
    def line(self,start,end,color,width=1,**options):
        self.painter.setPen(QPen(QColor(color),width));self.painter.setBrush(Qt.NoBrush)
        self.painter.drawLine(QPointF(*start),QPointF(*end))
    def polyline(self,points,color,width=1,closed=False,**options):
        for a,b in zip(points,points[1:]):self.line(a,b,color,width)
        if closed and points:self.line(points[-1],points[0],color,width)
    def rectangle(self,rect,color,width=1,fill=None,fill_alpha=1,**options):
        if fill:self.quad(rect,fill,fill_alpha)
        x0,y0,x1,y1=rect
        self.polyline(((x0,y0),(x1,y0),(x1,y1),(x0,y1)),color,width,True)
    def circle(self,point,radius,color,width=1,**options):
        self.painter.setPen(QPen(QColor(color),width));self.painter.setBrush(Qt.NoBrush)
        self.painter.drawEllipse(QPointF(*point),radius,radius)
    def text(self,point,text,color,font=("Segoe UI",10,""),anchor="nw",**options):
        qfont=QFont(font[0],font[1]);qfont.setBold("bold" in font[2])
        self.painter.setFont(qfont);self.painter.setPen(QColor(color))
        metrics=self.painter.fontMetrics();x,y=point
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
            self.owner._drag_select(e)
        elif event.buttons():
            self.owner._motion(e)
        else:
            self.owner._hover_select(e)
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.owner._release_select(self._event(event))
        else:
            self.owner._release(self._event(event))
        event.accept()

    def wheelEvent(self, event):
        self.owner._zoom(0.9 ** (event.angleDelta().y()/120))
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.owner._cancel_interaction()
        else:
            super().keyPressEvent(event)

    def focusOutEvent(self, event):
        self.owner._cancel_interaction()
        super().focusOutEvent(event)


class _PaintSurface(_InputSurface, QWidget):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def paintEvent(self, event):
        self.owner._paint_software(QPainter(self))


class _GLSurface(_InputSurface, QOpenGLWidget):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.error = None
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def initializeGL(self):
        self.error = None
        try:
            import moderngl
            from .gpu.renderer import ModernGLRenderer
            from .gpu.hud import GPUHudRenderer
            self.owner._renderer = ModernGLRenderer(moderngl.create_context(
                require=330, context=_QtGLLoader(self.context())))
            self.owner._hud = GPUHudRenderer(self.owner._renderer.ctx)
            self.context().aboutToBeDestroyed.connect(self.owner._release_gl)
            # Qt can recreate the context when the widget is reparented.
            # Re-upload retained snapshots rather than losing the visible scene.
            import inspect
            accepted=inspect.signature(self.owner._renderer.add_mesh).parameters
            for entry in self.owner._display_entries().values():
                self.owner._renderer.add_mesh(entry["handle"],**{k:v for k,v in entry["appearance"].items() if k in accepted})
            self.owner._apply_highlight_masks()
        except Exception as error:
            self.error = error
            if not getattr(self.owner,"_initializing",True):
                QTimer.singleShot(0,self.owner._render_failed)

    def paintGL(self):
        if self.error is not None or self.owner._closed:
            return
        try:
            owner = self.owner
            renderer = owner._renderer
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
            QTimer.singleShot(0, owner._render_failed)


class QtViewer(RetainedViewer, QWidget):
    """QWidget satisfying ViewerBackend through shared retained behavior."""
    render_error = Signal(str)
    backend_changed = Signal(str)
    def __init__(self, parent=None, *, backend="software", **options):
        QWidget.__init__(self, parent)
        allowed = {k: options[k] for k in ("width", "height", "bg", "interactive_fps", "shading", "interaction_profile") if k in options}
        self.initialize_state(**allowed)
        self._backend = backend
        self._initializing = True
        self._timers = set()
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
            self.surface.grabFramebuffer()
            if not self.surface.isValid() or self.surface.error is not None:
                error = self.surface.error or RuntimeError("Qt OpenGL context is unavailable")
                self.destroy()
                raise GPUUnavailableError(str(error), diagnostics=(type(error).__name__, str(error)))
        self._poll_updates()
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
                       stippled_transparency=False)

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

    def framebuffer_size(self):
        ratio = self.surface.devicePixelRatioF()
        return max(1, round(self.surface.width()*ratio)), max(1, round(self.surface.height()*ratio))

    def request_redraw(self):
        self.surface.update()

    def cancel_redraw(self):
        pass  # Qt coalesces update requests; closed state prevents drawing.

    def make_current(self):
        if self._backend == "gpu":
            self.surface.makeCurrent()

    def swap_buffers(self):
        pass  # QOpenGLWidget owns composition and presentation.

    def _mesh_changed(self, handle, change):
        self.make_current()
        return super()._mesh_changed(handle, change)

    def _apply_highlight_masks(self):
        self.make_current()
        return super()._apply_highlight_masks()

    def _restore_live_renderer(self):
        self.make_current()
        return super()._restore_live_renderer()

    def _show_animation_frame(self, index):
        self.make_current()
        return super()._show_animation_frame(index)

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

    def _release_gl(self):
        if self._backend == "gpu" and self._hud is not None:
            self.surface.makeCurrent()
            self._hud.release()
            self._renderer.release()
            self._hud = None
            self.surface.doneCurrent()

    def _render_failed(self):
        if self._closed:
            return
        error = self.surface.error
        self._backend_diagnostics += (f"Qt rendering failed: {error}",)
        self.setToolTip(self._backend_diagnostics[-1])
        self.render_error.emit(self._backend_diagnostics[-1])
        old=self.surface
        self._release_gl()
        old.hide()
        if getattr(self,"_allow_fallback",False):
            self._backend="software"
            self._renderer=_SoftwareResources();self._hud=_PainterHud()
            self.surface=_PaintSurface(self);self.canvas=self.surface
            self.layout().replaceWidget(old,self.surface)
            for entry in self._display_entries().values():
                self._renderer.add_mesh(entry["handle"],**entry["appearance"])
            self._apply_highlight_masks()
            self.surface.show();self.redraw()
            self.backend_changed.emit("software")
            old.deleteLater()

    def destroy(self):
        if self._closed:
            return
        self.stop_animation()
        self._closed = True
        for timer in tuple(self._timers):
            self.after_cancel(timer)
        self._release_gl()
        self._renderer.release()
        self.hide()
        self.deleteLater()

    def closeEvent(self, event):
        self.destroy()
        event.accept()

    def _paint_software(self, painter):
        painter.setRenderHint(QPainter.Antialiasing)
        ratio = self.surface.devicePixelRatioF()
        painter.scale(1/ratio, 1/ratio)
        painter.fillRect(0, 0, *self.viewport_size, QColor(self.bg))
        faces, lines, points = [], [], []
        for entry in self._display_entries().values():
            handle = entry["handle"]
            if handle.removed or not handle.visible:
                continue
            appearance = entry["appearance"]
            if appearance.get("depth_only"):
                continue
            for chunk_id, mesh in [(None, handle.mesh), *handle.chunks]:
                positions = mesh.positions
                if mesh.displacements is not None:
                    positions = positions + handle.deformation_scale*mesh.displacements
                world = positions @ handle.transform[:3,:3].T + handle.transform[:3,3]
                projected = [self.project_point(Point3D(*p)) for p in world]
                mask = self._renderer.masks.get(id(handle), {}) if chunk_id is None else self._renderer.chunk_masks.get((id(handle),chunk_id), {})
                hidden = set(mask.get("hidden_elements", ()))
                highlighted = set(mask.get("selected_elements", ())) | set(handle.selected_elements)
                preselected = set(mask.get("preselected_elements", ()))
                colors = appearance.get("face_colors")
                for index, triangle in enumerate(mesh.triangles):
                    element = int(mesh.triangle_to_element[index]) if mesh.triangle_to_element is not None else index
                    if mesh.active_elements is not None and not mesh.active_elements[element]:
                        continue
                    if element in hidden:
                        continue
                    verts = world[triangle]
                    if self._section_plane is not None and self._section_plane.enabled:
                        verts = np.array([p.to_tuple() for p in self._section_plane.clip_polygon(verts)])
                    screen = [self.project_point(Point3D(*p)) for p in verts]
                    if len(screen)<3 or any(p is None for p in screen):
                        continue
                    color = self._highlight_fill if element in highlighted else (colors[index if len(colors)==mesh.triangle_count else element] if colors else appearance["color"])
                    scalars=mesh.element_scalars if mesh.element_scalars is not None else mesh.node_scalars
                    if scalars is not None and not colors and element not in highlighted:
                        value=float(scalars[element]) if mesh.element_scalars is not None else float(np.mean(scalars[triangle]))
                        finite=scalars[np.isfinite(scalars)]
                        limits=appearance.get("scalar_range") or (float(finite.min()),float(finite.max())) if len(finite) else (0.,1.)
                        color=_interpolate_thickness_color(value,*limits) if np.isfinite(value) else appearance["invalid_color"]
                    if element in preselected and element not in highlighted:color="#ffd166"
                    faces.append((sum(p[2] for p in screen)/len(screen), screen, color, appearance))
                if mesh.lines is not None:
                    hidden_lines=set(mask.get("hidden_lines",()))
                    selected_lines=set(mask.get("selected_lines",()))
                    for index,(a,b) in enumerate(mesh.lines):
                        if index in hidden_lines:continue
                        segment=(Point3D(*world[a]),Point3D(*world[b]))
                        if self._section_plane is not None:segment=self._section_plane.clip_segment(*segment)
                        if segment is None:continue
                        a,b=(self.project_point(p) for p in segment)
                        if a is not None and b is not None:
                            style=dict(appearance)
                            if index in selected_lines:style["line_color"]=self._highlight_fill
                            lines.append((a,b,style))
                if mesh.point_indices is not None:
                    hidden_points=set(mask.get("hidden_points",()))
                    selected_points=set(mask.get("selected_points",()))
                    for index,i in enumerate(mesh.point_indices):
                        if index in hidden_points or projected[i] is None:continue
                        if self._section_plane is not None and not self._section_plane.contains(world[i]):continue
                        style=dict(appearance)
                        if index in selected_points:style["point_color"]=self._highlight_fill
                        points.append((projected[i],style))
        for _, screen, color, appearance in sorted(faces, key=lambda f:f[0], reverse=True):
            fill=QColor(color); fill.setAlphaF(appearance["opacity"])
            painter.setBrush(fill)
            painter.setPen(QPen(QColor(appearance["line_color"]), appearance["line_width"]) if self.show_mesh_lines else Qt.NoPen)
            painter.drawPolygon(QPolygonF([QPointF(*p[:2]) for p in screen]))
        for a,b,appearance in lines:
            painter.setPen(QPen(QColor(appearance["line_color"]),appearance["line_width"]))
            painter.drawLine(QPointF(*a[:2]),QPointF(*b[:2]))
        for p,appearance in points:
            painter.setBrush(QColor(appearance["point_color"]))
            painter.setPen(Qt.NoPen)
            r=appearance["point_size"]/2
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
