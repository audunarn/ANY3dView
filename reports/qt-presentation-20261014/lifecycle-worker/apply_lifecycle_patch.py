"""Exact-match patcher for src/any3dview/qt.py GL lifecycle ownership.

Each replacement must occur exactly once; the script fails closed otherwise.
Line endings are CRLF throughout, preserved by construction.
"""
from pathlib import Path

PATH = Path(__file__).resolve().parents[3] / "src/any3dview/qt.py"
text = PATH.read_text(encoding="utf-8", newline="")

EDITS = []

# A. __init__: ownership token, cached GL info, lifecycle counters/errors.
EDITS.append((
    '        self.initialize_state(**allowed)\r\n'
    '        self._backend = backend\r\n'
    '        self._initializing = True\r\n',
    '        self.initialize_state(**allowed)\r\n'
    '        self._backend = backend\r\n'
    '        # Explicit GL ownership: the token is the Qt context that the\r\n'
    '        # current renderer/HUD resources were allocated under; it is set\r\n'
    '        # only once both are constructed. Cached GL strings and counters\r\n'
    '        # are passive diagnostics and own no GL state.\r\n'
    '        self._gl_context = None\r\n'
    '        self._gl_info = {}\r\n'
    '        self._lifecycle_counters = {"initialize_gl_attempts": 0,\r\n'
    '            "context_recreations": 0, "gl_release_attempts": 0,\r\n'
    '            "gl_release_failures": 0, "destroy_calls": 0}\r\n'
    '        self._lifecycle_errors = []\r\n'
    '        self._initializing = True\r\n',
))

# B. __init__ GPU probe: raw grabFramebuffer exception rolls the constructor back.
EDITS.append((
    '            self.surface.resize(self.width, self.height)\r\n'
    '            self.surface.show()\r\n'
    '            self.surface.grabFramebuffer()\r\n'
    '            if not self.surface.isValid() or self.surface.error is not None:\r\n',
    '            self.surface.resize(self.width, self.height)\r\n'
    '            self.surface.show()\r\n'
    '            try:\r\n'
    '                self.surface.grabFramebuffer()\r\n'
    '            except Exception as error:\r\n'
    '                # Roll back the half-constructed viewer before the failure\r\n'
    '                # escapes, mirroring the invalid-context branch below.\r\n'
    '                self.destroy()\r\n'
    '                raise GPUUnavailableError(str(error), diagnostics=(type(error).__name__, str(error)))\r\n'
    '            if not self.surface.isValid() or self.surface.error is not None:\r\n',
))

# C. _capture_gl_info helper after _gl_surface_format.
EDITS.append((
    '    fmt.setDepthBufferSize(24)\r\n'
    '    return fmt\r\n'
    '\r\n'
    '\r\n'
    'class _GLSurface(_InputSurface, QOpenGLWidget):\r\n',
    '    fmt.setDepthBufferSize(24)\r\n'
    '    return fmt\r\n'
    '\r\n'
    '\r\n'
    'def _capture_gl_info(context):\r\n'
    '    """Cache identifying GL strings once, while the context is current."""\r\n'
    '    try:\r\n'
    '        info = context.info\r\n'
    '        return {"vendor": str(info.get("GL_VENDOR", "")),\r\n'
    '                "renderer": str(info.get("GL_RENDERER", "")),\r\n'
    '                "version": str(info.get("GL_VERSION", ""))}\r\n'
    '    except Exception:\r\n'
    '        return {}\r\n'
    '\r\n'
    '\r\n'
    'class _GLSurface(_InputSurface, QOpenGLWidget):\r\n',
))

# D. initializeGL: HUD failure cannot leak the renderer; token-bound callback.
EDITS.append((
    '    def initializeGL(self):\r\n'
    '        self.error = None\r\n'
    '        try:\r\n'
    '            import moderngl\r\n'
    '            from .gpu.renderer import ModernGLRenderer\r\n'
    '            from .gpu.hud import GPUHudRenderer\r\n'
    '            self.owner._renderer = ModernGLRenderer(moderngl.create_context(\r\n'
    '                require=330, context=_QtGLLoader(self.context())))\r\n'
    '            self.owner._hud = GPUHudRenderer(self.owner._renderer.ctx)\r\n'
    '            self.context().aboutToBeDestroyed.connect(self.owner._release_gl)\r\n'
    '            # Qt can recreate the context when the widget is reparented.\r\n'
    '            # Re-upload retained snapshots rather than losing the visible scene.\r\n'
    '            import inspect\r\n'
    '            accepted=inspect.signature(self.owner._renderer.add_mesh).parameters\r\n'
    '            for entry in self.owner._display_entries().values():\r\n'
    '                self.owner._renderer.add_mesh(entry["handle"],**{k:v for k,v in entry["appearance"].items() if k in accepted})\r\n'
    '            self.owner._apply_highlight_masks()\r\n'
    '        except Exception as error:\r\n'
    '            self.error = error\r\n'
    '            if not getattr(self.owner,"_initializing",True):\r\n'
    '                self._defer_render_failure()\r\n',
    '    def initializeGL(self):\r\n'
    '        self.error = None\r\n'
    '        owner = self.owner\r\n'
    '        owner._note_lifecycle("initialize_gl_attempts")\r\n'
    '        try:\r\n'
    '            import inspect\r\n'
    '            import moderngl\r\n'
    '            from .gpu.renderer import ModernGLRenderer\r\n'
    '            from .gpu.hud import GPUHudRenderer\r\n'
    '            context = self.context()\r\n'
    '            renderer = ModernGLRenderer(moderngl.create_context(\r\n'
    '                require=330, context=_QtGLLoader(context)))\r\n'
    '            try:\r\n'
    '                hud = GPUHudRenderer(renderer.ctx)\r\n'
    '            except BaseException:\r\n'
    '                # The context is current here; release the renderer\'s\r\n'
    '                # derived resources so a failing HUD constructor cannot\r\n'
    '                # leak them past the ownership token below.\r\n'
    '                renderer.release()\r\n'
    '                raise\r\n'
    '            owner._renderer = renderer\r\n'
    '            owner._hud = hud\r\n'
    '            if owner._gl_context is not None and owner._gl_context is not context:\r\n'
    '                owner._note_lifecycle("context_recreations")\r\n'
    '            owner._gl_context = context\r\n'
    '            owner._gl_info = _capture_gl_info(renderer.ctx)\r\n'
    '            # The closure carries the context token: a stale callback for a\r\n'
    '            # replaced context must not release the replacement\'s resources.\r\n'
    '            context.aboutToBeDestroyed.connect(lambda: owner._release_gl(context))\r\n'
    '            # Qt can recreate the context when the widget is reparented.\r\n'
    '            # Re-upload retained snapshots rather than losing the visible scene.\r\n'
    '            accepted=inspect.signature(renderer.add_mesh).parameters\r\n'
    '            for entry in owner._display_entries().values():\r\n'
    '                renderer.add_mesh(entry["handle"],**{k:v for k,v in entry["appearance"].items() if k in accepted})\r\n'
    '            owner._apply_highlight_masks()\r\n'
    '        except Exception as error:\r\n'
    '            self.error = error\r\n'
    '            if not getattr(self.owner,"_initializing",True):\r\n'
    '                self._defer_render_failure()\r\n',
))

# E. _release_gl: token guard, validity gate, independent releases, idempotence.
EDITS.append((
    '    def _release_gl(self):\r\n'
    '        if self._backend == "gpu" and self._hud is not None:\r\n'
    '            self.surface.makeCurrent()\r\n'
    '            self._hud.release()\r\n'
    '            self._renderer.release()\r\n'
    '            self._hud = None\r\n'
    '            self.surface.doneCurrent()\r\n',
    '    def _release_gl(self, context=None):\r\n'
    '        """Release GL resources owned by this viewer\'s tracked context.\r\n'
    '\r\n'
    '        The optional token is the Qt context whose aboutToBeDestroyed\r\n'
    '        fired; a stale callback for a replaced context must not release\r\n'
    '        the replacement\'s resources. Qt owns the native context: only\r\n'
    '        derived resources are released, never against an invalid context.\r\n'
    '        """\r\n'
    '        if self._backend != "gpu":\r\n'
    '            return\r\n'
    '        owner_context = getattr(self, "_gl_context", None)\r\n'
    '        if owner_context is None:\r\n'
    '            return  # No GL resources were allocated, or already released.\r\n'
    '        if context is not None and context is not owner_context:\r\n'
    '            return\r\n'
    '        self._gl_context = None\r\n'
    '        self._note_lifecycle("gl_release_attempts")\r\n'
    '        hud, renderer, surface = self._hud, self._renderer, self.surface\r\n'
    '        failures = []\r\n'
    '        try:\r\n'
    '            valid = bool(surface.isValid()) and bool(owner_context.isValid())\r\n'
    '        except RuntimeError:\r\n'
    '            valid = False  # Deleted Qt objects; native resources are gone.\r\n'
    '        if valid:\r\n'
    '            try:\r\n'
    '                surface.makeCurrent()\r\n'
    '            except RuntimeError as error:\r\n'
    '                failures.append(f"makeCurrent: {type(error).__name__}: {error}")\r\n'
    '                valid = False\r\n'
    '        if valid:\r\n'
    '            # Each owner releases independently so one failure cannot skip\r\n'
    '            # the others or the doneCurrent pairing.\r\n'
    '            for name, resource in (("hud", hud), ("renderer", renderer)):\r\n'
    '                if resource is None:\r\n'
    '                    continue\r\n'
    '                try:\r\n'
    '                    resource.release()\r\n'
    '                except BaseException as error:\r\n'
    '                    failures.append(f"{name}.release: {type(error).__name__}: {error}")\r\n'
    '            try:\r\n'
    '                surface.doneCurrent()\r\n'
    '            except RuntimeError as error:\r\n'
    '                failures.append(f"doneCurrent: {type(error).__name__}: {error}")\r\n'
    '        else:\r\n'
    '            failures.append("skipped GL release: Qt context unavailable")\r\n'
    '        for failure in failures:\r\n'
    '            self._note_lifecycle("gl_release_failures", error=RuntimeError(failure))\r\n'
    '        self._hud = None\r\n'
    '        self._renderer = None\r\n',
))

# F. destroy: count calls; release _renderer only when still owned.
EDITS.append((
    '    def destroy(self):\r\n'
    '        if self._closed:\r\n'
    '            return\r\n'
    '        self.stop_animation()\r\n',
    '    def destroy(self):\r\n'
    '        if self._closed:\r\n'
    '            return\r\n'
    '        self._note_lifecycle("destroy_calls")\r\n'
    '        self.stop_animation()\r\n',
))
EDITS.append((
    '        self._release_gl()\r\n'
    '        self._renderer.release()\r\n'
    '        self.hide()\r\n',
    '        self._release_gl()\r\n'
    '        if self._renderer is not None:\r\n'
    '            self._renderer.release()\r\n'
    '        self.hide()\r\n',
))

# G. _note_lifecycle + diagnostic_snapshot after the capabilities property.
EDITS.append((
    '                       stippled_transparency=False, smooth_normals=False)\r\n'
    '\r\n'
    '    def _queue_hover(self, event):\r\n',
    '                       stippled_transparency=False, smooth_normals=False)\r\n'
    '\r\n'
    '    def _note_lifecycle(self, event, error=None):\r\n'
    '        """Count a lifecycle event; retain bounded evidence of failures."""\r\n'
    '        counters = getattr(self, "_lifecycle_counters", None)\r\n'
    '        if counters is not None:\r\n'
    '            counters[event] = counters.get(event, 0) + 1\r\n'
    '        if error is not None:\r\n'
    '            errors = getattr(self, "_lifecycle_errors", None)\r\n'
    '            if errors is not None:\r\n'
    '                errors.append(f"{type(error).__name__}: {error}")\r\n'
    '                del errors[:-8]\r\n'
    '\r\n'
    '    def diagnostic_snapshot(self):\r\n'
    '        """Passive, JSON-compatible lifecycle diagnostics for this viewer.\r\n'
    '\r\n'
    '        Reads only cached state and Qt\'s isValid(); it never renders,\r\n'
    '        makes the context current, switches backends, queries live GL,\r\n'
    '        enumerates processes, touches project data or mutates ownership.\r\n'
    '        """\r\n'
    '        context = getattr(self, "_gl_context", None)\r\n'
    '        valid = False\r\n'
    '        if context is not None:\r\n'
    '            try:\r\n'
    '                valid = bool(context.isValid())\r\n'
    '            except RuntimeError:\r\n'
    '                valid = False  # Deleted Qt object.\r\n'
    '        info = getattr(self, "_gl_info", None) or {}\r\n'
    '        return {\r\n'
    '            "schema": "any3dview.qt.lifecycle",\r\n'
    '            "version": 1,\r\n'
    '            "active_backend": self._backend,\r\n'
    '            "closed": bool(self._closed),\r\n'
    '            "context_valid": valid,\r\n'
    '            "gl_vendor": str(info.get("vendor", "")),\r\n'
    '            "gl_renderer": str(info.get("renderer", "")),\r\n'
    '            "gl_version": str(info.get("version", "")),\r\n'
    '            "lifecycle_counters": dict(getattr(self, "_lifecycle_counters", None) or {}),\r\n'
    '            "lifecycle_errors": list(getattr(self, "_lifecycle_errors", None) or ()),\r\n'
    '        }\r\n'
    '\r\n'
    '    def _queue_hover(self, event):\r\n',
))

for index, (old, new) in enumerate(EDITS):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"edit {index}: expected 1 match, found {count}")
    text = text.replace(old, new, 1)

PATH.write_text(text, encoding="utf-8", newline="")
print(f"applied {len(EDITS)} edits; {len(text)} bytes")
